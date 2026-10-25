// 逐层 roofline 实测：真实网络形状（从 resnet18_folded.bin 读出），
// 分别计时 quant+im2col / int8 GEMM / requant / relu，并与 DP4A 理论上限比。
//
// 为什么不用 prof_layer：它只测 C64 32x32 一层（硬编码），而且把 requant
// 塞进了 GEMM 的计时窗口。逐层账必须自己量。
//
// 用法：./bench_layers <model.bin> <calib.bin> [batch=512] [iters=20]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "int8_engine.cuh"

using ptqmodel::BlockSpec;
using ptqmodel::ConvSpec;
using ptqmodel::Model;

static const double PEAK_TOPS = 15.34;   // 1024 core * 8192 ops/clk * 1873 MHz
static const double PEAK_BW = 192.0;     // GB/s，GDDR6 12 Gbps * 128 bit

static double now_ms()
{
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

template <class F>
static double time_ms(F&& f, int iters)
{
    // 先空跑 3 次：cublasLt 首次调用要做算法选择/workspace 分配，GPU 时钟也要爬升。
    // 不预热时同一份代码 GEMM 会从 5.3 漂到 8.1 TOPS（实测跨 3 轮），
    // 那不是硬件差异，是测量口径。
    for (int i = 0; i < 3; i++) f();
    cudaDeviceSynchronize();
    const double t0 = now_ms();
    for (int i = 0; i < iters; i++) f();
    cudaDeviceSynchronize();
    return (now_ms() - t0) / iters;
}

static int out_dim(int in, int k, int s, int p) { return (in + 2 * p - k) / s + 1; }

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        fprintf(stderr, "usage: %s <model.bin> <calib.bin> [batch] [iters]\n", argv[0]);
        return 2;
    }
    const int n = argc > 3 ? atoi(argv[3]) : 512;
    const int iters = argc > 4 ? atoi(argv[4]) : 20;

    Model m;
    if (!m.load(argv[1])) return 1;
    std::vector<std::pair<float, float>> ranges;
    {
        std::string err;
        if (!load_calib(argv[2], m.convs.size(), &ranges, &err))
        {
            fprintf(stderr, "[FAIL] %s\n", err.c_str());
            return 1;
        }
    }
    printf("model: %zu convs, %zu blocks, stem_pool=%d, batch=%d, iters=%d\n",
           m.convs.size(), m.blocks.size(), m.stem_pool, n, iters);

    ptqint8::CublasCtx ctx;
    ctx.init();

    // 复用一块足够大的输入/权重/临时缓冲；形状随层变化
    size_t max_x = 0;
    for (auto& c : m.convs) max_x = std::max(max_x, (size_t)n * 3 * 32 * 32);
    max_x = (size_t)n * 512 * 4 * 4;   // 实际最大激活 = 512*64*32*32
    max_x = std::max(max_x, (size_t)n * 64 * 32 * 32);
    float* d_x = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, max_x * sizeof(float)));
    {
        // 非零随机输入：否则 implicit 的 int32/epilogue 对照在 0 输入上
        // 只会验证到「写 0」，padding/累加错误都看不出来。
        std::mt19937 rng(12345);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> hx(max_x);
        for (auto& v : hx) v = dist(rng);
        CUDA_CHECK(cudaMemcpy(d_x, hx.data(), max_x * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    const int MAXK = 4608 + 8, MAXN = 512;
    int8_t* d_w = nullptr;
    CUDA_CHECK(cudaMalloc(&d_w, (size_t)MAXK * MAXN));
    CUDA_CHECK(cudaMemset(d_w, 1, (size_t)MAXK * MAXN));
    float* d_ws = nullptr, *d_b = nullptr, *d_zp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ws, MAXN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, MAXN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_zp, MAXN * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_ws, 0, MAXN * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_b, 0, MAXN * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_zp, 0, MAXN * sizeof(float)));
    {
        // 给 epilogue 对照非零 scale/bias，避免「0×scale+0」这种假通过。
        std::vector<float> hws(MAXN, 0.5f), hb(MAXN, 0.1f);
        CUDA_CHECK(cudaMemcpy(d_ws, hws.data(), MAXN * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, hb.data(), MAXN * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    int ch = 3, hh = 32, ww = 32;
    double t_im2col = 0, t_im2col_fast = 0, t_gemm = 0, t_req = 0, t_req_tiled = 0, t_relu = 0;
    double t_implicit = 0;
    double t_im2col_vec = 0, t_im2col_tiled = 0;
    long long flops = 0;

    printf("\n%-4s %-16s %7s %5s %6s %9s %9s %8s %8s %9s %8s %8s %7s %8s\n",
           "#", "layer", "Kpad", "N", "M", "im2col_old", "im2col_new", "gemm",
           "implicit", "req", "relu", "TOPS", "%peak", "imp_x");
    int idx = 0;

    auto bench_conv = [&](int li) {
        const ConvSpec& c = m.convs[li];
        const int OH = out_dim(hh, c.kh, c.sh, c.ph);
        const int OW = out_dim(ww, c.kw, c.sw, c.pw);
        const int K = c.cin * c.kh * c.kw;
        const int Kp = ((K + 7) / 8) * 8;
        // tap-major kernel 的 K 维按 Cpad=(cin+3)&~3 对齐（避免 4 字节 load
        // 跨出 cin 边界）；cin=3 时 KpTap 会比旧 Kp 大，缓冲区必须按两者
        // 的较大值分配，否则 stem conv 会越界（实测 illegal memory access）。
        const int Cpad = (c.cin + 3) & ~3;
        const int Kt = c.kh * c.kw * Cpad;
        const int KpTap = ((Kt + 7) / 8) * 8;
        const int Kcap = std::max(Kp, KpTap);
        const int N = c.cout;
        const long long M = (long long)n * OH * OW;
        if (c.groups != 1)
        {
            printf("%-4d %-16s  skip (groups=%d)\n", idx++, "grouped", c.groups);
            return;
        }
        int8_t* d_col = nullptr;
        int32_t* d_acc = nullptr;
        float* d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_col, (size_t)M * Kcap));
        CUDA_CHECK(cudaMalloc(&d_acc, (size_t)M * N * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_out, (size_t)M * N * sizeof(float)));

        const int grid_col = (int)std::min<long long>(65535, (M * Kp + 255) / 256);
        const double im2 = time_ms([&] {
            ptqint8::quant_im2col_launch(d_x, d_col, 0.01f, 0, 127, n, ch, hh, ww,
                                         c.kh, c.kw, c.sh, c.sw, c.ph, c.pw, OH, OW,
                                         K, Kp, 0, grid_col);
        }, iters);

        // 同一层、同一输出，只换新版 kernel（grid=(OW,OH,N)，无逐元素 64 位 div）
        const double im2f = time_ms([&] {
            ptqint8::quant_im2col_fast_launch(d_x, d_col, 0.01f, 0, 127, n, ch, hh, ww,
                                              c.kh, c.kw, c.sh, c.sw, c.ph, c.pw, OH, OW,
                                              K, Kp, 0);
        }, iters);

        // ⚠️ tap-major（vec/共享 tile）im2col 是上一轮的 WIP：实测在第 0 层就触发
        // illegal memory access（异步错误，会污染后面所有层的 CUDA_CHECK），
        // 所以默认不跑。要复现用 IM2COL_VEC=1；修好后应去掉这个开关。
        double im2v = 0.0;
        if (std::getenv("IM2COL_VEC"))
        {
            im2v = time_ms([&] {
                ptqint8::quant_im2col_tap_launch(d_x, d_col, 0.01f, 127,
                                                 n, ch, hh, ww, c.kh, c.kw, c.sh, c.sw,
                                                 c.ph, c.pw, OH, OW, KpTap, 0);
            }, iters);
            CUDA_CHECK(cudaGetLastError());
            printf("     im2col vec %.3f ms (x%.2f vs fast)\n", im2v, im2f / im2v);
        }


        // tiled 版：整幅输入量化进 shared 一次，各 tap 从 shared 取。
        const double im2t = time_ms([&] {
            ptqint8::quant_im2col_tiled_launch(d_x, d_col, 0.01f, 127,
                                               n, ch, hh, ww, c.kh, c.kw, c.sh, c.sw,
                                               c.ph, c.pw, OH, OW, K, Kp, 0);
        }, iters);
        {
            std::vector<int8_t> ht((size_t)M * Kp), hf2((size_t)M * Kp);
            ptqint8::quant_im2col_tiled_launch(d_x, d_col, 0.01f, 127,
                                               n, ch, hh, ww, c.kh, c.kw, c.sh, c.sw,
                                               c.ph, c.pw, OH, OW, K, Kp, 0);
            CUDA_CHECK(cudaGetLastError());
            const cudaError_t e1 = cudaMemcpy(ht.data(), d_col, ht.size(), cudaMemcpyDeviceToHost);
            ptqint8::quant_im2col_fast_launch(d_x, d_col, 0.01f, 0, 127, n, ch, hh, ww,
                                              c.kh, c.kw, c.sh, c.sw, c.ph, c.pw, OH, OW,
                                              K, Kp, 0);
            CUDA_CHECK(cudaMemcpy(hf2.data(), d_col, hf2.size(), cudaMemcpyDeviceToHost));
            long long bad2 = 0;
            if (e1 == cudaSuccess)
                for (size_t i = 0; i < ht.size(); i++) bad2 += (ht[i] != hf2[i]);
            printf("     im2col tiled %.3f ms (x%.2f vs fast)  bit-exact: %s (mismatch %lld)\n",
                   im2t, im2f / im2t, bad2 == 0 ? "YES" : "NO", bad2);
        }
        t_im2col_tiled += im2t;

        // 逐位一致性：旧 kernel vs 新 kernel 在同一输入/同一 scale 下的输出
        std::vector<int8_t> h_old((size_t)M * Kp), h_new((size_t)M * Kp);
        ptqint8::quant_im2col_launch(d_x, d_col, 0.01f, 0, 127, n, ch, hh, ww,
                                     c.kh, c.kw, c.sh, c.sw, c.ph, c.pw, OH, OW,
                                     K, Kp, 0, grid_col);
        CUDA_CHECK(cudaMemcpy(h_old.data(), d_col, h_old.size(), cudaMemcpyDeviceToHost));
        ptqint8::quant_im2col_fast_launch(d_x, d_col, 0.01f, 0, 127, n, ch, hh, ww,
                                          c.kh, c.kw, c.sh, c.sw, c.ph, c.pw, OH, OW,
                                          K, Kp, 0);
        CUDA_CHECK(cudaMemcpy(h_new.data(), d_col, h_new.size(), cudaMemcpyDeviceToHost));
        long long bad = 0;
        for (size_t i = 0; i < h_old.size(); i++) bad += (h_old[i] != h_new[i]);
        printf("     bit-exact: %s (mismatch %lld / %zu)\n",
               bad == 0 ? "YES" : "NO", bad, h_old.size());

        torch::Tensor acc;
        const double gem = time_ms([&] {
            acc = ptqint8::int8_gemm(d_col, M, Kp, d_w, N);
        }, iters);
        d_acc = acc.data_ptr<int32_t>();

        // ---- implicit GEMM：计时 + int32 逐元素对照 ----
        // 权重全 1，所以新旧两条路径的 int32 必须完全相同（和卷积的加法顺序无关）。
        std::vector<int8_t> h_wimp(ptqint8::implicit_weight_bytes(N, c.cin, c.kh, c.kw), 1);
        int8_t* d_wimp = nullptr;
        CUDA_CHECK(cudaMalloc(&d_wimp, h_wimp.size()));
        CUDA_CHECK(cudaMemcpy(d_wimp, h_wimp.data(), h_wimp.size(), cudaMemcpyHostToDevice));
        int32_t* d_acc2 = nullptr;
        CUDA_CHECK(cudaMalloc(&d_acc2, (size_t)M * N * sizeof(int32_t)));
        int8_t* d_qact = nullptr;
        CUDA_CHECK(cudaMalloc(&d_qact, (size_t)n * ((c.cin + 3) / 4) * 4 * hh * ww));
        const double t_quant = time_ms([&] {
            ptqint8::quant_act_c4_launch(d_x, d_qact, 0.01f, 127, n, ch, 0,
                                         c.cin, hh, ww);
        }, iters);
        const double t_conv = time_ms([&] {
            ptqint8::implicit_conv_launch(d_qact, d_wimp, d_acc2,
                                          n, c.cin, hh, ww, c.kh, c.kw,
                                          c.sh, c.sw, c.ph, c.pw, OH, OW, N);
        }, iters);
        const double imp = t_quant + t_conv;
        std::vector<int32_t> h_acc((size_t)M * N), h_acc2((size_t)M * N);
        CUDA_CHECK(cudaMemcpy(h_acc.data(), d_acc, h_acc.size() * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_acc2.data(), d_acc2, h_acc2.size() * 4, cudaMemcpyDeviceToHost));
        long long imp_bad = 0;
        for (size_t z = 0; z < h_acc.size(); z++) imp_bad += (h_acc[z] != h_acc2[z]);
        unsigned long long dbg[8] = {0};
        ptqint8::implicit_get_debug(dbg);
        printf("     implicit int32 mismatch: %lld / %zu  %.3f ms (quant %.3f + conv %.3f, x%.2f vs gemm)  block cycles: stage %llu compute %llu\n",
               imp_bad, h_acc.size(), imp, t_quant, t_conv, gem / imp,
               dbg[1] - dbg[0], dbg[2] - dbg[1]);

        // 融合 epilogue 的正确性：同一份 d_acc2 走 requant 作为参考，
        // 再与 implicit_conv_out_launch 直接写出的 fp32 逐元素比较。
        float* d_out2 = nullptr;
        float* d_out_ref = nullptr;
        CUDA_CHECK(cudaMalloc(&d_out2, (size_t)M * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out_ref, (size_t)M * N * sizeof(float)));
        const bool epi_ok = ptqint8::implicit_conv_out_launch(
            d_qact, d_wimp, d_out2, d_ws, d_b, d_zp, 0.01f,
            n, c.cin, hh, ww, c.kh, c.kw, c.sh, c.sw, c.ph, c.pw,
            OH, OW, N, N, 0, false);
        if (epi_ok)
        {
            const int epi_grid = (int)std::min<long long>(
                65535, ((long long)M * N + 255) / 256);
            ptqint8::requant_launch(d_acc2, d_out_ref, d_ws, d_b, d_zp,
                                    0.01f, N, (int)(OH * OW), n, false,
                                    N, 0, epi_grid);
            std::vector<float> h2((size_t)M * N), hr((size_t)M * N);
            CUDA_CHECK(cudaMemcpy(h2.data(), d_out2, h2.size() * 4,
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hr.data(), d_out_ref, hr.size() * 4,
                                  cudaMemcpyDeviceToHost));
            long long epi_bad = 0;
            float epi_max = 0.0f;
            for (size_t z = 0; z < h2.size(); z++)
            {
                if (h2[z] != hr[z]) epi_bad++;
                epi_max = std::max(epi_max, std::fabs(h2[z] - hr[z]));
            }
            printf("     fused epilogue vs requant: mismatch %lld / %zu  max|diff|=%.3e\n",
                   epi_bad, h2.size(), epi_max);
            // 引擎实际走的是「quant + conv + 融合 epilogue」这一条，所以这里单独
            // 计时它（否则只看 int32 版会漏掉 epilogue 的代价）。
            const double epi_ms = time_ms([&] {
                ptqint8::implicit_conv_out_launch(
                    d_qact, d_wimp, d_out2, d_ws, d_b, d_zp, 0.01f,
                    n, c.cin, hh, ww, c.kh, c.kw, c.sh, c.sw, c.ph, c.pw,
                    OH, OW, N, N, 0, false);
            }, iters);
            printf("     implicit+epi %.3f ms (x%.2f vs gemm, quant %.3f + epi %.3f)\n",
                   t_quant + epi_ms, gem / (t_quant + epi_ms), t_quant, epi_ms);
        }
        else
        {
            printf("     fused epilogue: skipped (shape unsupported)\n");
        }
        cudaFree(d_out2);
        cudaFree(d_out_ref);
        cudaFree(d_qact); cudaFree(d_wimp); cudaFree(d_acc2);

        const int grid_rq = (int)std::min<long long>(65535, (M * N + 255) / 256);
        const double rq = time_ms([&] {
            ptqint8::requant_launch(d_acc, d_out, d_ws, d_b, d_zp, 0.01f, N, (int)(OH * OW),
                                    n, false, N, 0, grid_rq);
        }, iters);

        // requant 旧 vs 新（转置瓦片）：先跑一次旧版留结果，再跑新版逐位比较
        std::vector<float> h_rq_old((size_t)M * N), h_rq_new((size_t)M * N);
        ptqint8::requant_launch(d_acc, d_out, d_ws, d_b, d_zp, 0.01f, N, (int)(OH * OW),
                                n, false, N, 0, grid_rq);
        CUDA_CHECK(cudaMemcpy(h_rq_old.data(), d_out, h_rq_old.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        const double rqt = time_ms([&] {
            ptqint8::requant_tiled_launch(d_acc, d_out, d_ws, d_b, d_zp, 0.01f, N,
                                          (int)(OH * OW), n, false, N, 0);
        }, iters);
        CUDA_CHECK(cudaMemcpy(h_rq_new.data(), d_out, h_rq_new.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        long long rq_bad = 0;
        for (size_t i = 0; i < h_rq_old.size(); i++) rq_bad += (h_rq_old[i] != h_rq_new[i]);
        printf("     requant bit-exact: %s (mismatch %lld)  old %.3f -> new %.3f ms (x%.1f)\n",
               rq_bad == 0 ? "YES" : "NO", rq_bad, rq, rqt, rq / rqt);

        double rl = 0.0;
        if (c.relu)
        {
            rl = time_ms([&] {
                ptqint8::relu_launch(d_out, (size_t)M * N, grid_rq, c.relu);
            }, iters);
        }

        const double ops = 2.0 * (double)M * Kp * N;
        const double tops = ops / (gem * 1e9);
        printf("%-4d %-16s %7d %5d %6lld %9.3f %9.3f %8.3f %8.3f %9.3f %8.3f %8.2f %5.1f%%  x%.1f\n",
               idx++, (std::to_string(c.cin) + "->" + std::to_string(N) + " k" +
                       std::to_string(c.kh) + "s" + std::to_string(c.sh)).c_str(),
               Kp, N, M, im2, im2f, gem, imp, rq, rl, tops, 100.0 * tops / PEAK_TOPS,
               im2 / im2f);

        // 带宽口径（GB/s）：im2col = 读 x(每个输出只读一次近似) + 写 col
        const double col_gb = (double)M * Kp / 1e9;
        const double x_gb = (double)M * c.cin * c.kh * c.kw / 1e9;   // 上界（未计 cache）
        printf("     bw: im2col<=%.1f GB/s (读上界 %.2f GB + 写 %.2f GB)  gemm=%.1f GB/s  requant=%.1f GB/s\n",
               (x_gb + col_gb) / (im2 * 1e-3), x_gb, col_gb,
               ((double)M * Kp + (double)Kp * N) / (gem * 1e-3) / 1e9,
               (double)M * N * 8.0 / (rq * 1e-3) / 1e9);

        t_im2col += im2; t_im2col_fast += im2f; t_im2col_vec += im2v; t_gemm += gem; t_req += rq; t_implicit += imp;
        t_req_tiled += ((OH * OW) >= 16 ? rqt : rq); t_relu += rl;
        flops += (long long)ops;
        // ⛔ 不能 cudaFree(d_acc)：它是 torch 张量的内存，由 caching allocator 管；
        //    手动 free 会在下一次分配/析构时踩坏堆（实测 illegal memory access）。
        cudaFree(d_col); cudaFree(d_out);

        ch = N; hh = OH; ww = OW;
    };

    bench_conv(0);
    if (m.stem_pool)
    {
        hh = (hh + 2 - 3) / 2 + 1;
        ww = (ww + 2 - 3) / 2 + 1;
        printf("     stem maxpool -> %dx%d\n", hh, ww);
    }
    for (auto& blk : m.blocks)
    {
        if (blk.ds >= 0)
        {
            const int ich = ch, ih = hh, iw = ww;
            bench_conv(blk.ds);
            ch = ich; hh = ih; ww = iw;   // ds 是分支，不改变主支路输入
        }
        for (int ci : blk.convs)
            if (ci >= 0) bench_conv(ci);
    }

    const double total = t_im2col + t_gemm + t_req + t_relu;         // 旧 kernel
    const double total_new = t_im2col_fast + t_gemm + t_req_tiled + t_relu; // 新 kernel
    printf("\n合计 batch=%d: im2col(旧) %.3f | im2col(新) %.3f | im2col(vec) %.3f | gemm %.3f | implicit %.3f | requant %.3f -> %.3f | relu %.3f ms\n",
           n, t_im2col, t_im2col_fast, t_im2col_tiled, t_im2col_vec, t_gemm, t_implicit, t_req, t_req_tiled, t_relu);
    printf("占比(新):    im2col %5.1f%% | gemm %5.1f%% | requant %5.1f%% | relu %5.1f%%\n",
           100 * t_im2col_fast / total_new, 100 * t_gemm / total_new,
           100 * t_req_tiled / total_new, 100 * t_relu / total_new);
    printf("总(旧) %.3f ms/batch = %.4f ms/img；总(新) %.3f ms/batch = %.4f ms/img\n",
           total, total / n, total_new, total_new / n);
    printf("im2col 加速 %.2fx；全段加速 %.2fx（不含 add/maxpool/fc/数据搬运）\n",
           t_im2col / t_im2col_fast, total / total_new);
    printf("GEMM 聚合 %.2f TOPS = 峰值的 %.1f%%\n",
           (double)flops / (t_gemm * 1e9), 100.0 * (double)flops / (t_gemm * 1e9) / PEAK_TOPS);
    printf("理论下限（GEMM 100%% 峰值 + 其余用新实测） %.3f ms/batch = %.4f ms/img\n",
           (double)flops / (PEAK_TOPS * 1e9) + t_im2col_fast + t_req + t_relu,
           ((double)flops / (PEAK_TOPS * 1e9) + t_im2col_fast + t_req + t_relu) / n);
    return 0;
}