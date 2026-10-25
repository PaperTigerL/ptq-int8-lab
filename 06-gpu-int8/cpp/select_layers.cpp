// 逐层 A/B：implicit GEMM vs im2col+cublas —— 生成「按层选择表」的唯一来源。
//
// 为什么单独一个程序而不塞进 bench_layers：
//   bench_layers 是历史 roofline 表（含旧/新 im2col、requant 转置、理论下限），
//   跑一遍几十秒且输出是给人看的；层选择表需要**窄而严格**的口径：
//     · 同一 batch、同一层输入/权重/scale（真实权重，不是全 1 占位）；
//     · 两条路都按引擎实际调用顺序计时（im2col(tiled)+GEMM+requant / quant_c4+conv+融合 epi）；
//     · 先 100 次预热再测（这台卡空闲 1350 MHz、持续负载 1873-1920 MHz，
//       不预热会把频率差 1.4 倍误读成代码差 —— ROOFLINE §10.4）；
//     · 两路**交错**计时取各自最快（避免先后顺序被时钟漂移带偏）；
//     · int32 累加必须逐元素相同（这是「选择只影响性能、不影响数值」的判据）；
//     · 融合 epilogue 的 fp32 输出再与 requant 对照一次（双保险）。
//   任何一项不满足，该层不写入表（宁可退回保守的 cublas）。
//
// 用法：
//   ./select_layers resnet18_folded.bin calib_minmax.bin \
//       [--batch 512] [--warmup 100] [--iters 15] \
//       [--json results/layer_impl.json] [--header cpp/layer_impl_default.h]
//
// 输出（stdout）：逐层表 + 合计；--json/--header 落盘（后者直接给引擎当默认表）。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "int8_engine.cuh"

using ptqmodel::BlockSpec;
using ptqmodel::ConvSpec;
using ptqmodel::Model;
using ptqint8::LayerImpl;
using ptqint8::LayerImplEntry;

static int out_dim(int in, int k, int s, int p) { return (in + 2 * p - k) / s + 1; }

static int grid_for(long long total)
{
    long long g = (total + 255) / 256;
    if (g < 1) g = 1;
    if (g > 65535) g = 65535;
    return (int)g;
}

static int read_clock_mhz()
{
    FILE* p = popen("nvidia-smi --query-gpu=clocks.current.graphics "
                    "--format=csv,noheader,nounits 2>/dev/null", "r");
    int v = 0;
    if (p)
    {
        if (fscanf(p, "%d", &v) != 1) v = 0;
        pclose(p);
    }
    return v;
}

struct Row
{
    int layer = -1;
    std::string shape;
    int cin = 0, cout = 0, kh = 0, kw = 0, sh = 1, sw = 1, ph = 0, pw = 0;
    int h = 0, w = 0;
    long long M = 0;
    int Kpad = 0, batch = 0;
    bool fused = false;              // implicit 是否走融合 epilogue
    bool fp32_ok = false;            // fp32 输出是否在 1e-6 内等价
    double t_col = 0.0, t_imp = 0.0; // ms / 整层 / 该 batch
    LayerImpl choice = LayerImpl::Cublas;
    long long bad32 = -1;
    long long bad_fp32 = -1;
    double max_fp32 = 0.0;
};

static double ev_ms(cudaEvent_t a, cudaEvent_t b)
{
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, a, b);
    return (double)ms;
}

static void write_json(const std::string& path, const std::vector<Row>& rows,
                       int batch, int warmup, int iters, int clock, int group_skips)
{
    std::ofstream f(path);
    if (!f) { fprintf(stderr, "写 %s 失败\n", path.c_str()); return; }
    f << "{\n";
    f << "  \"format\": \"ptq-layer-impl/1\",\n";
    f << "  \"note\": \"select_layers 实测：同 batch、100+ 预热、两路交错计时取最快、int32 逐元素相同\",\n";
    f << "  \"batch\": " << batch << ",\n";
    f << "  \"warmup\": " << warmup << ",\n";
    f << "  \"iters\": " << iters << ",\n";
    f << "  \"clock_mhz\": " << clock << ",\n";
    f << "  \"grouped_skipped\": " << group_skips << ",\n";
    f << "  \"layers\": [\n";
    for (size_t i = 0; i < rows.size(); i++)
    {
        const Row& r = rows[i];
        f << "    {\"layer\": " << r.layer
          << ", \"shape\": \"" << r.shape << "\""
          << ", \"cin\": " << r.cin << ", \"cout\": " << r.cout
          << ", \"kh\": " << r.kh << ", \"kw\": " << r.kw
          << ", \"sh\": " << r.sh << ", \"sw\": " << r.sw
          << ", \"ph\": " << r.ph << ", \"pw\": " << r.pw
          << ", \"h\": " << r.h << ", \"w\": " << r.w
          << ", \"impl\": \"" << (r.choice == LayerImpl::Implicit ? "implicit" : "cublas") << "\""
          << ", \"t_cublas_ms\": " << r.t_col
          << ", \"t_implicit_ms\": " << r.t_imp
          << ", \"implicit_fused_epilogue\": " << (r.fused ? "true" : "false")
          << ", \"int32_mismatch\": " << r.bad32
          << ", \"fp32_bit_exact\": " << (r.bad_fp32 == 0 ? "true" : "false")
          << ", \"fp32_mismatch\": " << r.bad_fp32
          << ", \"fp32_maxdiff\": " << r.max_fp32
          << "}" << (i + 1 < rows.size() ? "," : "") << "\n";
    }
    f << "  ]\n}\n";
    printf("[select] JSON → %s\n", path.c_str());
}

static void write_header(const std::string& path, const std::vector<Row>& rows, int batch)
{
    std::ofstream f(path);
    if (!f) { fprintf(stderr, "写 %s 失败\n", path.c_str()); return; }
    f << "// 自动生成：./select_layers resnet18_folded.bin calib_minmax.bin --batch " << batch
      << " --json ... --header ...\n"
      << "// 不要手改：改判据请改 select_layers.cpp，然后重跑生成。\n"
      << "// t_cublas_ms / t_implicit_ms 是生成时（batch=" << batch
      << "）的实测，供人读，不参与运行期判断。\n"
      << "// ⚠️ 本文件在 layer_impl.h 的 namespace ptqint8 内部被 include，不能再开 namespace。\n"
      << "#pragma once\n\n"
      << "inline const LayerImplEntry kDefaultLayerImpl[] = {\n";
    for (const Row& r : rows)
    {
        char buf[512];
        snprintf(buf, sizeof(buf),
                 "    {%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, LayerImpl::%s, %.4f, %.4f, %d, false},",
                 r.layer, r.cin, r.cout, r.kh, r.kw, r.sh, r.sw, r.ph, r.pw, r.h, r.w,
                 r.choice == LayerImpl::Implicit ? "Implicit" : "Cublas",
                 r.t_col, r.t_imp, r.batch);
        f << buf << "\n";
    }
    f << "};\n"
      << "inline constexpr int kDefaultLayerImplCount = " << rows.size() << ";\n";
    printf("[select] header → %s（%zu 条）\n", path.c_str(), rows.size());
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        fprintf(stderr,
                "usage: %s <model.bin> <calib.bin> [--batch N] [--warmup N] [--iters N] "
                "[--json out.json] [--header out.h]\n", argv[0]);
        return 2;
    }
    const char* model_path = argv[1];
    const char* calib_path = argv[2];
    int batch = 512, warmup = 100, iters = 15;
    std::string json_path, header_path;
    for (int i = 3; i < argc; i++)
    {
        if (!strcmp(argv[i], "--batch") && i + 1 < argc) batch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--json") && i + 1 < argc) json_path = argv[++i];
        else if (!strcmp(argv[i], "--header") && i + 1 < argc) header_path = argv[++i];
        else { fprintf(stderr, "未知参数 %s\n", argv[i]); return 2; }
    }
    if (batch < 32) { fprintf(stderr, "batch 太小（<32）\n"); return 2; }
    if (warmup < 0 || iters < 1) { fprintf(stderr, "warmup/iters 非法\n"); return 2; }

    Model m;
    if (!m.load(model_path)) return 1;
    std::vector<std::pair<float, float>> ranges;
    {
        std::string err;
        if (!load_calib(calib_path, m.convs.size(), &ranges, &err))
        {
            fprintf(stderr, "[FAIL] %s\n", err.c_str());
            return 1;
        }
    }
    std::vector<ptq::TensorQuant> aq(m.convs.size());
    for (size_t i = 0; i < m.convs.size(); i++)
        aq[i] = ptq::tensor_quant_from_range(ranges[i].first, ranges[i].second, 8,
                                             ptq::ActRangeMode::AbsMax);

    // 输入：非零随机（0 输入会让 padding/边界错误看起来像"通过"）
    size_t max_x = 0;
    for (auto& c : m.convs)
        max_x = std::max(max_x, (size_t)batch * c.cin * 32 * 32);
    max_x = std::max(max_x, (size_t)batch * 64 * 32 * 32);
    float* d_x = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, max_x * sizeof(float)));
    {
        std::vector<float> hx(max_x);
        std::mt19937 rng(12345);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& v : hx) v = dist(rng);
        CUDA_CHECK(cudaMemcpy(d_x, hx.data(), max_x * sizeof(float), cudaMemcpyHostToDevice));
    }

    const int clock0 = read_clock_mhz();
    printf("model: %zu convs | batch=%d | warmup=%d | iters=%d | clock=%d MHz（开始）\n",
           m.convs.size(), batch, warmup, iters, clock0);

    std::vector<Row> rows;
    std::vector<char> done(m.convs.size(), 0);
    int group_skips = 0;

    // 当前层的输入形状（随拓扑推进；ds 分支由调用方保存/恢复）
    int ch = 3, hh = 32, ww = 32;

    // 逐层 A/B 主体（形状按拓扑推进：与 bench_layers 同一条前进路径）
    // 说明：ch/hh/ww 是外层变量，进入时是这一层的输入形状，退出时推进到输出形状；
    //       ds 分支由调用方保存/恢复，因为它与主支路共用同一个输入。
    auto bench_one = [&](int li) {
        if (li < 0 || (size_t)li >= m.convs.size()) return;
        if (done[li]) return;
        done[li] = 1;
        const ConvSpec& c = m.convs[li];
        const int ch_in = ch, hh_in = hh, ww_in = ww;
        const int OH = out_dim(hh_in, c.kh, c.sh, c.ph);
        const int OW = out_dim(ww_in, c.kw, c.sw, c.pw);
        const int Lpos = OH * OW;
        const int N = c.cout;
        const long long M = (long long)batch * Lpos;
        const int K = c.cin * c.kh * c.kw;
        const int Kp = ((K + 7) / 8) * 8;
        const int Cpad = (c.cin + 3) & ~3;
        const int Ktap = c.kh * c.kw * Cpad;
        const int KpadTap = ((Ktap + 7) / 8) * 8;
        const int Kcap = std::max(Kp, KpadTap);

        Row r;
        r.layer = li; r.batch = batch;
        r.cin = c.cin; r.cout = N; r.kh = c.kh; r.kw = c.kw;
        r.sh = c.sh; r.sw = c.sw; r.ph = c.ph; r.pw = c.pw;
        r.h = hh_in; r.w = ww_in; r.M = M; r.Kpad = Kp;
        char shape[64];
        snprintf(shape, sizeof(shape), "%d->%d k%ds%d", c.cin, N, c.kh, c.sh);
        r.shape = shape;

        if (c.groups != 1)
        {
            group_skips++;
            r.choice = LayerImpl::Cublas;
            r.bad32 = r.bad_fp32 = -2;   // -2 = 未测（分组层走专用 kernel）
            rows.push_back(r);
            printf("%-3d %-16s skip（groups=%d，专用 kernel 不参与选择）\n",
                   li, shape, c.groups);
            return;
        }

        // ---- 权重/scale/bias/zp：与引擎 build 完全同口径 ----
        ptq::ChannelQuant cq = ptq::quantize_weight_per_channel_sym(
            c.weight.data(), N, K, 8);
        std::vector<int8_t> w_dev((size_t)Kp * N, 0);
        for (int o = 0; o < N; o++)
            for (int k = 0; k < K; k++)
                w_dev[(size_t)k * N + o] = cq.q[(size_t)o * K + k];
        std::vector<float> corr(N, 0.0f);
        for (int o = 0; o < N; o++)
        {
            long long sum = 0;
            for (int k = 0; k < K; k++) sum += cq.q[(size_t)o * K + k];
            corr[o] = (float)(-(double)aq[li].zero_point * (double)aq[li].scale *
                              (double)cq.scale[o] * (double)sum);
        }
        std::vector<int8_t> wimp;
        ptqint8::repack_implicit_weight(cq.q.data(), N, c.cin, c.kh, c.kw, &wimp);
        const float ascale = aq[li].scale;
        const bool fuse_relu = (c.relu == 1);
        const bool relu6 = (c.relu == 2);

        int8_t* d_col = nullptr;
        int8_t* d_qact = nullptr;
        int8_t* d_w = nullptr;
        int8_t* d_wimp = nullptr;
        float* d_ws = nullptr;
        float* d_b = nullptr;
        float* d_zp = nullptr;
        int32_t* d_acc_imp = nullptr;
        float* d_out = nullptr;
        float* d_out2 = nullptr;
        CUDA_CHECK(cudaMalloc(&d_col, (size_t)M * Kcap));
        CUDA_CHECK(cudaMalloc(&d_qact, (size_t)batch * ((c.cin + 3) / 4) * 4 * hh_in * ww_in));
        CUDA_CHECK(cudaMalloc(&d_w, (size_t)Kp * N));
        CUDA_CHECK(cudaMalloc(&d_wimp, wimp.size()));
        CUDA_CHECK(cudaMalloc(&d_ws, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_zp, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_acc_imp, (size_t)M * N * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_out, (size_t)M * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out2, (size_t)M * N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_w, w_dev.data(), w_dev.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_wimp, wimp.data(), wimp.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ws, cq.scale.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, c.bias.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_zp, corr.data(), N * sizeof(float), cudaMemcpyHostToDevice));

        auto run_col = [&]() {
            if (Lpos >= 4)
                ptqint8::quant_im2col_tiled_launch(
                    d_x, d_col, ascale, 127, batch, ch_in, hh_in, ww_in,
                    c.kh, c.kw, c.sh, c.sw, c.ph, c.pw, OH, OW, K, Kp, 0);
            else
                ptqint8::quant_im2col_fast_launch(
                    d_x, d_col, ascale, 0, 127, batch, ch_in, hh_in, ww_in,
                    c.kh, c.kw, c.sh, c.sw, c.ph, c.pw, OH, OW, K, Kp, 0);
            torch::Tensor acc_t = ptqint8::int8_gemm(d_col, M, Kp, d_w, N);
            const int32_t* acc = acc_t.data_ptr<int32_t>();
            if (Lpos >= 16)
                ptqint8::requant_tiled_launch(acc, d_out, d_ws, d_b, d_zp, ascale,
                                              N, Lpos, batch, fuse_relu, N, 0);
            else
                ptqint8::requant_launch(acc, d_out, d_ws, d_b, d_zp, ascale,
                                        N, Lpos, batch, fuse_relu, N, 0,
                                        grid_for((long long)M * N));
            if (relu6)
                ptqint8::relu_launch(d_out, (size_t)M * N, grid_for((long long)M * N), 2);
        };

        // 融合 epilogue 是否支持：与引擎一样先试一次，失败则退回 acc+requant。
        bool fused = false;
        {
            CUDA_CHECK(cudaMemset(d_out2, 0, (size_t)M * N * sizeof(float)));
            ptqint8::quant_act_c4_launch(d_x, d_qact, ascale, 127, batch, ch_in, 0,
                                         c.cin, hh_in, ww_in);
            fused = ptqint8::implicit_conv_out_launch(
                d_qact, d_wimp, d_out2, d_ws, d_b, d_zp, ascale,
                batch, c.cin, hh_in, ww_in, c.kh, c.kw, c.sh, c.sw, c.ph, c.pw,
                OH, OW, N, N, 0, fuse_relu, true);
            CUDA_CHECK(cudaGetLastError());
        }
        auto run_imp = [&]() {
            ptqint8::quant_act_c4_launch(d_x, d_qact, ascale, 127, batch, ch_in, 0,
                                         c.cin, hh_in, ww_in);
            if (fused)
            {
                ptqint8::implicit_conv_out_launch(
                    d_qact, d_wimp, d_out2, d_ws, d_b, d_zp, ascale,
                    batch, c.cin, hh_in, ww_in, c.kh, c.kw, c.sh, c.sw, c.ph, c.pw,
                    OH, OW, N, N, 0, fuse_relu, true);
            }
            else
            {
                ptqint8::implicit_conv_launch(
                    d_qact, d_wimp, d_acc_imp,
                    batch, c.cin, hh_in, ww_in, c.kh, c.kw, c.sh, c.sw, c.ph, c.pw,
                    OH, OW, N);
                if (Lpos >= 16)
                    ptqint8::requant_tiled_launch(d_acc_imp, d_out2, d_ws, d_b, d_zp,
                                                  ascale, N, Lpos, batch, fuse_relu, N, 0);
                else
                    ptqint8::requant_launch(d_acc_imp, d_out2, d_ws, d_b, d_zp, ascale,
                                            N, Lpos, batch, fuse_relu, N, 0,
                                            grid_for((long long)M * N));
            }
            if (relu6)
                ptqint8::relu_launch(d_out2, (size_t)M * N, grid_for((long long)M * N), 2);
        };

        // ---- 预热：100 次，两路交替（让时钟一起爬满） ----
        for (int i = 0; i < warmup; i++)
        {
            if (i & 1) { run_col(); run_imp(); }
            else { run_imp(); run_col(); }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());

        // ---- 计时：两路交错，每次一个 event 对；取各自最快 ----
        std::vector<cudaEvent_t> ev(4 * (size_t)iters);
        for (auto& e : ev) CUDA_CHECK(cudaEventCreate(&e));
        for (int i = 0; i < iters; i++)
        {
            CUDA_CHECK(cudaEventRecord(ev[4 * i + 0]));
            run_col();
            CUDA_CHECK(cudaEventRecord(ev[4 * i + 1]));
            CUDA_CHECK(cudaEventRecord(ev[4 * i + 2]));
            run_imp();
            CUDA_CHECK(cudaEventRecord(ev[4 * i + 3]));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
        double best_col = 1e30, best_imp = 1e30;
        for (int i = 0; i < iters; i++)
        {
            best_col = std::min(best_col, ev_ms(ev[4 * i + 0], ev[4 * i + 1]));
            best_imp = std::min(best_imp, ev_ms(ev[4 * i + 2], ev[4 * i + 3]));
        }
        for (auto& e : ev) cudaEventDestroy(e);

        // ---- 正确性 1：int32 累加逐元素相同 ----
        {
            if (Lpos >= 4)
                ptqint8::quant_im2col_tiled_launch(
                    d_x, d_col, ascale, 127, batch, ch_in, hh_in, ww_in,
                    c.kh, c.kw, c.sh, c.sw, c.ph, c.pw, OH, OW, K, Kp, 0);
            else
                ptqint8::quant_im2col_fast_launch(
                    d_x, d_col, ascale, 0, 127, batch, ch_in, hh_in, ww_in,
                    c.kh, c.kw, c.sh, c.sw, c.ph, c.pw, OH, OW, K, Kp, 0);
            torch::Tensor acc_t = ptqint8::int8_gemm(d_col, M, Kp, d_w, N);
            const int32_t* acc_c = acc_t.data_ptr<int32_t>();
            ptqint8::implicit_conv_launch(
                d_qact, d_wimp, d_acc_imp,
                batch, c.cin, hh_in, ww_in, c.kh, c.kw, c.sh, c.sw, c.ph, c.pw,
                OH, OW, N);
            CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<int32_t> hc((size_t)M * N), hi((size_t)M * N);
            CUDA_CHECK(cudaMemcpy(hc.data(), acc_c, hc.size() * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hi.data(), d_acc_imp, hi.size() * 4, cudaMemcpyDeviceToHost));
            long long bad = 0;
            for (size_t z = 0; z < hc.size(); z++) bad += (hc[z] != hi[z]);
            r.bad32 = bad;
            // 让 acc_t 活到 memcpy 之后再析构
        }
        // ---- 正确性 2：最终 fp32 输出（融合 epi vs requant）逐元素对照 ----
        run_col();
        CUDA_CHECK(cudaMemcpy(d_out2, d_out, (size_t)M * N * sizeof(float),
                              cudaMemcpyDeviceToDevice));
        {
            std::vector<float> hc((size_t)M * N);
            CUDA_CHECK(cudaMemcpy(hc.data(), d_out2, hc.size() * 4, cudaMemcpyDeviceToHost));
            run_imp();
            std::vector<float> hi2((size_t)M * N);
            CUDA_CHECK(cudaMemcpy(hi2.data(), d_out2, hi2.size() * 4, cudaMemcpyDeviceToHost));
            long long bad = 0;
            double mx = 0.0;
            for (size_t z = 0; z < hc.size(); z++)
            {
                if (hc[z] != hi2[z]) { bad++; mx = std::max(mx, (double)std::fabs(hc[z] - hi2[z])); }
            }
            r.bad_fp32 = bad;
            r.max_fp32 = mx;
        }

        r.fused = fused;
        r.t_col = best_col;
        r.t_imp = best_imp;
        // 硬判据：int32 累加必须逐元素相同（任务要求，也是「只换实现不改数值」的前提）。
        // 软判据：融合 epilogue 与 requant 的 fp32 在少数形状上差几个 ulp（算式等价、
        // 舍入顺序不同），用 1e-6 绝对容差兜底；超过即视为实现不等价，该层退回 cublas。
        // 这条容差不是"放水"：端到端精度另有 verify.sh 的 10000 张硬门槛。
        const bool fp32_ok = (r.bad_fp32 == 0) || (r.max_fp32 <= 1e-6);
        r.fp32_ok = fp32_ok;
        if (r.bad32 == 0 && fp32_ok)
            r.choice = (best_imp < best_col) ? LayerImpl::Implicit : LayerImpl::Cublas;
        else
            r.choice = LayerImpl::Cublas;
        rows.push_back(r);

        printf("%-3d %-16s h=%-2d M=%-7lld %7.3f %7.3f  %-8s 省 %6.3f  "
               "int32 %s  fp32 %s(%.1e)%s\n",
               li, shape, hh, M, best_col, best_imp,
               r.choice == LayerImpl::Implicit ? "implicit" : "cublas",
               std::fabs(best_col - best_imp),
               r.bad32 == 0 ? "exact" : "BAD",
               r.bad_fp32 == 0 ? "exact" : (fp32_ok ? "ulp" : "BAD"), r.max_fp32,
               fused ? "  [fused epi]" : "  [acc+requant]");

        cudaFree(d_col); cudaFree(d_qact); cudaFree(d_w); cudaFree(d_wimp);
        cudaFree(d_ws); cudaFree(d_b); cudaFree(d_zp);
        cudaFree(d_acc_imp); cudaFree(d_out); cudaFree(d_out2);
        ch = N; hh = OH; ww = OW;   // 推进到本层输出 = 下一层输入
    };

    // 按导出拓扑走：stem → (maxpool) → 每个 block 的 ds 分支 → 主支路
    bench_one(0);
    if (m.stem_pool)
    {
        hh = (hh + 2 - 3) / 2 + 1;
        ww = (ww + 2 - 3) / 2 + 1;
        printf("     stem maxpool → %dx%d\n", hh, ww);
    }
    for (const BlockSpec& blk : m.blocks)
    {
        if (blk.ds >= 0)
        {
            const int ich = ch, ih = hh, iw = ww;   // ds 是分支，不推进主支路
            bench_one(blk.ds);
            ch = ich; hh = ih; ww = iw;
        }
        for (int ci : blk.convs)
            if (ci >= 0) bench_one(ci);
    }
    CUDA_CHECK(cudaFree(d_x));

    std::sort(rows.begin(), rows.end(),
              [](const Row& a, const Row& b) { return a.layer < b.layer; });

    double sum_col = 0, sum_imp = 0, sum_mix = 0;
    int n_imp = 0, n_col = 0, n_bad = 0;
    for (const Row& r : rows)
    {
        sum_col += r.t_col;
        sum_imp += r.t_imp;
        sum_mix += (r.choice == LayerImpl::Implicit) ? r.t_imp : r.t_col;
        if (r.choice == LayerImpl::Implicit) n_imp++; else n_col++;
        if (r.bad32 > 0 || (r.bad32 == 0 && !r.fp32_ok)) n_bad++;   // -2 = 分组层未测
    }
    const int clock1 = read_clock_mhz();
    printf("\n合计（batch=%d，%d 层可测 + %d 层分组跳过）：\n", batch,
           (int)rows.size() - group_skips, group_skips);
    printf("  cublas 全走 %.3f ms/batch = %.4f ms/img\n", sum_col, sum_col / batch);
    printf("  implicit 全走 %.3f ms/batch = %.4f ms/img\n", sum_imp, sum_imp / batch);
    printf("  按层混合   %.3f ms/batch = %.4f ms/img（%d implicit / %d cublas）\n",
           sum_mix, sum_mix / batch, n_imp, n_col);
    printf("  混合相比 cublas 省 %.3f ms/batch（%.1f%%）\n", sum_col - sum_mix,
           100.0 * (sum_col - sum_mix) / std::max(1e-9, sum_col));
    printf("  int32/fp32 逐元素不一致的层数：%d（必须 0）\n", n_bad);
    printf("  clock: 开始 %d MHz → 结束 %d MHz\n", clock0, clock1);

    if (!json_path.empty())
        write_json(json_path, rows, batch, warmup, iters, clock1, group_skips);
    if (!header_path.empty())
        write_header(header_path, rows, batch);
    return n_bad == 0 ? 0 : 1;
}
