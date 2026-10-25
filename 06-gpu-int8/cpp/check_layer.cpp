// 单层数值校验：GPU int8 conv vs CPU 参考实现。
//
// 为什么单独写这个：全网精度 9.40%（=随机猜）时，**不能靠调参碰运气**，
// 必须把问题隔离到单层，逐元素对比。上一轮就是靠这个方法找到 ncnn batch
// 的 virtual 分派问题的。
//
// 参考实现刻意写得又慢又直白（host 循环 + fp32 累加），
// 它唯一的作用是「正确」，不参与性能对比。
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "int8_engine.cuh"
#include <torch/torch.h>

static bool g_quiet = false;

// CPU 参考：per-channel sym 权重 + per-tensor asym 激活
// 直接用 ptq_quant.h 的口径，保证测的是 kernel/GEMM 而非量化语义本身。
static void cpu_ref_conv(const ConvSpec& c,
                         const float* in, int n, int h, int w,
                         float ascale, int azp, int a_bits,
                         const std::vector<int8_t>& wq, const std::vector<float>& ws,
                         float* out)
{
    const int OH = (h + 2 * c.ph - c.kh) / c.sh + 1;
    const int OW = (w + 2 * c.pw - c.kw) / c.sw + 1;
    const int K = c.cin * c.kh * c.kw;
    const int a_qmax = (1 << (a_bits - 1)) - 1;   // 对称码空间，与引擎一致

    // zp 修正
    std::vector<double> zp_corr(c.cout, 0.0);
    for (int o = 0; o < c.cout; o++)
    {
        long long s = 0;
        for (int k = 0; k < K; k++) s += wq[(size_t)o * K + k];
        zp_corr[o] = -(double)azp * (double)ascale * (double)ws[o] * (double)s;
    }

    // ⚠️ 这里必须**直接用码字**做整数累加，不能"先反量化成 float、算完再除
    // scale 量化回去"—— 那是双重舍入，会引入 ±1 码字的误差，
    // 实测表现为 max|diff| ≈ 13.7、63.5% 元素超 1e-3。
    // GPU 路径是 fp32 → 一次量化 → int8 → int32 累加，所以参考必须是纯整数。
    std::vector<int> xq((size_t)c.cin * h * w);
    for (int b = 0; b < n; b++)
    {
        const float* src = in + (size_t)b * c.cin * h * w;
        for (int i = 0; i < c.cin * h * w; i++)
        {
            double t = std::floor(src[i] / ascale + 0.5);   // round-half-away
            if (t < -(double)(a_qmax + 1)) t = -(double)(a_qmax + 1);
            if (t > (double)a_qmax) t = (double)a_qmax;
            xq[i] = (int)t;
        }
        for (int co = 0; co < c.cout; co++)
        {
            for (int oh = 0; oh < OH; oh++)
            {
                for (int ow = 0; ow < OW; ow++)
                {
                    long long acc = 0;
                    for (int ci = 0; ci < c.cin; ci++)
                        for (int i = 0; i < c.kh; i++)
                            for (int j = 0; j < c.kw; j++)
                            {
                                const int ih = oh * c.sh + i - c.ph;
                                const int iw = ow * c.sw + j - c.pw;
                                // padding 区不贡献 —— 与 GPU kernel 里
                                // "越界先 v=0.0 再量化得到码字 -zp" 的效果等价于
                                // 「这一项乘 0」吗？不等价！
                                //   -zp 在 int8 里可能非 0（zp>128 时甚至溢出）。
                                //   所以正确做法是显式跳过，与 GPU 侧对齐后才敢下结论。
                                if (ih < 0 || ih >= h || iw < 0 || iw >= w) continue;
                                const int qi = xq[((size_t)ci * h + ih) * w + iw];
                                const int qidx = (ci * c.kh + i) * c.kw + j;
                                acc += (long long)qi * (long long)wq[(size_t)co * K + qidx];
                            }
                    double y = (double)acc * (double)ws[co] * (double)ascale
                             + c.bias[co] + zp_corr[co];
                    if (y < 0.0) y = 0.0;
                    out[((size_t)b * c.cout + co) * OH * OW + (size_t)oh * OW + ow] = (float)y;
                }
            }
        }
    }
}

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        fprintf(stderr, "usage: %s <model.bin> <calib.bin> <layer_idx> [--n N]\n", argv[0]);
        return 2;
    }
    const int li = atoi(argv[3]);
    int n = 4, h = 32, w = 32;
    for (int i = 4; i < argc; i++)
        if (!strcmp(argv[i], "--n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--h") && i + 1 < argc) h = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--w") && i + 1 < argc) w = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--quiet")) g_quiet = true;

    Model model;
    if (!model.load(argv[1])) return 1;
    std::vector<std::pair<float, float>> ranges;
    {
        std::string cerr;
        if (!load_calib(argv[2], model.convs.size(), &ranges, &cerr))
        {
            fprintf(stderr, "[FAIL] 校准文件不可用：%s\n", cerr.c_str());
            return 1;
        }
    }
    if (li < 0 || li >= (int)model.convs.size()) { fprintf(stderr, "层号越界\n"); return 2; }

    const ConvSpec& c = model.convs[li];
    if (c.groups != 1)
    {
        fprintf(stderr, "[FAIL] check_layer 的参考实现只支持 groups=1（本层 groups=%d）；"
                        "分组/depthwise 请用整网对齐\n", c.groups);
        return 2;
    }
    const int K = c.cin * c.kh * c.kw;
    const int Kpad = ((K + 7) / 8) * 8;
    const int OH = (h + 2 * c.ph - c.kh) / c.sh + 1;
    const int OW = (w + 2 * c.pw - c.kw) / c.sw + 1;

    printf("层 %d: cout=%d cin=%d k=%dx%d s=%d p=%d | K=%d Kpad=%d OH=%d OW=%d\n",
           li, c.cout, c.cin, c.kh, c.kw, c.sh, c.ph, K, Kpad, OH, OW);

    // 用 CPU 量化出参考权重与激活 scale
    const int w_bits = 8, a_bits = 8;
    ptq::ChannelQuant cq = ptq::quantize_weight_per_channel_sym(c.weight.data(), c.cout, K, w_bits);
    // 直接调用引擎同一函数，避免参考实现与引擎口径漂移：
    //   默认 AbsMax = max(|lo|,|hi|)/qmax，与 09 校准一致；
    //   引擎加 --act-minmax 时，这里要显式传 MinMax 才能对上。
    const ptq::TensorQuant aq = ptq::tensor_quant_from_range(
        ranges[li].first, ranges[li].second, a_bits);
    const float ascale = aq.scale;
    const int azp = aq.zero_point;
    const int a_qmax = (1 << (a_bits - 1)) - 1;

    // 随机输入（落在 [0, hi] 内，与真实激活同域）
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> ud(ranges[li].first, ranges[li].second);
    std::vector<float> in((size_t)n * c.cin * h * w);
    for (auto& v : in) v = ud(rng);

    printf("\n[CPU 参考]\n");
    std::vector<float> ref((size_t)n * c.cout * OH * OW);
    cpu_ref_conv(c, in.data(), n, h, w, ascale, azp, a_bits, cq.q, cq.scale, ref.data());

    // [GPU 路径]
    printf("\n[GPU int8]\n");
    ptqint8::CublasCtx ctx;
    ctx.init();
    std::vector<int8_t> wdev((size_t)Kpad * c.cout, 0);
    for (int o = 0; o < c.cout; o++)
        for (int k = 0; k < K; k++)
            wdev[(size_t)k * c.cout + o] = cq.q[(size_t)o * K + k];
    int8_t* w_gpu = nullptr;
    float* ws_gpu = nullptr; float* b_gpu = nullptr; float* zp_gpu = nullptr;
    CUDA_CHECK(cudaMalloc(&w_gpu, wdev.size()));
    CUDA_CHECK(cudaMemcpy(w_gpu, wdev.data(), wdev.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&ws_gpu, c.cout * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(ws_gpu, cq.scale.data(), c.cout * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&b_gpu, c.cout * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(b_gpu, c.bias.data(), c.cout * sizeof(float), cudaMemcpyHostToDevice));
    std::vector<float> corr(c.cout, 0.0f);
    for (int o = 0; o < c.cout; o++)
    {
        long long s = 0;
        for (int k = 0; k < K; k++) s += wdev[(size_t)k * c.cout + o];
        corr[o] = (float)(-(double)azp * (double)ascale * (double)cq.scale[o] * (double)s);
    }
    CUDA_CHECK(cudaMalloc(&zp_gpu, corr.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(zp_gpu, corr.data(), corr.size() * sizeof(float), cudaMemcpyHostToDevice));

    float* d_in = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, in.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, in.data(), in.size() * sizeof(float), cudaMemcpyHostToDevice));

    const long long L = OH * OW, M = (long long)n * L;
    int8_t* col = nullptr;
    CUDA_CHECK(cudaMalloc(&col, (size_t)M * Kpad));
    ptqint8::quant_im2col_launch(d_in, col, ascale, azp, a_qmax,
                                 n, c.cin, h, w, c.kh, c.kw, c.sh, c.sw, c.ph, c.pw,
                                 OH, OW, K, Kpad, 0,
                                 (int)std::min<long long>(65535, (M * Kpad + 255) / 256));
    int32_t* acc = nullptr;
    CUDA_CHECK(cudaMalloc(&acc, (size_t)M * c.cout * sizeof(int32_t)));
    torch::Tensor acc_t = ptqint8::int8_gemm(col, M, Kpad, w_gpu, c.cout);
    acc = acc_t.data_ptr<int32_t>();
    float* got = nullptr;
    CUDA_CHECK(cudaMalloc(&got, (size_t)M * c.cout * sizeof(float)));
    ptqint8::requant_launch(acc, got, ws_gpu, b_gpu, zp_gpu, ascale,
                            c.cout, (int)L, n, true, c.cout, 0,
                            (int)std::min<long long>(65535, (M * c.cout + 255) / 256));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> gout((size_t)M * c.cout);
    CUDA_CHECK(cudaMemcpy(gout.data(), got, gout.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // ---- 中间量 dump：GPU 的 im2col 与 int32 ----
    // 目的是把「量化 / im2col / GEMM / requant」四段隔离开，
    // 逐段与 CPU 参考比，而不是只看最终输出然后瞎猜。
    {
        std::vector<int8_t> gcol((size_t)M * Kpad);
        CUDA_CHECK(cudaMemcpy(gcol.data(), col, gcol.size(), cudaMemcpyDeviceToHost));
        std::vector<int32_t> gacc((size_t)M * c.cout);
        CUDA_CHECK(cudaMemcpy(gacc.data(), acc, gacc.size() * 4, cudaMemcpyDeviceToHost));

        // ---- implicit GEMM 的 int32 逐元素对照 ----
        // 判据是 0 mismatch：A/B 都是 int8，整数和与累加顺序无关。
        std::vector<int8_t> wimp;
        ptqint8::repack_implicit_weight(cq.q.data(), c.cout, c.cin, c.kh, c.kw, &wimp);
        int8_t* wimp_gpu = nullptr;
        CUDA_CHECK(cudaMalloc(&wimp_gpu, wimp.size()));
        CUDA_CHECK(cudaMemcpy(wimp_gpu, wimp.data(), wimp.size(), cudaMemcpyHostToDevice));
        int32_t* acc2 = nullptr;
        CUDA_CHECK(cudaMalloc(&acc2, (size_t)M * c.cout * sizeof(int32_t)));
        const int C4 = (c.cin + 3) / 4;
        int8_t* qact = nullptr;
        CUDA_CHECK(cudaMalloc(&qact, (size_t)n * C4 * 4 * h * w));
        ptqint8::quant_act_c4_launch(d_in, qact, ascale, a_qmax, n, c.cin, 0,
                                     c.cin, h, w);
        const bool imp_ok = ptqint8::implicit_conv_launch(
            qact, wimp_gpu, acc2, n, c.cin, h, w,
            c.kh, c.kw, c.sh, c.sw, c.ph, c.pw, OH, OW, c.cout);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<int32_t> gacc2((size_t)M * c.cout);
        CUDA_CHECK(cudaMemcpy(gacc2.data(), acc2, gacc2.size() * 4, cudaMemcpyDeviceToHost));
        long long imp_bad = 0; int32_t imp_max = 0;
        long long imp_first = -1;
        for (size_t i = 0; i < gacc2.size(); i++)
        {
            const int32_t d = gacc[i] > gacc2[i] ? gacc[i] - gacc2[i] : gacc2[i] - gacc[i];
            if (d) { imp_bad++; if (d > imp_max) imp_max = d; if (imp_first < 0) imp_first = (long long)i; }
        }
        printf("implicit GEMM: %s | int32 mismatch %lld / %zu | max|diff| %d | first %lld\n",
               imp_ok ? "launched" : "UNSUPPORTED", imp_bad, gacc2.size(), imp_max, imp_first);
        cudaFree(qact); cudaFree(acc2); cudaFree(wimp_gpu);

        // CPU 参考的 im2col 与 acc。
        // ⚠️ 这里必须**按 n 分别量化** —— 上一版对所有 row 都用 batch 0 的
        //    xqi，于是 n>0 的所有元素天然不一致，报告出"84% 不一致"
        //    的假象。**校验代码自己的 bug 被误读成 kernel 的 bug**，
        //    这已经是第三次踩"探针错误伪装成实现错误"（前两次：int8 编码溢出、
        //    move 后用已移空的 vector）。
        std::vector<int> xqi((size_t)n * c.cin * h * w);
        for (size_t i = 0; i < xqi.size(); i++)
        {
            double t = std::floor(in[i] / ascale + 0.5);
            if (t < -(double)(a_qmax + 1)) t = -(double)(a_qmax + 1);
            if (t > (double)a_qmax) t = (double)a_qmax;
            xqi[i] = (int)t;
        }
        long long colbad = 0;
        long long accbad = 0;
        int32_t first_acc_g = 0, first_acc_r = 0;
        long long first_acc_idx = -1;
        for (long long row = 0; row < M; row++)
        {
            const int ow = (int)(row % OW);
            const long long t = row / OW;
            const int oh = (int)(t % OH);
            const int nn = (int)(t / OH);
            for (int k = 0; k < Kpad; k++)
            {
                int ref_q = 0;
                if (k < K)
                {
                    const int j = k % c.kw, t2 = k / c.kw;
                    const int i = t2 % c.kh, cc = t2 / c.kh;
                    const int ih = oh * c.sh + i - c.ph;
                    const int iw = ow * c.sw + j - c.pw;
                    if (ih >= 0 && ih < h && iw >= 0 && iw < w)
                        ref_q = xqi[((size_t)nn * c.cin + cc) * h * w + (size_t)ih * w + iw];
                }
                if ((int)gcol[(size_t)row * Kpad + k] != ref_q) colbad++;
            }
            for (int co = 0; co < c.cout; co++)
            {
                long long a = 0;
                for (int k = 0; k < K; k++)
                {
                    const int j = k % c.kw, t2 = k / c.kw;
                    const int i = t2 % c.kh, cc = t2 / c.kh;
                    const int ih = oh * c.sh + i - c.ph;
                    const int iw = ow * c.sw + j - c.pw;
                    if (ih < 0 || ih >= h || iw < 0 || iw >= w) continue;
                    a += (long long)xqi[((size_t)nn * c.cin + cc) * h * w
                                          + (size_t)ih * w + iw]
                       * (long long)cq.q[(size_t)co * K + k];
                }
                if ((int32_t)a != gacc[(size_t)row * c.cout + co])
                {
                    accbad++;
                    if (first_acc_idx < 0)
                    {
                        first_acc_idx = row * c.cout + co;
                        first_acc_g = gacc[(size_t)row * c.cout + co];
                        first_acc_r = (int32_t)a;
                    }
                }
            }
        }
        // 逐点对照（默认开）。定位问题时它是最直接的证据：
        // 能看出 GPU 读的到底是哪个 (n,c,ih,iw)。
        // 批量验收时可以用 --quiet 关掉，让输出只剩汇总数字。
        if (!g_quiet)
        {
            printf("  row0 col[0..11] gpu =");
            for (int k = 0; k < 12; k++) printf(" %d", (int)gcol[k]);
            printf("\n  row0 期望        =");
            for (int k = 0; k < 12; k++)
            {
                const int j = k % c.kw, t2 = k / c.kw;
                const int i = t2 % c.kh, cc = t2 / c.kh;
                const int ih = 0 * c.sh + i - c.ph;
                const int iw = 0 * c.sw + j - c.pw;
                int eq = 0;
                if (ih >= 0 && ih < h && iw >= 0 && iw < w)
                    eq = xqi[((size_t)0 * c.cin + cc) * h * w + (size_t)ih * w + iw];
                printf(" %d", eq);
            }
            printf("\n");
        }
        printf("im2col 不一致 %lld / %lld\n", colbad, M * Kpad);
        printf("int32 不一致 %lld / %lld", accbad, M * c.cout);
        if (first_acc_idx >= 0)
            printf("，首个 idx=%lld gpu=%d ref=%d", first_acc_idx, first_acc_g, first_acc_r);
        printf("\n");
    }

    double maxdiff = 0.0;
    long long bad = 0, first = -1;
    for (size_t i = 0; i < gout.size(); i++)
    {
        const double d = std::fabs((double)gout[i] - ref[i]);
        if (d > maxdiff) maxdiff = d;
        if (d > 1e-3) { bad++; if (first < 0) first = (long long)i; }
    }
    printf("元素 %zu | max|diff| = %.6g | 超 1e-3 的 %lld 个 (%.2f%%)\n",
           gout.size(), maxdiff, bad, 100.0 * bad / gout.size());
    if (first >= 0)
        printf("首个超差下标 %lld → ref=%.6f gpu=%.6f\n", first, ref[first], gout[first]);
    printf("%s\n", maxdiff < 1e-3 ? "单层一致 ✓" : "单层不一致 ✗");
    return maxdiff < 1e-3 ? 0 : 1;
}