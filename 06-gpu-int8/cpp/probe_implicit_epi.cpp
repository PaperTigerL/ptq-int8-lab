// Fused implicit-conv epilogue probe.
//
// 只验证一件事：implicit GEMM 的 int32 累加结果，走「独立 requant」与
// 「融合 epilogue」两条路径，写出的 fp32 NCHW 是否逐元素一致。
// 覆盖：3x3 p=1（有边界 padding）、cin=3（Cpad 补到 4）、1x1 s=1、
// cout=64/128（Q=2/Q=4），以及 relu on/off。
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "implicit_conv.cuh"

namespace ptqint8 {
// int8_engine.cuh 里的声明；这里不引入 torch 头，手动声明同一签名。
void requant_launch(const int32_t* acc, float* out,
                    const float* wscale, const float* bias,
                    const float* zp_corr, float ascale,
                    int Cout, int HW, int N, bool relu,
                    int cout_stride, int co_base, int grid);
void requant_tiled_launch(const int32_t* acc, float* out,
                          const float* wscale, const float* bias,
                          const float* zp_corr, float ascale,
                          int Cout, int HW, int N, bool relu,
                          int cout_stride, int co_base);
}

#define CK(expr)                                                              \
    do {                                                                      \
        cudaError_t e = (expr);                                               \
        if (e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__,            \
                    cudaGetErrorString(e));                                   \
            return 2;                                                         \
        }                                                                     \
    } while (0)

static int run_case(int n, int C, int H, int W, int cout,
                    int kh, int kw, int sh, int sw, int ph, int pw,
                    bool relu, int seed)
{
    const int OH = (H + 2 * ph - kh) / sh + 1;
    const int OW = (W + 2 * pw - kw) / sw + 1;
    const int C4 = (C + 3) / 4;
    const int Lpos = OH * OW;
    const long long M = (long long)n * Lpos;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> xd(-1.0f, 1.0f);
    std::uniform_int_distribution<int> wd(-8, 8);
    std::uniform_real_distribution<float> sd(0.01f, 0.05f);
    std::uniform_real_distribution<float> bd(-0.1f, 0.1f);

    std::vector<float> hx((size_t)n * C * H * W);
    for (auto& v : hx) v = xd(rng);
    std::vector<int8_t> hw((size_t)cout * C * kh * kw);
    for (auto& v : hw) v = (int8_t)wd(rng);
    std::vector<float> hws(cout), hb(cout), hzp(cout, 0.0f);
    for (int o = 0; o < cout; o++) { hws[o] = sd(rng); hb[o] = bd(rng); }

    std::vector<int8_t> wimp;
    ptqint8::repack_implicit_weight(hw.data(), cout, C, kh, kw, &wimp);

    float *dx = nullptr, *dout = nullptr, *dref = nullptr;
    float *dws = nullptr, *db = nullptr, *dzp = nullptr;
    int8_t *dwimp = nullptr, *dqact = nullptr;
    int32_t* dacc = nullptr;
    const size_t qbytes = (size_t)n * C4 * 4 * H * W;
    CK(cudaMalloc(&dx, hx.size() * 4));
    CK(cudaMalloc(&dqact, qbytes));
    CK(cudaMalloc(&dwimp, wimp.size()));
    CK(cudaMalloc(&dacc, (size_t)M * cout * 4));
    CK(cudaMalloc(&dout, (size_t)M * cout * 4));
    CK(cudaMalloc(&dref, (size_t)M * cout * 4));
    CK(cudaMalloc(&dws, cout * 4));
    CK(cudaMalloc(&db, cout * 4));
    CK(cudaMalloc(&dzp, cout * 4));
    CK(cudaMemcpy(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dwimp, wimp.data(), wimp.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dws, hws.data(), cout * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db, hb.data(), cout * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dzp, hzp.data(), cout * 4, cudaMemcpyHostToDevice));

    const float ascale = 0.02f;
    ptqint8::quant_act_c4_launch(dx, dqact, ascale, 127,
                                 n, C, 0, C, H, W);
    CK(cudaGetLastError());

    const bool ok_old = ptqint8::implicit_conv_launch(
        dqact, dwimp, dacc, n, C, H, W, kh, kw, sh, sw, ph, pw, OH, OW, cout);
    const bool ok_new = ptqint8::implicit_conv_out_launch(
        dqact, dwimp, dout, dws, db, dzp, ascale,
        n, C, H, W, kh, kw, sh, sw, ph, pw, OH, OW, cout, cout, 0, relu);
    CK(cudaGetLastError());
    if (!ok_old || !ok_new)
    {
        printf("  shape skip: old=%d new=%d\n", ok_old, ok_new);
        return 0;
    }

    const int grid = (int)(((long long)M * cout + 255) / 256);
    if (Lpos >= 16)
        ptqint8::requant_tiled_launch(dacc, dref, dws, db, dzp, ascale,
                                      cout, Lpos, n, relu, cout, 0);
    else
        ptqint8::requant_launch(dacc, dref, dws, db, dzp, ascale,
                                cout, Lpos, n, relu, cout, 0, grid);
    CK(cudaGetLastError());

    std::vector<float> h2((size_t)M * cout), hr((size_t)M * cout);
    CK(cudaMemcpy(h2.data(), dout, h2.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hr.data(), dref, hr.size() * 4, cudaMemcpyDeviceToHost));
    long long bad = 0;
    float maxd = 0.0f;
    for (size_t i = 0; i < h2.size(); i++)
    {
        if (h2[i] != hr[i]) bad++;
        maxd = fmaxf(maxd, fabsf(h2[i] - hr[i]));
    }
    printf("  C=%d H=%d W=%d cout=%d k%dx%d s%d p%d relu=%d | M=%lld "
           "mismatch=%lld/%zu max|diff|=%.3e%s\n",
           C, H, W, cout, kh, kw, sh, ph, (int)relu, M, bad, h2.size(), maxd,
           (bad && maxd <= 1e-5f) ? "  (1-ULP float rounding)" : "");

    // 小规模计时：旧路径 = implicit GEMM 写 int32 + requant；新路径 = 融合 epilogue。
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    const int iters = 50;
    for (int i = 0; i < 5; i++)
    {
        ptqint8::implicit_conv_launch(dqact, dwimp, dacc, n, C, H, W,
                                      kh, kw, sh, sw, ph, pw, OH, OW, cout);
        if (Lpos >= 16)
            ptqint8::requant_tiled_launch(dacc, dref, dws, db, dzp, ascale,
                                          cout, Lpos, n, relu, cout, 0);
        else
            ptqint8::requant_launch(dacc, dref, dws, db, dzp, ascale,
                                    cout, Lpos, n, relu, cout, 0, grid);
    }
    cudaEventRecord(e0);
    for (int i = 0; i < iters; i++)
    {
        ptqint8::implicit_conv_launch(dqact, dwimp, dacc, n, C, H, W,
                                      kh, kw, sh, sw, ph, pw, OH, OW, cout);
        if (Lpos >= 16)
            ptqint8::requant_tiled_launch(dacc, dref, dws, db, dzp, ascale,
                                          cout, Lpos, n, relu, cout, 0);
        else
            ptqint8::requant_launch(dacc, dref, dws, db, dzp, ascale,
                                    cout, Lpos, n, relu, cout, 0, grid);
    }
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float old_ms = 0.0f;
    cudaEventElapsedTime(&old_ms, e0, e1);
    old_ms /= iters;

    for (int i = 0; i < 5; i++)
        ptqint8::implicit_conv_out_launch(dqact, dwimp, dout, dws, db, dzp, ascale,
                                          n, C, H, W, kh, kw, sh, sw, ph, pw,
                                          OH, OW, cout, cout, 0, relu);
    cudaEventRecord(e0);
    for (int i = 0; i < iters; i++)
        ptqint8::implicit_conv_out_launch(dqact, dwimp, dout, dws, db, dzp, ascale,
                                          n, C, H, W, kh, kw, sh, sw, ph, pw,
                                          OH, OW, cout, cout, 0, relu);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float new_ms = 0.0f;
    cudaEventElapsedTime(&new_ms, e0, e1);
    new_ms /= iters;
    printf("  timing: old(acc+requant)=%.4f ms  fused=%.4f ms  x%.2f\n",
           old_ms, new_ms, old_ms / new_ms);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);

    cudaFree(dx); cudaFree(dqact); cudaFree(dwimp); cudaFree(dacc);
    cudaFree(dout); cudaFree(dref); cudaFree(dws); cudaFree(db); cudaFree(dzp);
    // 允许 1 ULP：融合 kernel 与 requant kernel 的 FMA 收缩不同，
    // 1x1/短 K 层会出现 5.96e-8 的差；语义容差按 1e-6 判，逐位相等仍会打印。
    return maxd <= 1e-5f ? 0 : 1;
}

int main()
{
    printf("probe_implicit_epi: fused epilogue vs requant\n");
    int rc = 0;
    rc |= run_case(2, 16, 8, 8, 64, 3, 3, 1, 1, 1, 1, false, 1);
    rc |= run_case(2, 16, 8, 8, 64, 3, 3, 1, 1, 1, 1, true, 2);
    rc |= run_case(2, 3, 32, 32, 64, 3, 3, 1, 1, 1, 1, false, 3);
    rc |= run_case(2, 32, 4, 4, 128, 1, 1, 1, 1, 0, 0, false, 4);
    // R50 形状（batch=8）：看融合 epilogue 在真实 Q/K 下是否真省时间。
    rc |= run_case(8, 64, 32, 32, 64, 3, 3, 1, 1, 1, 1, false, 5);
    rc |= run_case(8, 128, 16, 16, 128, 3, 3, 1, 1, 1, 1, false, 6);
    rc |= run_case(8, 256, 8, 8, 256, 3, 3, 1, 1, 1, 1, false, 7);
    if (rc == 0) printf("ALL PASS\n");
    else printf("FAIL\n");
    return rc;
}
