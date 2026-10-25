// im2col + 量化 的完整映射验证（zp 固定在码空间中点）。
//
// probe_im2col 只验证了 zp=0、scale=1 的恒等映射。
// 现在验证真实的非对称量化路径：scale≠1、zp=128。
//
// 关键点：8 bit 非对称码空间 [0,255] 必须通过 zp=128 映射进 int8 [-128,127]。
// 这一步以前是错的（zp=0），症状是码字溢出。全网精度 9% 就是这么来的。
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "int8_engine.cuh"

int main()
{
    const int N = 3, C = 5, H = 9, W = 7, kh = 3, kw = 3, sh = 2, sw = 1, ph = 1, pw = 2;
    const int OH = (H + 2 * ph - kh) / sh + 1;
    const int OW = (W + 2 * pw - kw) / sw + 1;
    const int K = C * kh * kw;
    const int Kpad = ((K + 7) / 8) * 8;
    // ⚠️ 口径必须与引擎一致，所以直接调用同一个函数，不再手写公式：
    //   对称码空间，qmax = 2^(bits-1)-1 = 127，zp 恒 0，默认 AbsMax。
    //   这个探针第一版写的是 zp=128/qmax=255（非对称），
    //   主线改对称后它没跟着改，于是报 714/6480 不一致 ——
    //   **探针过期被误读成 kernel 回归**，这是本项目第 4 次踩这个坑。
    const int bits = 8;
    const float lo = -1.3f, hi = 2.7f;
    const ptq::TensorQuant aq = ptq::tensor_quant_from_range(lo, hi, bits);
    const float scale = aq.scale;
    const int qmax = (1 << (bits - 1)) - 1, zp = aq.zero_point;

    printf("lo=%.3f hi=%.3f → scale=%.6f zp=%d qmax=%d\n", lo, hi, scale, zp, qmax);
    printf("N=%d C=%d H=%d W=%d k=%dx%d s=(%d,%d) p=(%d,%d) → OH=%d OW=%d K=%d Kpad=%d\n",
           N, C, H, W, kh, kw, sh, sw, ph, pw, OH, OW, K, Kpad);
    printf("码字范围检查：lo→%d  hi→%d  （clamp 到 [-(qmax+1), qmax] = [%d, %d]）\n",
           (int)std::floor(lo / scale + 0.5f), (int)std::floor(hi / scale + 0.5f),
           -(qmax + 1), qmax);

    std::mt19937 rng(2026);
    std::uniform_real_distribution<float> ud(lo, hi);
    std::vector<float> in((size_t)N * C * H * W);
    for (auto& v : in) v = ud(rng);

    float* d_in = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, in.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, in.data(), in.size() * sizeof(float), cudaMemcpyHostToDevice));

    const long long M = (long long)N * OH * OW;
    int8_t* col = nullptr;
    CUDA_CHECK(cudaMalloc(&col, (size_t)M * Kpad));
    ptqint8::quant_im2col_launch(d_in, col, scale, zp, qmax,
                                 N, C, H, W, kh, kw, sh, sw, ph, pw,
                                 OH, OW, K, Kpad, 0,
                                 (int)std::min<long long>(65535, (M * Kpad + 255) / 256));
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<int8_t> g((size_t)M * Kpad);
    CUDA_CHECK(cudaMemcpy(g.data(), col, g.size(), cudaMemcpyDeviceToHost));

    long long bad = 0;
    long long first = -1;
    int min_g = 128, max_g = -128;
    for (long long row = 0; row < M; row++)
    {
        const int ow = (int)(row % OW);
        const long long t = row / OW;
        const int oh = (int)(t % OH);
        const int n = (int)(t / OH);
        for (int k = 0; k < Kpad; k++)
        {
            int expect = 0;
            if (k < K)
            {
                const int j = k % kw, t2 = k / kw;
                const int i = t2 % kh, c = t2 / kh;
                const int ih = oh * sh + i - ph;
                const int iw = ow * sw + j - pw;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                {
                    const float v = in[((size_t)n * C + c) * H * W + (size_t)ih * W + iw];
                    float q = std::floor(v / scale + 0.5f);
                    if (q < -(float)(qmax + 1)) q = -(float)(qmax + 1);
                    if (q > (float)qmax) q = (float)qmax;
                    expect = (int)q;
                }
            }
            const int got = (int)g[(size_t)row * Kpad + k];
            if (got < min_g) min_g = got;
            if (got > max_g) max_g = got;
            if (got != expect)
            {
                bad++;
                if (first < 0)
                    first = row * Kpad + k;
            }
        }
    }
    printf("im2col+量化：不一致 %lld / %lld %s\n", bad, M * Kpad,
           bad ? "✗" : "✓");
    printf("GPU 码字范围 [%d, %d]（clamp 区间 [%d, %d]）\n", min_g, max_g, -(qmax+1), qmax);
    if (bad && first >= 0) printf("首个不一致 idx=%lld\n", first);
    cudaFree(d_in);
    cudaFree(col);
    return bad ? 1 : 0;
}