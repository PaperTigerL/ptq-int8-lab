// im2col 映射的最小验证：把 GPU kernel 的 (row,k) 分解规则
// 与 CPU 侧的索引公式逐点比对，**不涉及任何量化**。
//
// 为什么单独测：前面整体比对给出「im2col 不一致 84%」，
// 但那可能是量化、GEMM、requant 三者的叠加。必须先把 im2col
// 这一段的索引映射单独钉死，否则后面全是无效劳动。
//
// 做法：用**可直接预测的输入** x[n][c][h][w] = n*1000000 + c*10000 + h*100 + w，
// 这样任何一个输出码字都能反推出它读的是哪个 (n,c,h,w)，
// 出错时能立刻看出「读了哪里的值」。
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "int8_engine.cuh"

int main(int argc, char** argv)
{
    const int N = 2, C = 4, H = 8, W = 8, kh = 3, kw = 3, sh = 2, sw = 2, ph = 1, pw = 1;
    const int OH = (H + 2 * ph - kh) / sh + 1;
    const int OW = (W + 2 * pw - kw) / sw + 1;
    const int K = C * kh * kw;
    const int Kpad = ((K + 7) / 8) * 8;

    printf("N=%d C=%d H=%d W=%d k=%dx%d s=%d p=%d → OH=%d OW=%d K=%d Kpad=%d\n",
           N, C, H, W, kh, kw, sh, ph, OH, OW, K, Kpad);

    // scale=1, zp=0 → 量化是恒等映射（只做 clamp 到 [0,255]）
    //
    // ⚠️ 编码值必须落在 **[0, 127]**（int8 正半边）才能反推来源。
    //   踩过两次：
    //     v1 用 n*1e6+c*1e4+... 最大 2e6 → 全被 clamp 成 127（假阳性）
    //     v2 用 n*64+c*16+h*4+w 最大 191 → 128 溢出成 -128（又是假阳性）
    //   现在 n*32+c*16+h*4+w，最大 1*32+3*16+7*4+7 = 95 < 127 ✓
    std::vector<float> in((size_t)N * C * H * W);
    auto code = [](int n, int c, int h, int w) { return n * 32 + c * 16 + h * 4 + w; };
    for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
            for (int h = 0; h < H; h++)
                for (int w = 0; w < W; w++)
                    in[((size_t)n * C + c) * H * W + h * W + w] =
                        (float)code(n, c, h, w);

    float* d_in = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, in.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, in.data(), in.size() * sizeof(float), cudaMemcpyHostToDevice));

    const long long M = (long long)N * OH * OW;
    int8_t* col = nullptr;
    CUDA_CHECK(cudaMalloc(&col, (size_t)M * Kpad));
    ptqint8::quant_im2col_launch(d_in, col, 1.0f, 0, 255,
                                 N, C, H, W, kh, kw, sh, sw, ph, pw,
                                 OH, OW, K, Kpad, 0,
                                 (int)std::min<long long>(65535, (M * Kpad + 255) / 256));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int8_t> g((size_t)M * Kpad);
    CUDA_CHECK(cudaMemcpy(g.data(), col, g.size(), cudaMemcpyDeviceToHost));

    long long bad = 0;
    long long first = -1;
    for (long long row = 0; row < M; row++)
    {
        // row -> (n, oh, ow)：与 kernel 内部一致
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
                    expect = code(n, c, ih, iw);
            }
            const int got = (int)g[(size_t)row * Kpad + k];
            if (got != expect)
            {
                bad++;
                if (first < 0)
                {
                    first = row * Kpad + k;
                    printf("首个不一致: row=%lld (n=%d oh=%d ow=%d) k=%d → gpu=%d 期望=%d (n,c,ih,iw)\n",
                           row, n, oh, ow, k, got, expect);
                }
            }
        }
    }
    printf("im2col 映射：不一致 %lld / %lld %s\n", bad, M * Kpad,
           bad ? "✗" : "✓（索引映射正确）");
    cudaFree(d_in);
    cudaFree(col);
    return bad ? 1 : 0;
}