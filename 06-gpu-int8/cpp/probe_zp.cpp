// 极简 zp 行为验证：输入全 0.5，看 kernel 输出的码字到底减没减 zp。
#include <cstdint>
//
// 背景：check_layer 报「row0 col[5] gpu=127 期望=-1」。
// 127 恰好等于 255-128，看起来像 zp 没被减。
// 但同一个 .o 被 probe_im2col_quant 验证通过（0/6480）——
// 两者结论矛盾，所以必须用最小 case 把 kernel 的 zp 行为单独钉死。
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "int8_engine.cuh"   // 声明一律 include，
                              // 手写会因参数个数/顺序对不上而链接失败

int main()
{
    const int N = 1, C = 64, H = 32, W = 32, kh = 3, kw = 3, sh = 1, sw = 1, ph = 1, pw = 1;
    const int OH = 32, OW = 32;
    const int K = C * kh * kw;
    const int Kpad = ((K + 7) / 8) * 8;
    const float scale = (2.0f - (-1.3f)) / 127.0f;  // 对称码空间
    const int zp = 0;   // 对称量化 zp 恒为 0

    std::vector<float> in((size_t)N * C * H * W, 0.5f);
    float* d = nullptr;
    cudaMalloc(&d, in.size() * 4);
    cudaMemcpy(d, in.data(), in.size() * 4, cudaMemcpyHostToDevice);

    const long long M = (long long)N * OH * OW;
    int8_t* col = nullptr;
    cudaMalloc(&col, (size_t)M * Kpad);
    ptqint8::quant_im2col_launch(d, col, scale, zp, 255, N, C, H, W, kh, kw, sh, sw, ph, pw,
                        OH, OW, K, Kpad, 0, 65535);
    cudaError_t e = cudaDeviceSynchronize();
    std::vector<signed char> g((size_t)M * Kpad);
    cudaMemcpy(g.data(), col, g.size(), cudaMemcpyDeviceToHost);

    printf("输入恒为 0.5 | scale=%.9g zp=%d | sync=%s\n", scale, zp, cudaGetErrorString(e));
    printf("期望码字 = round(0.5/scale) = %d\n", (int)(0.5f / scale + 0.5f));
    printf("gpu col[0..7] =");
    for (int i = 0; i < 8; i++) printf(" %d", (int)g[i]);
    printf("\n");
    return 0;
}