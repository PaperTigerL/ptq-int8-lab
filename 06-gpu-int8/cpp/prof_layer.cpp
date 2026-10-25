// 分段计时：把全网拆成「量化+im2col / GEMM / requant+ReLU / 残差 / fc」
// 四段，测出各自的占比。
//
// 为什么必须做：优化到 161× 之后，剩下的 3 倍不知道在哪。
// 直觉会猜「GEMM」，但 int8 tensor core 只有 3.5 TOPS，
// 而 im2col 矩阵搬运的数据量是 GEMM 输出的 kh*kw 倍 —— 谁才是瓶颈
// 必须用数据说话，不能猜。
#include <chrono>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "int8_engine.cuh"

#define TIME(label, n_iter, body)                                              \
    do {                                                                       \
        body;                                                                  \
        CUDA_CHECK(cudaDeviceSynchronize());                                   \
        const auto _t0 = std::chrono::steady_clock::now();                     \
        for (int _i = 0; _i < (n_iter); _i++) { body; }                       \
        CUDA_CHECK(cudaDeviceSynchronize());                                   \
        const auto _t1 = std::chrono::steady_clock::now();                     \
        const double _ms =                                                     \
            std::chrono::duration<double, std::milli>(_t1 - _t0).count() /     \
            (n_iter);                                                          \
        printf("  %-24s %8.3f ms/batch  %5.1f%%\n", label, _ms,                \
               100.0 * _ms / total_ms);                                        \
        acc_ms += _ms;                                                         \
    } while (0)

int main(int argc, char** argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s <model.bin> <calib.bin> [batch]\n", argv[0]); return 2; }
    const int n = (argc > 3) ? atoi(argv[3]) : 2000;

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

    Int8Engine engine;
    if (!engine.build(model, 8, 8)) return 1;
    {
        std::string cerr;
        if (!engine.set_calib(ranges, &cerr))
        {
            fprintf(stderr, "[FAIL] set_calib: %s\n", cerr.c_str());
            return 1;
        }
    }
    engine.alloc_scratch(n);

    const int c = 64, h = 32, w = 32, kh = 3, kw = 3, K = c * kh * kw;
    const int OH = 32, OW = 32;
    const long long M = (long long)n * OH * OW;

    const float scale = 0.01f, zp = 0;
    const int qmax = 127;

    float* d_in = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, (size_t)n * c * h * w * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_in, 0, (size_t)n * c * h * w * sizeof(float)));

    std::vector<int8_t> wdev((size_t)K * 64, 3);
    int8_t* w_gpu = nullptr;
    CUDA_CHECK(cudaMalloc(&w_gpu, wdev.size()));
    CUDA_CHECK(cudaMemcpy(w_gpu, wdev.data(), wdev.size(), cudaMemcpyHostToDevice));

    ptqint8::CublasCtx ctx;
    ctx.init();

    int8_t* col = nullptr;
    CUDA_CHECK(cudaMalloc(&col, (size_t)M * K));
    int32_t* acc = nullptr;
    CUDA_CHECK(cudaMalloc(&acc, (size_t)M * 64 * sizeof(int32_t)));
    float* o = nullptr;
    CUDA_CHECK(cudaMalloc(&o, (size_t)M * 64 * sizeof(float)));
    float* sc = nullptr;
    CUDA_CHECK(cudaMalloc(&sc, 64 * sizeof(float)));
    CUDA_CHECK(cudaMemset(sc, 0, 64 * sizeof(float)));
    float* bi = nullptr;
    CUDA_CHECK(cudaMalloc(&bi, 64 * sizeof(float)));
    CUDA_CHECK(cudaMemset(bi, 0, 64 * sizeof(float)));
    float* zp_c = nullptr;
    CUDA_CHECK(cudaMalloc(&zp_c, 64 * sizeof(float)));
    CUDA_CHECK(cudaMemset(zp_c, 0, 64 * sizeof(float)));

    const int iters = 20;
    double acc_ms = 0.0;

    // 先整体跑一次做基线（时间记在 total 里）
    {
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 3; i++)
        {
            ptqint8::quant_im2col_launch(d_in, col, scale, zp, qmax, n, c, h, w,
                                         kh, kw, 1, 1, 1, 1, OH, OW, K, K, 0, 65535);
            auto t = ptqint8::int8_gemm(col, M, K, w_gpu, 64);
            ptqint8::requant_launch(t.data_ptr<int32_t>(), o, sc, bi, zp_c, scale,
                                    64, OH * OW, n, false, 64, 0, 65535);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        acc_ms = std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - t0).count() / 3;
        printf("单层 (C64 32x32 k3, batch=%d) 合计 %.3f ms\n", n, acc_ms);
    }
    const double total_ms = acc_ms;
    acc_ms = 0.0;

    printf("分段占比：\n");
    TIME("量化+im2col", iters,
         ptqint8::quant_im2col_launch(d_in, col, scale, zp, qmax, n, c, h, w,
                                      kh, kw, 1, 1, 1, 1, OH, OW, K, K, 0, 65535));
    {
        torch::Tensor t = ptqint8::int8_gemm(col, M, K, w_gpu, 64);
        TIME("int8 GEMM", iters, {
            auto tt = ptqint8::int8_gemm(col, M, K, w_gpu, 64);
            ptqint8::requant_launch(tt.data_ptr<int32_t>(), o, sc, bi, zp_c, scale,
                                    64, OH * OW, n, false, 64, 0, 65535);
        });
    }
    {
        auto tt = ptqint8::int8_gemm(col, M, K, w_gpu, 64);
        const int32_t* ap = tt.data_ptr<int32_t>();
        TIME("requant", iters,
             ptqint8::requant_launch(ap, o, sc, bi, zp_c, scale, 64, OH * OW, n, false, 64, 0, 65535));
        TIME("relu", iters, ptqint8::relu_launch(o, (size_t)M * 64, 65535, 1));
        TIME("add", iters, ptqint8::add_launch(o, o, o, (size_t)M * 64, 65535));
        TIME("maxpool3x3s2p1", iters, ptqint8::maxpool_launch(d_in, o, n, c, h, w, 65535));
    }

    printf("\n换算：全网 20 层若都与本层同代价 → %.1f ms/batch = %.4f ms/img\n",
           total_ms * 20, total_ms * 20 / n);
    return 0;
}