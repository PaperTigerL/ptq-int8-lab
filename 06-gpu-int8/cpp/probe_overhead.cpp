// 端到端「非卷积段」微基准：只有数字能决定要不要优化，不靠猜。
// 测：H2D（pageable vs pinned）/ 残差 add_relu（各 stage 形状）/ maxpool。
// 用法：./probe_overhead [reps=200]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "int8_engine.cuh"   // 只为 add_relu_launch / maxpool_launch 声明

static double now_ms()
{
    return (double)clock() * 1000.0 / CLOCKS_PER_SEC;   // 仅 fallback，不用
}

int main(int argc, char** argv)
{
    const int reps = argc > 1 ? atoi(argv[1]) : 200;
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    auto bench = [&](const char* tag, size_t bytes, auto&& fn) {
        for (int i = 0; i < 20; i++) fn();
        cudaDeviceSynchronize();
        cudaEventRecord(a);
        for (int i = 0; i < reps; i++) fn();
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, a, b);
        const double per = ms / reps;
        printf("%-42s %8.3f ms/次  %7.1f GB/s\n", tag, per,
               bytes > 0 ? (double)bytes * reps / (ms * 1e-3) / 1e9 : 0.0);
        return per;
    };

    // ---- H2D：512 张 3×32×32 float = 6.29 MB，与 main.cpp 完全同尺寸 ----
    const size_t n_float = (size_t)512 * 3 * 32 * 32;
    const size_t bytes = n_float * sizeof(float);
    float* d = nullptr;
    cudaMalloc(&d, bytes);
    std::vector<float> h(n_float, 0.5f);
    float* hp = nullptr;
    cudaMallocHost((void**)&hp, bytes);
    memcpy(hp, h.data(), bytes);
    bench("H2D pageable 6.29MB", bytes, [&] { cudaMemcpy(d, h.data(), bytes, cudaMemcpyHostToDevice); });
    bench("H2D pinned   6.29MB", bytes, [&] { cudaMemcpy(d, hp, bytes, cudaMemcpyHostToDevice); });

    // ---- 残差 add：R18 的 8 次（每 stage 2 次），按真实形状 ----
    const long long elems[4] = {512LL * 64 * 64, 512LL * 128 * 16, 512LL * 256 * 4, 512LL * 512};
    const char* names[4] = {"add_relu 512x64x64", "add_relu 512x128x4x4",
                            "add_relu 512x256x2x2", "add_relu 512x512x1"};
    double add_total = 0.0;
    for (int k = 0; k < 4; k++)
    {
        float* x = nullptr; float* y = nullptr;
        cudaMalloc(&x, elems[k] * sizeof(float));
        cudaMalloc(&y, elems[k] * sizeof(float));
        cudaMemset(x, 0, elems[k] * sizeof(float));
        cudaMemset(y, 0, elems[k] * sizeof(float));
        // add_relu(out, a, b, n, grid)：读 a+b 写 out；这里 out==a 模拟引擎
        int grid = (int)((elems[k] + 255) / 256);
        if (grid > 65535) grid = 65535;
        const double ms = bench(names[k], 3 * elems[k] * sizeof(float), [&] {
            ptqint8::add_relu_launch(x, x, y, (size_t)elems[k], grid);
        });
        add_total += 2.0 * ms;   // 每 stage 2 次
        cudaFree(x); cudaFree(y);
    }
    printf("%-42s %8.3f ms/次\n", "add_relu 8 次合计（每 stage 2 次）", add_total);

    // ---- maxpool：stem 后 8x8 ----
    float* mp_in = nullptr; float* mp_out = nullptr;
    cudaMalloc(&mp_in, (size_t)512 * 64 * 8 * 8 * 4);
    cudaMalloc(&mp_out, (size_t)512 * 64 * 3 * 3 * 4);   // (8+2-3)/2+1 = 4? 实际 4x4
    cudaMemset(mp_in, 0, (size_t)512 * 64 * 64 * 4);
    const double mp = bench("maxpool 512x64x8x8 → 3x3", (size_t)512 * 64 * 64 * 4 * 2, [&] {
        ptqint8::maxpool_launch(mp_in, mp_out, 512, 64, 8, 8, (int)((512 * 64 * 9 + 255) / 256));
    });
    printf("maxpool %.3f ms\n", mp);
    cudaFree(mp_in); cudaFree(mp_out);
    cudaFreeHost(hp);
    cudaFree(d);
    return 0;
}
