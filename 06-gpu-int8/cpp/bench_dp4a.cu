// DP4A 峰值燃烧器：用来实测 GTX 1650 Ti 在**持续 int8 计算负载**下的
// 真实 boost 频率（nvidia-smi 的空闲频率 1350 MHz 不是 boost）。
//
// 每线程 8 条独立 __dp4a，避免依赖链把 IPC 压住；1024 块 × 256 线程
// = 26 万线程，覆盖 16 个 SM 的 1024 个 CUDA core。
//
// 用法：./bench_dp4a <秒>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void burn(int iters, int seed, int* sink)
{
    int a = seed + threadIdx.x, b = seed * 3 + blockIdx.x;
    int acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    int acc4 = 0, acc5 = 0, acc6 = 0, acc7 = 0;
    for (int i = 0; i < iters; ++i)
    {
        acc0 = __dp4a(a, b, acc0);
        acc1 = __dp4a(b, a, acc1);
        acc2 = __dp4a(a, b, acc2);
        acc3 = __dp4a(b, a, acc3);
        acc4 = __dp4a(a, b, acc4);
        acc5 = __dp4a(b, a, acc5);
        acc6 = __dp4a(a, b, acc6);
        acc7 = __dp4a(b, a, acc7);
    }
    int s = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7;
    if (s == 123456789) *sink = s;
}

int main(int argc, char** argv)
{
    const int secs = argc > 1 ? atoi(argv[1]) : 30;
    int* d = nullptr;
    cudaMalloc(&d, 4);
    const int blocks = 1024, threads = 256;
    // 预热一次，拿到单趟耗时并让时钟爬升
    burn<<<blocks, threads>>>(1000000, 7, d);
    cudaDeviceSynchronize();
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0);
    float elapsed = 0.0f;
    long long passes = 0;
    while (elapsed < secs * 1000.0f)
    {
        burn<<<blocks, threads>>>(1000000, 7, d);
        cudaDeviceSynchronize();
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        cudaEventElapsedTime(&elapsed, e0, e1);
        passes++;
    }
    const double dp4a = (double)passes * blocks * threads * 1000000.0 * 8.0;
    const double ops = dp4a * 8.0; // 每条 dp4a = 4 MAC = 8 int8 ops
    printf("burn %.2f s  passes=%lld  dp4a=%.3f G  int8_ops=%.3f G  => %.2f TOPS\n",
           elapsed / 1000.0, passes, dp4a / 1e9, ops / 1e9,
           ops / (elapsed / 1000.0) / 1e12);
    return 0;
}
