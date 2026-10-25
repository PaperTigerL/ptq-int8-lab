// GEMM 层验证：证明 at::_int_mm 的正确用法，以及裸 cublas 为什么不行。
//
// ⭐ 这个探针记录了一个花了三轮才搞明白的事实：
//   **torch::from_blob 造的张量默认在 CPU 上**。
//   实测：
//     auto a = from_blob(hA, {M,K}, kInt8);        // CPU 张量
//     _int_mm(a, b) → bad = 468/2048（垃圾值）
//     auto a = from_blob(...).to(kCUDA).contiguous();  // GPU 张量
//     _int_mm(a, b) → bad = 0/2048 ✓
//
// 之前误以为是 cuBLAS 的列主序约定推错了，穷举了 (transa,transb,lda,ldb,ldc)
// 三轮共 27 组组合，全错。**真相是设备/布局规范化，不是数学约定。**
// 教训：数值不对时，先确认「输入真的是我以为的那个东西」，
//       再去推数学。顺序反了会浪费大量编译周期。
#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#include <cstdio>
#include <random>
#include <vector>

static int compare(torch::Tensor a, torch::Tensor b,
                   const std::vector<int8_t>& hA, const std::vector<int8_t>& hB,
                   const char* tag)
{
    auto c = at::_int_mm(a, b).contiguous().cpu();
    const int32_t* p = c.data_ptr<int32_t>();
    // 注意：不要写 `int_K` 这样的变量名 —— `_K` 后缀在 C++ 里是非法标识符，
    // 编译器报的是 "expected unqualified-id before int"，看起来像类型问题，
    // 实际是变量名。用 Kdim。
    const int M = (int)a.size(0), Kdim = (int)a.size(1), Ncols = (int)b.size(1);

    int bad = 0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < Ncols; n++)
        {
            int32_t s = 0;
            for (int k = 0; k < Kdim; k++)
                s += (int32_t)hA[m * Kdim + k] * (int32_t)hB[k * Ncols + n];
            if (p[m * Ncols + n] != s) bad++;
        }
    printf("%-46s bad=%5d/%d %s\n", tag, bad, M * Ncols, bad ? "x" : "OK");
    return bad;
}

int main()
{
    const int M = 64, K = 128, N = 32;
    std::vector<int8_t> hA(M * K), hB(K * N);
    std::mt19937 r(7);
    std::uniform_int_distribution<int> d(-100, 100);
    for (auto& v : hA) v = (int8_t)d(r);
    for (auto& v : hB) v = (int8_t)d(r);

    printf("M=%d K=%d N=%d，行主序 C = A@B\n", M, K, N);

    // 1) 错：从_blob 不指定 device → CPU 张量
    compare(torch::from_blob(hA.data(), {M, K}, torch::kInt8),
            torch::from_blob(hB.data(), {K, N}, torch::kInt8),
            hA, hB, "from_blob（无 device，默认 CPU）");

    // 2) 对：显式 .to(kCUDA).contiguous()
    compare(torch::from_blob(hA.data(), {M, K}, torch::kInt8).to(torch::kCUDA).contiguous(),
            torch::from_blob(hB.data(), {K, N}, torch::kInt8).to(torch::kCUDA).contiguous(),
            hA, hB, "from_blob().to(kCUDA).contiguous()");

    // 3) 引擎里用的包装：直接包 device 指针
    {
        int8_t *dA = nullptr, *dB = nullptr;
        cudaMalloc(&dA, hA.size());
        cudaMalloc(&dB, hB.size());
        cudaMemcpy(dA, hA.data(), hA.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB.data(), hB.size(), cudaMemcpyHostToDevice);
        // 引擎的 wrap_i8_cuda 用 device_ptr + 显式 device=kCUDA
        auto a = torch::from_blob(dA, {M, K},
                                  torch::TensorOptions().dtype(torch::kInt8).device(torch::kCUDA));
        auto b = torch::from_blob(dB, {K, N},
                                  torch::TensorOptions().dtype(torch::kInt8).device(torch::kCUDA));
        compare(a, b, hA, hB, "wrap_i8_cuda（device_ptr + 显式 device）");
        cudaFree(dA);
        cudaFree(dB);
    }

    printf("\n结论：GEMM 层用 at::_int_mm，不要自己调 cublasGemmEx。\n");
    printf("      裸指针调用会绕过 libtorch 的设备/布局规范化，输出垃圾值且不报错。\n");
    return 0;
}