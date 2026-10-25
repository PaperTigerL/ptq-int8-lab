// int8 GEMM 封装（libtorch 版）。
//
// ⭐ 为什么不用裸 cublasGemmEx：
//   试过三轮（probe_gemm/2/3），穷举了 (transa,transb,lda,ldb,ldc,m,n,k)
//   的全部组合，结果全错或 illegal value。原因不是组合不存在，
//   而是 **裸指针绕过了 libtorch 的设备/布局规范化**：
//   `from_blob` 造出的张量默认在 **CPU** 上，必须 `.to(kCUDA).contiguous()`
//   才是合规输入（实测：不加这两步时 bad=468/2048，加了之后 bad=0）。
//   继续自己推导 cuBLAS 的列主序约定，是在重新实现 libtorch 已经做对的事。
//
// 结论：**GEMM 交给 at::_int_mm，量化 kernel 保持自有实现。**
// 分工清晰：
//   · 量化语义 + im2col 融合  → 我们的 CUDA kernel（真正要改的部分）
//   · int8 GEMM              → libtorch / cuBLAS（tensor core 的入口）
//   这也正好是「改语义、不重造轮子」的分界。
#pragma once

#include <ATen/ATen.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#include <vector>

namespace ptqint8 {

// fp32 sgemm 用的 handle。**只有最后一层 fc 走它** —— fp32 的 512×10
// 矩阵乘法用 cuBLAS 完全够，不要为了"统一"把它也换掉。
struct CublasCtx
{
    cublasHandle_t handle = nullptr;
    bool init() { return cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS; }
    void destroy() { if (handle) { cublasDestroy(handle); handle = nullptr; } }
};

// 零拷贝地包一个 device 指针为 CUDA int8 Tensor。
// ⚠️ 必须传 device_ptr（不是 host 指针）并显式 device=kCUDA：
//    省略 device 会得到 CPU 张量，_int_mm 会静默走 CPU 或报奇怪的错。
inline torch::Tensor wrap_i8_cuda(void* dev_ptr, long rows, long cols)
{
    return torch::from_blob(dev_ptr, {rows, cols},
                            torch::TensorOptions().dtype(torch::kInt8).device(torch::kCUDA));
}

// 行主序 C(M,N) = A(M,K) · B(K,N)，全 int8 → int32。
// A、B 都是我们自己 kernel 产出的 device 内存，形状必须显式匹配其布局。
inline torch::Tensor int8_gemm(void* a_dev, long M, long K,
                               void* b_dev, long N)
{
    auto a = wrap_i8_cuda(a_dev, M, K);
    auto b = wrap_i8_cuda(b_dev, K, N);
    return at::_int_mm(a, b);   // (M,N) int32
}

} // namespace ptqint8