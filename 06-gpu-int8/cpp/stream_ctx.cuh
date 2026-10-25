// 每个线程一个当前 CUDA stream。
//
// 为什么要 thread_local 而不是给每个 launcher 加一个 stream 参数：
//   所有 kernel launcher 都在 int8_kernels.cu / implicit_conv.cu 里，
//   加参数要改 ~30 个调用点和 ~30 个签名，而多流 harness 是「一个线程
//   一个引擎 + 一个 stream」，线程局部变量正好是它的自然映射。
//   inline thread_local（C++17）在两个 .cu 之间共享同一个实例。
//
// 默认 0 = legacy default stream，单流路径完全不变。
#pragma once

#include <cuda_runtime.h>

namespace ptqint8 {

inline thread_local cudaStream_t g_stream = nullptr;

inline void set_stream(cudaStream_t s) { g_stream = s; }
inline cudaStream_t get_stream() { return g_stream; }

} // namespace ptqint8
