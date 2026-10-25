#!/usr/bin/env python3
"""Fused int8 InnerProduct GEMV: one warp per output, dp4a, fused dequant.

The current int8 path zeroes a 16MB scratchpad every forward, launches a
partial-product kernel with a 64x8 shared-memory reduction, then a second
kernel that sums the partials and dequantizes.  For the model's single FC
(512 -> 10) the used scratch is only 80 ints, so the 16MB memset dominates.

Fast path (input 1-D, contiguous, in_size % 4 == 0, weight flat
[num_output][in_size]):
  - one warp per output, each lane dp4a-sums 4 int8 values per iteration,
    warp-shuffle reduction, lane 0 dequantizes + bias + activation;
  - one kernel, no scratchpad, no memset;
  - int32 addition is associative and the total < 2^31, so logits are
    bit-identical to the two-kernel path;
  - NCNN_CUDA_IP_GEMV=0 disables the fast path (old path remains).
"""
import io

PATH = "/home/user/ncnn-cuda/src/layer/cuda/innerproduct_cuda.cu"

INC_OLD = '#include "innerproduct_cuda.h"'
INC_NEW = '#include <cstdlib>\n\n#include "innerproduct_cuda.h"'

KERNEL = r'''
__device__ __forceinline__ int ip_dp4a_i8(int a, int b, int c)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    int r = c;
    signed char aa[4], bb[4];
    *reinterpret_cast<int*>(aa) = a;
    *reinterpret_cast<int*>(bb) = b;
    for (int i = 0; i < 4; i++) r += (int)aa[i] * (int)bb[i];
    return r;
#endif
}

// 输入 1-D int8、权重 [num_output][in_size] int8；一个 warp 负责一个输出。
__global__ void gpu_innerproduct_cuda_forward_int8_gemv(
    const signed char* __restrict__ input, int in_size,
    const signed char* __restrict__ weight,
    const float* __restrict__ weight_data_int8_scales,
    const float* __restrict__ bias_data,
    const float* __restrict__ bottom_blob_int8_scale,
    const float* __restrict__ activation_params,
    int num_output, bool bias_term, int activation_type,
    float* __restrict__ output)
{
    const int lane = threadIdx.x & 31;
    const int co = blockIdx.x;
    if (co >= num_output) return;

    const signed char* w = weight + (size_t)co * in_size;
    int acc = 0;
    for (int i = lane * 4; i + 3 < in_size; i += 128)
    {
        const int a = *reinterpret_cast<const int*>(input + i);
        const int b = *reinterpret_cast<const int*>(w + i);
        acc = ip_dp4a_i8(a, b, acc);
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);

    if (lane != 0) return;

    const float ws = weight_data_int8_scales[co];
    const float scale_in = (ws == 0.0f) ? 0.0f : 1.0f / (*bottom_blob_int8_scale * ws);
    float y = (float)acc * scale_in;
    if (bias_term) y += bias_data[co];

    if (activation_type == 1)
    {
        if (y < 0.0f) y = 0.0f;
    }
    else if (activation_type == 3 && activation_params)
    {
        y = fminf(fmaxf(y, activation_params[0]), activation_params[1]);
    }
    output[co] = y;
}

'''

FASTPATH = """
    // 单 FC、1-D 输入时走融合 GEMV：省掉 16MB scratchpad memset 与第二个 kernel。
    const char* ip_gemv_env = std::getenv("NCNN_CUDA_IP_GEMV");
    const bool ip_gemv_off = ip_gemv_env && ip_gemv_env[0] == '0' && ip_gemv_env[1] == '\\0';
    const bool ip_gemv_ok = !ip_gemv_off
        && bottom_blob.elemsize == 1u
        && bottom_blob.h == 1 && bottom_blob.c == 1
        && bottom_blob.w > 0 && (bottom_blob.w & 3) == 0
        && info.gpu_weight_data && info.gpu_weight_data->w == bottom_blob.w * info.num_output
        && (info.activation_type == 0 || info.activation_type == 1 || info.activation_type == 3);
    if (ip_gemv_ok)
    {
        gpu_innerproduct_cuda_forward_int8_gemv<<<info.num_output, 32>>>(
            static_cast<const signed char*>(bottom_blob.get_craw_data()),
            bottom_blob.w,
            static_cast<const signed char*>(info.gpu_weight_data->get_craw_data()),
            static_cast<const float*>(info.gpu_weight_data_int8_scales->get_craw_data()),
            info.bias_term ? static_cast<const float*>(info.gpu_bias_data->get_craw_data()) : nullptr,
            static_cast<const float*>(info.gpu_bottom_blob_int8_scale),
            static_cast<const float*>(info.gpu_activation_params->get_craw_data()),
            info.num_output, info.bias_term, info.activation_type,
            static_cast<float*>(top_blob.get_raw_data()));
        return 0;
    }

    checkCudaErrors(cudaMemset(gpu_scratchpad_memory, 0, gpu_scratchpad_memory_size));"""

MARK = """    checkCudaErrors(cudaMemset(gpu_scratchpad_memory, 0, gpu_scratchpad_memory_size));"""


def main():
    with io.open(PATH, "r", encoding="utf-8") as f:
        text = f.read()
    if MARK not in text:
        raise SystemExit("memset marker missing")
    # insert the fast-path branch only in the int8 launcher (first occurrence
    # after the int8 launcher signature)
    sig = "int innerproduct_cuda_forward_int8(const CudaMat& bottom_blob"
    pos = text.index(sig)
    mpos = text.index(MARK, pos)
    text = text[:mpos] + FASTPATH + text[mpos + len(MARK):]
    # add the GEMV kernel before the int8 launcher
    pos = text.index(sig)
    text = text[:pos] + KERNEL + text[pos:]
    if INC_OLD in text:
        text = text.replace(INC_OLD, INC_NEW, 1)
    with io.open(PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print("fused int8 GEMV added")


if __name__ == "__main__":
    main()
