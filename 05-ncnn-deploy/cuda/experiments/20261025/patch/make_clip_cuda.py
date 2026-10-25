#!/usr/bin/env python3
"""Add a minimal CUDA Clip layer so unfused Clip references can run."""
import io
import os

BASE = "/home/user/ncnn-cuda/src/layer/cuda"

H = r'''#ifndef LAYER_CLIP_CUDA_H
#define LAYER_CLIP_CUDA_H

#include "clip.h"

namespace ncnn {

class Clip_cuda : virtual public Clip
{
public:
    Clip_cuda();

    virtual int forward_inplace(CudaMat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_CLIP_CUDA_H
'''

CPP = r'''#include "clip_cuda.h"

#include <math.h>
#include <stdint.h>

namespace ncnn {

int clip_cuda_forward_inplace(float* d_input, int size, float minv, float maxv);
int clip_cuda_forward_inplace_int8(signed char* d_input, int size, int minv, int maxv);

Clip_cuda::Clip_cuda()
{
    support_cuda = true;
}

int Clip_cuda::forward_inplace(CudaMat& bottom_top_blob, const Option& /*opt*/) const
{
    const int total_size = (int)bottom_top_blob.total();

    if (bottom_top_blob.elemsize == 1u)
    {
        // 与 CPU Clip::forward_inplace_int8 的 float2int8(min/max) 保持一致。
        int min8 = (int)roundf(min);
        int max8 = (int)roundf(max);
        if (min8 > 127) min8 = 127;
        if (min8 < -127) min8 = -127;
        if (max8 > 127) max8 = 127;
        if (max8 < -127) max8 = -127;
        return clip_cuda_forward_inplace_int8(
            static_cast<signed char*>(bottom_top_blob.get_raw_data()), total_size, min8, max8);
    }

    return clip_cuda_forward_inplace(
        static_cast<float*>(bottom_top_blob.get_raw_data()), total_size, min, max);
}

} // namespace ncnn
'''

CU = r'''#include "clip_cuda.h"

namespace ncnn {

__global__ void clip_forward_inplace_kernel(float* __restrict__ data, int size, float minv, float maxv)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    float v = data[i];
    if (v < minv) v = minv;
    if (v > maxv) v = maxv;
    data[i] = v;
}

__global__ void clip_forward_inplace_int8_kernel(signed char* __restrict__ data, int size, int minv, int maxv)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    int v = (int)data[i];
    if (v < minv) v = minv;
    if (v > maxv) v = maxv;
    data[i] = (signed char)v;
}

void clip_cuda_forward_inplace(float* d_input, int size, float minv, float maxv)
{
    clip_forward_inplace_kernel<<<(size + 255) / 256, 256>>>(d_input, size, minv, maxv);
}

void clip_cuda_forward_inplace_int8(signed char* d_input, int size, int minv, int maxv)
{
    clip_forward_inplace_int8_kernel<<<(size + 255) / 256, 256>>>(d_input, size, minv, maxv);
}

} // namespace ncnn
'''

for name, content in (("clip_cuda.h", H), ("clip_cuda.cpp", CPP), ("clip_cuda.cu", CU)):
    with io.open(os.path.join(BASE, name), "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
print("wrote clip_cuda.{h,cpp,cu}")
