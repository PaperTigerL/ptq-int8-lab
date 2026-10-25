#!/usr/bin/env python3
"""Fuse valid-padding into the max-pooling kernel.

Old path per image: copy_make_border (pad kernel) + fill(-FLT_MAX) + max kernel
with a device-side offset table.
New path (pad_mode==1, max pooling): one kernel reads the unbordered blob and
checks window bounds; out-of-range taps are the max-identity value.  For max
pooling the result is bit-identical (padding is -FLT_MAX / -128 and never
wins).  Other pooling types / pad modes keep the old path.

NCNN_CUDA_POOL_FUSE_PAD=0 disables the fused path (default on for A/B).
"""
import io

CPP = "/home/user/ncnn-cuda/src/layer/cuda/pooling_cuda.cpp"
CU = "/home/user/ncnn-cuda/src/layer/cuda/pooling_cuda.cu"

CPP_DECL_OLD = """int pooling_cuda_forward_global(const CudaMat& bottom_blob, CudaMat& top_blob, const Pooling_cuda::Pooling_info& info);
int pooling_cuda_forward(const CudaMat& bottom_blob, CudaMat& top_blob, const Pooling_cuda::Pooling_info& info);"""
CPP_DECL_NEW = """int pooling_cuda_forward_global(const CudaMat& bottom_blob, CudaMat& top_blob, const Pooling_cuda::Pooling_info& info);
int pooling_cuda_forward(const CudaMat& bottom_blob, CudaMat& top_blob, const Pooling_cuda::Pooling_info& info);
int pooling_cuda_forward_max_pad(const CudaMat& bottom_blob, CudaMat& top_blob,
                                 int kernel_w, int kernel_h, int stride_w, int stride_h,
                                 int pad_left, int pad_top);"""

INC_OLD = """#include <chrono>
#include <float.h>"""
INC_NEW = """#include <chrono>
#include <cstdlib>
#include <float.h>"""

CPP_BRANCH_OLD = """    CudaMat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;"""
CPP_BRANCH_NEW = """    // valid-padding max pooling：把边界检查融进 max kernel，省掉 pad kernel、
    // bordered 中间缓冲和 fill。padding 值是 max 的单位元，结果逐位不变。
    const char* pool_fuse_env = getenv("NCNN_CUDA_POOL_FUSE_PAD");
    const bool pool_fuse_off = pool_fuse_env && pool_fuse_env[0] == '0' && pool_fuse_env[1] == '\\0';
    if (!pool_fuse_off && pooling_type == PoolMethod_MAX && !global_pooling
        && pad_mode == 1 && (pad_left + pad_right + pad_top + pad_bottom) > 0
        && kernel_w <= 8 && kernel_h <= 8)
    {
        const int outw = (w + pad_left + pad_right - kernel_w) / stride_w + 1;
        const int outh = (h + pad_top + pad_bottom - kernel_h) / stride_h + 1;
        top_blob.create(outw, outh, channels, elemsize, opt.blob_cuda_allocator);
        if (top_blob.empty()) return -100;

        pooling_cuda_forward_max_pad(bottom_blob, top_blob, kernel_w, kernel_h,
                                     stride_w, stride_h, pad_left, pad_top);
        return 0;
    }

    CudaMat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;"""

CU_KERNEL = r'''
// valid-padding max pooling: 读未 padding 的 blob，窗口越界处视为 max 单位元。
template <typename T>
__global__ void gpu_pooling_cuda_forward_max_pad_kernel(const T* __restrict__ input,
                                                        int in_w, int in_h, int in_cstep,
                                                        T* __restrict__ output,
                                                        int out_w, int out_h, int out_cstep,
                                                        int kernel_w, int kernel_h,
                                                        int stride_w, int stride_h,
                                                        int pad_left, int pad_top,
                                                        T pad_value, int total)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int ox = idx % out_w;
    const int t = idx / out_w;
    const int oy = t % out_h;
    const int c = t / out_h;

    const T* base = input + (size_t)c * in_cstep;
    const int sx0 = ox * stride_w - pad_left;
    const int sy0 = oy * stride_h - pad_top;

    T m = pad_value;
    for (int ky = 0; ky < kernel_h; ky++)
    {
        const int sy = sy0 + ky;
        if (sy < 0 || sy >= in_h) continue;
        const T* row = base + (size_t)sy * in_w;
        for (int kx = 0; kx < kernel_w; kx++)
        {
            const int sx = sx0 + kx;
            if (sx < 0 || sx >= in_w) continue;
            const T v = row[sx];
            if (v > m) m = v;
        }
    }

    output[(size_t)c * out_cstep + (size_t)oy * out_w + ox] = m;
}

void pooling_cuda_forward_max_pad(const CudaMat& bottom_blob, CudaMat& top_blob,
                                  int kernel_w, int kernel_h, int stride_w, int stride_h,
                                  int pad_left, int pad_top)
{
    const int total = (int)top_blob.c * top_blob.h * top_blob.w;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    if (bottom_blob.elemsize == 1u)
    {
        gpu_pooling_cuda_forward_max_pad_kernel<signed char><<<blocks, threads>>>(
            static_cast<const signed char*>(bottom_blob.get_craw_data()),
            bottom_blob.w, bottom_blob.h, (int)bottom_blob.cstep,
            static_cast<signed char*>(top_blob.get_raw_data()),
            top_blob.w, top_blob.h, (int)top_blob.cstep,
            kernel_w, kernel_h, stride_w, stride_h, pad_left, pad_top,
            (signed char)-128, total);
    }
    else
    {
        gpu_pooling_cuda_forward_max_pad_kernel<float><<<blocks, threads>>>(
            static_cast<const float*>(bottom_blob.get_craw_data()),
            bottom_blob.w, bottom_blob.h, (int)bottom_blob.cstep,
            static_cast<float*>(top_blob.get_raw_data()),
            top_blob.w, top_blob.h, (int)top_blob.cstep,
            kernel_w, kernel_h, stride_w, stride_h, pad_left, pad_top,
            -FLT_MAX, total);
    }
}

'''


def repl(path, old, new):
    with io.open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if old not in text:
        raise SystemExit("marker missing in %s: %r" % (path, old[:70]))
    text = text.replace(old, new, 1)
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def main():
    repl(CPP, CPP_DECL_OLD, CPP_DECL_NEW)
    repl(CPP, INC_OLD, INC_NEW)
    repl(CPP, CPP_BRANCH_OLD, CPP_BRANCH_NEW)
    with io.open(CU, "r", encoding="utf-8") as f:
        cu = f.read()
    pos = cu.rindex("}")
    cu = cu[:pos] + CU_KERNEL + "}\n"
    with io.open(CU, "w", encoding="utf-8", newline="\n") as f:
        f.write(cu)
    print("patched fused maxpool padding")


if __name__ == "__main__":
    main()
