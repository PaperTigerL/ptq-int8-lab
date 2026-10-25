#!/usr/bin/env python3
"""Split-K for the implicit int8 convolution (small feature maps).

Motivation: one warp per output pixel, 32 output channels per block. On the
4x4 stage (M=16, Cout=512) that is only 32 blocks (256 warps) on a 14-SM GPU
with ~56 resident blocks, so 1152 dependent dp4a per lane cannot be hidden.

Design:
  - split K = C4*kh*kw into S contiguous ranges;
  - grid (M/8, Cout/32, S); each block writes an int32 partial;
  - a reduction kernel sums the S partials and applies the original epilogue;
  - int32 addition is associative and the total < 2^31, so the accumulator is
    identical to the non-split kernel -> logits bit-identical;
  - S is selected from output blocks vs resident blocks; layers that already
    fill the GPU (or taps<9 / K<512) keep the original kernel;
  - NCNN_CUDA_SPLIT=1|2|4 overrides S for experiments (1 = force off).
"""
import io

H_PATH = "/home/user/ncnn-cuda/src/layer/cuda/implicit_conv_cuda.h"
CPP_H_PATH = "/home/user/ncnn-cuda/src/layer/cuda/convolution_cuda.h"
CU_PATH = "/home/user/ncnn-cuda/src/layer/cuda/implicit_conv_cuda.cu"
CPP_PATH = "/home/user/ncnn-cuda/src/layer/cuda/convolution_cuda.cpp"

H_OLD = """void implicit_pack_padded_int8_c4_launch(const signed char* input, int in_cstep,
                                         int C, int H, int W,
                                         int pad_top, int pad_left,
                                         int pad_bottom, int pad_right,
                                         signed char* packed, int PH, int PW);"""
H_NEW = """void implicit_pack_padded_int8_c4_launch(const signed char* input, int in_cstep,
                                         int C, int H, int W,
                                         int pad_top, int pad_left,
                                         int pad_bottom, int pad_right,
                                         signed char* packed, int PH, int PW);

// Split-K：把 K=C4*kh*kw 拆成 split 段写 int32 部分和，再由 reduce 做 epilogue。
// partial 布局：[split][M][Cout]，M=OH*OW。
int implicit_conv_int8_recommend_split(int M, int Cout, int C4, int kh, int kw);

void implicit_conv_int8_split_launch(const signed char* packed,
                                     int C, int H, int W,
                                     const signed char* wimp,
                                     int Cout, int kh, int kw, int dh, int dw,
                                     int sh, int sw, int OH, int OW,
                                     int split, int* partial);

void implicit_conv_int8_reduce_launch(const int* partial, int split, int M, int Cout,
                                      const float* wscales, const float* bias,
                                      const float* bottom_scale, const float* top_scale,
                                      bool requantize, int activation_type,
                                      void* output, int out_cstep);"""

H_MEMBER_OLD = "    mutable CudaMat gpu_implicit_pack;"
H_MEMBER_NEW = """    mutable CudaMat gpu_implicit_pack;  // [C4][H*W][4] int8, 每层持久
    mutable CudaMat gpu_implicit_partial;  // [split][M][Cout] int32, split-K 部分和"""

CU_KERNELS = r'''
// Split-K convolution: same mapping as implicit_conv_int8_kernel, but each
// block only covers the K range [k0, k1) and stores an int32 partial.
__global__ void implicit_conv_int8_split_kernel(const signed char* __restrict__ packed,
                                                const signed char* __restrict__ wimp,
                                                int C4, int H, int W,
                                                int Cout, int kh, int kw, int dh, int dw,
                                                int sh, int sw, int OH, int OW,
                                                int split, int* __restrict__ partial)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int M = OH * OW;
    const int m = blockIdx.x * 8 + warp;
    const int co = blockIdx.y * 32 + lane;
    if (m >= M || co >= Cout) return;

    const int split_id = blockIdx.z;
    const int oh = m / OW;
    const int ow = m - oh * OW;
    const int S = H * W;
    const int K = kh * kw * C4;
    const int k0 = (int)((long long)K * split_id / split);
    const int k1 = (int)((long long)K * (split_id + 1) / split);
    const int* qin = reinterpret_cast<const int*>(packed);
    const int* win = reinterpret_cast<const int*>(wimp);

    int acc = 0;
    int k = k0;
    while (k < k1)
    {
        const int tap = k / C4;
        const int i = tap / kw;
        const int j = tap - i * kw;
        const int pix = (oh * sh + i * dh) * W + (ow * sw + j * dw);
        int c4 = k - tap * C4;
        int c4end = k1 - tap * C4;
        if (c4end > C4) c4end = C4;
        for (; c4 < c4end; c4++)
        {
            const int a = qin[(size_t)c4 * S + pix];
            const int w = win[((size_t)tap * C4 + c4) * Cout + co];
            acc = dp4a_i8(a, w, acc);
        }
        k = tap * C4 + c4end;
    }

    partial[((size_t)split_id * M + m) * Cout + co] = acc;
}

// 汇总 split 个 int32 部分和，并做与原 kernel 完全相同的 epilogue。
__global__ void implicit_conv_int8_reduce_kernel(const int* __restrict__ partial,
                                                 int split,
                                                 const float* __restrict__ wscales,
                                                 const float* __restrict__ bias,
                                                 const float* __restrict__ bottom_scale,
                                                 const float* __restrict__ top_scale,
                                                 bool requantize, int activation_type,
                                                 void* __restrict__ output, int out_cstep,
                                                 int total, int Cout)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int m = idx / Cout;
    const int co = idx - m * Cout;

    int acc = 0;
    for (int s = 0; s < split; s++)
        acc += partial[(size_t)s * total + idx];

    const float ws = wscales[co];
    const float scale_in = (ws == 0.0f) ? 0.0f : 1.0f / (*bottom_scale * ws);
    float y = (float)acc * scale_in;
    if (bias) y += bias[co];

    const size_t oidx = (size_t)co * out_cstep + m;
    if (requantize)
    {
        signed char q = float2int8_ncnn(y * (*top_scale));
        if (activation_type == 1 && q < 0) q = 0;
        reinterpret_cast<signed char*>(output)[oidx] = q;
    }
    else
    {
        if (activation_type == 1 && y < 0.0f) y = 0.0f;
        reinterpret_cast<float*>(output)[oidx] = y;
    }
}

int implicit_conv_int8_recommend_split(int M, int Cout, int C4, int kh, int kw)
{
    const int taps = kh * kw;
    const int K = C4 * taps;
    if (taps < 9 || K < 512 || M <= 0 || Cout <= 0) return 1;

    static int sm_count = -1;
    if (sm_count < 0)
    {
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) dev = 0;
        int n = 0;
        if (cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || n <= 0)
            n = 14;
        sm_count = n;
    }

    // 原 kernel：256 线程、64 寄存器 -> 约 4 块/SM 可驻留。
    const int resident_blocks = sm_count * 4;
    const int blocks = ((M + 7) / 8) * ((Cout + 31) / 32);
    int split = (blocks >= resident_blocks) ? 1 : (resident_blocks + blocks - 1) / blocks;
    if (split > 4) split = 4;
    if (split < 1) split = 1;
    return split;
}

void implicit_conv_int8_split_launch(const signed char* packed,
                                     int C, int H, int W,
                                     const signed char* wimp,
                                     int Cout, int kh, int kw, int dh, int dw,
                                     int sh, int sw, int OH, int OW,
                                     int split, int* partial)
{
    const int C4 = (C + 3) / 4;
    const int M = OH * OW;
    const dim3 grid((M + 7) / 8, (Cout + 31) / 32, split);
    implicit_conv_int8_split_kernel<<<grid, 256>>>(
        packed, wimp, C4, H, W, Cout, kh, kw, dh, dw, sh, sw, OH, OW, split, partial);
}

void implicit_conv_int8_reduce_launch(const int* partial, int split, int M, int Cout,
                                      const float* wscales, const float* bias,
                                      const float* bottom_scale, const float* top_scale,
                                      bool requantize, int activation_type,
                                      void* output, int out_cstep)
{
    const int total = M * Cout;
    implicit_conv_int8_reduce_kernel<<<(total + 255) / 256, 256>>>(
        partial, split, wscales, bias, bottom_scale, top_scale,
        requantize, activation_type, output, out_cstep, total, Cout);
}

} // namespace ncnn
'''

CPP_LAMBDA_ANCHOR = "    const size_t out_elemsize_fused = use_int8_requantize ? 1u : 4u;"
CPP_LAMBDA = r'''    const size_t out_elemsize_fused = use_int8_requantize ? 1u : 4u;

    // implicit conv 分发：小特征图按占用率选 split-K，其余走原 kernel。
    auto launch_implicit_conv = [&](const signed char* packed, int C, int IH, int IW,
                                    int OW_, int OH_) {
        const int C4 = (C + 3) / 4;
        const int M = OH_ * OW_;
        int split = implicit_conv_int8_recommend_split(M, num_output, C4, kernel_h, kernel_w);
        const char* split_env = std::getenv("NCNN_CUDA_SPLIT");
        if (split_env)
        {
            const int v = atoi(split_env);
            split = (v <= 1) ? 1 : (v > 4 ? 4 : v);
        }
        if (split > 1)
        {
            const size_t partial_elems = (size_t)split * M * num_output;
            if ((size_t)gpu_implicit_partial.total() < partial_elems)
                gpu_implicit_partial.create((int)partial_elems, (size_t)4u, _cuda_allocator);
            if (!gpu_implicit_partial.empty())
            {
                implicit_conv_int8_split_launch(
                    packed, C, IH, IW,
                    static_cast<const signed char*>(gpu_weight_data_implicit.get_craw_data()),
                    num_output, kernel_w, kernel_h, dilation_w, dilation_h,
                    stride_w, stride_h, OW_, OH_, split,
                    static_cast<int*>(gpu_implicit_partial.get_raw_data()));
                implicit_conv_int8_reduce_launch(
                    static_cast<const int*>(gpu_implicit_partial.get_raw_data()), split, M, num_output,
                    static_cast<const float*>(gpu_weight_data_int8_scales.get_craw_data()),
                    bias_term ? static_cast<const float*>(gpu_bias_data.get_craw_data()) : nullptr,
                    gpu_bottom_blob_int8_scale, gpu_top_blob_int8_scale,
                    use_int8_requantize, activation_type,
                    top_blob.get_raw_data(), (int)top_blob.cstep);
                return;
            }
        }
        implicit_conv_int8_launch(
            packed, C, IH, IW,
            static_cast<const signed char*>(gpu_weight_data_implicit.get_craw_data()),
            static_cast<const float*>(gpu_weight_data_int8_scales.get_craw_data()),
            bias_term ? static_cast<const float*>(gpu_bias_data.get_craw_data()) : nullptr,
            gpu_bottom_blob_int8_scale, gpu_top_blob_int8_scale,
            num_output, kernel_w, kernel_h, dilation_w, dilation_h,
            stride_w, stride_h, OW_, OH_,
            use_int8_requantize, activation_type,
            top_blob.get_raw_data(), (int)top_blob.cstep);
    };'''

CALL_FUSED_OLD = """            implicit_conv_int8_launch(
                static_cast<const signed char*>(gpu_implicit_pack.get_craw_data()),
                C, PH, PW,
                static_cast<const signed char*>(gpu_weight_data_implicit.get_craw_data()),
                static_cast<const float*>(gpu_weight_data_int8_scales.get_craw_data()),
                bias_term ? static_cast<const float*>(gpu_bias_data.get_craw_data()) : nullptr,
                gpu_bottom_blob_int8_scale, gpu_top_blob_int8_scale,
                num_output, kernel_w, kernel_h, dilation_w, dilation_h,
                stride_w, stride_h, outw, outh,
                use_int8_requantize, activation_type,
                top_blob.get_raw_data(), (int)top_blob.cstep);"""
CALL_FUSED_NEW = """            launch_implicit_conv(
                static_cast<const signed char*>(gpu_implicit_pack.get_craw_data()),
                C, PH, PW, outw, outh);"""

CALL_PLAIN_OLD = """            implicit_conv_int8_launch(
                static_cast<const signed char*>(gpu_implicit_pack.get_craw_data()),
                C, bh, bw,
                static_cast<const signed char*>(gpu_weight_data_implicit.get_craw_data()),
                static_cast<const float*>(gpu_weight_data_int8_scales.get_craw_data()),
                bias_term ? static_cast<const float*>(gpu_bias_data.get_craw_data()) : nullptr,
                gpu_bottom_blob_int8_scale, gpu_top_blob_int8_scale,
                num_output, kernel_w, kernel_h, dilation_w, dilation_h,
                stride_w, stride_h, outw, outh,
                use_int8_requantize, activation_type,
                top_blob.get_raw_data(), (int)top_blob.cstep);"""
CALL_PLAIN_NEW = """            launch_implicit_conv(
                static_cast<const signed char*>(gpu_implicit_pack.get_craw_data()),
                C, bh, bw, outw, outh);"""


def replace_once(path, old, new):
    with io.open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if old not in text:
        raise SystemExit("marker missing in %s: %r" % (path, old[:70]))
    text = text.replace(old, new, 1)
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def main():
    replace_once(H_PATH, H_OLD, H_NEW)
    replace_once(CPP_H_PATH, H_MEMBER_OLD, H_MEMBER_NEW)
    with io.open(CU_PATH, "r", encoding="utf-8") as f:
        cu = f.read()
    pos = cu.rindex("} // namespace ncnn")
    cu = cu[:pos] + CU_KERNELS
    with io.open(CU_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(cu)
    replace_once(CPP_PATH, CPP_LAMBDA_ANCHOR, CPP_LAMBDA)
    replace_once(CPP_PATH, CALL_FUSED_OLD, CALL_FUSED_NEW)
    replace_once(CPP_PATH, CALL_PLAIN_OLD, CALL_PLAIN_NEW)
    print("patched split-K")


if __name__ == "__main__":
    main()
