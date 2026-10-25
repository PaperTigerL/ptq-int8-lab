#!/usr/bin/env python3
"""Fuse constant-zero padding into the implicit int8 pack kernel.

Old implicit path per conv: copy_make_border (pad kernel) -> bordered int8
buffer -> pack kernel -> packed [C4][PH*PW][4].
New path: one pack kernel reads the unbordered int8 buffer and writes border
zeros directly, so the pad kernel and the intermediate bordered buffer vanish.

Equivalence: the packed bytes are identical (same zero border, same channel
values); conv kernels/weights/offsets are unchanged.  Only explicit
non-negative pads with pad_value == 0 take the fused path; SAME(-233/-234)
and non-zero pad values keep the old path.
"""
import io

H_PATH = "/home/user/ncnn-cuda/src/layer/cuda/implicit_conv_cuda.h"
CU_PATH = "/home/user/ncnn-cuda/src/layer/cuda/implicit_conv_cuda.cu"
CPP_PATH = "/home/user/ncnn-cuda/src/layer/cuda/convolution_cuda.cpp"

H_OLD = """void implicit_pack_int8_c4_launch(const signed char* input, int in_cstep,
                                  int C, int H, int W, signed char* packed);"""
H_NEW = """void implicit_pack_int8_c4_launch(const signed char* input, int in_cstep,
                                  int C, int H, int W, signed char* packed);

// 读未 padding 的 int8 CHW，直接写带零边界的 [C4][PH*PW][4]。
// 等价于旧路径 copy_make_border(pad=0) + implicit_pack_int8_c4_launch。
void implicit_pack_padded_int8_c4_launch(const signed char* input, int in_cstep,
                                         int C, int H, int W,
                                         int pad_top, int pad_left,
                                         int pad_bottom, int pad_right,
                                         signed char* packed, int PH, int PW);"""

CU_OLD_LAUNCH = """void implicit_pack_int8_c4_launch(const signed char* input, int in_cstep,
                                  int C, int H, int W, signed char* packed)
{
    const int S = H * W;
    const int C4 = (C + 3) / 4;
    const int total = C4 * S;
    implicit_pack_int8_c4_kernel<<<(total + 255) / 256, 256>>>(
        input, in_cstep, C, S, packed, C4);
}"""

CU_NEW = r'''// 读到边界外时写 0，等价于 copy_make_border(BORDER_CONSTANT, pad_value=0)。
__global__ void implicit_pack_padded_int8_c4_kernel(const signed char* __restrict__ input,
                                                    int in_cstep, int C, int H, int W,
                                                    int pad_top, int pad_left,
                                                    signed char* __restrict__ packed,
                                                    int PH, int PW, int C4)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int S = PH * PW;
    if (idx >= C4 * S) return;
    const int c4 = idx / S;
    const int pix = idx - c4 * S;
    const int py = pix / PW;
    const int px = pix - py * PW;
    const int y = py - pad_top;
    const int x = px - pad_left;
    int v = 0;
    if ((unsigned)y < (unsigned)H && (unsigned)x < (unsigned)W)
    {
#pragma unroll
        for (int lane = 0; lane < 4; lane++)
        {
            const int c = c4 * 4 + lane;
            if (c < C)
            {
                const int q = (int)(unsigned char)input[(size_t)c * in_cstep + (size_t)y * W + x];
                v |= (q & 0xff) << (lane * 8);
            }
        }
    }
    reinterpret_cast<int*>(packed)[idx] = v;
}

void implicit_pack_int8_c4_launch(const signed char* input, int in_cstep,
                                  int C, int H, int W, signed char* packed)
{
    const int S = H * W;
    const int C4 = (C + 3) / 4;
    const int total = C4 * S;
    implicit_pack_int8_c4_kernel<<<(total + 255) / 256, 256>>>(
        input, in_cstep, C, S, packed, C4);
}

void implicit_pack_padded_int8_c4_launch(const signed char* input, int in_cstep,
                                         int C, int H, int W,
                                         int pad_top, int pad_left,
                                         int pad_bottom, int pad_right,
                                         signed char* packed, int PH, int PW)
{
    (void)pad_bottom;
    (void)pad_right;
    const int C4 = (C + 3) / 4;
    const int total = C4 * PH * PW;
    implicit_pack_padded_int8_c4_kernel<<<(total + 255) / 256, 256>>>(
        input, in_cstep, C, H, W, pad_top, pad_left, packed, PH, PW, C4);
}'''

CPP_OLD = """    CudaMat bottom_blob_bordered;
    make_padding(bottom_blob_unbordered, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;"""

CPP_NEW = """    const char* implicit_env = std::getenv("NCNN_CUDA_IMPLICIT");
    // Default ON. NCNN_CUDA_IMPLICIT=0 keeps the old kernel as an explicit fallback.
    const bool use_implicit = !implicit_env || !(implicit_env[0] == '0' && implicit_env[1] == '\\0');
    const size_t out_elemsize_fused = use_int8_requantize ? 1u : 4u;

    // 显式非负 pad 且 pad_value==0 时，把 copy_make_border 融进 int8 pack：
    // 少一次 pad kernel，也少了 bordered 中间缓冲。
    const bool can_fuse_pad_pack = use_implicit && int8_scale_term
        && bottom_blob_unbordered.elempack == 1
        && !gpu_weight_data_implicit.empty()
        && (activation_type == 0 || activation_type == 1)
        && pad_value == 0.0f
        && pad_left >= 0 && pad_right >= 0 && pad_top >= 0 && pad_bottom >= 0
        && (pad_left + pad_right + pad_top + pad_bottom) > 0;

    if (can_fuse_pad_pack)
    {
        const int C = bottom_blob_unbordered.c;
        const int H = bottom_blob_unbordered.h;
        const int W = bottom_blob_unbordered.w;
        const int PH = H + pad_top + pad_bottom;
        const int PW = W + pad_left + pad_right;
        const int outw = (PW - kernel_extent_w) / stride_w + 1;
        const int outh = (PH - kernel_extent_h) / stride_h + 1;
        const int C4 = (C + 3) / 4;
        const size_t pack_elems = (size_t)C4 * PH * PW * 4;
        if ((size_t)gpu_implicit_pack.total() != pack_elems)
            gpu_implicit_pack.create((int)pack_elems, (size_t)1u, _cuda_allocator);
        if (!gpu_implicit_pack.empty())
        {
            implicit_pack_padded_int8_c4_launch(
                static_cast<const signed char*>(bottom_blob_unbordered.get_craw_data()),
                (int)bottom_blob_unbordered.cstep, C, H, W,
                pad_top, pad_left, pad_bottom, pad_right,
                static_cast<signed char*>(gpu_implicit_pack.get_raw_data()), PH, PW);

            top_blob.create(outw, outh, num_output, out_elemsize_fused, opt.blob_cuda_allocator);
            if (top_blob.empty()) return -100;
            implicit_conv_int8_launch(
                static_cast<const signed char*>(gpu_implicit_pack.get_craw_data()),
                C, PH, PW,
                static_cast<const signed char*>(gpu_weight_data_implicit.get_craw_data()),
                static_cast<const float*>(gpu_weight_data_int8_scales.get_craw_data()),
                bias_term ? static_cast<const float*>(gpu_bias_data.get_craw_data()) : nullptr,
                gpu_bottom_blob_int8_scale, gpu_top_blob_int8_scale,
                num_output, kernel_w, kernel_h, dilation_w, dilation_h,
                stride_w, stride_h, outw, outh,
                use_int8_requantize, activation_type,
                top_blob.get_raw_data(), (int)top_blob.cstep);
            return 0;
        }
    }

    CudaMat bottom_blob_bordered;
    make_padding(bottom_blob_unbordered, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;"""

# The later duplicate declarations must go away.
CPP_OLD_LATER = """    // int8
    size_t out_elemsize = use_int8_requantize ? 1u : 4u;

    const char* implicit_env = std::getenv("NCNN_CUDA_IMPLICIT");
    // Default ON. NCNN_CUDA_IMPLICIT=0 keeps the old kernel as an explicit fallback.
    const bool use_implicit = !implicit_env || !(implicit_env[0] == '0' && implicit_env[1] == '\\0');
    if (use_implicit && int8_scale_term"""
CPP_NEW_LATER = """    // int8
    size_t out_elemsize = out_elemsize_fused;

    if (use_implicit && int8_scale_term"""


def main():
    for path, old, new in ((H_PATH, H_OLD, H_NEW), (CU_PATH, CU_OLD_LAUNCH, CU_NEW),
                           (CPP_PATH, CPP_OLD, CPP_NEW), (CPP_PATH, CPP_OLD_LATER, CPP_NEW_LATER)):
        with io.open(path, "r", encoding="utf-8") as f:
            text = f.read()
        if old not in text:
            raise SystemExit("marker missing in %s: %r" % (path, old[:60]))
        text = text.replace(old, new, 1)
        with io.open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    print("patched fused pad+pack")


if __name__ == "__main__":
    main()
