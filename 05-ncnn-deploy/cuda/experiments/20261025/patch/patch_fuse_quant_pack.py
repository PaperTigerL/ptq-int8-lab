#!/usr/bin/env python3
"""H-074: fuse float->int8 quantize into the pack kernel (opt-in).

Current flow for every conv with float input:
    quantize_float32_to_int8 (float CHW -> int8 CHW)  +  implicit_pack_* (int8 CHW -> [C4][HW][4])
Two kernels and an extra int8 intermediate. The fused kernel reads the float CHW
directly, applies the same q = clamp(round(v * bottom_scale)) and writes the same
packed int32, so packed bytes and all downstream integers are unchanged.

Env: NCNN_CUDA_FUSE_QUANT_PACK=1.
"""
import io
import sys

CU = "/home/user/ncnn-cuda/src/layer/cuda/implicit_conv_cuda.cu"
HDR = "/home/user/ncnn-cuda/src/layer/cuda/implicit_conv_cuda.h"
CPP = "/home/user/ncnn-cuda/src/layer/cuda/convolution_cuda.cpp"

CU_KERNELS = r"""
// 与 quantize_cuda.cu 的 gpu_float2int8 完全相同的取整与截断。
__device__ __forceinline__ signed char implicit_quant_i8(float v)
{
    int int32 = static_cast<int>(round(v));
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
}

// fuse: float CHW -> [C4][H*W][4] int8，等价于 quantize_float32_to_int8 + pack。
__global__ void implicit_quant_pack_int8_c4_kernel(const float* __restrict__ input,
                                                   int in_cstep, int C, int H, int W,
                                                   float scale,
                                                   signed char* __restrict__ packed,
                                                   int C4)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int S = H * W;
    if (idx >= C4 * S) return;
    const int c4 = idx / S;
    const int pix = idx - c4 * S;
    int v = 0;
#pragma unroll
    for (int lane = 0; lane < 4; lane++)
    {
        const int c = c4 * 4 + lane;
        if (c < C)
        {
            const float x = input[(size_t)c * in_cstep + pix];
            const int q = (int)implicit_quant_i8(x * scale);
            v |= (q & 0xff) << (lane * 8);
        }
    }
    reinterpret_cast<int*>(packed)[idx] = v;
}

// 同上，但读未 padding 的 float CHW，边界写 0（等价于 pad_value=0 的量化结果）。
__global__ void implicit_quant_pack_padded_int8_c4_kernel(const float* __restrict__ input,
                                                          int in_cstep, int C, int H, int W,
                                                          int pad_top, int pad_left,
                                                          float scale,
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
                const float f = input[(size_t)c * in_cstep + (size_t)y * W + x];
                const int q = (int)implicit_quant_i8(f * scale);
                v |= (q & 0xff) << (lane * 8);
            }
        }
    }
    reinterpret_cast<int*>(packed)[idx] = v;
}

"""

CU_LAUNCHERS = r"""
void implicit_quant_pack_int8_c4_launch(const float* input, int in_cstep,
                                        int C, int H, int W, float scale,
                                        signed char* packed)
{
    const int S = H * W;
    const int C4 = (C + 3) / 4;
    const int total = C4 * S;
    implicit_quant_pack_int8_c4_kernel<<<(total + 255) / 256, 256>>>(
        input, in_cstep, C, H, W, scale, packed, C4);
}

void implicit_quant_pack_padded_int8_c4_launch(const float* input, int in_cstep,
                                               int C, int H, int W,
                                               int pad_top, int pad_left,
                                               float scale, signed char* packed,
                                               int PH, int PW)
{
    const int C4 = (C + 3) / 4;
    const int total = C4 * PH * PW;
    implicit_quant_pack_padded_int8_c4_kernel<<<(total + 255) / 256, 256>>>(
        input, in_cstep, C, H, W, pad_top, pad_left, scale, packed, PH, PW, C4);
}

"""

CU_ANCHOR = "// 读到边界外时写 0，等价于 copy_make_border(BORDER_CONSTANT, pad_value=0)。\n"

HDR_ANCHOR = "} // namespace ncnn\n"
HDR_DECL = """// float CHW -> [C4][H*W][4] int8：quantize_float32_to_int8 + pack 的融合版。
void implicit_quant_pack_int8_c4_launch(const float* input, int in_cstep,
                                        int C, int H, int W, float scale,
                                        signed char* packed);

void implicit_quant_pack_padded_int8_c4_launch(const float* input, int in_cstep,
                                               int C, int H, int W,
                                               int pad_top, int pad_left,
                                               float scale, signed char* packed,
                                               int PH, int PW);

"""

OLD_QUANT = """    CudaMat bottom_blob_unbordered = bottom_blob;
    if (elemsize != 1)
    {
        Option opt_g = opt;
"""
NEW_QUANT = """    const char* implicit_env = std::getenv("NCNN_CUDA_IMPLICIT");
    // Default ON. NCNN_CUDA_IMPLICIT=0 keeps the old kernel as an explicit fallback.
    const bool use_implicit = !implicit_env || !(implicit_env[0] == '0' && implicit_env[1] == '\\0');

    CudaMat bottom_blob_unbordered = bottom_blob;
    // opt-in：float 输入先量化再 pack 的两颗 kernel 合成一颗（逐位一致）。
    static const bool fuse_quant_pack = []() {
        const char* e = std::getenv("NCNN_CUDA_FUSE_QUANT_PACK");
        return e && e[0] == '1' && e[1] == '\\0';
    }();
    const bool pads_all_zero = pad_left == 0 && pad_right == 0 && pad_top == 0 && pad_bottom == 0;
    const bool pads_fusable = pad_value == 0.0f
        && pad_left >= 0 && pad_right >= 0 && pad_top >= 0 && pad_bottom >= 0;
    const bool quant_pack_fusable = fuse_quant_pack && use_implicit && elemsize != 1
        && bottom_blob.elempack == 1 && int8_scale_term
        && !gpu_weight_data_implicit.empty()
        && (activation_type == 0 || activation_type == 1 || activation_type == 3)
        && (pads_all_zero || pads_fusable);
    if (elemsize != 1 && !quant_pack_fusable)
    {
        Option opt_g = opt;
"""

OLD_ENV = """    const char* implicit_env = std::getenv("NCNN_CUDA_IMPLICIT");
    // Default ON. NCNN_CUDA_IMPLICIT=0 keeps the old kernel as an explicit fallback.
    const bool use_implicit = !implicit_env || !(implicit_env[0] == '0' && implicit_env[1] == '\\0');
    const size_t out_elemsize_fused = use_int8_requantize ? 1u : 4u;
"""
NEW_ENV = """    const size_t out_elemsize_fused = use_int8_requantize ? 1u : 4u;
"""

FUSED_BRANCH = """    // 融合分支：直接用 float 输入做 quantize+pack，省掉中间 int8 缓冲和一颗 kernel。
    if (quant_pack_fusable)
    {
        const int C = bottom_blob.c;
        const int H = bottom_blob.h;
        const int W = bottom_blob.w;
        const int C4 = (C + 3) / 4;
        if (pads_all_zero)
        {
            const int outw = (W - kernel_extent_w) / stride_w + 1;
            const int outh = (H - kernel_extent_h) / stride_h + 1;
            const size_t pack_elems = (size_t)C4 * H * W * 4;
            if ((size_t)gpu_implicit_pack.total() != pack_elems)
                gpu_implicit_pack.create((int)pack_elems, (size_t)1u, _cuda_allocator);
            if (!gpu_implicit_pack.empty())
            {
                implicit_quant_pack_int8_c4_launch(
                    static_cast<const float*>(bottom_blob.get_craw_data()),
                    (int)bottom_blob.cstep, C, H, W, bottom_blob_int8_scale,
                    static_cast<signed char*>(gpu_implicit_pack.get_raw_data()));
                top_blob.create(outw, outh, num_output, out_elemsize_fused, opt.blob_cuda_allocator);
                if (top_blob.empty()) return -100;
                launch_implicit_conv(static_cast<const signed char*>(gpu_implicit_pack.get_craw_data()),
                                     C, H, W, outw, outh);
                return 0;
            }
        }
        else
        {
            const int PH = H + pad_top + pad_bottom;
            const int PW = W + pad_left + pad_right;
            const int outw = (PW - kernel_extent_w) / stride_w + 1;
            const int outh = (PH - kernel_extent_h) / stride_h + 1;
            const size_t pack_elems = (size_t)C4 * PH * PW * 4;
            if ((size_t)gpu_implicit_pack.total() != pack_elems)
                gpu_implicit_pack.create((int)pack_elems, (size_t)1u, _cuda_allocator);
            if (!gpu_implicit_pack.empty())
            {
                implicit_quant_pack_padded_int8_c4_launch(
                    static_cast<const float*>(bottom_blob.get_craw_data()),
                    (int)bottom_blob.cstep, C, H, W, pad_top, pad_left,
                    bottom_blob_int8_scale,
                    static_cast<signed char*>(gpu_implicit_pack.get_raw_data()), PH, PW);
                top_blob.create(outw, outh, num_output, out_elemsize_fused, opt.blob_cuda_allocator);
                if (top_blob.empty()) return -100;
                launch_implicit_conv(static_cast<const signed char*>(gpu_implicit_pack.get_craw_data()),
                                     C, PH, PW, outw, outh);
                return 0;
            }
        }
        // pack 缓冲分配失败：补回量化，继续走原路径。
        Option opt_g = opt;
        opt_g.blob_allocator = opt.workspace_allocator;
        opt_g.blob_cuda_allocator = opt.workspace_cuda_allocator;
        quantize_float32_to_int8(bottom_blob, bottom_blob_unbordered, bottom_blob_int8_scale, opt_g);
    }

    // 显式非负 pad 且 pad_value==0 时，把 copy_make_border 融进 int8 pack：
"""

CPP_FUSED_ANCHOR = "    // 显式非负 pad 且 pad_value==0 时，把 copy_make_border 融进 int8 pack：\n"


def patch(path, pairs):
    text = io.open(path, encoding="utf-8").read()
    for old, new, count in pairs:
        if text.count(old) != count:
            sys.exit("marker count mismatch in %s (%d != %d): %r" %
                     (path, text.count(old), count, old[:70]))
        text = text.replace(old, new)
    io.open(path, "w", encoding="utf-8", newline="\n").write(text)


patch(CU, [(CU_ANCHOR, CU_KERNELS + CU_LAUNCHERS + CU_ANCHOR, 1)])
patch(HDR, [(HDR_ANCHOR, HDR_DECL + HDR_ANCHOR, 1)])
patch(CPP, [
    (OLD_QUANT, NEW_QUANT, 1),
    (OLD_ENV, NEW_ENV, 1),
    (CPP_FUSED_ANCHOR, FUSED_BRANCH, 1),
])
print("fuse quant+pack patched")
