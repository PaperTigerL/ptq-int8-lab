#!/usr/bin/env python3
"""H-079: drop padded pack (border write amplification) by making the conv
kernel skip out-of-range taps.

Today a padded conv does quant_pack_padded: it writes (H+2p)*(W+2p) packed
int32 where only H*W are real payload; for 2x2 maps that is 4x the writes.
Variant: pack unpadded [C4][H*W][4] and let the conv skip taps outside the
feature map (same as multiplying by zero padding -> integer sum unchanged).

Env: NCNN_CUDA_CONV_BORDER=1. Only affects the fused quant+pack path.
"""
import io
import sys

CU = "/home/user/ncnn-cuda/src/layer/cuda/implicit_conv_cuda.cu"
HDR = "/home/user/ncnn-cuda/src/layer/cuda/implicit_conv_cuda.h"
CPP = "/home/user/ncnn-cuda/src/layer/cuda/convolution_cuda.cpp"

KERNEL = r"""
// 与 implicit_conv_int8_kernel 相同，但输入是**未 padding** 的 [C4][H*W][4]：
// 越界的 tap 直接跳过（等价于乘以 0 padding），权重下标仍按 tap 递增。
__global__ void implicit_conv_int8_kernel_border(const signed char* __restrict__ packed,
                                                 const signed char* __restrict__ wimp,
                                                 const float* __restrict__ wscales,
                                                 const float* __restrict__ bias,
                                                 const float* __restrict__ bottom_scale,
                                                 const float* __restrict__ top_scale,
                                                 int C4, int H, int W,
                                                 int Cout, int kh, int kw, int dh, int dw,
                                                 int sh, int sw, int OH, int OW,
                                                 int pad_top, int pad_left,
                                                 bool requantize, int activation_type,
                                                 const float* __restrict__ activation_params,
                                                 void* __restrict__ output, int out_cstep)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int M = OH * OW;
    const int m = blockIdx.x * 8 + warp;
    const int co = blockIdx.y * 32 + lane;
    if (m >= M || co >= Cout) return;

    const int oh = m / OW;
    const int ow = m - oh * OW;
    const int S = H * W;
    const int* qin = reinterpret_cast<const int*>(packed);
    const int* win = reinterpret_cast<const int*>(wimp);

    int acc = 0;
    int tap = 0;
    for (int i = 0; i < kh; i++)
    {
        const int ih = oh * sh + i * dh - pad_top;
        for (int j = 0; j < kw; j++)
        {
            const int iw = ow * sw + j * dw - pad_left;
            if ((unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W)
            {
                const int pix = ih * W + iw;
                for (int c4 = 0; c4 < C4; c4++)
                {
                    const int a = qin[(size_t)c4 * S + pix];
                    const int w = win[((size_t)tap * C4 + c4) * Cout + co];
                    acc = dp4a_i8(a, w, acc);
                }
            }
            tap++;
        }
    }

    const float ws = wscales[co];
    const float scale_in = (ws == 0.0f) ? 0.0f : 1.0f / (*bottom_scale * ws);
    float y = (float)acc * scale_in;
    if (bias) y += bias[co];

    const size_t oidx = (size_t)co * out_cstep + m;
    if (requantize)
    {
        signed char q = float2int8_ncnn(y * (*top_scale));
        if (activation_type == 1 && q < 0) q = 0;
        else if (activation_type == 3 && activation_params)
        {
            const signed char qmin = float2int8_ncnn(activation_params[0]);
            const signed char qmax = float2int8_ncnn(activation_params[1]);
            if (q < qmin) q = qmin;
            if (q > qmax) q = qmax;
        }
        reinterpret_cast<signed char*>(output)[oidx] = q;
    }
    else
    {
        if (activation_type == 1 && y < 0.0f) y = 0.0f;
        else if (activation_type == 3 && activation_params)
        {
            y = fminf(fmaxf(y, activation_params[0]), activation_params[1]);
        }
        reinterpret_cast<float*>(output)[oidx] = y;
    }
}

"""

LAUNCHER = r"""
void implicit_conv_int8_launch_border(const signed char* packed,
                                      int C, int H, int W,
                                      int pad_top, int pad_left,
                                      const signed char* wimp,
                                      const float* wscales,
                                      const float* bias,
                                      const float* bottom_scale,
                                      const float* top_scale,
                                      int Cout, int kh, int kw, int dh, int dw,
                                      int sh, int sw, int OH, int OW,
                                      bool requantize, int activation_type,
                                      const float* activation_params,
                                      void* output, int out_cstep)
{
    const int C4 = (C + 3) / 4;
    const int M = OH * OW;
    const dim3 grid((M + 7) / 8, (Cout + 31) / 32);
    implicit_conv_int8_kernel_border<<<grid, 256>>>(
        packed, wimp, wscales, bias, bottom_scale, top_scale,
        C4, H, W, Cout, kh, kw, dh, dw, sh, sw, OH, OW,
        pad_top, pad_left, requantize, activation_type, activation_params, output, out_cstep);
}

"""

CU_ANCHOR = "// 一个 warp = 同一行的两个相邻输出像素"

HDR_ANCHOR = "} // namespace ncnn\n"
HDR_DECL = """// 未 padding 的 packed 输入 + 越界 tap 跳过（等价 pad=0）。
void implicit_conv_int8_launch_border(const signed char* packed,
                                      int C, int H, int W,
                                      int pad_top, int pad_left,
                                      const signed char* wimp,
                                      const float* wscales,
                                      const float* bias,
                                      const float* bottom_scale,
                                      const float* top_scale,
                                      int Cout, int kh, int kw, int dh, int dw,
                                      int sh, int sw, int OH, int OW,
                                      bool requantize, int activation_type,
                                      const float* activation_params,
                                      void* output, int out_cstep);

"""

OLD_BRANCH = """        else
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
"""
NEW_BRANCH = """        else
        {
            const int PH = H + pad_top + pad_bottom;
            const int PW = W + pad_left + pad_right;
            const int outw = (PW - kernel_extent_w) / stride_w + 1;
            const int outh = (PH - kernel_extent_h) / stride_h + 1;
            // 省掉 pad 写放大：pack 不扩边，conv 跳过越界 tap（opt-in）。
            static const bool conv_border = []() {
                const char* e = std::getenv("NCNN_CUDA_CONV_BORDER");
                return e && e[0] == '1' && e[1] == '\\0';
            }();
            if (conv_border)
            {
                const size_t unpadded_elems = (size_t)C4 * H * W * 4;
                if ((size_t)gpu_implicit_pack.total() != unpadded_elems)
                    gpu_implicit_pack.create((int)unpadded_elems, (size_t)1u, _cuda_allocator);
                if (!gpu_implicit_pack.empty())
                {
                    implicit_quant_pack_int8_c4_launch(
                        static_cast<const float*>(bottom_blob.get_craw_data()),
                        (int)bottom_blob.cstep, C, H, W, bottom_blob_int8_scale,
                        static_cast<signed char*>(gpu_implicit_pack.get_raw_data()));
                    top_blob.create(outw, outh, num_output, out_elemsize_fused, opt.blob_cuda_allocator);
                    if (top_blob.empty()) return -100;
                    implicit_conv_int8_launch_border(
                        static_cast<const signed char*>(gpu_implicit_pack.get_craw_data()),
                        C, H, W, pad_top, pad_left,
                        static_cast<const signed char*>(gpu_weight_data_implicit.get_craw_data()),
                        static_cast<const float*>(gpu_weight_data_int8_scales.get_craw_data()),
                        bias_term ? static_cast<const float*>(gpu_bias_data.get_craw_data()) : nullptr,
                        gpu_bottom_blob_int8_scale, gpu_top_blob_int8_scale,
                        num_output, kernel_w, kernel_h, dilation_w, dilation_h,
                        stride_w, stride_h, outw, outh,
                        use_int8_requantize, activation_type, act_params,
                        top_blob.get_raw_data(), (int)top_blob.cstep);
                    return 0;
                }
            }
            const size_t pack_elems = (size_t)C4 * PH * PW * 4;
            if ((size_t)gpu_implicit_pack.total() != pack_elems)
                gpu_implicit_pack.create((int)pack_elems, (size_t)1u, _cuda_allocator);
            if (!gpu_implicit_pack.empty())
            {
"""


def patch(path, pairs):
    text = io.open(path, encoding="utf-8").read()
    for old, new, count in pairs:
        if text.count(old) != count:
            sys.exit("marker count mismatch in %s (%d != %d): %r" %
                     (path, text.count(old), count, old[:70]))
        text = text.replace(old, new)
    io.open(path, "w", encoding="utf-8", newline="\n").write(text)


patch(CU, [(CU_ANCHOR, KERNEL + LAUNCHER + CU_ANCHOR, 1)])
patch(HDR, [(HDR_ANCHOR, HDR_DECL + HDR_ANCHOR, 1)])
patch(CPP, [(OLD_BRANCH, NEW_BRANCH, 1)])
print("conv border patched")
