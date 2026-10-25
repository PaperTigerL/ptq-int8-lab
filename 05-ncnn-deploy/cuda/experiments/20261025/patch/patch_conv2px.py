#!/usr/bin/env python3
"""H-065: add a 2-pixels-per-warp implicit int8 conv kernel for 3x3 stride1.

One warp computes two horizontally adjacent output pixels and reuses each weight
int32 across both pixels. Integer accumulation order per pixel is unchanged
(identical dp4a sequence), so logits must stay bitwise identical.
Opt-in: NCNN_CUDA_CONV2PX=1.
"""
import io
import sys

CU = "/home/user/ncnn-cuda/src/layer/cuda/implicit_conv_cuda.cu"
HDR = "/home/user/ncnn-cuda/src/layer/cuda/implicit_conv_cuda.h"
CPP = "/home/user/ncnn-cuda/src/layer/cuda/convolution_cuda.cpp"

KERNEL = r"""
// 一个 warp = 同一行的两个相邻输出像素 (ow0, ow1)；每个权重 int32 在两个
// 像素间复用，输入 int 仍按像素分别读取。dp4a 序列与单像素 kernel 相同，
// 因此整数累加结果逐位一致。
__device__ __forceinline__ void implicit_conv_store(int acc, int m, int co,
                                                    const float* wscales,
                                                    const float* bias,
                                                    const float* bottom_scale,
                                                    const float* top_scale,
                                                    bool requantize, int activation_type,
                                                    const float* activation_params,
                                                    void* output, int out_cstep)
{
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

__global__ void implicit_conv_int8_kernel_2px(const signed char* __restrict__ packed,
                                              const signed char* __restrict__ wimp,
                                              const float* __restrict__ wscales,
                                              const float* __restrict__ bias,
                                              const float* __restrict__ bottom_scale,
                                              const float* __restrict__ top_scale,
                                              int C4, int H, int W,
                                              int Cout, int kh, int kw, int dh, int dw,
                                              int sh, int sw, int OH, int OW,
                                              bool requantize, int activation_type,
                                              const float* __restrict__ activation_params,
                                              void* __restrict__ output, int out_cstep)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int PW2 = (OW + 1) >> 1;              // 每行的像素对
    const int P = OH * PW2;                     // 总像素对
    const int p = blockIdx.x * 8 + warp;
    const int co = blockIdx.y * 32 + lane;
    if (p >= P || co >= Cout) return;

    const int oh = p / PW2;
    const int c2 = p - oh * PW2;
    const int ow0 = c2 * 2;
    const int ow1 = ow0 + 1;
    const bool has1 = ow1 < OW;

    const int S = H * W;
    const int* qin = reinterpret_cast<const int*>(packed);
    const int* win = reinterpret_cast<const int*>(wimp);

    int acc0 = 0;
    int acc1 = 0;
    int tap = 0;
    for (int i = 0; i < kh; i++)
    {
        const int ih = oh * sh + i * dh;
        for (int j = 0; j < kw; j++)
        {
            const int pix0 = ih * W + (ow0 * sw + j * dw);
            const int pix1 = ih * W + (ow1 * sw + j * dw);
            for (int c4 = 0; c4 < C4; c4++)
            {
                const int w = win[((size_t)tap * C4 + c4) * Cout + co];
                const int a0 = qin[(size_t)c4 * S + pix0];
                acc0 = dp4a_i8(a0, w, acc0);
                if (has1)
                {
                    const int a1 = qin[(size_t)c4 * S + pix1];
                    acc1 = dp4a_i8(a1, w, acc1);
                }
            }
            tap++;
        }
    }

    const int m0 = oh * OW + ow0;
    implicit_conv_store(acc0, m0, co, wscales, bias, bottom_scale, top_scale,
                        requantize, activation_type, activation_params, output, out_cstep);
    if (has1)
        implicit_conv_store(acc1, m0 + 1, co, wscales, bias, bottom_scale, top_scale,
                            requantize, activation_type, activation_params, output, out_cstep);
}

"""

LAUNCHER = r"""
void implicit_conv_int8_launch_2px(const signed char* packed,
                                   int C, int H, int W,
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
    const int PW2 = (OW + 1) >> 1;
    const int P = OH * PW2;
    const dim3 grid((P + 7) / 8, (Cout + 31) / 32);
    implicit_conv_int8_kernel_2px<<<grid, 256>>>(
        packed, wimp, wscales, bias, bottom_scale, top_scale,
        C4, H, W, Cout, kh, kw, dh, dw, sh, sw, OH, OW,
        requantize, activation_type, activation_params, output, out_cstep);
}

"""

CU_KERNEL_ANCHOR = "// 读到边界外时写 0，等价于 copy_make_border(BORDER_CONSTANT, pad_value=0)。\n"
CU_LAUNCH_ANCHOR = "// Split-K convolution: same mapping as implicit_conv_int8_kernel, but each\n"

HDR_ANCHOR = "} // namespace ncnn\n"
HDR_DECL = """// 3x3 stride1 的 2 像素/warp 版本；数值与 implicit_conv_int8_launch 逐位一致。
void implicit_conv_int8_launch_2px(const signed char* packed,
                                   int C, int H, int W,
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

CPP_ANCHOR = """        implicit_conv_int8_launch(
            packed, C, IH, IW,
"""
CPP_DISPATCH = """        // 3x3 stride1：2 像素/warp 寄存器分块（opt-in，逐位一致）。
        static const bool conv2px = []() {
            const char* e = std::getenv("NCNN_CUDA_CONV2PX");
            return e && e[0] == '1' && e[1] == '\\0';
        }();
        if (conv2px && kernel_w == 3 && kernel_h == 3
            && stride_w == 1 && stride_h == 1
            && dilation_w == 1 && dilation_h == 1 && OW_ >= 2)
        {
            implicit_conv_int8_launch_2px(
                packed, C, IH, IW,
                static_cast<const signed char*>(gpu_weight_data_implicit.get_craw_data()),
                static_cast<const float*>(gpu_weight_data_int8_scales.get_craw_data()),
                bias_term ? static_cast<const float*>(gpu_bias_data.get_craw_data()) : nullptr,
                gpu_bottom_blob_int8_scale, gpu_top_blob_int8_scale,
                num_output, kernel_w, kernel_h, dilation_w, dilation_h,
                stride_w, stride_h, OW_, OH_,
                use_int8_requantize, activation_type, act_params,
                top_blob.get_raw_data(), (int)top_blob.cstep);
            return;
        }

        implicit_conv_int8_launch(
            packed, C, IH, IW,
"""


def patch(path, pairs):
    text = io.open(path, encoding="utf-8").read()
    for old, new, count in pairs:
        if text.count(old) != count:
            sys.exit("marker count mismatch in %s (%d != %d): %r" %
                     (path, text.count(old), count, old[:60]))
        text = text.replace(old, new)
    io.open(path, "w", encoding="utf-8", newline="\n").write(text)


patch(CU, [
    (CU_KERNEL_ANCHOR, KERNEL + CU_KERNEL_ANCHOR, 1),
    (CU_LAUNCH_ANCHOR, LAUNCHER + CU_LAUNCH_ANCHOR, 1),
])
patch(HDR, [(HDR_ANCHOR, HDR_DECL + HDR_ANCHOR, 1)])
patch(CPP, [(CPP_ANCHOR, CPP_DISPATCH, 1)])
print("conv2px patched")
