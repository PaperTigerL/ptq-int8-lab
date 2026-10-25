#!/usr/bin/env python3
"""H-086: flat warp mapping for the implicit int8 conv.

Roofline says the kernel is not DRAM or dp4a-throughput bound; the loss is idle
warps when M=OH*OW is small: grid is (ceil(M/8), Cout/32) with 8 warps per
block, so M=1 runs 1/8 of every block and M=4 runs 4/8. Flatten the work to
gwid = m + M*co_group so every block's 8 warps are busy.

Per-pixel dp4a order is unchanged -> bitwise identical.
Env: NCNN_CUDA_CONV_FLAT=1.
"""
import io
import sys

CU = "/home/user/ncnn-cuda/src/layer/cuda/implicit_conv_cuda.cu"
HDR = "/home/user/ncnn-cuda/src/layer/cuda/implicit_conv_cuda.h"
CPP = "/home/user/ncnn-cuda/src/layer/cuda/convolution_cuda.cpp"

KERNEL = r"""
// flat 映射：warp 任务 = M * ceil(Cout/32)，gwid -> (m, co_group)。
// 小特征图（M=1/4）时不再有 7/8 或 4/8 的 warp 空转。
__global__ void implicit_conv_int8_kernel_flat(const signed char* __restrict__ packed,
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
    const int M = OH * OW;
    const int cog_count = (Cout + 31) / 32;
    const int gwid = blockIdx.x * 8 + warp;
    if (gwid >= M * cog_count) return;
    const int m = gwid % M;
    const int cog = gwid / M;
    const int co = cog * 32 + lane;
    if (co >= Cout) return;

    const int oh = m / OW;
    const int ow = m - oh * OW;
    const int S = H * W;
    const int* qin = reinterpret_cast<const int*>(packed);
    const int* win = reinterpret_cast<const int*>(wimp);

    int acc = 0;
    int tap = 0;
    for (int i = 0; i < kh; i++)
    {
        const int ih = oh * sh + i * dh;
        for (int j = 0; j < kw; j++)
        {
            const int pix = ih * W + (ow * sw + j * dw);
            for (int c4 = 0; c4 < C4; c4++)
            {
                const int a = qin[(size_t)c4 * S + pix];
                const int w = win[((size_t)tap * C4 + c4) * Cout + co];
                acc = dp4a_i8(a, w, acc);
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
void implicit_conv_int8_launch_flat(const signed char* packed,
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
    const int M = OH * OW;
    const int cog_count = (Cout + 31) / 32;
    const int warps = M * cog_count;
    const dim3 grid((warps + 7) / 8);
    implicit_conv_int8_kernel_flat<<<grid, 256>>>(
        packed, wimp, wscales, bias, bottom_scale, top_scale,
        C4, H, W, Cout, kh, kw, dh, dw, sh, sw, OH, OW,
        requantize, activation_type, activation_params, output, out_cstep);
}

"""

CU_ANCHOR = "// 一个 warp = 同一行的两个相邻输出像素"
HDR_ANCHOR = "} // namespace ncnn\n"
HDR_DECL = """// flat warp 映射（M*ceil(Cout/32) 个 warp 任务），数值逐位一致。
void implicit_conv_int8_launch_flat(const signed char* packed,
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

OLD_DISPATCH = """        implicit_conv_int8_launch(
            packed, C, IH, IW,
"""
NEW_DISPATCH = """        static const bool conv_flat = []() {
            const char* e = std::getenv("NCNN_CUDA_CONV_FLAT");
            return e && e[0] == '1' && e[1] == '\\0';
        }();
        if (conv_flat)
        {
            implicit_conv_int8_launch_flat(
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
                     (path, text.count(old), count, old[:70]))
        text = text.replace(old, new)
    io.open(path, "w", encoding="utf-8", newline="\n").write(text)


patch(CU, [(CU_ANCHOR, KERNEL + LAUNCHER + CU_ANCHOR, 1)])
patch(HDR, [(HDR_ANCHOR, HDR_DECL + HDR_ANCHOR, 1)])
patch(CPP, [(OLD_DISPATCH, NEW_DISPATCH, 1)])
print("flat warp patched")
