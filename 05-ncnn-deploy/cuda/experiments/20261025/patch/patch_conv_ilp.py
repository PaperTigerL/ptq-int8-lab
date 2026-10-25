#!/usr/bin/env python3
"""H-086: ILP variant of the implicit conv (2 independent dp4a chains).

Flat mapping did not add parallelism (it only repacked the same warp count).
Roofline says the kernel is latency-bound: the acc chain is 9*C4 dependent
dp4a with a weight load per step, and small layers have only 16-32 warps total.
Two accumulators over even/odd c4 halve the dependency chain and let the next
load issue while the previous dp4a is in flight. int addition is associative,
so acc0+acc1 is bitwise identical to the single-chain result.

Env: NCNN_CUDA_CONV_ILP=1.
"""
import io
import sys

CU = "/home/user/ncnn-cuda/src/layer/cuda/implicit_conv_cuda.cu"
HDR = "/home/user/ncnn-cuda/src/layer/cuda/implicit_conv_cuda.h"
CPP = "/home/user/ncnn-cuda/src/layer/cuda/convolution_cuda.cpp"

KERNEL = r"""
// 两个独立累加链（偶数/奇数 c4），缩短依赖链并让下一次 load 与上一次 dp4a 重叠。
// acc0 + acc1 是整数加法，结合律保证与原 kernel 逐位一致。
__global__ void implicit_conv_int8_kernel_ilp(const signed char* __restrict__ packed,
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
    const int m = blockIdx.x * 8 + warp;
    const int co = blockIdx.y * 32 + lane;
    if (m >= M || co >= Cout) return;

    const int oh = m / OW;
    const int ow = m - oh * OW;
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
            const int pix = ih * W + (ow * sw + j * dw);
            const int* qrow = qin + pix;
            const int* wrow = win + (size_t)tap * C4 * Cout + co;
            int c4 = 0;
            for (; c4 + 1 < C4; c4 += 2)
            {
                const int w0 = wrow[(size_t)c4 * Cout];
                const int a0 = qrow[(size_t)c4 * S];
                acc0 = dp4a_i8(a0, w0, acc0);
                const int w1 = wrow[(size_t)(c4 + 1) * Cout];
                const int a1 = qrow[(size_t)(c4 + 1) * S];
                acc1 = dp4a_i8(a1, w1, acc1);
            }
            if (c4 < C4)
            {
                const int w = wrow[(size_t)c4 * Cout];
                const int a = qrow[(size_t)c4 * S];
                acc0 = dp4a_i8(a, w, acc0);
            }
            tap++;
        }
    }

    int acc = acc0 + acc1;
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
void implicit_conv_int8_launch_ilp(const signed char* packed,
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
    const dim3 grid((M + 7) / 8, (Cout + 31) / 32);
    implicit_conv_int8_kernel_ilp<<<grid, 256>>>(
        packed, wimp, wscales, bias, bottom_scale, top_scale,
        C4, H, W, Cout, kh, kw, dh, dw, sh, sw, OH, OW,
        requantize, activation_type, activation_params, output, out_cstep);
}

"""

CU_ANCHOR = "// 一个 warp = 同一行的两个相邻输出像素"
HDR_ANCHOR = "} // namespace ncnn\n"
HDR_DECL = """// 双累加链（ILP）版本，数值逐位一致。
void implicit_conv_int8_launch_ilp(const signed char* packed,
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

OLD = """        static const bool conv_flat = []() {
            const char* e = std::getenv("NCNN_CUDA_CONV_FLAT");
            return e && e[0] == '1' && e[1] == '\\0';
        }();
"""
NEW = """        static const bool conv_ilp = []() {
            const char* e = std::getenv("NCNN_CUDA_CONV_ILP");
            return e && e[0] == '1' && e[1] == '\\0';
        }();
        if (conv_ilp)
        {
            implicit_conv_int8_launch_ilp(
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

        static const bool conv_flat = []() {
            const char* e = std::getenv("NCNN_CUDA_CONV_FLAT");
            return e && e[0] == '1' && e[1] == '\\0';
        }();
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
patch(CPP, [(OLD, NEW, 1)])
print("ilp patched")
