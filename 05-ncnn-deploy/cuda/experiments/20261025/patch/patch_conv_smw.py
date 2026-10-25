#!/usr/bin/env python3
"""H-090: per-block shared-memory weight staging for the implicit int8 conv.

Roofline (H-086) says the conv is latency/occupancy bound, not DRAM or dp4a
throughput bound: grid is (ceil(M/8), Cout/32) with 8 warps per block and every
warp in a block re-reads the *same* [k][lane] weight tile from global memory,
so the dependent weight load on the critical path happens 8 times per block.

Load the tile once per block into shared memory and let all warps in the block
consume it: weight global loads per block drop from active_warps*K rows to K
rows. The tile is staged in tap-aligned chunks sized to a fixed 8 KB shared
budget so per-block shared memory (and therefore occupancy) does not depend on
C. K order inside a chunk is unchanged (tap outer, c4 inner), so the per-pixel
dp4a accumulation sequence is bit-identical.

Blocks whose warps cannot share (M == 1, only warp 0 active) keep the original
kernel: staging would cost the same global traffic plus two barriers.

Env: NCNN_CUDA_CONV_SHMW=1 (default off).
"""
import io
import sys

CU = "/home/user/ncnn-cuda/src/layer/cuda/implicit_conv_cuda.cu"
HDR = "/home/user/ncnn-cuda/src/layer/cuda/implicit_conv_cuda.h"
CPP = "/home/user/ncnn-cuda/src/layer/cuda/convolution_cuda.cpp"

KERNEL = r"""
// 权重按块共享内存预取：块内最多 8 个 warp 共用同一 [k][lane] 权重分块。
// 原 kernel 每块的权重 global load 行数 = 活跃 warp 数 * K，这里压成 K 行；
// 分块按 tap 对齐、共享内存固定预算，K 内层顺序（tap 外、c4 内）不变 -> 逐位一致。
__global__ void implicit_conv_int8_kernel_smw(const signed char* __restrict__ packed,
                                              const signed char* __restrict__ wimp,
                                              const float* __restrict__ wscales,
                                              const float* __restrict__ bias,
                                              const float* __restrict__ bottom_scale,
                                              const float* __restrict__ top_scale,
                                              int C4, int H, int W,
                                              int Cout, int kh, int kw, int dh, int dw,
                                              int sh, int sw, int OH, int OW,
                                              int chunk_taps,
                                              bool requantize, int activation_type,
                                              const float* __restrict__ activation_params,
                                              void* __restrict__ output, int out_cstep)
{
    extern __shared__ int wsm[];   // [chunk_taps*C4][32]
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int M = OH * OW;
    const int m = blockIdx.x * 8 + warp;
    const int co_base = blockIdx.y * 32;
    const int co = co_base + lane;
    const int taps = kh * kw;
    const int* qin = reinterpret_cast<const int*>(packed);
    const int* win = reinterpret_cast<const int*>(wimp);
    // 注意：不能提前 return，装载与计算之间的 barrier 需要全块线程参与。
    const bool active = (m < M) && (co < Cout);
    const int stage_lanes = min(32, Cout - co_base);
    const int oh = active ? (m / OW) : 0;
    const int ow = active ? (m - oh * OW) : 0;
    const int S = H * W;

    int acc = 0;
    for (int t0 = 0; t0 < taps; t0 += chunk_taps)
    {
        const int nt = min(chunk_taps, taps - t0);
        const int nk = nt * C4;
        const int kbase = t0 * C4;
        // 协作装载：t = kl*32 + ln，同一 kl 的 32 个 lane 读连续的 32 个 int（128B 合并），
        // 写 wsm[t] 时 bank = lane -> 无冲突。
        for (int t = tid; t < nk * 32; t += blockDim.x)
        {
            const int kl = t >> 5;
            const int ln = t & 31;
            wsm[t] = (ln < stage_lanes) ? win[(size_t)(kbase + kl) * Cout + co_base + ln] : 0;
        }
        __syncthreads();
        if (active)
        {
            for (int tt = 0; tt < nt; tt++)
            {
                const int tap = t0 + tt;
                const int i = tap / kw;
                const int j = tap - i * kw;
                const int pix = (oh * sh + i * dh) * W + (ow * sw + j * dw);
                int koff = tt * C4 * 32;
#pragma unroll 4
                for (int c4 = 0; c4 < C4; c4++)
                {
                    const int a = qin[(size_t)c4 * S + pix];
                    const int w = wsm[koff + lane];
                    acc = dp4a_i8(a, w, acc);
                    koff += 32;
                }
            }
        }
        __syncthreads();
    }
    if (!active) return;

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
void implicit_conv_int8_launch_smw(const signed char* packed,
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
    const int taps = kh * kw;
    // 共享内存预算与 C 无关：chunk_taps = 预算 / (C4*32*4)，至少 1 个 tap。
    int chunk_taps = IMPLICIT_SMW_SHARED_BUDGET / (C4 * 32 * (int)sizeof(int));
    if (chunk_taps > taps) chunk_taps = taps;
    if (chunk_taps < 1) chunk_taps = 1;
    const size_t smem = (size_t)chunk_taps * C4 * 32 * sizeof(int);
    // 单个 tap 就超过动态共享内存默认上限的极端通道数：退回原 kernel，不能让 launch 静默失败。
    if (smem > 48 * 1024)
    {
        implicit_conv_int8_launch(packed, C, H, W, wimp, wscales, bias, bottom_scale, top_scale,
                                  Cout, kh, kw, dh, dw, sh, sw, OH, OW,
                                  requantize, activation_type, activation_params, output, out_cstep);
        return;
    }
    const dim3 grid((M + 7) / 8, (Cout + 31) / 32);
    implicit_conv_int8_kernel_smw<<<grid, 256, smem>>>(
        packed, wimp, wscales, bias, bottom_scale, top_scale,
        C4, H, W, Cout, kh, kw, dh, dw, sh, sw, OH, OW,
        chunk_taps, requantize, activation_type, activation_params, output, out_cstep);
}

"""

CONST = """
// 每块权重分块的共享内存预算（字节）。C4=16 -> 4 taps/块，C4=32 -> 2，C4>=64 -> 1。
#define IMPLICIT_SMW_SHARED_BUDGET 8192

"""

CU_ANCHOR = "// 一个 warp = 同一行的两个相邻输出像素"
HDR_ANCHOR = "} // namespace ncnn\n"
HDR_DECL = """// 权重按块共享内存预取（块内 8 warp 复用同一权重分块），数值逐位一致。
void implicit_conv_int8_launch_smw(const signed char* packed,
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
NEW_DISPATCH = """        static const bool conv_smw = []() {
            const char* e = std::getenv("NCNN_CUDA_CONV_SHMW");
            return e && e[0] == '1' && e[1] == '\\0';
        }();
        // M==1 时块内只有 1 个 warp，共享内存换不到任何复用，只多两次 barrier。
        if (conv_smw && OH_ * OW_ >= 2)
        {
            implicit_conv_int8_launch_smw(
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


patch(CU, [("#include <cstdint>", "#include <cstdint>\n" + CONST, 1),
           (CU_ANCHOR, KERNEL + LAUNCHER + CU_ANCHOR, 1)])
patch(HDR, [(HDR_ANCHOR, HDR_DECL + HDR_ANCHOR, 1)])
patch(CPP, [(OLD_DISPATCH, NEW_DISPATCH, 1)])
print("smw patched")
