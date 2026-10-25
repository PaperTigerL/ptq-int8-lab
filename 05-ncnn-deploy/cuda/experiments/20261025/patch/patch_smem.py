#!/usr/bin/env python3
"""Stage implicit-conv weights per tap in shared memory.

Old kernel: each warp (one output pixel) loads the same 32-channel weight row
from global memory.  8 warps per block therefore read the same weights 8 times,
and each dp4a sits behind a global load -> memory-latency bound.

New kernel: the 256 threads of a block cooperatively load the current tap's
[C4][32] weight tile into shared memory once, then all 8 warps consume it.
Global weight traffic per block drops ~8x and the dp4a inner loop reads smem
instead of global.  The mapping (warp = pixel, lane = output channel) and the
accumulation order (tap outer, C4 inner) are unchanged, so int32 results are
bit-identical.
"""
import io

PATH = "/home/user/ncnn-cuda/src/layer/cuda/implicit_conv_cuda.cu"
START = "// 一个 warp = 一个输出像素 (oh,ow)；lane = 一个输出通道 co。"
END = "void implicit_pack_int8_c4_launch"

NEW_KERNEL = r'''// 一个 warp = 一个输出像素 (oh,ow)；lane = 一个输出通道 co。
// 块的 8 个 warp 共享同一个 32 通道权重切片：每个 tap 开始前由 256 个线程
// 协作把该 tap 的 [C4][32] 权重装进 shared memory，之后 8 个 warp 复用。
// 全局权重流量降到约 1/8，dp4a 内循环读 smem 而不是 global。
// 映射（warp=像素、lane=输出通道）与累加顺序（tap 外层、C4 内层）不变，
// 因此 int32 结果逐位相同。
__global__ void implicit_conv_int8_kernel(const signed char* __restrict__ packed,
                                          const signed char* __restrict__ wimp,
                                          const float* __restrict__ wscales,
                                          const float* __restrict__ bias,
                                          const float* __restrict__ bottom_scale,
                                          const float* __restrict__ top_scale,
                                          int C4, int H, int W,
                                          int Cout, int kh, int kw, int dh, int dw,
                                          int sh, int sw, int OH, int OW,
                                          bool requantize, int activation_type,
                                          void* __restrict__ output, int out_cstep)
{
    extern __shared__ int swt[]; // 一个 tap 的 [C4][32] 权重

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int tid = threadIdx.x;
    const int M = OH * OW;
    const int m = blockIdx.x * 8 + warp;
    const int co_base = blockIdx.y * 32;
    const int co = co_base + lane;
    const bool active = (m < M) && (co < Cout);

    const int S = H * W;
    const int* qin = reinterpret_cast<const int*>(packed);
    const int* win = reinterpret_cast<const int*>(wimp);

    int acc = 0;
    int pix = 0;
    if (m < M)
    {
        const int oh = m / OW;
        const int ow = m - oh * OW;
        pix = oh * sh * W + ow * sw;
    }

    const int tile = C4 * 32;
    int tap = 0;
    for (int i = 0; i < kh; i++)
    {
        const int rowoff = i * dh * W;
        for (int j = 0; j < kw; j++)
        {
            // 上一 tap 的 smem 读取全部结束后再覆盖；本轮装填完再计算。
            __syncthreads();
            const int* wsrc = win + ((size_t)tap * C4) * Cout + co_base;
            for (int idx = tid; idx < tile; idx += blockDim.x)
            {
                const int c4 = idx >> 5;
                const int cc = idx & 31;
                int v = 0;
                if (co_base + cc < Cout) v = wsrc[(size_t)c4 * Cout + cc];
                swt[idx] = v;
            }
            __syncthreads();

            if (active)
            {
                const int p = pix + rowoff + j * dw;
                for (int c4 = 0; c4 < C4; c4++)
                    acc = dp4a_i8(qin[(size_t)c4 * S + p], swt[(c4 << 5) + lane], acc);
            }
            tap++;
        }
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
        reinterpret_cast<signed char*>(output)[oidx] = q;
    }
    else
    {
        if (activation_type == 1 && y < 0.0f) y = 0.0f;
        reinterpret_cast<float*>(output)[oidx] = y;
    }
}

'''

OLD_LAUNCH = """    const dim3 grid((M + 7) / 8, (Cout + 31) / 32);
    implicit_conv_int8_kernel<<<grid, 256>>>("""
NEW_LAUNCH = """    const dim3 grid((M + 7) / 8, (Cout + 31) / 32);
    const size_t smem_bytes = (size_t)C4 * 32 * sizeof(int);
    implicit_conv_int8_kernel<<<grid, 256, smem_bytes>>>("""


def main():
    with io.open(PATH, "r", encoding="utf-8") as f:
        text = f.read()
    a = text.index(START)
    b = text.index(END, a)
    text = text[:a] + NEW_KERNEL + text[b:]
    if OLD_LAUNCH not in text:
        raise SystemExit("launch marker not found")
    text = text.replace(OLD_LAUNCH, NEW_LAUNCH)
    with io.open(PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print("patched", PATH)


if __name__ == "__main__":
    main()
