#!/usr/bin/env python3
"""Replace the implicit int8 CUDA kernel with a 4-pixel-per-lane version.

Weight traffic drops 4x (one weight load feeds four dp4a accumulators), while
the per-output integer accumulation order (tap outer, c4 inner) is unchanged.
"""
import io

PATH = "/home/user/ncnn-cuda/src/layer/cuda/implicit_conv_cuda.cu"
START = "// 一个 warp = 一个输出像素 (oh,ow)；lane = 一个输出通道 co。"
END = "void implicit_pack_int8_c4_launch"

NEW_KERNEL = r'''// 一个 warp = 4 个相邻输出像素；lane = 一个输出通道 co。
// 每个 lane 用 4 个寄存器累加器：同一 tap/C4 的权重只读一次、复用 4 次，
// 权重带宽降到 1/4；输入 int 在 warp 内仍是 broadcast。
// 每个输出像素的累加顺序（tap 外层、c4 内层）与旧 kernel 完全一致，
// dp4a 是整数运算，因此结果逐位相同，精度不受影响。
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
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int M = OH * OW;
    const int m0 = (blockIdx.x * 8 + warp) * 4;
    const int co = blockIdx.y * 32 + lane;
    if (m0 >= M || co >= Cout) return;

    const int S = H * W;
    const int* qin = reinterpret_cast<const int*>(packed);
    const int* win = reinterpret_cast<const int*>(wimp);

    // 4 个像素的输入基址与输出偏移；valid 在 warp 内一致（m0 同一 warp 相同）。
    int base[4];
    int outoff[4];
    bool valid[4];
#pragma unroll
    for (int k = 0; k < 4; k++)
    {
        const int mk = m0 + k;
        valid[k] = mk < M;
        const int oh = valid[k] ? mk / OW : 0;
        const int ow = valid[k] ? mk - oh * OW : 0;
        base[k] = oh * sh * W + ow * sw;
        outoff[k] = oh * OW + ow;
    }

    int acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    int tap = 0;
    for (int i = 0; i < kh; i++)
    {
        const int rowoff = i * dh * W;
        for (int j = 0; j < kw; j++)
        {
            const int toff = rowoff + j * dw;
            const int* wrow = win + ((size_t)tap * C4) * Cout + co;
            for (int c4 = 0; c4 < C4; c4++)
            {
                const int w = wrow[(size_t)c4 * Cout];
                const int* q = qin + (size_t)c4 * S + toff;
                acc0 = dp4a_i8(q[base[0]], w, acc0);
                if (valid[1]) acc1 = dp4a_i8(q[base[1]], w, acc1);
                if (valid[2]) acc2 = dp4a_i8(q[base[2]], w, acc2);
                if (valid[3]) acc3 = dp4a_i8(q[base[3]], w, acc3);
            }
            tap++;
        }
    }

    const float ws = wscales[co];
    const float scale_in = (ws == 0.0f) ? 0.0f : 1.0f / (*bottom_scale * ws);
    const float b = bias ? bias[co] : 0.0f;
    const float ts = *top_scale;
    int accs[4] = {acc0, acc1, acc2, acc3};
#pragma unroll
    for (int k = 0; k < 4; k++)
    {
        if (!valid[k]) continue;
        float y = (float)accs[k] * scale_in + b;

        const size_t oidx = (size_t)co * out_cstep + outoff[k];
        if (requantize)
        {
            signed char q = float2int8_ncnn(y * ts);
            if (activation_type == 1 && q < 0) q = 0;
            reinterpret_cast<signed char*>(output)[oidx] = q;
        }
        else
        {
            if (activation_type == 1 && y < 0.0f) y = 0.0f;
            reinterpret_cast<float*>(output)[oidx] = y;
        }
    }
}

'''

OLD_LAUNCH = "    const dim3 grid((M + 7) / 8, (Cout + 31) / 32);"
NEW_LAUNCH = "    // 8 warps/block x 4 px/warp = 32 px/block\n    const dim3 grid((M + 31) / 32, (Cout + 31) / 32);"


def main():
    with io.open(PATH, "r", encoding="utf-8") as f:
        text = f.read()
    a = text.index(START)
    b = text.index(END, a)
    text = text[:a] + NEW_KERNEL + text[b:]
    if OLD_LAUNCH not in text:
        raise SystemExit("launch config marker not found")
    text = text.replace(OLD_LAUNCH, NEW_LAUNCH)
    with io.open(PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print("patched", PATH)


if __name__ == "__main__":
    main()
