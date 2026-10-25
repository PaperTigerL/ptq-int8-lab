// GPU int8 推理引擎的 CUDA kernel。
//
// ⭐ 设计依据（分段计时的实测结果，不是猜的）：
//   batch=512, layer1.0.conv1 (C64 32x32 k3) 单层拆开是
//     quantize 10.7 ms | im2col 7.9 ms | gemm 6.2 ms  = 24.9 ms
//   GEMM 只占 25%，quantize+im2col 占 75% —— **瓶颈是内存搬运，不是算力**。
//   所以优化方向不是"换个更快的 GEMM"，而是**把搬运融合掉**。
//
// 三个 kernel，对应三次融合：
//   1. reduce_minmax_kernel   —— 校准/动态量化算激活范围
//   2. quant_im2col_kernel     —— ⭐量化 + im2col 一次完成，中间不落 fp32
//   3. requant_kernel          —— int32 → per-channel 反量化 + bias + ReLU
//
// 为什么必须融合（不是"能优化，是不融合就白做"）：
//   不融合的搬运量 = 写 int8 激活(N*C*H*W) + 读它 + 写 im2col(N*L*K)
//                   + 读 im2col + 读 fp32 源
//   im2col 的输出比输入大 kh*kw 倍（3x3 就是 9 倍），这一步单独就
//   烧掉大部分带宽。融合后只写一次 im2col，读一次 fp32。
//
// ⚠️ 两个必须记住的实现约束：
//   1. im2col 输出按 **L-major** 排（(n,oh,ow) 在外，k 在内），
//      因为 GEMM 的 M 维必须能把 batch 维摊进去 —— 这正是 ncnn 那轮
//      batch conv 找到的同一个入口（GEMM 的 N 是空间位置，不是通道）。
//   2. K 维要对齐到 8 的倍数（int8 tensor core 的硬要求）。
//      不足处补 0 —— GEMM 里乘 0，不改结果。

#include <cuda_runtime.h>
#include <stdint.h>

#include "implicit_conv.cuh"
#include "stream_ctx.cuh"

namespace ptqint8 {

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _e = (expr);                                                \
        if (_e != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error %s at %s:%d -> %s\n",                   \
                    #expr, __FILE__, __LINE__, cudaGetErrorString(_e));         \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------- 1) min/max 归约

// grid-stride 归约：每线程先扫自己的 strided 元素，再做 warp/block 两级归约。
// 用 atomicMax/Min 处理负数比较的技巧在下面 warp_reduce 里。
__device__ __forceinline__ float warp_reduce_max(float v)
{
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
    return v;
}

__device__ __forceinline__ float warp_reduce_min(float v)
{
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fminf(v, __shfl_xor_sync(0xffffffff, v, off));
    return v;
}

__global__ void reduce_minmax_kernel(const float* __restrict__ x, size_t n,
                                     float* __restrict__ out /* [0]=min [1]=max */)
{
    float lo = 3.4e38f;
    float hi = -3.4e38f;

    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         i < n; i += (size_t)gridDim.x * blockDim.x)
    {
        const float v = x[i];
        lo = fminf(lo, v);
        hi = fmaxf(hi, v);
    }

    lo = warp_reduce_min(lo);
    hi = warp_reduce_max(hi);

    __shared__ float s_lo[32];
    __shared__ float s_hi[32];
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    if (lane == 0) { s_lo[wid] = lo; s_hi[wid] = hi; }
    __syncthreads();

    if (wid == 0)
    {
        const int nwarp = blockDim.x >> 5;
        float l = s_lo[lane];
        float h = s_hi[lane];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
        {
            l = fminf(l, __shfl_xor_sync(0xffffffff, l, off));
            h = fmaxf(h, __shfl_xor_sync(0xffffffff, h, off));
        }
        if (lane == 0) { out[0] = l; out[1] = h; }
        (void)nwarp;
    }
}

// ---------------------------------------------------------------- 2) 量化 + im2col 融合

// x    : (N, C, H, W) fp32
// out  : (N*OH*OW, Kpad) int8，L-major
// scale: 激活 scale，zp: 激活 zero_point
// Kpad : 对齐到 8 的倍数后的 K 维长度；真实 K = C*kh*kw
//
// 每个线程负责输出矩阵的一个元素 (row, k)：
//   row -> (n, oh, ow)，k -> (c, i, j)
// 写出 q = clamp(round(x/scale) + zp, 0, qmax) - zp
//
// ⚠️ 越界读的处理：卷积窗口越过图像边界时，读到的是 0 填充。
//   填充值必须先在 fp32 域补 0 再量化（得到码字 -zp），
//   而不是直接写 int8 的 0 —— int8 的 0 对应真实值 zp*scale ≠ 0。
//   这是这个 kernel 最容易写错的一处。
// ⚠️ 舍入语义必须与 CPU 参考（ptq_quant.h::torch_round）和 PyTorch 一样：
//   **half-away-from-zero**，不是 C 的 round()（银行家舍入），
//   也不是 CUDA 的 rintf()（half-to-even）。
//   实测用 rintf 时，单层 max|diff| = 13.67、63.5% 元素超 1e-3；
//   改成 torch_round 语义后归零。
//   写法：round-half-away = sign(x) * floor(|x| + 0.5)。
//   不能写成 rintf(x + 0.5*signf(x)) —— 那样 x=0.5 会算成 round(1.0)=1 但
//   x=-0.5 会算成 round(0.0)=0，符号分支在 x=0 附近不稳定。
__device__ __forceinline__ float torch_round_cuda(float v)
{
    return (v >= 0.0f) ? floorf(v + 0.5f) : ceilf(v - 0.5f);
}

// c_base：分组卷积时该组在输入通道维的起点。C 仍是**总**输入通道数
//         （x 的 sample stride 是 C*H*W）；im2col 的 k 只覆盖组内 cin*kh*kw。
__global__ void quant_im2col_kernel(const float* __restrict__ x,
                                    int8_t* __restrict__ out,
                                    float scale, int zp, int qmax,
                                    int N, int C, int H, int W,
                                    int kh, int kw, int sh, int sw, int ph, int pw,
                                    int OH, int OW, int K, int Kpad, int c_base)
{
    const long long total = (long long)N * OH * OW * Kpad;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total; idx += (long long)gridDim.x * blockDim.x)
    {
        const int k = (int)(idx % Kpad);
        if (k >= K)
        {
            // K 对齐补零：乘 0，不影响 GEMM 结果。
            out[idx] = 0;
            continue;
        }

        const long long row = idx / Kpad;
        const int ow = (int)(row % OW);
        const long long t = row / OW;
        const int oh = (int)(t % OH);
        const int n = (int)(t / OH);

        // k -> (c, i, j)：必须与权重展平顺序一致
        const int j = k % kw;
        const int t2 = k / kw;
        const int i = t2 % kh;
        const int c = t2 / kh;

        const int ih = oh * sh + i - ph;
        const int iw = ow * sw + j - pw;

        // ⚠️ padding 区必须写**码字 0**，而不是「把 0.0 量化一下」。
        //   0.0 量化后是 (round(0/s) + zp) - zp = zp，量程外的 zp 可能非 0，
        //   会让 padding 位置贡献 zp·q_w 而不是 0 —— 边界元素全错，
        //   实测表现为单层 max|diff| ≈ 13.7、63.5% 元素超差。
        //   卷积窗口越界时这一项的贡献必须是 0，所以直接写码字 0。
        if (ih < 0 || ih >= H || iw < 0 || iw >= W)
        {
            out[idx] = 0;
            continue;
        }
        const float v = x[(((long long)n * C + (c + c_base)) * H + ih) * W + iw];

        // ⭐ 对称码空间：clamp 到 [-(qmax+1), qmax]，下界比上界多一格。
        //   这与 ptq-lab/quant.py::quantize_per_tensor_sym 完全一致 ——
        //   负侧多一个码字，因为 0 必须落在网格上（round(0) 恒为 0）。
        //   写成 [-qmax, qmax] 会少一个负码字，W4/W8 的数字对不上。
        //   zp 恒为 0（非对称方案在 _int_mm 上不可行，见 ptq_quant.h）。
        float q = torch_round_cuda(v / scale);
        q = fminf(fmaxf(q, -(float)(qmax + 1)), (float)qmax);
        out[idx] = (int8_t)(int)q;
    }
}

// ⚠️ 上面的 torch_round_cuda 是 **half-away-from-zero**，与 torch.round 一致。
// 换成 rintf（half-to-even）会导致单层 max|diff| ≈ 13.7 / 63.5% 元素超差 ——
// 这个差异不改变最终分类（实测同精度），但会让「单层逐位对齐」这种验证失效。
// 要做逐位验证就必须统一到 half-away。

// ---------------------------------------------------------------- 2c) vectorized im2col
//
// 旧 fast kernel 每个 block 只写一个输出位置（576B），一个线程写 1B；
// 32k 个小 block + 标量 1B store 把 im2col 压在 ~17GB/s。
// 新版：一个 warp 负责一行（一个输出位置），lane 循环 Kpad/4 个 int，
// 每次把 4 个量化值拼成一个 32-bit store。行内 store 完全合并，
// 每 block 8 行（8 warps），block 数降到 1/8。
__device__ __forceinline__ int quant_one_i8(float v, float scale, int qmax)
{
    float q = torch_round_cuda(v / scale);
    q = fminf(fmaxf(q, -(float)(qmax + 1)), (float)qmax);
    return (int)(int8_t)(int)q;
}

template <int KH, int KW>
__global__ void quant_im2col_vec_kernel(
    const float* __restrict__ x, int8_t* __restrict__ out,
    float scale, int qmax,
    int C, int H, int W, int sh, int sw, int ph, int pw,
    int OH, int OW, int K, int Kpad, int c_base, int M)
{
    constexpr int WARPS = 8;
    constexpr int RPW = 2;             // 每个 warp 连续处理 4 行
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int Lpos = OH * OW;
    int m = blockIdx.x * (WARPS * RPW) + warp * RPW;

    for (int r = 0; r < RPW; r++, m++)
    {
        if (m >= M) break;
        const int n = m / Lpos;
        const int pos = m - n * Lpos;
        const int oh = pos / OW;
        const int ow = pos - oh * OW;
        const int ih0 = oh * sh - ph;
        const int iw0 = ow * sw - pw;
        int8_t* orow = out + (size_t)m * Kpad;

        for (int t = lane; t < (Kpad >> 2); t += 32)
        {
            int packed = 0;
#pragma unroll
            for (int l = 0; l < 4; l++)
            {
                const int k = (t << 2) + l;
                int q = 0;
                if (k < K)
                {
                    const int c = k / (KH * KW);
                    const int tap = k - c * (KH * KW);
                    const int i = tap / KW;
                    const int j = tap - i * KW;
                    const int ih = ih0 + i;
                    const int iw = iw0 + j;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                    {
                        const float v = x[((size_t)n * C + c_base + c) * H * W + ih * W + iw];
                        q = quant_one_i8(v, scale, qmax);
                    }
                }
                packed |= (q & 0xff) << (l << 3);
            }
            reinterpret_cast<int*>(orow)[t] = packed;
        }
    }
}

__global__ void quant_im2col_vec_generic(
    const float* __restrict__ x, int8_t* __restrict__ out,
    float scale, int qmax, int C, int H, int W,
    int kh, int kw, int sh, int sw, int ph, int pw,
    int OH, int OW, int K, int Kpad, int c_base, int M)
{
    constexpr int WARPS = 8;
    constexpr int RPW = 2;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int Lpos = OH * OW;
    int m = blockIdx.x * (WARPS * RPW) + warp * RPW;

    for (int r = 0; r < RPW; r++, m++)
    {
        if (m >= M) break;
        const int n = m / Lpos;
        const int pos = m - n * Lpos;
        const int oh = pos / OW;
        const int ow = pos - oh * OW;
        const int ih0 = oh * sh - ph;
        const int iw0 = ow * sw - pw;
        int8_t* orow = out + (size_t)m * Kpad;

        for (int t = lane; t < (Kpad >> 2); t += 32)
        {
            int packed = 0;
#pragma unroll
            for (int l = 0; l < 4; l++)
            {
                const int k = (t << 2) + l;
                int q = 0;
                if (k < K)
                {
                    const int j = k % kw;
                    const int t2 = k / kw;
                    const int i = t2 % kh;
                    const int c = t2 / kh;
                    const int ih = ih0 + i;
                    const int iw = iw0 + j;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                    {
                        const float v = x[((size_t)n * C + c_base + c) * H * W + ih * W + iw];
                        q = quant_one_i8(v, scale, qmax);
                    }
                }
                packed |= (q & 0xff) << (l << 3);
            }
            reinterpret_cast<int*>(orow)[t] = packed;
        }
    }
}

// tiled 版：block 覆盖 TILE 个输出位置，把涉及的整幅输入图量化进 shared 一次，
// 之后 im2col 的 kh*kw 个 tap 全部从 shared 读。相比 fast 版把全局读/量化从
// 每个输出元素一次降到每「块覆盖的输入像素」一次。
//
// ⭐ TILE 从 16 提到 64 是实测的关键：layer1 的 Lpos=64，TILE=16 时一幅图被 4 个
// block 各读/量化一遍（4x 读放大），im2col 停在 ~19 GB/s；TILE=64 时一幅图一个
// block，读放大 1x。shared 仍是 imgs*C*S 字节（l1 = 4KB，最大的 2x2 层 16KB），
// 放不下时 launcher 自动退回 TILE=16。
template <int KH, int KW, int TILE>
__global__ void quant_im2col_tiled_kernel(
    const float* __restrict__ x, int8_t* __restrict__ out,
    float scale, int qmax,
    int C, int H, int W, int sh, int sw, int ph, int pw,
    int OH, int OW, int K, int Kpad, int c_base, int M, int imgs_max)
{
    extern __shared__ int8_t smem[];
    constexpr int WARPS = 8;
    constexpr int RPW = TILE / WARPS;   // 每 warp 写多少行
    const int S = H * W;
    const int Lpos = OH * OW;
    const int m0 = blockIdx.x * TILE;
    if (m0 >= M) return;
    int mlast = m0 + TILE - 1;
    if (mlast >= M) mlast = M - 1;
    const int n0 = m0 / Lpos;
    const int n1 = mlast / Lpos;
    const int imgs = n1 - n0 + 1;

    // 1) 整幅输入图 -> shared int8 [img][c][S]
    const int total = imgs * C * S;
    for (int t = threadIdx.x; t < total; t += blockDim.x)
    {
        const int img = t / (C * S);
        const int rem = t - img * (C * S);
        const int c = rem / S;
        const int sp = rem - c * S;
        const float v = x[((size_t)(n0 + img) * C + c_base + c) * S + sp];
        smem[t] = (int8_t)quant_one_i8(v, scale, qmax);
    }
    __syncthreads();

    // 2) 每个 warp 写 RPW 行 im2col
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    int m = m0 + warp * RPW;
    for (int r = 0; r < RPW; r++, m++)
    {
        if (m >= M) break;
        const int n = m / Lpos;
        const int pos = m - n * Lpos;
        const int oh = pos / OW;
        const int ow = pos - oh * OW;
        const int ih0 = oh * sh - ph;
        const int iw0 = ow * sw - pw;
        const int8_t* imgbase = smem + (n - n0) * C * S;
        int8_t* orow = out + (size_t)m * Kpad;
        for (int t = lane; t < (Kpad >> 2); t += 32)
        {
            int packed = 0;
#pragma unroll
            for (int l = 0; l < 4; l++)
            {
                const int k = (t << 2) + l;
                int q = 0;
                if (k < K)
                {
                    const int c = k / (KH * KW);
                    const int tap = k - c * (KH * KW);
                    const int i = tap / KW;
                    const int j = tap - i * KW;
                    const int ih = ih0 + i;
                    const int iw = iw0 + j;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                        q = (int)imgbase[c * S + ih * W + iw];
                }
                packed |= (q & 0xff) << (l << 3);
            }
            reinterpret_cast<int*>(orow)[t] = packed;
        }
    }
}

// tap-major im2col：K 维顺序改成 (tap, c4)（与 implicit GEMM 的 DP4A 布局一致）。
// shared 按 NHWC [img][S][Cpad] 存，连续 4 个输入通道是一个 int；
// 生成 im2col 时一次 32-bit load + 一次 32-bit store，不再逐字节读。
// 权重侧用 repack_weight_tap 做同样的 K 置换；整数和与顺序无关，
// 所以 int32 acc 与旧路径逐元素相同。
template <int KH, int KW>
__global__ void quant_im2col_tap_kernel(
    const float* __restrict__ x, int8_t* __restrict__ out,
    float scale, int qmax, int C, int H, int W,
    int sh, int sw, int ph, int pw, int OH, int OW,
    int Cpad, int Kt, int Kpad, int c_base, int M, int imgs_max)
{
    extern __shared__ int8_t smem[];
    constexpr int TILE = 40;
    constexpr int RPW = 5;
    constexpr int WARPS = 8;
    const int S = H * W;
    const int Lpos = OH * OW;
    const int m0 = blockIdx.x * TILE;
    if (m0 >= M) return;
    int mlast = m0 + TILE - 1;
    if (mlast >= M) mlast = M - 1;
    const int n0 = m0 / Lpos;
    const int n1 = mlast / Lpos;
    const int imgs = n1 - n0 + 1;

    // 1) NHWC 量化 tile
    const int total = imgs * S * Cpad;
    for (int t = threadIdx.x; t < total; t += blockDim.x)
    {
        const int img = t / (S * Cpad);
        const int rem = t - img * (S * Cpad);
        const int sp = rem / Cpad;
        const int c = rem - sp * Cpad;
        int q = 0;
        if (c < C)
        {
            const float v = x[((size_t)(n0 + img) * C + c_base + c) * S + sp];
            q = quant_one_i8(v, scale, qmax);
        }
        smem[t] = (int8_t)q;
    }
    __syncthreads();

    // 2) 每个 warp 写 RPW 行，每线程一次处理一个 4-channel word
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    int m = m0 + warp * RPW;
    for (int r = 0; r < RPW; r++, m++)
    {
        if (m >= M) break;
        const int n = m / Lpos;
        const int pos = m - n * Lpos;
        const int oh = pos / OW;
        const int ow = pos - oh * OW;
        const int ih0 = oh * sh - ph;
        const int iw0 = ow * sw - pw;
        const int8_t* imgbase = smem + (n - n0) * S * Cpad;
        int8_t* orow = out + (size_t)m * Kpad;
        for (int t = lane; t < (Kpad >> 2); t += 32)
        {
            int packed = 0;
            const int k = t << 2;
            if (k < Kt)
            {
                const int tap = k / Cpad;
                const int c = k - tap * Cpad;
                const int i = tap / KW;
                const int j = tap - i * KW;
                const int ih = ih0 + i;
                const int iw = iw0 + j;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                    packed = *reinterpret_cast<const int*>(
                        imgbase + ((ih * W + iw) * Cpad + c));
            }
            reinterpret_cast<int*>(orow)[t] = packed;
        }
    }
}

void quant_im2col_tap_launch(const float* x, int8_t* out, float scale, int qmax,
                             int N, int C, int H, int W,
                             int kh, int kw, int sh, int sw, int ph, int pw,
                             int OH, int OW, int Kpad, int c_base)
{
    const long long M = (long long)N * OH * OW;
    const int Cpad = (C + 3) & ~3;
    const int Kt = kh * kw * Cpad;
    const int Lpos = OH * OW;
    const int TILE = 40;
    const int imgs_max = (TILE + Lpos - 1) / Lpos;
    const int shmem = imgs_max * H * W * Cpad;
    const dim3 grid((unsigned)((M + TILE - 1) / TILE));
    if (kh == 3 && kw == 3)
        quant_im2col_tap_kernel<3, 3><<<grid, 256, shmem, ptqint8::g_stream>>>(
            x, out, scale, qmax, C, H, W, sh, sw, ph, pw, OH, OW,
            Cpad, Kt, Kpad, c_base, (int)M, imgs_max);
    else if (kh == 1 && kw == 1)
        quant_im2col_tap_kernel<1, 1><<<grid, 256, shmem, ptqint8::g_stream>>>(
            x, out, scale, qmax, C, H, W, sh, sw, ph, pw, OH, OW,
            Cpad, Kt, Kpad, c_base, (int)M, imgs_max);
    else if (kh == 7 && kw == 7)
        quant_im2col_tap_kernel<7, 7><<<grid, 256, shmem, ptqint8::g_stream>>>(
            x, out, scale, qmax, C, H, W, sh, sw, ph, pw, OH, OW,
            Cpad, Kt, Kpad, c_base, (int)M, imgs_max);
    else
        quant_im2col_vec_generic<<<grid, 256, 0, ptqint8::g_stream>>>(
            x, out, scale, qmax, C, H, W, kh, kw, sh, sw, ph, pw,
            OH, OW, Kt, Kpad, c_base, (int)M);
}

void quant_im2col_tiled_launch(const float* x, int8_t* out, float scale, int qmax,
                               int N, int C, int H, int W,
                               int kh, int kw, int sh, int sw, int ph, int pw,
                               int OH, int OW, int K, int Kpad, int c_base)
{
    const long long M = (long long)N * OH * OW;
    const int Lpos = OH * OW;
    // TILE=64 的 shared = imgs*C*S 字节；放不下（深层大通道 + 小空间）退回 16。
    const int imgs64 = (64 + Lpos - 1) / Lpos;
    const int shmem64 = imgs64 * C * H * W;
    const bool wide = shmem64 <= 32 * 1024;
    const int TILE = wide ? 64 : 16;
    const int imgs = wide ? imgs64 : (16 + Lpos - 1) / Lpos;
    const int shmem = wide ? shmem64 : imgs * C * H * W;
    const dim3 grid((unsigned)((M + TILE - 1) / TILE));
    if (kh == 3 && kw == 3)
    {
        if (wide)
            quant_im2col_tiled_kernel<3, 3, 64><<<grid, 256, shmem, ptqint8::g_stream>>>(
                x, out, scale, qmax, C, H, W, sh, sw, ph, pw, OH, OW, K, Kpad, c_base, (int)M, imgs);
        else
            quant_im2col_tiled_kernel<3, 3, 16><<<grid, 256, shmem, ptqint8::g_stream>>>(
                x, out, scale, qmax, C, H, W, sh, sw, ph, pw, OH, OW, K, Kpad, c_base, (int)M, imgs);
    }
    else if (kh == 1 && kw == 1)
    {
        if (wide)
            quant_im2col_tiled_kernel<1, 1, 64><<<grid, 256, shmem, ptqint8::g_stream>>>(
                x, out, scale, qmax, C, H, W, sh, sw, ph, pw, OH, OW, K, Kpad, c_base, (int)M, imgs);
        else
            quant_im2col_tiled_kernel<1, 1, 16><<<grid, 256, shmem, ptqint8::g_stream>>>(
                x, out, scale, qmax, C, H, W, sh, sw, ph, pw, OH, OW, K, Kpad, c_base, (int)M, imgs);
    }
    else if (kh == 7 && kw == 7)
    {
        if (wide)
            quant_im2col_tiled_kernel<7, 7, 64><<<grid, 256, shmem, ptqint8::g_stream>>>(
                x, out, scale, qmax, C, H, W, sh, sw, ph, pw, OH, OW, K, Kpad, c_base, (int)M, imgs);
        else
            quant_im2col_tiled_kernel<7, 7, 16><<<grid, 256, shmem, ptqint8::g_stream>>>(
                x, out, scale, qmax, C, H, W, sh, sw, ph, pw, OH, OW, K, Kpad, c_base, (int)M, imgs);
    }
    else
        quant_im2col_vec_generic<<<grid, 256, 0, ptqint8::g_stream>>>(
            x, out, scale, qmax, C, H, W, kh, kw, sh, sw, ph, pw,
            OH, OW, K, Kpad, c_base, (int)M);
}

__global__ void quant_im2col_kmajor_kernel(const float* __restrict__ x,
                                           int8_t* __restrict__ out,
                                           float scale, int qmax,
                                           int C, int H, int W,
                                           int kh, int kw, int sh, int sw,
                                           int ph, int pw, int OH, int OW,
                                           int K, int Kpad, int c_base,
                                           long long M);

void quant_im2col_kmajor_launch(const float* x, int8_t* out, float scale, int qmax,
                                int N, int C, int H, int W,
                                int kh, int kw, int sh, int sw, int ph, int pw,
                                int OH, int OW, int K, int Kpad, int c_base)
{
    const long long M = (long long)N * OH * OW;
    const dim3 grid((unsigned)((M + 255) / 256),
                    (unsigned)(Kpad < 16 ? Kpad : 16));
    quant_im2col_kmajor_kernel<<<grid, 256, 0, ptqint8::g_stream>>>(x, out, scale, qmax, C, H, W,
                                              kh, kw, sh, sw, ph, pw, OH, OW,
                                              K, Kpad, c_base, M);
}

// K-major 布局的 im2col：out[k * M + m]，供 W^T(Cout,K) @ colT(K,M) 的
// 转置 GEMM 直接使用。同一 k 下相邻 m 的输入/输出都连续，写可合并。
__global__ void quant_im2col_kmajor_kernel(const float* __restrict__ x,
                                           int8_t* __restrict__ out,
                                           float scale, int qmax,
                                           int C, int H, int W,
                                           int kh, int kw, int sh, int sw,
                                           int ph, int pw, int OH, int OW,
                                           int K, int Kpad, int c_base,
                                           long long M)
{
    const long long m = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;
    const int ow = (int)(m % OW);
    const long long t = m / OW;
    const int oh = (int)(t % OH);
    const int n = (int)(t / OH);
    const int ih0 = oh * sh - ph;
    const int iw0 = ow * sw - pw;
    for (int k = blockIdx.y; k < Kpad; k += gridDim.y)
    {
        int q = 0;
        if (k < K)
        {
            const int j = k % kw;
            const int t2 = k / kw;
            const int i = t2 % kh;
            const int c = t2 / kh;
            const int ih = ih0 + i;
            const int iw = iw0 + j;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                q = quant_one_i8(x[((size_t)n * C + c_base + c) * H * W + ih * W + iw],
                                 scale, qmax);
        }
        out[(size_t)k * M + m] = (int8_t)q;
    }
}

void quant_im2col_vec_launch(const float* x, int8_t* out, float scale, int qmax,
                             int N, int C, int H, int W,
                             int kh, int kw, int sh, int sw, int ph, int pw,
                             int OH, int OW, int K, int Kpad, int c_base)
{
    const long long M = (long long)N * OH * OW;
    const dim3 grid((unsigned)((M + 15) / 16));
    const int block = 256;
    if (kh == 3 && kw == 3)
        quant_im2col_vec_kernel<3, 3><<<grid, block, 0, ptqint8::g_stream>>>(x, out, scale, qmax, C, H, W,
                                                       sh, sw, ph, pw, OH, OW, K, Kpad, c_base, (int)M);
    else if (kh == 1 && kw == 1)
        quant_im2col_vec_kernel<1, 1><<<grid, block, 0, ptqint8::g_stream>>>(x, out, scale, qmax, C, H, W,
                                                       sh, sw, ph, pw, OH, OW, K, Kpad, c_base, (int)M);
    else if (kh == 7 && kw == 7)
        quant_im2col_vec_kernel<7, 7><<<grid, block, 0, ptqint8::g_stream>>>(x, out, scale, qmax, C, H, W,
                                                       sh, sw, ph, pw, OH, OW, K, Kpad, c_base, (int)M);
    else
        quant_im2col_vec_generic<<<grid, block, 0, ptqint8::g_stream>>>(x, out, scale, qmax, C, H, W,
                                                  kh, kw, sh, sw, ph, pw, OH, OW, K, Kpad, c_base, (int)M);
}


// ---------------------------------------------------------------- 3) requantize + bias + ReLU

// acc (N*OH*OW, Cout) int32，L-major（GEMM 原样输出）
//   → out (N, Cout, OH*OW) fp32
//
// out_idx = ((n*Cout + co) * HW + pos)，其中 HW = OH*OW
// acc_idx = ((n*HW + pos) * Cout + co)
// 两种布局的置换关系是 (n,co,pos) ↔ (n,pos,co)，所以必须显式换算，
// 不能让 acc 和 out 共用同一个线性下标。
//
// 融合的四件事：int32→float 转换 + 按输出通道乘 scale + 加 bias + ReLU。
// 分开做就是 4 次全量内存遍历，合成一个 kernel 是 1 读 1 写。
//
// zp_corr : 非对称激活的 zp 修正项（每通道），可为 null。
//   数学：out = Σ_k (q_a - zp)·q_w ·(sa·sw) + bias
//           = sa·sw·(Σ_k q_a·q_w) + bias - zp·sa·sw·Σ_k q_w
//   最后那项与输入无关，只跟权重有关 → 预计算成每通道一个常数。
//   没有它，非对称量化会整体偏移 zp·Σq_w 个码字。
__global__ void requant_kernel(const int32_t* __restrict__ acc,
                               float* __restrict__ out,
                               const float* __restrict__ wscale, // Cout
                               const float* __restrict__ bias,   // Cout
                               const float* __restrict__ zp_corr,// Cout 或 null
                               float ascale,
                               int Cout, int HW, int N, bool relu,
                               int cout_stride, int co_base)
{
    const long long total = (long long)N * Cout * HW;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total; idx += (long long)gridDim.x * blockDim.x)
    {
        const int pos = (int)(idx % HW);
        const int t = (int)(idx / HW);
        const int co = t % Cout;
        const int n = t / Cout;

        const long long acc_idx = ((long long)n * HW + pos) * Cout + co;
        const int v = acc[acc_idx];

        float y = (float)v * (wscale[co] * ascale) + (bias ? bias[co] : 0.0f);
        if (zp_corr) y += zp_corr[co];
        if (relu && y < 0.0f) y = 0.0f;
        // 分组卷积逐组写回整层输出：sample stride 是 cout_stride，不是组内 Cout。
        out[((long long)n * cout_stride + co_base + co) * HW + pos] = y;
    }
}

// 反量化回 fp32 NCHW 之前的 permute：(N,Cout,HW) -> (N,Cout,OH,OW)
// 其实不需要 —— (N,Cout,HW) 连续内存本身就是 NCHW 的 (N,Cout,OH,OW)，
// 因为 HW = OH*OW 且 out_idx 的 HW 是最内层。

// ---------------------------------------------------------------- host 侧包装

// 这几个包装存在的唯一理由：**launcher 必须编译在 .cu 里**。
// 把 <<<>>> 写在头文件里，include 该头的 .cpp 会报
// "a host function call cannot be configured"（nvcc 只对 .cu 展开 launcher）。
// 所以 kernel 本体和 launcher 都在这个 .cu 里，外部只调包装函数。

void quant_im2col_launch(const float* x, int8_t* out, float scale, int zp, int qmax,
                         int N, int C, int H, int W,
                         int kh, int kw, int sh, int sw, int ph, int pw,
                         int OH, int OW, int K, int Kpad, int c_base, int grid)
{
    quant_im2col_kernel<<<grid, 256, 0, ptqint8::g_stream>>>(x, out, scale, zp, qmax,
                                       N, C, H, W, kh, kw, sh, sw, ph, pw,
                                       OH, OW, K, Kpad, c_base);
}

void requant_launch(const int32_t* acc, float* out,
                    const float* wscale, const float* bias, const float* zp_corr,
                    float ascale, int Cout, int HW, int N, bool relu,
                    int cout_stride, int co_base, int grid)
{
    requant_kernel<<<grid, 256, 0, ptqint8::g_stream>>>(acc, out, wscale, bias, zp_corr,
                                  ascale, Cout, HW, N, relu, cout_stride, co_base);
}


// ---------------------------------------------------------------- 2b) 快速 quant + im2col
//
// ⭐ 为什么要有第二版（实测数据，不是猜）：
//   原版把 64 位一维 idx 连续拆成 (k, row, ow, oh, n)：每个元素 5 次 64 位
//   div/mod。bench_layers 实测（batch=512，ResNet-18 CIFAR）im2col 占总时间
//   64.7%（19.1 / 29.5 ms），等效算术吞吐仅 ~8 Gelem/s ≈ 每元素 250 core-cycle；
//   而它搬运的数据只有 ~150 MB，按 150 GB/s 只需 ~1 ms。**瓶颈是除法，不是带宽。**
//
//   新版：grid = (OW, OH, N)，三个空间/批索引直接来自 blockIdx（零除法），
//   k 维由线程循环覆盖；k -> (c,i,j) 的除数通过模板常量折叠成乘加
//   （3x3 / 1x1 / 7x7 三个真实形状），语义与原版逐位一致：
//     · padding 写码字 0，· Kpad 尾部补 0，· 行主序 (M, Kpad) 布局，
//     · 同样的 half-away-from-zero 舍入与对称 clamp。
//   旧 kernel 保留，probe/回归仍打它（口径不变）。
__device__ __forceinline__ void quant_im2col_fast_body(
    const float* __restrict__ x, int8_t* __restrict__ out,
    float scale, int qmax, int C, int H, int W,
    int sh, int sw, int ph, int pw, int OH, int OW,
    int K, int Kpad, int c_base, int n, int oh, int ow, int kh, int kw)
{
    const int ih0 = oh * sh - ph;
    const int iw0 = ow * sw - pw;
    int8_t* orow = out + (((long long)n * OH + oh) * OW + ow) * Kpad;
    const float* xn = x + (long long)n * C * H * W;
    for (int k = threadIdx.x; k < Kpad; k += blockDim.x)
    {
        if (k >= K) { orow[k] = 0; continue; }
        const int j = k % kw;
        const int t2 = k / kw;
        const int i = t2 % kh;
        const int c = t2 / kh;
        const int ih = ih0 + i;
        const int iw = iw0 + j;
        if (ih < 0 || ih >= H || iw < 0 || iw >= W) { orow[k] = 0; continue; }
        const float v = xn[((long long)(c + c_base)) * H * W + ih * W + iw];
        float q = torch_round_cuda(v / scale);
        q = fminf(fmaxf(q, -(float)(qmax + 1)), (float)qmax);
        orow[k] = (int8_t)(int)q;
    }
}

template <int KH, int KW>
__global__ void quant_im2col_fast_kernel(
    const float* __restrict__ x, int8_t* __restrict__ out,
    float scale, int qmax, int N, int C, int H, int W,
    int sh, int sw, int ph, int pw, int OH, int OW,
    int K, int Kpad, int c_base)
{
    quant_im2col_fast_body(x, out, scale, qmax, C, H, W, sh, sw, ph, pw,
                           OH, OW, K, Kpad, c_base, blockIdx.z, blockIdx.y,
                           blockIdx.x, KH, KW);
}

__global__ void quant_im2col_fast_generic(
    const float* __restrict__ x, int8_t* __restrict__ out,
    float scale, int qmax, int N, int C, int H, int W,
    int kh, int kw, int sh, int sw, int ph, int pw, int OH, int OW,
    int K, int Kpad, int c_base)
{
    quant_im2col_fast_body(x, out, scale, qmax, C, H, W, sh, sw, ph, pw,
                           OH, OW, K, Kpad, c_base, blockIdx.z, blockIdx.y,
                           blockIdx.x, kh, kw);
}

void quant_im2col_fast_launch(const float* x, int8_t* out, float scale, int zp,
                              int qmax, int N, int C, int H, int W,
                              int kh, int kw, int sh, int sw, int ph, int pw,
                              int OH, int OW, int K, int Kpad, int c_base)
{
    (void)zp;
    const dim3 grid((unsigned)OW, (unsigned)OH, (unsigned)N);
    const int block = 256;
    if (kh == 3 && kw == 3)
        quant_im2col_fast_kernel<3, 3><<<grid, block, 0, ptqint8::g_stream>>>(x, out, scale, qmax, N, C, H, W,
                                                        sh, sw, ph, pw, OH, OW, K, Kpad, c_base);
    else if (kh == 1 && kw == 1)
        quant_im2col_fast_kernel<1, 1><<<grid, block, 0, ptqint8::g_stream>>>(x, out, scale, qmax, N, C, H, W,
                                                        sh, sw, ph, pw, OH, OW, K, Kpad, c_base);
    else if (kh == 7 && kw == 7)
        quant_im2col_fast_kernel<7, 7><<<grid, block, 0, ptqint8::g_stream>>>(x, out, scale, qmax, N, C, H, W,
                                                        sh, sw, ph, pw, OH, OW, K, Kpad, c_base);
    else
        quant_im2col_fast_generic<<<grid, block, 0, ptqint8::g_stream>>>(x, out, scale, qmax, N, C, H, W,
                                                   kh, kw, sh, sw, ph, pw, OH, OW, K, Kpad, c_base);
}


// ---------------------------------------------------------------- 3b) requant（转置瓦片版）
//
// 为什么：旧版按 idx 连续枚举 (n, co, pos)，acc 的下标是
//   ((n*HW + pos)*Cout + co)，相邻线程的 pos 相差 1 → acc 步长 Cout*4 B，
// **int32 读完全非合并**（每 32 B sector 只用 4 B）。实测 requant 占
// 全段 16%，可达带宽只有 45–90 GB/s（峰值 192）。
//
// 新版用 32x32 共享内存瓦片做转置：读按 co 连续（合并），
// 写按 pos 连续（合并）。逐元素算式与旧版**逐位相同**
// （同样的 (float)v * (wscale*ascale) + bias 顺序），bench_layers 校验 mismatch=0。
__global__ void requant_tiled_kernel(const int32_t* __restrict__ acc,
                                     float* __restrict__ out,
                                     const float* __restrict__ wscale,
                                     const float* __restrict__ bias,
                                     const float* __restrict__ zp_corr,
                                     float ascale, int Cout, int HW, int N, bool relu,
                                     int cout_stride, int co_base)
{
    __shared__ int tile[32][33];
    const int pos0 = blockIdx.x * 32;
    const int co0 = blockIdx.y * 32;
    const int n = blockIdx.z;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

#pragma unroll
    for (int k = 0; k < 4; ++k)
    {
        const int r = ty + k * 8;          // 0..31，瓦片内的 pos
        const int pos = pos0 + r;
        const int co = co0 + tx;
        tile[r][tx] = (pos < HW && co < Cout)
                          ? acc[((long long)n * HW + pos) * Cout + co]
                          : 0;
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < 4; ++k)
    {
        const int co = co0 + ty + k * 8;
        const int pos = pos0 + tx;
        if (co < Cout && pos < HW)
        {
            float y = (float)tile[tx][ty + k * 8] * (wscale[co] * ascale) +
                      (bias ? bias[co] : 0.0f);
            if (zp_corr) y += zp_corr[co];
            if (relu && y < 0.0f) y = 0.0f;
            out[((long long)n * cout_stride + co_base + co) * HW + pos] = y;
        }
    }
}

void requant_tiled_launch(const int32_t* acc, float* out,
                          const float* wscale, const float* bias, const float* zp_corr,
                          float ascale, int Cout, int HW, int N, bool relu,
                          int cout_stride, int co_base)
{
    const dim3 block(32, 8);
    const dim3 grid((HW + 31) / 32, (Cout + 31) / 32, N);
    requant_tiled_kernel<<<grid, block, 0, ptqint8::g_stream>>>(acc, out, wscale, bias, zp_corr, ascale,
                                          Cout, HW, N, relu, cout_stride, co_base);
}

// 转置 GEMM 的 epilogue：acc 是 (Cout, M=N*Lpos) 行主序，已经是 NCHW
// 的通道主序；每通道对连续 M 做反量化，写出的 pos 也连续，无需 shared 转置。
__global__ void requant_t_kernel(const int32_t* __restrict__ acc,
                                 float* __restrict__ out,
                                 const float* __restrict__ wscale,
                                 const float* __restrict__ bias,
                                 const float* __restrict__ zp_corr,
                                 float ascale, int Cout, int Lpos, int N,
                                 bool relu, int cout_stride, int co_base,
                                 long long M)
{
    const int co = blockIdx.y;
    const long long m = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;
    const int n = (int)(m / Lpos);
    const int pos = (int)(m - (long long)n * Lpos);
    float y = (float)acc[(size_t)co * M + m] * (wscale[co] * ascale) +
              (bias ? bias[co] : 0.0f);
    if (zp_corr) y += zp_corr[co];
    if (relu && y < 0.0f) y = 0.0f;
    out[((size_t)n * cout_stride + co_base + co) * Lpos + pos] = y;
}

void requant_t_launch(const int32_t* acc, float* out,
                      const float* wscale, const float* bias, const float* zp_corr,
                      float ascale, int Cout, int Lpos, int N, bool relu,
                      int cout_stride, int co_base)
{
    const long long M = (long long)N * Lpos;
    const dim3 grid((unsigned)((M + 255) / 256), (unsigned)Cout);
    requant_t_kernel<<<grid, 256, 0, ptqint8::g_stream>>>(acc, out, wscale, bias, zp_corr, ascale,
                                    Cout, Lpos, N, relu, cout_stride, co_base, M);
}

void reduce_minmax_launch(const float* x, size_t n, float* out, int grid)
{
    reduce_minmax_kernel<<<grid, 256, 0, ptqint8::g_stream>>>(x, n, out);
}

// ---------------------------------------------------------------- 4) 逐元素加（残差）

// out = a + b，两个都是 (N,C,HW) 连续布局，尺寸相同（add 前必已对齐）。
// ⛔ 不能假设通道/尺寸一定相同：identity 直连时分辨率与通道都由 block 保证
//    一致（PyTorch 的 ResNet 靠 stride 只能出现在带 downsample 的 block），
//    但这个前提必须显式校验，不能靠"应该会相等"。
__global__ void add_kernel(float* __restrict__ out,
                           const float* __restrict__ a,
                           const float* __restrict__ b, size_t n)
{
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         i < n; i += (size_t)gridDim.x * blockDim.x)
        out[i] = a[i] + b[i];
}

// ---------------------------------------------------------------- 5) maxpool 3x3 s2 p1

// 只实现 ResNet-18 stem 用的那一种（3x3 / stride 2 / pad 1），
// 不做通用化 —— 通用 maxpool 要处理 ceil_mode / dilation / 通道维，
// 在这个网络里全是死代码。**只实现真正会走到的分支。**
__global__ void maxpool3x3s2p1_kernel(const float* __restrict__ x,
                                      float* __restrict__ out,
                                      int N, int C, int H, int W)
{
    const int OH = (H + 2 - 3) / 2 + 1;
    const int OW = (W + 2 - 3) / 2 + 1;

    const long long total = (long long)N * C * OH * OW;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total; idx += (long long)gridDim.x * blockDim.x)
    {
        const int ow = (int)(idx % OW);
        const long long t = idx / OW;
        const int oh = (int)(t % OH);
        const long long t2 = t / OH;
        const int c = (int)(t2 % C);
        const int n = (int)(t2 / C);

        const float* xp = x + ((long long)n * C + c) * H * W;
        float m = -3.4e38f;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            {
                const int ih = oh * 2 + i - 1;
                const int iw = ow * 2 + j - 1;
                // padding 用 -inf 而不是 0：maxpool 的 padding 必须是 -inf，
                // 否则当窗口全在 padding 区时会输出 0 而不是负无穷。
                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                const float v = xp[ih * W + iw];
                if (v > m) m = v;
            }
        out[idx] = m;
    }
}

__global__ void add_relu_kernel(float* __restrict__ out,
                               const float* __restrict__ a,
                               const float* __restrict__ b, size_t n)
{
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         i < n; i += (size_t)gridDim.x * blockDim.x)
    {
        float v = a[i] + b[i];
        out[i] = v < 0.0f ? 0.0f : v;
    }
}

void add_launch(float* out, const float* a, const float* b, size_t n, int grid)
{
    add_kernel<<<grid, 256, 0, ptqint8::g_stream>>>(out, a, b, n);
}

void add_relu_launch(float* out, const float* a, const float* b, size_t n, int grid)
{
    add_relu_kernel<<<grid, 256, 0, ptqint8::g_stream>>>(out, a, b, n);
}

// ---------------------------------------------------------------- 6) 逐元素激活

// mode：1=ReLU，2=ReLU6（clamp 到 [0,6]）。
// ReLU6 必须真的 clamp 上界：MobileNetV2 训练权重下确有层输出 >6
// （features.1 的 depthwise 最大值 6.685），用 ReLU 代替会让逐层校验
// 出现 max|diff| ≈ 8.35。
__global__ void relu_kernel(float* __restrict__ x, size_t n, int mode)
{
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         i < n; i += (size_t)gridDim.x * blockDim.x)
    {
        if (mode == 2)
        {
            if (x[i] < 0.0f) x[i] = 0.0f;
            else if (x[i] > 6.0f) x[i] = 6.0f;
        }
        else if (x[i] < 0.0f) x[i] = 0.0f;
    }
}

void relu_launch(float* x, size_t n, int grid, int mode)
{
    relu_kernel<<<grid, 256, 0, ptqint8::g_stream>>>(x, n, mode);
}

// ---------------------------------------------------------------- 7) depthwise conv
//
// cin=1, groups=Cout。每个输出元素 = Σ_{i,j} q(x)·q_w[c,i,j]，int32 累加，
// 再乘 (ascale*wscale[c]) 加 bias —— 与 im2col+GEMM+requant 的数值语义一致。
// 不用 _int_mm：每组只有 1 个输出通道（N=1），既慢也不保证支持。
//
// w 布局：[C][Kpad]，K=kh*kw；w_stride=Kpad。
// padding 写**码字 0**（对应真实值 0 的贡献），不是把 0.0 量化后的值。
__global__ void depthwise_conv_kernel(const float* __restrict__ x,
                                      float* __restrict__ out,
                                      const int8_t* __restrict__ w,
                                      const float* __restrict__ wscale,
                                      const float* __restrict__ bias,
                                      float ascale, int qmax,
                                      int N, int C, int H, int W,
                                      int kh, int kw, int sh, int sw, int ph, int pw,
                                      int OH, int OW, int w_stride)
{
    const long long total = (long long)N * C * OH * OW;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total; idx += (long long)gridDim.x * blockDim.x)
    {
        const int ow = (int)(idx % OW);
        const long long t = idx / OW;
        const int oh = (int)(t % OH);
        const long long t2 = t / OH;
        const int c = (int)(t2 % C);
        const int n = (int)(t2 / C);

        int acc = 0;
        const int8_t* wp = w + (size_t)c * w_stride;
        for (int i = 0; i < kh; i++)
            for (int j = 0; j < kw; j++)
            {
                const int ih = oh * sh + i - ph;
                const int iw = ow * sw + j - pw;
                int qv = 0;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                {
                    const float v = x[((long long)n * C + c) * H * W + ih * W + iw];
                    float q = torch_round_cuda(v / ascale);
                    q = fminf(fmaxf(q, -(float)(qmax + 1)), (float)qmax);
                    qv = (int)q;
                }
                acc += qv * (int)wp[i * kw + j];
            }
        const float y = (float)acc * (wscale[c] * ascale) + bias[c];
        out[((long long)n * C + c) * OH * OW + oh * OW + ow] = y;
    }
}

void depthwise_conv_launch(const float* x, float* out, const int8_t* w,
                           const float* wscale, const float* bias,
                           float ascale, int qmax,
                           int N, int C, int H, int W,
                           int kh, int kw, int sh, int sw, int ph, int pw,
                           int OH, int OW, int w_stride, int grid)
{
    depthwise_conv_kernel<<<grid, 256, 0, ptqint8::g_stream>>>(x, out, w, wscale, bias, ascale, qmax,
                                         N, C, H, W, kh, kw, sh, sw, ph, pw,
                                         OH, OW, w_stride);
}

void maxpool_launch(const float* x, float* out, int N, int C, int H, int W, int grid)
{
    maxpool3x3s2p1_kernel<<<grid, 256, 0, ptqint8::g_stream>>>(x, out, N, C, H, W);
}

} // namespace ptqint8