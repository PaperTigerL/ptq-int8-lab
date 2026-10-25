// implicit GEMM（不落 im2col 的 W8A8 直接卷积）—— DP4A 实现。
//
// 与旧路径的关系：
//   旧：fp32 → quant+im2col (M,Kpad) int8 落显存 → at::_int_mm → int32 → requant
//   新：预量化一次 qact[n][c4][S][4] → block 内把「本 chunk 的全部有效 tap 权重」
//       一次搬进 shared，直接 __dp4a 累加 int32 → 融合 epilogue 写 fp32 NCHW。
//       im2col 矩阵和 int32 acc 都不落显存。
//
// 三个必须遵守的口径：
//   1. int32 累加结果与 GEMM 逐元素相同（整数加法与顺序无关），所以 epilogue
//      的算式逐位复用旧路径的 requant（bench_layers 校验 mismatch=0）。
//   2. 权重布局 w_imp[((tap*C4 + c4)*Cout + co)*4 + lane]，一次 32-bit load
//      就是 DP4A 需要的 4 个连续输入通道；同 k4 下 32 个输出通道的权重连续。
//   3. K 维顺序是 tap-major（tap 在外、通道 c4 在内），补零只影响乘 0。
//
// 微瓦片：warp 负责 P=8 个输出位置，lane 负责 Q=4（或 2）个输出通道，
// block = 8 warps = TM=64 个位置 × NT=32Q 个通道。
//
// ⭐ 四个「实测逼出来」的结构决定（都有数据支撑，不是保守）：
//   · 权重 slab 覆盖「一个 c4 chunk 的全部有效 tap」，sync 从每 tap 两次降到
//     每 chunk 两次。层0（stem k7s2）实测 7.59 ms = 每 block 98 次 __syncthreads
//     + 49 次 256B 权重读的纯延迟串行；改成每 chunk 两次后掉到 0.3 ms 量级。
//   · 只给「可能有效」的 tap 分配权重槽：1x1 空间的 3x3 层里 9 个 tap 只有 1 个
//     能落进图内（padding 之外全是边界），权重读和 sync 都省 9 倍。
//   · 输入 tile 缓存只在 taps>1 且 S>=4 时开；S=1 时缓存纯浪费（要 32KB shared，
//     占用率打到 1 block/SM）。
//   · 融合 epilogue 的转置 tile 行距 +1：不 +1 时 (lane*Q+q)*TM 的步长是 32 的
//     倍数，32 个 lane 全撞同一家 bank（32-way conflict，实测 epilogue 比主循环
//     还慢）。
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#include "implicit_conv.cuh"
#include "stream_ctx.cuh"

namespace ptqint8 {

// 舍入语义与 int8_kernels.cu 的 torch_round_cuda 完全一致：
// half-away-from-zero（与 torch.round 对齐），不是 rintf 的 half-to-even。
__device__ __forceinline__ float imp_round_away(float v)
{
    return (v >= 0.0f) ? floorf(v + 0.5f) : ceilf(v - 0.5f);
}

constexpr int kImpWarps = 8;
constexpr int kImpP = 8;                       // 每 warp 的输出位置数
constexpr int kImpTM = kImpP * kImpWarps;      // 每 block 的位置数（连续）
constexpr int kImpCH4Max = 32;                 // 一次缓存多少组 c4（上限）
constexpr size_t kImpEpiTileMax = 17 * 1024;   // 融合 epilogue 的转置 tile 上限
constexpr size_t kImpCacheMax = 24 * 1024;     // 输入 tile 缓存上限
constexpr size_t kImpSwMax = 4 * 1024;         // 权重 slab 目标上限（tap 合并用）

// epi_mode: 0 = 写 int32 acc（(M,Cout) 行主序）
//           1 = 融合反量化 + shared 转置后写 NCHW（Lpos > 1）
//           2 = 融合反量化直接写 NCHW（Lpos == 1，(m,co) 就是 NCHW）
// ch: 一个 c4 chunk 含多少组 c4（host 按 shared 预算算好）
// ⭐ __launch_bounds__(256, 4)：不写这一行时 3x3 变体用了 **229 个寄存器**
// （ptxas -v 实测），64K 寄存器 / (229×256) = 1.1 → 每 SM 只有 1 个 block =
// 8 warps = 12.5% 占用率。这个 kernel 是延迟受限的，占用率就是性能。
// 7x7（stem）更糟：255 个寄存器 + 2288 字节栈帧，2284 B spill stores。
// 4 blocks/SM ⇒ 64 寄存器上限，是实测出来的甜点（2 → 仍然 1 block/SM 的浪费，
// 8 → 开始 spill）。
template <int KH, int KW, int SH, int SW, int PH, int PW, int Q>
__global__ void __launch_bounds__(256, 4) implicit_conv_kernel(
    const int8_t* __restrict__ qin, const int8_t* __restrict__ w_imp,
    int32_t* __restrict__ acc_out, float* __restrict__ out,
    const float* __restrict__ wscale, const float* __restrict__ bias,
    const float* __restrict__ zp_corr, float ascale,
    int cout_stride, int co_base, bool relu,
    bool use_cache, int epi_mode, int ch, int tchunk,
    int C, int H, int W, int OH, int OW, int Cout,
    int imgs_max, int M)
{
    constexpr int NT = 32 * Q;
    constexpr int TM = kImpTM;
    constexpr int TMp = TM + 1;                // 转置 tile 行距（+1 消 bank 冲突）
    constexpr int TAPS = KH * KW;
    extern __shared__ int8_t smem[];
    __shared__ int tap_any[TAPS];
    __shared__ int slot_of_tap[TAPS];
    __shared__ int tap_of_slot[TAPS];
    __shared__ int nslots;
    // ⭐ masks 放 shared 而不是寄存器数组：TAPS=49 的 stem 会把 49 个 unsigned
    // 全压在寄存器里，直接挤爆 → 本地内存溢出（spill），实测该层比 GEMM 慢
    // 10 倍。它本来就是 **warp 内统一** 的值，一个字节/项就够，读是广播。
    __shared__ unsigned char smask[kImpWarps][TAPS];

    const int S = H * W;
    const int Lpos = OH * OW;
    const int C4 = (C + 3) >> 2;
    const int m0 = blockIdx.x * TM;
    if (m0 >= M) return;
    const int co0 = blockIdx.y * NT;
    int mlast = m0 + TM - 1;
    if (mlast >= M) mlast = M - 1;
    const int n0 = m0 / Lpos;
    const int imgs = mlast / Lpos - n0 + 1;
    const size_t sin_bytes = use_cache
        ? ((KH == 1 && KW == 1) ? (size_t)C4 * TM * 4
                                : (size_t)C4 * imgs_max * S * 4)
        : 0;
    int8_t* sin = smem;
    int* sw = reinterpret_cast<int*>(smem + ((sin_bytes + 15) & ~(size_t)15));
    // 权重 slab 之后放一个恒零 int：越界位置读它，贡献 0。
    int* zero = sw + (size_t)tchunk * ch * NT;
    if (threadIdx.x == 0) *zero = 0;
    if (threadIdx.x < TAPS) tap_any[threadIdx.x] = 0;

    if (use_cache)
    {
        if (KH == 1 && KW == 1)
        {
            for (int t = threadIdx.x; t < C4 * TM; t += blockDim.x)
            {
                const int c4 = t / TM;
                const int p = t - c4 * TM;
                const int m = m0 + p;
                if (m >= M) continue;
                const int n = m / Lpos;
                const int pos = m - n * Lpos;
                const int oh = pos / OW;
                const int ow = pos - oh * OW;
                const int sp = (oh * SH) * W + ow * SW;
                const int8_t* src = qin + (((size_t)n * C4 + c4) * S + sp) * 4;
                int8_t* dst = sin + (((size_t)c4 * TM + p) * 4);
                *reinterpret_cast<int*>(dst) = *reinterpret_cast<const int*>(src);
            }
        }
        else if (S >= 4)
        {
            const int total_vec = (C4 * imgs * S) >> 2;
            for (int t = threadIdx.x; t < total_vec; t += blockDim.x)
            {
                const int w = t << 2;
                const int c4 = w / (imgs * S);
                const int rem = w - c4 * (imgs * S);
                const int img = rem / S;
                const int s = rem - img * S;
                const int8_t* src = qin + ((size_t)(n0 + img) * C4 + c4) * S * 4 + s * 4;
                int8_t* dst = sin + (((size_t)c4 * imgs + img) * S + s) * 4;
                *reinterpret_cast<int4*>(dst) = *reinterpret_cast<const int4*>(src);
            }
        }
        else
        {
            for (int t = threadIdx.x; t < C4 * imgs; t += blockDim.x)
            {
                const int c4 = t / imgs;
                const int img = t - c4 * imgs;
                const int8_t* src = qin + (((size_t)(n0 + img) * C4 + c4) * S) * 4;
                int8_t* dst = sin + (((size_t)c4 * imgs + img) * S) * 4;
                *reinterpret_cast<int*>(dst) = *reinterpret_cast<const int*>(src);
            }
        }
    }

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int mbase = m0 + warp * kImpP;
    int h0[kImpP], w0[kImpP], off[kImpP];
    bool alive[kImpP];
#pragma unroll
    for (int p = 0; p < kImpP; p++)
    {
        const int m = mbase + p;
        alive[p] = (m < M);
        h0[p] = w0[p] = off[p] = 0;
        if (alive[p])
        {
            const int n = m / Lpos;
            const int pos = m - n * Lpos;
            const int oh = pos / OW;
            const int ow = pos - oh * OW;
            h0[p] = oh * SH - PH;
            w0[p] = ow * SW - PW;
            if (KH == 1 && KW == 1)
                off[p] = use_cache ? (m - m0) * 4
                                   : ((size_t)n * C4 * S + h0[p] * W + w0[p]) * 4;
            else
                off[p] = use_cache ? ((n - n0) * S + h0[p] * W + w0[p]) * 4
                                   : ((size_t)n * C4 * S + h0[p] * W + w0[p]) * 4;
        }
    }

    // zero / tap_any / 输入 tile 三件事都要在下面用到之前落地。
    __syncthreads();
#pragma unroll
    for (int i = 0; i < KH; i++)
#pragma unroll
        for (int j = 0; j < KW; j++)
        {
            unsigned mk = 0;
#pragma unroll
            for (int p = 0; p < kImpP; p++)
                if (alive[p])
                {
                    const int ih = h0[p] + i;
                    const int iw = w0[p] + j;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) mk |= (1u << p);
                }
            smask[warp][i * KW + j] = (unsigned char)mk;
            if (lane == 0 && mk) atomicOr(&tap_any[i * KW + j], 1);
        }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        int sc = 0;
        for (int t = 0; t < TAPS; t++)
        {
            if (tap_any[t])
            {
                tap_of_slot[sc] = t;
                slot_of_tap[t] = sc;
                sc++;
            }
            else
                slot_of_tap[t] = -1;
        }
        nslots = sc;
    }
    __syncthreads();
    const int ns = nslots;

    int acc[kImpP * Q];
#pragma unroll
    for (int z = 0; z < kImpP * Q; z++) acc[z] = 0;

    for (int s0 = 0; s0 < ns; s0 += tchunk)
    {
      const int sn = (ns - s0) < tchunk ? (ns - s0) : tchunk;
      for (int c4b = 0; c4b < C4; c4b += ch)
      {
        const int c4n = (C4 - c4b) < ch ? (C4 - c4b) : ch;
        // 一次把本 chunk 的全部有效 tap 权重搬进 shared（l 连续 → 合并读）
        {
            const int total = sn * c4n * NT;
            for (int t = threadIdx.x; t < total; t += blockDim.x)
            {
                const int s = t / (c4n * NT);
                const int rem = t - s * (c4n * NT);
                const int c4 = rem / NT;
                const int l = rem - c4 * NT;
                const int tap = tap_of_slot[s0 + s];
                sw[(size_t)s * c4n * NT + t - s * (c4n * NT)] =
                    *reinterpret_cast<const int*>(
                        w_imp + (((size_t)(tap * C4 + c4b + c4) * Cout + co0 + l) << 2));
            }
        }
        __syncthreads();
        // ⚠️ 这里**不能**展开：9 个 tap 全展开会让每个 p/q 的地址与中间值同时
        // 活着，实测把寄存器从 64 推到 229（→ 1 block/SM）。展开省下的那点
        // 索引算术远不值这个代价。
#pragma unroll 1
        for (int tap = 0; tap < TAPS; tap++)
            {
                const int slot = slot_of_tap[tap];
                if (slot < s0 || slot >= s0 + sn) continue;
                const int ti = tap / KW;
                const int tap_off = (ti * W + (tap - ti * KW)) * 4;
                const int* swp = sw + (size_t)(slot - s0) * c4n * NT;
                const unsigned mk = smask[warp][tap];
                bool valid[kImpP];
#pragma unroll
                for (int p = 0; p < kImpP; p++)
                    valid[p] = ((mk >> p) & 1u) != 0;
                for (int c4 = 0; c4 < c4n; c4++)
                {
                    const int8_t* cbase;
                    if (use_cache)
                        cbase = sin + (size_t)(c4b + c4) *
                                          ((KH == 1 && KW == 1) ? (TM * 4) : (imgs * S * 4));
                    else
                        cbase = qin + (size_t)(c4b + c4) * S * 4;
                    if (Q == 4)
                    {
                        const int4 wv = *reinterpret_cast<const int4*>(swp + c4 * NT + lane * 4);
#pragma unroll
                        for (int p = 0; p < kImpP; p++)
                        {
                            const int8_t* ap = valid[p] ? cbase + off[p] + tap_off
                                                        : reinterpret_cast<const int8_t*>(zero);
                            const int a = *reinterpret_cast<const int*>(ap);
                            acc[p * Q + 0] = __dp4a(a, wv.x, acc[p * Q + 0]);
                            acc[p * Q + 1] = __dp4a(a, wv.y, acc[p * Q + 1]);
                            acc[p * Q + 2] = __dp4a(a, wv.z, acc[p * Q + 2]);
                            acc[p * Q + 3] = __dp4a(a, wv.w, acc[p * Q + 3]);
                        }
                    }
                    else
                    {
                        const int2 wv = *reinterpret_cast<const int2*>(swp + c4 * NT + lane * 2);
#pragma unroll
                        for (int p = 0; p < kImpP; p++)
                        {
                            const int8_t* ap = valid[p] ? cbase + off[p] + tap_off
                                                        : reinterpret_cast<const int8_t*>(zero);
                            const int a = *reinterpret_cast<const int*>(ap);
                            acc[p * Q + 0] = __dp4a(a, wv.x, acc[p * Q + 0]);
                            acc[p * Q + 1] = __dp4a(a, wv.y, acc[p * Q + 1]);
                        }
                    }
                }
            }
        __syncthreads();
      }
    }

    if (epi_mode == 1)
    {
        __syncthreads();
        float* sout = reinterpret_cast<float*>(smem);
#pragma unroll
        for (int p = 0; p < kImpP; p++)
        {
            const int m = mbase + p;
#pragma unroll
            for (int q = 0; q < Q; q++)
            {
                const int co = co0 + lane * Q + q;
                float y = (float)acc[p * Q + q] * (wscale[co] * ascale) +
                          (bias ? bias[co] : 0.0f);
                if (zp_corr) y += zp_corr[co];
                if (relu && y < 0.0f) y = 0.0f;
                sout[(lane * Q + q) * TMp + (warp * kImpP + p)] = (m < M) ? y : 0.0f;
            }
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < NT * TM; idx += blockDim.x)
        {
            const int co_local = idx / TM;
            const int pos_local = idx - co_local * TM;
            const int m = m0 + pos_local;
            if (m >= M) continue;
            const int n = m / Lpos;
            const int pos = m - n * Lpos;
            out[((size_t)n * cout_stride + co_base + co0 + co_local) * Lpos + pos] =
                sout[co_local * TMp + pos_local];
        }
    }
    else if (epi_mode == 2)
    {
        // Lpos == 1：acc 的 (m, co) 就是 NCHW 的 (n, cout)，无需转置。
#pragma unroll
        for (int p = 0; p < kImpP; p++)
        {
            const int m = mbase + p;
            if (m >= M) continue;
            const int co = co0 + lane * Q;
            float* dst = out + ((size_t)m * cout_stride + co_base + co);
            float y[Q];
#pragma unroll
            for (int q = 0; q < Q; q++)
            {
                float v = (float)acc[p * Q + q] * (wscale[co + q] * ascale) +
                          (bias ? bias[co + q] : 0.0f);
                if (zp_corr) v += zp_corr[co + q];
                if (relu && v < 0.0f) v = 0.0f;
                y[q] = v;
            }
            if (Q == 4)
                *reinterpret_cast<float4*>(dst) = make_float4(y[0], y[1], y[2], y[3]);
            else
                *reinterpret_cast<float2*>(dst) = make_float2(y[0], y[1]);
        }
    }
    else
    {
#pragma unroll
        for (int p = 0; p < kImpP; p++)
        {
            const int m = mbase + p;
            if (m < M)
            {
                const int cb = co0 + lane * Q;
                if (Q == 4)
                    *reinterpret_cast<int4*>(acc_out + (size_t)m * Cout + cb) =
                        make_int4(acc[p * Q + 0], acc[p * Q + 1],
                                  acc[p * Q + 2], acc[p * Q + 3]);
                else
                    *reinterpret_cast<int2*>(acc_out + (size_t)m * Cout + cb) =
                        make_int2(acc[p * Q + 0], acc[p * Q + 1]);
            }
        }
    }
}

struct ImplicitEpi
{
    float* out = nullptr;
    const float* wscale = nullptr;
    const float* bias = nullptr;
    const float* zp_corr = nullptr;
    float ascale = 0.0f;
    int cout_stride = 0;
    int co_base = 0;
    bool relu = false;
};

// c4 chunk 与 tap chunk 的选择：
//   ch 保持旧口径（min(C4,32)）—— 实测这是占用率/同步次数的最佳点，
//   slab 大小不变（l1 4KB、N=128 的层 16KB），不改动已验证的行为。
//   tchunk 只在一个 c4 chunk 的权重很小时才 >1：典型是 stem（C=3 → C4=1，
//   每 tap 只有 256B），把 49 个 tap 分成 ceil(49/tchunk) 次搬，
//   sync 从 98 次降到个位数，而 shared 仍压在 4KB。
static void implicit_pick_chunks(int alloc_slots, int NT, int C4, int* ch_out, int* tch_out)
{
    int ch = C4 < kImpCH4Max ? C4 : kImpCH4Max;
    if (ch < 1) ch = 1;
    int tch = 1;
    while (tch < alloc_slots && (size_t)(tch + 1) * ch * NT * 4 <= kImpSwMax) tch++;
    *ch_out = ch;
    *tch_out = tch;
}

template <int KH, int KW, int SH, int SW, int PH, int PW, int Q>
static bool implicit_launch_shape(const int8_t* qact, const int8_t* w_imp,
                                  int32_t* acc, const ImplicitEpi& epi,
                                  bool use_cache,
                                  int n, int C, int h, int w, int oh, int ow, int cout)
{
    constexpr int NT = 32 * Q;
    constexpr int TM = kImpTM;
    constexpr int TAPS = KH * KW;
    const int S = h * w;
    const int Lpos = oh * ow;
    const int C4 = (C + 3) >> 2;
    if (cout % NT != 0) return false;
    const long long Mll = (long long)n * Lpos;
    if (Mll > 0x7fffffffLL) return false;
    const int M = (int)Mll;
    // 1x1 空间（S==1）时 3x3 的 9 个 tap 里只有中心那个能落进图内。
    const int alloc_slots = (S == 1) ? 1 : TAPS;
    const int imgs_max = use_cache ? (TM + Lpos - 1) / Lpos : 1;
    const size_t sin_b = !use_cache
        ? 0
        : ((KH == 1 && KW == 1) ? (size_t)C4 * TM * 4
                                : (size_t)C4 * imgs_max * S * 4);
    int ch = 1, tch = 1;
    implicit_pick_chunks(alloc_slots, NT, C4, &ch, &tch);
    const size_t sw_b = (size_t)tch * ch * NT * 4;
    const size_t base_b = ((sin_b + 15) & ~(size_t)15) + sw_b + 4;

    int epi_mode = 0;
    size_t out_b = 0;
    if (epi.out)
    {
        if (Lpos == 1)
            epi_mode = 2;
        else
        {
            out_b = (size_t)NT * (TM + 1) * sizeof(float);
            if (out_b > kImpEpiTileMax) return false;
            epi_mode = 1;
        }
    }
    const size_t shmem = base_b > out_b ? base_b : out_b;
    if (shmem > 64 * 1024) return false;
    const dim3 grid((unsigned)((M + TM - 1) / TM), (unsigned)(cout / NT));
    auto kern = implicit_conv_kernel<KH, KW, SH, SW, PH, PW, Q>;
    if (shmem + 1024 > 48 * 1024)
    {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024);
        cudaGetLastError();
    }
    kern<<<grid, 256, shmem, g_stream>>>(qact, w_imp, acc, epi.out, epi.wscale, epi.bias,
                                        epi.zp_corr, epi.ascale, epi.cout_stride,
                                        epi.co_base, epi.relu, use_cache, epi_mode, ch,
                                        tch, C, h, w, oh, ow, cout, imgs_max, M);
    return cudaGetLastError() == cudaSuccess;
}

template <int KH, int KW, int SH, int SW, int PH, int PW>
static bool implicit_try_shape(const int8_t* qact, const int8_t* w_imp, int32_t* acc,
                               const ImplicitEpi& epi, bool allow_direct,
                               int n, int C, int h, int w, int oh, int ow, int cout)
{
    const int S = h * w;
    const int Lpos = oh * ow;
    const int C4 = (C + 3) >> 2;
    const bool q4 = (cout % 128 == 0);
    const int NT = q4 ? 128 : 64;
    if (cout % NT != 0) return false;

    // 输入 tile 缓存的自动判据（见文件头的实测说明）。
    bool cache = (KH * KW > 1) && (S >= 4);
    if (!allow_direct) cache = true;             // A/B：强制缓存路
    if (cache)
    {
        const int imgs_max = (kImpTM + Lpos - 1) / Lpos;
        const size_t sin_b = (size_t)C4 * imgs_max * S * 4;
        const int slots = (S == 1) ? 1 : KH * KW;
        int ch = 1, tch = 1;
        implicit_pick_chunks(slots, NT, C4, &ch, &tch);
        const size_t sw_b = (size_t)tch * ch * NT * 4;
        if (sin_b + sw_b > kImpCacheMax) cache = false;
    }

    if (q4)
    {
        if (implicit_launch_shape<KH, KW, SH, SW, PH, PW, 4>(
                qact, w_imp, acc, epi, cache, n, C, h, w, oh, ow, cout))
            return true;
        if (cache && implicit_launch_shape<KH, KW, SH, SW, PH, PW, 4>(
                         qact, w_imp, acc, epi, false, n, C, h, w, oh, ow, cout))
            return true;
    }
    else
    {
        if (implicit_launch_shape<KH, KW, SH, SW, PH, PW, 2>(
                qact, w_imp, acc, epi, cache, n, C, h, w, oh, ow, cout))
            return true;
        if (cache && implicit_launch_shape<KH, KW, SH, SW, PH, PW, 2>(
                         qact, w_imp, acc, epi, false, n, C, h, w, oh, ow, cout))
            return true;
    }
    return false;
}

static bool implicit_dispatch(const int8_t* qact, const int8_t* w_imp, int32_t* acc,
                              const ImplicitEpi& epi, bool allow_direct,
                              int n, int C, int h, int w,
                              int kh, int kw, int sh, int sw, int ph, int pw,
                              int oh, int ow, int cout)
{
    if (kh == 7 && kw == 7 && sh == 2 && sw == 2 && ph == 3 && pw == 3)
        return implicit_try_shape<7, 7, 2, 2, 3, 3>(qact, w_imp, acc, epi, allow_direct,
                                                   n, C, h, w, oh, ow, cout);
    if (kh == 3 && kw == 3 && sh == 1 && sw == 1 && ph == 1 && pw == 1)
        return implicit_try_shape<3, 3, 1, 1, 1, 1>(qact, w_imp, acc, epi, allow_direct,
                                                   n, C, h, w, oh, ow, cout);
    if (kh == 3 && kw == 3 && sh == 2 && sw == 2 && ph == 1 && pw == 1)
        return implicit_try_shape<3, 3, 2, 2, 1, 1>(qact, w_imp, acc, epi, allow_direct,
                                                   n, C, h, w, oh, ow, cout);
    if (kh == 3 && kw == 3 && sh == 1 && sw == 1 && ph == 0 && pw == 0)
        return implicit_try_shape<3, 3, 1, 1, 0, 0>(qact, w_imp, acc, epi, allow_direct,
                                                   n, C, h, w, oh, ow, cout);
    if (kh == 1 && kw == 1 && sh == 1 && sw == 1 && ph == 0 && pw == 0)
        return implicit_try_shape<1, 1, 1, 1, 0, 0>(qact, w_imp, acc, epi, allow_direct,
                                                   n, C, h, w, oh, ow, cout);
    if (kh == 1 && kw == 1 && sh == 2 && sw == 2 && ph == 0 && pw == 0)
        return implicit_try_shape<1, 1, 2, 2, 0, 0>(qact, w_imp, acc, epi, allow_direct,
                                                   n, C, h, w, oh, ow, cout);
    return false;
}

bool implicit_conv_launch(const int8_t* qact, const int8_t* w_imp,
                          int32_t* acc, int n, int C, int h, int w,
                          int kh, int kw, int sh, int sw, int ph, int pw,
                          int oh, int ow, int cout, bool allow_direct)
{
    ImplicitEpi epi;
    return implicit_dispatch(qact, w_imp, acc, epi, allow_direct, n, C, h, w,
                             kh, kw, sh, sw, ph, pw, oh, ow, cout);
}

bool implicit_conv_out_launch(const int8_t* qact, const int8_t* w_imp, float* out,
                              const float* wscale, const float* bias,
                              const float* zp_corr, float ascale,
                              int n, int C, int h, int w,
                              int kh, int kw, int sh, int sw, int ph, int pw,
                              int oh, int ow, int cout,
                              int cout_stride, int co_base, bool relu,
                              bool allow_direct)
{
    ImplicitEpi epi;
    epi.out = out;
    epi.wscale = wscale;
    epi.bias = bias;
    epi.zp_corr = zp_corr;
    epi.ascale = ascale;
    epi.cout_stride = cout_stride;
    epi.co_base = co_base;
    epi.relu = relu;
    return implicit_dispatch(qact, w_imp, nullptr, epi, allow_direct, n, C, h, w,
                             kh, kw, sh, sw, ph, pw, oh, ow, cout);
}

// 全局平均池化：一个 block 处理一个 (n,c)。
__global__ void gap_kernel(const float* __restrict__ x, float* __restrict__ out, int HW)
{
    const int idx = blockIdx.x;
    const float* p = x + (size_t)idx * HW;
    float s = 0.0f;
    for (int k = threadIdx.x; k < HW; k += blockDim.x) s += p[k];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        s += __shfl_down_sync(0xffffffffu, s, off);
    __shared__ float sm[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) sm[warp] = s;
    __syncthreads();
    if (threadIdx.x == 0)
    {
        float t = 0.0f;
        const int nw = (blockDim.x + 31) >> 5;
        for (int i = 0; i < nw; i++) t += sm[i];
        out[idx] = t / (float)HW;
    }
}

void gap_launch(const float* x, float* out, int n, int C, int HW)
{
    if (HW <= 0) return;
    gap_kernel<<<n * C, 128, 0, g_stream>>>(x, out, HW);
}

// 激活预量化：fp32 NCHW → int8 [n][c4][S][4]。
__global__ void quant_act_c4_kernel(const float* __restrict__ x,
                                    int8_t* __restrict__ out,
                                    float scale, int qmax,
                                    int c_total, int c_base, int C, int S,
                                    long long total)
{
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int s = (int)(idx % S);
    const long long t = idx / S;
    const int C4 = (C + 3) >> 2;
    const int c4 = (int)(t % C4);
    const int n = (int)(t / C4);
    int qv[4];
#pragma unroll
    for (int l = 0; l < 4; l++)
    {
        const int cc = (c4 << 2) + l;
        if (cc < C)
        {
            const float v = x[((size_t)n * c_total + c_base + cc) * S + s];
            float q = imp_round_away(v / scale);
            q = fminf(fmaxf(q, -(float)(qmax + 1)), (float)qmax);
            qv[l] = (int)q;
        }
        else
            qv[l] = 0;
    }
    uchar4 u;
    u.x = (unsigned char)(int8_t)qv[0];
    u.y = (unsigned char)(int8_t)qv[1];
    u.z = (unsigned char)(int8_t)qv[2];
    u.w = (unsigned char)(int8_t)qv[3];
    *reinterpret_cast<uchar4*>(out + (((size_t)n * C4 + c4) * S + s) * 4) = u;
}

void quant_act_c4_launch(const float* x, int8_t* out, float scale, int qmax,
                         int n, int c_total, int c_base, int c_group,
                         int h, int w)
{
    const int S = h * w;
    const int C4 = (c_group + 3) >> 2;
    const long long total = (long long)n * C4 * S;
    const int grid = (int)((total + 255) / 256);
    quant_act_c4_kernel<<<grid, 256, 0, g_stream>>>(x, out, scale, qmax, c_total,
                                                    c_base, c_group, S, total);
}

} // namespace ptqint8

void ptqint8::implicit_get_debug(unsigned long long* out)
{
    for (int i = 0; i < 8; i++) out[i] = 0;
}
