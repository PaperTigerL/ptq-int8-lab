// Implicit GEMM (direct convolution) 接口 —— 不落 im2col 的 W8A8 路径。
//
// 旧路径：fp32 -> quant+im2col (M,Kpad) int8 落显存 -> at::_int_mm -> int32
// 新路径：conv kernel 的 shared tile 里直接量化输入 + 生成 A 片段，DP4A 累加
//         到 int32 (M,Cout)，再走同一个 requant。im2col 矩阵从不落显存。
//
// 权重布局（每个 group 独立）：
//   w_imp[((tap * C4 + c4) * cout + co) * 4 + lane]
//     tap = i*kw + j（卷积窗口内位置，tap-major）
//     c4  = c/4（输入通道按 4 对齐；cin 不是 4 的倍数时尾部补 0）
//     lane= c%4
// 这样一次 32-bit load 就是 DP4A 需要的 4 个连续输入通道，
// 同一 k4 下 32 个输出通道的权重也是连续的 128B（合并读）。
#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

namespace ptqint8 {

inline size_t implicit_weight_bytes(int cout, int cin, int kh, int kw)
{
    const int c4 = (cin + 3) / 4;
    return (size_t)kh * kw * c4 * cout * 4;
}

// q 是现有 per-channel 对称量化结果，布局 [cout][cin*kh*kw]（Cout-major）。
// 仅支持普通卷积（groups=1）；分组/depthwise 继续走旧路径。
inline void repack_implicit_weight(const int8_t* q, int cout, int cin,
                                   int kh, int kw, std::vector<int8_t>* out)
{
    const int taps = kh * kw;
    const int c4 = (cin + 3) / 4;
    out->assign((size_t)taps * c4 * cout * 4, 0);
    for (int o = 0; o < cout; o++)
        for (int c = 0; c < cin; c++)
            for (int t = 0; t < taps; t++)
            {
                const int k = c * taps + t;   // 与 ConvSpec.weight 的 [cin][kh][kw] 一致
                (*out)[(((size_t)(t * c4 + (c >> 2)) * cout + o) << 2) + (c & 3)] =
                    q[(size_t)o * cin * taps + k];
            }
}

// 预量化：fp32 NCHW -> int8 [n][c4][H*W][4]（每个 group 单独一块）。
// 一次全局 pass 替代每个 block 重复量化同一张输入图（grid.y 个输出通道块会
// 重复读同一输入，旧实现把这份开销乘了 2~16 倍）。
void quant_act_c4_launch(const float* x, int8_t* out, float scale, int qmax,
                         int n, int c_total, int c_base, int c_group,
                         int h, int w);

void implicit_get_debug(unsigned long long* out);

// qact: quant_act_c4_launch 的输出（单个 group）。
// w_imp: 上面 repack 出来的布局（单个 group 的基址）。
// acc: (M=n*OH*OW, cout) int32 行主序，与 at::_int_mm 的输出逐元素一致。
bool implicit_conv_launch(const int8_t* qact, const int8_t* w_imp, int32_t* acc,
                          int n, int c_group, int h, int w,
                          int kh, int kw, int sh, int sw, int ph, int pw,
                          int oh, int ow, int cout, bool allow_direct = true);

// 全局平均池化（在 GPU 上，避免 n*C*HW 的两次 memcpy + 主机同步）。
// 输出 out[n*C] = mean over HW。R18-CIFAR 末端 HW==1，此时与旧的主机实现
// 逐位相同（求和是恒等映射）；HW>1 时求和顺序与主机版不同，末位可能差 1 ulp。
void gap_launch(const float* x, float* out, int n, int C, int HW);

// 融合 epilogue：int32 累加后直接做 per-channel 反量化 + bias + zp 修正 + ReLU，
// 写 fp32 NCHW（cout_stride/co_base 处理分组写回）。中间不落 int32 acc，
// 也不启动单独的 requant kernel。内部用 shared-memory 转置保证 NCHW 写合并。
// 形状不支持或 shared 超限时返回 false，调用方回退旧的 acc+requant 路径。
bool implicit_conv_out_launch(const int8_t* qact, const int8_t* w_imp, float* out,
                              const float* wscale, const float* bias,
                              const float* zp_corr, float ascale,
                              int n, int c_group, int h, int w,
                              int kh, int kw, int sh, int sw, int ph, int pw,
                              int oh, int ow, int cout,
                              int cout_stride, int co_base, bool relu,
                              bool allow_direct = true);

} // namespace ptqint8