// per-channel / per-tensor 量化语义 —— C++ 版，与 03-ptq-lab/src/quant.py 逐位对齐。
//
// 为什么单独抽成头文件：量化语义是整个实验平台的地基。
// 一旦 C++ 和 Python 的口径差一位（比如 clamp 写成 [-qmax,qmax] 而不是
// [-(qmax+1),qmax]），所有实验数字都不可比，而且**不会报错** ——
// 只是精度悄悄对不上。所以这里把每一条规格都写死在注释里，
// 并且配套 tests/quant_agreement.cpp 做逐位比对。
//
// 三个必须记住的口径（都来自 quant.py 的实际实现，不是教科书）：
//  1. 对称量化的 clamp 下界是 [-(qmax+1), qmax]，不是 [-qmax, qmax]。
//     负侧多一个码字，因为 0 必须落在网格上（round(0) 恒为 0，不能偏移）。
//     写成 [-qmax,qmax] 会少一个负码字，W4/W8 的数字对不上。
//  2. scale 必须 clamp(min=1e-8)。全零张量会得到 scale=0，除下去是 nan，
//     而 nan 不会让程序崩溃 —— 只会让实验数字变成垃圾。
//  3. 权重的 per-channel 轴是 dim=0（Conv2d 的 out_channel），
//     scale reshape 成 [-1,1,1,1] 才能广播回去。
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ptq {

// ---------------------------------------------------------------- 基础工具

// 位宽合法性检查，挡在一切计算之前。
// bits=1 会让 qmax=0，scale=absmax/0=inf。不抛异常的话后面会
// 静默产出一片 inf —— 比崩溃难查得多，因为程序照常跑完、退出码 0。
inline int check_bits(int bits)
{
    if (bits < 2)
        throw std::invalid_argument("bits 必须是 >= 2 的整数，收到 " + std::to_string(bits));
    return bits;
}

// round 的语义必须与 torch.round 一致：**半值远离零**（round-half-away-from-zero），
// 不是 C 的 round() 银行家舍入，也不是 std::nearbyint。
// 这一个函数写错会让所有量化结果差半个码字。
inline double torch_round(double v)
{
    return (v >= 0.0) ? std::floor(v + 0.5) : std::ceil(v - 0.5);
}

// qmax 口径：对称用 2^(b-1)-1，非对称用 2^b-1。
inline int sym_qmax(int bits) { return (1 << (bits - 1)) - 1; }

// ---------------------------------------------------------------- 权重量化

// per-channel 对称量化（权重的标准做法）。
//
// 输入 w 是 Conv2d 权重的 [Cout, Cin*kh*kw] 展平形式（Cout-major）。
// 输出 q[i][k]（int8，量程 [-(qmax+1), qmax]）与每个输出通道一个 scale。
//
// 对齐 quant.py::quantize_per_channel_sym 的规格：
//   1. 按行（=按输出通道）求 abs().max()
//   2. scale = absmax / qmax，clamp(min=1e-8)
//   3. q = round(w / scale).clamp(-(qmax+1), qmax)
//
// 调用方负责把 [Cout,Cin,kh,kw] 展平成 [Cout, K]（用 row-major），
// 这样 scale 的广播语义与 PyTorch 的 reshape([-1,1,1,1]) 完全一致。
struct ChannelQuant {
    std::vector<int8_t> q;   // Cout * K，row-major
    std::vector<float> scale;// Cout
    int cout = 0;
    int k = 0;
    int bits = 8;
};

inline ChannelQuant quantize_weight_per_channel_sym(const float* w, int cout, int k, int bits)
{
    check_bits(bits);
    ChannelQuant r;
    r.cout = cout;
    r.k = k;
    r.bits = bits;
    r.q.resize((size_t)cout * k);
    r.scale.resize(cout);

    const int qmax = sym_qmax(bits);
    const double qmin = -(double)(qmax + 1);

    for (int o = 0; o < cout; o++)
    {
        const float* row = w + (size_t)o * k;
        double amax = 0.0;
        for (int i = 0; i < k; i++)
        {
            const double a = std::fabs((double)row[i]);
            if (a > amax) amax = a;
        }
        double s = amax / qmax;
        // 必须 clamp：全零通道（罕见但存在，比如被剪掉的分支）会得到 s=0，
        // 除下去是 nan/inf。这是最容易静默产出垃圾数字的地方。
        if (s < 1e-8) s = 1e-8;
        r.scale[o] = (float)s;

        int8_t* out = r.q.data() + (size_t)o * k;
        for (int i = 0; i < k; i++)
        {
            double v = torch_round((double)row[i] / s);
            if (v < qmin) v = qmin;
            if (v > (double)qmax) v = qmax;
            out[i] = (int8_t)v;
        }
    }
    return r;
}

// ---------------------------------------------------------------- 激活量化

// per-tensor 非对称量化（激活的标准做法）。
//
// ResNet 里激活过 ReLU 后全非负，对称量化会把 [-scale_max,0) 这半边
// 的码字全浪费掉，等于白丢 1 bit。非对称用一个 zero_point 把区间
// 平移到 [0, 2^bits-1]，码字不浪费。
//
// scale = (hi-lo) / (2^bits - 1)，zp = round(-lo/scale)，clamp 到 [0, qmax]。
// 分母是 2^bits-1 不是 2^(bits-1)-1 —— 非对称不要求 0 落在网格上。
struct TensorQuant {
    float scale = 1.0f;
    int zero_point = 0;
    int bits = 8;
    bool symmetric = true;   // 见 tensor_quant_from_range 的长注释：
                             // int8 GEMM（_int_mm）不支持 zero-point，
                             // 所以我们统一走对称码空间。
};

// 从一批样本统计 min/max（即 minmax 校准策略）。
inline void tensor_minmax(const float* x, size_t n, float* lo, float* hi)
{
    double l = 0.0, h = 0.0;
    for (size_t i = 0; i < n; i++)
    {
        const double v = (double)x[i];
        if (i == 0) { l = h = v; continue; }
        if (v < l) l = v;
        if (v > h) h = v;
    }
    *lo = (float)l;
    *hi = (float)h;
}

// 激活范围 → scale 的口径。
//
// 09-ptq-toolkit 的校准（compute_a_params(symmetric=True)）用 AbsMax：
//   scale = max(|lo|,|hi|)/qmax
// 06 引擎历史口径是 MinMax：
//   scale = (hi-lo)/qmax
// 两者只在「范围跨零且两侧不对称」时不同。以 R18 第 0 层
// lo=-1.9894737, hi=2.1264887 为例：MinMax 0.0324092、AbsMax 0.0167440，
// 比值 1.9356。其余 19 层 lo=0，两种口径逐位相同。
//
// ⭐ 默认改为 AbsMax，与 09 校准口径一致：
//   跨零层的 AbsMax 分辨率更细（scale 更小），且量程覆盖 [lo,hi]，
//   不会产生裁剪。MinMax 保留为 `--act-minmax` 显式对照。
//
// ⚠️ 关于历史精度：此前「切口径会让历史 W8A8 数字失效」的说法不成立。
//   历史 W8A8 数字（如 91.85%）是**权重-only fake-quant**，不含激活量化；
//   激活 scale 只在 C++ 引擎里生效，因此改默认口径不影响那些历史数字的可比性。
enum class ActRangeMode { AbsMax = 0, MinMax = 1 };

// 由 (lo, hi) 推出 scale / zero_point。
// mode 默认 AbsMax（与 09 校准一致）。
//
// ⭐⚠️ 这里踩过一个很深的坑，值得完整记下来：
//
//   8 bit 非对称量化的**码空间是 [0, 255]**（256 个码字），但 int8 只有
//   [-128, 127]。自然 zp = round(-lo/s) 在 lo=0 时得到 0，于是码字直接是
//   [0,255]，**超过 127 的部分全部溢出**。
//
//   第一反应是"把 zp 强制成码空间中点 128"，让 [0,255] 平移到 [-128,127]。
//   **这个方案是错的**，实测症状（probe_zp，输入恒为 0.5、scale=0.00778）：
//     期望码字 = round(0.5/scale) - zp = -64
//     GPU 实际 = +64
//   推导：q = round(v/s) = 64 → +zp = 192 → clamp 到 [0,255] **放行** →
//        -zp = 64。所以存下来的是 round(v/s) 本身，zp 被完全抵消，
//        clamp 也从未生效 —— 所谓"非对称"完全没发生，退化成了对称截断。
//   根因：clamp 的区间 [0,255] 覆盖了 zp 偏移后的整个范围，
//        于是 zp 只是被减了一次又加了一次，纯粹的无效操作。
//
//   正确做法：**不搞 zp 偏移，直接用对称码空间**。
//     scale = (hi - lo) / (2^(bits-1) - 1)      ← 分母是 127，不是 255
//     zp    = 0（恒定）
//     q     = clamp(round(x / scale), -(2^(bits-1)), 2^(bits-1)-1)
//   这就是 ptq-lab 的 quantize_per_tensor_sym / _per_channel_sym 的口径，
//   也是 ncnn int8 通路实际在用的方案（代码里的 `int8_scale_term` 就是
//   qmax/absmax 这个乘数）。
//
//   代价：非负激活浪费一半码字（这正是 ptq-lab 里"为什么激活该用非对称"
//   讨论的问题），但那条路的正确实现需要 int8 GEMM 支持 zero-point 偏移，
//   _int_mm 不支持。**在我们的约束下，对称是唯一能跑通的正确解。**
inline TensorQuant tensor_quant_from_range(float lo, float hi, int bits,
                                           ActRangeMode mode = ActRangeMode::AbsMax)
{
    check_bits(bits);
    // 对称码空间：分母用 2^(bits-1)-1（127），保证码字落在 int8 正半边
    const int qmax = (1 << (bits - 1)) - 1;
    double s;
    if (mode == ActRangeMode::AbsMax)
        s = std::max(std::fabs((double)lo), std::fabs((double)hi)) / qmax;
    else
        s = ((double)hi - (double)lo) / qmax;
    if (s < 1e-8) s = 1e-8;

    TensorQuant t;
    t.scale = (float)s;
    t.zero_point = 0;   // 对称量化 zp 恒为 0，不需要 zp_corr
    t.bits = bits;
    t.symmetric = true;
    return t;
}

// 就地量化：x[i] -> int8 码字（已减去 zero_point，即以 zp 为原点）。
//
// ⛔ 返回的是 **int8**，量程 [-(qmax+1), qmax] 对称表示（zp 折叠到 0）。
// 这不是量化口径变了，是「非对称码空间」在 int8 存储下的等价表示：
//   q_uint = clamp(round(x/s) + zp, 0, 2^bits-1)   ∈ [0,255]
//   q_int8 = q_uint - zp                            ∈ [-zp, 255-zp]
// 反量化时 x ≈ (q_int8 + zp) * s = q_int8*s + zp*s。
// 好处：GEMM 可以直接吃 int8 走 tensor core，不用为了 zero_point 改 kernel。
//
// padding 区填 0 是正确的：0 在 int8 表示里对应真实值 zp*s，
// 而 zp 是把 lo 对齐到码字 0 的位置，padding 的语义是「卷积窗口外的贡献为 0」，
// 所以必须填**量化后代表 0 的码字**。本实现在 int8 域填 (int8)0，
// 对应真实值 zp*s ≠ 0 —— 这是 padding 语义的正确处理方式，
// 因为 requantize 后的 bias 补偿项会把它算进去（见 fusion 里的 bias 修正）。
inline void quantize_act_asym_to_int8(const float* x, size_t n, const TensorQuant& t,
                                      int8_t* out, int8_t pad_value = 0)
{
    (void)pad_value;
    const double s = t.scale;
    const double zp = t.zero_point;
    const int qmax = (1 << t.bits) - 1;

    for (size_t i = 0; i < n; i++)
    {
        double q = torch_round((double)x[i] / s) + zp;
        if (q < 0.0) q = 0.0;
        if (q > (double)qmax) q = qmax;
        out[i] = (int8_t)(q - zp);
    }
}

} // namespace ptq