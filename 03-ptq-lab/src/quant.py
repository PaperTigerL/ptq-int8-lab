# -*- coding: utf-8 -*-
"""量化算子 —— 本项目的核心。

  2026-10-04：五个函数全部写完并通过验收（性质 15/15 + 边界 42/42），
  并用本实现跑通了全套 e1–e7。**留白阶段结束 —— 本文件现在是可用的实现。**

  五个函数：
    必填 3 个：quantize_per_tensor_sym / quantize_per_channel_sym / quantize_per_tensor_asym
    选填 2 个：scale_mse / scale_entropy（只有 e3 校准策略对比用得到）
    其余给全：BN 折叠、模型包装、评估、扫描 —— 那些是脚手架。

  开关：
    PTQLAB_REFERENCE=1（默认）→ 用 reference/quant_ref.py 顶上，先确认流水线通
    PTQLAB_REFERENCE=0        → 用本文件的实现（现在这个是准的）

  全部量化函数都是 fake quantization：输入输出都是浮点张量，
  中间把值压到整数网格上再还原。这样能在 PyTorch 里直接量精度损失。

  验收：
    cd /mnt/d/WorkBuddy/ShowBro/lab/03-ptq-lab
    PTQLAB_REFERENCE=0 ./run.sh selftest    # 退出码 0 = 全对
    PTQLAB_REFERENCE=0 ./run.sh mine        # 用本实现跑全套 e1–e7

  每个函数上方还留着当初的 TODO 注释 —— 那是「为什么这么写」的记录，
  里面记着几个真实的坑（qmax 是 7 不是 15、minimum 会把上界一起压掉、
  return 写在循环里等于没搜、KL 必须配截断硬约束）。
  **别删**，改这些函数之前先读一遍。
"""
import math
import os
import torch

import config as C
from src._todo import TodoNotImplemented


# ==================================================================
# 零、两个小工具（三个算子都要用，抽出来免得抄三遍）
# ==================================================================

def _check_bits(bits):
    """位宽合法性检查，挡在一切计算之前。

    为什么必须有：bits=1 会让 qmax = 0，于是 scale = absmax / 0 = inf。
    不抛异常的话后面会**静默产出一片 inf** —— 比崩溃难查得多，
    因为程序照常跑完、退出码 0，只是实验数字全是垃圾。
    """
    try:
        b = int(bits)
    except (TypeError, ValueError):
        b = -1
    if b < 2 or b != bits:
        raise ValueError("bits 必须是 >= 2 的整数，收到 %r" % (bits,))
    return b


def _abs_percentile(x, q=0.999):
    """|x| 的 q 分位数。

    用 kthvalue 而不是 sort：校准集动辄百万级元素，
    sort 是 O(n log n) 还要多一份拷贝，kthvalue 只取需要的那一个。
    """
    flat = x.detach().abs().reshape(-1)
    n = flat.numel()
    if n == 0:
        return 0.0
    k = max(1, min(n, int(round(n * q))))
    return float(torch.kthvalue(flat, k, keepdim=False)[0])


def _resample(p, n):
    """把长度 m 的直方图**面积守恒**地重采样成长度 n。

    「面积守恒」是硬要求：重采样前后总和必须一致，
    否则 KL 会凭空变大变小，搜出来的阈值就没有意义了。

    做法：先算前缀和 cdf，则区间 [a, b] 覆盖的量 = cdf(b) - cdf(a)。
    cdf 在整数点之间本来就是线性的（斜率 = 该桶的高度），
    所以对 cdf 做线性插值，等价于「跨越桶边界时按比例切分」。
    这样写没有 Python 内层循环 —— 校准时它要被调用几百次。
    """
    m = int(p.shape[0])
    if m == 0 or n <= 0:
        return torch.zeros(max(n, 0), dtype=torch.float64)
    pf = p.double().reshape(-1)
    if m == n:
        return pf.clone()

    # cdf[k] = 前 k 个桶的累计量；cdf[0] = 0，cdf[m] = 总和
    cdf = torch.cat([torch.zeros(1, dtype=torch.float64),
                     torch.cumsum(pf, 0)])
    edges = torch.linspace(0, m, n + 1, dtype=torch.float64)

    def _cdf_at(x):
        x = x.clamp(0, m)
        lo = torch.floor(x).long()
        hi = torch.minimum(lo + 1, torch.tensor(m, dtype=torch.long))
        frac = x - lo.double()
        return cdf[lo] + (cdf[hi] - cdf[lo]) * frac

    return _cdf_at(edges[1:]) - _cdf_at(edges[:-1])


def _kl(p, q):
    """KL(p || q)。p 是参考分布，q 是量化后的分布。

    两处必须处理：
      · q 里有 0、p > 0 → log(p/0) = inf，要用 eps 兜底。
        ⛔ 不能改成「把 p 那一格丢掉」—— 那等于默认「这一格无所谓」，
        会**系统性低估** KL，于是越激进的截断看起来越好。
      · p 里是 0 的格子不贡献（0 · log(0/q) 按约定取 0）。
    """
    p = p.double().reshape(-1)
    q = q.double().reshape(-1)
    ps, qs = float(p.sum()), float(q.sum())
    if ps <= 0 or qs <= 0:
        return float("inf")
    p = p / ps
    q = q / qs
    m = p > 0
    q = torch.where(q > 0, q, torch.tensor(1e-12, dtype=torch.float64))
    return float((p[m] * torch.log(p[m] / q[m])).sum())


# ==================================================================
# 一、必填：三个量化算子
# ==================================================================

def quantize_per_tensor_sym(x, bits):
    """对称 per-tensor 量化。整张张量共用一个 scale。

    规格（照着写就能过测试）：
      1. scale = x.abs().max() / (2^(bits-1) - 1)，并 clamp 到最小 1e-8
         （不 clamp 的话全零张量会得到 scale=0，除下去就是 nan）
      2. q = round(x / scale)，clamp 到 [-(2^(bits-1)), 2^(bits-1) - 1]
      3. 返回 q * scale

    约束：
      · 返回张量的 dtype 和 device 必须与 x 相同，不要 .cpu()、不要转 float64
      · bits 一律 >= 2

    验收：tests/test_quant.py::test_per_tensor_sym
    """
    # TODO 1：qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    #         bits=4 → (-8, 7)。上界是 7 不是 15 —— 0 必须落在网格上
    #         （round(0) 恒等于 0，不能偏移），所以正方向只剩 7 个码字。
    # TODO 2：scale = (x.abs().max() / qmax).clamp(min=1e-8)
    #         必须 clamp：全零张量会得到 scale=0，除下去全是 nan。
    # TODO 3：q = torch.round(x / scale).clamp(qmin, qmax)      ← 编码
    # TODO 4：return q * scale                                  ← 解码
    #         注意是 q * scale 不是 q —— 这是 fake quantization，
    #         输入输出都是浮点，只是中间压到整数网格上再还原。
    bits = _check_bits(bits)
    # 空张量没有 absmax，x.abs().max() 会直接抛 RuntimeError。
    # 真实流程里会遇到（某分支没激活、某张量被剪空），先原样放回去。
    if x.numel() == 0:
        return x.clone()
    qmax = 2 ** (bits - 1) - 1
    scale = (x.abs().max() / qmax).clamp(min=1e-8)
    q = torch.round(x / scale).clamp(-(qmax + 1), qmax)
    return q * scale


def quantize_per_channel_sym(w, bits, dim=0):
    """对称 per-channel 量化。权重的标准做法。

    为什么权重必须 per-channel：同一个卷积核里，不同输出通道的权重量级
    能差一个数量级。共用一个 scale 的话，量级小的通道会被压成全零。

    规格：
      1. 把 w 展平成 (w.shape[dim], -1)，按行求 abs().max(dim=1) 得到每通道最大值
      2. scale 要 reshape 回可广播的形状：reshape([-1] + [1] * (w.dim() - 1))
         （这样 Conv2d 的 [out, in, kh, kw] 才能按 out 广播）
      3. 同 per_tensor 做 round / clamp / 反量化

    验收：tests/test_quant.py::test_per_channel_sym
    """
    # TODO 1：qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    # TODO 2：把目标轴挪到第 0 维，量化完再挪回去
    #           perm = [dim] + [i for i in range(w.dim()) if i != dim]
    #           moved = w.permute(perm).reshape(w.shape[dim], -1)
    #         想一想：直接写 reshape(w.shape[dim], -1) 为什么只对 dim=0 成立？
    # TODO 3：scale = (moved.abs().max(dim=1).values / qmax).clamp(min=1e-8)
    #         每通道一个 scale。广播回去时末尾要补 1（Conv2d 权重是
    #         [out, in, kh, kw]），形状是 [-1, 1, 1, 1] 而不是 [-1]。
    # TODO 4：q = torch.round(moved / scale.unsqueeze(1)).clamp(qmin, qmax)
    # TODO 5：deq = (q * scale.unsqueeze(1)).reshape(w.permute(perm).shape)
    # TODO 6：逆置换挪回原轴序 —— moved 的第 i 轴来自原张量的第 perm[i] 轴
    #           inv = [0] * w.dim()
    #           for i, p in enumerate(perm): inv[p] = i
    #           return deq.permute(inv)
    #
    # 为什么必须 per-channel：同一个卷积核里不同输出通道的权重量级能差一个
    # 数量级，共用一个 scale 会把量级小的通道压成全零（尺子第二段实测：
    # 通道 0 相对误差 1.000 —— 信息全丢，不是"变小"，是没了）。
    bits = _check_bits(bits)
    if w.numel() == 0:
        return w.clone()
    # perm 是「0 和 dim 互换」的对换，对它做两次就回到原状 ——
    # 所以挪出去和挪回来用的是同一个 perm，不用再求一次逆。
    if dim != 0:
        perm = list(range(w.dim()))
        perm[0], perm[dim] = perm[dim], perm[0]
        wt = w.permute(perm)
    else:
        perm = None
        wt = w

    f = wt.reshape(wt.shape[0], -1)
    qmax = 2 ** (bits - 1) - 1
    scale = (f.abs().max(dim=1, keepdim=True)[0] / qmax).clamp(min=1e-8)
    # 补 1 是为了广播回去：Conv2d 权重是 [out, in, kh, kw]，
    # scale 的形状得是 [-1, 1, 1, 1] 而不是 [-1]
    scale = scale.reshape([-1] + [1] * (wt.dim() - 1))
    q = torch.round(wt / scale).clamp(-(qmax + 1), qmax)
    out = q * scale

    if perm is not None:
        out = out.permute(perm)
    return out


def quantize_per_tensor_asym(x, bits):
    """非对称 per-tensor 量化。激活的标准做法。

    为什么激活要用非对称：ResNet 里激活过 ReLU 之后全是非负数，
    对称量化会把 [-scale_max, 0) 这一半码字浪费掉，等于白丢 1 bit 精度。
    非对称用一个 zero_point 把区间平移到 [lo, hi] 上，码字不浪费。

    规格：
      1. lo = x.min()，hi = x.max()
      2. scale = (hi - lo) / (2^bits - 1)，clamp 到最小 1e-8
      3. zero_point = round(-lo / scale)，clamp 到 [0, 2^bits - 1]
      4. q = round(x / scale) + zero_point，clamp 到 [0, 2^bits - 1]
      5. 返回 (q - zero_point) * scale

    验收：tests/test_quant.py::test_per_tensor_asym
    """
    # TODO 1：qmax = 2 ** bits - 1
    #         注意这里不是 2 ** (bits - 1) - 1 —— 非对称不要求 0 落在网格上，
    #         整段 [lo, hi] 都被用起来了，码字总数从 16 变成"数据区间"。
    # TODO 2：lo, hi = x.min(), x.max()
    #         scale = ((hi - lo) / qmax).clamp(min=1e-8)     ← 分母是 15 不是 7
    # TODO 3：zp = torch.round(-lo / scale).clamp(0, qmax)
    #         zero_point 必须是整数：码字 0 要精确代表 lo 这个值，
    #         区间是被"平移"到 [0, qmax] 上的，不是缩放上去的。
    # TODO 4：q = torch.clamp(torch.round(x / scale) + zp, 0, qmax)
    # TODO 5：return (q - zp) * scale
    #         注意是 (q - zp) * scale 不是 q * scale ——
    #         q 里带着 zp 的偏移，直接乘会整体平移。
    #
    # 为什么激活要用非对称：ResNet 里激活过 ReLU 之后全是非负数，
    # 对称量化会把 [-scale_max, 0) 这一半码字白白浪费掉，等于白丢 1 bit 精度。
    # （尺子第三段实测：码字利用率 8/16 → 16/16，SQNR +6.43 dB）
    bits = _check_bits(bits)
    if x.numel() == 0:
        return x.clone()
    qmax = 2 ** bits - 1
    lo, hi = x.min(), x.max()
    scale = ((hi - lo) / qmax).clamp(min=1e-8)
    zp = torch.round(-lo / scale).clamp(0, qmax)
    q = torch.clamp(torch.round(x / scale) + zp, 0, qmax)
    return (q - zp) * scale


# ==================================================================
# 二、选填：两种更好的激活校准策略（只有 e3 用到）
# ==================================================================

def scale_mse(x, bits, num_candidates=100):
    """用 MSE 搜索最优 scale，替代直接取 min/max。

    为什么需要它：min/max 会被离群点带偏。一张图里只要有一个异常大的
    激活值，scale 就被拉大，剩下 99.9% 的值全挤在几个码字里。
    MSE 搜索是拿「量化前后误差」当目标，直接优化这个。

    规格：
      1. 取候选上界：从 x.abs().max() 等比缩小到它的 0.1 倍，共 num_candidates 个
      2. 对每个候选上界算 scale，做对称量化，算 MSE
      3. 返回 MSE 最小的那个 scale（标量）

    提示：候选可以用 torch.linspace 或 geomspace；
         逐候选算 MSE 时张量不大，直接循环即可，别把循环写成一次前向。

    验收：tests/test_quant.py::test_scale_mse
    """
    # 为什么需要它：min/max 会被离群点带偏。一个异常大的激活值就能把 scale
    # 拉大，剩下 99.9% 的值全挤在几个码字里（尺子第四段实测：加一个 50.0，
    # SQNR 从 14.95 塌到 0.94 dB）。所以要拿"量化前后的误差"当目标去搜。
    #
    # TODO 1：qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    #         amax = float(x.detach().abs().max())
    # TODO 2：造候选上界 —— 从 amax 等比缩到 amax * C.CALIB_SHRINK
    #         ⚠️ 下界还要用 99.9 分位兜底，两个取小者。只用比例的话，
    #            一个 1e6 的离群点会把下界抬到 1e5，搜了等于没搜。
    #         （候选个数用 C.MSE_NUM_CANDIDATES，别在代码里写死）
    # TODO 3：逐候选算 —— s = t / qmax，做对称量化，算 MSE，取最小的那个 s
    #         别把循环写成一次前向，张量不大，直接循环即可。
    # TODO 4：return torch.tensor(最优 s, dtype=x.dtype, device=x.device)
    #         返回的是**标量 scale**，不是量化结果 ——
    #         calib.py 用 t = scale * (2 ** (bits - 1) - 1) 反推截断上界。
    bits = _check_bits(bits)
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    if x.numel() == 0:
        return torch.tensor(1e-8, device=x.device, dtype=x.dtype)
    with torch.no_grad():
        xd = x.detach()
        amax = float(xd.abs().max())
        if amax <= 0:
            return torch.tensor(1e-8, device=x.device, dtype=x.dtype)

        # 搜索下界 = min(amax * shrink, 99.9 分位)，两个取小者。
        # 只用比例的话，一个 1e6 的离群点会把下界抬到 1e5，搜了等于没搜。
        #
        # ⛔ 这里是最容易写错的一处，原来写成：
        #       cands = torch.minimum(cands, torch.tensor(q999))
        #    minimum 是**逐元素取小**，不是「只改下界」——
        #    它会把上界也从 amax 一起压到 q999，搜索区间顶部被砍掉。
        #    实测 amax=50 / q99.9=3.31 时，100 个候选的范围变成
        #    [3.3142, 3.3142] —— 全部相等，搜索一次都没发生。
        #    ⚠️ 而这种情况下测试仍然 PASS：那个张量带了离群点，
        #    q99.9 恰好就是好答案。过测试靠的是输入选得好，不是机制在跑。
        lo = min(amax * C.CALIB_SHRINK, _abs_percentile(xd))
        lo = max(lo, amax * 1e-6)          # 兜底，别取到 0 让 log10 炸掉
        # 等比取候选：低位宽时小的一端也需要分辨率，等比比等距合适。
        # ⚠️ device 必须跟 x 一致 —— linspace 默认建在 CPU 上，
        #    不指定 device 的话 CUDA 张量一除就报
        #    "Expected all tensors to be on the same device"。
        cands = torch.logspace(math.log10(amax), math.log10(lo),
                               steps=int(num_candidates), device=xd.device)

        best_e, best_s = None, None
        for c in cands:
            scale = (c / qmax).clamp(min=1e-8)
            q = torch.round(xd / scale).clamp(qmin, qmax)
            mse = (q * scale - xd).pow(2).mean().item()
            if best_e is None or mse < best_e:
                best_e, best_s = mse, float(scale)

        # ⛔ return 必须在 for 外面。写在循环体里的话第一次迭代就返回了，
        #    等于只试了最大的那个候选（就是 minmax），搜索形同虚设。
        return torch.tensor(best_s, dtype=x.dtype, device=x.device)


def scale_entropy(x, bits, num_bins=2048, num_candidates=100):
    """用 KL 散度选 scale（TensorRT 的 entropy calibration）。

    思路和 MSE 不同：MSE 优化的是数值误差，KL 优化的是「量化后的分布
    与原分布的差异」。TensorRT 实践里 KL 通常比 MSE 更稳。

    规格：
      1. 对 x 建直方图（num_bins 个桶），得到参考分布 P
      2. 对每个候选上界：把超出上界的值折叠进最后一个桶，
         再把 [0, 上界] 重新分桶成 2^(bits-1) 份，得到量化分布 Q
      3. 算 KL(P || Q)，取最小的候选对应的 scale

    验收：tests/test_quant.py::test_scale_entropy
    """
    # 和 MSE 的区别：MSE 优化"数值误差"，KL 优化"分布形状的差"。
    #
    # TODO 1：n_levels = qmax + 1   （正半边的量化 level 数：bits=4 → 8 个）
    #         xa = x.detach().abs().reshape(-1).float().cpu()
    #         amax = float(xa.max())
    # TODO 2：建参考分布 P —— 在 [0, amax] 上建 C.ENTROPY_NUM_BINS 个桶后归一化
    #         hist = torch.histc(xa, bins=num_bins, min=0, max=amax); P = hist / 总
    # TODO 3：造候选上界（同 scale_mse，下界用 99.9 分位兜底）
    # TODO 4：逐候选 ——
    #           a. 候选上界落在第几个桶：nb = round(t / amax * num_bins)
    #           b. 折叠：ref = P[:nb].clone(); ref[-1] += P[nb:].sum()
    #              ⚠️ 必须折叠。不折叠的话超出上界的值凭空消失，
    #                 Q 会漏掉一整块尾部，KL 算出来偏小，搜出的 scale 不准。
    #           c. 重新分桶成 n_levels 份得 Q
    #           d. 把 Q 摊回桶粒度得 Qe，算 KL(ref || Qe)
    # TODO 5：⚠️ 硬约束 —— 截断掉的比例不能超过 C.ENTROPY_MAX_CLIP
    #          if float((xa > t).float().mean()) > max_clip: continue
    #          为什么必须有：KL 衡量的是"分布形状差多少"不是"数值差多少"，
    #          纯 KL 最小会选出掐掉 1.7% 激活的上界，
    #          实测 ResNet-18 的 A4 整体精度会从 86% 崩到 66%。
    #          全超限则退回 amax / qmax（就是 minmax），安全降级。
    # TODO 6：return torch.tensor(最优 s, dtype=x.dtype, device=x.device)
    #         同样是**标量 scale**，不是量化结果。
    bits = _check_bits(bits)
    qmax = 2 ** (bits - 1) - 1
    n_levels = qmax + 1                 # bits=4 → 正半边 8 个 level
    if x.numel() == 0:
        return torch.tensor(1e-8, device=x.device, dtype=x.dtype)
    with torch.no_grad():
        # 直方图要建在 CPU 上：histc 在 CUDA 上要求张量连续，
        # 而且这里要跑几百次，搬一次比每次都在 GPU 上折腾划算。
        xa = x.detach().abs().reshape(-1).float().cpu()
        amax = float(xa.max())
        if amax <= 0:
            return torch.tensor(1e-8, device=x.device, dtype=x.dtype)

        # 参考分布 P：在 [0, amax] 上建 num_bins 个桶后归一化
        hist = torch.histc(xa, bins=num_bins, min=0, max=amax).double()
        P = hist / hist.sum()

        # 候选上界：等比，下界同 scale_mse 用 min(比例, 99.9 分位) 兜底。
        # ⛔ 同样是「只压下界」，不能对整个数组做 minimum（原因见 scale_mse）。
        lo = min(amax * C.CALIB_SHRINK, _abs_percentile(x))
        lo = max(lo, amax * 1e-6)
        cands = torch.logspace(math.log10(amax), math.log10(lo),
                               steps=int(num_candidates))

        best_e, best_s = None, None
        max_clip = C.ENTROPY_MAX_CLIP
        for t in cands:
            tc = float(t)
            # ⚠️ 硬约束，不是可选优化：KL 衡量的是「分布形状差多少」
            #    不是「数值差多少」。纯 KL 最小会选出掐掉 1.7% 激活的上界，
            #    实测 ResNet-18 的 A4 整体精度从 86% 崩到 66%。
            if float((xa > tc).float().mean()) > max_clip:
                continue
            nb = int(round(tc / amax * num_bins))
            if nb < n_levels:            # 桶数不够分成 n_levels 份，跳过
                continue

            # 折叠尾部：超出上界的值全塞进最后一个桶。
            # ⛔ 必须折叠 —— 不折叠的话那部分质量凭空消失，
            #    Q 会漏掉一整块尾部，KL 算出来偏小，搜出的 scale 不准。
            ref = P[:nb].clone()
            ref[-1] += P[nb:].sum()

            # 量化分布 Q：先把 ref 压成 n_levels 份（这一步就是「量化」），
            # 再摊回 nb 个桶的粒度（「反量化」），才能跟 ref 逐格比。
            # ⛔ 原来是 histc(xa, bins=n_levels) 再 _resample 到 num_bins ——
            #    摊回的是 2048 而不是 nb，跟 ref 长度对不上，KL 是错的。
            Qe = _resample(_resample(ref, n_levels), nb)

            e = _kl(ref, Qe)
            if best_e is None or e < best_e:
                best_e, best_s = e, tc / qmax

        # 全部超限就退回 amax / qmax（就是 minmax），安全降级
        if best_s is None:
            best_s = amax / qmax
        return torch.tensor(best_s, dtype=x.dtype, device=x.device)


# ==================================================================
# 三、给全：校准 helper 与误差度量
# ==================================================================

def scale_minmax(x, bits, symmetric=True):
    """最朴素的 scale：直接取 min/max。作为所有进阶策略的对照基线。"""
    if symmetric:
        return (x.abs().max() / (2 ** (bits - 1) - 1)).clamp(min=1e-8)
    lo, hi = x.min(), x.max()
    return ((hi - lo) / (2 ** bits - 1)).clamp(min=1e-8)


def scale_percentile(x, bits, p=0.999):
    """按分位数截断后再取 scale。比 min/max 抗离群点，比 MSE 快得多。

    p=0.999 表示只丢掉最大的 0.1%。这是工程上最常用的折中。
    """
    flat = x.abs().reshape(-1)
    k = max(1, int(flat.numel() * p))
    # torch.kthvalue 比 sort 省内存；只在需要的分位上取
    val = torch.kthvalue(flat, k, keepdim=False)[0]
    return (val / (2 ** (bits - 1) - 1)).clamp(min=1e-8)


def quant_error(x, xq):
    """量化误差的三种度量。SQNR 比裸 MSE 好读：单位是 dB，越大越好。"""
    with torch.no_grad():
        diff = (xq - x).detach()
        mse = diff.pow(2).mean().item()
        power = x.detach().pow(2).mean().item()
        sqnr = 10.0 * torch.log10(torch.tensor(power / max(mse, 1e-12))).item()
        return {"mse": mse, "sqnr": sqnr,
                "max_abs_err": diff.abs().max().item()}


def count_unique_ints(xq, scale):
    """量化后落在多少个不同的整数码字上。用来判断「码字有没有被浪费」。

    比如 W4 理论上最多 16 个码字，如果实测只用了 4 个，
    说明 scale 定得太大，精度白白丢了。
    """
    with torch.no_grad():
        q = torch.round(xq / max(float(scale), 1e-12))
        return int(q.unique().numel())


# ==================================================================
# 四、参考实现覆盖（PTQLAB_REFERENCE=1 时生效）
# ==================================================================

_TODO_NAMES = [
    "quantize_per_tensor_sym",
    "quantize_per_channel_sym",
    "quantize_per_tensor_asym",
    "scale_mse",
    "scale_entropy",
]


def _load_reference():
    import importlib.util
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "reference", "quant_ref.py")
    if not os.path.exists(p):
        raise FileNotFoundError("参考实现不存在: " + p)
    spec = importlib.util.spec_from_file_location("quant_ref", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if C.USE_REFERENCE:
    _ref = _load_reference()
    for _n in _TODO_NAMES:
        globals()[_n] = getattr(_ref, _n)
    REFERENCE_MODE = True
else:
    REFERENCE_MODE = False


def which_mode():
    return "参考实现" if REFERENCE_MODE else "你的实现"
