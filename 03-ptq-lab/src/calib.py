# -*- coding: utf-8 -*-
"""激活校准：用校准集统计每层激活范围，固化为 a_scale / a_zero。

这是「静态量化」的核心步骤。上一轮的做法是在每次 forward 里现算 min/max，
那叫动态量化 —— 部署时芯片不会这么干，而且那样做各种校准策略根本比不出差别。
"""
import torch
import torch.nn as nn

import config as C
import src.quant as Q


def _quantile(x, q):
    """取分位数。用 kthvalue 而不是 sort，省一次全量排序。"""
    f = x.reshape(-1)
    n = f.numel()
    k = max(1, min(n, int(round(n * q))))
    return float(torch.kthvalue(f, k, keepdim=False)[0])


@torch.no_grad()
def collect_activations(model, loader, samples_per_layer=200000, device=None):
    """跑一遍校准集，收集每个 Conv2d 的输入激活（降采样后保存）。

    为什么不整存：ResNet-18 深层激活是 [B,512,8,8]，1024 张图就是 268MB/层，
    二十多层一起存会直接把 15GB 内存吃光。
    降采样到每层 20 万个值，对统计量（min/max/分位/直方图）影响可以忽略。
    """
    device = device or C.DEVICE
    store = {}
    hooks = []

    for nm, mod in model.named_modules():
        if not isinstance(mod, nn.Conv2d):
            continue

        def mk(nm_):
            def hk(m, inp, out):
                x = inp[0]
                if not isinstance(x, torch.Tensor):
                    return
                f = x.detach().reshape(-1)
                if f.numel() > samples_per_layer:
                    # ⚠️ 索引必须在 CPU 上生成。CUDA 的随机数不受 torch.manual_seed
                    # 控制 —— 用它采样的话，同一个 CALIB_SEED 两次跑出来的校准集
                    # 是不一样的，实验无法复现。实测后果：e3 的 entropy 档在两次
                    # 运行里一个 86.46%、一个 67.66%，差 19 个点。
                    idx = torch.randint(0, f.numel(), (samples_per_layer,),
                                        device="cpu").to(f.device)
                    f = f[idx]
                f = f.float().cpu()
                cur = store.get(nm_)
                if cur is None:
                    store[nm_] = f
                else:
                    cat = torch.cat([cur, f])
                    if cat.numel() > samples_per_layer:
                        idx = torch.randperm(cat.numel())[:samples_per_layer]
                        cat = cat[idx]
                    store[nm_] = cat
            return hk

        hooks.append(mod.register_forward_hook(mk(nm_=nm)))

    model.eval()
    for x, _ in loader:
        model(x.to(device, non_blocking=True))
    for h in hooks:
        h.remove()

    return store


def compute_a_params(store, a_bits, strategy="minmax", percentile=0.999,
                     symmetric=False):
    """按策略算每层的激活量化参数。

    minmax      : 直接取 min/max。对照基线，最容易被离群点带偏。
    percentile  : 掐掉两端各 (1-p) 后再取。工程上最常用的折中。
    mse         : 搜索让量化误差最小的截断（调你的 Q.scale_mse）。
    entropy     : 搜索让 KL 散度最小的截断（调你的 Q.scale_entropy）。

    symmetric=False（默认）：非对称。qmax = 2^bits - 1，带 zero_point，
        整段 [lo, hi] 铺满全部码字。这是本平台的既有行为，也是
        TFLite/LiteRT 对激活的标准做法。

    symmetric=True：对称。0 必须落在网格上，区间被钉死在 [-t, +t]，
        t = max(|lo|, |hi|)，zero_point 恒为 0。
        ⚠️ 这正是 ncnn int8 通路的能力边界 —— src/layer/quantize.cpp 的
        float2int8() 把码字截在 ±127 且没有 zero_point。
        当激活过 ReLU（lo≈0、hi=H）时，负半轴那一半码字全部浪费，
        等效于白丢 1 bit。本实验就是要把这个代价量化出来。
    """
    if a_bits <= 0:
        return {}, {}

    s_out, z_out = {}, {}
    qmax_asym = 2 ** a_bits - 1             # 非对称码字上界
    qmax_sym = 2 ** (a_bits - 1) - 1        # 对称正半轴上界（0 必须落在网格上）

    for nm, x in store.items():
        lo, hi = float(x.min()), float(x.max())

        if strategy == "percentile":
            lo = _quantile(x, 1.0 - percentile)
            hi = _quantile(x, percentile)
        elif strategy == "mse":
            t = float(Q.scale_mse(x, a_bits)) * (2 ** (a_bits - 1) - 1)
            lo = max(lo, -t)
            hi = min(hi, t)
        elif strategy == "entropy":
            t = float(Q.scale_entropy(x, a_bits)) * (2 ** (a_bits - 1) - 1)
            lo = max(lo, -t)
            hi = min(hi, t)

        if symmetric:
            # 对称：区间必须是 [-t, +t]，t 取两端绝对值的较大者。
            # 于是 [lo, hi] 只用掉了 [-t, +t] 的一部分 —— 少掉的那部分
            # 就是相对非对称浪费掉的码字。
            t = max(abs(lo), abs(hi))
            scale = t / qmax_sym
            scale = max(scale, 1e-8)
            zp = 0.0
        else:
            scale = (hi - lo) / qmax_asym
            scale = max(scale, 1e-8)
            zp = int(round(-lo / scale))
            zp = max(0, min(qmax_asym, zp))

        s_out[nm] = torch.tensor(scale, dtype=torch.float32)
        z_out[nm] = torch.tensor(float(zp), dtype=torch.float32)

    return s_out, z_out


def calibrate(model, loader, a_bits, strategy="minmax", percentile=0.999,
              device=None, symmetric=False):
    """一步到位：收集激活 → 算参数。返回 (a_scales, a_zeros, store)。

    symmetric 透传给 compute_a_params，决定用的是非对称码空间
    （[0, 2^bits-1] 带 zp）还是对称码空间（[-(2^(b-1)-1), 2^(b-1)-1]）。
    """
    store = collect_activations(model, loader, device=device)
    s, z = compute_a_params(store, a_bits, strategy=strategy,
                            percentile=percentile, symmetric=symmetric)
    return s, z, store
