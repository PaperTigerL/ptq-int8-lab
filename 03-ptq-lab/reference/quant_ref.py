# -*- coding: utf-8 -*-
"""量化算子的参考实现。

只在 PTQLAB_REFERENCE=1（默认）时被 src/quant.py 加载覆盖。

建议用法：自己填完 TODO 之后，再来对照一遍这个文件 ——
重点看边界处理（clamp、广播形状、dtype 保持），这些是新手最容易漏的。
"""
import torch

_EPS = 1e-12


def _sym_qmax(bits):
    """对称量化的整数上界，顺带把非法位宽挡在门口。

    bits=1 会让 qmax=0，于是 scale = absmax / 0 = inf ——
    不抛异常的话会**静默产生一堆 inf**，比崩溃难查得多。
    """
    if not isinstance(bits, int) or bits < 2:
        raise ValueError("bits 必须是 >= 2 的整数，收到 %r" % (bits,))
    return 2 ** (bits - 1) - 1


def quantize_per_tensor_sym(x, bits):
    qmax = _sym_qmax(bits)
    # 空张量没有 absmax，x.abs().max() 会直接抛 RuntimeError。
    if x.numel() == 0:
        return x.clone()
    scale = (x.abs().max() / qmax).clamp(min=1e-8)
    q = torch.round(x / scale).clamp(-(qmax + 1), qmax)
    return q * scale


def quantize_per_channel_sym(w, bits, dim=0):
    qmax = _sym_qmax(bits)
    if w.numel() == 0:
        return w.clone()
    if dim != 0:
        perm = list(range(w.dim()))
        perm[0], perm[dim] = perm[dim], perm[0]
        wt = w.permute(perm)
    else:
        perm = None
        wt = w

    f = wt.reshape(wt.shape[0], -1)
    scale = (f.abs().max(dim=1, keepdim=True)[0] / qmax).clamp(min=1e-8)
    scale = scale.reshape([-1] + [1] * (wt.dim() - 1))
    q = torch.round(wt / scale).clamp(-(qmax + 1), qmax)
    out = q * scale

    if perm is not None:
        out = out.permute(perm)
    return out


def quantize_per_tensor_asym(x, bits):
    if not isinstance(bits, int) or bits < 2:
        raise ValueError("bits 必须是 >= 2 的整数，收到 %r" % (bits,))
    if x.numel() == 0:
        return x.clone()
    lo, hi = x.min(), x.max()
    qmax = 2 ** bits - 1
    scale = ((hi - lo) / qmax).clamp(min=1e-8)
    zp = torch.round(-lo / scale).clamp(0, qmax)
    q = (torch.round(x / scale) + zp).clamp(0, qmax)
    return (q - zp) * scale


def scale_mse(x, bits, num_candidates=100):
    with torch.no_grad():
        qmax = _sym_qmax(bits)
        if x.numel() == 0:
            return torch.tensor(1e-8, device=x.device, dtype=x.dtype)
        amax = float(x.detach().abs().max())
        if amax <= 0:
            return torch.tensor(1e-8, device=x.device, dtype=x.dtype)

        cands = torch.linspace(amax, amax * 0.1, num_candidates)
        best_e, best_s = None, None
        xd = x.detach()
        for c in cands:
            s = torch.tensor(float(c) / qmax, device=x.device, dtype=x.dtype).clamp(min=1e-8)
            q = torch.round(xd / s).clamp(-(qmax + 1), qmax)
            e = ((q * s - xd) ** 2).mean()
            if best_e is None or float(e) < best_e:
                best_e, best_s = float(e), s
        return best_s


def _resample(p, n):
    """把长度 m 的直方图面积守恒地重采样成长度 n。"""
    m = int(p.shape[0])
    idx = torch.linspace(0, m, n + 1, dtype=torch.float64)
    out = torch.zeros(n, dtype=torch.float64)
    pf = p.double()
    for j in range(n):
        a, b = float(idx[j]), float(idx[j + 1])
        ia = int(a)
        ib = int(b)
        s = 0.0
        if ib > ia:
            s += float(pf[min(ia, m - 1)]) * (min(ia + 1, b) - a)
            for k in range(ia + 1, min(ib, m)):
                s += float(pf[k])
            if ib < m:
                s += float(pf[ib]) * (b - ib)
        else:
            s += float(pf[min(ia, m - 1)]) * (b - a)
        out[j] = s
    return out


def _kl(p, q):
    assert p.shape == q.shape, "KL 形状不一致: %s vs %s" % (
        tuple(p.shape), tuple(q.shape))
    p = p.double()
    q = q.double()
    ps, qs = p.sum(), q.sum()
    if ps <= 0 or qs <= 0:
        return float("inf")
    p = p / ps
    q = q / qs
    m = p > 0
    q = torch.where(q > 0, q, torch.tensor(_EPS, dtype=torch.float64))
    return float((p[m] * torch.log(p[m] / q[m])).sum())


def scale_entropy(x, bits, num_bins=2048, num_candidates=100):
    """TensorRT 的 entropy calibration：选让 KL(P||Q) 最小的截断阈值。"""
    with torch.no_grad():
        qmax = _sym_qmax(bits)
        n_q = qmax + 1
        xa = x.detach().abs().float().cpu()
        # 排除精确零。ReLU 之后激活里有大量精确 0，它们会在第 0 个桶
        # 堆成一根巨柱，让 KL 误判「阈值压到很小也没关系」，
        # 结果九成激活被截断 —— 实测 A4 精度会崩到 67%。
        nz = xa[xa > 0]
        if nz.numel() == 0:
            return torch.tensor(1e-8, device=x.device, dtype=x.dtype)
        amax = float(nz.max())
        if amax <= 0:
            return torch.tensor(1e-8, device=x.device, dtype=x.dtype)

        hist = torch.histc(nz, bins=num_bins, min=0, max=amax).double()
        hist = hist / (hist.sum() + _EPS)

        cands = torch.linspace(amax, amax * 0.1, num_candidates)
        best_kl, best_c = None, amax
        for c in cands:
            cf = float(c)
            i = max(1, int(round(cf / amax * num_bins)))
            i = min(i, num_bins)
            sliced = hist[:i].clone()
            tail = hist[i:].sum()
            sliced[-1] = sliced[-1] + tail
            # 先合并到 n_q 桶（这就是量化后的分布），再上采样回 i 桶，
            # 这样 P 与 Q 同长度，KL 才有意义。
            q_small = _resample(sliced, n_q)
            q = _resample(q_small, int(sliced.shape[0]))
            kl = _kl(sliced, q)
            if best_kl is None or kl < best_kl:
                best_kl, best_c = kl, cf

        return torch.tensor(best_c / qmax, device=x.device,
                            dtype=x.dtype).clamp(min=1e-8)
