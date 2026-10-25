# -*- coding: utf-8 -*-
"""逐层量化敏感度分析。

方法：每次只把一层压到目标位宽，其余层全部保持 FP32，测精度掉多少。
掉得多的 = 硬骨头 = 混合精度里必须保高位宽的层。

注意这是「单独量化」而不是「累积量化」：
  单独量化的好处是各层互不干扰，敏感度可比较、可排序；
  代价是它不等于最终混合精度的真实表现（层间误差会累积）。
  所以 e5 拿到混合配置后必须再实测一次，不能拿敏感度直接推算精度。
"""
import copy

import torch
import torch.nn as nn

import config as C
import src.model as M
import src.eval as E


@torch.no_grad()
def layer_sensitivity(model, loader=None, bits=4, device=None,
                      max_batches=None, base_acc=None):
    """返回 (list of dict, base_acc)。

    每项: {name, acc, delta, delta_pct}
    delta = base_acc - acc  (>=0 表示量化后变差)
    """
    device = device or C.DEVICE
    if base_acc is None:
        base_acc = E.evaluate(model, loader, device, max_batches)

    names = M.conv_layer_names(model)
    out = []
    for i, nm in enumerate(names):
        mq = M.build_quant_model(model, w_bits=0, a_bits=0,
                                 per_layer_w={nm: bits})
        mq = mq.to(device)
        acc = E.evaluate(mq, loader, device, max_batches)
        out.append({
            "name": nm,
            "acc": acc,
            "delta": base_acc - acc,
            "delta_pct": (base_acc - acc) * 100,
        })
        del mq
        if device == "cuda":
            torch.cuda.empty_cache()
    out.sort(key=lambda d: d["delta"])
    return out, base_acc


def split_by_sensitivity(sens, keep_top_k, low_bits=4, high_bits=8):
    """敏感度最高的 keep_top_k 层保 high_bits，其余压到 low_bits。

    sens 已按 delta 升序排好（最不敏感的在最前）。

    注意 Python 切片陷阱：sens[-0:] 等价于 sens[0:]，是「全部元素」而不是「空」。
    不特判的话 keep_top_k=0 会变成全保护，扫描表第一档直接失真。
    """
    if keep_top_k <= 0:
        return {d["name"]: low_bits for d in sens}
    keep_top_k = min(keep_top_k, len(sens))
    protected = {d["name"]: high_bits for d in sens[-keep_top_k:]}
    rest = {d["name"]: low_bits for d in sens[:-keep_top_k]}
    per_layer = {}
    per_layer.update(rest)
    per_layer.update(protected)
    return per_layer


def avg_bits(model, per_layer_w, default=0):
    """按参数量加权算平均位宽 —— 不能简单对层数取平均。

    第一层 conv1 只有 9408 个参数，深层 conv 有 23 万个，
    各占一位算平均会严重高估小层的影响。
    """
    tot_params = 0
    tot_bits = 0
    for nm, mod in model.named_modules():
        if not isinstance(mod, nn.Conv2d):
            continue
        n = mod.weight.numel()
        b = per_layer_w.get(nm, default)
        tot_params += n
        tot_bits += n * b
    return tot_bits / max(tot_params, 1)
