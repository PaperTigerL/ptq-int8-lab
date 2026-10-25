#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断：entropy 校准为什么给 conv1 选出一个只有 amax 10% 的阈值。

背景：把 ptq-lab 的 entropy scale 灌进 ncnn，C++ 侧精度 37.63%；
Python 侧同口径 38.19%，两边一致地烂，所以问题在 ptq-lab 侧。
conv1 的 a_scale=0.001668、a_zero=128，反推出截断阈值 t≈0.213，
而 conv1 输入（归一化图像）的真实范围是 ±2 —— 差了十倍。

本脚本把中间量全部打出来，定位是「分布假设错」还是「约束失效」。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import config as C
import src.model as M
import src.quant as Q
from src.data import make_calib_loader


def main():
    C.configure("cifar10", "resnet18")
    m = M.load_fp32(C.CKPT_MAIN).to(C.DEVICE)
    M.fold_all_bn(m)
    m.eval()

    captured = {}

    def mk(nm):
        def hk(mod, inp, out):
            captured[nm] = inp[0].detach()
        return hk

    hs = []
    for nm, mod in m.named_modules():
        if isinstance(mod, torch.nn.Conv2d):
            hs.append(mod.register_forward_hook(mk(nm)))

    # 只跑一小批，够看分布
    loader = make_calib_loader()
    x0 = next(iter(loader))[0].to(C.DEVICE)
    with torch.no_grad():
        m(x0)
    for h in hs:
        h.remove()

    x = captured["conv1"].float()
    xa = x.abs().reshape(-1)
    amax = float(xa.max())

    print("=== conv1 输入激活（归一化后的图像）===")
    print("  shape          : %s" % (tuple(x.shape),))
    print("  min / max      : %.4f / %.4f" % (float(x.min()), float(x.max())))
    print("  |x| max        : %.4f" % amax)
    print("  |x| 分位       : p50=%.4f p90=%.4f p99=%.4f p99.9=%.4f"
          % (float(torch.quantile(xa, 0.5)), float(torch.quantile(xa, 0.9)),
             float(torch.quantile(xa, 0.99)), float(torch.quantile(xa, 0.999))))

    s = float(Q.scale_entropy(x, 8))
    t = s * 127
    clip_frac = float((xa > t).float().mean())
    print("")
    print("=== scale_entropy 的产出 ===")
    print("  返回 scale s   : %.8f" % s)
    print("  反推阈值 t     : %.6f  (= s * 127)" % t)
    print("  t / amax       : %.4f%%" % (t / amax * 100))
    print("  |x| > t 的比例 : %.4f%%   (约束上限 ENTROPY_MAX_CLIP=%.2f%%)"
          % (clip_frac * 100, C.ENTROPY_MAX_CLIP * 100))
    print("  -> 约束%s" % ("失效/未触发" if clip_frac > C.ENTROPY_MAX_CLIP
                          else "正常触发"))

    # 对照：minmax 口径
    s_mm = float(Q.scale_minmax(x, 8, symmetric=False))
    print("")
    print("=== 对照：minmax（非对称）===")
    print("  scale          : %.8f  zero≈%.1f"
          % (s_mm, round(-float(x.min()) / s_mm)))

    # 再看一个深层，确认是不是只有 conv1 这样
    print("")
    print("=== 各层 entropy vs minmax 的 scale 比值 ===")
    print("  %-24s %10s %10s %8s" % ("层", "entropy", "minmax", "ent/min"))
    for nm in ["conv1", "layer1.0.conv1", "layer1.0.conv2",
               "layer2.0.conv1", "layer4.1.conv2"]:
        if nm not in captured:
            continue
        xx = captured[nm].float()
        se = float(Q.scale_entropy(xx, 8))
        sm = float(Q.scale_minmax(xx, 8, symmetric=False))
        print("  %-24s %10.6f %10.6f %8.3f" % (nm, se, sm, se / sm))


if __name__ == "__main__":
    main()
