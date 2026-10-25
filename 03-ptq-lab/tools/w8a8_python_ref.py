#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""纯 W8A8 + 指定激活定标策略的 Python 侧精度，用于和 C++ 桥的结果对照。

为什么单独写一个：e7 里的 entropy 是和「混合精度 + AdaRound」混在一起的
（85.45%），拿它当「W8A8 + entropy」的 Python 基线是错的口径。
C++ 桥跑出 entropy 只有 37.63%，必须有一个干净的 Python 数字才能判断
是「桥弄坏了」还是「entropy 在这个模型上本身就不行」。

用法：
    ~/pyenv-torch/bin/python tools/w8a8_python_ref.py --strategy entropy
    ~/pyenv-torch/bin/python tools/w8a8_python_ref.py --strategy minmax
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import config as C
import src.calib as CB
import src.eval as E
import src.model as M
from src.data import make_calib_loader, make_test_loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--w-bits", type=int, default=8)
    ap.add_argument("--a-bits", type=int, default=8)
    ap.add_argument("--strategy", default="minmax",
                    choices=["minmax", "percentile", "mse", "entropy"])
    args = ap.parse_args()

    C.configure(args.dataset, args.arch)

    m = M.load_fp32(C.CKPT_MAIN).to(C.DEVICE)
    M.fold_all_bn(m)
    m.eval()

    fp32 = E.evaluate(m, make_test_loader())
    print("FP32 精度        : %.4f" % fp32, flush=True)

    a_scales, a_zeros, store = CB.calibrate(
        m, make_calib_loader(), args.a_bits, strategy=args.strategy)
    qm = M.build_quant_model(m, w_bits=args.w_bits, a_bits=args.a_bits,
                             a_scales=a_scales, a_zeros=a_zeros).to(C.DEVICE)

    acc = E.evaluate(qm, make_test_loader())
    print("W%dA%d + %-10s: %.4f" % (args.w_bits, args.a_bits, args.strategy, acc),
          flush=True)

    # 激活 scale 的几个样本，便于和 C++ 表里的数值对账
    ks = list(a_scales.keys())[:3]
    for k in ks:
        print("  %-24s a_scale=%.6f a_zero=%.1f"
              % (k, float(a_scales[k]), float(a_zeros[k])), flush=True)


if __name__ == "__main__":
    main()
