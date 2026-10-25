# -*- coding: utf-8 -*-
"""导出每层 conv 输入的 (min, max)，供 C++ int8 引擎做静态激活量化。

为什么在 Python 侧做校准：
  C++ 侧要复刻一遍 fp32 卷积的正确性（等于重写 cuDNN），收益为零、
  风险极高。而 fp32 前向在 Python 侧已经是**验证过的**（87.16% 与 ncnn 逐位一致）。
  真正要改的量化语义在 C++（ptq_quant.h），校准只是给它喂 range。

⛔ 校准集必须与训练集同分布、与测试集分开：
  用测试集校准 = 在测试集上调参 = 精度虚高。
  这里用训练集的前 N 张（config.CKPT_MAIN 那个模型的训练分布）。

hooks 的关键点：
  `register_forward_pre_hook` 拿到的输入是**该层 conv 的输入**，
  正是量化器要夹住的位置（quant.py 的 QuantConv2d 就是在 conv 前量化 x）。
  BN 已折叠，所以这里看到的就是量化后要面对的输入分布。
"""
import os
import struct
import sys

import torch
import torch.nn as nn

sys.path.insert(0, "/mnt/d/WorkBuddy/ShowBro/lab/03-ptq-lab")
import config as C
import src.model as M
import src.data as D

MAGIC = b"PTQC"


@torch.no_grad()
def collect(model, loader, n_calib, device):
    convs = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    stats = {n: [3.4e38, -3.4e38] for n, _ in convs}
    handles = []

    def mk(name):
        def hook(mod, inp):
            x = inp[0].detach().float()
            lo = float(x.min())
            hi = float(x.max())
            if lo < stats[name][0]: stats[name][0] = lo
            if hi > stats[name][1]: stats[name][1] = hi
        return hook

    for n, m in convs:
        handles.append(m.register_forward_pre_hook(mk(n)))

    seen = 0
    for images, _labels in loader:
        if seen >= n_calib:
            break
        take = min(images.shape[0], n_calib - seen)
        model(images[:take].to(device))
        seen += take
    for h in handles:
        h.remove()
    return [(stats[n][0], stats[n][1]) for n, _ in convs], seen


def main():
    device = "cuda"
    model = M.fold_all_bn(M.load_fp32()).to(device).eval()

    n_calib = int(os.environ.get("PTQ_CALIB_N", "512"))
    # 用 ptq-lab 现成的校准集工厂：固定种子置换、前 N 张训练集、可复现。
    # ⛔ 它取的是**训练集**（load_datasets 的第一个返回值），不是测试集。
    #    用测试集校准 = 在测试集上调参 = 精度虚高，这个坑不能踩。
    loader = D.make_calib_loader(size=n_calib)

    ranges, seen = collect(model, loader, n_calib, device)
    print("校准样本 %d 张（训练集），conv 层 %d" % (seen, len(ranges)))

    blob = bytearray()
    blob += MAGIC
    blob += struct.pack("<i", len(ranges))
    for lo, hi in ranges:
        print("  %-34s lo=%-12.5g hi=%-12.5g" % ("", lo, hi))
        blob += struct.pack("<2f", lo, hi)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib_minmax.bin")
    with open(out, "wb") as f:
        f.write(bytes(blob))
    print("写出 %s（%.2f KB）" % (out, len(blob) / 1024))


if __name__ == "__main__":
    main()