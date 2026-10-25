#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 ptq-lab 的量化配置落成 json，供 05-ncnn-deploy 的 make_table.py 消费。

为什么需要这个文件：`QuantConv2d.forward()` 每次前向都现算权重 scale
（`Q.quantize_per_channel_sym(w, bits)`），算完即丢，从不落盘。
要把这份配置搬进 ncnn，必须先把它固化成文件。

产出 json 的结构：
    {
      "meta": {...},                    # 口径，供复现和审计
      "layers": {
        "<pytorch 层名>": {
          "w_scale": [float, ...],      # 逐输出通道，长度 = out_channels
          "a_scale": float,             # per-tensor 激活 scale
          "a_zero":  float              # per-tensor 激活 zero point
        }, ...
      }
    }

⚠️ 覆盖范围：只包含 ptq-lab 真正量化了的层（Conv2d）。
   fc / InnerProduct 不在 ptq-lab 的量化范围里，本文件不产出它的 scale，
   由 make_table.py 明确回退到 KL 基线表并打印告警 —— 不静默混口径。

用法：
    python tools/dump_scales.py                          # 默认 CIFAR-10 + ResNet-18
    python tools/dump_scales.py --strategy entropy       # 换激活定标策略
    python tools/dump_scales.py --dataset cifar100 --arch resnet18
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import config as C
import src.calib as CB
import src.model as M
import src.quant as Q
from src.data import make_calib_loader


def log(m):
    print(m, flush=True)


def weight_scales_per_channel(w, bits):
    """按 `Q.quantize_per_channel_sym` 的同一公式算逐通道权重 scale。

    这里刻意重算而不是从 forward 里钩出来，是为了可验证：
    下面 verify_weight_scales() 会用「重算的 scale 量化结果 == 原函数输出」
    来证明这个公式和线上用的是同一个。公式抄自 src/quant.py:211。
    """
    qmax = 2 ** (bits - 1) - 1
    f = w.reshape(w.shape[0], -1)
    scale = (f.abs().max(dim=1)[0] / qmax).clamp(min=1e-8)
    return scale


def verify_weight_scales(model, bits):
    """自检：重算的 scale 必须和 `Q.quantize_per_channel_sym` 等价。

    这是整个 dump 的可信度基础。如果这个断言挂了，说明公式抄错了，
    此时产出的 json 全是废的 —— 所以它必须在写文件之前跑。
    """
    worst = 0.0
    checked = 0
    for nm, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Conv2d):
            continue
        w = mod.weight.data
        s = weight_scales_per_channel(w, bits)

        # 用重算的 scale 手工量化一遍
        qmax = 2 ** (bits - 1) - 1
        f = w.reshape(w.shape[0], -1)
        q = torch.round(f / s.unsqueeze(1)).clamp(-(qmax + 1), qmax)
        deq = (q * s.unsqueeze(1)).reshape(w.shape)

        ref = Q.quantize_per_channel_sym(w, bits)
        d = float((deq - ref).abs().max())
        worst = max(worst, d)
        checked += 1

    if checked == 0:
        raise RuntimeError("模型里一个 Conv2d 都没找到，检查 load_fp32 是否成功")
    if worst > 1e-6:
        raise RuntimeError(
            "权重 scale 自检失败：重算结果与 Q.quantize_per_channel_sym 最大差 %.3e。"
            "公式与 src/quant.py 不一致，产出的 json 不可用。" % worst)
    log("  自检通过：%d 个 Conv2d，重算 scale 与原函数最大差 %.2e" % (checked, worst))
    return worst


def main():
    ap = argparse.ArgumentParser(
        description="把 ptq-lab 的量化配置（逐通道权重 scale + 激活 scale）落成 json")
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--w-bits", type=int, default=8, help="权位宽（默认 8）")
    ap.add_argument("--a-bits", type=int, default=8, help="激位宽（默认 8）")
    ap.add_argument("--strategy", default="minmax",
                    choices=["minmax", "percentile", "mse", "entropy"],
                    help="激活定标策略（默认 minmax）。ncnn 的 KL 不在其中，"
                         "正是要在 ncnn 里 A/B 的对象")
    ap.add_argument("--percentile", type=float, default=0.999)
    ap.add_argument("--symmetric", action="store_true",
                    help="激活用对称码空间（zp 恒为 0）。给对称-only 的引擎"
                         "（如 ncnn）用；默认非对称。")
    ap.add_argument("--out", default=None,
                    help="输出 json 路径。默认 results/<ds>_<arch>/"
                         "scales_w{wbits}a{abits}_{strategy}[_sym].json")
    args = ap.parse_args()

    C.configure(args.dataset, args.arch)

    log("模型     : %s + %s" % (args.dataset, args.arch))
    log("权重     : %s" % C.CKPT_MAIN)
    log("口径     : W%dA%d，激活定标 %s，%s"
        % (args.w_bits, args.a_bits, args.strategy,
           "对称" if args.symmetric else "非对称"))

    # fold_all_bn 必须做：ONNX 导出时 do_constant_folding=True 也折了 BN，
    # 不折的话算出来的权重 scale 对不上 ncnn 里真正的卷积权重。
    m = M.load_fp32(C.CKPT_MAIN).to(C.DEVICE)
    M.fold_all_bn(m)
    m.eval()

    log("")
    log("[1/3] 权重 scale 自检")
    verify_weight_scales(m, args.w_bits)

    log("[2/3] 激活校准（%s）" % args.strategy)
    a_scales, a_zeros, store = CB.calibrate(
        m, make_calib_loader(), args.a_bits,
        strategy=args.strategy, percentile=args.percentile,
        symmetric=args.symmetric)
    log("  校准集 %d 张，覆盖 %d 层" % (C.CALIB_SIZE, len(store)))

    log("[3/3] 落盘")
    layers = {}
    for nm, mod in m.named_modules():
        if not isinstance(mod, torch.nn.Conv2d):
            continue
        ws = weight_scales_per_channel(mod.weight.data, args.w_bits)
        entry = {
            "w_scale": [float(v) for v in ws.flatten().tolist()],
            "out_channels": int(mod.weight.shape[0]),
            "kernel": list(mod.weight.shape[1:]),
        }
        if nm in a_scales:
            entry["a_scale"] = float(a_scales[nm])
            entry["a_zero"] = float(a_zeros[nm])
        layers[nm] = entry

    if args.out:
        out = args.out
    else:
        rdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "%s_%s" % (args.dataset, args.arch))
        os.makedirs(rdir, exist_ok=True)
        out = os.path.join(rdir, "scales_w%da%d_%s%s.json"
                           % (args.w_bits, args.a_bits, args.strategy,
                              "_sym" if args.symmetric else ""))

    payload = {
        "meta": {
            "dataset": args.dataset,
            "arch": args.arch,
            "w_bits": args.w_bits,
            "a_bits": args.a_bits,
            "act_strategy": args.strategy,
            "a_symmetric": bool(args.symmetric),
            "percentile": args.percentile,
            "calib_size": C.CALIB_SIZE,
            "calib_seed": C.CALIB_SEED,
            "ckpt": C.CKPT_MAIN,
            "bn_folded": True,
            "covered": "Conv2d only（fc/InnerProduct 不在 ptq-lab 量化范围）",
        },
        "layers": layers,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=1)

    log("")
    log("写出 %s" % out)
    log("  %d 层，每层含 w_scale(%d 通道) / a_scale / a_zero"
        % (len(layers), layers[next(iter(layers))]["out_channels"]))
    log("")
    log("下一步：05-ncnn-deploy/scripts/make_table.py --scales %s" % out)


if __name__ == "__main__":
    main()
