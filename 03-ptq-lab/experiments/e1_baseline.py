#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验 1：确认基线，并验证 BN 折叠没有改变模型。

看着像走过场，其实是最重要的一道自检。
BN 折叠是数学等价变换 —— 折叠前后的精度必须完全一致。
如果不一致，说明折叠代码写错了，后面所有量化结论全部作废。
这一步不过，别往下走。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import config as C
import src.model as M
import src.eval as E
import src.quant as Q


def log(m):
    print(m, flush=True)


def main():
    log("=" * 72)
    log("实验 1 · 基线确认 + BN 折叠自检")
    log("=" * 72)
    log("设备     : %s" % C.device_info())
    log("量化算子 : %s" % Q.which_mode())
    log("")

    torch.manual_seed(C.CALIB_SEED)

    # ---------- 1. FP32 主基线 ----------
    log("[1] 加载主基线（ResNet-18 / CIFAR-10 / 50 epoch）")
    m = M.load_fp32(C.CKPT_MAIN).to(C.DEVICE)
    r_fp32 = E.evaluate_detail(m)
    log("    FP32 accuracy : %.2f%%   (%d/%d)   %.1fs   peak %.0fMB" % (
        r_fp32["acc"] * 100, r_fp32["correct"], r_fp32["total"],
        r_fp32["seconds"], r_fp32["peak_mem_mb"]))

    # ---------- 2. BN 折叠 + 等价性自检 ----------
    n_before = M.count_bn(m)
    M.fold_all_bn(m)
    n_after = M.count_bn(m)
    r_fold = E.evaluate_detail(m)
    delta = abs(r_fold["acc"] - r_fp32["acc"])
    ok = delta < 1e-9

    log("")
    log("[2] BN 折叠")
    log("    BN 层数      : %d -> %d" % (n_before, n_after))
    log("    折叠后 accuracy: %.2f%%" % (r_fold["acc"] * 100))
    log("    与折叠前之差 : %+.6f%%" % ((r_fold["acc"] - r_fp32["acc"]) * 100))
    log("    判定         : %s" % (
        "PASS 数学等价，可以继续" if ok else
        "FAIL 折叠改变了模型，必须先修 model.py 再往下走"))

    # ---------- 3. 弱基线对照（说明为什么不用它） ----------
    # ⚠️ 弱基线（3 epoch）只对默认组合存在。换 (数据集 × 模型) 时若没训过弱基线，
    #    这一段跳过即可 —— 它只是"说明为什么不用欠拟合基线"的对照，不影响任何结论。
    log("")
    if not os.path.isfile(C.CKPT_WEAK):
        log("[3] 欠拟合对照：跳过（该组合没有训过 3 epoch 弱基线）")
        log("    这一步只是说明「为什么不用欠拟合模型谈掉点」，不影响任何结论。")
        r_weak = None
    else:
        log("[3] 对照：欠拟合基线（3 epoch）会怎样")
        mw = M.load_fp32(C.CKPT_WEAK).to(C.DEVICE)
        M.fold_all_bn(mw)
        r_weak = E.evaluate_detail(mw)
        log("    弱基线 accuracy: %.2f%%" % (r_weak["acc"] * 100))
        log("    与主基线相差  : %.2f%%  <- 上一轮实验用的就是这个基线" % (
            (r_fp32["acc"] - r_weak["acc"]) * 100))
        log("    结论：在欠拟合模型上谈「量化掉点」没有意义，")
        log("          掉点里混着欠拟合的噪声，分不清是谁造成的。")
        del mw
        torch.cuda.empty_cache()

    # ---------- 落盘 ----------
    payload = {
        "exp": "e1_baseline",
        "device": C.DEVICE,
        "quant_mode": Q.which_mode(),
        "fp32_acc": r_fp32["acc"],
        "fp32_detail": r_fp32,
        "bn_before": n_before,
        "bn_after": n_after,
        "fold_acc": r_fold["acc"],
        "fold_delta": r_fold["acc"] - r_fp32["acc"],
        "fold_equiv_ok": ok,
        # r_weak 可能为 None（该组合没训过 3epoch 弱基线）
        "weak_acc": r_weak["acc"] if r_weak else None,
        "weak_gap": (r_fp32["acc"] - r_weak["acc"]) if r_weak else None,
    }
    path = E.save_result("e1_baseline", payload)
    log("")
    log("结果已存: %s" % path)
    log("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
