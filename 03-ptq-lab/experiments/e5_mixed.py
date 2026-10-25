#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验 5：混合精度 —— 本项目要拿到的第一个「提升」。

核心主张：全 W4A8 = 85.28%，但 20 层不是平等的。
把软柿子压到更低位、硬骨头保 8bit，可以在**平均位宽更低**的前提下
拿到比统一 W4 更高的精度。这就是可量化的提升。

评价口径必须用「平均位宽 vs 精度」，而且位宽要按参数量加权（见 sens.avg_bits）。
只看层数平均是自欺 —— conv1 只有 9408 个参数，深层有 23 万个，权重完全不同。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import config as C
import src.model as M
import src.eval as E
import src.calib as CB
import src.sens as S
import src.quant as Q
from src.data import make_calib_loader


def log(m):
    print(m, flush=True)


def main():
    log("=" * 72)
    log("实验 5 · 混合精度（敏感度驱动的位宽分配）")
    log("=" * 72)
    log("设备     : %s" % C.device_info())
    log("量化算子 : %s" % Q.which_mode())
    log("")

    torch.manual_seed(C.CALIB_SEED)
    m = M.load_fp32(C.CKPT_MAIN).to(C.DEVICE)
    M.fold_all_bn(m)
    base = E.evaluate(m)
    log("[0] FP32 基线 : %.2f%%" % (base * 100))

    # ---------- 激活校准（固定 A8，因为 e2 证明 A8 几乎免费） ----------
    log("")
    log("[1] 激活校准（A8, minmax）—— e2 已证明 A8 只掉 0.05%，固定用它")
    store = CB.collect_activations(m, make_calib_loader())
    a_s, a_z = CB.compute_a_params(store, 8, strategy="minmax")
    log("    完成，%d 层" % len(a_s))

    # ---------- 敏感度扫描 ----------
    log("")
    log("[2] 逐层敏感度扫描（复用 e4 的方法）...")
    t0 = time.time()
    sens, _ = S.layer_sensitivity(m, bits=4, base_acc=base)
    log("    完成，耗时 %.0fs" % (time.time() - t0))

    # ---------- 两个统一位宽基线 ----------
    log("")
    log("[3] 基线：统一位宽")
    baselines = {}
    for wb in (8, 4):
        mq = M.build_quant_model(m, w_bits=wb, a_bits=8,
                                 a_scales=a_s, a_zeros=a_z).to(C.DEVICE)
        acc = E.evaluate(mq)
        baselines["W%dA8" % wb] = {"acc": acc,
                                   "avg_bits": float(wb)}
        log("    W%dA8  %6.2f%%   (平均权重位宽 %.1f)" % (wb, acc * 100, wb))
        del mq
        torch.cuda.empty_cache()

    # ---------- 混合精度扫描 ----------
    log("")
    log("[4] 混合精度：保住最敏感的前 k 层用 8bit，其余压到 4bit")
    log("    %-8s %10s %12s %14s" % ("k(保8)", "accuracy", "vs W4A8", "平均位宽"))
    log("    " + "-" * 48)

    rows = []
    for k in (0, 2, 4, 6, 8, 10, 12, 20):
        per_layer = S.split_by_sensitivity(sens, keep_top_k=k,
                                           low_bits=4, high_bits=8)
        ab = S.avg_bits(m, per_layer)
        mq = M.build_quant_model(m, w_bits=4, a_bits=8,
                                 per_layer_w=per_layer,
                                 a_scales=a_s, a_zeros=a_z).to(C.DEVICE)
        acc = E.evaluate(mq)
        rows.append({"k": k, "acc": acc, "avg_bits": ab,
                     "gain_vs_w4": acc - baselines["W4A8"]["acc"]})
        log("    %-8d %9.2f%% %+11.2f%% %13.2f" % (
            k, acc * 100, (acc - baselines["W4A8"]["acc"]) * 100, ab))
        del mq
        torch.cuda.empty_cache()
    log("    " + "-" * 48)

    # ---------- 找最优 ----------
    log("")
    log("[5] 结论")
    w4 = baselines["W4A8"]
    w8 = baselines["W8A8"]

    # ⚠️ 不能直接按「精度最高」选：k=20 是全部保 8bit（平均 8.00 bit），
    #    等于根本没压缩，精度自然逼近 FP32 —— 把它选成"最优"毫无意义。
    #    必须在「平均位宽 <= MIXED_MAX_AVG_BITS」的约束下比精度。
    cands = [r for r in rows if r["avg_bits"] <= C.MIXED_MAX_AVG_BITS]
    if not cands:
        best = None
        lo = min(rows, key=lambda r: r["avg_bits"])
        log("    ⚠️ 没有任何配置满足「平均权重位宽 <= %.1f」—— 混合精度在本配置下不成立。"
            % C.MIXED_MAX_AVG_BITS)
        log("       扫描表里位宽最低的是 k=%d / 平均 %.2f bit / 精度 %.2f%%"
            % (lo["k"], lo["avg_bits"], lo["acc"] * 100))
    else:
        best = max(cands, key=lambda r: r["acc"])
        log("    最优混合配置 : 保 %d 层 8bit，平均权重位宽 %.2f，精度 %.2f%%"
            % (best["k"], best["avg_bits"], best["acc"] * 100))
        log("    对比统一 W4A8: %.2f%% (平均 4.00 bit)  -> %+.2f%%"
            % (w4["acc"] * 100, (best["acc"] - w4["acc"]) * 100))
        log("    对比统一 W8A8: %.2f%% (平均 8.00 bit)  -> %+.2f%%"
            % (w8["acc"] * 100, (best["acc"] - w8["acc"]) * 100))
        log("    （k=20 那种全保 8bit 精度更高，但它没有压缩，不算数。）")

    if best is not None:
        b5 = best
        log("")
        log("    【关键判据】平均位宽 <= 5.0 的最佳配置：")
        log("      保 %d 层 / 平均 %.2f bit / 精度 %.2f%%" %
            (b5["k"], b5["avg_bits"], b5["acc"] * 100))
        log("      相对统一 W4A8 提升 %+.2f%%" % (b5["gain_vs_w4"] * 100))
        if b5["gain_vs_w4"] > 0:
            log("      -> 混合精度有效：同样甚至更低的位宽，精度更高。")
        else:
            log("      -> 混合精度在本配置下没捞到收益，需要换思路（见手册第 6 节）。")

    E.save_result("e5_mixed", {
        "exp": "e5_mixed",
        "device": C.DEVICE,
        "quant_mode": Q.which_mode(),
        "fp32_acc": base,
        "baselines": baselines,
        "rows": rows,
        "sens": sens,
    })
    log("")
    log("结果已存 results/")
    log("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
