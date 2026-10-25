#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验 2：RTN 量化阶梯 —— 先搞清楚「掉点主要来自谁」。

阶梯设计的关键是 A32 这一档：激活不量化、只量化权重。
    对比 W4A32 与 W4A4，就能把「激活量化单独造成的掉点」摘出来。
    对比 W8A32 与 W4A32，就是「权重量化单独造成的掉点」。
不这么拆，你只会看到一个总数，不知道该往哪使劲。

RTN = Round To Nearest，最朴素的四舍五入，是所有进阶方法的对照基线。
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
import src.quant as Q
from src.data import make_calib_loader


def log(m):
    print(m, flush=True)


def main():
    log("=" * 72)
    log("实验 2 · RTN 量化阶梯（W/A 位宽各档对比）")
    log("=" * 72)
    log("设备     : %s" % C.device_info())
    log("量化算子 : %s" % Q.which_mode())
    log("校准集   : %d 张，seed=%d，策略=minmax" % (C.CALIB_SIZE, C.CALIB_SEED))
    log("")

    torch.manual_seed(C.CALIB_SEED)

    # ---------- 准备：FP32 + BN 折叠 ----------
    m = M.load_fp32(C.CKPT_MAIN).to(C.DEVICE)
    M.fold_all_bn(m)
    r_fp32 = E.evaluate_detail(m)
    log("[0] FP32 基线 : %.2f%%" % (r_fp32["acc"] * 100))
    log("")

    # ---------- 采集激活（只跑一次，后面各档复用） ----------
    log("[1] 采集校准激活 ...")
    t0 = time.time()
    calib_loader = make_calib_loader()
    store = CB.collect_activations(m, calib_loader)
    log("    收集到 %d 层，耗时 %.1fs" % (len(store), time.time() - t0))
    log("")

    # ---------- 各档位跑一遍 ----------
    LADDER = [
        (8, 8), (8, 0),
        (4, 8), (4, 0),
        (4, 4), (3, 8), (3, 4),
    ]
    a_cache = {}
    results = {}

    log("[2] 逐档评估")
    log("    %-10s %10s %12s %10s" % ("配置", "accuracy", "相对FP32", "耗时"))
    log("    " + "-" * 46)

    for wb, ab in LADDER:
        if ab not in a_cache:
            if ab > 0:
                a_cache[ab] = CB.compute_a_params(store, ab, strategy="minmax")
            else:
                a_cache[ab] = ({}, {})
        a_s, a_z = a_cache[ab]

        mq = M.build_quant_model(m, w_bits=wb, a_bits=ab,
                                 a_scales=a_s, a_zeros=a_z).to(C.DEVICE)
        t1 = time.time()
        r = E.evaluate_detail(mq)
        tag = "W%dA%d" % (wb, ab) if ab else "W%dA32" % wb
        results[tag] = r["acc"]
        log("    %-10s %9.2f%% %+11.2f%% %9.1fs" % (
            tag, r["acc"] * 100, (r["acc"] - r_fp32["acc"]) * 100, time.time() - t1))
        del mq
        torch.cuda.empty_cache()

    log("    " + "-" * 46)

    # ---------- 拆解贡献 ----------
    log("")
    log("[3] 掉点归因（这是本实验真正要看的东西）")
    acc = r_fp32["acc"]

    def gap(a, b):
        if a in results and b in results:
            return (results[a] - results[b]) * 100
        return None

    rows = [
        ("权重 8->4，激活不量化", "W4A32", "W8A32"),
        ("权重 8->4，激活 A8  ", "W4A8", "W8A8"),
        ("激活 FP32->A8，权重 W4", "W4A8", "W4A32"),
        ("激活 A8->A4，权重 W4  ", "W4A4", "W4A8"),
        ("权重 4->3，激活 A8   ", "W3A8", "W4A8"),
    ]
    for label, a, b in rows:
        g = gap(a, b)
        if g is None:
            continue
        log("    %-24s %+7.2f%%   (%s vs %s)" % (label, g, a, b))
    log("")
    log("    读法：哪一档掉得最狠，改进预算就押在哪。")

    # ---------- 落盘 ----------
    E.save_result("e2_rtn_ladder", {
        "exp": "e2_rtn_ladder",
        "device": C.DEVICE,
        "quant_mode": Q.which_mode(),
        "calib_size": C.CALIB_SIZE,
        "calib_strategy": "minmax",
        "fp32_acc": acc,
        "results": results,
    })
    log("")
    log("结果已存 results/")
    log("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
