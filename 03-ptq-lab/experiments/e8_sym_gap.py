#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验 8：对称 vs 非对称 —— 丢掉 zero_point 到底值多少？

⚠️ 这是**决策探针**，不是 e1–e7 那条教学序列的延续。

它的唯一目的是回答一个工程决策问题：

    ncnn 的 int8 通路只支持对称码（±127、无 zero_point，
    见 src/layer/quantize.cpp 的 float2int8）。我们要给它加 sub-8bit，
    那么 —— 要不要顺带支持非对称激活？

答案取决于「丢 zp 的代价」随位宽怎么变。本实验把它量出来。

背景（决定了本实验为什么非做不可）：
    ptq-lab 侧算出的激活量化参数一直是非对称的（calib.py 里
    qmax = 2^bits - 1，带 zero_point）。所以 e2 阶梯里那个
    W4A4 = 83.90% 是「引擎支持非对称」前提下的数字。
    ncnn 只跑对称，那个数字拿不到 —— 差多少，没人量过。

两层设计（先快后慢）：
    sqnr（默认，分钟级）：不跑推理，直接对每层激活算两种码空间的 SQNR 差。
    e2e （小时级）      ：跑端到端精度，拿能写进论文的数字。

判据（先定死，避免事后挑数字）：
    4bit 下 asym − sym 的端到端精度差
        > 1.00pp  → 支持非对称，走创新路线（专利只覆盖 8bit）
        < 0.33pp  → 只做对称 int4，省掉 padding / eltwise / requantize 三处改造
    0.33pp = CIFAR-10 测试集 10000 张的标准误，低于它等于没差别。
"""
import argparse
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


# ------------------------------------------------------------------ 量化
# 这两个函数复现 QuantConv2d.forward 的码空间语义，保证探针和真实模型一致。
# 改动 forward 时这里必须同步改，否则探针测的东西就不是模型跑的东西。

def _quant_asym(x, bits):
    """非对称：码字 [0, 2^bits - 1]，zero_point 平移整段 [lo, hi]。"""
    qmax = 2 ** bits - 1
    lo, hi = float(x.min()), float(x.max())
    scale = max((hi - lo) / qmax, 1e-8)
    zp = int(round(-lo / scale))
    zp = max(0, min(qmax, zp))
    q = (torch.round(x / scale) + zp).clamp(0, qmax)
    return (q - zp) * scale, scale, zp


def _quant_sym(x, bits):
    """对称：码字 [-(2^(bits-1)), 2^(bits-1) - 1]，0 必须落在网格上。"""
    qmax = 2 ** (bits - 1) - 1
    qmin = -(qmax + 1)
    lo, hi = float(x.min()), float(x.max())
    t = max(abs(lo), abs(hi))          # 区间被钉死在 [-t, +t]
    scale = max(t / qmax, 1e-8)
    q = torch.round(x / scale).clamp(qmin, qmax)
    return q * scale, scale, 0


def _sqnr(x, xh):
    """信号量化噪声比（dB）。越大越好。"""
    sig = torch.mean(x.float() ** 2)
    err = torch.mean((x.float() - xh.float()) ** 2)
    if err <= 0:
        return float("inf")
    return float(10 * torch.log10(sig / err))


def _code_usage(lo, hi):
    """对称码空间下激活实际占用的比例。

    50% 表示负半轴那一半码字全浪费 —— 也就是白丢 1 bit。
    100% 表示激活关于 0 对称，非对称相对对称没有优势。
    """
    t = max(abs(lo), abs(hi))
    if t <= 0:
        return 1.0
    return (hi - lo) / (2.0 * t)


# ------------------------------------------------------------------ 第一层

def mode_sqnr(args, store):
    """不跑推理，直接量每层激活在两种码空间下的 SQNR 差。"""
    bits_list = [int(b) for b in args.bits.split(",")]
    log("=" * 78)
    log("实验 8 · 第一层：SQNR 判据（asym vs sym，逐层）")
    log("=" * 78)
    log("设备     : %s" % C.device_info())
    log("量化算子 : %s" % Q.which_mode())
    log("校准集   : %d 张，seed=%d，策略=minmax" % (C.CALIB_SIZE, C.CALIB_SEED))
    log("位宽     : %s" % bits_list)
    log("")

    summary = {}
    for bits in bits_list:
        rows = []
        for nm in sorted(store.keys()):
            x = store[nm]
            lo, hi = float(x.min()), float(x.max())

            xa, sa, za = _quant_asym(x, bits)
            xs, ss, _zs = _quant_sym(x, bits)

            rows.append({
                "layer": nm,
                "lo": lo,
                "hi": hi,
                "usage": _code_usage(lo, hi),
                "sqnr_asym": _sqnr(x, xa),
                "sqnr_sym": _sqnr(x, xs),
                "delta_db": _sqnr(x, xa) - _sqnr(x, xs),
            })

        deltas = [r["delta_db"] for r in rows]
        usages = [r["usage"] for r in rows]
        deltas_sorted = sorted(deltas)
        med = deltas_sorted[len(deltas_sorted) // 2]

        summary[bits] = {
            "n_layers": len(rows),
            "delta_mean_db": sum(deltas) / len(deltas),
            "delta_median_db": med,
            "delta_max_db": max(deltas),
            "delta_min_db": min(deltas),
            "usage_mean": sum(usages) / len(usages),
            "usage_min": min(usages),
        }

        log("── A%d ──────────────────────────────────────────────" % bits)
        log("    %-22s %10s %10s %9s %8s" % (
            "层", "SQNR_asym", "SQNR_sym", "差值dB", "码字利用"))
        # 按「最吃亏」排序：差值最大的层就是最该支持非对称的层
        for r in sorted(rows, key=lambda r: -r["delta_db"])[:8]:
            log("    %-22s %9.2f %10.2f %+9.2f %7.1f%%" % (
                r["layer"][:22], r["sqnr_asym"], r["sqnr_sym"],
                r["delta_db"], r["usage"] * 100))
        log("    %-22s %9s %10s %+9.2f %7s" % (
            "…（共 %d 层）" % len(rows), "", "",
            summary[bits]["delta_mean_db"], ""))
        log("    均值 %+.2f dB / 中位 %+.2f dB / 最大 %+.2f dB / 最小 %+.2f dB"
            % (summary[bits]["delta_mean_db"], summary[bits]["delta_median_db"],
               summary[bits]["delta_max_db"], summary[bits]["delta_min_db"]))
        log("    码字利用率 均值 %.1f%% / 最低 %.1f%%"
            % (summary[bits]["usage_mean"] * 100,
               summary[bits]["usage_min"] * 100))
        log("")

    log("=" * 78)
    log("汇总：丢 zero_point 的代价（dB，越大越该支持非对称）")
    log("    %-6s %10s %10s %10s %10s" % ("位宽", "均值", "中位", "最大", "码字利用"))
    log("    " + "-" * 50)
    for bits in bits_list:
        s = summary[bits]
        log("    A%-5d %+9.2f %+10.2f %+10.2f %9.1f%%" % (
            bits, s["delta_mean_db"], s["delta_median_db"],
            s["delta_max_db"], s["usage_mean"] * 100))
    log("")
    log("    读法：")
    log("      · 差值 ≈ 6 dB 是理论上限（码字正好少一半）。")
    log("      · 差值 < 6 dB 说明该层激活不是单边的（lo 明显 < 0），")
    log("        非对称优势小 —— 这类层用对称就够了。")
    log("      · 码字利用率 50% = 负半轴全浪费 = 白丢 1 bit。")

    E.save_result("e8_sym_gap_sqnr", {
        "exp": "e8_sym_gap_sqnr",
        "device": C.DEVICE,
        "quant_mode": Q.which_mode(),
        "calib_size": C.CALIB_SIZE,
        "bits": bits_list,
        "summary": {str(k): v for k, v in summary.items()},
    })
    log("")
    log("结果已存 results/")
    return 0


# ------------------------------------------------------------------ 第二层

def mode_e2e(args, store, m_fp32):
    """跑端到端精度。只在 SQNR 趋势明显时才需要这一层。"""
    combos = []
    for pair in args.combos.split(","):
        wb, ab = pair.split(":")
        combos.append((int(wb), int(ab)))

    log("=" * 78)
    log("实验 8 · 第二层：端到端精度（asym vs sym）")
    log("=" * 78)
    log("设备     : %s" % C.device_info())
    log("量化算子 : %s" % Q.which_mode())
    log("配置     : %s" % ["W%dA%d" % c for c in combos])
    log("")

    r_fp32 = E.evaluate_detail(m_fp32)
    fp32_acc = r_fp32["acc"]
    log("[0] FP32 基线 : %.2f%%" % (fp32_acc * 100))
    log("")

    a_cache = {}
    results = {}
    log("[1] 逐配置评估")
    log("    %-10s %12s %12s %10s %8s" % (
        "配置", "asym(非对称)", "sym(对称)", "差值pp", "耗时"))
    log("    " + "-" * 58)

    for wb, ab in combos:
        row = {}
        for sym in (False, True):
            key = (ab, sym)
            if key not in a_cache:
                a_cache[key] = CB.compute_a_params(
                    store, ab, strategy="minmax", symmetric=sym)
            a_s, a_z = a_cache[key]

            mq = M.build_quant_model(m_fp32, w_bits=wb, a_bits=ab,
                                     a_scales=a_s, a_zeros=a_z,
                                     a_symmetric=sym).to(C.DEVICE)
            t1 = time.time()
            r = E.evaluate_detail(mq)
            row["asym" if not sym else "sym"] = r["acc"]
            log("    %-10s %11.2f%% %11s %9s %7.1fs" % (
                "W%dA%d-%s" % (wb, ab, "asym" if not sym else "sym"),
                r["acc"] * 100, "", "", time.time() - t1))
            del mq
            torch.cuda.empty_cache()

        gap = (row["asym"] - row["sym"]) * 100
        tag = "W%dA%d" % (wb, ab)
        results[tag] = {
            "asym": row["asym"], "sym": row["sym"], "gap_pp": gap,
        }
        log("    %-10s %11.2f%% %11.2f%% %+9.2f %8s" % (
            tag, row["asym"] * 100, row["sym"] * 100, gap, ""))
        log("    " + "-" * 58)

    log("")
    log("[2] 判据")
    log("    4bit 下 asym − sym 的差：")
    for tag in ("W4A4", "W8A4"):
        if tag not in results:
            continue
        g = results[tag]["gap_pp"]
        verdict = ("支持非对称（走创新路线）" if g > 1.00
                   else "只做对称（省三处改造）" if g < 0.33
                   else "灰色地带，需补证据")
        log("      %-6s %+7.2f pp  →  %s" % (tag, g, verdict))
    log("")
    log("    阈值：> 1.00pp 支持；< 0.33pp（CIFAR-10 标准误）放弃。")

    E.save_result("e8_sym_gap_e2e", {
        "exp": "e8_sym_gap_e2e",
        "device": C.DEVICE,
        "quant_mode": Q.which_mode(),
        "calib_size": C.CALIB_SIZE,
        "fp32_acc": fp32_acc,
        "results": results,
    })
    log("")
    log("结果已存 results/")
    return 0


def main():
    ap = argparse.ArgumentParser(description="实验 8：对称 vs 非对称判据")
    ap.add_argument("--mode", choices=["sqnr", "e2e"], default="sqnr",
                    help="sqnr=分钟级逐层信噪比（默认）；e2e=端到端精度")
    ap.add_argument("--bits", default="8,6,4,3,2",
                    help="sqnr 模式扫的激活位宽，逗号分隔")
    ap.add_argument("--combos", default="8:8,4:8,4:4",
                    help="e2e 模式跑的配置，格式 w:a，逗号分隔")
    args = ap.parse_args()

    torch.manual_seed(C.CALIB_SEED)

    m = M.load_fp32(C.CKPT_MAIN).to(C.DEVICE)
    M.fold_all_bn(m)

    log("[准备] 采集校准激活 ...")
    t0 = time.time()
    store = CB.collect_activations(m, make_calib_loader())
    log("       收集到 %d 层，耗时 %.1fs" % (len(store), time.time() - t0))
    log("")

    if args.mode == "sqnr":
        return mode_sqnr(args, store)
    return mode_e2e(args, store, m)


if __name__ == "__main__":
    sys.exit(main())
