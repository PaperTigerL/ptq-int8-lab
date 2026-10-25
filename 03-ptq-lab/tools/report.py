#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总 results/ 下所有实验的产出，打一张总表。

用法：./run.sh status

每次实验都带时间戳存新文件（保留历史），所以同一个实验会有多个结果。
这里只显示每个实验**最新**的那一份 —— 否则旧的、修 bug 之前的数据
会跟新数据混在一起，看起来像自相矛盾。
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C


def load_latest():
    """按 exp 分组，每组只留文件时间最新的一份。"""
    groups = {}
    for p in glob.glob(os.path.join(C.RESULT_DIR, "*.json")):
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
        except Exception as e:
            d = {"_err": str(e)}
        key = d.get("exp") or os.path.basename(p)
        mt = os.path.getmtime(p)
        if key not in groups or mt > groups[key][0]:
            groups[key] = (mt, os.path.basename(p), d)
    return [(fn, d) for _, fn, d in sorted(groups.values(), key=lambda t: t[1])]


def main():
    items = load_latest()
    print("=" * 72)
    print("实验结果汇总  (results/，每个实验只显示最新一份)")
    print("=" * 72)

    if not items:
        print("还没有任何结果。先跑： ./run.sh e1")
        return 0

    for fname, d in items:
        if "_err" in d:
            print("")
            print("  [%s]  读取失败: %s" % (fname, d["_err"]))
            continue
        print("")
        print("  [%s]   <- %s" % (d.get("exp", "?"), fname))
        print("   设备=%s  算子=%s" % (d.get("device", "?"), d.get("quant_mode", "?")))

        if "fp32_acc" in d:
            print("   FP32 基线      : %.2f%%" % (d["fp32_acc"] * 100))

        if d.get("exp") == "e1_baseline":
            print("   BN 折叠        : %d -> %d, 等价=%s, 差 %.6f%%"
                  % (d.get("bn_before", -1), d.get("bn_after", -1),
                     d.get("fold_equiv_ok"), (d.get("fold_delta") or 0) * 100))
            print("   弱基线(3 epoch): %.2f%%  相差 %.2f%%"
                  % (d.get("weak_acc", 0) * 100, d.get("weak_gap", 0) * 100))

        if d.get("exp") == "e2_rtn_ladder":
            for k, v in sorted(d.get("results", {}).items()):
                print("   %-8s %6.2f%%  (%+.2f%%)"
                      % (k, v * 100, (v - d["fp32_acc"]) * 100))

        if d.get("exp") == "e3_act_calib":
            for k, v in sorted(d.get("results", {}).items(),
                               key=lambda kv: -kv[1]):
                print("   %-12s %6.2f%%  (%+.2f%%)"
                      % (k, v * 100, (v - d["fp32_acc"]) * 100))

        if d.get("exp") == "e4_sensitivity":
            s = d.get("sens", [])
            if s:
                print("   最敏感(不能压): " + ", ".join(
                    "%s %.2f%%" % (x["name"], x["delta_pct"]) for x in s[-3:][::-1]))
                print("   最不敏感(可压): " + ", ".join(
                    "%s %.2f%%" % (x["name"], x["delta_pct"]) for x in s[:3]))
                print("   敏感度极差    : %.2f%%" % d.get("spread_pct", 0))

        if d.get("exp") == "e5_mixed":
            for k, v in d.get("baselines", {}).items():
                print("   %-8s %6.2f%%  (平均 %.1f bit)"
                      % (k, v["acc"] * 100, v["avg_bits"]))
            # ⚠️ 不能直接按精度选：k=20 是全部保 8bit（平均 8.00 bit），
            #    等于没压缩，精度必然最高 —— 选它出来毫无意义。
            #    必须限定「平均位宽 <= MIXED_MAX_AVG_BITS」。
            rows5 = d.get("rows", [])
            cands = [r for r in rows5 if r["avg_bits"] <= C.MIXED_MAX_AVG_BITS]
            best = max(cands, key=lambda r: r["acc"]) if cands else None
            if best:
                print("   最优混合      : 保 %d 层 / 平均 %.2f bit / %.2f%%"
                      % (best["k"], best["avg_bits"], best["acc"] * 100))
            else:
                print("   最优混合      : 无（没有配置的平均位宽 <= %.1f）"
                      % C.MIXED_MAX_AVG_BITS)

        if d.get("exp") == "e6_adaround":
            print("   RTN      W%dA8 : %.2f%%  (%+.2f%%)"
                  % (d.get("bits", 4), d.get("rtn_acc", 0) * 100,
                     (d.get("rtn_acc", 0) - d["fp32_acc"]) * 100))
            print("   AdaRound W%dA8 : %.2f%%  (%+.2f%%)"
                  % (d.get("bits", 4), d.get("adaround_acc", 0) * 100,
                     (d.get("adaround_acc", 0) - d["fp32_acc"]) * 100))
            print("   提升          : %+.2f%%" % (d.get("gain", 0) * 100))

        if d.get("exp") == "e7_combo":
            res = d.get("results", {})
            b0 = res.get("2. W4A8 RTN")
            base = b0["acc"] if isinstance(b0, dict) else None
            print("   组合主表（同一次运行，表内可比）:")
            for k, v in res.items():
                if not isinstance(v, dict):
                    continue
                acc = v["acc"]
                tail = "  (%+.2f%%)" % ((acc - base) * 100) if base else ""
                print("     %-24s %6.2f%%  平均 %.2f bit%s"
                      % (k, acc * 100, v.get("avg_bits", 0), tail))
            if d.get("ladder"):
                print("   Pareto 前沿（low/high -> 精度 @ 平均位宽）:")
                for r in d["ladder"]:
                    print("     W%d/W%d  %6.2f%%  @ %.2f bit"
                          % (r["low"], r["high"], r["acc"] * 100, r["avg_bits"]))
            if d.get("robust"):
                print("   稳健性（换校准 seed）: " + ", ".join(
                    "%d:%.2f%%" % (r["seed"], r["acc"] * 100) for r in d["robust"])
                    + "   极差 %.2f%%" % d.get("spread_pct", 0))

    print("")
    print("-" * 72)
    print("共 %d 个实验有结果（历史版本保留在 results/ 里，未显示）" % len(items))
    return 0


if __name__ == "__main__":
    sys.exit(main())
