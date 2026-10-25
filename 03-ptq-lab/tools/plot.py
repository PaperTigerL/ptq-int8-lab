#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把实验结果画成图（论文用）。

用法：./run.sh plot   或   python tools/plot.py

图里一律用**英文标签**：WSL 里没装中文字体，写中文会渲染成一堆方框。
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C

FIGDIR = os.path.join(C.RESULT_DIR, "figs")

# 实验脚本里的 tag 是中文（给人读的），画图时换成英文。
TAG_EN = {
    "1. W8A8 RTN": "1. W8A8 RTN",
    "2. W4A8 RTN": "2. W4A8 RTN",
    "3. W4A8 + AdaRound": "3. W4A8 + AdaRound",
    "4. 混合 W4/W8 RTN": "4. Mixed W4/W8 RTN",
    "5. 混合 + AdaRound": "5. Mixed + AdaRound",
    "6. 混合+AdaRound+A4 entropy": "6. Mixed+AdaRound+A4",
}

# 手工指定每个标签的偏移，避免顶部密集区文字互相压住。
# (dx, dy) 是点的像素偏移；None 表示这个点不标注（和别的点几乎重合）。
LABEL_OFFSET = {
    "1. W8A8 RTN": (7, -15),
    "2. W4A8 RTN": (7, 5),
    "3. W4A8 + AdaRound": (7, -15),
    "4. 混合 W4/W8 RTN": None,
    "5. 混合 + AdaRound": (7, 7),
    "6. 混合+AdaRound+A4 entropy": (7, -15),
    "W8A8": (6, -14),
    "W4A8": (6, 5),
    "W4A4": (6, 5),
    "W3A8": (6, 5),
    "W3A4": (6, 5),
    "mixed 8/3": (7, 5),
    "mixed 6/4": (7, -15),
    "mixed 8/6": (7, 5),
}


def latest(exp):
    """取该实验最新的一份结果。"""
    best = None
    for p in glob.glob(os.path.join(C.RESULT_DIR, "*.json")):
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        if d.get("exp") != exp:
            continue
        mt = os.path.getmtime(p)
        if best is None or mt > best[0]:
            best = (mt, d)
    return best[1] if best else None


def fig_bitwidth_curve(e7, e2, path):
    """精度 vs 平均权重位宽。论文最核心的一张图。"""
    pts = []
    if e7:
        for tag, v in e7.get("results", {}).items():
            pts.append((0, v["avg_bits"], v["acc"] * 100, tag, "combo"))
        for L in e7.get("ladder", []):
            pts.append((1, L["avg_bits"], L["acc"] * 100,
                        "mixed %d/%d" % (L["high"], L["low"]), "ladder"))
    if e2:
        for k, v in e2.get("results", {}).items():
            if k.endswith("A32"):
                continue
            b = 8.0 if k.startswith("W8") else (4.0 if k.startswith("W4") else 3.0)
            pts.append((2, b, v * 100, k, "rtn"))

    if not pts:
        print("  [skip] fig1 没有数据")
        return False

    fp32 = (e7 or e2 or {}).get("fp32_acc", 0) * 100

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    styles = {"rtn": ("o", "#9E9E9E"), "combo": ("s", "#C62828"),
              "ladder": ("^", "#1565C0")}
    seen_lab = set()
    plotted = []

    # 顺序画：ours 优先占位，ladder 次之，uniform RTN 最后。
    # e7 的 ladder (4,8) 与「混合 W4/W8」是同一个配置，坐标几乎重合，去重。
    for pri, x, y, tag, kind in sorted(pts, key=lambda t: (t[0], t[1])):
        if any(abs(x - px) < 0.05 and abs(y - py) < 0.35 for px, py in plotted):
            continue
        plotted.append((x, y))
        mk, col = styles[kind]
        lab = {"rtn": "Uniform RTN", "combo": "Ours (combined)",
               "ladder": "Mixed-precision ladder"}.get(kind)
        if lab in seen_lab:
            lab = None
        else:
            seen_lab.add(lab)
        ax.scatter(x, y, marker=mk, s=70, color=col, label=lab, zorder=3)
        off = LABEL_OFFSET.get(tag, (6, 4))
        if off is not None:
            ax.annotate(TAG_EN.get(tag, tag), (x, y),
                        textcoords="offset points", xytext=off,
                        fontsize=8, color="#444")

    if fp32:
        ax.axhline(fp32, ls="--", lw=1, color="#2E7D32")
        ax.text(0.015, fp32 + 0.08, "FP32 %.2f%%" % fp32,
                transform=ax.get_yaxis_transform(),
                fontsize=9, color="#2E7D32")

    ax.set_xlabel("Average weight bits (parameter-count weighted)")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("ResNet-18 / CIFAR-10 : accuracy vs average weight bit-width")
    ax.grid(alpha=0.25, ls=":")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(min(p[2] for p in pts) - 2.5, max(fp32, max(p[2] for p in pts)) + 1.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def fig_sensitivity(e4, e5, path):
    """逐层敏感度条形图。"""
    sens = None
    for d in (e4, e5):
        if d and d.get("sens"):
            sens = d["sens"]
            break
    if not sens:
        print("  [skip] fig2 没有敏感度数据（先跑 e4 或 e5）")
        return False

    names = [s["name"] for s in sens]
    vals = [s["delta_pct"] for s in sens]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#C62828" if v > 0.3 else ("#1565C0" if v > 0 else "#9E9E9E")
              for v in vals]
    ax.bar(range(len(names)), vals, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_ylabel("Accuracy drop when this layer alone is quantized to W4 (%)")
    ax.set_title("Per-layer quantization sensitivity (higher = must protect)")
    ax.grid(axis="y", alpha=0.25, ls=":")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    print("=" * 60)
    print("生成图表 -> %s" % FIGDIR)
    print("=" * 60)

    e2 = latest("e2_rtn_ladder")
    e4 = latest("e4_sensitivity")
    e5 = latest("e5_mixed")
    e7 = latest("e7_combo")

    ok = 0
    p1 = os.path.join(FIGDIR, "fig1_acc_vs_bits.png")
    if fig_bitwidth_curve(e7, e2, p1):
        print("  [ok] fig1_acc_vs_bits.png")
        ok += 1
    p2 = os.path.join(FIGDIR, "fig2_layer_sensitivity.png")
    if fig_sensitivity(e4, e5, p2):
        print("  [ok] fig2_layer_sensitivity.png")
        ok += 1

    print("")
    print("生成 %d 张图" % ok)
    if ok == 0:
        print("没有可用数据 —— 先跑 ./run.sh all")
    return 0


if __name__ == "__main__":
    sys.exit(main())
