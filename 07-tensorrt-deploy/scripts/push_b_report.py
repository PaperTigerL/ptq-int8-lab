# -*- coding: utf-8 -*-
"""push-20261025b：位置分解的 A/B 汇总（同场对照，别跨场引用绝对值）。

产出三张表：
  1) 主 A/B 表：同场基线与各档 dataset_median / steady_median / 精度
  2) 逐层 A/B：被分解的卷积在 base / decomp 两种图里各自的 ms（同场、同 batch）
  3) MAC 账：每个被改写的卷积的 MAC，用来判断「省下的算力」和「省下的时间」是否脱钩

用法:
    python push_b_report.py                       # 默认 results/push-20261025b/bench.jsonl
    python push_b_report.py --pairs int8_c1x1_base:int8_decall ...
"""
import argparse
import json
from pathlib import Path

BASE_MS = 11.6
# 被 push_decompose.py 改写的卷积（与 bench profile 名字里能对上的子串）
TARGETS = [
    "/layer3/layer3.0/conv2/Conv",
    "/layer3/layer3.1/conv1/Conv",
    "/layer3/layer3.1/conv2/Conv",
    "/layer4/layer4.0/conv1/Conv",
]


def load(path):
    recs = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip().startswith("{")]
    return {r.get("tag"): r for r in recs}


def fmt(x, nd=4):
    return "-" if x is None else f"{x:.{nd}f}"


def main_table(recs):
    print("| tag | batch | opt | dio | acc | ds_min | ds_med | x | steady_med | x | planMB | build_s |")
    print("|" + "---|" * 12)
    for tag, r in sorted(recs.items(), key=lambda kv: (kv[1].get("dataset_median") or 9e9)):
        dm = r.get("dataset_median")
        sm = r.get("steady_median")
        print("| " + " | ".join([
            tag, str(r.get("batch")), str(r.get("opt")), "Y" if r.get("direct_io") else "-",
            "-" if r.get("acc") is None else f"{r['acc']:.2f}%",
            fmt(r.get("dataset_min")), fmt(dm), f"{BASE_MS/dm:.0f}x" if dm else "-",
            fmt(sm), f"{BASE_MS/sm:.0f}x" if sm else "-",
            fmt(r.get("plan_mb"), 1), fmt(r.get("build_s"), 0),
        ]) + " |")


def layer_ms(profile, sub):
    """profile 里所有名字包含 sub 的层时间之和（融合后的层名会带 + 后缀）。"""
    return sum(v for k, v in profile if sub in k)


def layer_pair(base, var, label):
    pb, pv = base.get("profile"), var.get("profile")
    if not pb or not pv:
        print(f"（{label}：缺 profile）")
        return
    print(f"\n### {label} 逐层（ms/batch，batch={base.get('batch')}）\n")
    print("| 卷积 | base | decomp | Δ | 比值 | base 占比 |")
    print("|---|---|---|---|---|---|")
    tb = sum(v for _, v in pb)
    tot_b = tot_v = 0.0
    for t in TARGETS:
        b = layer_ms(pb, t)
        v = layer_ms(pv, t.replace("/Conv", "/Conv/decomp_conv")) + layer_ms(pv, t.replace("/Conv", "/Conv/decomp_d2s"))
        if b == 0 and v == 0:
            continue
        tot_b += b
        tot_v += v
        print(f"| {t.split('/Conv')[0]} | {b:.3f} | {v:.3f} | {v-b:+.3f} | {b/v if v else float('nan'):.2f}x | "
              f"{100*b/tb:.1f}% |")
    print(f"| **合计** | **{tot_b:.3f}** | **{tot_v:.3f}** | **{tot_v-tot_b:+.3f}** | "
          f"**{tot_b/tot_v if tot_v else float('nan'):.2f}x** | |")
    print(f"\n整机 profile 合计 {tb:.3f} -> {sum(v for _, v in pv):.3f} ms/batch；"
          f"dataset {base.get('dataset_median')} -> {var.get('dataset_median')}，"
          f"steady {base.get('steady_median')} -> {var.get('steady_median')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="results/push-20261025b/bench.jsonl")
    ap.add_argument("--pairs", nargs="*", default=[
        "fp16_c1x1_o5dio_base:fp16_decall_o5dio",
        "int8_c1x1_base:int8_decall",
    ])
    args = ap.parse_args()
    recs = load(args.jsonl)
    print(f"# push-20261025b A/B（{args.jsonl}，{len(recs)} 档）\n")
    main_table(recs)
    for p in args.pairs:
        b, v = p.split(":")
        if b in recs and v in recs:
            layer_pair(recs[b], recs[v], f"{b} -> {v}")
        else:
            print(f"\n（缺 {p}）")


if __name__ == "__main__":
    main()
