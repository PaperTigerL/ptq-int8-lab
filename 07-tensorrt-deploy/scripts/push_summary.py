# -*- coding: utf-8 -*-
"""把 bench.jsonl 汇总成 markdown 表（push-20261025）。"""
import json
import sys
from pathlib import Path

BASE_MS = 11.6


def row(r):
    tag = r.get("tag", "")
    acc = r.get("acc")
    ds_min = r.get("dataset_min")
    ds_med = r.get("dataset_median")
    st_med = r.get("steady_median")
    gp = r.get("graph_median")
    g = r.get("graph")
    def f(x, nd=4):
        return "-" if x is None else f"{x:.{nd}f}"
    def xx(x):
        return "-" if not x else f"{BASE_MS / x:.0f}x"
    return "| " + " | ".join([
        tag, str(r.get("batch")), str(r.get("opt")), str(r.get("tactic")),
        f"{r.get('workspace_gb')}", "-" if acc is None else f"{acc:.2f}%",
        f(ds_min), f(ds_med), xx(ds_med), f(st_med), xx(st_med),
        f(gp), "-" if g is None else str(g), f"{r.get('plan_mb')}", f"{r.get('build_s')}",
    ]) + " |"


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/push-20261025/bench.jsonl")
    if not path.exists():
        raise SystemExit(f"缺 {path}")
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip().startswith("{")]
    print(f"# bench.jsonl：{len(recs)} 条")
    hdr = ("| tag | batch | opt | tactic | ws | acc | ds_min | ds_med | x | steady_med | x | "
           "graph_med | graph | planMB | build_s |")
    sep = "|" + "---|" * 15
    print(hdr)
    print(sep)
    for r in sorted(recs, key=lambda x: (x.get("dataset_median") or 9e9)):
        print(row(r))


if __name__ == "__main__":
    main()
