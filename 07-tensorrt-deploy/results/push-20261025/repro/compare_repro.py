# -*- coding: utf-8 -*-
"""复现对账：把 repro/bench.jsonl 的数字与 07 原轮同配置档比一遍。

判据（写死在文件里，避免“事后看数字再定标准”）：
    延迟  dataset_median / steady_median 偏差 <= 3%
    精度  |Δ| <= 0.5 pp
基准（07 原轮 bench.jsonl 的 int8_c1x1_pb 档）：
    dataset 0.00744、steady 0.00790、精度 87.22%
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]                      # 07-tensorrt-deploy/
ORIG = REPO / "results" / "push-20261025" / "bench.jsonl"
REPRO = HERE / "bench.jsonl"

BASE = dict(dataset=0.00744, steady=0.00790, acc=87.22)
TAG = "int8_c1x1_pb"
TOL_LAT = 0.03
TOL_ACC = 0.5


def load(path):
    if not path.exists():
        return []
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln.startswith("{"):
            out.append(json.loads(ln))
    return out


def pick(recs, tag):
    return next((r for r in recs if r.get("tag") == tag), None)


def pct(v, base):
    return 100.0 * (v - base) / base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tee", default="")
    args = ap.parse_args()
    out = []

    def say(s=""):
        print(s)
        out.append(s)

    orig = pick(load(ORIG), TAG)
    repro = [r for r in load(REPRO) if str(r.get("tag", "")).startswith("repro_int8")]

    say("== 复现对账：int8 c1x1, batch 512, opt3, skip1000, preboost100 ==")
    say(f"基准（07 原轮 {TAG}）: dataset {BASE['dataset']:.5f} / "
        f"steady {BASE['steady']:.5f} / acc {BASE['acc']:.2f}%")
    if orig:
        say(f"  原轮记录读数: dataset {orig['dataset_median']:.5f} / "
            f"steady {orig['steady_median']:.5f} / acc {orig['acc']:.2f}%"
            f"  (engine {orig['engine']}, build {orig['build_s']}s)")
    else:
        say("  ⚠️ 没找到原轮记录（bench.jsonl 被改过？）")
    if not repro:
        say("✗ repro/bench.jsonl 里没有 repro_int8* 记录")
        return 1

    say("")
    hdr = f"{'复现档':<22}{'dataset':>10}{'Δ':>9}{'steady':>10}{'Δ':>9}{'acc':>8}{'Δ':>8}{'build':>8}"
    say(hdr)
    say("-" * len(hdr))
    ok = True
    for r in repro:
        d, s, a = r["dataset_median"], r["steady_median"], r["acc"]
        dd, ds, da = pct(d, BASE["dataset"]), pct(s, BASE["steady"]), a - BASE["acc"]
        good = abs(dd) <= 100 * TOL_LAT and abs(ds) <= 100 * TOL_LAT and abs(da) <= TOL_ACC
        ok = ok and good
        say(f"{r['tag']:<22}{d:>10.5f}{dd:>8.2f}%{s:>10.5f}{ds:>8.2f}%"
            f"{a:>8.2f}{da:>+7.2f}pp{r['build_s']:>8.0f}s  {'PASS' if good else 'FAIL'}")

    say("")
    say(f"判据: |Δlatency| <= {100*TOL_LAT:.0f}% (dataset 与 steady 各自)、|Δacc| <= {TOL_ACC} pp")
    say(("✓ 复现通过：同一量级，数字可交付" if ok
         else "✗ 复现失败：偏离超出判据，需查原因（不能改判据去凑）"))

    if args.tee:
        Path(args.tee).write_text("\n".join(out) + "\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
