#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""量化模型体检：一条命令跑完所有模型，出对照表并判定是否回归。

为什么需要它
------------
阶段 3 之后手上有 4 个有效模型（FP32 / W8A8 / W4A4 对称 / W4A4 非对称），
每次改 ncnn 的量化通路，都要挨个跑 eval、再用眼睛比 5 个数字。
这个过程有两个问题：
  1. 慢 —— 4 个模型全量是 5~10 分钟，改动小的时候不想跑
  2. 会漏 —— 肉眼看"78.42 vs 78.42"很容易漏掉 78.42 → 78.19

所以把期望值写成配置，让工具自己判。**超容差就是 FAIL，退出码非 0**，
这样它能直接放进脚本或 CI，而不是一份需要人读的报告。

用法
----
    python3 scripts/bench.py                       # 全量，10000 张
    python3 scripts/bench.py --quick 1000          # 快速回归，1000 张
    python3 scripts/bench.py --only W4A4-asym      # 只跑一个
    python3 scripts/bench.py --json out.json       # 落盘

注意：--quick 时测试集变小，精度会有 ±1pp 左右抖动，判定自动放宽到 --quick-tol。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parent.parent
EVAL = LAB / "cpp" / "build" / "cifar10_eval"
DATA = LAB / "data_raw" / "test_batch.raw"
DEFAULT_CFG = Path(__file__).resolve().parent / "bench_models.json"

# 期望值来自 docs/阶段1 与 docs/阶段3b 的实测记录
FALLBACK_CFG = {
    "threads": 1,
    "models": [
        {"tag": "FP32", "param": "models/resnet18_cifar10_opt.param",
         "bin": "models/resnet18_cifar10_opt.bin", "expect": 0.8716},
        {"tag": "W8A8", "param": "models/resnet18_cifar10_int8.param",
         "bin": "models/resnet18_cifar10_int8.bin", "expect": 0.8705},
        {"tag": "W4A4-sym", "param": "models/resnet18_cifar10_w4_packed.param",
         "bin": "models/resnet18_cifar10_w4_packed.bin", "expect": 0.7842},
        {"tag": "W4A4-asym", "param": "models/resnet18_cifar10_w4_asym.param",
         "bin": "models/resnet18_cifar10_w4_asym.bin", "expect": 0.8392},
    ],
}

RE_ACC = re.compile(r"精度\s*:\s*([\d.]+)%")
RE_MS = re.compile(r"([\d.]+)\s*ms/张")


def run_one(m, threads: int, limit: int, data: Path) -> dict:
    cmd = [
        str(EVAL), "--param", str(LAB / m["param"]), "--bin", str(LAB / m["bin"]),
        "--data", str(data), "--threads", str(threads), "--tag", m["tag"],
    ]
    if limit:
        cmd += ["--limit", str(limit)]

    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0

    if p.returncode != 0:
        return {"tag": m["tag"], "ok": False, "error": (p.stderr or p.stdout)[-300:],
                "wall": wall, "n": limit or 10000}

    acc = RE_ACC.search(p.stdout)
    ms = RE_MS.search(p.stdout)
    if not acc:
        return {"tag": m["tag"], "ok": False, "error": "输出里找不到精度行",
                "raw": p.stdout[-300:], "wall": wall, "n": limit or 10000}

    return {
        "tag": m["tag"],
        "ok": True,
        "acc": float(acc.group(1)) / 100.0,
        "ms": float(ms.group(1)) if ms else None,
        "wall": wall,
        "n": limit or 10000,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("--quick", type=int, default=0, help="只跑前 N 张（快速回归）")
    ap.add_argument("--quick-tol", type=float, default=0.012,
                    help="quick 模式的精度容差（默认 1.2pp）")
    ap.add_argument("--tol", type=float, default=0.002,
                    help="全量模式的精度容差（默认 0.2pp）")
    ap.add_argument("--only", default=None, help="只跑 tag 匹配的模型")
    ap.add_argument("--json", default=None, help="结果落盘路径")
    args = ap.parse_args()

    if not EVAL.is_file():
        print(f"[FAIL] 评测器不存在：{EVAL}\n先在 cpp/build 下 make")
        return 2
    if not DATA.is_file():
        print(f"[FAIL] 测试数据不存在：{DATA}")
        return 2

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8")) \
        if Path(args.config).is_file() else FALLBACK_CFG
    models = cfg["models"]
    if args.only:
        models = [m for m in models if args.only.lower() in m["tag"].lower()]
        if not models:
            print(f"[FAIL] 没有 tag 匹配 {args.only}")
            return 2

    tol = args.quick_tol if args.quick else args.tol
    mode = f"quick({args.quick} 张, tol=±{tol*100:.1f}pp)" if args.quick \
        else f"full(10000 张, tol=±{tol*100:.1f}pp)"

    print(f"模型     : {len(models)} 个")
    print(f"模式     : {mode}")
    print(f"线程     : {cfg.get('threads', 1)}")
    print()
    print(f"{'tag':<18} {'acc':>8} {'expect':>8} {'Δ':>8} {'ms/img':>8} {'判定':>6}")
    print("-" * 62)

    rows = []
    failed = 0
    for m in models:
        r = run_one(m, cfg.get("threads", 1), args.quick, DATA)
        if not r["ok"]:
            print(f"{r['tag']:<18} {'--':>8} {'--':>8} {'--':>8} {'--':>8} {'ERR':>6}")
            print(f"    {r['error']}")
            rows.append(r)
            failed += 1
            continue

        exp = m["expect"]
        d = r["acc"] - exp
        ok = abs(d) <= tol
        if not ok:
            failed += 1
        r.update({"expect": exp, "delta": d, "pass": ok})
        ms = f"{r['ms']:.2f}" if r["ms"] is not None else "--"
        print(f"{r['tag']:<18} {r['acc']*100:7.2f}% {exp*100:7.2f}% "
              f"{d*100:+7.2f} {ms:>8} {'PASS' if ok else 'FAIL':>6}")
        rows.append(r)

    print("-" * 62)
    print(f"{'全部通过' if failed == 0 else f'{failed} 个未通过'}")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "mode": "quick" if args.quick else "full",
            "limit": args.quick or 10000,
            "tol": tol,
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "rows": rows,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结果已存 {args.json}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
