#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""敏感层保护 Pareto 扫描：把 top-K 敏感层保成 FP16，看精度/速度/体积怎么变。

这是阶段 2 路线 A。机制（已验证）：ncnn 量化表里没有某层的条目 ->
ncnn2int8 的 quantize_convolution() 第 169-171 行 continue -> 该层保持
fp16 权重、激活不量化。

排序取自 ptq-lab e4 的敏感度结果（不硬编码层名）。

⚠️ 口径 caveat：e4 的敏感度是 **W4** 口径测的（把某层压到 4bit 看掉多少点），
   而我们这里保护的是 **INT8**（不量化 -> FP16）。两个位宽不同，
   排序只是个代理。这是已知局限，扫描结果本身会校验排序对不对。

用法：
    python3 scripts/pareto_scan.py                       # 默认 K=0,1,2,4,8
    python3 scripts/pareto_scan.py --ks 1,3,6            # 自定义档位
    python3 scripts/pareto_scan.py --limit 2000          # 快速试跑
"""
import argparse
import json
import os
import re
import subprocess
import sys

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PTQ = os.path.join(os.path.dirname(LAB), "03-ptq-lab")

DEFAULT_SCALES = os.path.join(
    PTQ, "results/cifar10_resnet18/scales_w8a8_minmax_mine.json")
DEFAULT_SENS = os.path.join(
    PTQ, "results/cifar10_resnet18/e4_sensitivity_1004-111246.json")


def log(m):
    print(m, flush=True)


def hr():
    print("-" * 64, flush=True)


def run(cmd, quiet=True):
    """跑命令，返回 (returncode, stdout+stderr)。"""
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def load_ranking(sens_path):
    """按 delta 升序取层名（delta 越负 = 越敏感 = 越该保护）。"""
    with open(sens_path) as f:
        d = json.load(f)
    items = sorted(d["sens"], key=lambda e: e["delta_pct"])
    return [e["name"] for e in items]


def main():
    ap = argparse.ArgumentParser(
        description="敏感层保护 Pareto 扫描（INT8 + FP16 混合）")
    ap.add_argument("--scales", default=DEFAULT_SCALES)
    ap.add_argument("--sens", default=DEFAULT_SENS)
    ap.add_argument("--param", default=os.path.join(LAB, "models/resnet18_cifar10_opt.param"))
    ap.add_argument("--bin", default=os.path.join(LAB, "models/resnet18_cifar10_opt.bin"))
    ap.add_argument("--baseline", default=os.path.join(LAB, "tables/resnet18_cifar10_kl.table"))
    ap.add_argument("--ks", default="0,1,2,4,8")
    ap.add_argument("--limit", type=int, default=0, help="0 = 全量 10000")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--ncnn2int8", default=os.path.expanduser(
        "~/ncnn-build/ncnn/build/tools/quantize/ncnn2int8"))
    ap.add_argument("--runner", default=os.path.join(LAB, "cpp/build/cifar10_eval"))
    ap.add_argument("--work", default=os.path.join(LAB, "models/_pareto"))
    ap.add_argument("--out", default=os.path.join(LAB, "results/pareto_sensitive.json"))
    args = ap.parse_args()

    for f in (args.scales, args.sens, args.param, args.bin, args.baseline,
              args.ncnn2int8, args.runner):
        if not os.path.exists(f):
            sys.exit("[缺] %s 不存在" % f)

    ranking = load_ranking(args.sens)
    ks = [int(k) for k in args.ks.split(",")]
    os.makedirs(args.work, exist_ok=True)

    log("敏感层保护 Pareto 扫描（INT8 + FP16 混合）")
    log("  scale 来源 : %s" % os.path.basename(args.scales))
    log("  敏感度排序 : %s" % os.path.basename(args.sens))
    log("  档位       : K = %s" % args.ks)
    log("  评测       : %s 张，%d 线程" % (args.limit or 10000, args.threads))
    log("")
    log("  排序（前 8）：%s" % ", ".join(ranking[:8]))

    results = []
    for k in ks:
        keep = ",".join(ranking[:k]) if k > 0 else ""
        tag = "k%d" % k
        tbl = os.path.join(args.work, "%s.table" % tag)
        oparam = os.path.join(args.work, "%s.param" % tag)
        obin = os.path.join(args.work, "%s.bin" % tag)

        hr()
        log("### K=%d  保护 %d 层 %s" % (k, k, ("（基线，全 INT8）" if k == 0 else "")))
        if keep:
            log("    保护：%s" % keep)

        # 1. 造表
        rc, out = run(
            'python3 %s/scripts/make_table.py --scales "%s" --param "%s" '
            '--baseline "%s" --out "%s" %s'
            % (LAB, args.scales, args.param, args.baseline, tbl,
               ('--keep-fp16 "%s"' % keep) if keep else ""))
        if rc != 0:
            log("  [失败] 造表失败：%s" % out[-500:])
            continue

        # 2. 量化
        rc, out = run('"%s" "%s" "%s" "%s" "%s" "%s"'
                      % (args.ncnn2int8, args.param, args.bin, oparam, obin, tbl))
        if rc != 0:
            log("  [失败] ncnn2int8：%s" % out[-500:])
            continue
        nq = len(re.findall(r"^quantize_convolution ", out, re.M))

        # 3. 评测
        rc, out = run(
            '"%s" --param "%s" --bin "%s" --data "%s/data_raw/test_batch.raw" '
            '--threads %d --limit %d --tag "%s"'
            % (args.runner, oparam, obin, LAB, args.threads, args.limit, tag))
        if rc != 0:
            log("  [失败] 评测：%s" % out[-500:])
            continue

        acc = re.search(r"精度\s*:\s*([\d.]+)%", out)
        ms = re.search(r"([\d.]+)\s*ms/张", out)
        size_kb = os.path.getsize(obin) / 1024.0

        r = {
            "k": k,
            "kept_fp16": ranking[:k],
            "quantized_layers": nq,
            "acc": float(acc.group(1)) if acc else None,
            "ms_per_image": float(ms.group(1)) if ms else None,
            "bin_kb": round(size_kb, 1),
        }
        results.append(r)
        log("  结果：精度 %s  延迟 %s ms/张  bin %s KB  （量化 %d 层）"
            % (r["acc"], r["ms_per_image"], r["bin_kb"], nq))

    # ---- 汇总 ----
    hr()
    log("")
    log("Pareto 汇总")
    log("")
    log("  %4s %10s %12s %12s %10s" % ("K", "精度", "ms/张", "bin/KB", "量化层"))
    base = results[0] if results and results[0]["k"] == 0 else None
    for r in results:
        d_acc = ""
        if base and r["acc"] is not None and base["acc"] is not None:
            d = r["acc"] - base["acc"]
            d_acc = " (%+.2fpp)" % d
        log("  %4d %9s%% %12s %12s %10d%s"
            % (r["k"],
               ("%.2f" % r["acc"]) if r["acc"] is not None else "-",
               ("%.3f" % r["ms_per_image"]) if r["ms_per_image"] else "-",
               r["bin_kb"], r["quantized_layers"], d_acc))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"meta": {"scales": args.scales, "sens": args.sens,
                            "limit": args.limit, "threads": args.threads},
                   "results": results}, f, indent=1)
    log("")
    log("写出 %s" % args.out)

    # 结论提示
    if base and len(results) > 1:
        best = max((r for r in results if r["acc"] is not None),
                   key=lambda r: r["acc"])
        log("")
        log("最高精度在 K=%d：%.2f%%（基线 K=0 是 %.2f%%）"
            % (best["k"], best["acc"], base["acc"]))


if __name__ == "__main__":
    main()
