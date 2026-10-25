#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按已归位的 json 重建各组合的 all.csv（results/<dataset>_<arch>/all.csv）。

背景：2026-10-17 参数化后，落盘路径从 results/ 根目录改成 results/<组合>/ 子目录。
移动 json 时 all.csv 跟不上（行与文件会失配），所以从 json 重新生成——
json 是唯一事实来源，csv 只是派生视图。

用法：
    ~/pyenv-torch/bin/python tools/rebuild_csv_all.py
"""
import csv
import glob
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "results")


def flatten(payload, prefix=""):
    """把嵌套 dict 拍平成  a.b.c -> v。list 整体转字符串。"""
    out = {}
    for k, v in payload.items():
        key = "%s%s" % (prefix, k)
        if isinstance(v, dict):
            out.update(flatten(v, prefix=key + "."))
        elif isinstance(v, (list, tuple)):
            out[key] = "; ".join(str(x) for x in v)
        else:
            out[key] = v
    return out


def rebuild(subdir):
    d = os.path.join(RESULTS, subdir)
    files = sorted(glob.glob(os.path.join(d, "*.json")))
    if not files:
        print("  %-24s 无 json，跳过" % subdir)
        return 0

    rows = []
    skipped = 0
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as exc:
            print("  ! %s 解析失败: %s" % (os.path.basename(f), exc))
            continue

        # ⚠️ **跳过开发期的冒烟测试数据**（2026-10-17 加）。
        #   e6 的正式数据是 iters=20000；冒烟测试用 --iters 30/50，
        #   10-04 那批老数据还没有 iters 字段（旧版实验）。
        #   曾因此让 30 步的结果排在正式数据前面，rebuild 后 all.csv 里
        #   "最新一行" 变成冒烟数据 —— 引用时极容易看错。
        #   判据：**只认 iters == 20000**，其余（缺字段/小数值）一律过滤。
        exp = payload.get("exp")
        iters = payload.get("iters")
        if exp == "e6_adaround":
            try:
                official = int(iters) == 20000
            except (TypeError, ValueError):
                official = False
            if not official:
                skipped += 1
                continue

        row = {"exp": exp or "?",
               "tag": os.path.basename(f).rsplit("_", 1)[-1].replace(".json", ""),
               "file": os.path.basename(f)}
        row.update(flatten(payload))
        rows.append(row)

    if not rows:
        print("  %-24s 全部被过滤（%d 个冒烟），跳过" % (subdir, skipped))
        return 0

    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)

    out = os.path.join(d, "all.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    note = "（已过滤 %d 个冒烟）" % skipped if skipped else ""
    print("  %-24s %2d 行 -> %s %s"
          % (subdir, len(rows), os.path.relpath(out, BASE), note))
    return len(rows)


def main():
    print("重建 all.csv（json 为唯一事实来源）")
    print("-" * 60)
    subs = [d for d in sorted(os.listdir(RESULTS))
            if os.path.isdir(os.path.join(RESULTS, d))
            and not d.startswith(("_", "archive", "figs"))]
    total = 0
    for s in subs:
        total += rebuild(s)
    print("-" * 60)
    print("共 %d 行" % total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
