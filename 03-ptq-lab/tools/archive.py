#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 results/ 顶层的旧结果归档，只留每个实验最新的一份。

为什么需要这个脚本
------------------
每次跑实验都是往 results/ 顶层新写一个带时间戳的文件（e1_baseline_1004-110945.json），
从来不清。跑上十来轮之后顶层就堆了几十个同名不同戳的文件：

  - all.csv 会跟着涨到几十行，一行的实验夹着多个互相冲突的版本；
  - 想引用「论文主表那一份」时得靠肉眼挑时间戳；
  - 而 plot.py / report.py 其实都是按 mtime 取每个实验的最新一份，
    也就是说**顶层那堆旧文件从来没被用过，只是噪音**。

归档规则（和 plot.py 的 latest() 完全同一个语义）
------------------------------------------------
  每个 exp（e1_baseline / e2_rtn_ladder / ... / e7_combo）只保留 mtime 最大的那一份，
  其余全部挪到 results/archive/<归档时刻>/ 下，原文件名不变、mtime 不变。

  ⛔ 只挪不删。归档目录同样被 .gitignore 排除（可重跑的东西不进版本库），
     留着只是防误删，确认不用了再整个目录删掉。

幂等
----
  跑第二次时顶层已经只剩「每实验最新一份」，没有任何文件需要挪，脚本会直接告诉你
  「没有需要归档的」，一个字节都不动。

只依赖标准库，不需要 torch。

用法
    python tools/archive.py          # 归档
    python tools/archive.py --dry    # 只打印会挪哪些，不动手
"""
import glob
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULT_DIR = os.path.join(ROOT, "results")
ARCHIVE_DIR = os.path.join(RESULT_DIR, "archive")


def exp_of(path, payload=None):
    """取实验名。优先信 json 里的 exp 字段，没有再从文件名前缀推。"""
    name = os.path.basename(path)[:-5]          # 去掉 .json
    if payload:
        e = payload.get("exp")
        if e:
            return e
    for e in ("e1_baseline", "e2_rtn_ladder", "e3_act_calib",
              "e4_sensitivity", "e5_mixed", "e6_adaround", "e7_combo"):
        if name.startswith(e + "_"):
            return e
    return name


def main():
    dry = "--dry" in sys.argv

    files = sorted(glob.glob(os.path.join(RESULT_DIR, "*.json")))
    if not files:
        print("results/ 顶层没有 json，没什么可归档的。")
        return 0

    # 按 exp 分组，各组内按 mtime 排序，只留最大的那个
    groups = {}
    for p in files:
        mt = os.path.getmtime(p)
        try:
            import json
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            d = None
        groups.setdefault(exp_of(p, d), []).append((mt, p))

    keep, move = [], []
    for e in sorted(groups):
        items = sorted(groups[e], key=lambda t: t[0])
        keep.append(items[-1][1])
        move.extend(p for _, p in items[:-1])

    print("=" * 60)
    print("results/ 归档检查")
    print("=" * 60)
    print("  顶层 json 总数 : %d" % len(files))
    print("  实验数         : %d" % len(groups))
    print("  保留（各最新） : %d" % len(keep))
    print("  待归档         : %d" % len(move))

    if not move:
        print("")
        print("没有需要归档的 —— 每个实验在顶层都只有一份，已是干净状态。")
        return 0

    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = os.path.join(ARCHIVE_DIR, stamp)
    if dry:
        print("")
        print("[dry-run] 将归档到 %s" % dest)
        for p in move:
            print("    %s" % os.path.basename(p))
        return 0

    os.makedirs(dest, exist_ok=True)
    n = 0
    for p in move:
        shutil.move(p, os.path.join(dest, os.path.basename(p)))
        n += 1

    # 旧 all.csv 一起留档：它是历史全量，重建后顶层那份只含留下的几行
    csv_path = os.path.join(RESULT_DIR, "all.csv")
    if os.path.exists(csv_path):
        shutil.copy2(csv_path, os.path.join(dest, "all_before_%s.csv" % stamp))

    print("")
    print("已归档 %d 个 -> %s" % (n, dest))
    print("顶层剩 %d 个 json（每个实验各一份最新）" % len(keep))

    # 重建 all.csv，让它跟顶层对上。
    # 归档后顶层少了文件，不重建的话 all.csv 还留着旧行，跟顶层就对不上了。
    sys.path.insert(0, HERE)
    try:
        import rebuild_csv
        rebuild_csv.main()
    except Exception as e:
        print("（all.csv 重建没成功：%s —— 手动跑 ./run.sh csv 补一次）" % e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
