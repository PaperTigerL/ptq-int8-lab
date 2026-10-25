# -*- coding: utf-8 -*-
"""验收：08-report 两张表里每个数字都能指到 results 文件/某个 README 的具体行。

用法（在 lab/08-report 下，或 lab/ 下带路径）：
    python scripts/verify_traceability.py

为什么单独写这个文件：内联 heredoc 里的 f-string 会被 shell 吃掉引号转义，
验证逻辑要落在文件里。

2026-10-25 起要验两张表：
    results/summary.csv              主表（TRT = preboost+steady，源 bench.jsonl）
    results/summary_legacy_idle.csv  旧口径脚注表（idle 门控，源 idle_final.txt）
"""
import csv
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPORT = HERE.parent
LAB = REPORT.parent

MAIN = REPORT / "results" / "summary.csv"
LEGACY = REPORT / "results" / "summary_legacy_idle.csv"


def _num_in_line(val, line):
    """val 代表的数值是否出现在 line 里（按浮点值比，不按字符串）。"""
    try:
        v = float(val)
    except ValueError:
        return val in line
    for m in re.finditer(r"\d+\.?\d*", line):
        try:
            if abs(float(m.group(0)) - v) < 1e-6:
                return True
        except ValueError:
            continue
    return False


def check_md(src, val, acc):
    """源是 markdown 的某一行：延迟值（和精度值）必须出现在那一行里。"""
    f, ln = src.rsplit(":", 1)
    p = LAB / f
    if not p.exists():
        return False, "文件不存在 " + f
    lines = p.read_text(encoding="utf-8").splitlines()
    if int(ln) > len(lines):
        return False, "行号越界 %s（文件只有 %d 行）" % (src, len(lines))
    line = lines[int(ln) - 1]
    # 延迟必须出现在**指到的那一行**；精度允许出现在同文件的别处
    # （README 的表格里延迟在一行、精度常在同一节另一行，逼它同行只会逼人造假）。
    ok = _num_in_line(val, line)
    if ok and acc:
        ok = any(_num_in_line(acc, x) for x in lines)
        if not ok:
            return False, "精度 %s 在该文件里找不到" % acc
    return ok, line.strip()[:80]


def check_jsonl(src, val, acc):
    """源是 bench.jsonl 的某个 tag：按 tag 取记录，比 dataset_median 与 acc。"""
    f, tag = src.rsplit(":", 1)
    p = LAB / f
    if not p.exists():
        return False, "文件不存在 " + f
    rec = None
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        r = json.loads(ln)
        if r.get("tag") == tag:
            rec = r
    if rec is None:
        return False, "bench.jsonl 里没有 tag=" + tag
    ok = abs(float(rec["dataset_median"]) - float(val)) < 1e-9
    if acc:
        ok = ok and abs(float(rec["acc"]) - float(acc)) < 1e-9
    return ok, "tag=%s dataset=%.5f acc=%s" % (tag, rec["dataset_median"], rec.get("acc"))


def check_txt(src, val, acc, mn=None):
    """源是 txt 汇总段：median / min / 精度 三个值都在文本里。"""
    p = LAB / src
    if not p.exists():
        return False, "文件不存在 " + src
    txt = p.read_text(encoding="utf-8")
    ok = (str(val) in txt and str(acc) in txt)
    if mn:
        ok = ok and str(mn) in txt
    return ok, "在汇总段里找到 median/min/精度 三个值"


def check(row, cols):
    src = row["source"]
    val = row[cols["median"]]
    acc = row[cols.get("acc", "")] if cols.get("acc") else ""
    if ".md:" in src:
        return check_md(src, val, acc)
    if ".jsonl:" in src:
        return check_jsonl(src, val, acc)
    mn = row[cols["min"]] if cols.get("min") else None
    return check_txt(src, val, acc, mn)


def run(path, cols, title):
    print("== " + title + " ==")
    if not path.exists():
        print("  [FAIL] 缺 " + str(path))
        return 0, 1
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    ok = 0
    for r in rows:
        hit, detail = check(r, cols)
        ok += hit
        print("  [%s] %-16s %8s ms/img  → %s" %
              ("OK " if hit else "FAIL", r["route"], r[cols["median"]], r["source"]))
        if not hit:
            print("         该行内容: " + detail)
    print("  %s (%d/%d)" % ("✓ 全部可溯源" if ok == len(rows) else "✗ 有不可溯源",
                            ok, len(rows)))
    return ok, len(rows)


def main():
    ok1, n1 = run(MAIN,
                  dict(median="ms_per_img_median", acc="accuracy_pct"),
                  "主表 results/summary.csv（TRT = preboost+steady）")
    print("")
    ok2, n2 = run(LEGACY,
                  dict(median="ms_per_img_median", acc="accuracy_pct",
                       min="ms_per_img_min"),
                  "旧口径脚注表 results/summary_legacy_idle.csv（idle 门控）")
    print("")
    ok, n = ok1 + ok2, n1 + n2
    print("  %s (%d/%d)" % ("✓ 两张表全部可溯源" if ok == n else "✗ 有不可溯源",
                            ok, n))
    return 0 if ok == n else 1


if __name__ == "__main__":
    sys.exit(main())
