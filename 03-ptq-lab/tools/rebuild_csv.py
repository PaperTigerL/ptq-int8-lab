# -*- coding: utf-8 -*-
"""从 results/*.json 重建 results/all.csv。

什么时候需要跑：老版本的 save_result 每次拿「当前行的键」当表头直接追加，
导致不同实验的列互相错位（e2 的 1024 被塞进 fp32_acc 那一列），
并且嵌套结构（e7 的 results / ladder / robust）会被整段丢掉。
修完之后跑一次这个脚本，就能按统一列名把历史结果重排一遍。

不需要 torch —— 只依赖标准库，在任何环境都能跑。

用法：
    python tools/rebuild_csv.py
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.resultio import append_csv_row, flatten_payload  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(os.path.dirname(HERE), "results")


def main():
    files = sorted(glob.glob(os.path.join(RESULT_DIR, "*.json")))
    if not files:
        print("results/ 下没有 json，先跑 ./run.sh e1")
        return 1

    csv_path = os.path.join(RESULT_DIR, "all.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)

    keys = []
    n = 0
    for p in files:
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
        except Exception as e:
            print("跳过（读不了）: %s   %s" % (os.path.basename(p), e))
            continue

        base = os.path.basename(p)[:-5]          # 去掉 .json
        exp = d.get("exp") or base
        tag = base[len(exp) + 1:] if base.startswith(exp + "_") else base
        row = {"exp": exp, "tag": tag, "file": base + ".json"}
        row.update(flatten_payload(d))
        keys = append_csv_row(csv_path, row)
        n += 1

    print("OK")
    print("  结果数 : %d" % n)
    print("  列数   : %d" % len(keys))
    print("  输出   : %s" % csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
