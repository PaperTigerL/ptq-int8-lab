# -*- coding: utf-8 -*-
"""合并所有 idle_*.txt 的测量，按 util 分档统计（min + median 并列）。

⭐ 为什么是「合并多个文件」而不是「读一个文件」：
   低档轮（util<=20%）很难一次凑够。实测一轮 6 次里常只有 1~2 轮低档，
   而 1~2 轮的 median 就等于那个值本身，统计上不可靠。
   所以 idle_final.sh 每次写一个带时间戳的文件（不覆盖），
   本脚本把所有历史测量合并，分档后才有样本量。

⭐ 为什么必须按 util 分档：
   实测同一批测量里，fp32 在 util 1% 时是 0.0274 ms/img，
   util 98~100% 时是 0.5867 —— **差 21 倍**。
   把所有轮次混在一起求中位，得到的数字既不是性能也不是下界，
   而是没有意义的加权平均。

分档规则（与 idle_final.sh 的门控一致）
   低档 util <= 20%   GPU 安静，该轮是可用的性能值
   高档 util >  20%   有负载压上来，该轮是「被占用时的下界」

⚠️ 低档样本 < min_low（默认 3）时会明确告警 —— 那时 median 不可信。

用法
    ~/pyenv-torch/bin/python summarize_idle.py# 合并全部，写出 idle_summary.txt
    ~/pyenv-torch/bin/python summarize_idle.py --dry      # 只打印
    ~/pyenv-torch/bin/python summarize_idle.py --min-low 5 # 低档样本量下限
"""
import argparse
import glob
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
OUT = RESULTS / "idle_summary.txt"
BASE = 11.6          # ncnn CPU fp32 ms/img，全项目基线
MODELS = ["fp32", "fp16", "int8"]

# 开跑前采样的 util（分档依据）；第二式兼容旧的「跑完后」格式
RE_NEW = re.compile(r"round (\d+): ([\d.]+) ms/img.*?\(开跑前 GPU (\d+) MiB / (\d+)%\)")
RE_OLD = re.compile(r"round (\d+): ([\d.]+) ms/img.*?\(GPU (\d+) MiB / (\d+)%\)")
RE_ANY = re.compile(r"round (\d+): ([\d.]+) ms/img")


def parse_file(path):
    """从一份测量文件里抽出 {model: (rounds, acc, thresh)}。"""
    txt = path.read_text(encoding="utf-8")
    out = {}
    for m in MODELS:
        key = f"--- {m} batch="
        if key not in txt:
            continue
        sec = txt.split(key)[1].split("--- ")[0]
        acc = re.search(r"精度 ([\d.]+)%", sec)
        th = re.search(r"判据 util<= (\d+)%", sec)
        thresh = int(th.group(1)) if th else 20

        rows = RE_NEW.findall(sec) or RE_OLD.findall(sec)
        if rows:
            rounds = [(float(v), int(u)) for _, v, _, u in rows]
        else:
            rounds = [(float(v), -1) for _, v in RE_ANY.findall(sec)]
        if rounds:
            out[m] = (rounds, float(acc.group(1)) if acc else None, thresh)
    return out


def median(xs):
    s = sorted(xs)
    return s[len(s) // 2] if s else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--min-low", type=int, default=3,
                    help="低档样本量下限（少于则告警 median 不可信）")
    args = ap.parse_args()

    files = sorted(glob.glob(str(RESULTS / "idle_*.txt")))
    if not files:
        print("没找到任何 idle_*.txt（先跑 idle_final.sh 或 wait_then_measure.sh）")
        return 1

    merged = {m: [] for m in MODELS}
    accs, thresh, n_files = {}, 20, 0
    for f in files:
        got = parse_file(Path(f))
        if not got:
            continue
        n_files += 1
        for m, (rounds, acc, th) in got.items():
            merged[m].extend(rounds)
            if acc:
                accs[m] = acc
            thresh = th

    total = sum(len(v) for v in merged.values())
    L = ["", "=" * 74,
         f"======== 汇总（合并 {n_files} 份测量文件 / {total} 轮，按 util 分档）========",
         "=" * 74,
         f"数据源：results/idle_*.txt",
         "",
         f"分档规则：低档 util <= {thresh}%（GPU 安静，可用的性能值）",
         f"        高档 util >  {thresh}%（有负载压着，是「被占用时的下界」）",
         "",
         "%-7s %-6s %-6s %-9s %-9s %-12s %-8s" %
         ("配置", "档位", "轮数", "min", "中位", "util 区间", "精度"),
         "-" * 74]

    rows = []
    for m in MODELS:
        rounds = merged[m]
        if not rounds:
            L.append(f"{m:<7} (未测)")
            continue
        low = [(v, u) for v, u in rounds if 0 <= u <= thresh]
        high = [(v, u) for v, u in rounds if u > thresh]

        for tag, group in (("低档", low), ("高档", high)):
            if not group:
                continue
            vals = [v for v, _ in group]
            us = [u for _, u in group]
            s = sorted(vals)
            med = median(vals)
            warn = f" ⚠️<{args.min_low}" if (tag == "低档" and len(group) < args.min_low) else ""
            L.append(f"{m:<7} {tag:<6} {len(group):<6} {s[0]:<9.4f} {med:<9.4f} "
                     f"{f'{min(us)}-{max(us)}%':<12} "
                     f"{(str(accs.get(m)) + '%') if accs.get(m) else '-':<8}{warn}")
            rows.append((m, tag, med, len(group)))

    L += ["-" * 74, "", "【结论】可信值 = 低档轮次的中位数："]

    for m in MODELS:
        lo = [x for x in rows if x[0] == m and x[1] == "低档"]
        hi = [x for x in rows if x[0] == m and x[1] == "高档"]
        if lo:
            _, _, med, n = lo[0]
            tag = f"（低档 {n} 轮）"
            if n < args.min_low:
                tag += f" ⚠️ 少于 {args.min_low} 轮，median 仅供参考"
            if hi:
                tag += f"；另有 {hi[0][3]} 轮高占用未计入"
            L.append(f"  {m:<7} {med:.4f} ms/img = {BASE / med:>6.0f}×  {tag}")
        elif hi:
            _, _, med, n = hi[0]
            L.append(f"  {m:<7} {med:.4f} ms/img = {BASE / med:>6.0f}×  "
                     f"⛔ 无低档轮（{n} 轮全在高占用）")
            L.append(f"          ↑ 这是被占用时的**下界**，不是性能上限。")
        else:
            L.append(f"  {m:<7} (未测)")

    no_low = [m for m in MODELS
              if not [x for x in rows if x[0] == m and x[1] == "低档"]]
    thin = [x for x in rows if x[1] == "低档" and x[3] < args.min_low]

    if no_low:
        L += ["", f"⛔ 以下配置**没有低档轮**：{', '.join(no_low)}",
              "   它们的数字只能读作「被占用时的下界」。",
              "   攒够低档样本：反复跑",
              "   `WAIT_SECONDS=900 ROUNDS=6 bash wait_then_measure.sh`",
              "   本脚本会自动合并所有 idle_*.txt 重算。"]
    if thin:
        L += ["", f"⚠️ 低档样本不足 {args.min_low} 轮："
                  + ", ".join(f"{x[0]}({x[3]}轮)" for x in thin),
              "   median 在这么少的样本上不稳定，建议继续累积。"]

    L += ["",
          "口径说明：",
          "  · 基线 ncnn CPU fp32 = 11.6 ms/img = 1×",
          "  · min = 性能上限；median = 对外报数口径",
          "  · util 是每轮**开跑前**的采样值（本进程还没跑，采到的就是外部负载）",
          "  · 被别的进程占着只会让数字偏慢 → 被占时的数据是下界不是上界",
          "  · 分档不是形式主义：实测 fp32 在 util 1% 时 0.0274 ms/img，",
          "    util 98~100% 时 0.5867 —— 差 21 倍"]

    block = "\n".join(L)
    print(block)
    if not args.dry:
        with open(OUT, "w", encoding="utf-8") as f:
            f.write(block + "\n")
        print(f"\n已写出 {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
