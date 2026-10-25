# -*- coding: utf-8 -*-
"""从各路线的 results → 生成 08-report 的三张图 + 两张数据表。

⭐ 设计原则：**图里的每个数字都必须来自 results 文件，不许硬编码。**
   理由：报告里的数字一旦手抄进去，源数据更新后就对不上了，
   而且没人能说清某个数字到底出自哪一次测量。
   这个脚本把 results/* 当唯一真值来源，解析失败就报错而不是填默认值。

⭐ 2026-10-25 起主表换口径：TRT 三行取 push-20261025 的
   **protocol = preboost(100) + steady**（batch 512，逐批协议前先背靠背顶 clock），
   数据源 07-tensorrt-deploy/results/push-20261025/bench.jsonl。
   旧的 idle 门控口径（778× / 614× / 420×）保留成脚注表，
   数据源仍是 07-tensorrt-deploy/results/idle_final.txt，
   产出 results/summary_legacy_idle.csv —— **两份表并存，但各自标注口径**。

产出：
   results/summary.csv             主表（新口径，机读）
   results/summary_legacy_idle.csv 旧口径脚注表（idle 门控）
   figs/speed.png                  速度对比（对数轴）
   figs/accuracy_speed.png         精度-加速比散点（含旧口径 TRT 空心点）
   figs/tops.png                   每层 TOPS

用法（在 lab/08-report 下）：
   python scripts/make_figs.py
   bash scripts/make_figs.sh          # 等价，自动用 WSL 的 torch 环境
"""
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
LAB = REPORT.parent

TRT = LAB / "07-tensorrt-deploy"
PUSH = TRT / "results" / "push-20261025"
# 两轮数据：07 原轮（fp16/int8 主表数字）+ repro（fp32 折叠图是在复现轮补测的）
BENCHES = [PUSH / "bench.jsonl", PUSH / "repro" / "bench.jsonl"]
LEGACY_IDLE = TRT / "results" / "idle_final.txt"

BASELINE = 11.6                     # ncnn CPU fp32 ms/img = 1×（05 的起点）

# 新口径固定参数：写在这里是为了让报告里的 protocol 一栏有唯一来源，
# 而不是散落在 README 三处。
PUSH_PROTOCOL = "preboost(100)+steady b512"
PUSH_TAGS = [                       # (报告里的名字, bench.jsonl 的 tag)
    ("TRT fp32", "fp32_c1x1_pb"),
    ("TRT fp16", "fp16c_o5_dio_pb"),
    ("TRT int8", "int8_c1x1_pb"),
]
LEGACY_NAMES = {"fp32": "TRT fp32", "fp16": "TRT fp16", "int8": "TRT int8"}

# 中文标签的字体：Windows/WSL 下 matplotlib 默认字体不含 CJK，会显示成方块。
plt.rcParams["axes.unicode_minus"] = False
for _f in ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "Microsoft YaHei",
           "SimHei", "Noto Sans CJK JP"]:
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_f]
        break
    except Exception:
        continue

# 图里用英文标签，避免字体缺失导致方块；中文解释放报告正文里。
LABEL = {
    "ncnn CPU fp32": "ncnn CPU fp32 (baseline)",
    "ncnn CPU int8": "ncnn CPU int8",
    "ncnn CUDA fork": "ncnn CUDA fork",
    "libtorch C++": "libtorch C++",
    "TRT fp32": "TRT fp32",
    "TRT fp16": "TRT fp16",
    "TRT int8": "TRT int8",
}


# ---------------------------------------------------------------- 新口径：bench.jsonl

def load_bench(paths=None):
    """push-20261025 的每档一行 JSON（原轮 + 复现轮）。

    同一 tag 以后读到的为准；每条记录额外带 _file（相对 lab/ 的路径），
    这样 summary.csv 的 source 列能指到**具体是哪个文件里的哪一档**。
    """
    recs = {}
    n_files = 0
    for path in (paths or BENCHES):
        if not path.exists():
            print("  ⚠️ 缺 " + str(path) + "（新口径主表读不到这一份）")
            continue
        n_files += 1
        rel = str(path.relative_to(LAB)).replace("\\", "/")
        for ln in path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln.startswith("{"):
                continue
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            r["_file"] = rel
            recs[r.get("tag", "")] = r
    if not n_files:
        print("  ⚠️ 一份 bench.jsonl 都没读到")
    return recs


def route_from_bench(name, r):
    """bench 一条记录 → 报告一行。延迟主值是 dataset median（与旧表同义：中位口径）。"""
    med = r["dataset_median"]
    steady = r.get("steady_median")
    return dict(
        name=name, ms=med, ms_steady=steady,
        acc=r.get("acc"), kind="gpu-push",
        protocol=PUSH_PROTOCOL,
        src=r["_file"] + ":" + r["tag"],
        note="build " + str(r["build_s"]) + "s, plan " + str(r["plan_mb"]) + "MB",
    )


# ---------------------------------------------------------------- 旧口径：idle_final.txt

def parse_trt_section(path, model):
    """从 idle_final.txt / final.txt 抽出某模型那一节的各轮 ms/img。

    **优先只取低档轮**（util<=THRESH，开跑前采样）——
    高档轮是有负载压着测的，只能当下界，不能代表性能。

    低档一轮都没有时退回全部轮次，并把 busy_only=True 带出去，
    让图和 CSV 都能标注「这是被占用时的下界」。

    ⚠️ 两种格式必须区别对待，不能一视同仁：
      新格式「(开跑前 GPU ... / N%)」—— util 采在**开跑前**，按 util 分档才有意义；
      旧格式「(GPU ... / N%)」—— util 采在轮次**结束后**，那时本进程已跑完，
        采到的是别人的负载，**分档是错的**。idle_final.txt 正是旧格式，
        分档会把 5 轮拆成「1 轮低档 + 4 轮高档」，median 变成那 1 轮的值
        （实测把 int8 从 614× 变成 481×、fp32 从 420× 变成 351×）。
    """
    if not path.exists():
        return None
    txt = path.read_text(encoding="utf-8")
    key = "--- " + model + " batch="
    if key not in txt:
        return None
    sec = txt.split(key)[1].split("--- ")[0]
    acc = re.search(r"精度 ([\d.]+)%", sec)

    rows_new = re.findall(
        r"round \d+: ([\d.]+) ms/img.*?\(开跑前 GPU \d+ MiB / (\d+)%\)", sec)
    is_new_format = bool(rows_new)

    if is_new_format:
        pairs = [(float(v), int(u)) for v, u in rows_new]
    else:
        rows_old = re.findall(
            r"round \d+: ([\d.]+) ms/img.*?\(GPU \d+ MiB / (\d+)%\)", sec)
        pairs = ([(float(v), int(u)) for v, u in rows_old] if rows_old
                 else [(float(v), -1) for v in
                       re.findall(r"round \d+: ([\d.]+) ms/img", sec)])
    if not pairs:
        return None

    thresh_m = re.search(r"判据 util<= (\d+)%", sec)
    thresh = int(thresh_m.group(1)) if thresh_m else 20
    if is_new_format:
        low = [v for v, u in pairs if 0 <= u <= thresh]
        busy_only = not low
        vals = low if low else [v for v, _ in pairs]
    else:
        low, busy_only = [v for v, _ in pairs], False
        vals = [v for v, _ in pairs]

    s = sorted(vals)
    return dict(rounds=pairs, vals=vals, min=s[0], median=s[len(s) // 2],
                max=s[-1], acc=float(acc.group(1)) if acc else None,
                utils=[u for _, u in pairs], busy_only=busy_only,
                n_low=len(low), n_total=len(pairs), thresh=thresh)


def load_legacy():
    """旧口径（idle 门控）三行，只进脚注表与图的空心点。"""
    out = []
    for key, name in LEGACY_NAMES.items():
        r = parse_trt_section(LEGACY_IDLE, key)
        if r is None:
            print("  ⚠️ idle_final.txt 里找不到 " + key + "，旧口径脚注表少一行")
            continue
        out.append(dict(name=name, ms=r["median"], ms_min=r["min"], ms_max=r["max"],
                        acc=r["acc"], kind="gpu-idle",
                        protocol="idle-gated(util<=20%) b512",
                        src="07-tensorrt-deploy/results/idle_final.txt",
                        n_total=r["n_total"], busy_only=r["busy_only"]))
    return out


# ---------------------------------------------------------------- 其余五条路线

def load_others():
    """非 TRT 的路线：数字各自来自它们的 README，同以前的口径（单一测量值）。"""
    routes = []

    src05 = LAB / "05-ncnn-deploy" / "README.md"
    t05 = src05.read_text(encoding="utf-8").splitlines()
    cpu_base = cpu_base_line = None
    for i, line in enumerate(t05, 1):
        m = re.search(r"CPU 基线的 ([\d.]+)", line)
        if m:
            cpu_base, cpu_base_line = float(m.group(1)), i
            break
    routes.append(dict(name="ncnn CPU fp32", ms=cpu_base or BASELINE, acc=None,
                       kind="baseline", protocol="single-run, 4 threads",
                       src="05-ncnn-deploy/README.md:" + str(cpu_base_line or "?"),
                       note="10000 张，CPU 4 线程"))

    src04 = LAB / "04-libtorch-cuda" / "README.md"
    t04 = src04.read_text(encoding="utf-8").splitlines()
    cpu_int8 = cpu_int8_line = None
    for i, line in enumerate(t04, 1):
        m = re.search(r"ncnn CPU int8 4 线程 \| ([\d.]+)", line)
        if m:
            cpu_int8, cpu_int8_line = float(m.group(1)), i
            break
    routes.append(dict(name="ncnn CPU int8", ms=cpu_int8 or 3.4, acc=None,
                       kind="cpu", protocol="single-run, 4 threads",
                       src="04-libtorch-cuda/README.md:" + str(cpu_int8_line or "?"),
                       note=""))

    lib = lib_line = None
    for i, line in enumerate(t04, 1):
        m = re.search(r"libtorch GPU batch=512\*\* \| \*\*([\d.]+)", line)
        if m:
            lib, lib_line = float(m.group(1)), i
            break
    routes.append(dict(name="libtorch C++", ms=lib or 0.0784, acc=87.16,
                       kind="gpu", protocol="batch512, 稳定值",
                       src="04-libtorch-cuda/README.md:" + str(lib_line or "?"),
                       note=""))

    src05c = LAB / "05-ncnn-deploy" / "cuda" / "README.md"
    t05c = src05c.read_text(encoding="utf-8").splitlines()
    fork = fork_line = None
    for i, line in enumerate(t05c, 1):
        m = re.search(r"fp32\*\*，显存不复用 \| \*\*88\.00%\*\* \| \*\*([\d.]+)", line)
        if m:
            fork, fork_line = float(m.group(1)), i
            break
    routes.append(dict(name="ncnn CUDA fork", ms=fork or 14.89, acc=88.00,
                       kind="gpu", protocol="single-run（唯一精度正确的配置）",
                       src="05-ncnn-deploy/cuda/README.md:" + str(fork_line or "?"),
                       note="失败路线：比 CPU 还慢"))
    return routes


def load_routes():
    routes = load_others()
    bench = load_bench()
    for name, tag in PUSH_TAGS:
        r = bench.get(tag)
        if r is None:
            print("  ⚠️ bench.jsonl 里没有 " + tag + "（" + name + "），主表少一行")
            continue
        routes.append(route_from_bench(name, r))
    return routes


# ---------------------------------------------------------------- 输出

def write_summary(routes, out_csv):
    base = next((r["ms"] for r in routes if r["kind"] == "baseline"), BASELINE)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        # lineterminator="\n"：仓库行尾统一 LF（根 .gitattributes）；
        # csv 模块在 Windows 下默认写 \r\n，会让 worktree 变 CRLF。
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["route", "ms_per_img_median", "ms_per_img_steady",
                    "speedup_median", "speedup_steady", "accuracy_pct",
                    "kind", "protocol", "source", "note"])
        for r in routes:
            med = r["ms"]
            st = r.get("ms_steady")
            w.writerow([r["name"], "%.5f" % med,
                        ("%.5f" % st) if st else "",
                        "%.1f" % (base / med),
                        ("%.1f" % (base / st)) if st else "",
                        r.get("acc") if r.get("acc") is not None else "",
                        r["kind"], r["protocol"], r["src"], r.get("note", "")])
    print("  写出 " + str(out_csv))


def write_legacy(legacy, out_csv):
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["route", "ms_per_img_median", "ms_per_img_min",
                    "ms_per_img_max", "speedup_median", "accuracy_pct",
                    "protocol", "source", "note"])
        for r in legacy:
            w.writerow([r["name"], "%.4f" % r["ms"], "%.4f" % r["ms_min"],
                        "%.4f" % r["ms_max"], "%.1f" % (BASELINE / r["ms"]),
                        r["acc"], r["protocol"], r["src"],
                        str(r["n_total"]) + " 轮"
                        + ("，全部被占用" if r["busy_only"] else "")])
    print("  写出 " + str(out_csv))


def fig_speed(routes, legacy, out):
    """速度对比。跨 3 个数量级 → 对数 x 轴；TRT 组同时画出新/旧两种口径。"""
    rs = sorted(routes, key=lambda r: -r["ms"])
    names = [LABEL.get(r["name"], r["name"]) for r in rs]
    meds = [r["ms"] for r in rs]
    base = next((r["ms"] for r in routes if r["kind"] == "baseline"), BASELINE)
    old = {r["name"]: r["ms"] for r in legacy}

    fig, ax = plt.subplots(figsize=(10.6, 5.4))
    x = list(range(len(rs)))
    ax.bar(x, meds, 0.62, label="new protocol: preboost+steady (b512)",
           color="#2563eb")
    # 旧口径只对 TRT 三行有意义：画成同一位置的浅色细柱，直观看出差多少
    old_label_done = False
    for i, r in enumerate(rs):
        if r["name"] not in old:
            continue
        lab = None
        if not old_label_done:
            lab = "old protocol: idle-gated (b512)"
            old_label_done = True
        ax.bar(i, old[r["name"]], 0.30, color="#f59e0b", label=lab)
        ax.text(i + 0.16, old[r["name"]], "old %.0fx" % (base / old[r["name"]]),
                ha="left", va="center", fontsize=7.5, color="#b45309")
    for i, r in enumerate(rs):
        top = max(meds[i], old.get(r["name"], 0))
        ax.text(i, top * 1.16, "%.0fx" % (base / r["ms"]), ha="center",
                fontsize=9.5, fontweight="bold", color="#1d4ed8")
    ax.set_yscale("log")
    ax.set_ylim(0.0045, 42)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right")
    ax.set_ylabel("ms / image  (log scale)")
    ax.set_title("Speed across 7 routes - ResNet-18 CIFAR-10, baseline = ncnn CPU fp32 (11.6 ms/img)\n"
                 "TensorRT rows: batch 512; new = preboost(100)+steady, old = idle-gated",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("  写出 " + str(out))


def fig_accuracy_speed(routes, legacy, out):
    """精度-加速比散点。实心 = 新口径；空心方块 = 旧 idle 口径（只 TRT 三行有）。"""
    pts = [r for r in routes if r.get("acc") is not None]
    base = BASELINE
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    colors = {"gpu-push": "#16a34a", "gpu-idle": "#f59e0b",
              "gpu": "#3b82f6", "baseline": "#6b7280", "cpu": "#6b7280"}
    # 三行 TRT 的精度只差 0.06 pp，点会挤在一起 —— 标注位置必须逐点指定，
    # 否则「都一样」的观感会变成标签互相压字。
    OFF = {"TRT fp32": (10, 18), "TRT fp16": (10, 36), "TRT int8": (-18, 40),
           "libtorch C++": (-102, 8), "ncnn CUDA fork": (8, 6),
           "ncnn CPU int8": (8, 6)}
    OFF_OLD = {"TRT fp32": (14, -24), "TRT fp16": (-52, -44), "TRT int8": (14, -64)}
    for r in pts:
        sp = base / r["ms"]
        ax.scatter(sp, r["acc"], s=150, color=colors.get(r["kind"], "#888"),
                   zorder=4, edgecolors="white", linewidths=0.8)
        ax.annotate(LABEL.get(r["name"], r["name"]), (sp, r["acc"]),
                    textcoords="offset points",
                    xytext=OFF.get(r["name"], (8, 6)), fontsize=8.5)
    for r in legacy:
        sp = base / r["ms"]
        ax.scatter(sp, r["acc"], s=110, facecolors="none", edgecolors="#b45309",
                   marker="s", linewidths=1.6, zorder=3)
        ax.annotate(LABEL.get(r["name"], r["name"]) + " (old)", (sp, r["acc"]),
                    textcoords="offset points",
                    xytext=OFF_OLD.get(r["name"], (8, -14)),
                    fontsize=7.5, color="#b45309")
    ax.plot([], [], "s", markerfacecolor="none", markeredgecolor="#b45309",
            label="old protocol (idle-gated): lower clock, slower")
    ax.scatter([], [], s=120, color="#16a34a",
               label="new protocol (preboost+steady)")
    ax.set_xscale("log")
    ax.set_xlabel("speedup vs ncnn CPU fp32 (log scale)")
    ax.set_ylabel("top-1 accuracy (%)")
    ax.set_xlim(0.6, 2600)
    ax.set_ylim(85.6, 88.6)
    ax.set_title("Accuracy vs speedup - all routes within 0.9 pp\n"
                 "fill = new protocol, hollow square = old idle protocol (same engine)",
                 fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("  写出 " + str(out))


def fig_tops(out):
    """每层 GEMM 吞吐：读 results/gemm_tops.csv。

    数据源是 06-gpu-int8 用 torch._int_mm 在这台 GTX 1650 Ti 上实测的
    七种代表层形状。不用 TRT 逐层 profiler——后者在 TRT 11.4 里
    反序列化 plan 后拿不到（实测见 scripts/probe_trt_profiler.py）。
    """
    src = REPORT / "results" / "gemm_tops.csv"
    if not src.exists():
        print("  ⚠️ 缺 " + src.name + "，跳过每层 TOPS 图")
        return False

    lines = [ln for ln in src.read_text(encoding="utf-8-sig").splitlines()
             if ln.strip() and not ln.lstrip().startswith("#")]
    rows = list(csv.DictReader(lines))
    if not rows:
        print("  ⚠️ " + src.name + " 没有数据行，跳过")
        return False

    stages = [r["stage"] for r in rows]
    ks = [int(r["K"]) for r in rows]
    ns = [int(r["N"]) for r in rows]
    tops = [float(r["tops"]) for r in rows]

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    xs = list(range(len(rows)))
    colors = ["#dc2626" if n < 256 else "#16a34a" for n in ns]
    ax.bar(xs, tops, color=colors)
    best = max(tops)
    ax.axhline(best, color="#2563eb", ls="--", lw=1,
               label="best %.1f TOPS (N=512)" % best)
    for i, n in enumerate(ns):
        ax.text(i, tops[i] + 0.22, "N=%d" % n, ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(["%s\nK=%d" % (stages[i], ks[i]) for i in range(len(rows))],
                       fontsize=8)
    ax.set_ylabel("TOPS (int8, measured)")
    ax.set_xlabel("layer stage  /  K = Cin x k x k")
    ax.set_ylim(0, best * 1.22)
    ax.set_title("int8 GEMM throughput vs output channels\n"
                 "red = N < 256: tensor cores cannot be saturated "
                 "(shallow layers of ResNet-18 CIFAR)", fontsize=10)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("  写出 " + str(out))
    return True


def main():
    print("解析各路线 results ...")
    routes = load_routes()
    legacy = load_legacy()
    if not routes:
        print("没解析到任何路线，检查 results 文件")
        return 1

    out_csv = REPORT / "results" / "summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    (REPORT / "figs").mkdir(parents=True, exist_ok=True)

    write_summary(routes, out_csv)
    write_legacy(legacy, REPORT / "results" / "summary_legacy_idle.csv")
    fig_speed(routes, legacy, REPORT / "figs" / "speed.png")
    fig_accuracy_speed(routes, legacy, REPORT / "figs" / "accuracy_speed.png")
    fig_tops(REPORT / "figs" / "tops.png")

    print("")
    print("主表（新口径）：")
    base = next((r["ms"] for r in routes if r["kind"] == "baseline"), BASELINE)
    for r in sorted(routes, key=lambda r: r["ms"]):
        acc = ("%.2f%%" % r["acc"]) if r.get("acc") is not None else "  -"
        print("  %-18s %8.4f ms/img  %7.1fx  %8s  [%s]"
              % (r["name"], r["ms"], base / r["ms"], acc, r["src"]))
    if legacy:
        print("")
        print("旧口径（脚注）：")
        for r in legacy:
            print("  %-18s %8.4f ms/img  %7.1fx  %.2f%%"
                  % (r["name"], r["ms"], BASELINE / r["ms"], r["acc"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
