# -*- coding: utf-8 -*-
"""bench：从 06/07 的 results 文件读已有基准，**不重新测 GPU**。

⭐ 为什么 M1 不重新测：
   1. GPU 让给用户（不占用他的游戏）
   2. 更重要的：重测会引入「机器负载」这个变量，让 golden test 不可复现。
      08-report 那边已经吃过亏 —— 同一份代码在不同负载下能差 21 倍
      （idle_final.txt 的附录记录了这件事）。

所以 M1 的 bench 是**引用**而不是**测量**。要真测，
等 M3 有了自己的引擎与统一的门控脚本再说。
"""
import re
from pathlib import Path
from typing import Dict, List

# 基线：ncnn CPU fp32 = 11.6 ms/img（全项目统一基准）
BASELINE_MS = 11.6


def _find_lab() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "06-gpu-int8").is_dir() and (p / "03-ptq-lab").is_dir():
            return p
    raise FileNotFoundError("找不到 lab/")


def _parse_summary_csv(lab: Path) -> List[Dict]:
    """读 08-report 的 summary.csv（它已经是七路线同口径的机读版）。"""
    p = lab / "08-report" / "results" / "summary.csv"
    if not p.exists():
        return []
    import csv
    out = []
    with open(p, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append(dict(
                route=row["route"],
                ms_per_img=float(row["ms_per_img_median"]),
                ms_min=float(row["ms_per_img_min"]),
                speedup=float(row["speedup_median"]),
                speedup_min=float(row["speedup_min"]),
                accuracy_pct=(float(row["accuracy_pct"])
                              if row["accuracy_pct"] else None),
                kind=row["kind"],
                source=row["source"],
            ))
    return out


def _parse_idle_final(lab: Path) -> Dict:
    """读 07 的 idle_final.txt 汇总段（门控口径的三行）。"""
    p = lab / "07-tensorrt-deploy" / "results" / "idle_final.txt"
    if not p.exists():
        return {}
    txt = p.read_text(encoding="utf-8")
    out = {}
    for tag in ("fp32", "fp16", "int8"):
        m = re.search(
            rf"^{tag}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+\s+(\S+)\s+([\d.]+)%"
            rf"\s+(\d+)\s+(\d+)\s*$", txt, re.M)
        if m:
            out[tag] = dict(min=float(m.group(1)), median=float(m.group(2)),
                            max=float(m.group(3)), util_range=m.group(4),
                            acc=float(m.group(5)),
                            speedup_median=int(m.group(6)),
                            speedup_min=int(m.group(7)))
    return out


def _parse_engine_acc(lab: Path) -> Dict:
    """读 06 的端到端结果（自写 int8 引擎的精度）。"""
    p = lab / "06-gpu-int8" / "results" / "e2e_compare.txt"
    if not p.exists():
        return {}
    txt = p.read_text(encoding="utf-8")
    out = {}
    for key, pat in (("engine_acc", r"GPU int8 引擎\s+([\d.]+)%"),
                     ("numpy_ref_acc", r"numpy 独立参考\s+([\d.]+)%"),
                     ("argmax_agree", r"argmax 一致[：:]\s*([\d.]+)%")):
        m = re.search(pat, txt)
        if m:
            out[key] = float(m.group(1))
    for key, pat in (("w4a8_acc", r"W4A8\s+([\d.]+)%"),
                     ("w6a8_acc", r"W6A8\s+([\d.]+)%")):
        m = re.search(pat, txt)
        if m:
            out[key] = float(m.group(1))
    return out


def collect() -> Dict:
    """汇总所有可引用的基准数据。"""
    lab = _find_lab()
    rows = _parse_summary_csv(lab)
    idle = _parse_idle_final(lab)
    engine = _parse_engine_acc(lab)
    return dict(
        baseline=dict(route="ncnn CPU fp32", ms_per_img=BASELINE_MS,
                     note="全项目统一基准 1×"),
        rows=rows,
        trt_idle=idle,
        engine=engine,
        provenance=dict(
            summary_csv="08-report/results/summary.csv",
            idle_final="07-tensorrt-deploy/results/idle_final.txt",
            engine_e2e="06-gpu-int8/results/e2e_compare.txt",
            note="全部为**已有结果的引用**，非本次重测",
        ),
    )
