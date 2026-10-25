# -*- coding: utf-8 -*-
"""report：把 build + bench 汇总成 Markdown 与 JSON。

两份输出内容一致，区别只是格式 —— Markdown 给人看，JSON 给下游程序。
每个数字都带出处（source 列 / provenance），不出现无源数字。
"""
import json
from pathlib import Path
from typing import Dict, Tuple

from ..bench import collect


def fmt_speedup(x: float) -> str:
    """倍数格式化：< 10× 保留一位小数，否则取整。

    ⛔ 之前一律用 `{:.0f}`，把 ncnn CUDA fork 的 **0.8× 显示成了「1×」**——
       一条比基线还慢的路线，在报告里看起来和基线一样快。
       这类错误比数字错更糟：它让读者得出相反的结论。
    """
    return f"{x:.1f}×" if x < 10 else f"{x:.0f}×"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def build_payload(out_dir="results") -> Dict:
    """扫描 results/build_*.json（命名：build_<model>_w<bits>.json）。"""
    out = Path(out_dir)
    builds = []
    for p in sorted(out.glob("build_*_w*.json")):
        d = _load(p)
        if d:
            d["_src"] = p.name          # 溯源：数字来自哪个 json
            builds.append(d)
    return dict(builds=builds, bench=collect())


def render_markdown(d: Dict) -> str:
    L = ["# PTQ 工具链报告（M2 · 三模型 × 两宽度）", "",
         "> 本报告由 `ptq report` 生成。所有基准数字是**对 06/07 已有结果的引用**，",
         "> 不是本次重测 —— GPU 让给用户，且重测会引入机器负载这个变量。", ""]

    # ---- 构建：三模型 × 两宽度 ----
    bs = d["builds"]
    L += ["## 一、量化构建（三模型 × W8/W4）", "",
          "| 模型 | 拓扑 | 宽度 | fp32 基线 | 量化后 | 掉点 | 导出 | 自检 |",
          "|---|---|---|---|---|---|---|---|"]
    for r in bs:
        ex = r["exported"]
        L.append(f"| {r.get('model_name', r.get('model'))} "
                 f"| {r.get('topo_kind', '?')} "
                 f"| W{r['w_bits']}A{r['a_bits']} "
                 f"| {r['fp32_acc']:.2f}% | {r['quant_acc']:.2f}% "
                 f"| {r['acc_drop']:.2f} pp "
                 f"| `{Path(ex['path']).name}`（{ex['n_conv']} conv / "
                 f"{ex['n_block']} block / {ex['bytes'] / 1e6:.1f} MB） "
                 f"| {ex.get('selfcheck_maxdiff', 0):.3g} |")
    L += ["",
          "> ⚠️ **「量化后精度」是权重-only 的 fake-quant，不含激活量化**——",
          "> 激活量化在 C++ 引擎侧才真正生效，所以真实 W8A8 引擎精度会更低。",
          "> W4 的掉点尤其要按「权重-only」读：真实 W4A8 会更低。",
          "",
          "> `自检`列是导出时的拓扑自检 max|diff|，必须为 0（或 fp32 舍入级 1e-5）。", ""]

    # ---- 基准 ----
    L += ["## 二、基准（引用 06/07 已有结果）", "",
          "| 路线 | ms/img (median) | 加速 | top-1 | 出处 |",
          "|---|---|---|---|---|"]
    for r in d["bench"]["rows"]:
        acc = f"{r['accuracy_pct']:.2f}%" if r.get("accuracy_pct") else "—"
        L.append(f"| {r['route']} | {r['ms_per_img']:.4f} | "
                 f"{fmt_speedup(r['speedup'])} | {acc} | `{r['source']}` |")
    L += [""]

    # ---- 引擎侧 ----
    eng = d["bench"].get("engine") or {}
    if eng:
        L += ["### 自写 CUDA int8 引擎（06）的端到端结果", "",
              "| 项 | 值 |", "|---|---|"]
        if "engine_acc" in eng:
            L.append(f"| GPU int8 引擎精度（全量 10000 张）| {eng['engine_acc']:.2f}% |")
        if "numpy_ref_acc" in eng:
            L.append(f"| numpy 独立参考精度 | {eng['numpy_ref_acc']:.2f}% |")
        if "argmax_agree" in eng:
            L.append(f"| argmax 一致率 | {eng['argmax_agree']:.1f}% |")
        for k, label in (("w6a8_acc", "W6A8 精度"), ("w4a8_acc", "W4A8 精度")):
            if k in eng:
                L.append(f"| {label} | {eng[k]:.2f}% |")
        L += ["", f"出处：`{d['bench']['provenance']['engine_e2e']}`", ""]

    # ---- 门控口径 ----
    idle = d["bench"].get("trt_idle") or {}
    if idle:
        L += ["### TensorRT 三精度（门控口径）", "",
              "| 配置 | min | median | util 区间 | 精度 | 加速(median) |",
              "|---|---|---|---|---|---|"]
        for tag in ("fp32", "fp16", "int8"):
            if tag in idle:
                v = idle[tag]
                L.append(f"| TRT {tag} | {v['min']:.4f} | {v['median']:.4f} | "
                         f"{v['util_range']} | {v['acc']:.2f}% | "
                         f"{v['speedup_median']}× |")
        L += ["", f"出处：`{d['bench']['provenance']['idle_final']}`", ""]

    L += ["## 三、数据出处", ""]
    for k, v in d["bench"]["provenance"].items():
        if k != "note":
            L.append(f"- `{k}`：`../lab/{v}`" if not v.startswith("06") else f"- `{k}`：`{v}`")
    L += ["", d["bench"]["provenance"]["note"], ""]
    return "\n".join(L)


def write_reports(out_dir="results") -> Tuple[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = build_payload(out_dir)

    md_path = out / "report.md"
    md_path.write_text(render_markdown(d), encoding="utf-8")
    js_path = out / "report.json"
    js_path.write_text(json.dumps(d, indent=2, ensure_ascii=False),
                       encoding="utf-8")
    return str(md_path), str(js_path)
