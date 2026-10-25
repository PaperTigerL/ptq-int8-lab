# -*- coding: utf-8 -*-
"""MobileNetV2 · top4_all 混合精度包的独立自检与掉点对比。

用途：T-20261023-01 的「混合精度包 + 自检脚本可复跑」验收项。
与 R50 的 `tests/test_mixed_precision.py` 同协议，但参数固定到
`mv2img10` 10ep 权重，且不硬编码层名（从 I2 JSON 读取）。

流程：
  1. 从 results/i2/layer_sensitivity_mobilenet_v2.json 读 top4_all
  2. 校验 4 层 W8 / 48 层 W4，层名必须真实存在
  3. 回读 results/mixed_precision/mobilenet_v2_top4_all 包（若已导出）
  4. 同一 test 前 512 张：fp32 / 全 W4 per-channel / 全 W4 g32 / top4_all
  5. 断言 top4_all 掉点 <=3pp 且优于全 W4；结果写 JSON

用法：
    CUDA_VISIBLE_DEVICES='' PYTHONPATH=. \
      /home/user/pyenv-torch/bin/python tests/test_mv2_mixed.py
"""
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import torch                                                      # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from ptq_toolkit.export.mixed_check import check_package           # noqa: E402
from ptq_toolkit.models import get_model                           # noqa: E402
from ptq_toolkit.quant.mixed import (explicit_plan,                # noqa: E402
                                     w8_layers_from_json)
from ptq_toolkit.quant.ptq import quantize_weight_sym              # noqa: E402

CKPT = Path("/mnt/d/WorkBuddy/ShowBro/paper-code/ckpts/"
            "ckpt_mobilenet_v2_mv2img10_ep9.pth")
I2_JSON = ROOT / "results/i2/layer_sensitivity_mobilenet_v2.json"
PKG = ROOT / "results/mixed_precision/mobilenet_v2_top4_all"
EVAL_N = 512
OUT = ROOT / "results/mixed_precision/mv2_allw4_vs_top4_all_n512.json"
FAILED = []


def check(name, cond, detail=""):
    print(f"  {'✓' if cond else '✗'} {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILED.append(name)


def test_plan():
    print("\n[1] MV2 I2 计划 top4_all")
    if not I2_JSON.is_file():
        check("I2 JSON 存在", False, str(I2_JSON))
        return None
    w8, meta = w8_layers_from_json(I2_JSON, "top4_all")
    model = get_model("mobilenet_v2")
    names = [n for n, m in model.named_modules()
             if isinstance(m, torch.nn.Conv2d)]
    check("plan = top4_all", meta.get("plan") == "top4_all", str(meta.get("plan")))
    check("W8 层数 = 4", len(w8) == 4, str(w8))
    check("W4 层数 = 48", meta.get("n_w4") == 48, str(meta.get("n_w4")))
    check("W8 层名全部存在于模型", all(n in names for n in w8))
    check("W8 名单无重复", len(set(w8)) == len(w8))
    check("I2 记录 52 conv", len(names) == 52, str(len(names)))
    return w8


def test_package(w8):
    print("\n[2] 混合精度包磁盘回读自检")
    if not (PKG / "manifest.json").is_file():
        check("包存在", False, str(PKG))
        return
    rep = check_package(PKG)
    check("check_package 全部通过", rep["ok"],
          f"{sum(1 for c in rep['checks'] if not c['ok'])}/{len(rep['checks'])} 失败")
    check("52 conv / 18 block",
          rep["n_conv"] == 52 and rep["n_block"] == 18,
          f"{rep['n_conv']}/{rep['n_block']}")
    check("4 层 W8 + 48 层 W4",
          rep["n_w8"] == 4 and rep["n_w4"] == 48,
          f"{rep['n_w8']}/{rep['n_w4']}")
    check("W8 名单与 I2 一致", rep["w8_layers"] == w8, str(rep["w8_layers"]))
    m = json.loads((PKG / "manifest.json").read_text(encoding="utf-8"))
    check("manifest 记录 5 个产物", len(m.get("files", {})) == 5,
          str(sorted(m.get("files", {}))))
    check("拓扑自检 max|diff| = 0",
          m.get("topology_selfcheck_maxdiff") == 0.0,
          str(m.get("topology_selfcheck_maxdiff")))
    check("source ckpt 有 SHA256",
          bool((m.get("source_weights") or {}).get("sha256")),
          (m.get("source_weights") or {}).get("sha256", "")[:12])


def _eval_with_plan(model, test, plan):
    import copy
    m = copy.deepcopy(model)
    for name, mod in m.named_modules():
        if not isinstance(mod, torch.nn.Conv2d):
            continue
        spec = plan[name]
        w = mod.weight.detach().cpu().numpy()
        q = quantize_weight_sym(w, spec.bits, spec.scheme, spec.group_size)
        mod.weight.data.copy_(
            torch.from_numpy(q.dequant()).to(
                mod.weight.device, mod.weight.dtype).reshape(mod.weight.shape))
    from ptq_toolkit.pipeline import load_ptq_lab
    lab = load_ptq_lab()
    return 100.0 * float(lab["evaluate"](m, test, device="cpu"))


def test_compare(w8):
    print(f"\n[3] 全 W4 vs top4_all 掉点对比（前 {EVAL_N} 张，CPU）")
    if w8 is None or not CKPT.is_file():
        check("权重存在", False, str(CKPT))
        return
    from ptq_toolkit.pipeline import (get_loaders, load_ptq_lab,
                                      load_trained_weights)

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)

    lab = load_ptq_lab()
    model = get_model("mobilenet_v2")
    n_sd, miss, unexp = load_trained_weights(model, str(CKPT))
    check("训练权重键集合干净", not miss and not unexp,
          f"{n_sd} tensors, missing={miss[:2]}, unexpected={unexp[:2]}")
    model = lab["fold_all_bn"](model.eval()).eval()
    _calib, test, _n_train, n_test = get_loaders(calib_n=1, eval_n=EVAL_N)
    check("test loader 样本数", n_test == EVAL_N, str(n_test))

    base = 100.0 * float(lab["evaluate"](model, test, device="cpu"))
    check(f"fp32 基线 {base:.2f}% >= 85%", base >= 85.0)

    plan_top4 = explicit_plan(model, w8)
    all_ch = explicit_plan(model, [], w4_scheme="per_channel")
    all_g32 = explicit_plan(model, [], w4_group_size=32)
    rows = []
    for key, plan in (("all_w4_per_channel", all_ch),
                      ("all_w4_per_group32", all_g32),
                      ("top4_all", plan_top4)):
        acc = _eval_with_plan(model, test, plan)
        rows.append(dict(candidate=key, acc=round(acc, 4),
                         drop=round(base - acc, 4)))
        print(f"  {key:<20} {acc:6.2f}%  掉点 {base - acc:5.2f} pp")

    by = {r["candidate"]: r for r in rows}
    top4 = by["top4_all"]
    check("top4_all 掉点 <=3.0 pp（512 张）", top4["drop"] <= 3.0,
          f"{top4['drop']:.2f} pp")
    check("top4_all 优于全 W4 per-channel",
          top4["drop"] < by["all_w4_per_channel"]["drop"],
          f"{top4['drop']:.2f} < {by['all_w4_per_channel']['drop']:.2f}")
    check("top4_all 优于全 W4 per-group32",
          top4["drop"] < by["all_w4_per_group32"]["drop"],
          f"{top4['drop']:.2f} < {by['all_w4_per_group32']['drop']:.2f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(
        model="mobilenet_v2", weights=str(CKPT), eval_n=EVAL_N,
        device="cpu", fp32_acc=round(base, 4),
        protocol="训练权重→BN 折叠→同一 test 前 512 张→逐层 QDQ"
                 "（fc fp32）→evaluate；不含激活量化",
        rows=rows,
        note="W8 掉点见 ptq eval 的 m2_20261023 结果；本脚本只做混合精度对比。",
    ), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  · 结果 → {OUT}")


def main():
    print("MobileNetV2 top4_all 自检（CPU-only）")
    w8 = test_plan()
    test_package(w8)
    test_compare(w8)
    print()
    if FAILED:
        print(f"FAIL：{len(FAILED)} 项未通过：{FAILED}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
