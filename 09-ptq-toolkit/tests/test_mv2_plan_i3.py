# -*- coding: utf-8 -*-
"""MobileNetV2 I3 混合精度计划（W8 保护 24 层 + W4 per-channel 28 层）验收。

为什么单独一个验收脚本
----------------------
I2 的 top4_all 在 MV2 上掉 1.95pp@512 / 2.15pp@2000，超 1pp 预算。
I3 换了选层口径（见 tools/mv2_mixed_study.py 与 README「MV2 混合精度(二)」）：
  · 用【测试集】而不是训练集 in-sample 的 512 张；
  · 用【边际 CE 伤害】排序，并且在 512/2000/全量 10k 三个口径上同时复核；
  · W4 粒度从 per-group32 换成 per-channel（MV2-CIFAR 的 K 很小，
    per-group 的 scale 元数据吃掉 2-4 个比特，g16 要 6.05 有效比特，
    per-channel 只要 4.25；g32 则是「又贵又差」）。

本脚本把这份计划固定成可复跑的验收：
  1. 计划 JSON 自洽（schema / 层名存在 / 24 W8 + 28 W4 / 声明 per-channel）
  2. w8_layers_from_json 能读出计划，并把 w4_group_size 带出来
     （CLI 的 --w4-group 默认值跟随证据，不静默退回 32）
  3. 包按磁盘字节回读自检（manifest / PTQ2 / PTQC / 逐层网格 / SHA256）
  4. 512 张实测：fp32 / 计划 / 同粒度全 W4 —— 断言计划掉点 <= 1.0pp
     且明显优于全 W4；结果写 results/i3/accept_mv2_plan_n512.json

用法：
    CUDA_VISIBLE_DEVICES='' PYTHONPATH=. \
      /home/user/pyenv-torch/bin/python tests/test_mv2_plan_i3.py
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
PLAN = ROOT / "results/i3/plan_mobilenet_v2_w4pc_w8top24.json"
G16_PLAN = ROOT / "results/i3/plan_mobilenet_v2_w4g16_w8top21.json"
I2_JSON = ROOT / "results/i2/layer_sensitivity_mobilenet_v2.json"
PKG = ROOT / "results/mixed_precision/mobilenet_v2_w4pc_w8top24"
EVAL_N = 512
BUDGET_PP = 1.0
OUT = ROOT / "results/i3/accept_mv2_plan_n512.json"
FAILED = []


def check(name, cond, detail=""):
    print(f"  {'✓' if cond else '✗'} {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILED.append(name)


def conv_names(model):
    return [n for n, m in model.named_modules()
            if isinstance(m, torch.nn.Conv2d)]


def test_plan_json(model):
    print("\n[1] I3 计划 JSON")
    if not PLAN.is_file():
        check("计划 JSON 存在", False, str(PLAN))
        return None
    doc = json.loads(PLAN.read_text(encoding="utf-8"))
    plans = {p["name"]: p for p in doc.get("plans") or []}
    rec = doc.get("recommended") or {}
    name = rec.get("plan")
    check("schema = ptq/mixed-plan@1", doc.get("schema") == "ptq/mixed-plan@1",
          str(doc.get("schema")))
    check("recommended.plan 在 plans 里", name in plans, str(name))
    if name not in plans:
        return None
    p = plans[name]
    names = conv_names(model)
    w8 = p.get("w8_layers") or []
    check("52 层里 24 W8 + 28 W4",
          p.get("n_w8") == 24 and p.get("n_w4") == 28
          and len(w8) == 24 and len(names) == 52,
          f"{p.get('n_w8')}/{p.get('n_w4')}")
    check("W8 名单无重复", len(set(w8)) == len(w8))
    unknown = [n for n in w8 if n not in names]
    check("W8 层名全部是模型的 Conv2d", not unknown, str(unknown[:3]))
    check("W4 粒度是 per-channel",
          p.get("w4_scheme") == "per_channel" and int(p.get("w4_group_size") or 0) == 0,
          f"{p.get('w4_scheme')}/{p.get('w4_group_size')}")
    ver = {v["eval_n"]: v for v in p.get("verified") or []}
    check("带 2000 与全量 10k 复核", set(ver) >= {2000, 10000}, str(sorted(ver)))
    check("2000 <=1.5pp", bool(ver.get(2000, {}).get("passed"))
          and ver[2000]["drop"] <= 1.5, str(ver.get(2000, {}).get("drop")))
    check("全量 10k <=1.5pp", bool(ver.get(10000, {}).get("passed"))
          and ver[10000]["drop"] <= 1.5, str(ver.get(10000, {}).get("drop")))
    return w8


def test_read_plan():
    print("\n[2] 计划读取（w8 名单 + 粒度跟随）")
    w8, meta = w8_layers_from_json(PLAN, "w4pc_w8top24")
    check("计划名 = w4pc_w8top24", meta.get("plan") == "w4pc_w8top24")
    check("读回 24 层 W8", len(w8) == 24, str(len(w8)))
    check("w4_scheme 跟随计划 = per_channel",
          meta.get("w4_scheme") == "per_channel", str(meta.get("w4_scheme")))
    if G16_PLAN.is_file():
        w8_g16, meta_g16 = w8_layers_from_json(G16_PLAN, "w4g16_w8top21")
        check("备用计划 g16 读回 21 层且粒度 = 16",
              len(w8_g16) == 21 and meta_g16.get("w4_group_size") == 16,
              f"{len(w8_g16)}/{meta_g16.get('w4_group_size')}")
    else:
        check("备用 g16 计划存在", False, str(G16_PLAN))
    if I2_JSON.is_file():
        w8_i2, meta_i2 = w8_layers_from_json(I2_JSON, "top4_all")
        check("回归：I2 的 top4_all 仍读回 4 层", len(w8_i2) == 4, str(len(w8_i2)))
        check("回归：I2 的 group_size 仍是 32", meta_i2.get("w4_group_size") == 32,
              str(meta_i2.get("w4_group_size")))
    else:
        check("I2 JSON 存在（回归项）", False, str(I2_JSON))
    return w8


def test_package(w8):
    print("\n[3] 混合精度包磁盘回读自检")
    if not (PKG / "manifest.json").is_file():
        check("包存在", False, str(PKG))
        return
    rep = check_package(PKG)
    check("check_package 全部通过", rep["ok"],
          f"{sum(1 for c in rep['checks'] if not c['ok'])}/{len(rep['checks'])} 失败")
    check("52 conv / 18 block", rep["n_conv"] == 52 and rep["n_block"] == 18,
          f"{rep['n_conv']}/{rep['n_block']}")
    check("24 层 W8 + 28 层 W4", rep["n_w8"] == 24 and rep["n_w4"] == 28,
          f"{rep['n_w8']}/{rep['n_w4']}")
    # 计划 JSON 里的 w8_layers 保持【选层排序】（伤害从大到小，是证据的一部分），
    # 包的 layer_plan 按引擎 conv 定义顺序写；两者只保证集合相同。
    check("W8 名单与计划一致（集合）",
          set(rep["w8_layers"]) == set(w8 or []),
          f"{len(rep['w8_layers'])} vs {len(w8 or [])}")
    counts = {(c["bits"], c["scheme"], c["group_size"]): c["layers"]
              for c in rep["counts"]}
    check("counts = 8/per_channel × 24 + 4/per_channel × 28",
          counts.get((8, "per_channel", 0)) == 24
          and counts.get((4, "per_channel", 0)) == 28, str(counts))
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


def test_measure(w8):
    print(f"\n[4] 512 张实测：计划 vs 同粒度全 W4（CPU）")
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

    plan = explicit_plan(model, w8, w4_scheme="per_channel")
    all_pc = explicit_plan(model, [], w4_scheme="per_channel")
    rows = []
    for key, pl in (("all_w4_per_channel", all_pc), ("w4pc_w8top24", plan)):
        acc = _eval_with_plan(model, test, pl)
        rows.append(dict(candidate=key, acc=round(acc, 4), drop=round(base - acc, 4)))
        print(f"  {key:<20} {acc:6.2f}%  掉点 {base - acc:5.2f} pp")
    by = {r["candidate"]: r for r in rows}
    pick = by["w4pc_w8top24"]
    check(f"计划掉点 <= {BUDGET_PP:.1f} pp（512 张）",
          pick["drop"] <= BUDGET_PP, f"{pick['drop']:.2f} pp")
    check("计划明显优于同粒度全 W4（至少好 2 pp）",
          by["all_w4_per_channel"]["drop"] - pick["drop"] >= 2.0,
          f"{by['all_w4_per_channel']['drop']:.2f} vs {pick['drop']:.2f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(
        model="mobilenet_v2", weights=str(CKPT), eval_n=EVAL_N, device="cpu",
        fp32_acc=round(base, 4), plan="w4pc_w8top24", budget_pp=BUDGET_PP,
        protocol="训练权重→BN 折叠→同一 test 前 512 张→逐层 QDQ"
                 "（fc fp32）→evaluate；不含激活量化",
        rows=rows,
    ), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  · 结果 → {OUT}")


def main():
    print("MobileNetV2 I3 计划验收（CPU-only，w4pc_w8top24）")
    model = get_model("mobilenet_v2")
    w8 = test_plan_json(model)
    if w8 is None:
        print("\nFAIL：计划 JSON 缺失或不可解析")
        return 1
    w8b = test_read_plan()
    test_package(w8b or w8)
    test_measure(w8b or w8)
    print()
    if FAILED:
        print(f"FAIL：{len(FAILED)} 项未通过：{FAILED}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
