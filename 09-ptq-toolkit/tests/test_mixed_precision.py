# -*- coding: utf-8 -*-
"""I2 结论落地：per-layer 混合精度（top4_all）的计划、包格式与掉点对比。

覆盖：
  1. 从 I2 敏感度 JSON 读推荐计划：top4_all = 4 层 W8 + 49 层 W4
  2. 显式 per-layer 覆盖的校验（拼错层名必须报错，重复名去重）
  3. 混合精度权重包的**磁盘回读自检**（manifest/PTQ2/PTQC/quant_config/
     layer_plan/hash/逐层 bits 网格）
  4. 同一模型上 all-W4（per-channel / per-group32）vs top4_all 的掉点对比
     —— 用现有 M2 test harness（训练权重 → BN 折叠 → 同一 test loader →
     CPU 前向），只取前 512 张，避免重复全量 10k 评测

用法：
    python tests/test_mixed_precision.py            # 全跑
    python tests/test_mixed_precision.py --skip-eval  # 跳过精度对比（~60 s）
"""
import json
import os
import subprocess
import sys
from pathlib import Path

# ⛔ CPU-only：必须在 import torch 之前清掉 CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np                                                # noqa: E402
import torch                                                      # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from ptq_toolkit.export.mixed_check import check_package           # noqa: E402
from ptq_toolkit.models import get_model                           # noqa: E402
from ptq_toolkit.quant.mixed import (build_layer_plan,             # noqa: E402
                                     explicit_plan,
                                     make_explicit_candidate,
                                     plan_layer_list, w8_layers_from_json)
from ptq_toolkit.quant.ptq import quantize_weight_sym              # noqa: E402

R50_CKPT = Path("/mnt/d/WorkBuddy/ShowBro/paper-code/ckpts/"
                "ckpt_resnet50_r50aug20_ep18.pth")
I2_JSON = ROOT / "results/i2/layer_sensitivity_resnet50.json"
PKG = ROOT / "results/mixed_precision/resnet50_top4_all"
EVAL_N = 512
FAILED = []

EXPECTED_W8 = ["conv1", "layer1.0.conv1", "layer1.0.conv3", "layer1.1.conv1"]


def check(name, cond, detail=""):
    print(f"  {'✓' if cond else '✗'} {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILED.append(name)


# ---------------------------------------------------------------- 1) I2 计划

def test_i2_plan():
    print("\n[1] I2 推荐计划 top4_all")
    w8, meta = w8_layers_from_json(I2_JSON)
    check("推荐计划名 = top4_all", meta["plan"] == "top4_all", str(meta["plan"]))
    check("W8 层 = conv1 + layer1.0.conv1/conv3 + layer1.1.conv1",
          w8 == EXPECTED_W8, str(w8))
    check("W8 层数 4 / W4 层数 49",
          meta.get("n_w8") == 4 and meta.get("n_w4") == 49,
          f"{meta.get('n_w8')}/{meta.get('n_w4')}")
    check("I2 实测掉点 0.0 pp（512 张 in-sample）",
          meta.get("drop_pp") == 0.0, str(meta.get("drop_pp")))

    # 指定别的计划必须精确命中，不许静默退回默认
    _, meta_structural = w8_layers_from_json(I2_JSON, "structural")
    check("指定 structural 计划可读取",
          meta_structural["plan"] == "structural"
          and len(meta_structural["w8_layers"]) == 6)

    try:
        w8_layers_from_json(I2_JSON, "no_such_plan")
        check("未知计划名应报错", False)
    except KeyError:
        check("未知计划名应报错", True)


# ---------------------------------------------------------------- 2) 显式覆盖

def test_explicit_plan():
    print("\n[2] 显式 per-layer 覆盖")
    model = get_model("resnet50")
    names = [n for n, m in model.named_modules()
             if isinstance(m, torch.nn.Conv2d)]
    plan = explicit_plan(model, EXPECTED_W8)
    check("53 层全部有配置", len(plan) == 53 and list(plan) == names,
          f"{len(plan)} 层")
    check("4 层 W8 per-channel",
          sum(1 for s in plan.values() if s.bits == 8) == 4
          and all(plan[n].bits == 8 and plan[n].scheme == "per_channel"
                  for n in EXPECTED_W8))
    check("49 层 W4 per-group32",
          sum(1 for s in plan.values()
              if s.bits == 4 and s.scheme == "per_group"
              and s.group_size == 32) == 49)

    cand = make_explicit_candidate("top4_all", EXPECTED_W8)
    check("Candidate 路径与 explicit_plan 完全一致",
          build_layer_plan(model, cand) == plan)

    try:
        explicit_plan(model, ["conv1", "layer1.0.conv999"])
        check("拼错层名应报错", False)
    except ValueError:
        check("拼错层名应报错", True)

    dup = explicit_plan(model, ["conv1", "conv1"] + EXPECTED_W8[1:])
    check("重复层名去重后仍 4 层 W8",
          sum(1 for s in dup.values() if s.bits == 8) == 4)

    rows = plan_layer_list(plan)
    check("逐层列表顺序 = Conv2d 定义顺序", [r["name"] for r in rows] == names)


# ---------------------------------------------------------------- 3) 包自检

def test_package():
    print("\n[3] 混合精度包磁盘回读自检")
    manifest = PKG / "manifest.json"
    if not manifest.is_file():
        print(f"  · SKIP：{PKG} 不存在；生成命令见 README「混合精度」")
        return
    binp = PKG / "resnet50_top4_all.ptq2.bin"
    cfgp = PKG / "quant_config_resnet50_top4_all.json"
    if not binp.is_file() or not cfgp.is_file():
        print(f"  · SKIP：权重/量化参数未落盘（.gitignore 忽略大文件）；"
              f"先跑 README 的 export 命令")
        return
    rep = check_package(PKG)
    check("check_package 全部通过", rep["ok"],
          f"{sum(1 for c in rep['checks'] if not c['ok'])}/{len(rep['checks'])} 失败")
    check("53 conv / 16 block", rep["n_conv"] == 53 and rep["n_block"] == 16,
          f"{rep['n_conv']}/{rep['n_block']}")
    check("4 层 W8 + 49 层 W4",
          rep["n_w8"] == 4 and rep["n_w4"] == 49,
          f"{rep['n_w8']}/{rep['n_w4']}")
    check("W8 名单与 I2 推荐一致", rep["w8_layers"] == EXPECTED_W8,
          str(rep["w8_layers"]))
    m = json.loads(manifest.read_text(encoding="utf-8"))
    check("manifest 记录 5 个产物哈希",
          len(m.get("files", {})) == 5, str(sorted(m.get("files", {}))))
    check("拓扑自检 max|diff| = 0",
          m.get("topology_selfcheck_maxdiff") == 0.0,
          str(m.get("topology_selfcheck_maxdiff")))
    check("PTQC 长度 8+8n = 432 B",
          (PKG / "calib_resnet50_top4_all.ptqc").stat().st_size == 432)
    check("source ckpt 有 SHA256",
          (m.get("source_weights") or {}).get("sha256", "").startswith(
              "b16cc5aa"), (m.get("source_weights") or {}).get("sha256", "")[:12])


def test_cli_check():
    print("\n[4] ptq mixed-check CLI")
    if not (PKG / "manifest.json").is_file() \
            or not (PKG / "resnet50_top4_all.ptq2.bin").is_file():
        print("  · SKIP：包不存在")
        return
    r = subprocess.run(
        [sys.executable, "-m", "ptq_toolkit", "mixed-check", str(PKG),
         "--no-grid"],
        cwd=str(ROOT), capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)})
    check("mixed-check rc=0", r.returncode == 0,
          (r.stdout + r.stderr).strip().splitlines()[-1]
          if (r.stdout or r.stderr) else "")


# ---------------------------------------------------------------- 5) 掉点对比

def _eval_with_plan(model, test, plan, device="cpu"):
    """把 plan 逐层 QDQ 到权重上（fc 保持 fp32），返回 top-1（%）。"""
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
    return 100.0 * float(lab["evaluate"](m, test, device=device))


def test_accuracy_compare():
    print(f"\n[5] 全 W4 vs top4_all 掉点对比（前 {EVAL_N} 张，CPU 前向）")
    if not R50_CKPT.is_file():
        print(f"  · SKIP：找不到 {R50_CKPT}")
        return

    from ptq_toolkit.pipeline import (get_loaders, load_ptq_lab,
                                      load_trained_weights)

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)

    lab = load_ptq_lab()
    model = get_model("resnet50")
    n_sd, miss, unexp = load_trained_weights(model, str(R50_CKPT))
    check("训练权重键集合干净", not miss and not unexp,
          f"{n_sd} tensors, missing={miss[:2]}, unexpected={unexp[:2]}")
    model = lab["fold_all_bn"](model.eval()).eval()
    _calib, test, _n_train, n_test = get_loaders(calib_n=1, eval_n=EVAL_N)
    check("test loader 样本数", n_test == EVAL_N, str(n_test))

    base = 100.0 * float(lab["evaluate"](model, test, device="cpu"))
    check(f"fp32 基线 {base:.2f}% >= 90%", base >= 90.0)

    w8, _meta = w8_layers_from_json(I2_JSON)
    plan_top4 = explicit_plan(model, w8)
    all_ch = explicit_plan(model, [], w4_scheme="per_channel")
    all_g32 = explicit_plan(model, [], w4_group_size=32)

    # explicit_plan 的空 W8 名单 = 全 W4；这里直接构造对照（不走 w8_layers API）
    rows = []
    for key, plan in (("all_w4_per_channel", all_ch),
                      ("all_w4_per_group32", all_g32),
                      ("i2_top4_all", plan_top4)):
        acc = _eval_with_plan(model, test, plan)
        rows.append(dict(candidate=key, acc=round(acc, 4),
                         drop=round(base - acc, 4)))
        print(f"  {key:<20} {acc:6.2f}%  掉点 {base - acc:5.2f} pp")

    by = {r["candidate"]: r for r in rows}
    top4 = by["i2_top4_all"]
    check("top4_all 掉点 <= 1.0 pp（512 张）", top4["drop"] <= 1.0,
          f"{top4['drop']:.2f} pp")
    check("top4_all 优于全 W4 per-channel",
          top4["drop"] < by["all_w4_per_channel"]["drop"],
          f"{top4['drop']:.2f} < {by['all_w4_per_channel']['drop']:.2f}")
    check("top4_all 优于全 W4 per-group32",
          top4["drop"] < by["all_w4_per_group32"]["drop"],
          f"{top4['drop']:.2f} < {by['all_w4_per_group32']['drop']:.2f}")

    out = ROOT / "results/mixed_precision/accuracy_allw4_vs_top4_all_n512.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        model="resnet50", weights=str(R50_CKPT), eval_n=EVAL_N,
        device="cpu", fp32_acc=round(base, 4),
        protocol="训练权重→BN 折叠→同一 test loader（前 512 张）→"
                 "逐层 QDQ（fc fp32）→evaluate；不含激活量化",
        rows=rows,
        note="与 I2 的 512 张 in-sample 口径不同：I2 含 A8 fake-quant，"
             "这里只量化权重；两者不可直接相减。",
    ), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  · 结果 → {out.relative_to(ROOT)}")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    skip_eval = "--skip-eval" in argv
    print("I2 混合精度落地测试\n" + "-" * 62)
    test_i2_plan()
    test_explicit_plan()
    test_package()
    test_cli_check()
    if skip_eval:
        print("\n[5] SKIP（--skip-eval）")
    else:
        test_accuracy_compare()
    print("\n" + "-" * 62)
    if FAILED:
        print(f"✗ {len(FAILED)} 项失败：")
        for f in FAILED:
            print(f"  - {f}")
        return 1
    print("✓ 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
