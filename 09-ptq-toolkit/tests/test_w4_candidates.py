# -*- coding: utf-8 -*-
"""W4 per-group / per-channel / 混合精度候选的单测与导出校验。

覆盖：
  1. per-group 量化的 scale/码字与逐组参考实现逐值一致
  2. 组间 scale 独立；group_size>=K 退化为 per-channel
  3. 混合精度计划：首/末/downsample/depthwise 的保护规则
  4. QDQ 后的权重确实落在 W4/W8 网格上（不是「名义 W4、实际 int8」）
  5. 真实导出产物：PTQ2 完整解析、PTQC 长度/层数、每层 bits 与码字范围
  6. 可选：调用 06 的 CPU probe（ptq_model_probe）做跨语言校验

用法：
    python tests/test_w4_candidates.py
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from ptq_toolkit.export.ptqc import read_ptqc                      # noqa: E402
from ptq_toolkit.export.ptq2_info import conv_weight, read_ptq2_info  # noqa: E402
from ptq_toolkit.models import get_model                           # noqa: E402
from ptq_toolkit.quant.mixed import (CANDIDATES, build_layer_plan,  # noqa: E402
                                     plan_summary)
from ptq_toolkit.quant.ptq import (quantize_per_channel_sym,        # noqa: E402
                                   quantize_per_group_sym,
                                   quantize_weight_sym, qmax_of,
                                   round_half_away)

FAILED = []
OUT = ROOT / "results" / "w4_candidates"
C06 = ROOT.parent / "06-gpu-int8" / "cpp"


def check(name, cond, detail=""):
    print(f"  {'✓' if cond else '✗'} {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILED.append(name)


# ---------------------------------------------------------------- 1) per-group

def _ref_group(w, bits, gs):
    """逐组参考实现（显式 for 循环，不复用被测代码的分组逻辑）。"""
    qmax = qmax_of(bits)
    flat = w.reshape(w.shape[0], -1).astype(np.float64)
    cout, k = flat.shape
    n_groups = (k + gs - 1) // gs
    scales = np.zeros((cout, n_groups), dtype=np.float64)
    codes = np.zeros((cout, k), dtype=np.int64)
    for c in range(cout):
        for g in range(n_groups):
            seg = flat[c, g * gs:(g + 1) * gs]
            amax = float(np.abs(seg).max()) if seg.size else 0.0
            s = max(amax / qmax, 1e-8)
            scales[c, g] = s
            codes[c, g * gs:g * gs + seg.size] = np.clip(
                round_half_away(seg / s), -(qmax + 1), qmax).astype(np.int64)
    return scales, codes


def test_per_group_semantics():
    print("\n[1] per-group 量化语义")
    rng = np.random.default_rng(0)
    w = (rng.standard_normal((3, 70)) * 4.0).astype(np.float32)
    for bits in (4, 8):
        for gs in (1, 16, 32, 64, 128):
            q = quantize_per_group_sym(w, bits, gs)
            ref_s, ref_c = _ref_group(w, bits, gs)
            check(f"W{bits} gs={gs} scale 逐值一致",
                  q.scale.shape == ref_s.shape and
                  np.allclose(q.scale, ref_s, rtol=1e-6, atol=1e-7),
                  f"{q.scale.shape}")
            check(f"W{bits} gs={gs} 码字逐值一致",
                  np.array_equal(q.code.reshape(w.shape[0], -1), ref_c))
            check(f"W{bits} gs={gs} dequant 形状",
                  q.dequant().shape == w.shape)


def test_group_independence():
    print("\n[2] 组间 scale 独立 / gs>=K 退化")
    w = np.zeros((2, 64), dtype=np.float32)
    w[0, :32] = 0.5
    w[0, 32:] = 8.0
    q1 = quantize_per_group_sym(w, 4, 32)
    w2 = w.copy()
    w2[0, 32:] = 80.0
    q2 = quantize_per_group_sym(w2, 4, 32)
    check("改第二组不影响第一组 scale",
          np.array_equal(q1.scale[0, 0:1], q2.scale[0, 0:1]))
    check("第二组 scale 随幅值变大",
          q2.scale[0, 1] > q1.scale[0, 1] * 5,
          f"{q1.scale[0,1]:.4g} -> {q2.scale[0,1]:.4g}")

    for gs in (64, 65, 1024):
        a = quantize_per_group_sym(w, 4, gs).dequant()
        b = quantize_per_channel_sym(w, 4).dequant()
        check(f"gs={gs} >= K 时与 per-channel 逐值一致",
              np.array_equal(a, b))
    try:
        quantize_per_group_sym(w, 4, 0)
        check("gs=0 应报错", False)
    except ValueError:
        check("gs=0 应报错", True)


def test_group_dispatch():
    print("\n[3] quantize_weight_sym 分派")
    w = np.random.default_rng(1).standard_normal((4, 40)).astype(np.float32)
    a = quantize_weight_sym(w, 4, "per_group", 16)
    b = quantize_per_group_sym(w, 4, 16)
    check("per_group 分派到同一实现",
          np.array_equal(a.code, b.code) and np.allclose(a.scale, b.scale))
    c = quantize_weight_sym(w, 4, "per_channel")
    d = quantize_per_channel_sym(w, 4)
    check("per_channel 分派到同一实现",
          np.array_equal(c.code, d.code) and np.allclose(c.scale, d.scale))
    try:
        quantize_weight_sym(w, 4, "unknown")
        check("未知 scheme 应报错", False)
    except ValueError:
        check("未知 scheme 应报错", True)


# ---------------------------------------------------------------- 4) 混合计划

def test_mixed_plans():
    print("\n[4] 混合精度计划")
    mv2 = get_model("mobilenet_v2")
    plan = build_layer_plan(mv2, CANDIDATES["w4a8_mixed_g32"])
    summary = plan_summary(plan)
    convs = [(n, m) for n, m in mv2.named_modules()
             if isinstance(m, torch.nn.Conv2d)]
    first, last = convs[0][0], convs[-1][0]
    check("MV2 首 conv W8", plan[first].bits == 8, first)
    check("MV2 末 conv W8", plan[last].bits == 8, last)
    dw = [n for n, m in convs if m.groups > 1 and m.groups == m.in_channels]
    check("MV2 depthwise 全部 W8",
          len(dw) == 17 and all(plan[n].bits == 8 for n in dw),
          f"{len(dw)} 层")
    pw = [n for n, m in convs
          if not (m.groups > 1 and m.groups == m.in_channels)]
    check("MV2 非 depthwise 中存在 W4 g32",
          any(plan[n].bits == 4 and plan[n].scheme == "per_group"
              and plan[n].group_size == 32 for n in pw))
    check("MV2 W8 层数 = 首 + 17 dw + 末 = 19",
          len(summary["w8_layers"]) == 19, str(len(summary["w8_layers"])))

    r50 = get_model("resnet50")
    p50 = build_layer_plan(r50, CANDIDATES["w4a8_mixed_g32"])
    s50 = plan_summary(p50)
    names = [n for n, m in r50.named_modules()
             if isinstance(m, torch.nn.Conv2d)]
    check("R50 首/末 conv W8",
          p50[names[0]].bits == 8 and p50[names[-1]].bits == 8)
    ds = [n for n in names if "downsample" in n]
    check("R50 downsample 全部 W8",
          len(ds) == 4 and all(p50[n].bits == 8 for n in ds), f"{len(ds)} 层")
    check("R50 中间层 W4 g32",
          any(p50[n].bits == 4 for n in names if "downsample" not in n
              and n not in (names[0], names[-1])))
    check("R50 W8 层数 = 首 + 4 ds + 末 = 6",
          len(s50["w8_layers"]) == 6, str(len(s50["w8_layers"])))

    for key in ("w8a8", "w4a8_ch", "w4a8_g32"):
        p = build_layer_plan(mv2, CANDIDATES[key])
        check(f"{key} 全部同配置",
              len(set((s.bits, s.scheme, s.group_size)
                      for s in p.values())) == 1)


# ---------------------------------------------------------------- 5) QDQ 网格

def test_qdq_grid():
    print("\n[5] QDQ 后权重落在 W4 网格")
    rng = np.random.default_rng(7)
    w = (rng.standard_normal((5, 96)) * 2.0).astype(np.float32)
    q = quantize_per_group_sym(w, 4, 32)
    d = q.dequant().reshape(5, -1)
    ok_grid, max_lev = True, 0
    for c in range(w.shape[0]):
        for g in range(q.scale.shape[1]):
            seg = d[c, g * 32:(g + 1) * 32]
            s = float(q.scale[c, g])
            code = np.rint(seg / s)
            max_lev = max(max_lev, len(np.unique(code)))
            if np.abs(seg - code * s).max() > 1e-5 * max(1.0, np.abs(w).max()):
                ok_grid = False
    check("每组最多 16 个码字（W4 对称码空间）", max_lev <= 16, f"max={max_lev}")
    check("反量化值全部落在 code*scale 网格", ok_grid)
    check("W4 码字在 [-8,7]",
          q.code.min() >= -8 and q.code.max() <= 7,
          f"[{q.code.min()},{q.code.max()}]")


# ---------------------------------------------------------------- 6) 真实产物

def _quant_entries(cfg):
    return list(cfg["weights"].items())


def _check_export_pair(model_key, cand_key):
    binp = OUT / f"{model_key}_{cand_key}.ptq2.bin"
    ptqcp = OUT / f"calib_{model_key}_{cand_key}.ptqc"
    cfgp = OUT / f"quant_config_{model_key}_{cand_key}.json"
    if not (binp.is_file() and ptqcp.is_file() and cfgp.is_file()):
        print(f"  · SKIP {model_key}/{cand_key}（产物缺失）")
        return
    info = read_ptq2_info(binp)
    ranges = read_ptqc(ptqcp)
    cfg = json.loads(cfgp.read_text(encoding="utf-8"))
    entries = _quant_entries(cfg)

    check(f"{model_key}/{cand_key} PTQ2 完整解析",
          info["n_conv"] == len(entries) and info["n_block"] > 0,
          f"{info['n_conv']} conv / {info['n_block']} block")
    check(f"{model_key}/{cand_key} PTQC 层数一致",
          len(ranges) == info["n_conv"],
          f"{len(ranges)} vs {info['n_conv']}")
    check(f"{model_key}/{cand_key} PTQC 长度精确 8+8n",
          ptqcp.stat().st_size == 8 + 8 * len(ranges))
    check(f"{model_key}/{cand_key} PTQC 有限且 lo<=hi",
          all(np.isfinite(lo) and np.isfinite(hi) and lo <= hi
              for lo, hi in ranges))

    data = binp.read_bytes()
    ok_grid, ok_range, ok_meta = True, True, True
    n_w4 = 0
    for i, (name, wcfg) in enumerate(entries):
        conv = info["convs"][i]
        if wcfg.get("bits") not in (4, 8):
            ok_meta = False
            continue
        if wcfg["bits"] == 4:
            n_w4 += 1
        w = conv_weight(data, conv).reshape(conv["cout"], -1)
        scale = np.asarray(wcfg["scale"], dtype=np.float32)
        if wcfg.get("scheme") == "per_group":
            gs = int(wcfg["group_size"])
            ng = scale.shape[1]
            if wcfg.get("scale_shape") != [int(conv["cout"]), ng]:
                ok_meta = False
                continue
            for g in range(ng):
                seg = w[:, g * gs:(g + 1) * gs]
                s = scale[:, g:g + 1]
                code = round_half_away(seg / s)
                qmax = 7 if wcfg["bits"] == 4 else 127
                if code.min() < -(qmax + 1) or code.max() > qmax:
                    ok_range = False
                if np.abs(seg - code * s).max() > 1e-5 * max(1.0, float(np.abs(seg).max())):
                    ok_grid = False
        else:
            if scale.shape != (conv["cout"],):
                ok_meta = False
                continue
            code = round_half_away(w / scale[:, None])
            qmax = 7 if wcfg["bits"] == 4 else 127
            if code.min() < -(qmax + 1) or code.max() > qmax:
                ok_range = False
            if np.abs(w - code * scale[:, None]).max() > 1e-5 * max(
                    1.0, float(np.abs(w).max())):
                ok_grid = False
    check(f"{model_key}/{cand_key} 码字范围合法", ok_range)
    check(f"{model_key}/{cand_key} 权重落在声明的 bits 网格", ok_grid)
    check(f"{model_key}/{cand_key} scale 形状与 scheme 一致", ok_meta)
    if cand_key == "w8a8":
        check(f"{model_key}/{cand_key} 基线不含 W4 层", n_w4 == 0, f"{n_w4} 层")
    else:
        check(f"{model_key}/{cand_key} 存在 W4 层", n_w4 > 0, f"{n_w4} 层")

    if model_key == "mobilenet_v2":
        check(f"{model_key}/{cand_key} MV2 语义计数",
              info["n_depthwise"] == 17 and info["n_relu6"] == 35
              and info["n_residual2"] == 10,
              f"dw={info['n_depthwise']} relu6={info['n_relu6']} "
              f"res2={info['n_residual2']}")
    if model_key == "resnet50":
        check(f"{model_key}/{cand_key} R50 语义计数",
              info["n_conv"] == 53 and info["n_block"] == 16
              and info["n_residual1"] == 16,
              f"{info['n_conv']}/{info['n_block']}/{info['n_residual1']}")


def test_baseline_conv_matches_m2_accept():
    """新导出链路的 W8/W4 per-channel conv 权重应与 M2 验收产物逐位一致。

    旧 pipeline 会连 fc 一起 QDQ，新导出有意保持 fc fp32（PTQ2 的 fc 走
    引擎 fp32 cuBLAS），所以只比对 conv 部分。
    """
    print("\n[6] 与 m2_accept 既有产物的 conv 权重对齐")
    pairs = [
        ("resnet50", "w8a8", ROOT / "results/m2_accept/resnet50_folded_w8a8.bin"),
        ("resnet50", "w4a8_ch", ROOT / "results/m2_accept/resnet50_folded_w4a8.bin"),
        ("mobilenet_v2", "w8a8",
         ROOT / "results/m2_accept/mobilenet_v2_folded_w8a8.bin"),
        ("mobilenet_v2", "w4a8_ch",
         ROOT / "results/m2_accept/mobilenet_v2_folded_w4a8.bin"),
    ]
    for model, cand, old in pairs:
        new = OUT / f"{model}_{cand}.ptq2.bin"
        if not (old.is_file() and new.is_file()):
            print(f"  · SKIP {model}/{cand}（缺 {old.name} 或新产物）")
            continue
        a = read_ptq2_info(old)
        b = read_ptq2_info(new)
        same_meta = all(
            (x["cout"], x["cin"], x["kh"], x["kw"], x["relu"], x["groups"])
            == (y["cout"], y["cin"], y["kh"], y["kw"], y["relu"], y["groups"])
            for x, y in zip(a["convs"], b["convs"]))
        da, db = old.read_bytes(), new.read_bytes()
        max_diff = max(
            float(np.abs(conv_weight(da, x) - conv_weight(db, y)).max())
            for x, y in zip(a["convs"], b["convs"]))
        check(f"{model}/{cand} conv 元信息一致", same_meta)
        check(f"{model}/{cand} conv 权重逐位一致", max_diff == 0.0,
              f"max|diff|={max_diff:g}")


def test_real_exports():
    print("\n[7] 真实导出产物")
    if not OUT.is_dir():
        print(f"  · SKIP：{OUT} 不存在")
        return
    for model_key in ("resnet50", "mobilenet_v2"):
        for cand_key in CANDIDATES:
            _check_export_pair(model_key, cand_key)


def test_m2_accept_ptqc_pairing():
    """M2 验收既有 PTQ2 与本次补出的 PTQC 必须按层数/大小严格配对。"""
    print("\n[8] m2_accept 既有 bin + 本次 PTQC 配对")
    rows = [
        ("resnet50", "w8a8", 53, 432),
        ("resnet50", "w4a8", 53, 432),
        ("mobilenet_v2", "w8a8", 52, 424),
        ("mobilenet_v2", "w4a8", 52, 424),
    ]
    for model, bits, n_conv, n_bytes in rows:
        binp = ROOT / "results/m2_accept" / f"{model}_folded_{bits}.bin"
        ptqcp = ROOT / "results/m2_accept" / f"calib_{model}_folded_{bits}.ptqc"
        if not (binp.is_file() and ptqcp.is_file()):
            print(f"  · SKIP {model}/{bits}")
            continue
        info = read_ptq2_info(binp)
        ranges = read_ptqc(ptqcp)
        check(f"m2_accept {model}/{bits} n_conv={n_conv}",
              info["n_conv"] == n_conv, str(info["n_conv"]))
        check(f"m2_accept {model}/{bits} PTQC {n_bytes} B / {n_conv} 层",
              ptqcp.stat().st_size == n_bytes and len(ranges) == n_conv,
              f"{ptqcp.stat().st_size} B / {len(ranges)}")


def test_cpp_probe():
    print("\n[9] 06 CPU probe 跨语言校验")
    probe = C06 / "build" / "ptq_model_probe"
    if not probe.is_file():
        print(f"  · SKIP：{probe} 未构建")
        return
    if not OUT.is_dir():
        print(f"  · SKIP：{OUT} 不存在")
        return
    for binp in sorted(OUT.glob("*.ptq2.bin")):
        ptqcp = OUT / f"calib_{binp.name[:-len('.ptq2.bin')]}.ptqc"
        if not ptqcp.is_file():
            check(f"{binp.name} 对应 PTQC 存在", False)
            continue
        r = subprocess.run([str(probe), str(binp), str(ptqcp)],
                           capture_output=True, text=True)
        check(f"{binp.name} probe rc=0", r.returncode == 0,
              (r.stdout + r.stderr).strip().splitlines()[-1]
              if (r.stdout or r.stderr) else "")


def main() -> int:
    torch.manual_seed(0)
    print("W4 候选 / 导出校验\n" + "-" * 62)
    test_per_group_semantics()
    test_group_independence()
    test_group_dispatch()
    test_mixed_plans()
    test_qdq_grid()
    test_baseline_conv_matches_m2_accept()
    test_real_exports()
    test_m2_accept_ptqc_pairing()
    test_cpp_probe()
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
