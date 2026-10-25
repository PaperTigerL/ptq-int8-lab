# -*- coding: utf-8 -*-
"""I2：逐层权重量化敏感度（W4/W8 + A8），两阶段采样。

协议（与 H-20261022-003 / M2 对齐，但显式加入激活量化）：
  1. 训练权重 → BN 折叠 → 同一份 512 张训练子集（get_loaders 的 calib）
  2. 用这 512 张做 A8 校准：per-tensor **对称 AbsMax**，scale=max(|lo|,|hi|)/127
  3. 基线 W32A8：权重 fp32 + 每层 conv 输入做 A8 fake-quant
  4. **阶段 A（排序）**：在前 64 张上逐层只量化该层权重
     （W8 per-channel / W4 per-group 32），其余层 fp32 + A8，得到粗排序；
     64 张的 top-1 分辨率是 1.5625 pp。
  5. **阶段 B（复核）**：取 W4/W8 粗排 top-N 的并集，在全部 512 张上重测；
     512 张的 top-1 分辨率是 0.195 pp。最终排序 = 复核层按 512 掉点排，
     其余层保持 64 张估计值并显式标记 basis。
  6. 混合精度计划在 512 张上做**真实组合评测**（不假设掉点可相加）。
  7. 选满足 target/budget 且 W8 层数最少的计划，输出层清单。

⛔ 只跑 CPU：导入 torch 前清空 CUDA_VISIBLE_DEVICES，并断言 CUDA 不可见。
⛔ 这是**校准子集内**的敏感度排序（in-sample），不是泛化精度估计。
"""
import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = {
    "resnet50": "/mnt/d/WorkBuddy/ShowBro/paper-code/ckpts/"
                "ckpt_resnet50_r50aug20_ep18.pth",
    "mobilenet_v2": "/mnt/d/WorkBuddy/ShowBro/paper-code/ckpts/"
                    "ckpt_mobilenet_v2_mv2aug20_ep19.pth",
}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="逐层权重量化敏感度（CPU-only，64 排序 + 512 复核）")
    ap.add_argument("--model", default="resnet50",
                    choices=["resnet50", "mobilenet_v2"])
    ap.add_argument("--weights", default=None,
                    help="默认用 DEFAULT_WEIGHTS 里的训练 checkpoint")
    ap.add_argument("--calib-n", type=int, default=512,
                    help="A8 校准张数，也是复核上限")
    ap.add_argument("--rank-n", type=int, default=64,
                    help="阶段 A 粗排张数")
    ap.add_argument("--recheck-n", type=int, default=512,
                    help="阶段 B 复核张数（不超过 calib-n）")
    ap.add_argument("--recheck-top-w4", type=int, default=10,
                    help="按 W4 粗排取前 N 层做 512 复核")
    ap.add_argument("--recheck-top-w8", type=int, default=6,
                    help="按 W8 粗排取前 N 层做 512 复核")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--w4-group", type=int, default=32)
    ap.add_argument("--budget-pp", type=float, default=3.0)
    ap.add_argument("--target-pp", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out-dir", default="results/i2")
    return ap.parse_args(argv)


def round_away(x):
    """half-away-from-zero，与 06 的 torch_round / 09 round_half_away 同语义。"""
    return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)


class ActivationQDQ:
    """在每个 Conv2d 的输入上做 per-tensor A8 fake-quant（AbsMax）。"""

    def __init__(self, scales):
        self.scales = scales
        self.enabled = True
        self.handles = []

    def attach(self, model):
        for name, mod in model.named_modules():
            if not isinstance(mod, nn.Conv2d):
                continue
            if name not in self.scales:
                raise KeyError(f"激活 scale 缺失：{name}")
            self.handles.append(
                mod.register_forward_pre_hook(self._make(name)))

    def _make(self, name):
        scale = float(self.scales[name])
        inv = 1.0 / scale

        def hook(_mod, args):
            if not self.enabled:
                return None
            q = args[0] * inv
            q = round_away(q)
            q = torch.clamp(q, -128, 127)
            q = q * scale
            return (q,) + tuple(args[1:])

        return hook

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def qdq_layer(mod, bits, scheme, group_size):
    """就地替换该层权重为 QDQ 后的 fp32 值；返回原值副本供恢复。"""
    orig = mod.weight.detach().clone()
    w = mod.weight.detach().cpu().numpy()
    q = quantize_weight_sym(w, bits, scheme, group_size)
    new = torch.from_numpy(q.dequant()).to(
        device=mod.weight.device, dtype=mod.weight.dtype)
    mod.weight.data.copy_(new.reshape_as(mod.weight))
    return orig


def restore_layer(mod, orig):
    mod.weight.data.copy_(orig)


def snapshot(convs):
    return {name: mod.weight.detach().clone() for name, mod in convs}


def restore_all(convs, snap):
    for name, mod in convs:
        mod.weight.data.copy_(snap[name])


def eval_acc(model, images, labels, batch_size):
    """在已物化的 (images, labels) 上测 top-1；inference_mode + channels_last。"""
    correct = 0
    total = int(labels.numel())
    with torch.inference_mode():
        for i in range(0, total, batch_size):
            x = images[i:i + batch_size]
            y = labels[i:i + batch_size]
            correct += int((model(x).argmax(1) == y).sum())
    return 100.0 * correct / max(total, 1)


def measure_layer(model, mod, bits, scheme, gs,
                  images, labels, batch_size):
    orig = qdq_layer(mod, bits, scheme, gs)
    try:
        return eval_acc(model, images, labels, batch_size)
    finally:
        restore_layer(mod, orig)


def collect_calib(calib_loader, calib_n):
    xs, ys = [], []
    for x, y in calib_loader:
        xs.append(x)
        ys.append(y)
    images = torch.cat(xs, dim=0)
    labels = torch.cat(ys, dim=0)
    if labels.numel() < calib_n:
        raise RuntimeError(
            f"校准集只有 {labels.numel()} 张，少于 --calib-n={calib_n}")
    return images[:calib_n], labels[:calib_n]


def main(argv=None):
    args = parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)

    global torch, nn, quantize_weight_sym
    import torch
    import torch.nn as nn

    from ptq_toolkit.models import get_model
    from ptq_toolkit.pipeline import (get_loaders, load_ptq_lab,
                                      load_trained_weights)
    from ptq_toolkit.quant.ptq import quantize_weight_sym

    if torch.cuda.is_available():
        raise SystemExit("[FAIL] CUDA 可见；本单要求纯 CPU，拒绝运行")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    weights = args.weights or DEFAULT_WEIGHTS[args.model]
    if not Path(weights).is_file():
        raise SystemExit(f"[FAIL] 找不到权重：{weights}")

    t_all = time.time()
    lab = load_ptq_lab()
    model = get_model(args.model)
    n_sd, miss, unexp = load_trained_weights(model, weights)
    if miss or unexp:
        raise SystemExit(
            f"[FAIL] 权重键集合不干净：missing={miss} unexpected={unexp}")
    model = lab["fold_all_bn"](model.to("cpu").eval()).to("cpu").eval()
    print(f"模型 {args.model} · 权重 {n_sd} 个张量 · BN 已折叠")

    calib_loader, _, _, _ = get_loaders(calib_n=args.calib_n, eval_n=1)
    images, labels = collect_calib(calib_loader, args.calib_n)
    n_calib = int(labels.numel())
    n_rank = min(args.rank_n, n_calib)
    n_recheck = min(args.recheck_n, n_calib)
    if n_rank < 1 or n_recheck < 1:
        raise SystemExit("[FAIL] rank-n/recheck-n 必须为正")
    print(f"校准 {n_calib} 张；阶段 A 粗排 {n_rank} 张；"
          f"阶段 B 复核 {n_recheck} 张（batch={args.batch_size}）")

    # ---- A8 校准：per-tensor 对称 AbsMax（用全部 512 张）----
    torch.manual_seed(args.seed)
    t0 = time.time()
    a_scales, a_zeros, store = lab["calibrate"](
        model, calib_loader, a_bits=8, symmetric=True)
    convs = [(n, m) for n, m in model.named_modules()
             if isinstance(m, nn.Conv2d)]
    if len(convs) != len(a_scales):
        raise RuntimeError(
            f"校准层数 {len(a_scales)} != conv 层数 {len(convs)}")
    if any(abs(float(a_zeros[n])) > 0 for n in a_scales):
        raise RuntimeError("对称 AbsMax 校准的 zero_point 应为 0")
    scales = {n: float(a_scales[n]) for n in a_scales}
    ranges = {}
    for name, _m in convs:
        v = store[name]
        ranges[name] = (float(v.min()), float(v.max()))
    print(f"A8 AbsMax 校准完成：{len(convs)} 层（{time.time() - t0:.1f} s）")

    # channels_last 在 CPU/MKLDNN 上快约 25%；校准完成后切换，输入一起转。
    model = model.to(memory_format=torch.channels_last)
    images = images.contiguous(memory_format=torch.channels_last)
    rank_images = images[:n_rank].contiguous(
        memory_format=torch.channels_last)
    rank_labels = labels[:n_rank]
    recheck_images = images[:n_recheck].contiguous(
        memory_format=torch.channels_last)
    recheck_labels = labels[:n_recheck]

    # ---- 基线 ----
    t0 = time.time()
    fp32_acc_rank = eval_acc(model, rank_images, rank_labels, args.batch_size)
    fp32_acc_recheck = eval_acc(
        model, recheck_images, recheck_labels, args.batch_size)
    print(f"fp32 基线：rank@{n_rank} {fp32_acc_rank:.4f}% · "
          f"recheck@{n_recheck} {fp32_acc_recheck:.4f}%"
          f"（{time.time() - t0:.1f} s）")

    aq = ActivationQDQ(scales)
    aq.attach(model)
    try:
        t0 = time.time()
        w32a8_rank = eval_acc(
            model, rank_images, rank_labels, args.batch_size)
        w32a8_recheck = eval_acc(
            model, recheck_images, recheck_labels, args.batch_size)
        print(f"W32A8 基线：rank@{n_rank} {w32a8_rank:.4f}% · "
              f"recheck@{n_recheck} {w32a8_recheck:.4f}%"
              f"（{time.time() - t0:.1f} s）")

        rows = []
        for idx, (name, mod) in enumerate(convs):
            rows.append({
                "index": idx,
                "name": name,
                "in_channels": int(mod.in_channels),
                "out_channels": int(mod.out_channels),
                "groups": int(mod.groups),
                "kernel": [int(mod.kernel_size[0]),
                           int(mod.kernel_size[1])],
                "K": int(mod.in_channels // mod.groups
                         * mod.kernel_size[0] * mod.kernel_size[1]),
                "depthwise": bool(mod.groups > 1
                                  and mod.groups == mod.in_channels),
                "downsample": "downsample" in name,
                "first": idx == 0,
                "last": idx == len(convs) - 1,
                "act_lo": round(ranges[name][0], 6),
                "act_hi": round(ranges[name][1], 6),
                "act_scale": round(scales[name], 9),
            })

        # ---- 阶段 A：64 张粗排（W8、W4 各一轮）----
        t0 = time.time()
        for idx, (name, mod) in enumerate(convs):
            rec = rows[idx]
            acc8 = measure_layer(
                model, mod, 8, "per_channel", 0,
                rank_images, rank_labels, args.batch_size)
            acc4 = measure_layer(
                model, mod, 4, "per_group", args.w4_group,
                rank_images, rank_labels, args.batch_size)
            rec["w8_acc_rank"] = round(acc8, 4)
            rec["w4_acc_rank"] = round(acc4, 4)
            rec["w8_drop_pp_rank"] = round(w32a8_rank - acc8, 4)
            rec["w4_drop_pp_rank"] = round(w32a8_rank - acc4, 4)
            print(f"  rank [{idx:02d}/{len(convs)}] {name:<42} "
                  f"W8 {rec['w8_drop_pp_rank']:+7.3f} pp  "
                  f"W4 {rec['w4_drop_pp_rank']:+7.3f} pp")
        print(f"阶段 A 完成：{len(convs)} 层 × 2 位宽 @ {n_rank} 张"
              f"（{time.time() - t0:.1f} s）")

        by_w4 = sorted(rows, key=lambda r: (-r["w4_drop_pp_rank"], r["index"]))
        by_w8 = sorted(rows, key=lambda r: (-r["w8_drop_pp_rank"], r["index"]))
        for rank, rec in enumerate(by_w4):
            rec["rank_w4_rank_n"] = rank + 1
        for rank, rec in enumerate(by_w8):
            rec["rank_w8_rank_n"] = rank + 1

        recheck_names = []
        for rec in (by_w4[:args.recheck_top_w4]
                    + by_w8[:args.recheck_top_w8]):
            if rec["name"] not in recheck_names:
                recheck_names.append(rec["name"])
        recheck_set = set(recheck_names)
        print(f"阶段 B 复核层：{len(recheck_names)} 层"
              f"（W4 top{args.recheck_top_w4} ∪ W8 top{args.recheck_top_w8}）")

        # ---- 阶段 B：512 张复核 ----
        t0 = time.time()
        by_name = {r["name"]: r for r in rows}
        for i, name in enumerate(recheck_names):
            rec = by_name[name]
            mod = dict(convs)[name]
            acc8 = measure_layer(
                model, mod, 8, "per_channel", 0,
                recheck_images, recheck_labels, args.batch_size)
            acc4 = measure_layer(
                model, mod, 4, "per_group", args.w4_group,
                recheck_images, recheck_labels, args.batch_size)
            rec["w8_acc_recheck"] = round(acc8, 4)
            rec["w4_acc_recheck"] = round(acc4, 4)
            rec["w8_drop_pp_recheck"] = round(w32a8_recheck - acc8, 4)
            rec["w4_drop_pp_recheck"] = round(w32a8_recheck - acc4, 4)
            print(f"  recheck [{i + 1:02d}/{len(recheck_names)}] {name:<42} "
                  f"W8 {rec['w8_drop_pp_recheck']:+7.3f} pp  "
                  f"W4 {rec['w4_drop_pp_recheck']:+7.3f} pp")
        print(f"阶段 B 完成（{time.time() - t0:.1f} s）")

        # 最终字段：复核层用 512 实测，其余层保留 64 估计并标记 basis。
        for rec in rows:
            if rec["name"] in recheck_set:
                rec["eval_n"] = n_recheck
                rec["basis"] = f"recheck@{n_recheck}"
                rec["w8_acc"] = rec["w8_acc_recheck"]
                rec["w4_acc"] = rec["w4_acc_recheck"]
                rec["w8_drop_pp"] = rec["w8_drop_pp_recheck"]
                rec["w4_drop_pp"] = rec["w4_drop_pp_recheck"]
            else:
                rec["eval_n"] = n_rank
                rec["basis"] = f"rank@{n_rank}"
                rec["w8_acc"] = rec["w8_acc_rank"]
                rec["w4_acc"] = rec["w4_acc_rank"]
                rec["w8_drop_pp"] = rec["w8_drop_pp_rank"]
                rec["w4_drop_pp"] = rec["w4_drop_pp_rank"]

        # 最终顺序：复核层按 512 W4 掉点排前面，其余按 64 估计排后面。
        final_order = (
            sorted((r for r in rows if r["name"] in recheck_set),
                   key=lambda r: (-r["w4_drop_pp"], r["index"]))
            + sorted((r for r in rows if r["name"] not in recheck_set),
                     key=lambda r: (-r["w4_drop_pp"], r["index"])))
        for rank, rec in enumerate(final_order):
            rec["rank"] = rank + 1

        # ---- 混合精度计划（512 张真实组合评测）----
        structural = {r["name"] for r in rows
                      if r["first"] or r["last"] or r["downsample"]
                      or r["depthwise"]}
        non_structural = [r for r in final_order
                          if r["name"] not in structural]
        top_names = [r["name"] for r in non_structural]
        top_all = [r["name"] for r in final_order]

        plan_specs = [("all_w4", set()), ("structural", set(structural))]
        for k in (4, 8, 12):
            if k < len(top_names):
                plan_specs.append((f"top{k}_only", set(top_names[:k])))
        for k in (4, 8):
            if k < len(top_all):
                plan_specs.append((f"top{k}_all", set(top_all[:k])))
            if k < len(top_names):
                plan_specs.append(
                    (f"structural_top{k}",
                     set(structural) | set(top_names[:k])))

        plans = []
        t0 = time.time()
        for plan_name, protected in plan_specs:
            snap = snapshot(convs)
            try:
                for name, mod in convs:
                    if name in protected:
                        qdq_layer(mod, 8, "per_channel", 0)
                    else:
                        qdq_layer(mod, 4, "per_group", args.w4_group)
                acc = eval_acc(
                    model, recheck_images, recheck_labels, args.batch_size)
            finally:
                restore_all(convs, snap)
            est = sum(by_name[n]["w4_drop_pp"]
                      for n in by_name if n not in protected)
            plans.append({
                "name": plan_name,
                "n_w8": len(protected),
                "n_w4": len(convs) - len(protected),
                "estimated_drop_pp": round(est, 4),
                "acc": round(acc, 4),
                "drop_pp": round(w32a8_recheck - acc, 4),
                "w8_layers": sorted(
                    protected, key=lambda n: by_name[n]["index"]),
            })
            print(f"  计划 {plan_name:<18} W8={len(protected):2d} "
                  f"acc={acc:.4f}%  掉点 {w32a8_recheck - acc:+7.3f} pp")
        print(f"计划评测完成（{time.time() - t0:.1f} s）")

        def pick(threshold):
            ok = [p for p in plans if p["drop_pp"] <= threshold]
            if not ok:
                return None
            return min(ok, key=lambda p: (p["n_w8"], p["drop_pp"]))

        rec = pick(args.target_pp)
        basis = f"target<={args.target_pp}pp, min W8 layers"
        if rec is None:
            rec = pick(args.budget_pp)
            basis = (f"target 不可达；budget<={args.budget_pp}pp, "
                     "min W8 layers")
        if rec is None:
            rec = min(plans, key=lambda p: (p["drop_pp"], p["n_w8"]))
            basis = "target/budget 均不可达；取掉点最小计划"
        best = min(plans, key=lambda p: (p["drop_pp"], p["n_w8"]))

        rec_set = set(rec["w8_layers"])
        recommended = {
            "plan": rec["name"],
            "selection_basis": basis,
            "budget_pp": args.budget_pp,
            "target_pp": args.target_pp,
            "n_w8": rec["n_w8"],
            "n_w4": rec["n_w4"],
            "acc": rec["acc"],
            "drop_pp": rec["drop_pp"],
            "estimated_drop_pp": rec["estimated_drop_pp"],
            "w8_layers": rec["w8_layers"],
            "w4_layers": [r["name"] for r in rows
                          if r["name"] not in rec_set],
            "best_accuracy_plan": best["name"],
            "best_accuracy_drop_pp": best["drop_pp"],
            "measured_on": f"{n_recheck} 张校准子集（in-sample）",
        }
    finally:
        aq.detach()

    # ---- 落盘 ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    top_sensitive = [{
        "rank": r["rank"],
        "name": r["name"],
        "index": r["index"],
        "w4_drop_pp": r["w4_drop_pp"],
        "w8_drop_pp": r["w8_drop_pp"],
        "eval_n": r["eval_n"],
        "basis": r["basis"],
        "structural": bool(r["first"] or r["last"]
                           or r["downsample"] or r["depthwise"]),
        "bits_in_recommended": 8 if r["name"] in rec_set else 4,
    } for r in final_order[:12]]

    doc = {
        "schema": "ptq/i2-layer-sensitivity@2",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": args.model,
        "weights": str(weights),
        "weights_tensors": n_sd,
        "device": "cpu",
        "threads": args.threads,
        "torch_version": torch.__version__,
        "cpu": platform.processor() or platform.machine(),
        "calib_n": n_calib,
        "rank_n": n_rank,
        "recheck_n": n_recheck,
        "recheck_top_w4": args.recheck_top_w4,
        "recheck_top_w8": args.recheck_top_w8,
        "recheck_layers": recheck_names,
        "eval_batch_size": args.batch_size,
        "sampling": (
            f"两阶段：先用前 {n_rank} 张（top-1 分辨率 "
            f"{100.0 / n_rank:.3f} pp）对全部 {len(convs)} 层做 W4/W8 粗排，"
            f"再取 W4 top{args.recheck_top_w4} ∪ W8 top{args.recheck_top_w8}"
            f"（{len(recheck_names)} 层）在前 {n_recheck} 张"
            f"（分辨率 {100.0 / n_recheck:.3f} pp）复核。"
            "最终排序优先使用 512 复核值；未复核层保留 64 估计值并标 basis。"),
        "calib_source": "get_loaders(calib_n) 的训练集前 N 张；与 A8 校准同一批",
        "activation": ("A8 per-tensor symmetric AbsMax; "
                       "scale=max(|lo|,|hi|)/127; zp=0"),
        "w8_scheme": "per_channel",
        "w4_scheme": "per_group",
        "w4_group_size": args.w4_group,
        "note": ("逐层只量化一层；W32A8 为基线；in-sample 评测，"
                 "不代表全量/GPU 精度。64 张粗排分辨率 1.5625 pp，"
                 "512 张复核分辨率 0.195 pp；未复核层的 drop 为 64 张估计值。"),
        "baselines": {
            "fp32_acc_rank": round(fp32_acc_rank, 4),
            "w32a8_acc_rank": round(w32a8_rank, 4),
            "a8_drop_pp_rank": round(fp32_acc_rank - w32a8_rank, 4),
            "fp32_acc_recheck": round(fp32_acc_recheck, 4),
            "w32a8_acc_recheck": round(w32a8_recheck, 4),
            "a8_drop_pp_recheck": round(fp32_acc_recheck - w32a8_recheck, 4),
            "fp32_acc": round(fp32_acc_recheck, 4),
            "w32a8_acc": round(w32a8_recheck, 4),
            "a8_drop_pp": round(fp32_acc_recheck - w32a8_recheck, 4),
        },
        "layers": [dict(r) for r in final_order],
        "top_sensitive": top_sensitive,
        "plans": plans,
        "recommendation": recommended,
        "runtime_s": round(time.time() - t_all, 1),
        "command": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
    }
    json_path = out_dir / f"layer_sensitivity_{args.model}.json"
    json_path.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = out_dir / f"layer_sensitivity_{args.model}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "name", "index", "rank", "basis", "eval_n",
            "w8_drop_pp_rank", "w4_drop_pp_rank",
            "w8_drop_pp_recheck", "w4_drop_pp_recheck",
            "w8_drop_pp", "w4_drop_pp",
            "w8_acc", "w4_acc", "act_scale", "act_lo", "act_hi",
            "groups", "K", "depthwise", "downsample", "first", "last",
            "bits_in_recommended"])
        for r in final_order:
            w.writerow([
                r["name"], r["index"], r["rank"], r["basis"], r["eval_n"],
                r["w8_drop_pp_rank"], r["w4_drop_pp_rank"],
                r.get("w8_drop_pp_recheck", ""),
                r.get("w4_drop_pp_recheck", ""),
                r["w8_drop_pp"], r["w4_drop_pp"],
                r["w8_acc"], r["w4_acc"], r["act_scale"], r["act_lo"],
                r["act_hi"], r["groups"], r["K"], int(r["depthwise"]),
                int(r["downsample"]), int(r["first"]), int(r["last"]),
                8 if r["name"] in rec_set else 4])

    print()
    print(f"推荐计划：{recommended['plan']} · W8 {recommended['n_w8']} 层 / "
          f"W4 {recommended['n_w4']} 层 · 掉点 {recommended['drop_pp']:+.3f} pp"
          f"（{recommended['selection_basis']}）")
    print("Top-12 敏感层（复核层按 512 掉点，其余按 64 估计）：")
    for t in top_sensitive:
        print(f"  {t['rank']:2d}. {t['name']:<42} "
              f"W4 {t['w4_drop_pp']:+7.3f} pp  "
              f"W8 {t['w8_drop_pp']:+7.3f} pp  [{t['basis']}]"
              f"{'  结构化' if t['structural'] else ''}")
    print(f"→ {json_path}")
    print(f"→ {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
