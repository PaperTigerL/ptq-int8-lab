#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MobileNetV2 W4 mixed-precision study harness (CPU-only, ASCII output).

Why this file exists
--------------------
The I2 sensitivity list (results/i2/layer_sensitivity_mobilenet_v2.json) ranked
layers by "quantize ONE layer to W4, keep the rest fp32".  That metric has two
problems for MV2:
  * resolution is 1/n_test (n=64 -> 1.56 pp), so the 64-image ranking is noise;
  * it never measures the *accumulated* damage of many layers at W4, which is
    what the delivered package actually does.

This harness measures the quantity the package is judged on: full-network
weight-only QDQ accuracy on the SAME test prefix used by "ptq eval".
It also covers the scheme grid (per-channel / per-group sizes / symmetric vs
asymmetric) and records the metadata cost of each option.

Modes
-----
  grid      all-W4 under a scheme grid -> accuracy drop + effective bits/weight
  marginal  damage of "everything W8 except layer L at W4" (leave-one-out)
  greedy    round-wise greedy: flip the least damaging layers to W4, re-measure
  curve     cumulative accuracy for the k least damaging layers kept at W8
  plan      evaluate one concrete plan (w8 layer list + W4 scheme)

Every mode writes a JSON evidence file; stdout stays ASCII so Windows/WSL
console encodings cannot mangle it.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CKPT_DEFAULT = ("/mnt/d/WorkBuddy/ShowBro/paper-code/ckpts/"
                "ckpt_mobilenet_v2_mv2img10_ep9.pth")


# ------------------------------------------------------------------ quantizers
def round_half_away(x):
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def qdq_sym(w, bits, scheme="per_channel", gs=0):
    """Symmetric QDQ. Mirrors ptq_toolkit.quant.ptq (per_channel / per_group)."""
    from ptq_toolkit.quant.ptq import (quantize_per_channel_sym,
                                       quantize_per_group_sym)
    if scheme == "per_channel":
        return quantize_per_channel_sym(w, bits).dequant()
    return quantize_per_group_sym(w, bits, gs).dequant()


def qdq_asym(w, bits, gs=0):
    """Asymmetric (zero-point) QDQ along the flattened K axis.

    gs=0 -> one (scale, zero) per output channel (per-channel asym).
    Otherwise groups of gs along K (per-group asym).
    Not used by any exported package yet: the 06 engine only implements
    symmetric int8.  Measured here to answer "does the zero point matter?".
    """
    qmax = (1 << bits) - 1
    wf = w.reshape(w.shape[0], -1).astype(np.float64)
    cout, k = wf.shape
    if gs and gs > 0:
        ng = (k + gs - 1) // gs
        pad = np.zeros((cout, ng * gs), dtype=np.float64)
        pad[:, :k] = wf
        groups = pad.reshape(cout, ng, gs)
        span = ng * gs
    else:
        groups = wf[:, None, :]
        span = k
    lo = groups.min(axis=2)
    hi = groups.max(axis=2)
    scale = np.maximum((hi - lo) / qmax, 1e-12)
    zero = np.clip(round_half_away(-lo / scale), 0, qmax)
    code = np.clip(round_half_away(groups / scale[..., None] + zero[..., None]),
                   0, qmax)
    deq = (code - zero[..., None]) * scale[..., None]
    deq = deq.reshape(cout, span)[:, :k]
    return deq.reshape(w.shape)


def qdq(w, key):
    """key = (kind, bits, scheme, gs) with kind in {sym, asym}."""
    kind, bits, scheme, gs = key
    if kind == "asym":
        return qdq_asym(w, bits, gs)
    return qdq_sym(w, bits, scheme, gs)


def key_name(key):
    kind, bits, scheme, gs = key
    if kind == "asym":
        return "W%d_asym_%s" % (bits, ("per_group%d" % gs) if gs else "per_channel")
    return "W%d_%s" % (bits, ("per_group%d" % gs) if scheme == "per_group" else "per_channel")


def key_cost(weight_shapes, key):
    """Effective bits/weight including fp32 scale (+ int8 zero point if asym).

    weight_shapes: list of (name, numel, cout, k) for every conv.
    """
    kind, bits, scheme, gs = key
    tot_w = 0
    tot_bits = 0.0
    for _n, numel, cout, k in weight_shapes:
        if scheme == "per_group" and gs and k > gs:
            n_scales = cout * ((k + gs - 1) // gs)
        else:
            n_scales = cout
        n_zero = n_scales if kind == "asym" else 0
        tot_w += numel
        tot_bits += bits * numel + 32.0 * n_scales + 8.0 * n_zero
    return tot_bits / max(tot_w, 1)


# ------------------------------------------------------------------ model setup
LAB = None


def setup(args):
    global LAB
    import torch
    from ptq_toolkit.pipeline import (get_loaders, load_ptq_lab,
                                     load_trained_weights)
    from ptq_toolkit.models import get_model

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)

    LAB = load_ptq_lab()
    model = get_model("mobilenet_v2")
    n_sd, miss, unexp = load_trained_weights(model, args.weights)
    if miss or unexp:
        raise SystemExit("ckpt key set unclean: missing=%s unexpected=%s"
                         % (miss[:3], unexp[:3]))
    model = LAB["fold_all_bn"](model.eval()).eval()
    _calib, test, _ntr, n_test = get_loaders(calib_n=1, eval_n=args.eval_n)
    if n_test != args.eval_n:
        raise SystemExit("test loader has %d samples, wanted %d"
                         % (n_test, args.eval_n))
    print("ckpt: %s (%d tensors)" % (args.weights, n_sd))
    print("eval_n=%d threads=%d device=cpu" % (args.eval_n, args.threads))
    return model, test


def convs_of(model):
    import torch
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, torch.nn.Conv2d)]


def weight_shapes(convs):
    out = []
    for name, mod in convs:
        k = int(np.prod(mod.weight.shape[1:]))
        out.append((name, int(mod.weight.numel()), int(mod.weight.shape[0]), k))
    return out


def build_cache(convs, keys):
    import torch
    cache = {}
    t0 = time.time()
    for key in keys:
        d = {}
        for name, mod in convs:
            w = mod.weight.detach().cpu().numpy().astype(np.float64)
            dq = qdq(w, key)
            d[name] = torch.from_numpy(
                np.ascontiguousarray(dq)).to(mod.weight.dtype).reshape(
                    mod.weight.shape)
        cache[key] = d
    print("qdq cache: %d keys x %d layers in %.1f s"
          % (len(keys), len(convs), time.time() - t0))
    return cache


def set_plan(model, cache, plan):
    """plan: dict layer_name -> qdq key."""
    for name, mod in convs_of(model):
        mod.weight.data.copy_(cache[plan[name]][name])


def evaluate(model, test):
    return 100.0 * float(LAB["evaluate"](model, test, device="cpu"))


def ce_loss(model, test):
    """Mean cross entropy on the same test prefix.

    Accuracy on 512 images moves in 0.195 pp steps, so ranking layers by it
    produces a lot of ties.  CE is continuous and monotone-ish with accuracy,
    which makes it a better *selection* signal; accuracy stays the hard gate.
    """
    import torch
    import torch.nn.functional as F
    tot, n = 0.0, 0
    with torch.no_grad():
        for x, y in test:
            out = model(x)
            tot += float(F.cross_entropy(out, y, reduction="sum"))
            n += int(y.numel())
    return tot / max(n, 1)


# ------------------------------------------------------------------ modes
def mode_grid(args, model, test):
    specs = []
    for kind in args.kinds:
        for scheme_gs in args.schemes:
            if scheme_gs in ("pc", "per_channel"):
                scheme, gs = "per_channel", 0
            else:
                scheme, gs = "per_group", int(scheme_gs.lstrip("g_"))
            specs.append((kind, 4, scheme, gs))
    specs.append(("sym", 8, "per_channel", 0))
    convs = convs_of(model)
    cache = build_cache(convs, specs)
    shapes = weight_shapes(convs)
    base = evaluate(model, test)
    base_loss = ce_loss(model, test)
    print("fp32 %.4f%%  ce %.5f (n=%d)" % (base, base_loss, args.eval_n))
    rows = []
    for key in specs:
        set_plan(model, cache, {n: key for n, _ in convs})
        acc = evaluate(model, test)
        row = dict(config=key_name(key), acc=round(acc, 4),
                   drop=round(base - acc, 4),
                   ce=round(ce_loss(model, test), 6),
                   eff_bits=round(key_cost(shapes, key), 3))
        rows.append(row)
        print("  %-22s %7.4f%%  drop %6.4f pp  eff_bits %.2f"
              % (row["config"], acc, base - acc, row["eff_bits"]))
    doc = dict(mode="grid", model="mobilenet_v2", weights=args.weights,
               eval_n=args.eval_n, fp32_acc=round(base, 4),
               protocol="train weights -> BN fold -> test prefix -> per-conv "
                        "QDQ (fc fp32) -> evaluate; weight-only, no act quant",
               rows=rows)
    write(args, "w4_scheme_grid_n%d" % args.eval_n, doc)


def mode_marginal(args, model, test):
    key = parse_key(args.w4)
    w8 = ("sym", 8, "per_channel", 0)
    convs = convs_of(model)
    names = [n for n, _ in convs]
    cache = build_cache(convs, [w8, key])
    base = evaluate(model, test)          # all W8
    base_loss = ce_loss(model, test)
    print("all-W8 baseline %.4f%%  ce %.5f" % (base, base_loss))
    rows = []
    t0 = time.time()
    for i, name in enumerate(names):
        plan = {n: w8 for n in names}
        plan[name] = key
        set_plan(model, cache, plan)
        acc = evaluate(model, test)
        loss = ce_loss(model, test)
        rows.append(dict(name=name, acc=round(acc, 4), damage=round(base - acc, 4),
                         ce=round(loss, 6), dce=round(loss - base_loss, 6)))
        print("[%2d/%d] %-24s %7.4f%%  damage %6.4f pp  dce %+.5f  (%.0fs)"
              % (i + 1, len(names), name, acc, base - acc, loss - base_loss,
                 time.time() - t0))
    rows.sort(key=lambda r: (r["damage"], r["dce"]))
    for i, r in enumerate(rows):
        r["rank"] = i + 1
    doc = dict(mode="marginal", model="mobilenet_v2", weights=args.weights,
               eval_n=args.eval_n, w4_config=key_name(key), w8_acc=round(base, 4),
               protocol="all convs W8 per-channel, one layer at %s; damage = "
                        "acc(all W8) - acc(plan)" % key_name(key),
               rows=rows)
    write(args, "marginal_%s_n%d" % (key_name(key), args.eval_n), doc)


def mode_curve(args, model, test):
    key = parse_key(args.w4)
    w8 = ("sym", 8, "per_channel", 0)
    convs = convs_of(model)
    names = [n for n, _ in convs]
    doc = json.loads(Path(args.marginal).read_text(encoding="utf-8"))
    # 注意方向：order 是「先保护谁」的顺序 —— 伤害最大的排在最前面。
    # 升序会把最无害的层先抬到 W8，曲线几乎不动（踩过一次）。
    if args.order == "dce":
        keyf = lambda r: (-r["dce"], -r["damage"])
    elif args.order == "damage":
        keyf = lambda r: (-r["damage"], -r["dce"])
    elif args.order == "depth":
        keyf = lambda r: names.index(r["name"])
    else:
        raise SystemExit("bad --order %r" % args.order)
    order = [r["name"] for r in sorted(doc["rows"], key=keyf)]
    print("order=%s first 8: %s" % (args.order, ",".join(order[:8])))
    cache = build_cache(convs, [w8, key])
    base = evaluate(model, test)
    print("w8 baseline %.4f%%" % base)
    rows = []
    for k in range(0, min(len(order), args.max_w8) + 1):
        w8set = set(order[:k])
        plan = {n: (w8 if n in w8set else key) for n in names}
        set_plan(model, cache, plan)
        acc = evaluate(model, test)
        loss = ce_loss(model, test)
        rows.append(dict(n_w8=k, acc=round(acc, 4), drop=round(base - acc, 4),
                         ce=round(loss, 6), w8_layers=order[:k]))
        print("  k=%2d W8  %7.4f%%  drop %6.4f pp  ce %.5f" % (k, acc, base - acc, loss))
    out = dict(mode="curve", model="mobilenet_v2", weights=args.weights,
               eval_n=args.eval_n, w4_config=key_name(key), order=args.order,
               n_conv=len(names), w8_acc=round(base, 4),
               protocol="train weights -> BN fold -> test prefix -> per-conv "
                        "QDQ (fc fp32) -> evaluate; weight-only",
               marginal_source=args.marginal, rows=rows)
    write(args, "curve_%s_%s_n%d" % (key_name(key), args.order, args.eval_n), out)


def mode_greedy(args, model, test):
    """Round-wise greedy: each round, measure damage of every remaining W4
    candidate against the CURRENT plan, move the cheapest ones to W8, then
    measure the exact plan."""
    key = parse_key(args.w4)
    w8 = ("sym", 8, "per_channel", 0)
    convs = convs_of(model)
    names = [n for n, _ in convs]
    cache = build_cache(convs, [w8, key])
    base = evaluate(model, test)
    print("all-W8 baseline %.4f%%  target drop <= %.2f pp"
          % (base, args.target_pp))
    w8set = set()
    history = []
    t0 = time.time()
    while True:
        cands = [n for n in names if n not in w8set]
        scored = []
        for name in cands:
            plan = {n: (w8 if n in w8set else key) for n in names}
            plan[name] = w8
            set_plan(model, cache, plan)
            scored.append((base - evaluate(model, test), name))
        scored.sort()
        picked = [name for _d, name in scored[:args.per_round]]
        w8set |= set(picked)
        plan = {n: (w8 if n in w8set else key) for n in names}
        set_plan(model, cache, plan)
        acc = evaluate(model, test)
        row = dict(n_w8=len(w8set), acc=round(acc, 4),
                   drop=round(base - acc, 4),
                   added=picked,
                   marginal=[dict(name=n, damage=round(d, 4)) for d, n in scored])
        history.append(row)
        print("round: k=%2d W8  %7.4f%%  drop %6.4f pp  (+%s)  (%.0fs)"
              % (len(w8set), acc, base - acc, ",".join(picked),
                 time.time() - t0))
        if base - acc <= args.target_pp or len(w8set) == len(names):
            break
    doc = dict(mode="greedy", model="mobilenet_v2", weights=args.weights,
               eval_n=args.eval_n, w4_config=key_name(key),
               w8_acc=round(base, 4), target_pp=args.target_pp,
               per_round=args.per_round, history=history)
    write(args, "greedy_%s_n%d" % (key_name(key), args.eval_n), doc)


def mode_plan(args, model, test):
    key = parse_key(args.w4)
    w8 = ("sym", 8, "per_channel", 0)
    convs = convs_of(model)
    names = [n for n, _ in convs]
    if args.plan:
        pdoc = json.loads(Path(args.plan).read_text(encoding="utf-8"))
        w8_layers = pdoc.get("w8_layers")
        if not isinstance(w8_layers, list):
            raise SystemExit("%s has no w8_layers list" % args.plan)
    else:
        w8_layers = [s.strip() for s in args.w8_layers.split(",") if s.strip()]
    unknown = set(w8_layers) - set(names)
    if unknown:
        raise SystemExit("not a conv layer: %s" % sorted(unknown)[:5])
    cache = build_cache(convs, [w8, key])
    base = evaluate(model, test)
    plan = {n: (w8 if n in set(w8_layers) else key) for n in names}
    set_plan(model, cache, plan)
    acc = evaluate(model, test)
    print("plan: W8 %d / W4 %d (%s)  eval_n=%d"
          % (len(w8_layers), len(names) - len(w8_layers), key_name(key),
             args.eval_n))
    print("fp32 %.4f%%  quant %.4f%%  drop %.4f pp  budget %.2f pp -> %s"
          % (base, acc, base - acc, args.budget_pp,
             "PASS" if base - acc <= args.budget_pp else "FAIL"))
    doc = dict(mode="plan", model="mobilenet_v2", weights=args.weights,
               eval_n=args.eval_n, w4_config=key_name(key),
               fp32_acc=round(base, 4), quant_acc=round(acc, 4),
               drop=round(base - acc, 4), budget_pp=args.budget_pp,
               passed=bool(base - acc <= args.budget_pp),
               n_w8=len(w8_layers), w8_layers=w8_layers,
               protocol="train weights -> BN fold -> test prefix -> per-conv "
                        "QDQ (fc fp32) -> evaluate; weight-only")
    write(args, "plan_%s_n%d" % (args.tag or "adhoc", args.eval_n), doc)
    return doc


def mode_mkplan(args, model, test):
    """Turn a measured curve into an I2-compatible plan JSON.

    Rule: smallest k (fewest W8 layers) whose *measured* drop on the curve
    dataset is within the target.  No model needed - pure bookkeeping over
    numbers that were already measured, so the plan cannot drift from the
    evidence it cites.
    """
    doc = json.loads(Path(args.curve).read_text(encoding="utf-8"))
    rows = sorted(doc["rows"], key=lambda r: r["n_w8"])
    by_k = {r["n_w8"]: r for r in rows}
    if args.k is not None:
        if args.k not in by_k:
            raise SystemExit("curve has no k=%d row (max %d)"
                             % (args.k, max(by_k)))
        pick = by_k[args.k]
    else:
        pick = None
        for r in rows:
            if r["drop"] <= args.target_pp and r["n_w8"] > 0:
                pick = r
                break
        if pick is None:
            raise SystemExit("curve never reaches target %.2f pp (best %.4f)"
                             % (args.target_pp, min(r["drop"] for r in rows)))
    key = parse_key(args.w4)
    name = args.plan_name or ("w4g%d_w8top%d" % (key[3], pick["n_w8"]))
    # 复核证据（n=2000 / 全量）必须与这份计划同一层名单，否则拒绝合并 ——
    # 让「计划 JSON 里引用的掉点」不可能来自别的计划。
    verified = []
    for vp in (args.verify or []):
        v = json.loads(Path(vp).read_text(encoding="utf-8"))
        if list(v.get("w8_layers") or []) != list(pick["w8_layers"]):
            raise SystemExit("%s: w8 layer list differs from the curve row" % vp)
        verified.append(dict(source=str(vp), eval_n=v["eval_n"],
                             acc=v["quant_acc"], drop=v["drop"],
                             budget_pp=v.get("budget_pp"),
                             passed=v.get("passed")))
    verified.sort(key=lambda v: v["eval_n"])
    plan = dict(name=name, n_w8=pick["n_w8"],
                n_w4=int(doc.get("n_conv", 52)) - pick["n_w8"],
                w4_scheme=key[2], w4_group_size=key[3],
                eval_n=doc["eval_n"], acc=pick["acc"], drop_pp=pick["drop"],
                basis="measured on the same protocol as the delivered package "
                      "(test prefix, weight-only QDQ, fc fp32)",
                verified=verified,
                w8_layers=pick["w8_layers"])
    out = dict(
        schema="ptq/mixed-plan@1",
        model="mobilenet_v2",
        weights=doc["weights"],
        protocol=doc.get("protocol") or (
            "train weights -> BN fold -> test prefix -> per-conv QDQ "
            "(fc fp32) -> evaluate; weight-only, no act quant"),
        selection_rule=args.rule or (
            "smallest n_w8 whose measured drop <= %.2f pp on n=%d; order = "
            "per-layer marginal CE damage (all-W8 baseline, one layer at W4) "
            "descending" % (args.target_pp, doc["eval_n"])),
        source_curve=str(args.curve),
        plans=[plan],
        recommended=dict(plan=name, n_w8=plan["n_w8"], n_w4=plan["n_w4"],
                         drop_pp=plan["drop_pp"], acc=plan["acc"],
                         w8_layers=plan["w8_layers"],
                         selection_basis="measured, not estimated"),
    )
    write(args, args.tag or ("plan_mobilenet_v2_" + name), out)
    print("plan %s: W8 %d / W4 %d %s g%d  acc %.4f%%  drop %.4f pp (n=%d)"
          % (name, plan["n_w8"], plan["n_w4"], key[2], key[3],
             plan["acc"], plan["drop_pp"], doc["eval_n"]))


def parse_key(spec):
    """spec: 'pc' / 'g32' / 'pc_asym' / 'g16_asym' (all W4)."""
    kind = "asym" if spec.endswith("_asym") else "sym"
    body = spec[:-5] if kind == "asym" else spec
    if body in ("pc", "per_channel"):
        return (kind, 4, "per_channel", 0)
    if body.startswith("g"):
        return (kind, 4, "per_group", int(body[1:]))
    raise SystemExit("bad --w4 spec: %r" % spec)


def write(args, stem, doc):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / ("%s.json" % stem)
    p.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print("evidence -> %s" % p)


def main(argv=None):
    ap = argparse.ArgumentParser(description="MV2 W4 mixed-precision study")
    ap.add_argument("mode", choices=["grid", "marginal", "greedy", "curve",
                                    "plan", "mkplan"])
    ap.add_argument("--weights", default=CKPT_DEFAULT)
    ap.add_argument("--eval-n", type=int, default=512)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out-dir", default="results/i3")
    ap.add_argument("--tag", default="")
    ap.add_argument("--w4", default="pc", help="pc | g16 | g32 | g64 | pc_asym | g32_asym")
    ap.add_argument("--kinds", nargs="*", default=["sym", "asym"])
    ap.add_argument("--schemes", nargs="*", default=["pc", "g16", "g32", "g64", "g128"])
    ap.add_argument("--marginal", default=None)
    ap.add_argument("--curve", default=None)
    ap.add_argument("--k", type=int, default=None,
                    help="mkplan: fixed W8 count instead of the smallest")
    ap.add_argument("--verify", nargs="*", default=None,
                    help="mkplan: plan-mode JSONs (n=2000/full) to merge")
    ap.add_argument("--rule", default="", help="mkplan: selection rule text")
    ap.add_argument("--plan-name", default=None,
                    help="mkplan: plan key written into the JSON")
    ap.add_argument("--plan", default=None)
    ap.add_argument("--w8-layers", default="")
    ap.add_argument("--max-w8", type=int, default=24)
    ap.add_argument("--target-pp", type=float, default=1.0)
    ap.add_argument("--budget-pp", type=float, default=1.0)
    ap.add_argument("--per-round", type=int, default=4)
    ap.add_argument("--order", default="dce", choices=["dce", "damage", "depth"])
    args = ap.parse_args(argv)

    if args.mode == "mkplan":
        mode_mkplan(args, None, None)
        return 0
    model, test = setup(args)
    fn = dict(grid=mode_grid, marginal=mode_marginal, greedy=mode_greedy,
              curve=mode_curve, plan=mode_plan)[args.mode]
    fn(args, model, test)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
