#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""激活 scale 溢出探针：定位 percentile / entropy 灌进 ncnn 后掉精度的原因。

待解释的现象（docs/阶段2 §3.5）：同一座桥，minmax 忠实、percentile 掉 3.12pp、
entropy 掉 17.14pp。zero_point 假设已被证伪，所以需要新假设：

    **激活 scale 注入值过大 → 整数量化溢出截断**。

理由链：ptq-lab 的激活是 asym，量程 qmax = 2^bits - 1 = 255，
scale = 量程/255（除数）。make_table 写入前取倒数，所以 ncnn 侧注入的是
255/量程（乘数）。percentile / entropy 会**截断**量程，于是：
量程变小 → scale 变小 → 注入的乘数变大 → 运行时真实激活越过阈值 →
q = round(x * 乘数) 冲出 255 被截断。

两组证据，本脚本一次跑完：

  A. 溢出倍数（纯读 json，不跑模型，秒出）
     倍数 = a_scale_minmax / a_scale_strategy = 真实量程 / 截断后量程。
     这是「各策略到底溢出了多少」的直接读数。

  B. 溢出容忍度（真跑 C++ 引擎）
     拿 minmax 表（唯一的忠实桥）把激活 scale 整体乘 f，看精度何时塌。
     如果掉精度的拐点和 A 算出的倍数对得上，机制就锁定了。

用法：
    python3 scripts/overflow_probe.py                    # A + B + C 全跑
    python3 scripts/overflow_probe.py --factors 1,1.5,2  # 自定义 B 档位
    python3 scripts/overflow_probe.py --skip-run         # 只算 A，不跑引擎
    python3 scripts/overflow_probe.py --no-causal        # 跳过 C（省 4 次评测）
    python3 scripts/overflow_probe.py --limit 2000       # 快速试跑
"""
import argparse
import json
import os
import re
import subprocess
import sys

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PTQ = os.path.join(os.path.dirname(LAB), "03-ptq-lab")
SCALES_DIR = os.path.join(PTQ, "results/cifar10_resnet18")

# asym 激活量程：q = round(x / scale) + zp ∈ [0, qmax]，见 src/quant.py 的
# quantize_per_tensor_asym（qmax = 2 ** bits - 1，不是 2 ** (bits-1) - 1）
QMAX = 255


def log(m):
    print(m, flush=True)


def hr():
    print("-" * 68, flush=True)


def run(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def load_scales(name):
    path = os.path.join(SCALES_DIR, "scales_w8a8_%s.json" % name)
    if not os.path.exists(path):
        sys.exit("[缺] %s 不存在" % path)
    with open(path) as f:
        return json.load(f), path


# ------------------------------------------------------------------ A
def part_a(strategies):
    """算各策略相对 minmax 的溢出倍数。"""
    base, _ = load_scales("minmax_mine")
    log("A. 溢出倍数（相对 minmax，asym 量程 qmax = %d）" % QMAX)
    log("")
    log("  %-14s %10s %10s %10s   %s"
        % ("策略", "最坏倍数", "max|q|", "溢出层数", "最坏层"))
    log("  " + "-" * 64)

    out = {"qmax": QMAX, "ref": "minmax_mine", "strategies": {}}
    for s in strategies:
        d, _ = load_scales("%s_mine" % s)
        ratios = {}
        for name, e in base["layers"].items():
            if name not in d["layers"]:
                continue
            a_ref = e["a_scale"]
            a_s = d["layers"][name]["a_scale"]
            if a_s <= 0:
                continue
            ratios[name] = a_ref / a_s          # >1 = 溢出
        if not ratios:
            log("  %-14s  （没有可比层，跳过）" % s)
            continue
        worst = max(ratios, key=lambda k: ratios[k])
        w = ratios[worst]
        n_ovf = sum(1 for v in ratios.values() if v > 1.0)
        log("  %-14s %9.2fx %10.1f %8d/%d   %s"
            % (s, w, QMAX * w, n_ovf, len(ratios), worst))
        out["strategies"][s] = {
            "worst_ratio": round(w, 3),
            "max_abs_q": round(QMAX * w, 1),
            "worst_layer": worst,
            "overflow_layers": n_ovf,
            "compared_layers": len(ratios),
            "ratios": {k: round(v, 3) for k, v in ratios.items()},
        }
    log("")
    log("  读法：minmax 自身是 1.00x（构造使然，q 恰好顶到 %d）；" % QMAX)
    log("        倍数 > 1 的层，运行时 q 会冲出量程被截断。")
    return out


# ------------------------------------------------------------------ 公共
def build_and_eval(args, scales_json, tag):
    """一份 scale json → 造表 → ncnn2int8 → C++ 评测。返回精度/延迟。"""
    tbl = os.path.join(args.work, "%s.table" % tag)
    oparam = os.path.join(args.work, "%s.param" % tag)
    obin = os.path.join(args.work, "%s.bin" % tag)

    rc, out = run(
        'python3 %s/scripts/make_table.py --scales "%s" --param "%s" '
        '--baseline "%s" --out "%s" %s'
        % (LAB, scales_json, args.param, args.baseline, tbl,
           "--no-sym-fix" if args.no_sym_fix else ""))
    if rc != 0:
        log("  [失败] 造表 %s：%s" % (tag, out[-500:]))
        return None

    rc, out = run('"%s" "%s" "%s" "%s" "%s" "%s"'
                  % (args.ncnn2int8, args.param, args.bin, oparam, obin, tbl))
    if rc != 0:
        log("  [失败] ncnn2int8 %s：%s" % (tag, out[-500:]))
        return None

    rc, out = run(
        '"%s" --param "%s" --bin "%s" --data "%s/data_raw/test_batch.raw" '
        '--threads %d --limit %d --tag "%s"'
        % (args.runner, oparam, obin, LAB, args.threads, args.limit, tag))
    if rc != 0:
        log("  [失败] 评测 %s：%s" % (tag, out[-500:]))
        return None

    acc = re.search(r"精度\s*:\s*([\d.]+)%", out)
    ms = re.search(r"([\d.]+)\s*ms/张", out)
    return {"acc": float(acc.group(1)) if acc else None,
            "ms_per_image": float(ms.group(1)) if ms else None}


def dump_json(d, path):
    with open(path, "w") as fp:
        json.dump(d, fp, indent=1)
    return path


# ------------------------------------------------------------------ B
def part_b(args):
    """把 minmax 表的激活 scale 整体乘 f，跑 ncnn 看精度怎么塌。"""
    src, src_path = load_scales("minmax_mine")
    os.makedirs(args.work, exist_ok=True)

    log("")
    log("B. 溢出容忍度（minmax 表，激活 scale × f，真跑 C++ 引擎）")
    log("")

    results = []
    for f in [float(x) for x in args.factors.split(",")]:
        tag = "f%g" % f
        # 注入到 ncnn 的乘数 = 1 / a_scale。要让它乘 f，就把 a_scale 除 f。
        d = json.loads(json.dumps(src))
        for e in d["layers"].values():
            e["a_scale"] = e["a_scale"] / f
        d["meta"]["note"] = ("overflow probe：激活 scale 相对 %s 放大 %.4gx"
                             % (src_path, f))
        sj = dump_json(d, os.path.join(args.work, "%s.json" % tag))

        r = build_and_eval(args, sj, tag)
        if r is None:
            continue
        r["factor"] = f
        results.append(r)
        log("  f=%-6s 精度 %s%%   %s ms/张"
            % (f, ("%.2f" % r["acc"]) if r["acc"] is not None else "-",
               ("%.3f" % r["ms_per_image"]) if r["ms_per_image"] else "-"))

    # 汇总
    if results:
        base = results[0]
        log("")
        log("  %8s %10s %12s" % ("f", "精度", "对基线"))
        log("  " + "-" * 32)
        for r in results:
            d = ""
            if base["acc"] is not None and r["acc"] is not None:
                d = "%+.2fpp" % (r["acc"] - base["acc"])
            log("  %8s %9s%% %12s"
                % (r["factor"],
                   ("%.2f" % r["acc"]) if r["acc"] is not None else "-", d))
    return results


# ------------------------------------------------------------------ C
def part_c(args, ratios):
    """因果回验：把策略自己的激活 scale 缩小，看精度是否回来。

    A+B 只证明相关。C 才是因果：若把 percentile / entropy 的注入 scale
    缩小（q 不再被 ±127 截断），精度就从 83.88% / 68.88% 回到 87% 附近，
    那么掉点的原因就是溢出截断，而不是「asym→sym 的固有损失」或别的什么。

    缩小倍数两种给法（注意是**注入 scale 缩小**，即 a_scale 放大）：
      --causal-shrink 0      → 各策略用自己的最坏溢出倍数（彻底消除溢出）
      --causal-shrink 2.0079 → 统一用 255/127（asym 255 码 -> ncnn ±127 码
                               的口径修正，让每层恰好截在策略自己选的阈值上）
    每组都带一个原样对照（同一策略、不作弊），所以 before/after 都是实测。
    """
    fixed = args.causal_shrink if args.causal_shrink > 0 else None
    log("")
    log("C. 因果回验：缩小注入的激活 scale（每组都带原样对照）%s"
        % ("，统一 ×%.4f" % fixed if fixed else "，各策略用自身最坏溢出倍数"))
    log("")

    results = []
    for s, info in ratios.items():
        w = fixed if fixed else info["worst_ratio"]
        d, path = load_scales("%s_mine" % s)

        # 对照：原样
        r0 = build_and_eval(args, path, "c_%s_base" % s)
        # 修正：a_scale × w  =>  注入乘数 ÷ w  =>  max|q| 回到 255
        d2 = json.loads(json.dumps(d))
        for e in d2["layers"].values():
            e["a_scale"] = e["a_scale"] * w
        d2["meta"]["note"] = ("overflow probe 因果回验：激活 scale ×%.4g 消除溢出"
                              % w)
        r1 = build_and_eval(args, dump_json(
            d2, os.path.join(args.work, "c_%s_fix.json" % s)), "c_%s_fix" % s)

        if r0 is None or r1 is None:
            continue
        results.append({"strategy": s, "worst_ratio": w,
                        "acc_before": r0["acc"], "acc_after": r1["acc"],
                        "acc_python": None})
        log("  %-12s 缩小 %.3fx：C++  %.2f%% -> %.2f%%  (%+.2fpp)"
            % (s, w, r0["acc"], r1["acc"], r1["acc"] - r0["acc"]))

    if results:
        log("")
        log("  %-12s %8s %10s %10s %12s"
            % ("策略", "缩小倍数", "原样", "缩小后", "恢复"))
        log("  " + "-" * 54)
        for r in results:
            log("  %-12s %7.2fx %9.2f%% %9.2f%% %11.2fpp"
                % (r["strategy"], r["worst_ratio"], r["acc_before"],
                   r["acc_after"], r["acc_after"] - r["acc_before"]))
        log("")
        log("  判据：若「消除溢出」把精度拉回到 minmax 桥的 87.17% 附近，")
        log("        机制即为溢出截断，可以结案。")
    return results


def main():
    ap = argparse.ArgumentParser(
        description="激活 scale 溢出探针（定位 percentile/entropy 掉精度的机制）")
    ap.add_argument("--strategies", default="percentile,entropy",
                    help="要和 minmax 比的策略（A 部分）")
    ap.add_argument("--factors", default="1,2,4,8",
                    help="激活 scale 放大倍数（B 部分）")
    ap.add_argument("--param", default=os.path.join(LAB, "models/resnet18_cifar10_opt.param"))
    ap.add_argument("--bin", default=os.path.join(LAB, "models/resnet18_cifar10_opt.bin"))
    ap.add_argument("--baseline", default=os.path.join(LAB, "tables/resnet18_cifar10_kl.table"))
    ap.add_argument("--limit", type=int, default=0, help="0 = 全量 10000")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--ncnn2int8", default=os.path.expanduser(
        "~/ncnn-build/ncnn/build/tools/quantize/ncnn2int8"))
    ap.add_argument("--runner", default=os.path.join(LAB, "cpp/build/cifar10_eval"))
    ap.add_argument("--work", default=os.path.join(LAB, "models/_ovf"))
    ap.add_argument("--out", default=os.path.join(LAB, "results/overflow_probe.json"))
    ap.add_argument("--skip-run", action="store_true", help="只算 A，不跑 C++ 引擎")
    ap.add_argument("--no-sym-fix", action="store_true",
                    help="传给 make_table.py：关掉 127/255 码空间换算，"
                         "复现 2026-10-18 修正前的基准")
    ap.add_argument("--no-causal", action="store_true", help="跳过 C（省 4 次评测）")
    ap.add_argument("--causal-shrink", type=float, default=0.0,
                    help="C 部分统一用的**注入 scale 缩小倍数**（= a_scale 放大倍数）；"
                         "0 = 各策略用自己的最坏溢出倍数。"
                         "传 255/127 = 2.0079 表示「asym 255 码 -> ncnn ±127 码」"
                         "的口径修正")
    args = ap.parse_args()

    log("激活 scale 溢出探针")
    log("  评测：%s 张，%d 线程" % (args.limit or 10000, args.threads))
    log("")
    hr()

    a = part_a([s for s in args.strategies.split(",") if s])
    b = []
    c = []
    if not args.skip_run:
        for f in (args.param, args.bin, args.baseline, args.ncnn2int8, args.runner):
            if not os.path.exists(f):
                sys.exit("[缺] %s 不存在" % f)
        b = part_b(args)
        c = [] if args.no_causal else part_c(args, a["strategies"])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"meta": {"qmax": QMAX, "limit": args.limit,
                            "threads": args.threads,
                            "factors": args.factors},
                   "overflow_ratio": a, "tolerance": b, "causal": c}, f, indent=1)
    log("")
    log("写出 %s" % args.out)


if __name__ == "__main__":
    main()
