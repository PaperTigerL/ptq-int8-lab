# -*- coding: utf-8 -*-
"""`ptq` 命令行入口：build / export / eval / ptqc / mixed-check / bench / report。

设计原则：
  · build 只做**量化与导出**（CPU 可跑，不碰 GPU）
  · bench **不重新测 GPU** —— 读 06/07 已有的 results 文件。
    M1 阶段 GPU 让给用户，且重测会引入「机器负载」这个变量，
    让 golden test 不可复现。真正的 GPU 基准留给后续（M3）。
  · report 把 build + bench 汇总成 Markdown 与 JSON
  · export/eval 是 W4 候选与显式 per-layer 混合精度（I2 top4_all）的
    导出/权重-only 评测入口；mixed-check 按磁盘字节回读自检混合精度包

用法：
    python -m ptq_toolkit build   --w-bits 8 --a-bits 8
    python -m ptq_toolkit export  --model resnet50 --weights ckpt.pth \
        --plan-json results/i2/layer_sensitivity_resnet50.json --plan-name top4_all \
        --out-dir results/mixed_precision/resnet50_top4_all
    python -m ptq_toolkit eval    --model resnet50 --weights ckpt.pth \
        --plan-json ... --plan-name top4_all --eval-n 2000
    python -m ptq_toolkit mixed-check results/mixed_precision/resnet50_top4_all
    python -m ptq_toolkit bench
    python -m ptq_toolkit report
    python -m ptq_toolkit ptqc --ranges ranges.json --out calib.ptqc \
        --weights results/resnet18_folded_w8a8.bin
"""
import argparse
import json
import sys
from pathlib import Path


def cmd_build(args) -> int:
    from . import pipeline
    from .models import REGISTRY

    targets = ([args.model] if args.model != "all"
               else sorted(REGISTRY))
    rc = 0
    for mi, mk in enumerate(targets):
        if mi:
            print("")
        try:
            r = pipeline.build(model_key=mk, w_bits=args.w_bits,
                               a_bits=args.a_bits, calib_n=args.calib_n,
                               eval_n=args.eval_n, weights=args.weights,
                               out_dir=args.out_dir, device=args.device,
                               verbose=not args.quiet)
        except Exception as e:
            print(f"[FAIL] {mk}: {type(e).__name__}: {str(e)[:160]}")
            rc = 1
            continue
        out = Path(args.out_dir) / f"build_{mk}_w{args.w_bits}.json"
        out.write_text(json.dumps(r.to_dict(), indent=2, ensure_ascii=False),
                       encoding="utf-8")
        print(f"构建摘要 → {out}")
    return rc


def cmd_bench(args) -> int:
    """bench：汇总 06/07 的**已有**基准结果，不重新测 GPU。

    输出放到 results/bench.json，供 report 消费。
    """
    from .bench import collect
    rows = collect()
    out = Path(args.out_dir) / "bench.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"基准（读自 06/07 已有结果）→ {out}")
    print(f"{'路线':<18}{'ms/img':>10}{'加速':>10}{'精度':>10}")
    print("-" * 48)
    for r in rows["rows"]:
        acc = f"{r['accuracy_pct']:.2f}%" if r.get("accuracy_pct") else "-"
        from .report.render import fmt_speedup
        print(f"{r['route']:<18}{r['ms_per_img']:>10.4f}"
              f"{fmt_speedup(r['speedup']):>10}{acc:>10}")
    return 0


def cmd_report(args) -> int:
    from .report import write_reports
    md, js = write_reports(out_dir=args.out_dir)
    print(f"Markdown → {md}")
    print(f"JSON     → {js}")
    return 0


def _resolve_plan_args(args):
    """把 --plan-json/--plan-name/--w8-layers 解析成 (w8_layers, meta)。"""
    from .quant.mixed import w8_layers_from_json

    if args.plan_json:
        if args.w8_layers:
            raise ValueError("--plan-json 与 --w8-layers 只能二选一")
        return w8_layers_from_json(args.plan_json, args.plan_name)
    if args.w8_layers:
        layers = [s.strip() for s in args.w8_layers.split(",") if s.strip()]
        if not layers:
            raise ValueError("--w8-layers 为空")
        return layers, dict(plan=args.plan_name or "custom",
                            source="--w8-layers (CLI)")
    return None, None


def _resolve_w4(args, plan_meta):
    """解析 W4 的 (scheme, group_size)：命令行显式值 > 计划 JSON 声明 > 默认。

    ⭐ 为什么不让 --w4-group 默认 32、scheme 默认 per_group 就直接用：
       计划 JSON（I2/I3）里带着这份计划**实测时用的**粒度。
       若命令行忘了传、或传了别的值，会静默导出一个「与证据不符」的包，
       而包自检照样全过（自检只验包内部一致，不验与证据一致）。
       所以两个默认值都改成 None，按「显式 > 证据 > per_group/32」解析；
       per-channel 时 group_size 必须归 0（否则 layer_plan 与 quant_config
       的 group_size 不一致，mixed-check 会报「逐层配置不符」）。
    """
    scheme = args.w4_scheme or (plan_meta or {}).get("w4_scheme") or "per_group"
    if scheme not in ("per_channel", "per_group"):
        raise ValueError(f"w4 scheme 只支持 per_channel/per_group，收到 {scheme!r}")
    if scheme == "per_channel":
        return scheme, 0
    if args.w4_group is not None:
        return scheme, int(args.w4_group)
    from_plan = (plan_meta or {}).get("w4_group_size")
    if from_plan:
        return scheme, int(from_plan)
    return scheme, 32


def cmd_export(args) -> int:
    """导出 W4/W8 候选或显式混合精度计划的 PTQ2 + PTQC（只校准，不评测）。"""
    from . import pipeline

    try:
        w8_layers, plan_meta = _resolve_plan_args(args)
        w4_scheme, w4_group = _resolve_w4(args, plan_meta)
        results = pipeline.export_candidates(
            model_key=args.model,
            weights=args.weights,
            candidate_keys=None if w8_layers is not None else args.candidates,
            calib_n=args.calib_n,
            a_bits=args.a_bits,
            out_dir=args.out_dir,
            device=args.device,
            verbose=not args.quiet,
            w8_layers=w8_layers,
            w4_group_size=w4_group,
            w4_scheme=w4_scheme,
            plan_name=(plan_meta or {}).get("plan"),
            plan_meta=plan_meta,
        )
    except Exception as e:
        print(f"[FAIL] {args.model}: {type(e).__name__}: {e}")
        return 1
    for r in results:
        print(f"{r.candidate:<18} PTQ2 {r.n_conv} conv / "
              f"{r.exported['bytes'] / 1e6:.1f} MB · "
              f"PTQC {r.ptqc['bytes']} B · "
              f"W8 {len(r.layer_plan.get('w8_layers') or r.plan['w8_layers'])}"
              f"/{r.plan['n_layers']} 层")
        if r.selfcheck:
            nbad = sum(1 for c in r.selfcheck["checks"] if not c["ok"])
            print(f"  包自检：{'PASS' if r.selfcheck['ok'] else 'FAIL'}"
                  f"（{len(r.selfcheck['checks'])} 项，失败 {nbad}）"
                  f" · manifest {r.manifest.get('path')}")
    return 0


def cmd_eval(args) -> int:
    """按 M2 协议做权重-only CPU 精度评测（不跑激活量化/GPU）。"""
    from . import pipeline

    try:
        w8_layers, plan_meta = _resolve_plan_args(args)
        w4_scheme, w4_group = _resolve_w4(args, plan_meta)
        doc = pipeline.eval_candidates(
            model_key=args.model,
            weights=args.weights,
            candidate_keys=None if w8_layers is not None else args.candidates,
            eval_n=args.eval_n,
            budget_pp=args.budget_pp,
            out_dir=args.out_dir,
            device=args.device,
            verbose=not args.quiet,
            w8_layers=w8_layers,
            w4_group_size=w4_group,
            w4_scheme=w4_scheme,
            plan_name=(plan_meta or {}).get("plan"),
            plan_meta=plan_meta,
        )
    except Exception as e:
        print(f"[FAIL] {args.model}: {type(e).__name__}: {e}")
        return 1

    print(f"{'候选':<18}{'fp32':>8}{'量化':>8}{'掉点':>8}  判定")
    print("-" * 52)
    n_pass = 0
    for r in doc["candidates"]:
        verdict = "PASS" if r["passed"] else "FAIL"
        n_pass += int(r["passed"])
        print(f"{r['candidate']:<18}{r['fp32_acc']:>8.2f}"
              f"{r['quant_acc']:>8.2f}{r['acc_drop']:>8.2f}  {verdict}")
    print(f"\n{args.model}：{n_pass}/{len(doc['candidates'])} 个候选 <= "
          f"{doc['budget_pp']:.1f} pp · 结果 {doc['path']}")
    return 0


def cmd_ptqc(args) -> int:
    """把按引擎 conv 顺序排好的 (lo,hi) 列表写成 PTQC。

    输入 JSON 是 `[[lo, hi], ...]`（也接受 {"ranges": [...]}）。
    顺序必须等于引擎 conv 顺序：`named_modules()` 过滤 `nn.Conv2d` 的
    定义顺序，与 exporter.py / topoblocks.py 的导出顺序一致。
    `--weights` 给 PTQW/PTQ2 产物时，会读文件头 n_conv 并强制校验层数。
    """
    from .export.ptqc import weight_conv_count, write_ptqc

    try:
        doc = json.loads(Path(args.ranges).read_text(encoding="utf-8"))
        if isinstance(doc, dict):
            doc = doc.get("ranges")
        if not isinstance(doc, list):
            raise ValueError("ranges JSON 应为 [[lo,hi], ...] 或 {\"ranges\": [...]}")
        info = write_ptqc(args.out, doc)
    except Exception as e:
        print(f"[FAIL] {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print(f"PTQC → {info['path']}（n={info['n']} 层，{info['bytes']} B）")
    print("层顺序：ranges 数组顺序 = 引擎 conv 顺序"
          "（named_modules 过滤 Conv2d 的定义顺序）")
    if args.weights:
        try:
            magic, n_conv = weight_conv_count(args.weights)
        except Exception as e:
            print(f"[FAIL] 读权重头失败：{type(e).__name__}: {e}", file=sys.stderr)
            return 1
        print(f"权重 {args.weights}：magic={magic} n_conv={n_conv}")
        if n_conv != info["n"]:
            print(f"[FAIL] 层数不一致：PTQC n={info['n']} != 权重 n_conv={n_conv}；"
                  f"引擎 set_calib 会拒绝", file=sys.stderr)
            return 1
        print("层数一致")
    return 0


def cmd_mixed_check(args) -> int:
    """回读一个混合精度包（manifest + PTQ2 + PTQC + quant_config）并自检。"""
    from .export.mixed_check import check_package

    try:
        rep = check_package(args.package, check_grid=not args.no_grid)
    except Exception as e:
        print(f"[FAIL] {args.package}: {type(e).__name__}: {e}",
              file=sys.stderr)
        return 1
    for c in rep.get("checks", []):
        print(f"  {'✓' if c['ok'] else '✗'} {c['name']}"
              f"{'  ' + c['detail'] if c.get('detail') else ''}")
    if rep.get("n_conv"):
        print(f"\n{rep['n_conv']} conv · W8 {rep['n_w8']} / W4 {rep['n_w4']} · "
              f"{rep['package']}")
    print("✓ 全部通过" if rep.get("ok") else
          f"✗ {sum(1 for c in rep.get('checks', []) if not c['ok'])} 项失败")
    return 0 if rep.get("ok") else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="ptq",
        description="PTQ 工具链：PyTorch 模型 + 校准数据 → 量化 → 导出 → 报告")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="折叠 + 校准 + 量化 + 导出（不碰 GPU）")
    b.add_argument("--model", default="resnet18",
                   choices=["resnet18", "resnet50", "mobilenet_v2", "all"],
                   help="模型名；all = 依次构建全部（registry 里已注册的）")
    b.add_argument("--w-bits", type=int, default=8, choices=[4, 8])
    b.add_argument("--a-bits", type=int, default=8, choices=[4, 8])
    b.add_argument("--calib-n", type=int, default=512, help="校准样本数")
    b.add_argument("--weights", default=None,
                   help="加载训练好的 checkpoint（tools/finetune.py 的产物）。"
                        "在 BN 折叠之前加载；键集合不干净会直接报错")
    b.add_argument("--eval-n", type=int, default=0,
                   help="评测样本数上限（0=全量；随机初始化模型默认取 2000，"
                        "因为全量在 CPU 上要 20 小时且精度无区别）")
    b.add_argument("--out-dir", default="results")
    b.add_argument("--device", default="cpu")
    b.add_argument("--quiet", action="store_true")
    b.set_defaults(func=cmd_build)

    n = sub.add_parser("bench", help="汇总 06/07 已有基准（不重新测 GPU）")
    n.add_argument("--out-dir", default="results")
    n.set_defaults(func=cmd_bench)

    r = sub.add_parser("report", help="生成 Markdown / JSON 报告")
    r.add_argument("--out-dir", default="results")
    r.set_defaults(func=cmd_report)

    e = sub.add_parser(
        "export", help="导出 W4/W8 候选的 PTQ2+PTQC（校准后直接导出，不评测）")
    e.add_argument("--model", required=True,
                   choices=["resnet50", "mobilenet_v2"],
                   help="只支持走 PTQ2 的两个模型")
    e.add_argument("--weights", default=None,
                   help="训练好的 checkpoint；键集合不干净会直接报错")
    e.add_argument("--candidates", default="w8a8,w4a8_ch,w4a8_g32,w4a8_mixed_g32",
                   help="逗号分隔的候选名，见 quant/mixed.py::CANDIDATES")
    e.add_argument("--plan-json", default=None,
                   help="显式 per-layer 计划的 JSON（I2 敏感度文件或 "
                        "{\"w8_layers\": [...]}）；与 --candidates 二选一")
    e.add_argument("--plan-name", default=None,
                   help="从 --plan-json 的 plans 里选（默认 recommended.plan）")
    e.add_argument("--w8-layers", default=None,
                   help="逗号分隔的显式 W8 层名；与 --plan-json 二选一")
    e.add_argument("--w4-scheme", default=None,
                   choices=["per_channel", "per_group"],
                   help="未列入 W8 的层用哪种 W4 粒度；"
                        "默认沿用 --plan-json 里声明值，再退回 per_group")
    e.add_argument("--w4-group", type=int, default=None,
                   help="未列入 W8 的层用 W4 per-group 的组大小；"
                        "默认沿用 --plan-json 里声明值，再退回 32")
    e.add_argument("--calib-n", type=int, default=512)
    e.add_argument("--a-bits", type=int, default=8, choices=[4, 8])
    e.add_argument("--out-dir", default="results/w4_candidates")
    e.add_argument("--device", default="cpu")
    e.add_argument("--quiet", action="store_true")
    e.set_defaults(func=cmd_export)

    v = sub.add_parser(
        "eval", help="权重-only CPU 精度评测（M2 协议；不含激活量化）")
    v.add_argument("--model", required=True,
                   choices=["resnet50", "mobilenet_v2"],
                   help="只支持走 PTQ2 的两个模型")
    v.add_argument("--weights", required=True,
                   help="训练好的 checkpoint；不做训练权重评测没有意义")
    v.add_argument("--candidates", default="w8a8,w4a8_ch,w4a8_g32,w4a8_mixed_g32",
                   help="逗号分隔的候选名，见 quant/mixed.py::CANDIDATES")
    v.add_argument("--plan-json", default=None,
                   help="显式 per-layer 计划的 JSON（I2 敏感度文件或 "
                        "{\"w8_layers\": [...]}）；与 --candidates 二选一")
    v.add_argument("--plan-name", default=None,
                   help="从 --plan-json 的 plans 里选（默认 recommended.plan）")
    v.add_argument("--w8-layers", default=None,
                   help="逗号分隔的显式 W8 层名；与 --plan-json 二选一")
    v.add_argument("--w4-scheme", default=None,
                   choices=["per_channel", "per_group"],
                   help="未列入 W8 的层用哪种 W4 粒度；"
                        "默认沿用 --plan-json 里声明值，再退回 per_group")
    v.add_argument("--w4-group", type=int, default=None,
                   help="未列入 W8 的层用 W4 per-group 的组大小；"
                        "默认沿用 --plan-json 里声明值，再退回 32")
    v.add_argument("--eval-n", type=int, default=2000,
                   help="测试集前 N 张；0=全量 10000")
    v.add_argument("--budget-pp", type=float, default=3.0,
                   help="掉点预算（百分点），默认 3.0")
    v.add_argument("--out-dir", default="results/w4_candidates")
    v.add_argument("--device", default="cpu")
    v.add_argument("--quiet", action="store_true")
    v.set_defaults(func=cmd_eval)

    q = sub.add_parser(
        "ptqc", help="按 conv 顺序的 (lo,hi) 列表 → 06 引擎可读的 PTQC")
    q.add_argument("--ranges", required=True,
                   help="JSON：[[lo,hi], ...]，顺序=引擎 conv 顺序")
    q.add_argument("--out", required=True, help="输出 .ptqc 路径")
    q.add_argument("--weights", default=None,
                   help="可选 PTQW/PTQ2 产物；读头 n_conv 校验层数一致")
    q.set_defaults(func=cmd_ptqc)

    m = sub.add_parser(
        "mixed-check", help="回读混合精度包（manifest/PTQ2/PTQC/配置）并自检")
    m.add_argument("package", help="包目录（内含 manifest.json）")
    m.add_argument("--no-grid", action="store_true",
                   help="跳过逐层权重网格校验（只看结构与哈希）")
    m.set_defaults(func=cmd_mixed_check)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
