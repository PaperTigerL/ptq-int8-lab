# -*- coding: utf-8 -*-
"""Stage 0：纯 CPU 推理延迟/吞吐仪表（固定输入、线程、预热，median-of-3）。

基线定义（后续 1000x 优化以它为分母，口径必须固定）：
  · 模型：registry 里的 fp32 模型，BN 已折叠（部署路径不含 BN）
  · 输入：固定 seed 的 (N,3,32,32) float32，全 batch 复用同一张量
  · 线程：torch intra-op = --threads，inter-op = 1；OMP/MKL 同步设置
  · 预热：--warmup 次；计时：连续 3 轮，每轮 --iters 次
  · 指标：每轮**墙钟** median 延迟，三轮 median 的极差 ≤3% 才算稳定。
    延迟按墙钟定义；多线程下 process CPU time 是各线程求和（≈墙钟×利用率），
    只作辅助诊断，不参与稳定判定。
  · 三轮**交错采样**（round-robin）：宿主负载波动对三轮影响相同，
    median 极差才反映测量稳定性，而不是宿主噪声。`--sequential` 可关。

⛔ 只跑 CPU：导入 torch 前把 CUDA_VISIBLE_DEVICES 置空，并在运行时断言
   torch.cuda.is_available() 为 False。
"""
import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Stage 0 CPU inference benchmark")
    ap.add_argument("--model", default="resnet18",
                    choices=["resnet18", "resnet50", "mobilenet_v2"])
    ap.add_argument("--weights", default=None,
                    help="训练 checkpoint；resnet18 省略时用 03 的默认权重")
    ap.add_argument("--batch-sizes", default="1,8",
                    help="逗号分隔，例如 1,8,32")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30,
                    help="每轮计时迭代数")
    ap.add_argument("--runs", type=int, default=3,
                    help="测量轮数；极差按每轮 median 计算")
    ap.add_argument("--input-seed", type=int, default=1234)
    ap.add_argument("--out", default=None,
                    help="默认 results/stage0/stage0_<model>.json")
    ap.add_argument("--max-spread-pct", type=float, default=3.0)
    ap.add_argument("--sequential", action="store_true",
                    help="关闭三路交错，按传统 run1/run2/run3 顺序采样")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # 必须在 import torch 之前设置，确保任何库都不会初始化 CUDA。
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)

    import torch
    from ptq_toolkit.models import get_model
    from ptq_toolkit.pipeline import load_ptq_lab, load_trained_weights

    if torch.cuda.is_available():
        raise SystemExit("[FAIL] CUDA 可见；本单要求纯 CPU，拒绝运行")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.input_seed)

    lab = load_ptq_lab()
    model = get_model(args.model)
    n_sd = 0
    if args.weights:
        n_sd, miss, unexp = load_trained_weights(model, args.weights)
        if miss or unexp:
            raise SystemExit(
                f"[FAIL] 权重键集合不干净：missing={miss} unexpected={unexp}")
    model = lab["fold_all_bn"](model.to("cpu").eval()).to("cpu").eval()

    batch_sizes = [int(s) for s in args.batch_sizes.split(",") if s.strip()]
    if not batch_sizes or any(b <= 0 for b in batch_sizes):
        raise SystemExit("[FAIL] --batch-sizes 必须是正整数列表")

    out = Path(args.out) if args.out else Path(
        f"results/stage0/stage0_{args.model}.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    def build_doc(results):
        return {
            "schema": "ptq/stage0-bench@1",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "model": args.model,
            "weights": args.weights or "(registry default)",
            "weights_tensors": n_sd,
            "device": "cpu",
            "bn_folded": True,
            "threads": args.threads,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "torch_version": torch.__version__,
            "cpu": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count(),
            "input_seed": args.input_seed,
            "input_shape": [3, 32, 32],
            "warmup": args.warmup,
            "iters_per_run": args.iters,
            "sample_count": args.iters,
            "runs": args.runs,
            "max_spread_pct": args.max_spread_pct,
            "results": results,
            "all_stable": all(r["stable"] for r in results),
            "command": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
        }

    results = []
    with torch.inference_mode():
        for bs in batch_sizes:
            g = torch.Generator().manual_seed(args.input_seed)
            x = torch.randn(bs, 3, 32, 32, generator=g, dtype=torch.float32)

            for _ in range(args.warmup):
                model(x)

            wall = [[] for _ in range(args.runs)]
            cpu = [[] for _ in range(args.runs)]

            def once(r):
                w0, c0 = time.perf_counter_ns(), time.process_time_ns()
                model(x)
                w1, c1 = time.perf_counter_ns(), time.process_time_ns()
                wall[r].append((w1 - w0) / 1e6)
                cpu[r].append((c1 - c0) / 1e6)

            if args.sequential:
                for r in range(args.runs):
                    for _ in range(args.iters):
                        once(r)
            else:
                for i in range(args.iters):
                    for j in range(args.runs):
                        once((i + j) % args.runs)

            runs = []
            for r in range(args.runs):
                lat_wall, lat_cpu = wall[r], cpu[r]
                runs.append({
                    "run": r + 1,
                    "median_ms": round(statistics.median(lat_wall), 4),
                    "median_cpu_ms": round(statistics.median(lat_cpu), 4),
                    "mean_ms": round(statistics.fmean(lat_wall), 4),
                    "std_ms": round(statistics.pstdev(lat_wall), 4),
                    "mean_cpu_ms": round(statistics.fmean(lat_cpu), 4),
                    "std_cpu_ms": round(statistics.pstdev(lat_cpu), 4),
                    "min_ms": round(min(lat_wall), 4),
                    "max_ms": round(max(lat_wall), 4),
                    "iters_ms": [round(v, 4) for v in lat_wall],
                    "iters_cpu_ms": [round(v, 4) for v in lat_cpu],
                })

            medians = [r["median_ms"] for r in runs]
            cpu_medians = [r["median_cpu_ms"] for r in runs]
            med = statistics.median(medians)
            med_cpu = statistics.median(cpu_medians)
            spread = (max(medians) - min(medians)) / med * 100.0 if med else 0.0
            spread_cpu = (max(cpu_medians) - min(cpu_medians)) / med_cpu * 100.0 \
                if med_cpu else 0.0
            rec = {
                "batch_size": bs,
                "warmup": args.warmup,
                "iters_per_run": args.iters,
                "sample_count": args.iters,
                "runs": runs,
                "median_ms": round(med, 4),
                "median_cpu_ms": round(med_cpu, 4),
                "median_std_ms": round(statistics.median(
                    [r["std_ms"] for r in runs]), 4),
                "run_medians_ms": medians,
                "run_medians_cpu_ms": cpu_medians,
                "spread_pct": round(spread, 3),
                "spread_cpu_pct": round(spread_cpu, 3),
                "stable": bool(spread <= args.max_spread_pct),
                "throughput_img_s": round(bs * 1000.0 / med, 2) if med else None,
                "throughput_img_cpu_s": round(bs * 1000.0 / med_cpu, 2)
                if med_cpu else None,
            }
            results.append(rec)
            # 每个 batch 完成即落盘：即使后续 batch 超时，已完成的数据不丢。
            out.write_text(
                json.dumps(build_doc(results), indent=2, ensure_ascii=False),
                encoding="utf-8")
            print(f"  batch={bs:<3} median={med:8.3f} ms "
                  f"spread={spread:5.2f}%  "
                  f"(cpu_sum {med_cpu:8.3f} ms, spread_cpu {spread_cpu:5.2f}%)  "
                  f"throughput={rec['throughput_img_s']:.1f} img/s  "
                  f"{'PASS' if rec['stable'] else 'FAIL'}")

    doc = build_doc(results)
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"Stage 0 → {out}")
    return 0 if doc["all_stable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
