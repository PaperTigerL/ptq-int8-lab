#!/usr/bin/env python3
"""8-round interleaved A/B: normal forward vs captured CUDA Graph replay."""
import json
import os
import random
import re
import statistics
import subprocess

LAB = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda"
BASE = os.path.join(LAB, "ncnn_cuda_eval_2000c_lazyw8")   # 8959d4c, default path
CAND = os.path.join(LAB, "ncnn_cuda_graph_eval")          # + NCNN_CUDA_GRAPH=1
SHIM = os.path.join(LAB, "graph_shim.so")
P = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/models/resnet18_cifar10_int8_old.param"
B = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/models/resnet18_cifar10_int8_old.bin"
D = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/data_raw/test_batch.raw"
OUT = os.path.join(LAB, "results-h048")
RE_ACC = re.compile(r"^精度 (\d+\.\d+)% \((\d+)/(\d+)\)", re.M)
RE_GPU = re.compile(r"^最优 GPU kernel\s+([\d.]+)", re.M)
RE_WALL = re.compile(r"^最优 含传输墙钟\s+([\d.]+)", re.M)


def env(cand):
    e = dict(os.environ)
    e["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib:/home/user/cuda/lib64"
    for k in ("NCNN_CUDA_IMPLICIT", "NCNN_CUDA_IP_GEMV", "NCNN_CUDA_BINARYOP_VEC",
              "NCNN_CUDA_POOL_FUSE_PAD", "NCNN_CUDA_LAZY_W8", "NCNN_CUDA_GRAPH",
              "NCNN_CUDA_GRAPH_DUMP", "LD_PRELOAD"):
        e.pop(k, None)
    if cand:
        e["NCNN_CUDA_GRAPH"] = "1"
        e["LD_PRELOAD"] = SHIM
    return e


def run(binary, cand):
    cmd = [binary, P, B, D, "--int8", "--limit", "512", "--batch", "64",
           "--repeat", "1", "--warmup", "30"]
    text = subprocess.run(cmd, env=env(cand), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=900).stdout.decode("utf-8", "replace")
    acc = RE_ACC.search(text)
    gpu = RE_GPU.search(text)
    wall = RE_WALL.search(text)
    if not acc or not gpu or not wall:
        raise SystemExit("run failed:\n" + text[-800:])
    return float(gpu.group(1)), float(wall.group(1)), int(acc.group(2)) / int(acc.group(3)) * 100.0


def boot(vals, iters=10000, seed=12345):
    rnd = random.Random(seed)
    meds = sorted(statistics.median([vals[rnd.randrange(len(vals))] for _ in range(len(vals))]) for _ in range(iters))
    return meds[int(0.025 * iters)], meds[int(0.975 * iters)]


def main():
    rounds = int(os.environ.get("ROUNDS", "8"))
    os.makedirs(OUT, exist_ok=True)
    bg, cg, bw, cw = [], [], [], []
    for r in range(rounds):
        order = ("base", "cand") if r % 2 == 0 else ("cand", "base")
        v = {}
        for which in order:
            g, w, acc = run(BASE if which == "base" else CAND, which == "cand")
            v[which] = (g, w)
            print("round %d %s: gpu %.4f wall %.4f (acc %.2f%%)" % (r + 1, which, g, w, acc), flush=True)
        bg.append(v["base"][0]); cg.append(v["cand"][0])
        bw.append(v["base"][1]); cw.append(v["cand"][1])
        print("round %d gpu_improv=%+.2f%% wall_improv=%+.2f%%" % (
            r + 1, (v["base"][0] - v["cand"][0]) / v["base"][0] * 100,
            (v["base"][1] - v["cand"][1]) / v["base"][1] * 100), flush=True)
    imprs = [(b - c) / b * 100 for b, c in zip(bg, cg)]
    wimprs = [(b - c) / b * 100 for b, c in zip(bw, cw)]
    med = statistics.median(imprs)
    lo, hi = boot(imprs)
    wmed = statistics.median(wimprs)
    wlo, whi = boot(wimprs)
    decision = "default_on" if (lo > 0 and med >= 2.0) else "keep_opt_in"
    print("SUMMARY gpu base_med=%.4f cand_med=%.4f med=%.2f%% ci=[%.2f,%.2f] " % (
        statistics.median(bg), statistics.median(cg), med, lo, hi))
    print("SUMMARY wall base_med=%.4f cand_med=%.4f med=%.2f%% ci=[%.2f,%.2f] -> %s" % (
        statistics.median(bw), statistics.median(cw), wmed, wlo, whi, decision))
    with open(os.path.join(OUT, "ab_graph.json"), "w", encoding="utf-8") as f:
        json.dump({"base": BASE, "cand": CAND, "rounds": rounds,
                   "base_gpu": bg, "cand_gpu": cg, "base_wall": bw, "cand_wall": cw,
                   "gpu_improvements": imprs, "gpu_median_pct": med, "gpu_ci95": [lo, hi],
                   "wall_improvements": wimprs, "wall_median_pct": wmed, "wall_ci95": [wlo, whi],
                   "decision": decision}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
