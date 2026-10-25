#!/usr/bin/env python3
"""H-055: 8-round A/B, graph default-on vs explicit NCNN_CUDA_GRAPH=0."""
import json
import os
import random
import re
import statistics
import subprocess

LAB = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda"
BIN_OFF = os.path.join(LAB, "ncnn_cuda_eval_2000c_lazyw8")   # 8959d4c 普通路径
BIN_ON = os.path.join(LAB, "ncnn_cuda_graph_eval")            # graph 默认开
SHIM = os.path.join(LAB, "graph_shim.so")
P = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/models/resnet18_cifar10_int8_old.param"
B = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/models/resnet18_cifar10_int8_old.bin"
D = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/data_raw/test_batch.raw"
OUT = os.path.join(LAB, "results-h055")
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
        e["LD_PRELOAD"] = SHIM            # graph 默认开（env NCNN_CUDA_GRAPH 未设）
    # off = 8959d4c 普通二进制，环境保持干净
    return e


def run(cand):
    cmd = [BIN_ON if cand else BIN_OFF, P, B, D, "--int8", "--limit", "512", "--batch", "64",
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
        order = ("off", "on") if r % 2 == 0 else ("on", "off")
        v = {}
        for which in order:
            g, w, acc = run(which == "on")
            v[which] = (g, w)
            print("round %d %s: gpu %.4f wall %.4f (acc %.2f%%)" % (r + 1, which, g, w, acc), flush=True)
        bg.append(v["off"][0]); cg.append(v["on"][0])
        bw.append(v["off"][1]); cw.append(v["on"][1])
        print("round %d gpu_improv=%+.2f%% wall_improv=%+.2f%%" % (
            r + 1, (v["off"][0] - v["on"][0]) / v["off"][0] * 100,
            (v["off"][1] - v["on"][1]) / v["off"][1] * 100), flush=True)
    imprs = [(b - c) / b * 100 for b, c in zip(bg, cg)]
    wimprs = [(b - c) / b * 100 for b, c in zip(bw, cw)]
    med = statistics.median(imprs); lo, hi = boot(imprs)
    wmed = statistics.median(wimprs); wlo, whi = boot(wimprs)
    decision = "default_on" if (lo > 0 and med >= 2.0) else "revert"
    print("SUMMARY gpu off_med=%.4f on_med=%.4f med=%.2f%% ci=[%.2f,%.2f]" % (
        statistics.median(bg), statistics.median(cg), med, lo, hi))
    print("SUMMARY wall off_med=%.4f on_med=%.4f med=%.2f%% ci=[%.2f,%.2f] -> %s" % (
        statistics.median(bw), statistics.median(cw), wmed, wlo, whi, decision))
    with open(os.path.join(OUT, "ab_graph_default_on.json"), "w", encoding="utf-8") as f:
        json.dump({"binary_off": BIN_OFF, "binary_on": BIN_ON, "rounds": rounds,
                   "off_gpu": bg, "on_gpu": cg, "off_wall": bw, "on_wall": cw,
                   "gpu_improvements": imprs, "gpu_median_pct": med, "gpu_ci95": [lo, hi],
                   "wall_improvements": wimprs, "wall_median_pct": wmed, "wall_ci95": [wlo, whi],
                   "decision": decision}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
