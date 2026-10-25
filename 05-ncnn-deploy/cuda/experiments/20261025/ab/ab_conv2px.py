#!/usr/bin/env python3
"""H-065: 8-round A/B, conv2px off vs on (same binary)."""
import json
import os
import random
import re
import statistics
import subprocess

LAB = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda"
BIN = os.path.join(LAB, "ncnn_cuda_eval_2000c_conv2px")
P = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/models/resnet18_cifar10_int8_old.param"
B = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/models/resnet18_cifar10_int8_old.bin"
D = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/data_raw/test_batch.raw"
OUT = os.path.join(LAB, "results-h065")
RE_ACC = re.compile(r"^精度 (\d+\.\d+)% \((\d+)/(\d+)\)", re.M)
RE_GPU = re.compile(r"^最优 GPU kernel\s+([\d.]+)", re.M)
RE_WALL = re.compile(r"^最优 含传输墙钟\s+([\d.]+)", re.M)


def env(on):
    e = dict(os.environ)
    e["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib:/home/user/cuda/lib64"
    for k in list(e):
        if k.startswith("NCNN_CUDA_") or k == "LD_PRELOAD":
            e.pop(k, None)
    if on:
        e["NCNN_CUDA_CONV2PX"] = "1"
    return e


def run(on):
    cmd = [BIN, P, B, D, "--int8", "--limit", "512", "--batch", "64",
           "--repeat", "1", "--warmup", "30"]
    text = subprocess.run(cmd, env=env(on), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=900).stdout.decode("utf-8", "replace")
    acc = RE_ACC.search(text); gpu = RE_GPU.search(text); wall = RE_WALL.search(text)
    if not acc or not gpu or not wall:
        raise SystemExit("run failed:\n" + text[-600:])
    return float(gpu.group(1)), float(wall.group(1)), int(acc.group(2)) / int(acc.group(3)) * 100.0


def boot(vals, iters=10000, seed=12345):
    rnd = random.Random(seed)
    meds = sorted(statistics.median([vals[rnd.randrange(len(vals))] for _ in range(len(vals))]) for _ in range(iters))
    return meds[int(0.025 * iters)], meds[int(0.975 * iters)]


def main():
    rounds = int(os.environ.get("ROUNDS", "8"))
    os.makedirs(OUT, exist_ok=True)
    og, ng, ow, nw = [], [], [], []
    for r in range(rounds):
        order = ("off", "on") if r % 2 == 0 else ("on", "off")
        v = {}
        for which in order:
            g, w, acc = run(which == "on")
            v[which] = (g, w)
            print("round %d %s: gpu %.4f wall %.4f acc %.2f%%" % (r + 1, which, g, w, acc), flush=True)
        og.append(v["off"][0]); ng.append(v["on"][0])
        ow.append(v["off"][1]); nw.append(v["on"][1])
        print("round %d gpu_improv=%+.2f%%" % (
            r + 1, (v["off"][0] - v["on"][0]) / v["off"][0] * 100), flush=True)
    gi = [(a - b) / a * 100 for a, b in zip(og, ng)]
    wi = [(a - b) / a * 100 for a, b in zip(ow, nw)]
    gmed = statistics.median(gi); gci = boot(gi)
    wmed = statistics.median(wi); wci = boot(wi)
    decision = "default_on" if (gci[0] > 0 and gmed >= 2.0) else "keep_opt_in"
    print("SUMMARY gpu off_med=%.4f on_med=%.4f med=%+.2f%% ci=[%.2f,%.2f]" % (
        statistics.median(og), statistics.median(ng), gmed, gci[0], gci[1]))
    print("SUMMARY wall off_med=%.4f on_med=%.4f med=%+.2f%% ci=[%.2f,%.2f] -> %s" % (
        statistics.median(ow), statistics.median(nw), wmed, wci[0], wci[1], decision))
    with open(os.path.join(OUT, "ab_conv2px.json"), "w", encoding="utf-8") as f:
        json.dump({"rounds": rounds, "off_gpu": og, "on_gpu": ng, "off_wall": ow, "on_wall": nw,
                   "gpu_impr": gi, "gpu_median_pct": gmed, "gpu_ci95": list(gci),
                   "wall_impr": wi, "wall_median_pct": wmed, "wall_ci95": list(wci),
                   "decision": decision}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
