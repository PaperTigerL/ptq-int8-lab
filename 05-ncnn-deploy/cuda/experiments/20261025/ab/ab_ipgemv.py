#!/usr/bin/env python3
"""8-round interleaved A/B: a82d8d7 (fusion off) vs default-on fusion binary."""
import json
import os
import random
import re
import statistics
import subprocess
import sys

LAB = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda"
BASE = os.path.join(LAB, "ncnn_cuda_eval_2000c_binop")      # a82d8d7, fusion off
CAND = os.path.join(LAB, "ncnn_cuda_eval_2000c_ipgemv")    # default on
P = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/models/resnet18_cifar10_int8_old.param"
B = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/models/resnet18_cifar10_int8_old.bin"
D = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/data_raw/test_batch.raw"
OUT = os.path.join(LAB, "results-h035")
RE_ACC = re.compile(r"^精度 (\d+\.\d+)% \((\d+)/(\d+)\)", re.M)
RE_GPU = re.compile(r"^最优 GPU kernel\s+([\d.]+)", re.M)


def env():
    e = dict(os.environ)
    e["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib:/home/user/cuda/lib64"
    for k in ("NCNN_CUDA_IMPLICIT", "NCNN_CUDA_IP_GEMV", "NCNN_CUDA_BINARYOP_VEC", "NCNN_CUDA_POOL_FUSE_PAD"):
        e.pop(k, None)
    return e


def run(binary):
    cmd = [binary, P, B, D, "--int8", "--limit", "512", "--batch", "64",
           "--repeat", "1", "--warmup", "30", "--reset-every", "1"]
    text = subprocess.run(cmd, env=env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=600).stdout.decode("utf-8", "replace")
    acc = RE_ACC.search(text)
    gpu = RE_GPU.search(text)
    if not acc or not gpu:
        raise SystemExit("run failed:\n" + text[-600:])
    return float(gpu.group(1)), float(acc.group(1))


def boot(vals, iters=10000, seed=12345):
    rnd = random.Random(seed)
    meds = sorted(statistics.median([vals[rnd.randrange(len(vals))] for _ in range(len(vals))]) for _ in range(iters))
    return meds[int(0.025 * iters)], meds[int(0.975 * iters)]


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    os.makedirs(OUT, exist_ok=True)
    bvals, cvals = [], []
    for r in range(rounds):
        order = ("base", "cand") if r % 2 == 0 else ("cand", "base")
        v = {}
        for which in order:
            ms, acc = run(BASE if which == "base" else CAND)
            v[which] = ms
            print("round %d %s: %.4f (acc %.2f%%)" % (r + 1, which, ms, acc), flush=True)
        bvals.append(v["base"]); cvals.append(v["cand"])
        print("round %d improv=%+.2f%%" % (r + 1, (v["base"] - v["cand"]) / v["base"] * 100), flush=True)
    imprs = [(b - c) / b * 100 for b, c in zip(bvals, cvals)]
    med = statistics.median(imprs)
    lo, hi = boot(imprs)
    decision = "default_on_confirmed" if (lo > 0 and med >= 2.0) else "revert"
    print("SUMMARY base_med=%.4f base_min=%.4f cand_med=%.4f cand_min=%.4f med_impr=%.2f%% ci=[%.2f,%.2f] -> %s" % (
        statistics.median(bvals), min(bvals), statistics.median(cvals), min(cvals), med, lo, hi, decision))
    with open(os.path.join(OUT, "ab_ipgemv.json"), "w", encoding="utf-8") as f:
        json.dump({"base": BASE, "cand": CAND, "rounds": rounds,
                   "base_runs": bvals, "cand_runs": cvals, "improvements": imprs,
                   "median_improvement_pct": med, "ci95": [lo, hi], "decision": decision},
                  f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
