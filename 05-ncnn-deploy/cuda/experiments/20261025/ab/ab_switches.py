#!/usr/bin/env python3
"""Interleaved A/B for the three opt-in switches (process-level alternation).

Same binary (a82d8d7), per-run env selects baseline/candidate:
  pool  : NCNN_CUDA_POOL_FUSE_PAD 0/1
  binop : NCNN_CUDA_BINARYOP_VEC 0/1
  relu  : NCNN_CUDA_FUSE_ACT unset/1
8 rounds, alternating baseline/candidate (order flips each round); each run is
512 images / batch 64 / warmup 30.  Reports per-round values, medians, minima,
paired improvements and a bootstrap 95% CI of the median improvement.
"""
import json
import os
import random
import re
import statistics
import subprocess
import sys

LAB = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda"
BIN = os.path.join(LAB, "ncnn_cuda_eval_2000c_binop")
P = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/models/resnet18_cifar10_int8_old.param"
B = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/models/resnet18_cifar10_int8_old.bin"
D = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/data_raw/test_batch.raw"
OUT = os.path.join(LAB, "results-h033")
RE_ACC = re.compile(r"^精度 (\d+\.\d+)% \((\d+)/(\d+)\)", re.M)
RE_GPU = re.compile(r"^最优 GPU kernel\s+([\d.]+)", re.M)


def env_for(pool, binop, fuse):
    e = dict(os.environ)
    e["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib:/home/user/cuda/lib64"
    e.pop("NCNN_CUDA_IMPLICIT", None)
    if pool is None:
        e.pop("NCNN_CUDA_POOL_FUSE_PAD", None)
    else:
        e["NCNN_CUDA_POOL_FUSE_PAD"] = str(pool)
    if binop is None:
        e.pop("NCNN_CUDA_BINARYOP_VEC", None)
    else:
        e["NCNN_CUDA_BINARYOP_VEC"] = str(binop)
    if fuse is None:
        e.pop("NCNN_CUDA_FUSE_ACT", None)
    else:
        e["NCNN_CUDA_FUSE_ACT"] = str(fuse)
    return e


def run(env):
    cmd = [BIN, P, B, D, "--int8", "--limit", "512", "--batch", "64",
           "--repeat", "1", "--warmup", "30", "--reset-every", "1"]
    text = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=600).stdout.decode("utf-8", "replace")
    acc = RE_ACC.search(text)
    gpu = RE_GPU.search(text)
    if not acc or not gpu:
        raise SystemExit("run failed:\n" + text[-800:])
    return float(gpu.group(1)), float(acc.group(1))


def bootstrap_ci(vals, iters=10000, seed=12345):
    rnd = random.Random(seed)
    meds = []
    n = len(vals)
    for _ in range(iters):
        sample = [vals[rnd.randrange(n)] for _ in range(n)]
        meds.append(statistics.median(sample))
    meds.sort()
    return meds[int(0.025 * iters)], meds[int(0.975 * iters)]


SWITCHES = {
    "pool": {"base": env_for(0, 0, None), "cand": env_for(1, 0, None)},
    "binop": {"base": env_for(1, 0, None), "cand": env_for(1, 1, None)},
    "relu": {"base": env_for(1, 0, None), "cand": env_for(1, 0, 1)},
}


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    os.makedirs(OUT, exist_ok=True)
    report = {"binary": BIN, "rounds": rounds, "images_per_run": 512, "switches": {}}
    for name, cfgs in SWITCHES.items():
        base_vals, cand_vals, accs = [], [], []
        print("== %s ==" % name, flush=True)
        for r in range(rounds):
            order = ("base", "cand") if r % 2 == 0 else ("cand", "base")
            vals = {}
            for which in order:
                ms, acc = run(cfgs[which])
                vals[which] = ms
                accs.append((which, acc))
            base_vals.append(vals["base"])
            cand_vals.append(vals["cand"])
            impr = (vals["base"] - vals["cand"]) / vals["base"] * 100.0
            print("round %d: base=%.4f cand=%.4f impr=%+.2f%%" % (r + 1, vals["base"], vals["cand"], impr), flush=True)
        imprs = [(b - c) / b * 100.0 for b, c in zip(base_vals, cand_vals)]
        med_impr = statistics.median(imprs)
        ci_lo, ci_hi = bootstrap_ci(imprs)
        decision = "default_on" if (ci_lo > 0 and med_impr >= 2.0) else "opt_in_off"
        report["switches"][name] = {
            "base_runs": base_vals, "cand_runs": cand_vals,
            "base_median": statistics.median(base_vals), "base_min": min(base_vals),
            "cand_median": statistics.median(cand_vals), "cand_min": min(cand_vals),
            "improvements": imprs, "median_improvement_pct": med_impr,
            "ci95": [ci_lo, ci_hi], "decision": decision,
        }
        print("SUMMARY %s base_med=%.4f base_min=%.4f cand_med=%.4f cand_min=%.4f "
              "med_impr=%.2f%% ci=[%.2f,%.2f] -> %s"
              % (name, statistics.median(base_vals), min(base_vals),
                 statistics.median(cand_vals), min(cand_vals), med_impr, ci_lo, ci_hi, decision), flush=True)
    with open(os.path.join(OUT, "ab_switches.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("json:", os.path.join(OUT, "ab_switches.json"))


if __name__ == "__main__":
    main()
