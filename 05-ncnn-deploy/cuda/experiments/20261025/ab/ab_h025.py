#!/usr/bin/env python3
"""Steady-state alternating A/B: e053c57 vs fused pad+pack (T-20261024-05)."""
import os
import re
import subprocess
import sys

LAB = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda"
P = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/models/resnet18_cifar10_int8_old.param"
B = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/models/resnet18_cifar10_int8_old.bin"
D = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/data_raw/test_batch.raw"
ENV = dict(os.environ)
ENV["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib:/home/user/cuda/lib64"
ENV.pop("NCNN_CUDA_IMPLICIT", None)
RE_ACC = re.compile(r"^精度 (\d+\.\d+)% \((\d+)/(\d+)\)", re.M)
RE_GPU = re.compile(r"^最优 GPU kernel\s+([\d.]+)", re.M)


def run(binary, repeat=1, warmup=20):
    cmd = [binary, P, B, D, "--int8", "--limit", "2000", "--batch", "64",
           "--repeat", str(repeat), "--warmup", str(warmup), "--reset-every", "1"]
    text = subprocess.run(cmd, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=1800).stdout.decode("utf-8", "replace")
    acc = RE_ACC.search(text)
    gpu = RE_GPU.search(text)
    if not acc or not gpu:
        raise SystemExit("run failed, tail:\n" + text[-800:])
    return int(acc.group(2)), float(gpu.group(1))


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    bins = {"e053c57": os.path.join(LAB, "ncnn_cuda_eval_2000c_e053c57"),
            "fused": os.path.join(LAB, "ncnn_cuda_eval_2000c")}
    print("burn-in ...", flush=True)
    for _ in range(2):
        run(bins["e053c57"])
    res = {k: [] for k in bins}
    for r in range(rounds):
        order = ("e053c57", "fused") if r % 2 == 0 else ("fused", "e053c57")
        for name in order:
            c, ms = run(bins[name])
            res[name].append(ms)
            print("%s round %d: correct=%d gpu=%.4f" % (name, r + 1, c, ms), flush=True)
    for name, vals in res.items():
        s = sorted(vals)
        med = s[len(s) // 2]
        print("%s: runs=%s median=%.4f min=%.4f" % (name, vals, med, min(vals)))


if __name__ == "__main__":
    main()
