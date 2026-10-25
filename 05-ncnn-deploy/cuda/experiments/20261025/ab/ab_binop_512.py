#!/usr/bin/env python3
"""A/B: 9561414 baseline vs fused maxpool padding, 512 images."""
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
ENV.pop("NCNN_CUDA_FUSE_ACT", None)
ENV["NCNN_CUDA_BINARYOP_VEC"] = "1"
RE_ACC = re.compile(r"^精度 (\d+\.\d+)% \((\d+)/(\d+)\)", re.M)
RE_GPU = re.compile(r"^最优 GPU kernel\s+([\d.]+)", re.M)


def run(binary):
    cmd = [binary, P, B, D, "--int8", "--limit", "512", "--batch", "512",
           "--repeat", "1", "--warmup", "20", "--reset-every", "1"]
    text = subprocess.run(cmd, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=600).stdout.decode("utf-8", "replace")
    acc = RE_ACC.search(text)
    gpu = RE_GPU.search(text)
    if not acc or not gpu:
        raise SystemExit("run failed:\n" + text[-600:])
    return int(acc.group(2)), float(gpu.group(1))


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    base = os.path.join(LAB, "ncnn_cuda_eval_2000c_pool")
    fused = os.path.join(LAB, "ncnn_cuda_eval_2000c_binop")
    res = {"base": [], "binop": []}
    for _ in range(2):
        run(base)
    for r in range(rounds):
        order = ("base", "binop") if r % 2 == 0 else ("binop", "base")
        for name in order:
            c, ms = run(base if name == "base" else fused)
            res[name].append(ms)
            print("%s round %d: correct=%d gpu=%.4f" % (name, r + 1, c, ms), flush=True)
    for name, vals in res.items():
        s = sorted(vals)
        print("%s: runs=%s median=%.4f min=%.4f" % (name, vals, s[len(s)//2], min(vals)))


if __name__ == "__main__":
    main()
