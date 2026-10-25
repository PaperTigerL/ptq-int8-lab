#!/usr/bin/env python3
import subprocess

BIN = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda/ncnn_cuda_graph_eval"
LOG = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda/results-h074/nodeprof_default.log"
syms = []
for line in subprocess.run(["nm", "-C", "-n", "--defined-only", BIN], stdout=subprocess.PIPE, text=True).stdout.splitlines():
    p = line.split(" ", 2)
    if len(p) == 3:
        try:
            syms.append((int(p[0], 16), p[2].strip()))
        except ValueError:
            pass
syms.sort()


def sym_at(off):
    lo, hi = 0, len(syms)
    while lo < hi:
        mid = (lo + hi) // 2
        if syms[mid][0] <= off:
            lo = mid + 1
        else:
            hi = mid
    return syms[lo - 1][1] if lo else "?"


nodes = []
for line in open(LOG):
    p = line.split()
    if p and p[0] == "NODE":
        nodes.append((int(p[1]), int(p[2])))
for i, off in nodes:
    s = sym_at(off) if off >= 0 else "?"
    if "quantize" in s:
        for j in range(max(0, i - 2), min(len(nodes), i + 3)):
            print(("-> " if j == i else "   ") + "%2d %s" % (j, sym_at(nodes[j][1]).split("(")[0]))
