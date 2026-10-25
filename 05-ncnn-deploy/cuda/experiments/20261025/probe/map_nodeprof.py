#!/usr/bin/env python3
"""H-071: map graph node offsets to symbols and rank kernel-only time."""
import collections
import subprocess
import sys

BIN = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda/ncnn_cuda_graph_eval"
LOG = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda/results-h071/nodeprof.log"

syms = []
out = subprocess.run(["nm", "-C", "-n", "--defined-only", BIN],
                     stdout=subprocess.PIPE, text=True).stdout
for line in out.splitlines():
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

total_ms = collections.Counter()
count = collections.Counter()
rows = []
for line in open(LOG):
    if not line.startswith("NODEPROF"):
        continue
    _, idx, off, ms = line.split()
    off = int(off)
    name = sym_at(off) if off >= 0 else "unknown"
    # 短名：去掉参数列表
    short = name.split("(")[0].split(" ")[-1]
    total_ms[short] += float(ms)
    count[short] += 1
    rows.append((int(idx), off, float(ms), short))

grand = sum(total_ms.values())
print("nodes=%d  kernel_total=%.4f ms/img" % (len(rows), grand))
print("== by kernel ==")
for name, ms in total_ms.most_common():
    print("%-55s n=%3d %8.4f ms/img %5.1f%%" % (name, count[name], ms, 100.0 * ms / grand))
