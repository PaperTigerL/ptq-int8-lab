#!/usr/bin/env python3
"""H-071: aggregate cumulative-prefix graph profile by kernel symbol."""
import collections
import subprocess

BIN = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda/ncnn_cuda_graph_eval"
LOG = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda/results-h071/nodeprof_cum.log"

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


node_off = {}
cum = {}
for line in open(LOG):
    p = line.split()
    if not p:
        continue
    if p[0] == "NODE":
        node_off[int(p[1])] = int(p[2])
    elif p[0] == "CUM":
        cum[int(p[1])] = float(p[2])

n = max(node_off) + 1
total = collections.Counter()
cnt = collections.Counter()
marg = {}
for i in range(n):
    m = cum.get(i + 1, 0.0) - cum.get(i, 0.0)
    marg[i] = m
    name = sym_at(node_off[i]) if node_off[i] >= 0 else "?"
    short = name.split("(")[0].split(" ")[-1]
    total[short] += m
    cnt[short] += 1

kernel_sum = sum(marg.values())
print("nodes=%d  replay_full=%.4f ms  sum_marginal=%.4f ms" % (n, cum.get(n, 0), kernel_sum))
print("== by kernel (graph marginal) ==")
for name, ms in total.most_common():
    print("%-52s n=%3d %8.4f ms/img %5.1f%%" % (name, cnt[name], ms, 100.0 * ms / kernel_sum))
print("== top single nodes ==")
for i, m in sorted(marg.items(), key=lambda kv: -kv[1])[:6]:
    print("node %2d %8.4f ms  %s" % (i, m, sym_at(node_off[i]).split("(")[0]))
