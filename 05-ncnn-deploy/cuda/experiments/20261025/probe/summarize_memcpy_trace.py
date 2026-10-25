#!/usr/bin/env python3
"""Summarize interposer trace: group by (kind, bytes) and by symbol."""
import subprocess
import collections
import sys

BIN = sys.argv[1] if len(sys.argv) > 1 else "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda/ncnn_cuda_eval_2000c_ipgemv"
TRACE = sys.argv[2] if len(sys.argv) > 2 else "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda/results-h039/memcpy_trace_128.txt"

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


dirs = {0: "H2H", 1: "H2D", 2: "D2H", 3: "D2D", 4: "DEFAULT"}
by_kind = collections.Counter()
by_kind_bytes = collections.Counter()
by_sym = collections.Counter()
by_sym_bytes = collections.Counter()
by_size = collections.Counter()
by_size_bytes = collections.Counter()
for line in open(TRACE):
    if line.startswith("#") or line.startswith("TOTAL"):
        continue
    kind, nbytes, count, total, off, mod = line.split(" ", 5)
    kind, nbytes, count, total, off = int(kind), int(nbytes), int(count), int(total), int(off)
    name = sym_at(off)
    d = dirs.get(kind, str(kind))
    by_kind[d] += count
    by_kind_bytes[d] += total
    by_sym[(d, name)] += count
    by_sym_bytes[(d, name)] += total
    by_size[(d, nbytes)] += count
    by_size_bytes[(d, nbytes)] += total

print("== by direction ==")
for d, n in by_kind.most_common():
    print("%-4s calls=%4d bytes=%d" % (d, n, by_kind_bytes[d]))
print("\n== by call site ==")
for (d, name), n in sorted(by_sym.items(), key=lambda kv: -kv[1]):
    print("%-4s calls=%4d bytes=%9d  %s" % (d, n, by_sym_bytes[(d, name)], name[:110]))
print("\n== by (direction, bytes) ==")
for (d, b), n in sorted(by_size.items(), key=lambda kv: -kv[1]):
    print("%-4s size=%7d calls=%4d bytes=%9d" % (d, b, n, by_size_bytes[(d, b)]))

print("\n== by (direction, size, call site) ==")
by_site_size = collections.Counter()
by_site_size_bytes = collections.Counter()
for line in open(TRACE):
    if line.startswith("#") or line.startswith("TOTAL"):
        continue
    kind, nbytes, count, total, off, mod = line.split(" ", 5)
    kind, nbytes, count, total, off = int(kind), int(nbytes), int(count), int(total), int(off)
    key = (dirs.get(kind, str(kind)), nbytes, sym_at(off))
    by_site_size[key] += count
    by_site_size_bytes[key] += total
for (d, b, name), n in sorted(by_site_size.items(), key=lambda kv: -kv[0][1]):
    print("%-4s size=%7d calls=%4d bytes=%9d  %s" % (d, b, n, by_site_size_bytes[(d, b, name)], name[:90]))
