#!/usr/bin/env python3
"""Map interposer caller offsets (module-relative) to nearest symbol via nm."""
import subprocess
import sys

BIN = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda/ncnn_cuda_eval_2000c_ipgemv"
TRACE = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda/results-h039/memcpy_trace_128.txt"

syms = []
out = subprocess.run(["nm", "-C", "-n", "--defined-only", BIN],
                     stdout=subprocess.PIPE, text=True).stdout
for line in out.splitlines():
    parts = line.split(" ", 2)
    if len(parts) == 3:
        try:
            addr = int(parts[0], 16)
        except ValueError:
            continue
        syms.append((addr, parts[1], parts[2]))
syms.sort()

rows = []
for line in open(TRACE):
    if line.startswith("#") or line.startswith("TOTAL"):
        continue
    kind, nbytes, count, total, off, mod = line.split(" ", 5)
    off = int(off)
    if off < 0:
        name = "(unknown module)"
    else:
        lo, hi = 0, len(syms)
        while lo < hi:
            mid = (lo + hi) // 2
            if syms[mid][0] <= off:
                lo = mid + 1
            else:
                hi = mid
        name = syms[lo - 1][2] if lo else "?"
    rows.append((int(count), int(nbytes), int(kind), int(total), off, name.strip()))

rows.sort(key=lambda r: (-r[0], -r[1]))
directions = {0: "H2H", 1: "H2D", 2: "D2H", 3: "D2D", 4: "DEFAULT"}
for count, nbytes, kind, total, off, name in rows:
    print("%-6s %7d B  x%-4d total=%9d  off=0x%x  %s" % (
        directions.get(kind, str(kind)), nbytes, count, total, off, name))
