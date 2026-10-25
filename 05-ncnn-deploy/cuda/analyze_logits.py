#!/usr/bin/env python3
"""Compute top-1/top-5 and logits diffs from dump files."""
import sys

RAW = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/data_raw/test_batch.raw"


def load(path):
    rows = []
    for line in open(path):
        p = line.split()
        if len(p) < 12:
            continue
        rows.append((int(p[0]), float.fromhex(p[2]) if False else [float.fromhex(x) for x in p[2:12]]))
    return rows


def labels(n):
    with open(RAW, "rb") as f:
        data = f.read()
    return [data[i * 3073] for i in range(n)]


def topk(rows, labs, k):
    ok = 0
    for idx, vals in rows:
        order = sorted(range(10), key=lambda j: vals[j], reverse=True)[:k]
        if labs[idx] in order:
            ok += 1
    return ok, len(rows)


def main():
    files = sys.argv[1:]
    loaded = {}
    for p in files:
        rows = load(p)
        loaded[p] = rows
        labs = labels(len(rows))
        t1 = topk(rows, labs, 1)
        t5 = topk(rows, labs, 5)
        print("%s n=%d top1=%d/%d (%.2f%%) top5=%d/%d (%.2f%%)" % (
            p, len(rows), t1[0], t1[1], 100.0 * t1[0] / max(t1[1], 1),
            t5[0], t5[1], 100.0 * t5[0] / max(t5[1], 1)))
    if len(files) == 2:
        a, b = loaded[files[0]], loaded[files[1]]
        n = min(len(a), len(b))
        maxdiff = 0.0
        neq = 0
        for i in range(n):
            if a[i][0] != b[i][0]:
                print("index mismatch at", i)
                break
            for x, y in zip(a[i][1], b[i][1]):
                d = abs(x - y)
                if d > maxdiff:
                    maxdiff = d
                if x != y:
                    neq += 1
        print("diff n=%d max_abs=%.9g differing_floats=%d" % (n, maxdiff, neq))


if __name__ == "__main__":
    main()
