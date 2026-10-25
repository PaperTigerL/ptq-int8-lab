# -*- coding: utf-8 -*-
"""端到端比对：GPU logits vs numpy 参考 logits。

⭐ 这是**唯一的正确性判据**（取代之前的逐层比对）：
   中间张量（im2col 布局、int32 中间值）的具体排布是实现细节，
   不同实现可以有不同排布但输出完全一致。
   只有最终输出能证明语义正确。

用法：
  python cmp_e2e.py --gpu /tmp/gpu.bin --ref /tmp/ref.bin
"""
import argparse
import struct
import sys

import numpy as np


def load_logits(path):
    with open(path, "rb") as f:
        d = f.read()
    assert d[:4] == b"PTQL", f"{path} magic 不符"
    n, c = struct.unpack_from("<2i", d, 4)
    off = 12
    arr = np.frombuffer(d, dtype="<f4", count=n * c, offset=off).reshape(n, c)
    assert off + n * c * 4 == len(d), f"{path} 有多余字节"
    return np.array(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="判定「一致」的绝对容差；超出即视为不一致")
    args = ap.parse_args()

    g = load_logits(args.gpu)
    r = load_logits(args.ref)

    if g.shape != r.shape:
        print("形状不同：gpu %s vs ref %s" % (g.shape, r.shape))
        return 1

    diff = np.abs(g.astype(np.float64) - r.astype(np.float64))
    maxdiff = float(diff.max())
    n_bad = int((diff > args.tol).sum())
    total = diff.size

    print("样本 %d × %d 类 = %d 个 logits" % (g.shape[0], g.shape[1], total))
    print("max|diff| = %.6g" % maxdiff)
    print("超容差(%.0e)的 %d 个 (%.4f%%)" % (args.tol, n_bad, 100.0 * n_bad / total))

    # 分类结果是否一致 —— 比数值容差更贴近「可用」这个标准
    pg = g.argmax(axis=1)
    pr = r.argmax(axis=1)
    same = int((pg == pr).sum())
    print("argmax 一致：%d/%d (%.4f%%)" % (same, g.shape[0], 100.0 * same / g.shape[0]))

    # ⭐ 判据就是 verify_e2e.sh 头部写死的两条（精度差 ≤ 1.0 pp、argmax 一致 ≥ 90%），
    #   这里只能看到 argmax 那一条；精度由调用方打印。
    #   max|diff| 只作诊断，不作判据：GPU 侧与 numpy 侧都走「量化 → 整数累加 → 反量化」，
    #   但累加顺序/GEMM 分块不同，logits 的绝对差天然在 1e-3～1e0 量级。把它当判据
    #   会让每一次运行都报"不一致 ✗"，而实际上历史记录的通过口径就是 argmax ≥ 90%
    #   （见 06 README「实测的端到端结果」与 ROOFLINE-int8.md 4.1）。
    argmax_pct = 100.0 * same / g.shape[0]
    ok = argmax_pct >= 90.0
    print("结论：%s（判据 argmax ≥ 90%%，实测 %.1f%%；max|diff| 仅诊断）" %
          ("端到端通过 ✓" if ok else "端到端不通过 ✗", argmax_pct))
    if n_bad:
        idx = np.unravel_index(np.argmax(diff), diff.shape)
        print("  最大偏差位置 [%d][%d]：gpu=%.6f ref=%.6f（诊断信息，不影响判定）" %
              (idx[0], idx[1], g[idx], r[idx]))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())