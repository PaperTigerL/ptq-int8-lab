# -*- coding: utf-8 -*-
"""逐 block 比对 GPU 与 numpy 参考的中间激活，找出分歧起点。

配合 locate_block（GPU 侧导出）使用：
  python locate_block.py --ref-prefix /tmp/ref_mid --gpu-prefix /tmp/gpu_mid

输出的每一行是「跑到第几个 block 之后」的 max|diff。
**第一个 max|diff 明显跳大的位置就是分歧起点**——
它的上一个 block 还是好的，说明错误就在那一个 block 里。
"""
import argparse
import struct

import numpy as np


def load_mid(path):
    with open(path, "rb") as f:
        d = f.read()
    assert d[:4] == b"PTQM", f"{path} magic 不符"
    n, c, h, w = struct.unpack_from("<4i", d, 4)
    off = 20
    return np.frombuffer(d, dtype="<f4", count=n * c * h * w, offset=off).reshape(n, c, h, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--n-blocks", type=int, default=8)
    ap.add_argument("--gpu-prefix", default="/tmp/gpu_mid_b")
    ap.add_argument("--ref-prefix", default="/tmp/ref_mid_b")
    args = ap.parse_args()

    print("%-8s %-22s %-12s %-12s %s" %
          ("block", "shape", "max|diff|", "相对", "判定"))
    prev_rel = 0.0
    first_bad = None
    for bi in range(args.n_blocks):
        try:
            g = load_mid("%s%d.bin" % (args.gpu_prefix, bi))
            r = load_mid("%s%d.bin" % (args.ref_prefix, bi))
        except FileNotFoundError as e:
            print("%-8d <缺失: %s>" % (bi, e))
            break
        if g.shape != r.shape:
            print("%-8d 形状不同 %s vs %s" % (bi, g.shape, r.shape))
            first_bad = bi
            break

        diff = np.abs(g.astype(np.float64) - r.astype(np.float64))
        mx = float(diff.max())
        scale = max(float(np.abs(r).max()), 1e-12)
        rel = mx / scale

        # 判定标准：相对偏差 > 0.5% 就算「这一层之后已经错了」。
        # 为什么用相对而不是绝对：不同层的激活量级差 10 倍以上，
        # 绝对阈值会让深层永远「通过」。
        bad = rel > 0.005
        if bad and first_bad is None:
            first_bad = bi
        jump = "  ← 分歧起点" if (bad and (first_bad == bi) and bi > 0 and prev_rel <= 0.005) else ""
        print("%-8d %-22s %-12.6g %-12.6g %s%s" %
              (bi, str(g.shape), mx, rel, "超差" if bad else "OK", jump))
        prev_rel = rel

    print()
    if first_bad is None:
        print("结论：全部 block 都在容差内 —— 与 GPU 实现一致 ✓")
    elif first_bad == 0:
        print("结论：第 0 个 block 就超差 —— 问题在 stem（conv1 + maxpool）")
    else:
        print("结论：分歧起点 = block %d（它的上一个 block %d 仍 OK）" %
              (first_bad, first_bad - 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())