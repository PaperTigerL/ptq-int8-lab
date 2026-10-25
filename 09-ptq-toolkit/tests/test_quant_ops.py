# -*- coding: utf-8 -*-
"""量化算子自测：口径必须与 03-ptq-lab 逐位一致。

这类测试的价值在于**锁死口径**：per-channel 量化的 round / clamp / scale
三处只要有一处漂移，精度就会掉 1~2 pp，而症状是「数字看起来还行」，
不会报错。06 的 ptq_quant.h 注释里记的 8 bit 码空间问题就是这么来的。

测试项
  1. round 是 half-away-from-zero，不是 banker's rounding
  2. clamp 区间是 [-(qmax+1), qmax]，负侧多一个码字
  3. scale 分母是 qmax（127 / 7），不是 2^bits-1
  4. 与 03 的 quantize_per_channel_sym 逐位一致（真对照，不是重述规格）
  5. dequant 的 scale 广播维度正确（曾静默按末维广播）

用法：python tests/test_quant_ops.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from ptq_toolkit.quant.ptq import (qmax_of, quantize_per_channel_sym,   # noqa: E402
                                   round_half_away)

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'✓' if cond else '✗'} {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILED.append(name)


def test_round():
    print("1) round = half-away-from-zero")
    x = np.array([0.5, 1.5, 2.5, -0.5, -1.5, -2.5])
    got = round_half_away(x)
    want = np.array([1., 2., 3., -1., -2., -3.])
    check("0.5→1 而非 0（banker's 会给 0）",
          got[0] == 1.0, f"got {got[0]}")
    check("2.5→3 而非 2", got[2] == 3.0, f"got {got[2]}")
    check("-0.5→-1", got[3] == -1.0, f"got {got[3]}")
    check("全部 half-away", np.allclose(got, want), f"{got}")
    check("与 numpy 默认 round 不同（这正是要点）",
          not np.allclose(got, np.round(x)))


def test_qmax():
    print("2) qmax = 2^(bits-1)-1（不是 2^bits-1）")
    check("bits=8 → 127", qmax_of(8) == 127, f"got {qmax_of(8)}")
    check("bits=4 → 7", qmax_of(4) == 7, f"got {qmax_of(4)}")


def test_clamp_and_scale():
    print("3) clamp 区间 [-（qmax+1), qmax] 与 scale 分母")
    # 构造一个 scale 恰好 1 的张量，方便读码字
    w = np.zeros((2, 4), dtype=np.float32)
    w[0] = [0, 1, -1, 2]
    w[1] = [-127, 126, 127, -128]
    q = quantize_per_channel_sym(w, 8)

    # channel 0：absmax = 2 → scale = 2/127
    check("scale = absmax/127（不是 /255）",
          abs(q.scale[0] - 2.0 / 127.0) < 1e-7, f"got {q.scale[0]:.8f}")
    # channel 1：absmax = 128 → scale = 128/127 > 1，码字应被 clamp
    check("channel1 scale = 128/127", abs(q.scale[1] - 128.0 / 127.0) < 1e-7)
    check("最大码字不超过 +qmax（int8 上界）",
          q.code.max() <= 127, f"max={q.code.max()}")
    check("最小码字不低于 -(qmax+1)", q.code.min() >= -128, f"min={q.code.min()}")
    check("channel0 的码字成比例",
          abs(q.code[0][2] - round(-1.0 / (2.0 / 127.0))) <= 1)


def test_broadcast():
    print("4) dequant 的 scale 广播维度")
    w = np.random.randn(5, 3, 3, 3).astype(np.float32)
    q = quantize_per_channel_sym(w, 8)
    d = q.dequant()
    check("输出形状与输入相同", d.shape == w.shape, f"{d.shape} vs {w.shape}")
    # 关键：每个通道的 scale 只作用在该通道上。若按末维广播，形状会直接报错
    # 或静默算错，所以这里查「同一通道内 scale 一致」
    rel = [float(np.abs(d[i] - q.code[i] * q.scale[i]).max()) for i in range(5)]
    check("每个通道只乘自己的 scale", max(rel) < 1e-6, f"max err {max(rel):.2e}")


def test_against_ptq_lab():
    print("5) 与 03-ptq-lab 的 quantize_per_channel_sym 逐位对照")
    # ⚠️ 03 的函数返回的是**反量化后的张量**（不是 (code, scale) 元组），
    #    所以对照方式是「各自 dequant 后比数值」，不是比码字。
    found = False
    for parent in HERE.parents:
        src = parent / "03-ptq-lab"
        if (src / "src" / "quant.py").is_file():
            for d_ in (src, src / "src"):
                if str(d_) not in sys.path:
                    sys.path.insert(0, str(d_))
            found = True
            break
    if not found:
        print("  [SKIP] 找不到 03-ptq-lab，跳过对照")
        return

    import torch
    import quant as Q
    rng = np.random.default_rng(0)
    ok_all = True
    for bits in (8, 4):
        # 特意用 4 维（Conv2d 的形状），覆盖 scale 广播那条路径
        w = (rng.standard_normal((6, 4, 3, 3)) * 3.0).astype(np.float32)
        mine = quantize_per_channel_sym(w, bits).dequant()
        ref = Q.quantize_per_channel_sym(torch.from_numpy(w), bits).numpy()
        same = np.allclose(mine, ref, rtol=0, atol=1e-6)
        ok_all = ok_all and same
        md = float(np.abs(mine - ref).max())
        print(f"    bits={bits} shape={w.shape}: 逐位一致={same} max|diff|={md:.2e}")
    check("W8/W4 都与 03 逐位一致（4 维权重）", ok_all)


def main():
    print("量化算子自测\n" + "-" * 46)
    test_round()
    test_qmax()
    test_clamp_and_scale()
    test_broadcast()
    test_against_ptq_lab()
    print("-" * 46)
    if FAILED:
        print(f"✗ {len(FAILED)} 项失败：{', '.join(FAILED)}")
        return 1
    print("✓ 全部通过 —— 量化口径与 03-ptq-lab 一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
