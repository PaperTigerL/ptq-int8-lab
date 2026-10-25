#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""量化算子的性质测试。

这里测的是**性质**，不是对答案 —— 你的实现只要满足这些性质就算对，
怎么写随你。判据都设计成「笨办法裁判」：
拿一个不需要被测函数就能算出来的参照物跟它比。

退出码
  0  全部通过
  1  有 FAIL
  3  有 TODO 还没填（不算错，是提醒）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import config as C
import src.quant as Q
from src._todo import TodoNotImplemented

ROWS = []


def rec(name, state, detail=""):
    ROWS.append((name, state, detail))
    if state != "PASS":
        pass


def check(name, cond, detail=""):
    rec(name, "PASS" if cond else "FAIL", detail)


def call(fn, *a, **k):
    """调用被测函数，三种结果分开记。

    ⛔ 早期版本只区分「拿到结果 / 没拿到」—— 于是 RuntimeError、TypeError
       这类**填了但写崩了**的情况，跟「还没填」一样被记成 TODO，
       退出码还是 3（提醒），错误就这么被藏起来了。

      · 正常返回           → PASS
      · TodoNotImplemented → TODO（还没填，不是错误）
      · 其它任何异常        → FAIL（填了，但写错了）
    """
    try:
        return fn(*a, **k), None, "PASS"
    except TodoNotImplemented as e:
        return None, str(e), "TODO"
    except Exception as e:
        return None, "%s: %s" % (type(e).__name__, e), "FAIL"


def sqnr(x, xq):
    mse = ((xq - x) ** 2).mean().item()
    pwr = (x ** 2).mean().item()
    return 10 * torch.log10(torch.tensor(pwr / max(mse, 1e-12))).item()


def is_on_grid(xq, scale):
    """量化后的值必须落在 scale 的整数倍上。"""
    q = xq / max(float(scale), 1e-12)
    return (q - torch.round(q)).abs().max().item() < 1e-3


# ---------------------------------------------------------------- 1
def test_per_tensor_sym():
    torch.manual_seed(0)
    x = torch.randn(1024, device=C.DEVICE) * 3.0
    xq, err, st = call(Q.quantize_per_tensor_sym, x, 8)
    if xq is None:
        rec("per_tensor_sym", st, err)
        return

    check("per_tensor_sym · dtype/device 不变",
          xq.dtype == x.dtype and xq.device == x.device,
          "%s vs %s" % (xq.dtype, x.dtype))

    sc = (x.abs().max() / 127)
    check("per_tensor_sym · 落在整数网格上", is_on_grid(xq, sc))

    e = sqnr(x, xq)
    check("per_tensor_sym · 8bit SQNR > 40dB", e > 40, "%.1fdB" % e)

    z = torch.zeros(16, device=C.DEVICE)
    zq, _, _ = call(Q.quantize_per_tensor_sym, z, 8)
    if zq is not None:
        check("per_tensor_sym · 全零输入不产生 nan",
              bool(torch.isfinite(zq).all()), "")

    # 码字不能浪费：8bit 至少要用掉 100 个不同整数
    n = Q.count_unique_ints(xq, sc)
    check("per_tensor_sym · 码字利用率 > 100/256", n > 100, "%d 个" % n)


# ---------------------------------------------------------------- 2
def test_per_channel_sym():
    """核心判据：通道间量级差 100 倍时，小通道不能被压成全零。

    这是 per-channel 存在的唯一理由。用 per-tensor 做一遍当反例。
    """
    torch.manual_seed(1)
    w = torch.randn(8, 4, 3, 3, device=C.DEVICE)
    w[0] *= 0.01      # 小通道
    w[1] *= 100.0     # 大通道

    wq, err, st = call(Q.quantize_per_channel_sym, w, 4)
    if wq is None:
        rec("per_channel_sym", st, err)
        return

    small_q = wq[0]
    small_o = w[0]
    rel = ((small_q - small_o).abs().max() / small_o.abs().max()).item()

    # 反例：per-tensor 会把小通道压没
    wt = Q.quantize_per_tensor_sym(w, 4)
    rel_pt = ((wt[0] - w[0]).abs().max() / w[0].abs().max()).item()

    check("per_channel_sym · 小通道相对误差 < 0.2", rel < 0.2, "%.3f" % rel)
    check("per_channel_sym · 明显优于 per-tensor",
          rel < rel_pt * 0.5,
          "per-channel %.3f  vs  per-tensor %.3f" % (rel, rel_pt))
    check("per_channel_sym · 各通道独立（大通道不为零）",
          float(wq[1].abs().max()) > 0)


# ---------------------------------------------------------------- 3
def test_per_tensor_asym():
    """核心判据：对 ReLU 后的非负输入，非对称必须比对称更省码字。

    对称量化会把 [-max, 0) 这一半码字浪费掉。
    """
    torch.manual_seed(2)
    x = torch.rand(4096, device=C.DEVICE) ** 2 * 5.0   # 非负，右偏
    xq, err, st = call(Q.quantize_per_tensor_asym, x, 4)
    if xq is None:
        rec("per_tensor_asym", st, err)
        return

    e_asym = sqnr(x, xq)
    xs = Q.quantize_per_tensor_sym(x, 4)
    e_sym = sqnr(x, xs)

    check("per_tensor_asym · 对非负输入优于对称量化",
          e_asym > e_sym + 1.0,
          "asym %.1fdB  vs  sym %.1fdB" % (e_asym, e_sym))
    check("per_tensor_asym · 量化后仍非负",
          float(xq.min()) >= -1e-6, "min=%.4f" % float(xq.min()))
    check("per_tensor_asym · dtype/device 不变",
          xq.dtype == x.dtype and xq.device == x.device)


# ---------------------------------------------------------------- 4
def test_scale_mse():
    """笨办法裁判：MSE 搜出来的 scale，误差不该比 minmax 更差。

    构造一个带离群点的分布 —— min/max 会被离群点带偏，
    MSE 搜索应当找到更紧的区间。
    """
    torch.manual_seed(3)
    x = torch.randn(20000, device=C.DEVICE)
    x[0] = 50.0        # 一个离群点
    x[1] = -48.0

    s, err, st = call(Q.scale_mse, x, 4)
    if s is None:
        rec("scale_mse", st, err)
        return

    qmax = 7
    s_mm = float(x.abs().max() / qmax)
    e_mse = ((torch.round(x / float(s)).clamp(-8, 7) * float(s) - x) ** 2).mean().item()
    e_mm = ((torch.round(x / s_mm).clamp(-8, 7) * s_mm - x) ** 2).mean().item()

    check("scale_mse · 优于 minmax", e_mse < e_mm,
          "mse %.5f  vs  minmax %.5f  (改善 %.1f%%)"
          % (e_mse, e_mm, (1 - e_mse / e_mm) * 100))
    check("scale_mse · 返回标量且为正", float(s) > 0, "s=%.5f" % float(s))


# ---------------------------------------------------------------- 5
def test_scale_entropy():
    torch.manual_seed(4)
    x = torch.randn(20000, device=C.DEVICE).abs()
    x[0] = 60.0

    s, err, st = call(Q.scale_entropy, x, 4)
    if s is None:
        rec("scale_entropy", st, err)
        return

    qmax = 7
    s_mm = float(x.abs().max() / qmax)
    e_kl = ((torch.round(x / float(s)).clamp(-8, 7) * float(s) - x) ** 2).mean().item()
    e_mm = ((torch.round(x / s_mm).clamp(-8, 7) * s_mm - x) ** 2).mean().item()

    check("scale_entropy · 不差于 minmax", e_kl <= e_mm * 1.05,
          "entropy %.5f  vs  minmax %.5f" % (e_kl, e_mm))
    check("scale_entropy · 返回标量且为正", float(s) > 0, "s=%.5f" % float(s))


def main():
    print("=" * 72)
    print("量化算子性质测试")
    print("=" * 72)
    print("设备     : %s" % C.device_info())
    print("被测对象 : %s" % Q.which_mode())
    print("")

    test_per_tensor_sym()
    test_per_channel_sym()
    test_per_tensor_asym()
    test_scale_mse()
    test_scale_entropy()

    print("%-26s %-6s %s" % ("检查项", "结果", "细节"))
    print("-" * 72)
    for n, s, d in ROWS:
        print("%-26s %-6s %s" % (n, s, d))
    print("-" * 72)

    n_pass = sum(1 for _, s, _ in ROWS if s == "PASS")
    n_fail = sum(1 for _, s, _ in ROWS if s == "FAIL")
    n_todo = sum(1 for _, s, _ in ROWS if s == "TODO")

    print("")
    print("通过 %d / 失败 %d / 未实现 %d" % (n_pass, n_fail, n_todo))
    if n_todo:
        print("")
        print("还有 %d 个 TODO 没填 —— 这不是错误，是提醒。" % n_todo)
        print("填完 src/quant.py 后重跑本测试。")
    if n_fail:
        print("")
        print("有 %d 项没通过，按上面「细节」列的数值定位。" % n_fail)

    if n_fail:
        return 1
    if n_todo:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
