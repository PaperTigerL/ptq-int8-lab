#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""量化算子的边界测试。

和 tests/test_quant.py 的分工：
  test_quant.py    测「性质」—— 精度够不够、有没有比基线好。那是 correctness。
  test_edge_cases.py 测「健壮性」—— 奇形怪状的输入会不会炸。那是 robustness。

边界不是凑数。量化算子在真实流程里会被喂到各种形状：
空张量（某个分支没激活）、单元素（bias）、常量张量（BN 折叠后某层全同）、
非 contiguous（transpose 的视图）、float64（numpy 转过来的）、
以及极端离群点（校准集里混进一张坏图）。任何一个炸掉，实验就跑不完。

退出码
  0  全部通过
  1  有 FAIL
"""
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import config as C
import src.quant as Q
from src._todo import TodoNotImplemented

ROWS = []


def check(name, cond, detail=""):
    ROWS.append((name, "PASS" if cond else "FAIL", detail))


def guard(fn, group):
    """跑一组测试，三种结果分开记。

      · TodoNotImplemented → TODO（还没填，不是错误，退出码 3）
      · 组内有断言没过     → FAIL（写在各组内部，走 check()）
      · 其它异常（崩了）   → FAIL，并指出崩在被测代码的哪一行

    ⛔ 早期版本只拦 TodoNotImplemented。于是一个 RuntimeError 就能让
       整个测试文件当场挂掉，后面的分组一行都看不见 ——
       跟「留白状态下也能逐步看进度」的初衷正好相反。
       实测触发方式：空张量上调用 x.abs().max()，
       抛 RuntimeError: max(): Expected reduction dim...，整个文件就死了。
    """
    try:
        fn()
    except TodoNotImplemented as e:
        ROWS.append((group, "TODO", str(e)))
    except Exception as e:
        tb = traceback.extract_tb(sys.exc_info()[2])[-1]
        ROWS.append((group, "FAIL", "%s: %s    崩在 %s:%d"
                     % (type(e).__name__, e, os.path.basename(tb.filename), tb.lineno)))


def finite(t):
    return bool(torch.isfinite(t).all())


# ================================================================= A 位宽边界
def test_bits_range():
    x = torch.randn(64, device=C.DEVICE)

    x2 = Q.quantize_per_tensor_sym(x, 2)
    check("bits=2 可用（最小位宽）",
          finite(x2) and x2.shape == x.shape,
          "码字 %d 个" % Q.count_unique_ints(x2, float(x.abs().max() / 1)))

    x16 = Q.quantize_per_tensor_sym(x, 16)
    e16 = ((x16 - x) ** 2).mean().item()
    check("bits=16 精度显著优于 bits=2",
          e16 < ((x2 - x) ** 2).mean().item(),
          "mse16=%.3e" % e16)

    try:
        Q.quantize_per_tensor_sym(x, 1)
        check("bits=1 应被明确拒绝", False, "没抛异常")
    except ValueError as e:
        check("bits=1 应被明确拒绝", True, "ValueError: %s" % e)


# ============================================================= B 形状与 dtype
def test_shapes_and_dtype():
    # 空张量：某分支没激活时会遇到。max() 在空张量上会抛错，必须挡住
    e = torch.empty(0, device=C.DEVICE)
    check("空张量 · sym 不崩", Q.quantize_per_tensor_sym(e, 8).numel() == 0)
    check("空张量 · asym 不崩", Q.quantize_per_tensor_asym(e, 8).numel() == 0)
    check("空张量 · per-channel 不崩",
          Q.quantize_per_channel_sym(torch.empty(0, 4, 3, 3, device=C.DEVICE), 8).numel() == 0)
    check("空张量 · scale_mse 不崩", float(Q.scale_mse(e, 8)) > 0)
    check("空张量 · scale_entropy 不崩", float(Q.scale_entropy(e, 8)) > 0)

    # 单元素 / 全零 / 常量
    one = torch.tensor([3.7], device=C.DEVICE)
    check("单元素张量", finite(Q.quantize_per_tensor_sym(one, 4)))
    check("全零张量不出 nan", finite(Q.quantize_per_tensor_sym(
        torch.zeros(32, device=C.DEVICE), 8)))
    const = torch.full((32,), 2.5, device=C.DEVICE)
    check("常量张量 · sym 不 nan", finite(Q.quantize_per_tensor_sym(const, 4)))
    check("常量张量 · asym 不 nan", finite(Q.quantize_per_tensor_asym(const, 4)))
    check("常量张量 · scale_mse 不 nan", math_ok(Q.scale_mse(const, 4)))

    # float64 必须原样保持（numpy 转进来的张量是 double）
    d = torch.randn(32, dtype=torch.float64)
    dq = Q.quantize_per_tensor_sym(d, 8)
    check("float64 输入保持 dtype", dq.dtype == torch.float64, str(dq.dtype))

    # 非 contiguous：transpose 出来的是视图，reshape 会踩坑
    base = torch.randn(16, 8, device=C.DEVICE)
    nc = base.transpose(0, 1)
    check("非 contiguous 输入",
          not nc.is_contiguous() and finite(Q.quantize_per_tensor_sym(nc, 8)))

    # per-channel 的形状与 dim
    w = torch.randn(8, 4, 3, 3, device=C.DEVICE)
    w[0] *= 0.01
    w[1] *= 100.0
    for dim in (0, 1, -1):
        wq = Q.quantize_per_channel_sym(w, 4, dim=dim)
        check("per-channel dim=%d · 形状不变" % dim, wq.shape == w.shape, str(wq.shape))
        check("per-channel dim=%d · dtype 不变" % dim, wq.dtype == w.dtype)

    # dim=1 时通道轴在 in_channels 上：每个 in_channel 独立，误差都得小
    wq1 = Q.quantize_per_channel_sym(w, 4, dim=1)
    rels = []
    for i in range(w.shape[1]):
        o = w[:, i]
        q = wq1[:, i]
        rels.append((q - o).abs().max().item() / o.abs().max().item())
    check("per-channel dim=1 · 各 in_channel 误差 < 0.2",
          max(rels) < 0.2, "max rel=%.3f" % max(rels))


def math_ok(s):
    s = float(s)
    return s > 0 and s == s and s != float("inf")


# ================================================================= C 数值性质
def test_numeric_properties():
    torch.manual_seed(11)
    x = torch.randn(512, device=C.DEVICE) * 2.0

    # 幂等：量化后的值已经在网格上，再量化一次不该再变
    xq1 = Q.quantize_per_tensor_sym(x, 4)
    xq2 = Q.quantize_per_tensor_sym(xq1, 4)
    check("sym 幂等（量化两次 == 一次）",
          torch.allclose(xq1, xq2, atol=1e-6),
          "max diff=%.2e" % float((xq1 - xq2).abs().max()))

    # 值域：对称量化后绝对值不该超过原 absmax（网格上界就是 absmax）
    check("sym 值域不超界",
          float(xq1.abs().max()) <= float(x.abs().max()) + 1e-5,
          "%.4f vs %.4f" % (float(xq1.abs().max()), float(x.abs().max())))

    # 确定性：同一输入重复调用必须一致（否则实验不可复现）
    check("sym 确定性", torch.allclose(xq1, Q.quantize_per_tensor_sym(x, 4)))

    # 非对称：全负输入（BN 折叠后某些层可能全负）
    neg = -torch.rand(512, device=C.DEVICE) * 3.0 - 1.0
    nq = Q.quantize_per_tensor_asym(neg, 4)
    check("asym · 全负输入不 nan 且保持符号",
          finite(nq) and float(nq.max()) <= 0 + 1e-5,
          "max=%.4f" % float(nq.max()))

    lo, hi = float(neg.min()), float(neg.max())
    sc = max((hi - lo) / 15, 1e-8)
    check("asym · 值域不超出 [lo-scale, hi+scale]",
          float(nq.min()) >= lo - sc - 1e-5 and float(nq.max()) <= hi + sc + 1e-5,
          "[%.3f, %.3f] vs [%.3f, %.3f]" % (float(nq.min()), float(nq.max()), lo, hi))

    xa = torch.rand(512, device=C.DEVICE) ** 2 * 5.0
    xaq = Q.quantize_per_tensor_asym(xa, 4)
    check("asym 幂等", torch.allclose(xaq, Q.quantize_per_tensor_asym(xaq, 4), atol=1e-6))
    check("asym · 非负输入量化后仍非负", float(xaq.min()) >= -1e-6,
          "min=%.6f" % float(xaq.min()))

    # 极端离群点：校准集混进坏图时的极端情况。
    # 判据不是「离群点下也要准」—— 那做不到。判据是「MSE 校准能把它救回来」，
    # 也就是这一条正是 scale_mse / scale_entropy 存在的理由。
    out = torch.randn(4096, device=C.DEVICE)
    out[0] = 1e6
    out[1] = -1e6
    oq = Q.quantize_per_tensor_sym(out, 8)
    check("极端离群点 · 不 inf 不 nan", finite(oq))

    # 极端离群点下 **MSE 也救不回来**：单个 1e6 的平方误差是 1e12，
    # 会完全主导目标函数 —— 容纳它反而是 MSE 意义下的最优解。
    # 这是 MSE 校准的能力边界，不是 bug。能救的是 percentile：
    # 它直接掐掉尾部、根本不看误差，所以不受离群点的量级影响。
    body = out[2:]
    e_mm = float(((oq[2:] - body) ** 2).mean())

    s_mse = float(Q.scale_mse(out, 8))
    oq2 = torch.round(out / s_mse).clamp(-128, 127) * s_mse
    e_mse = float(((oq2[2:] - body) ** 2).mean())

    s_pct = float(Q.scale_percentile(out, 8))
    oq3 = torch.round(out / s_pct).clamp(-128, 127) * s_pct
    e_pct = float(((oq3[2:] - body) ** 2).mean())

    check("极端离群点 · MSE 救不回、percentile 能救回",
          e_mse > e_mm * 0.5 and e_pct < e_mm * 0.01,
          "minmax %.3e / mse %.3e / percentile %.3e" % (e_mm, e_mse, e_pct))


# ================================================================= D 设备一致
def test_device_consistency():
    if not torch.cuda.is_available():
        check("CPU/CUDA 一致性", True, "无 CUDA，跳过")
        return
    torch.manual_seed(12)
    xc = torch.randn(256) * 2.0
    xg = xc.cuda()
    a = Q.quantize_per_tensor_sym(xc, 4).cpu()
    b = Q.quantize_per_tensor_sym(xg, 4).cpu()
    check("CPU/CUDA 一致性 · sym", torch.allclose(a, b, atol=1e-6),
          "max diff=%.2e" % float((a - b).abs().max()))

    xa = (torch.rand(256) ** 2 * 5.0)
    xag = xa.cuda()
    c = Q.quantize_per_tensor_asym(xa, 4).cpu()
    d = Q.quantize_per_tensor_asym(xag, 4).cpu()
    check("CPU/CUDA 一致性 · asym", torch.allclose(c, d, atol=1e-6),
          "max diff=%.2e" % float((c - d).abs().max()))


# ================================================================= E 校准策略
def test_calibration_strategies():
    torch.manual_seed(13)
    clean = torch.randn(20000, device=C.DEVICE)
    s_mm = float(clean.abs().max() / 7)

    # 判据是「不差于 minmax」，不是「等于 minmax」。
    # MSE 会主动收紧上界、牺牲尾部换主体精度 —— 即使没有离群点也照样收紧，
    # 这是它的设计目标，不是 bug。
    def sym_mse(x, s):
        return float(((torch.round(x / s).clamp(-8, 7) * s - x) ** 2).mean())

    s_mse = float(Q.scale_mse(clean, 4))
    check("scale_mse · 无离群点时也不差于 minmax",
          sym_mse(clean, s_mse) <= sym_mse(clean, s_mm) * 1.001,
          "err %.5f vs minmax %.5f (scale %.4f vs %.4f)"
          % (sym_mse(clean, s_mse), sym_mse(clean, s_mm), s_mse, s_mm))

    s_ent = float(Q.scale_entropy(clean, 4))
    check("scale_entropy · 返回标量且为正", s_ent > 0, "s=%.4f" % s_ent)

    # 带离群点时必须明显收紧
    dirty = clean.clone()
    dirty[0] = 50.0
    dirty[1] = -48.0
    s_mm_d = float(dirty.abs().max() / 7)
    s_mse_d = float(Q.scale_mse(dirty, 4))
    check("scale_mse · 有离群点时明显收紧",
          s_mse_d < s_mm_d * 0.5,
          "mse=%.4f vs minmax=%.4f" % (s_mse_d, s_mm_d))
    s_ent_d = float(Q.scale_entropy(dirty, 4))
    check("scale_entropy · 有离群点时明显收紧",
          s_ent_d < s_mm_d * 0.5,
          "entropy=%.4f vs minmax=%.4f" % (s_ent_d, s_mm_d))

    # 常量输入（BN 折叠后某层权重可能全同）
    const = torch.full((2048,), 1.5, device=C.DEVICE)
    check("scale_mse · 常量输入不 nan", math_ok(Q.scale_mse(const, 4)))
    check("scale_entropy · 常量输入不 nan", math_ok(Q.scale_entropy(const, 4)))


# ================================================================= F 主流程冒烟
def test_pipeline_smoke():
    """主流程冒烟：加载 → BN 折叠 → 建量化模型 → 前向一次。

    不跑完整评估（那是 e1-e7 的事，要几分钟）。这里只确认
    「量化算子接进真实模型后不会炸」，几十秒内能出结果。
    """
    ok, missing = C.require_assets(need_ckpt=True)
    if not ok:
        check("主流程冒烟", False, "缺依赖: %s" % ", ".join(p for _, p in missing))
        return

    import src.model as M
    m = M.load_fp32(C.CKPT_MAIN).to(C.DEVICE).eval()
    M.fold_all_bn(m)

    mq = M.build_quant_model(m, w_bits=8, a_bits=8).to(C.DEVICE).eval()
    with torch.no_grad():
        y = mq(torch.randn(4, 3, 32, 32, device=C.DEVICE))
    check("主流程冒烟 · W8A8 前向",
          finite(y) and y.shape[0] == 4 and y.shape[1] == 10,
          "输出 %s" % (list(y.shape),))

    mq4 = M.build_quant_model(m, w_bits=4, a_bits=4).to(C.DEVICE).eval()
    with torch.no_grad():
        y4 = mq4(torch.randn(4, 3, 32, 32, device=C.DEVICE))
    check("主流程冒烟 · W4A4 前向", finite(y4), "输出 %s" % (list(y4.shape),))


# ================================================================= G 可复现性
def test_calib_reproducible():
    """同一个 seed 下，两次采集的激活统计量必须完全一致。

    这条不是凑数。降采样原来用 CUDA 随机数（`torch.randint(..., device=f.device)`），
    而 CUDA 的随机数不受 `torch.manual_seed` 控制 —— 于是同一个 CALIB_SEED
    每次跑出来的校准集都不一样，实验根本不可复现。
    实测后果：e3 的 entropy 档两次运行一个 86.46%、一个 67.66%，差 19 个点。
    """
    import torch.nn as nn
    import torch.utils.data as tud
    import src.calib as CB

    data = torch.randn(256, 3, 8, 8)
    loader = tud.DataLoader(
        tud.TensorDataset(data, torch.zeros(256, dtype=torch.long)),
        batch_size=64)

    def collect():
        torch.manual_seed(C.CALIB_SEED)
        m = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU())
        m = m.to(C.DEVICE).eval()
        return CB.collect_activations(m, loader, samples_per_layer=100)

    s1 = collect()
    s2 = collect()
    ok = set(s1.keys()) == set(s2.keys())
    detail = "层数 %d" % len(s1)
    for k in s1:
        if k not in s2:
            ok = False
            break
        a, b = float(s1[k].abs().max()), float(s2[k].abs().max())
        if abs(a - b) > 1e-6 or s1[k].numel() != s2[k].numel():
            ok = False
            detail = "%s: %.6f vs %.6f (%d vs %d)" % (
                k, a, b, s1[k].numel(), s2[k].numel())
            break
    check("校准采样可复现（同 seed 两次一致）", ok, detail)


def main():
    print("=" * 72)
    print("量化算子边界测试")
    print("=" * 72)
    print("设备     : %s" % C.device_info())
    print("被测对象 : %s" % Q.which_mode())
    print("")

    guard(test_bits_range, "A 位宽边界组")
    guard(test_shapes_and_dtype, "B 形状与 dtype 组")
    guard(test_numeric_properties, "C 数值性质组")
    guard(test_device_consistency, "D 设备一致组")
    guard(test_calibration_strategies, "E 校准策略组")
    guard(test_calib_reproducible, "G 可复现性组")
    guard(test_pipeline_smoke, "F 主流程冒烟组")

    print("%-42s %-6s %s" % ("检查项", "结果", "细节"))
    print("-" * 72)
    for n, s, d in ROWS:
        print("%-42s %-6s %s" % (n, s, d))
    print("-" * 72)

    n_pass = sum(1 for _, s, _ in ROWS if s == "PASS")
    n_fail = sum(1 for _, s, _ in ROWS if s == "FAIL")
    n_todo = sum(1 for _, s, _ in ROWS if s == "TODO")
    print("")
    print("通过 %d / 失败 %d / 未实现 %d" % (n_pass, n_fail, n_todo))
    if n_todo:
        print("")
        print("还有 %d 组没跑起来 —— 对应的函数还没填，这是提醒不是错误。" % n_todo)
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
