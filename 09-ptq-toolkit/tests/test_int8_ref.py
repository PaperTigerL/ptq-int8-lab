# -*- coding: utf-8 -*-
"""int8 conv/gemm CPU 参考实现的单测。

锁定 06 引擎路径的六类语义：
  1. 对称 AbsMax 与非对称 zero-point 的 scale/码空间/零点完全分离
  2. 权重 per-channel 对称 W8/W4：scale 分母是 qmax，不是 2^bits-1
  3. 卷积的零填充 / 步长 / 分组 / depthwise
  4. 残差相加与 ReLU6 边界（激活必须在 add 之后）
  5. 激活跨零范围（lo<0<hi）不裁剪
  6. padding 补真实 0：对称补码字 0，非对称补 zero_point

用法：python tests/test_int8_ref.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from ptq_toolkit.ref_int8 import (AsymActQuant, SymActQuant, add_residual,  # noqa: E402
                                  conv2d_int8, gemm_int8,
                                  quantize_weight_per_channel_sym, relu6)
from ptq_toolkit.quant.ptq import round_half_away                     # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'✓' if cond else '✗'} {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILED.append(name)


def close(name, got, want, atol=1e-5, rtol=1e-5):
    got = np.asarray(got)
    want = np.asarray(want)
    diff = np.max(np.abs(got - want)) if got.size else 0.0
    check(name, got.shape == want.shape and np.allclose(
        got, want, atol=atol, rtol=rtol),
        f"max|diff|={diff:.3e}, shape={got.shape}")


def _torch():
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    return torch, F


def _dequant_input(aq, x):
    return aq.dequantize(aq.quantize(x))


def _dequant_weight(w, bits=8):
    code, scale = quantize_weight_per_channel_sym(w, bits)
    view = [scale.shape[0]] + [1] * (w.ndim - 1)
    return code.astype(np.float32) * scale.reshape(view)


# ---------------------------------------------------------------- 1) 激活口径

def test_activation_modes():
    print("1) 对称 AbsMax 与非对称 zp 的口径分离")
    lo, hi = -2.0, 4.0
    sym = SymActQuant.from_range(lo, hi)
    asym = AsymActQuant.from_range(lo, hi)

    check("AbsMax scale = max(|lo|,|hi|)/127",
          np.isclose(sym.scale, 4.0 / 127.0),
          f"got {sym.scale:.9f}")
    check("MinMax 公式仅作对照且 scale 更粗",
          np.isclose(SymActQuant.from_range(lo, hi, mode="minmax").scale,
                     6.0 / 127.0))
    check("MinMax scale > AbsMax scale",
          SymActQuant.from_range(lo, hi, mode="minmax").scale > sym.scale)

    check("对称码空间 [-(qmax+1), qmax]",
          (sym.qmin, sym.qmax) == (-128, 127),
          f"got ({sym.qmin}, {sym.qmax})")
    check("非对称码空间 [0, 2^bits-1]",
          (asym.qmin, asym.qmax) == (0, 255),
          f"got ({asym.qmin}, {asym.qmax})")
    check("对称 zero_point 恒为 0", sym.zero_point == 0)
    check("非对称 zero_point = round(-lo/scale)",
          asym.zero_point == 85,
          f"got zp={asym.zero_point}, scale={asym.scale:.9f}")

    xs = np.array([lo, 0.0, hi], dtype=np.float32)
    sym_codes = sym.quantize(xs)
    asym_codes = asym.quantize(xs)
    check("对称码字 [-64, 0, 127]（负侧未用满 -128）",
          np.array_equal(sym_codes, np.array([-64, 0, 127], dtype=np.int8)),
          f"got {sym_codes.tolist()}")
    check("非对称码字 [0, 85, 255]（零点居中平移）",
          np.array_equal(asym_codes, np.array([0, 85, 255], dtype=np.uint8)),
          f"got {asym_codes.tolist()}")
    check("同一真实 0 在两种口径下码字不同（0 vs 85）",
          sym_codes[1] != asym_codes[1])

    dense = np.linspace(lo, hi, 10001, dtype=np.float32)
    sym_dq = sym.dequantize(sym.quantize(dense))
    asym_dq = asym.dequantize(asym.quantize(dense))
    check("对称跨零范围无裁剪、误差 <=0.5*scale",
          np.max(np.abs(sym_dq - dense)) <= 0.5 * sym.scale + 1e-7,
          f"max err={np.max(np.abs(sym_dq - dense)):.3e}")
    check("非对称跨零范围无裁剪、误差 <=0.5*scale",
          np.max(np.abs(asym_dq - dense)) <= 0.5 * asym.scale + 1e-7,
          f"max err={np.max(np.abs(asym_dq - dense)):.3e}")
    check("对称端点码字都在 int8 范围内",
          np.all(sym.quantize(dense) >= -128)
          and np.all(sym.quantize(dense) <= 127))
    check("非对称端点码字都在 uint8 范围内",
          np.all(asym.quantize(dense) >= 0)
          and np.all(asym.quantize(dense) <= 255))

    sym_full = SymActQuant.from_range(-4.0, 4.0)
    check("对称零区间 lo=-4,hi=4 时端点是 -127/127",
          np.array_equal(sym_full.quantize(np.array([-4.0, 4.0])),
                         np.array([-127, 127], dtype=np.int8)))

    for bad in (lambda: SymActQuant(1.0, zero_point=1),
                lambda: AsymActQuant(1.0, zero_point=256),
                lambda: SymActQuant.from_range(1.0, -1.0)):
        try:
            bad()
            check("非法参数必须抛错", False)
        except (ValueError, TypeError):
            check("非法参数必须抛错", True)


def test_rounding_and_clamp():
    print("2) half-away-from-zero 与码字裁剪")
    sym = SymActQuant(scale=1.0)
    got = sym.quantize(np.array([0.5, -0.5, 1.5, -1.5, 2.5, -2.5]))
    check("±0.5 远离零取整（不是 banker's）",
          np.array_equal(got, np.array([1, -1, 2, -2, 3, -3],
                                       dtype=np.int8)),
          f"got {got.tolist()}")
    check("对称上界裁到 127 / 下界 -128",
          np.array_equal(sym.quantize(np.array([1e9, -1e9])),
                         np.array([127, -128], dtype=np.int8)))
    asym = AsymActQuant(scale=1.0, zero_point=128)
    check("非对称上界裁到 255 / 下界 0",
          np.array_equal(asym.quantize(np.array([1e9, -1e9])),
                         np.array([255, 0], dtype=np.uint8)))
    check("round_half_away 与 torch.round 同语义",
          np.array_equal(round_half_away(np.array([0.5, -0.5])),
                         np.array([1.0, -1.0])))


# ---------------------------------------------------------------- 2) 权重量化

def test_weight_quantization():
    print("3) per-channel 对称权重（W8/W4）")
    rng = np.random.default_rng(20261022)
    w = rng.normal(size=(4, 3, 3, 3)).astype(np.float32)
    w[0] = 0.0
    w[1, 0, 0, 0] = -5.0

    for bits in (8, 4):
        qmax = (1 << (bits - 1)) - 1
        code, scale = quantize_weight_per_channel_sym(w, bits)
        amax = np.abs(w.reshape(4, -1)).max(axis=1)
        check(f"W{bits} scale[c]=amax[c]/{qmax}",
              np.allclose(scale[1:], amax[1:] / qmax, rtol=1e-6))
        check(f"W{bits} 零通道 scale=1e-8 且码字全 0",
              np.isclose(scale[0], 1e-8) and np.all(code[0] == 0))
        check(f"W{bits} 码字在 [-(qmax+1), qmax]",
              np.all(code >= -(qmax + 1)) and np.all(code <= qmax),
              f"min={code.min()}, max={code.max()}")
        flat = w[1].reshape(-1)
        k = int(np.abs(flat).argmax())
        check(f"W{bits} 最大幅值映射到 ±qmax",
              abs(int(code[1].reshape(-1)[k])) == qmax,
              f"code={code[1].reshape(-1)[k]}, want ±{qmax}")


# ---------------------------------------------------------------- 3) 卷积

def _conv_case(act_cls, groups, stride, padding, activation, seed):
    rng = np.random.default_rng(seed)
    n, cin, h, w = 2, 4, 7, 8
    if groups == 4:
        cout = cin
    elif groups == 2:
        cout = 6
    else:
        cout = 4
    x = (rng.normal(size=(n, cin, h, w)).astype(np.float32) * 1.5 + 0.6)
    x[0, 0, 0, 0] = -3.0
    x[0, 1, 0, 0] = 5.0
    weight = rng.normal(size=(cout, cin // groups, 3, 3)).astype(np.float32)
    bias = rng.normal(size=cout).astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    aq = act_cls.from_range(lo, hi)

    got = conv2d_int8(x, weight, bias, act_quant=aq, stride=stride,
                      padding=padding, groups=groups,
                      activation=activation)
    torch, F = _torch()
    a_dq = torch.from_numpy(_dequant_input(aq, x))
    w_dq = torch.from_numpy(_dequant_weight(weight))
    exp = F.conv2d(a_dq, w_dq, torch.from_numpy(bias),
                   stride=stride, padding=padding, groups=groups)
    if activation == "relu":
        exp = torch.clamp(exp, min=0.0)
    elif activation == "relu6":
        exp = torch.clamp(exp, min=0.0, max=6.0)
    close(f"{act_cls.__name__} groups={groups} stride={stride} "
          f"pad={padding} act={activation}", got, exp.numpy().astype(np.float32),
          atol=2e-5, rtol=1e-4)
    oh = (h + 2 * padding - 3) // stride + 1
    ow = (w + 2 * padding - 3) // stride + 1
    check("输出形状按 OH/OW=(H/W+2p-k)/s+1",
          got.shape == (n, cout, oh, ow), f"got {got.shape}")
    return aq, x, weight, bias, got


def test_conv_variants():
    print("4) 卷积：零填充 / 步长 / 分组 / depthwise")
    for cls in (SymActQuant, AsymActQuant):
        _conv_case(cls, 1, 1, 1, None, 11)
        _conv_case(cls, 1, 2, 1, "relu", 12)
        _conv_case(cls, 2, 1, 1, "relu6", 13)
        _conv_case(cls, 4, 2, 1, None, 14)


def test_padding_is_real_zero():
    print("5) padding 补真实 0（非对称补 zp，不是字面 0）")
    x = np.ones((1, 1, 3, 3), dtype=np.float32)
    weight = np.ones((1, 1, 3, 3), dtype=np.float32)
    bias = np.zeros(1, dtype=np.float32)
    counts = np.array([[4, 6, 4], [6, 9, 6], [4, 6, 4]], dtype=np.float32)
    for cls in (SymActQuant, AsymActQuant):
        aq = cls.from_range(-1.0, 3.0)
        out = conv2d_int8(x, weight, bias, act_quant=aq, padding=1)[0, 0]
        code = int(aq.quantize(np.array([1.0], dtype=np.float32))[0])
        wcode, wscale = quantize_weight_per_channel_sym(weight, 8)
        want = (counts * float(code - aq.zero_point) * float(wcode.reshape(-1)[0])
                * aq.scale * float(wscale[0]))
        close(f"{cls.__name__} 3x3 卷积+p=1：输出正比于有效像素数",
              out, want, atol=1e-6, rtol=1e-6)
        if cls is AsymActQuant:
            check("非对称 zp 非 0，字面补码字 0 必然导致边界偏移",
                  aq.zero_point == 64, f"zp={aq.zero_point}")


# ---------------------------------------------------------------- 4) GEMM

def test_gemm():
    print("6) GEMM：int32 累积 + per-channel 反量化")
    rng = np.random.default_rng(20261023)
    a = (rng.normal(size=(5, 13)).astype(np.float32) * 2.0 + 0.5)
    a[0, 0] = -4.0
    a[0, 1] = 6.0
    weight = rng.normal(size=(7, 13)).astype(np.float32)
    bias = rng.normal(size=7).astype(np.float32)
    torch, F = _torch()
    for cls in (SymActQuant, AsymActQuant):
        aq = cls.from_range(float(a.min()), float(a.max()))
        for bits in (8, 4):
            got, acc = gemm_int8(a, weight, bias, act_quant=aq,
                                 w_bits=bits, return_acc=True)
            a_dq = torch.from_numpy(_dequant_input(aq, a))
            w_dq = torch.from_numpy(_dequant_weight(weight, bits))
            exp = F.linear(a_dq, w_dq, torch.from_numpy(bias))
            close(f"{cls.__name__} W{bits} gemm", got, exp.numpy(),
                  atol=2e-5, rtol=1e-4)
            acodes = aq.quantize(a).astype(np.int32) - aq.zero_point
            wcodes, _ = quantize_weight_per_channel_sym(weight, bits)
            want_acc = acodes @ wcodes.reshape(7, -1).astype(np.int32).T
            check(f"{cls.__name__} W{bits} int32 acc 与公式逐值一致",
                  np.array_equal(acc, want_acc),
                  f"max|diff|={np.max(np.abs(acc - want_acc))}")


# ---------------------------------------------------------------- 5) 后处理

def test_residual_relu6():
    print("7) 残差与 ReLU6 边界")
    x = np.array([-1.0, -1e-6, 0.0, 6.0, 6.0 + 1e-6, 100.0],
                 dtype=np.float32)
    close("ReLU6 边界 [-1,0,6,6+]",
          relu6(x), np.array([0.0, 0.0, 0.0, 6.0, 6.0, 6.0],
                             dtype=np.float32), atol=0, rtol=0)

    main = np.array([1.0, -2.0, 3.0, 6.5], dtype=np.float32)
    ident = np.array([-1.0, 0.5, -4.0, 1.0], dtype=np.float32)
    close("残差 add 后 ReLU（add 前不激活）",
          add_residual(main, ident, activation="relu"),
          np.array([0.0, 0.0, 0.0, 7.5], dtype=np.float32), atol=0, rtol=0)
    close("MobileNetV2 residual=2：add 后不激活",
          add_residual(main, ident),
          np.array([0.0, -1.5, -1.0, 7.5], dtype=np.float32), atol=0, rtol=0)
    try:
        add_residual(np.zeros((2, 2)), np.zeros((3, 3)))
        check("残差形状不一致必须抛错", False)
    except ValueError:
        check("残差形状不一致必须抛错", True)


# ---------------------------------------------------------------- 6) 入口校验

def test_validation():
    print("8) 输入校验")
    x = np.zeros((1, 4, 5, 5), dtype=np.float32)
    w = np.zeros((4, 2, 3, 3), dtype=np.float32)
    aq = SymActQuant.from_range(-1.0, 1.0)
    try:
        conv2d_int8(x, w, act_quant=aq, groups=3)
        check("groups 不整除必须抛错", False)
    except ValueError:
        check("groups 不整除必须抛错", True)
    try:
        conv2d_int8(x, w, act_quant="sym")
        check("act_quant 类型非法必须抛错", False)
    except TypeError:
        check("act_quant 类型非法必须抛错", True)


def main():
    test_activation_modes()
    test_rounding_and_clamp()
    test_weight_quantization()
    test_conv_variants()
    test_padding_is_real_zero()
    test_gemm()
    test_residual_relu6()
    test_validation()
    print()
    if FAILED:
        print(f"FAILED：{len(FAILED)} 项")
        for name in FAILED:
            print(f"  - {name}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
