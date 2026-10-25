# -*- coding: utf-8 -*-
"""int8 conv / gemm 的 CPU 参考实现（NumPy）。

口径见 docs/int8_conv_gemm_spec.md，与 06-gpu-int8 的引擎路径逐条对应：

  · 权重：per-channel 对称量化，qmax = 2^(bits-1)-1（W8=127 / W4=7），
    码字 clamp 到 [-(qmax+1), qmax]，zero_point 恒为 0。
  · 激活（引擎路径）：per-tensor 对称 AbsMax，
    scale = max(|lo|, |hi|) / qmax，zero_point 恒为 0。
  · 累积：int32；反量化：acc * (act_scale * weight_scale[cout]) + bias。
  · padding：补**真实 0**。对称路径的零码字就是 0；非对称路径必须补
    zero_point，否则反量化后是 -zp*scale 而不是 0。
  · 后处理：ReLU / ReLU6 在反量化后的 fp32 域做；残差在 fp32 域相加，
    ReLU 必须等 add 之后再做。

非对称激活（AsymActQuant）只用于把「zp 口径」做成可执行、可断言的参考，
06 引擎不执行它：int8 GEMM 不带 zero-point 修正，引擎统一走对称码空间。
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

from .quant.ptq import qmax_of, round_half_away

ACTIVATIONS = (None, "none", "relu", "relu6")


def _pair(v: Union[int, Tuple[int, int]], name: str) -> Tuple[int, int]:
    if isinstance(v, int):
        out = (v, v)
    elif isinstance(v, (tuple, list)) and len(v) == 2:
        out = (int(v[0]), int(v[1]))
    else:
        raise TypeError(f"{name} 必须是 int 或 (int, int)，收到 {v!r}")
    if out[0] < 0 or out[1] < 0:
        raise ValueError(f"{name} 不能为负：{out}")
    return out


def _check_bits(bits: int) -> None:
    if not isinstance(bits, int) or isinstance(bits, bool) or not 2 <= bits <= 8:
        raise ValueError(f"bits 必须是 2..8 的 int，收到 {bits!r}")


@dataclass(frozen=True)
class SymActQuant:
    """对称激活量化（引擎实际路径）：zp=0，码空间 [-(qmax+1), qmax]。"""

    scale: float
    bits: int = 8
    zero_point: int = 0

    def __post_init__(self):
        _check_bits(self.bits)
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError(f"scale 必须是有限正数，收到 {self.scale!r}")
        if self.zero_point != 0:
            raise ValueError("对称量化的 zero_point 必须为 0")

    @property
    def qmax(self) -> int:
        return qmax_of(self.bits)

    @property
    def qmin(self) -> int:
        return -(self.qmax + 1)

    @classmethod
    def from_range(cls, lo: float, hi: float, bits: int = 8,
                   mode: str = "absmax") -> "SymActQuant":
        """由 (lo, hi) 推 scale。

        mode="absmax" 是引擎默认：max(|lo|,|hi|)/qmax，跨零时也覆盖两端。
        mode="minmax" 仅作对照：(hi-lo)/qmax，分辨率更粗且与 09 校准不一致。
        """
        _check_bits(bits)
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo > hi:
            raise ValueError(f"非法范围 lo={lo}, hi={hi}")
        qmax = qmax_of(bits)
        if mode == "absmax":
            s = max(abs(float(lo)), abs(float(hi))) / qmax
        elif mode == "minmax":
            s = (float(hi) - float(lo)) / qmax
        else:
            raise ValueError(f"未知 mode={mode!r}；只允许 absmax / minmax")
        return cls(scale=max(s, 1e-8), bits=bits)

    def quantize(self, x: np.ndarray) -> np.ndarray:
        q = round_half_away(np.asarray(x, dtype=np.float64) / self.scale)
        return np.clip(q, self.qmin, self.qmax).astype(np.int8)

    def dequantize(self, q: np.ndarray) -> np.ndarray:
        return (np.asarray(q, dtype=np.float32) - self.zero_point) \
            * np.float32(self.scale)


@dataclass(frozen=True)
class AsymActQuant:
    """非对称激活量化（参考语义，引擎不支持）：码空间 [0, 2^bits-1]。"""

    scale: float
    zero_point: int
    bits: int = 8

    def __post_init__(self):
        _check_bits(self.bits)
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError(f"scale 必须是有限正数，收到 {self.scale!r}")
        if not 0 <= self.zero_point <= self.qmax:
            raise ValueError(
                f"zero_point={self.zero_point} 超出 [0, {self.qmax}]")

    @property
    def qmax(self) -> int:
        return (1 << self.bits) - 1

    @property
    def qmin(self) -> int:
        return 0

    @classmethod
    def from_range(cls, lo: float, hi: float,
                   bits: int = 8) -> "AsymActQuant":
        _check_bits(bits)
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo > hi:
            raise ValueError(f"非法范围 lo={lo}, hi={hi}")
        qmax = (1 << bits) - 1
        scale = max((float(hi) - float(lo)) / qmax, 1e-8)
        zp = int(np.clip(round_half_away(-float(lo) / scale), 0, qmax))
        return cls(scale=scale, zero_point=zp, bits=bits)

    def quantize(self, x: np.ndarray) -> np.ndarray:
        q = round_half_away(np.asarray(x, dtype=np.float64) / self.scale) \
            + self.zero_point
        return np.clip(q, self.qmin, self.qmax).astype(np.uint8)

    def dequantize(self, q: np.ndarray) -> np.ndarray:
        return (np.asarray(q, dtype=np.float32) - self.zero_point) \
            * np.float32(self.scale)


ActQuant = Union[SymActQuant, AsymActQuant]


def quantize_weight_per_channel_sym(
        weight: np.ndarray, bits: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """per-channel（axis=0）对称量化。返回 (int8 码字, float32 scale)。"""
    _check_bits(bits)
    w = np.asarray(weight, dtype=np.float64)
    if w.ndim < 2:
        raise ValueError(f"权重要求 >= 2 维，收到 {w.ndim} 维")
    qmax = qmax_of(bits)
    flat = w.reshape(w.shape[0], -1)
    amax = np.abs(flat).max(axis=1)
    scale = np.maximum(amax / qmax, 1e-8)
    code = round_half_away(flat / scale[:, None])
    code = np.clip(code, -(qmax + 1), qmax).astype(np.int8)
    return code.reshape(w.shape), scale.astype(np.float32)


def _apply_activation(y: np.ndarray, activation: Optional[str]) -> np.ndarray:
    if activation not in ACTIVATIONS:
        raise ValueError(f"未知 activation={activation!r}；允许 {ACTIVATIONS}")
    if activation in (None, "none"):
        return y
    if activation == "relu":
        return np.maximum(y, np.float32(0.0))
    return np.clip(y, np.float32(0.0), np.float32(6.0))


def conv2d_int8(
        x: np.ndarray,
        weight: np.ndarray,
        bias: Optional[np.ndarray] = None,
        *,
        act_quant: ActQuant,
        w_bits: int = 8,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        groups: int = 1,
        activation: Optional[str] = None,
        return_acc: bool = False,
):
    """int8 卷积参考实现，返回反量化后的 fp32 NCHW 输出（可选 int32 acc）。

    weight 形状 [Cout, Cin/groups, kh, kw]；bias 形状 [Cout]。
    padding 补真实 0：对称路径补码字 0，非对称路径补 zero_point。
    """
    if not isinstance(act_quant, (SymActQuant, AsymActQuant)):
        raise TypeError(f"act_quant 类型非法：{type(act_quant).__name__}")
    x = np.asarray(x, dtype=np.float32)
    w = np.asarray(weight, dtype=np.float32)
    if x.ndim != 4 or w.ndim != 4:
        raise ValueError(f"x/weight 必须是 4 维，收到 {x.ndim}/{w.ndim}")
    N, C, H, W = x.shape
    Cout, Cin_g, kh, kw = w.shape
    if groups < 1 or C % groups or Cout % groups:
        raise ValueError(
            f"groups={groups} 必须整除 Cin={C} 与 Cout={Cout}")
    if Cin_g != C // groups:
        raise ValueError(
            f"weight 第二维 {Cin_g} != Cin/groups={C // groups}")
    sh, sw = _pair(stride, "stride")
    ph, pw = _pair(padding, "padding")
    if sh == 0 or sw == 0:
        raise ValueError("stride 不能为 0")
    OH = (H + 2 * ph - kh) // sh + 1
    OW = (W + 2 * pw - kw) // sw + 1
    if OH <= 0 or OW <= 0:
        raise ValueError(f"输出尺寸非法：OH={OH}, OW={OW}")
    if bias is not None:
        bias = np.asarray(bias, dtype=np.float32).reshape(-1)
        if bias.shape[0] != Cout:
            raise ValueError(f"bias 长度 {bias.shape[0]} != Cout={Cout}")

    a_codes = act_quant.quantize(x)
    pad_code = np.asarray(act_quant.zero_point, dtype=a_codes.dtype)
    ap = np.full((N, C, H + 2 * ph, W + 2 * pw), pad_code,
                 dtype=a_codes.dtype)
    ap[:, :, ph:ph + H, pw:pw + W] = a_codes
    a_shift = ap.astype(np.int32) - int(act_quant.zero_point)

    w_codes, w_scales = quantize_weight_per_channel_sym(w, w_bits)
    w_flat = w_codes.reshape(Cout, -1).astype(np.int32)
    cin_pg = C // groups
    cout_pg = Cout // groups
    acc = np.zeros((N, Cout, OH, OW), dtype=np.int32)
    for g in range(groups):
        c0 = g * cin_pg
        o0 = g * cout_pg
        wg = w_flat[o0:o0 + cout_pg]
        for oh in range(OH):
            ih = oh * sh
            for ow in range(OW):
                iw = ow * sw
                patch = a_shift[:, c0:c0 + cin_pg, ih:ih + kh, iw:iw + kw]
                acc[:, o0:o0 + cout_pg, oh, ow] = \
                    patch.reshape(N, -1) @ wg.T

    factor = (np.float32(act_quant.scale)
              * w_scales.astype(np.float32))[None, :, None, None]
    y = acc.astype(np.float32) * factor
    if bias is not None:
        y = y + bias[None, :, None, None]
    y = _apply_activation(y, activation)
    if return_acc:
        return y, acc
    return y


def gemm_int8(
        a: np.ndarray,
        weight: np.ndarray,
        bias: Optional[np.ndarray] = None,
        *,
        act_quant: ActQuant,
        w_bits: int = 8,
        activation: Optional[str] = None,
        return_acc: bool = False,
):
    """int8 GEMM：A[M,K] × W[N,K]^T → fp32[M,N]（可选 int32 acc）。"""
    if not isinstance(act_quant, (SymActQuant, AsymActQuant)):
        raise TypeError(f"act_quant 类型非法：{type(act_quant).__name__}")
    a = np.asarray(a, dtype=np.float32)
    w = np.asarray(weight, dtype=np.float32)
    if a.ndim != 2 or w.ndim != 2:
        raise ValueError(f"a/weight 必须是 2 维，收到 {a.ndim}/{w.ndim}")
    if a.shape[1] != w.shape[1]:
        raise ValueError(f"K 不一致：a {a.shape} vs weight {w.shape}")
    if bias is not None:
        bias = np.asarray(bias, dtype=np.float32).reshape(-1)
        if bias.shape[0] != w.shape[0]:
            raise ValueError(f"bias 长度 {bias.shape[0]} != N={w.shape[0]}")

    a_codes = act_quant.quantize(a).astype(np.int32)
    a_shift = a_codes - int(act_quant.zero_point)
    w_codes, w_scales = quantize_weight_per_channel_sym(w, w_bits)
    acc = a_shift @ w_codes.reshape(w.shape[0], -1).astype(np.int32).T

    factor = np.float32(act_quant.scale) * w_scales.astype(np.float32)
    y = acc.astype(np.float32) * factor[None, :]
    if bias is not None:
        y = y + bias[None, :]
    y = _apply_activation(y, activation)
    if return_acc:
        return y, acc
    return y


def add_residual(x: np.ndarray, identity: np.ndarray,
                 activation: Optional[str] = None) -> np.ndarray:
    """残差相加（fp32 域），activation 在 add 之后做。

    对应引擎：block 末 conv 不激活 → add → ReLU（ResNet）；
    MobileNetV2 的 residual=2 在 add 后不激活。
    """
    x = np.asarray(x, dtype=np.float32)
    identity = np.asarray(identity, dtype=np.float32)
    if x.shape != identity.shape:
        raise ValueError(f"残差形状不一致：{x.shape} vs {identity.shape}")
    return _apply_activation(x + identity, activation)


def relu(x: np.ndarray) -> np.ndarray:
    return _apply_activation(np.asarray(x, dtype=np.float32), "relu")


def relu6(x: np.ndarray) -> np.ndarray:
    return _apply_activation(np.asarray(x, dtype=np.float32), "relu6")
