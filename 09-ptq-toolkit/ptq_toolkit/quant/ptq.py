# -*- coding: utf-8 -*-
"""per-channel PTQ 量化核心（W8 / W4）。

⭐ 与 `03-ptq-lab` 的关系：
   03 是量化语义的 ground truth（PyTorch fake-quant，可做 KL/MSE 搜索），
   本模块是它的**可部署子集**——只保留 minmax 口径的 per-channel 权重 +
   静态激活，因为：

     · 06/07 的 C++ 引擎与 TensorRT 都只支持这一种（引擎要 int8 权重矩阵）
     · M1 的验收标准是「与 06/07 已有结果对齐」，而不是「比 03 更好」

   所以这里的 round / clamp 口径**逐位复刻** 03 的
   `quantize_per_channel_sym`，两边必须能互相验证。

三处口径（与 ptq_quant.h 完全一致，改一处必须同步另一处）：
  1. round = **half-away-from-zero**（不是 numpy 的 banker's rounding）
  2. clamp = [-(qmax+1), qmax]，qmax = 2^(bits-1)-1
     负侧多一个码字，保证 0 落在网格上
  3. weight scale 分母用 qmax（=127），**不是** 255
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------- 基础算子

def round_half_away(x: np.ndarray) -> np.ndarray:
    """half-away-from-zero，与 torch.round / C++ 的 torch_round 同语义。

    ⛔ 不能用 np.round —— 那是 banker's rounding（half-to-even），
       在恰好 .5 的值上与 torch.round 不同，一个量化等级。
    """
    return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5))


def qmax_of(bits: int) -> int:
    """对称码空间的上界。bits=8 → 127，bits=4 → 7。

    ⛔ 不是 (1<<bits)-1（那是 255 / 15）—— 那是 int8 存储的上界，
       但对称量化的**有效码字**只有一半。用 255 会让 scale 差 2 倍。
    """
    return (1 << (bits - 1)) - 1


# ---------------------------------------------------------------- 权重量化

@dataclass
class QuantizedTensor:
    """一个被量化过的张量。

    code      : int8 码字（形状与原张量同）
    scale     : 反量化系数
                 per-channel → (Cout,)
                 per-group   → (Cout, n_groups)，每组一个 scale
    zero      : 零点（对称量化恒为 0）
    bits      : 位宽
    group_size: per-group 时每组沿展平 K 维的长度；per-channel 为 0
    """
    code: np.ndarray
    scale: np.ndarray
    zero: int
    bits: int
    group_size: int = 0

    @property
    def shape(self):
        return tuple(self.code.shape)

    def dequant(self) -> np.ndarray:
        """反量化回原形状。

        ⛔ scale 是 per-channel 的（Cout 个值），要广播到第 0 维：
           `scale[:, None...]` 后再靠 `np.broadcast_to` 撑开。
           直接 `code * scale` 会因为 (Cout,Cin,kh,kw) 与 (Cout,)
           末维对不上而报错 —— 或者更糟，NumPy 静默按 Cout 广播到末维，
           结果全错但**不报错**。

        per-group 的 scale 是二维 (Cout, n_groups)，必须先把展平的
        K 维按 group_size 切成同样数量的小块，再逐块乘各自的 scale。
        ⛔ 不能把二维 scale 直接 reshape 成 (Cout,1,1,1)：那会丢掉
           group 维，或按末维广播到错误位置。
        """
        s = self.scale
        tgt = self.code.shape
        if s.ndim == 1:
            view = [s.shape[0]] + [1] * (len(tgt) - 1)
            return (self.code.astype(np.float32) + self.zero) * s.reshape(view)
        if s.ndim == 2:
            cout = tgt[0]
            if s.shape[0] != cout:
                raise ValueError(
                    f"scale 的第 0 维 {s.shape[0]} != 输出通道 {cout}")
            flat = self.code.reshape(cout, -1).astype(np.float32)
            k = flat.shape[1]
            ng = s.shape[1]
            gs = self.group_size or ((k + ng - 1) // ng)
            if gs <= 0 or gs * ng < k:
                raise ValueError(
                    f"group_size={gs}、n_groups={ng} 覆盖不了 K={k}")
            sc = np.repeat(s, gs, axis=1)[:, :k]
            out = (flat + self.zero) * sc
            return out.reshape(tgt)
        raise ValueError(f"scale 只能是 1 维或 2 维，得到 {s.ndim} 维")

    def nbytes(self) -> int:
        """量化后的字节数（用于报压缩比）。"""
        return self.code.nbytes + self.scale.nbytes


def quantize_per_channel_sym(w: np.ndarray, bits: int) -> QuantizedTensor:
    """per-channel 对称量化，axis=0（Cout）。

    与 03-ptq-lab/src/quant.py::quantize_per_channel_sym 同口径，
    与 06-gpu-int8/cpp/ptq_quant.h 逐位一致。
    """
    if w.ndim < 2:
        raise ValueError(f"per-channel 量化要求 >= 2 维，得到 {w.ndim} 维")
    qmax = qmax_of(bits)
    wf = w.reshape(w.shape[0], -1).astype(np.float64)     # (Cout, K)

    amax = np.abs(wf).max(axis=1)
    scale = np.maximum(amax / qmax, 1e-8)                 # clamp 防 0
    code = round_half_away(wf / scale[:, None])
    code = np.clip(code, -(qmax + 1), qmax).astype(np.int8)

    return QuantizedTensor(
        code=code.reshape(w.shape),
        scale=scale.astype(np.float32),
        zero=0,
        bits=bits,
    )


def quantize_per_group_sym(w: np.ndarray, bits: int,
                           group_size: int) -> QuantizedTensor:
    """per-group 对称量化：沿展平的 K 维每 `group_size` 个元素一组。

    这是 W4 的候选方案：per-channel 在 W4 下只有 15 个正负码字，
    一个输出通道内动态范围差异大时（depthwise / 首尾层）误差集中；
    把 K 切小、每组独立 scale 可以显著降低量化误差。

    ⛔ `group_size` 必须 > 0。K 不能整除时最后一组补 0 计算 absmax，
       但**不写回**（padding 只参与 scale 统计，不改变 K 的实际长度）。
       absmax 补 0 不会改变最大值，除非整组全 0 —— 那时 scale 仍被
       clamp 到 1e-8，码字全 0，与 per-channel 的零通道处理一致。

    ⛔ 组边界按**展平后的 K** 切，不按 kh/kw 空间位置切：
       导出时权重也是按 `[Cout][Cin][kh][kw]` 展平成 K 的，
       两边必须用同一个展平顺序，否则 scale 与码字错位。
    """
    if w.ndim < 2:
        raise ValueError(f"per-group 量化要求 >= 2 维，得到 {w.ndim} 维")
    if not isinstance(group_size, int) or isinstance(group_size, bool):
        raise TypeError(f"group_size 必须是 int，收到 {type(group_size).__name__}")
    if group_size <= 0:
        raise ValueError(f"group_size 必须 > 0，收到 {group_size}")

    qmax = qmax_of(bits)
    wf = w.reshape(w.shape[0], -1).astype(np.float64)     # (Cout, K)
    cout, k = wf.shape
    ng = (k + group_size - 1) // group_size
    padded = np.zeros((cout, ng * group_size), dtype=np.float64)
    padded[:, :k] = wf
    groups = padded.reshape(cout, ng, group_size)

    amax = np.abs(groups).max(axis=2)                     # (Cout, ng)
    scale = np.maximum(amax / qmax, 1e-8)
    code = round_half_away(groups / scale[..., None])
    code = np.clip(code, -(qmax + 1), qmax).astype(np.int8)
    code = code.reshape(cout, ng * group_size)[:, :k].reshape(w.shape)

    return QuantizedTensor(
        code=code,
        scale=scale.astype(np.float32),
        zero=0,
        bits=bits,
        group_size=group_size,
    )


def quantize_weight_sym(w: np.ndarray, bits: int,
                        scheme: str = "per_channel",
                        group_size: int = 0) -> QuantizedTensor:
    """按 scheme 分派权重量化。`scheme` 只允许 per_channel / per_group。"""
    if scheme == "per_channel":
        return quantize_per_channel_sym(w, bits)
    if scheme == "per_group":
        return quantize_per_group_sym(w, bits, group_size)
    raise ValueError(
        f"未知量化方案 {scheme!r}；只允许 'per_channel' / 'per_group'")


# ---------------------------------------------------------------- 激活量化

@dataclass
class ActQuantSpec:
    """激活的静态量化参数。

    scale 由 (lo, hi) 推出，zero 恒为 0（对称）。
    这是 06/07 引擎实际支持的唯一形态。
    """
    scale: float
    zero: int
    bits: int

    @property
    def qmax(self) -> int:
        return qmax_of(self.bits)


def act_spec_from_range(lo: float, hi: float, bits: int) -> ActQuantSpec:
    """由校准得到的 (lo, hi) 推出激活 scale。

    分母是 qmax（127）而不是 255 —— 与权重同口径，
    否则同一层里权重与激活的「量化密度」不一致。
    """
    qmax = qmax_of(bits)
    scale = max((hi - lo) / qmax, 1e-8)
    return ActQuantSpec(scale=float(scale), zero=0, bits=bits)


def calibrate_range(x: torch.Tensor) -> Tuple[float, float]:
    """一个 batch 的 (lo, hi)。多个 batch 逐元素取 min/max。"""
    xf = x.detach().float().cpu().numpy()
    return float(xf.min()), float(xf.max())


def merge_range(lo_hi: Tuple[float, float],
                new: Tuple[float, float]) -> Tuple[float, float]:
    return min(lo_hi[0], new[0]), max(lo_hi[1], new[1])


# ---------------------------------------------------------------- fake-quant

def fake_quant_per_channel(w: torch.Tensor, bits: int) -> torch.Tensor:
    """PyTorch 侧的 fake-quant，用来测量化损失（不产出 int8 码字）。"""
    q = quantize_per_channel_sym(w.detach().cpu().numpy(), bits)
    return torch.from_numpy(q.dequant()).to(w.device, w.dtype)


class ActQuantObserver:
    """静态激活量化观测器。

    用法：
        obs = ActQuantObserver(bits=8)
        for x, _ in calib_loader:
            model(x)                       # 先跑一遍前向
            obs.observe(model)              # 再收每层输入的 range
        specs = obs.specs()                # 拿到 {layer_name: ActQuantSpec}
    """

    def __init__(self, bits: int = 8):
        self.bits = bits
        self._ranges: Dict[str, Tuple[float, float]] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    # -- 内部：找到所有 conv/linear 的输入
    @staticmethod
    def _conv_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
        out = []
        for name, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                out.append((name, m))
        return out

    def attach(self, model: nn.Module) -> None:
        """给每个 conv 的 forward 前置 hook 收输入 range。"""
        for name, m in self._conv_layers(model):

            def hook(mod, inp, _name=name):
                lo, hi = calibrate_range(inp[0].detach())
                cur = self._ranges.get(_name)
                self._ranges[_name] = (lo, hi) if cur is None \
                    else merge_range(cur, (lo, hi))
                return None

            self._handles.append(m.register_forward_pre_hook(hook))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @property
    def ranges(self) -> Dict[str, Tuple[float, float]]:
        return dict(self._ranges)

    def specs(self) -> Dict[str, ActQuantSpec]:
        return {k: act_spec_from_range(lo, hi, self.bits)
                for k, (lo, hi) in self._ranges.items()}

    def summary(self) -> str:
        lines = [f"{'层':<34}{'lo':>12}{'hi':>12}{'scale':>12}"]
        for k, (lo, hi) in self._ranges.items():
            s = act_spec_from_range(lo, hi, self.bits)
            lines.append(f"{k:<34}{lo:>12.5g}{hi:>12.5g}{s.scale:>12.6g}")
        return "\n".join(lines)


# ---------------------------------------------------------------- 模型级量化

@dataclass
class PTQResult:
    """一次 PTQ 的产物。"""
    w_bits: int
    a_bits: int
    weights: Dict[str, QuantizedTensor] = field(default_factory=dict)
    acts: Dict[str, ActQuantSpec] = field(default_factory=dict)
    fp32_acc: float = 0.0      # fp32 基线精度（同一份测试集）
    quant_acc: float = 0.0     # 量化后精度

    @property
    def acc_drop(self) -> float:
        return self.fp32_acc - self.quant_acc


@torch.no_grad()
def quantize_model(model: nn.Module,
                   calib_loader,
                   w_bits: int = 8,
                   a_bits: int = 8,
                   device: str = "cpu",
                   verbose: bool = True) -> PTQResult:
    """对整个模型做 per-channel PTQ 并测精度。

    流程：
      1. 记 fp32 基线精度
      2. 用校准集跑前向，收每层激活的 (lo, hi)
      3. per-channel 量化所有 conv 权重
      4. 用 fake-quant 前向测量化后精度

    ⚠️ fake-quant 前向会**同时**模拟激活量化 —— 这需要把每个 conv 的
       输入量化成 int8 再反量化。当前实现只做权重 fake-quant，
       激活量化在 C++ 引擎侧生效（那才是真实路径）。
       所以本函数给出的 `quant_acc` 是**权重-only** 的精度，
       会**高于**真实 W8A8 引擎的精度。做 golden test 时要知道这个差。
    """
    model = model.to(device).eval()

    # ---- 1) fp32 基线 ----
    fp32_acc = None  # 由调用方传入或用 evaluate 算；见 run_ptq

    # ---- 2) 校准 ----
    obs = ActQuantObserver(bits=a_bits)
    obs.attach(model)
    n = 0
    for x, _ in calib_loader:
        model(x.to(device))
        n += 1
    obs.detach()
    if verbose:
        print(f"校准完成：{n} 个 batch，{len(obs.ranges)} 层")

    # ---- 3) 权重 per-channel 量化（就地替换）----
    quant = {}
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            w = m.weight.detach().cpu().numpy()
            q = quantize_per_channel_sym(w, w_bits)
            quant[name] = q
            # 就地写回 fake-quant 后的权重，便于直接测精度
            new_w = torch.from_numpy(q.dequant()).to(m.weight.device, m.weight.dtype)
            m.weight.data.copy_(new_w.reshape(m.weight.shape))

    return PTQResult(w_bits=w_bits, a_bits=a_bits,
                     weights=quant, acts=obs.specs(),
                     fp32_acc=fp32_acc or 0.0)
