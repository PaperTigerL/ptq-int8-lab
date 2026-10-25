# -*- coding: utf-8 -*-
"""引擎侧 PTQC 校准文件（CPU-only 导出/读回）。

格式严格对齐 06 的 `load_calib`（06-gpu-int8/cpp/int8_engine.cuh:683-700）：

    magic   4 字节  "PTQC"
    n       int32   小端
    n 对    float32 (lo, hi) 小端

层数 / 顺序约束（06 强制，非本模块臆造）：
  · `Int8Engine::set_calib` 要求 `ranges.size() == layers_.size()`，
    否则直接 `std::exit(1)`（int8_engine.cuh:193-197）。
  · `layers_` 只由 conv 构成：`layers_.resize(m.convs.size())`、
    `aq_.resize(m.convs.size())`（int8_engine.cuh:145 / 184），`m.convs`
    来自 PTQW 里 `n_conv` 个 ConvSpec（int8_engine.cuh:92-104），**不含 fc**
    （fc 权重单独读进 `fc_w`/`fc_b`，见 int8_engine.cuh:121-126 / 176-182）。
  · 因此 PTQC 的层数 = 模型 Conv2d 数，且顺序必须与 09 导出权重的 conv
    遍历顺序一致 —— 即 `named_modules()` 过滤 `nn.Conv2d` 的定义顺序
    （见 exporter.py:69-70 `_conv_index`、exporter.py:86-87；
    topoblocks.py:101-102 同口径）。**不要按模型语义重排**。

与引擎 zp 口径的差异：
  · 06 引擎 `tensor_quant_from_range` 把激活 zero-point 强制钉在 0
    （对称码空间，见 ptq_quant.h:175-189），scale = (hi-lo)/127。
  · 09 pipeline 的 `calibrate(..., symmetric=True)` 用的是
    scale = max(|lo|,|hi|)/127，区间被钉在 [-t,t]。
  · 因此 PTQC 必须写**原始 minmax lo/hi**（与 06 export_calib.py 同口径），
    不能写 09 的对称 scale 反推区间 —— 否则一侧负半轴被重复压缩。
    两边在「ReLU 输出、lo≈0」时数值一致；lo 明显为负时 scale 会不同，
    这是跨工具口径差，报告里必须写明（见 posts/ptq/20261022-*）。
  · 本模块不预设对称/非对称：range→scale/zp 的推导留给引擎。

层顺序（引擎强制）：
  conv 的遍历顺序 = `model.named_modules()` 过滤 `nn.Conv2d` 的**定义顺序**，
  与 exporter.py `_conv_index`（exporter.py:67-70）和 topoblocks.py
  `conv_index`（topoblocks.py:85-87）完全一致。**不要按语义重排**：
  downsample conv 就停在其定义位置，而不是挪到 block 前。

本模块格式层只依赖标准库（struct / math），可在无 torch 环境跑格式校验；
`conv_names` / `write_ptqc_for_model` 仅在调用时才 import torch。
"""
import math
import struct
from pathlib import Path
from typing import List, Sequence, Tuple

MAGIC = b"PTQC"
_INT32_MAX = 2 ** 31 - 1


def write_ptqc(path, ranges: Sequence[Sequence[float]],
               n: int | None = None) -> dict:
    """写 PTQC。`ranges` 是 (lo, hi) 序列；可选 `n` 显式指定层数并做一致性校验。

    校验：0 <= n <= int32 上界、n == len(ranges)、每对 lo/hi 有限且 lo <= hi。
    返回 dict(path, n, bytes)。
    """
    if n is None:
        n = len(ranges)
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError(f"n 必须是 int，收到 {type(n).__name__}")
    if n < 0:
        raise ValueError(f"n 不能为负：{n}")
    if n > _INT32_MAX:
        raise ValueError(f"n 超出 int32 上界：{n}")
    if n != len(ranges):
        raise ValueError(f"n({n}) != len(ranges)({len(ranges)})")

    blob = bytearray(MAGIC)
    blob += struct.pack("<i", n)
    for i, pair in enumerate(ranges):
        if len(pair) != 2:
            raise ValueError(f"ranges[{i}] 应为 (lo, hi) 两元，收到 {len(pair)} 元")
        lo, hi = float(pair[0]), float(pair[1])
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise ValueError(f"ranges[{i}] 非有限值：lo={lo!r} hi={hi!r}")
        if lo > hi:
            raise ValueError(f"ranges[{i}] lo>hi：lo={lo} hi={hi}")
        blob += struct.pack("<2f", lo, hi)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(bytes(blob))
    return dict(path=str(p), n=n, bytes=len(blob))


def read_ptqc(path) -> List[Tuple[float, float]]:
    """读 PTQC，返回 [(lo, hi), ...]。校验 magic、n>=0、文件大小 == 8+8n。

    作为 write 的逆运算与文件自检用；语义与 06 的 `load_calib` 一致，
    但对越界/损坏更严格（直接抛异常而非段错误）。
    """
    data = Path(path).read_bytes()
    if len(data) < 4 or data[:4] != MAGIC:
        raise ValueError(f"{path}: magic 不符（应为 PTQC）")
    if len(data) < 8:
        raise ValueError(f"{path}: 头不完整（不足 8 字节）")
    n = struct.unpack_from("<i", data, 4)[0]
    if n < 0:
        raise ValueError(f"{path}: 头里 n 为负：{n}")
    expected = 8 + 8 * n
    if len(data) != expected:
        raise ValueError(
            f"{path}: 文件大小 {len(data)} != 8 + 8*{n} = {expected}")
    out: List[Tuple[float, float]] = []
    off = 8
    for _ in range(n):
        lo, hi = struct.unpack_from("<2f", data, off)
        off += 8
        out.append((lo, hi))
    return out


def weight_conv_count(path) -> Tuple[str, int]:
    """只读权重文件头，返回 (magic, n_conv)。

    支持 06 的 PTQW 与 09 的 PTQ2 —— 两者 magic 后第一个 int32 都是 n_conv
    （exporter.py:193-195 / topoblocks.py:247-250），所以 8 字节足够。
    用于在导出 PTQC 前校验层数与引擎要读的权重严格一致。
    """
    with open(path, "rb") as f:
        head = f.read(8)
    if len(head) < 8:
        raise ValueError(f"{path}: 文件过短（{len(head)} B < 8 B）")
    magic = head[:4]
    if magic not in (b"PTQW", b"PTQ2"):
        raise ValueError(f"{path}: magic 不符（实得 {magic!r}）")
    n = struct.unpack_from("<i", head, 4)[0]
    if n < 0:
        raise ValueError(f"{path}: 头里 n_conv 为负：{n}")
    return magic.decode("ascii"), n


def conv_names(model) -> List[str]:
    """按导出顺序列出 Conv2d 名字（named_modules 定义顺序，非语义重排）。"""
    import torch.nn as nn

    return [name for name, m in model.named_modules()
            if isinstance(m, nn.Conv2d)]


def ranges_from_store(store, names: Sequence[str]) -> List[Tuple[float, float]]:
    """按 `names` 顺序从激活 store 取 (min, max)。

    store 的 value 可以是 torch tensor / numpy array（需支持 .min()/.max()）。
    缺层 / 多余层都报错：静默跳过会让 PTQC 少一层，引擎 set_calib 直接
    std::exit(1)，错误信息只有「层数不匹配」，追不到是哪个模块漏了。
    """
    names = list(names)
    name_set = set(names)
    missing = [n for n in names if n not in store]
    if missing:
        raise ValueError(
            f"激活 store 缺 {len(missing)} 层：{missing[:6]}"
            f"{' ...' if len(missing) > 6 else ''}")
    extra = [n for n in store if n not in name_set]
    if extra:
        raise ValueError(
            f"激活 store 多出 {len(extra)} 个非 conv 键：{extra[:6]}"
            f"{' ...' if len(extra) > 6 else ''}")

    out: List[Tuple[float, float]] = []
    for i, name in enumerate(names):
        x = store[name]
        lo, hi = float(x.min()), float(x.max())
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise ValueError(
                f"第 {i} 层 {name!r} 的 range 非有限：lo={lo!r} hi={hi!r}")
        if lo > hi:
            raise ValueError(f"第 {i} 层 {name!r} 的 lo>hi：{lo} > {hi}")
        out.append((lo, hi))
    return out


def write_ptqc_for_model(path, model, store,
                         n_expected: int | None = None) -> dict:
    """按模型 conv 定义顺序，把激活 store 的原始 minmax 写成 PTQC。

    `n_expected` 传引擎侧权重的 conv 数（PTQW/PTQ2 头里的 n_conv），
    不一致直接报错 —— 这是 06 `set_calib` 的硬约束（int8_engine.cuh:193-197）。
    """
    names = conv_names(model)
    ranges = ranges_from_store(store, names)
    if n_expected is not None and len(names) != int(n_expected):
        raise ValueError(
            f"PTQC 层数 {len(names)} != 权重 conv 数 {n_expected}；"
            f"引擎 set_calib 会拒绝。检查导出模型与校准模型是否同一份。")
    info = write_ptqc(path, ranges)
    info.update(n_conv=len(names),
                first_layer=names[0] if names else None,
                last_layer=names[-1] if names else None)
    return info
