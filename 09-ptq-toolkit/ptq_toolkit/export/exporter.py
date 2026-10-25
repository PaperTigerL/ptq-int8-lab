# -*- coding: utf-8 -*-
"""导出 BN 折叠后的权重 + 网络拓扑，供 C++ int8 引擎加载。

⭐ 格式与 `06-gpu-int8/cpp/export_weights.py` **完全一致**（magic "PTQW"），
   所以本模块导出的文件能直接喂给 06 的引擎二进制。
   这不是巧合而是刻意：**golden test 要用同一份二进制做对齐**，
   格式各自演化的话，对齐就失去意义。

输出 `model_folded.bin`（小端）：
  magic "PTQW"
  n_conv int32  n_block int32  stem_pool int32
  每个 conv:  cout cin kh kw sh sw ph pw relu   (9 个 int32)
             weight cout*cin*kh*kw fp32（Cout-major）
             bias   cout fp32
  每个 block: conv1_idx conv2_idx downsample_idx stride   (4 个 int32)
               downsample_idx = -1 表示 identity 直连
  fc: fc_in fc_out (2 int32) + weight fc_out*fc_in + bias fc_out

**relu 标志的语义**（错了整网掉到随机猜水平，实测过）：
  stem 的 conv1 与 block 的 conv1  接 ReLU
  block 的 conv2 不接 —— ReLU 在残差 add 之后
  downsample 的 1×1 conv 不接 —— 它在残差那一侧

自检：用同一份拓扑在 PyTorch 侧复算 logits，与折叠后的模型比对，
      max|diff| 必须 < 1e-4。这道自检是必要的：
      拓扑错了单层校验查不出来（06 的教训：单层全绿而全网 7.6%）。
"""
import struct
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAGIC = b"PTQW"


class ConvRecord(NamedTuple):
    """一个 conv 的完整记录（元信息 + 权重 + bias）。"""
    cout: int
    cin: int
    kh: int
    kw: int
    sh: int
    sw: int
    ph: int
    pw: int
    relu: int          # 1=接 ReLU，0=不接
    weight: np.ndarray  # (cout, cin, kh, kw) fp32，Cout-major
    bias: np.ndarray    # (cout,) fp32

    def meta(self) -> Tuple[int, ...]:
        return (self.cout, self.cin, self.kh, self.kw,
                self.sh, self.sw, self.ph, self.pw, self.relu)


class Block(NamedTuple):
    """一个 BasicBlock。"""
    c1: int
    c2: int
    ds: int      # downsample conv 索引，-1 = identity 直连
    stride: int


def _conv_index(model: nn.Module) -> Dict[str, int]:
    """层名 → conv 索引。索引顺序 = named_modules 的定义顺序。"""
    return {name: i for i, (name, m) in enumerate(
        (n, mm) for n, mm in model.named_modules() if isinstance(mm, nn.Conv2d))}


def extract_topology(model: nn.Module,
                    stem_name: str = "conv1",
                    layers: Tuple[str, ...] = ("layer1", "layer2", "layer3", "layer4"),
                    ) -> Tuple[List[ConvRecord], List[Block], int]:
    """从模型抽出 (convs, blocks, stem_pool)。

    relu 标志由「谁在 add 之前」决定：
      stem conv1 / block.conv1 → 1
      block.conv2 / downsample → 0
    """
    idx = _conv_index(model)

    recs: List[ConvRecord] = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            w = m.weight.detach().cpu().contiguous().numpy()
            cout, cin, kh, kw = w.shape
            b = (m.bias.detach().cpu().contiguous().numpy()
                 if m.bias is not None else np.zeros(cout, dtype=np.float32))
            recs.append(ConvRecord(cout, cin, kh, kw,
                                  m.stride[0], m.stride[1],
                                  m.padding[0], m.padding[1],
                                  1, w, b))       # 默认接 ReLU，下面改

    blocks: List[Block] = []
    for lname in layers:
        layer = getattr(model, lname, None)
        if layer is None:
            continue
        for bi, blk in enumerate(layer):
            c1 = idx[f"{lname}.{bi}.conv1"]
            c2 = idx[f"{lname}.{bi}.conv2"]
            ds = idx.get(f"{lname}.{bi}.downsample.0", -1)
            blocks.append(Block(c1, c2, ds, blk.conv1.stride[0]))
            recs[c1] = recs[c1]._replace(relu=1)     # conv1 接 ReLU
            recs[c2] = recs[c2]._replace(relu=0)     # conv2 不接
            if ds >= 0:
                recs[ds] = recs[ds]._replace(relu=0) # downsample 不接

    stem_pool = 1 if stem_name in idx else 0
    return recs, blocks, stem_pool


def replay_topology(model: nn.Module, recs: List[ConvRecord],
                    blocks: List[Block], stem_pool: int,
                    x: torch.Tensor) -> torch.Tensor:
    """按导出的拓扑在 PyTorch 侧复算 logits（拓扑自检用）。"""
    def conv(i: int, t: torch.Tensor) -> torch.Tensor:
        r = recs[i]
        # ⚠️ 权重/bias 是 numpy 造出的 CPU 张量，必须搬到输入所在设备，
        #   否则 --device cuda 下 F.conv2d 报
        #   「Input type (torch.cuda.FloatTensor) and weight type
        #     (torch.FloatTensor) should be the same」。
        y = F.conv2d(t, torch.from_numpy(r.weight).to(t.device),
                     torch.from_numpy(r.bias).to(t.device),
                     (r.sh, r.sw), (r.ph, r.pw))
        return F.relu(y) if r.relu else y

    y = conv(0, x)
    if stem_pool:
        y = F.max_pool2d(y, 3, 2, 1)
    for blk in blocks:
        ident = conv(blk.ds, y) if blk.ds >= 0 else y
        t = conv(blk.c1, y)
        t = conv(blk.c2, t)
        y = F.relu(t + ident)
    pooled = y.mean(dim=(2, 3)).flatten(1)
    return model.fc(pooled)


def ensure_folded(model: nn.Module) -> nn.Module:
    """确保 BN 已折叠 —— **没有折叠就直接报错**。

    ⭐ 为什么必须报错而不是自动折叠：
       导出的是「conv 权重 + conv bias」，BN 的scale/shift 不在格式里。
       若模型里还有 BN，replay 会缺少 BN 而算错，
       但错误会以「自检 max|diff| = 3.9」的形式出现，
       看不出根因是「忘了折叠」。
       所以在入口就拦下来，报错信息直接说「还有 N 个 BN 没折叠」。

    为什么不自动折叠：折叠会改模型权重，而调用方可能还持有原模型
    （pipeline 里 fp32 基线精度就是用折叠后的模型量的）。
    悄悄改掉别人的模型是不可预期的副作用。
    """
    n_bn = sum(1 for _, m in model.named_modules()
               if isinstance(m, nn.BatchNorm2d))
    if n_bn:
        raise RuntimeError(
            f"模型里还有 {n_bn} 个 BatchNorm2d 未折叠。"
            f"导出的格式只含 conv 权重+bias，不含 BN 参数。"
            f"请先调用 03-ptq-lab 的 fold_all_bn(model)（pipeline 里已自动做）。")
    return model


def export_weights(model: nn.Module, path, x: torch.Tensor,
                   strict: bool = True,
                   **topo_kw) -> Dict:
    """导出折叠权重 + 拓扑，带拓扑自检。

    参数
      model  : **BN 已折叠**、eval 模式的模型
      path   : 输出 .bin
      x      : 一个输入 batch，仅用于自检（不参与写入）
      strict : 自检失败时抛异常（默认抛）
    """
    ensure_folded(model)
    recs, blocks, stem_pool = extract_topology(model, **topo_kw)

    # ---- 拓扑自检 ----
    with torch.no_grad():
        ref = model(x).float()
        got = replay_topology(model, recs, blocks, stem_pool, x).float()
    d = float((got - ref).abs().max())
    if strict and d > 1e-4:
        raise RuntimeError(
            f"拓扑自检失败：按导出拓扑复算的 logits 与模型直接前向 "
            f"max|diff| = {d:.4g}（应 < 1e-4）。"
            f"常见原因：relu 标志设错、或 block 的 conv 顺序不对。")

    # ---- 序列化 ----
    blob = bytearray()
    blob += MAGIC
    blob += struct.pack("<3i", len(recs), len(blocks), stem_pool)
    for r in recs:
        blob += struct.pack("<9i", *r.meta())
        blob += np.ascontiguousarray(r.weight, dtype="<f4").tobytes()
        blob += np.ascontiguousarray(r.bias, dtype="<f4").tobytes()
    for blk in blocks:
        blob += struct.pack("<4i", blk.c1, blk.c2, blk.ds, blk.stride)
    fw = model.fc.weight.detach().cpu().contiguous().numpy()
    fb = model.fc.bias.detach().cpu().contiguous().numpy()
    blob += struct.pack("<2i", fw.shape[1], fw.shape[0])
    blob += np.ascontiguousarray(fw, dtype="<f4").tobytes()
    blob += np.ascontiguousarray(fb, dtype="<f4").tobytes()

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(bytes(blob))
    return dict(path=str(p), n_conv=len(recs), n_block=len(blocks),
                stem_pool=stem_pool, bytes=len(blob),
                selfcheck_maxdiff=d)


def write_quant_config(quant, path, extra: Dict | None = None) -> Dict:
    """把量化参数（权重 scale + 激活 range）写成 JSON 文本。

    C++ 引擎需要的两样东西都在这里：
      · 权重 scale（per-channel，Cout 个float）
      · 激活 scale（per-layer，一个 float）
    这样引擎不必自己重新推导量化参数——
    **量化语义只在 Python 这一侧定义一次**，与 06 的做法一致。
    """
    import json
    doc = {
        "w_bits": quant.w_bits,
        "a_bits": quant.a_bits,
        "weights": {name: {"scale": qt.scale.tolist(),
                           "code_dtype": "int8",
                           "zero": qt.zero,
                           "bits": int(qt.bits),
                           "scheme": ("per_group" if getattr(qt, "group_size", 0)
                                      else "per_channel"),
                           "group_size": int(getattr(qt, "group_size", 0)),
                           "scale_shape": list(np.shape(qt.scale))}
                    for name, qt in quant.weights.items()},
        "acts": {name: {"scale": s.scale, "zero": s.zero, "qmax": s.qmax}
                 for name, s in quant.acts.items()},
    }
    if extra:
        doc.update(extra)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return dict(path=str(p), n_weights=len(doc["weights"]),
                n_acts=len(doc["acts"]))
