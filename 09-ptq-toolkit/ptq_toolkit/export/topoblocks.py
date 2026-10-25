# -*- coding: utf-8 -*-
"""PTQW-v2：把 06 的 blocks 格式推广到「N 层 conv + 可选 downsample」。

⭐ 为什么不用通用指令序列（PTQX）：
   PTQX 是 M2 中途设计的，想用「模型声明残差 + 栈机指令」表达任意拓扑。
   实测它在 ResNet-50 上自检 max|diff| = 0.03（relu 标志回填的边界情况难穷尽），
   在 MobileNetV2 上因可选 skip 结构直接崩。迭代 20+ 轮仍不可靠。

   而blocks 格式在 M1 已被 06 的引擎二进制验证过（11,172,032 参数逐位一致），
   推广到 N 层只需要改「conv 个数」这一个维度：
     · ResNet-18:2 层 + 可选 downsample（06 原有）
     · ResNet-50:  3 层 + 每个 stage 首block 有 downsample
     · MobileNetV2: 2 层 + **可选** skip（无 downsample）

   **能用已验证的机制扩展，就不要换新机制。**

blocks v2 格式（magic **PTQ2**，与 06 的 PTQW 并存）：
  magic "PTQ2" / n_conv / n_block / stem_pool / n_conv_per_block
  每个 conv:  cout cin kh kw sh sw ph pw relu (9 int32) + weight + bias
  每个 block: c0_idx c1_idx c2_idx ds_idx stride (5 int32)
              —— conv 索引按「块内第几个」递增；ds_idx = -1 表示 identity 直连
  fc: fc_in fc_out + weight + bias

⭐ conv 的 `relu` 标志语义（int32，不新增字段）：
     0 = 无激活（线性输出）
     1 = ReLU（ResNet 默认）
     2 = ReLU6（clamp 到 [0,6]，MobileNetV2）

⭐ block 的 `residual` 标志语义（int32，仍是一个字段）：
     0 = 无残差（纯串联，最后 conv 的 relu 在 conv() 里做）
     1 = 有残差 + 相加后接 ReLU（ResNet 默认）
     2 = 有残差 + 相加后**无**激活（MobileNetV2：project 是线性输出，加完就走）

C++ 侧的实现：把 block 里的 conv 从 c0 依次做到 c_{n-1}，
若有 ds 则先算 identity，最后按 residual(1/2) 决定是否 relu(sum)。
"""
import struct
from pathlib import Path
from typing import Dict, List, NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAGIC = b"PTQ2"

# conv 的 relu 标志 / replay 分派用：激活名 → int32 值。
ACT_INT = {"relu": 1, "relu6": 2}


class ConvRec(NamedTuple):
    cout: int
    cin: int
    kh: int
    kw: int
    sh: int
    sw: int
    ph: int
    pw: int
    relu: int          # 0=无激活, 1=ReLU, 2=ReLU6（见模块头语义说明）
    weight: np.ndarray
    bias: np.ndarray
    # depthwise 的分组数（MobileNetV2 用 groups=Cin；普通 conv 是 1）。
    # ⚛不存这一项的话，F.conv2d 会因通道不匹配报错
    #   （"weight of size [32,1,3,3] but input has 32 channels"）。
    groups: int = 1

    def meta(self):
        return (self.cout, self.cin, self.kh, self.kw,
                self.sh, self.sw, self.ph, self.pw, self.relu)


class BlockRec(NamedTuple):
    convs: tuple      # 块内 conv 的全局索引（按执行顺序）
    ds: int           # downsample conv 索引，-1 = identity 直连
    stride: int
    # 残差语义（int32，二进制仍是一个字段）：
    #   0 = 无残差（纯串联，最后一个 conv 的 relu 在 conv() 里做）
    #   1 = 有残差 + 相加后接 ReLU（ResNet 默认）
    #   2 = 有残差 + 相加后**无**激活（MobileNetV2：project 线性输出，加完就走）
    residual: int = 1


def conv_index(model: nn.Module) -> Dict[str, int]:
    return {n: i for i, (n, _) in enumerate(
        (nm, m) for nm, m in model.named_modules() if isinstance(m, nn.Conv2d))}


def build_blocks(model: nn.Module, plan: List[Dict], stem_pool: bool,
                 act: str = "relu"):
    """由 plan 构建 blocks。

    plan 每项：{"convs": [层名...], "downsample": 层名或None, "stride": int,
               可选 "_no_residual" / "_post_relu" / "_last_relu"}
    act: "relu"（ResNet 默认）或 "relu6"（MobileNetV2），决定「有激活」时
         conv 写 1 还是 2；不写死模型名，由 plan 顶层带 _act 时经 export_blocks 传入。
    """
    act_val = ACT_INT[act]
    idx = conv_index(model)
    convs_named = [(n, m) for n, m in model.named_modules()
                   if isinstance(m, nn.Conv2d)]

    # ⭐ stem conv 必须在 blocks 之外先执行。
    #   ResNet-18/50 的 stem 是第一个 conv（conv1）；MobileNetV2 是 stem.0。
    #   blocks 列表只覆盖残差块，stem 由调用方通过 stem_conv 指定。
    stem_conv = plan[0].get("_stem") if plan else None
    in_blocks = {n for p in plan for n in p["convs"]}
    in_blocks |= {p["downsample"] for p in plan if p.get("downsample")}
    outside = [n for n, _ in convs_named if n not in in_blocks]

    if stem_conv is None:
        stem_conv = outside[0] if outside else None
    if stem_conv is None:
        raise RuntimeError("找不到 stem conv（所有 conv 都属于某个 block）")

    # ⛔ plan 之外的 conv 只能有 stem 一个。
    #   MobileNetV2 曾漏掉 `features.18`（1×1 投影到 1280），
    #   症状极隐蔽：replay 跑完 mean 后只有 64 通道，fc 报
    #   "mat1 and mat2 shapes cannot be multiplied (2x64 and 512x10)" ——
    #   错误信息完全指不到「少跑了一个 conv」。
    leftover = [n for n in outside if n != stem_conv]
    if leftover:
        raise RuntimeError(
            f"topo_plan 漏了 {len(leftover)} 个不属于任何 block 的 conv："
            f"{leftover[:6]}{' ...' if len(leftover) > 6 else ''}\n"
            f"它们既不是 stem 也不在 plan 的 convs/downsample 里，"
            f"replay 时会被静默跳过 → 输出维度错但看不出原因。\n"
            f"请把它们补进 plan（例如 MobileNetV2 的 features.18 投影层）。")

    recs = []
    for _, m in convs_named:
        w = m.weight.detach().cpu().contiguous().numpy()
        cout, cin, kh, kw = w.shape
        b = (m.bias.detach().cpu().contiguous().numpy()
             if m.bias is not None else np.zeros(cout, dtype=np.float32))
        recs.append(ConvRec(cout, cin, kh, kw, m.stride[0], m.stride[1],
                            m.padding[0], m.padding[1], 1, w, b, m.groups))

    stem_idx = idx[stem_conv]

    # ---- 块级 residual 编码（0/1/2，见模块头）----
    #   _no_residual → 0（纯串联）；
    #   有残差且默认相加后接激活 → 1（ResNet）；
    #   有残差但 _post_relu=False → 2（MobileNetV2：project 线性输出，加完就走）。
    blocks = []
    for p in plan:
        cs = tuple(idx[n] for n in p["convs"])
        ds = idx[p["downsample"]] if p.get("downsample") else -1
        if p.get("_no_residual"):
            res = 0
        else:
            res = 1 if p.get("_post_relu", True) else 2
        blocks.append(BlockRec(cs, ds, int(p.get("stride", 1)), res))

    # ---- conv 的 relu 标志（0/1/2）----
    # 默认所有 conv 都接「当前激活」（act_val：ResNet=ReLU1 / MV2=ReLU6=2）。
    relu_of = {i: act_val for i in range(len(recs))}

    # stem conv 接激活（三个模型的 stem 后都有非线性）—— 保持 act_val。
    for p, blk in zip(plan, blocks):
        last = blk.convs[-1]
        # 「块内最后一个 conv 是否接激活」：
        #   显式 _last_relu 优先；否则沿用旧语义——
        #   有残差的块最后 conv 不激活（激活在相加后），无残差的块最后 conv 激活。
        if "_last_relu" in p:
            last_act = p["_last_relu"]
        else:
            last_act = (blk.residual == 0)
        if not last_act:
            relu_of[last] = 0
        # downsample 在残差那一侧，永不激活（ResNet-50/18；ds 恒 -1 for MV2）。
        if blk.ds >= 0 and blk.residual in (1, 2):
            relu_of[blk.ds] = 0

    recs = [r._replace(relu=(act_val if i == stem_idx else relu_of[i]))
            for i, r in enumerate(recs)]
    return recs, blocks, stem_pool, stem_idx


def replay_blocks(model, recs, blocks, stem_pool, x, stem_idx=0):
    def conv(i, t):
        r = recs[i]
        # groups 从 recs 拿 —— MobileNetV2 的 depthwise 是 groups=Cin，
        # 漏了它会报「weight of size [32,1,3,3] but input has 32 channels」
        # ⚠️⛔⛔ `groups` 是 F.conv2d 的**第 7 个位置参数**（在 dilation 之后）。
        #   写成第 6 个会被当成 dilation —— 实测 groups=32 传到 dilation 位时，
        #   PyTorch 按 groups=1 解释权重，报
        #   「Given groups=1, weight of size [32,1,3,3], expected input to have
        #     1 channels but got 32」，错误信息里的 groups 反而是错的 1，
        #   **极难定位**。这里一律用关键字参数。
        y = F.conv2d(t, torch.from_numpy(r.weight).to(t.device),
                     torch.from_numpy(r.bias).to(t.device),
                     stride=(r.sh, r.sw), padding=(r.ph, r.pw),
                     dilation=1, groups=r.groups)
        # relu 标志分派：0=无激活 / 1=ReLU / 2=ReLU6（clamp 到 [0,6]）。
        # ⚠️ 训练权重下 MV2 确有层输出 >6（如 features.1 的 depthwise 最大 6.685），
        #   用 F.relu 代替 ReLU6 会让自检 max|diff| 差到 8.35 —— 三处语义缺一不可。
        if r.relu == 1:
            return F.relu(y)
        if r.relu == 2:
            return F.hardtanh(y, 0.0, 6.0)
        return y

    y = conv(stem_idx, x)
    if stem_pool:
        y = F.max_pool2d(y, 3, 2, 1)
    for blk in blocks:
        t = y
        for ci in blk.convs:
            t = conv(ci, t)
        if blk.residual in (1, 2):
            ident = conv(blk.ds, y) if blk.ds >= 0 else y
            s = t + ident
            # residual=1（ResNet）相加后接 ReLU；residual=2（MV2）相加后不激活。
            y = F.relu(s) if blk.residual == 1 else s
        else:
            y = t          # 纯串联：最后一个 conv 的 relu 已在 conv() 里做了
    fc = [m for _, m in model.named_modules() if isinstance(m, nn.Linear)][-1]
    return fc(y.mean(dim=(2, 3)).flatten(1))


def export_blocks(model: nn.Module, path, x: torch.Tensor, plan: List[Dict],
                  stem_pool: bool | None = None, strict: bool = True) -> Dict:
    """导出 PTQ2 格式（含自检）。"""
    from .exporter import ensure_folded
    ensure_folded(model)
    if stem_pool is None:
        stem_pool = any(isinstance(m, nn.MaxPool2d)
                        for _, m in model.named_modules())

    # 激活类型由 plan 顶层/首项的 _act 声明（不写死模型名）：默认 "relu"。
    act = next((p["_act"] for p in plan if "_act" in p), "relu")
    recs, blocks, stem_pool, stem_idx = build_blocks(model, plan, stem_pool, act)

    with torch.no_grad():
        ref = model(x).float()
        got = replay_blocks(model, recs, blocks, stem_pool, x, stem_idx).float()
    d = float((got - ref).abs().max())
    if strict and d > 1e-4:
        raise RuntimeError(
            f"PTQ2 拓扑自检失败：max|diff| = {d:.4g}（应 < 1e-4）。"
            f"conv {len(recs)} 个 / block {len(blocks)} 个。"
            f"常见原因：plan 里的 convs 顺序或 downsample 指错。")
    _ = None

    blob = bytearray()
    blob += MAGIC
    blob += struct.pack("<4i", len(recs), len(blocks), int(stem_pool),
                        max(len(b.convs) for b in blocks))   # 4 = n_conv_per_block
    for r in recs:
        # 10 个 int32：前 9 个是 conv 元信息 + groups（第 10 个）
        blob += struct.pack("<9i", *r.meta()) + struct.pack("<i", r.groups)
        blob += np.ascontiguousarray(r.weight, dtype="<f4").tobytes()
        blob += np.ascontiguousarray(r.bias, dtype="<f4").tobytes()
    ncp = max(len(b.convs) for b in blocks)
    for blk in blocks:
        cs = tuple(blk.convs) + (-1,) * (ncp - len(blk.convs))
        # 固定 ncp + 3 个字段（ds / stride / residual），长度恒定，
        # 否则不同 blocks_per_block 的模型打包长度会不一致（实测 pack 报错）
        blob += struct.pack(f"<{ncp + 3}i", *cs, blk.ds, blk.stride, blk.residual)
    fw = [m for _, m in model.named_modules() if isinstance(m, nn.Linear)][-1]
    W = fw.weight.detach().cpu().contiguous().numpy()
    B = fw.bias.detach().cpu().contiguous().numpy()
    blob += struct.pack("<2i", W.shape[1], W.shape[0])
    blob += np.ascontiguousarray(W, dtype="<f4").tobytes()
    blob += np.ascontiguousarray(B, dtype="<f4").tobytes()

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(bytes(blob))
    return dict(path=str(p), n_conv=len(recs), n_block=len(blocks),
                stem_pool=int(stem_pool), bytes=len(blob),
                selfcheck_maxdiff=d, topo="blocks/PTQ2",
                stem_idx=stem_idx,
                max_convs_per_block=max(len(b.convs) for b in blocks))
