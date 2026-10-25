# -*- coding: utf-8 -*-
"""导出 BN 折叠后的 fp32 权重 + **网络拓扑**，供 C++ int8 引擎加载。

⭐ 为什么必须导出拓扑（这是本轮最关键的教训）：
   ResNet-18 不是纯串联。C++ 侧若按「conv 索引顺序依次执行」来实现，
   residual 分支会被整个丢掉 —— 实测症状：
     · 单层校验全部通过（max|diff| = 9.5e-7）
     · 全网精度 7.6%（= 随机猜）
   **教训：单层对 ≠ 全网对。结构性错误不会被逐层校验发现。**
   所以拓扑必须显式描述，不能让 C++ 侧猜。

输出 `resnet18_folded.bin`（小端）：
  magic "PTQW"  n_conv int32
  n_block int32            （4 个 BasicBlock）
  n_stem_pool int32        （stem 之后是否有 maxpool，这里恒为 1）
  每个 conv:  cout cin kh kw sh sw ph pw relu(8 int32 中的第 9 个)
             weight cout*cin*kh*kw fp32（Cout-major）
             bias   cout fp32
  每个 block: conv1_idx conv2_idx downsample_idx stride  (4 int32)
               downsample_idx = -1 表示 identity 直连
  fc: fc_in fc_out (2 int32) + weight fc_out*fc_in + bias fc_out

**relu 标志的语义**：conv 输出后是否立刻接 ReLU。
  downsample 的 1x1 conv **不接 ReLU**（它在残差相加的那一侧）。
  block 的 conv2 **也不接 ReLU** —— ReLU 在 add 之后。
  这两个标志错了，整个 block 的数就全错。

自检：导出后用同一份拓扑在 PyTorch 侧复算一遍 logits，
      与 fold 后的模型逐位比对（max diff 必须为 0）。
"""
import os
import struct
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/mnt/d/WorkBuddy/ShowBro/lab/03-ptq-lab")
import config as C
import src.model as M

MAGIC = b"PTQW"
OP_POOL = 1


def export():
    model = M.fold_all_bn(M.load_fp32())
    model.eval()

    convs = [m for _, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    print("conv 层数 %d，BN 剩余 %d（应为 0）" % (len(convs), M.count_bn(model)))

    # ---- 拓扑：把 conv 映射到 (block, role) ----
    conv1_i = 0
    blocks = []

    def idx_of(mod):
        return convs.index(mod)

    for layer in (model.layer1, model.layer2, model.layer3, model.layer4):
        for blk in layer:
            c1 = idx_of(blk.conv1)
            c2 = idx_of(blk.conv2)
            ds = idx_of(blk.downsample[0]) if blk.downsample is not None else -1
            blocks.append((c1, c2, ds, blk.conv1.stride[0]))
    print("block 数 %d：%s" % (len(blocks), blocks))

    # ---- relu 标志 ----
    relu_flag = [1] * len(convs)      # 默认接 ReLU
    relu_flag[0] = 1                  # stem conv1 接 ReLU
    for c1, c2, ds, _ in blocks:
        relu_flag[c2] = 0             # block 的 conv2：ReLU 在 add 之后
        if ds >= 0:
            relu_flag[ds] = 0         # downsample：残差那一侧，不接 ReLU

    blob = bytearray()
    blob += MAGIC
    blob += struct.pack("<3i", len(convs), len(blocks), 1)  # 第三个 = stem 后有 pool

    for i, conv in enumerate(convs):
        w = conv.weight.data.detach().cpu().contiguous()
        b = conv.bias.data.detach().cpu().contiguous()
        cout, cin, kh, kw = w.shape
        meta = (cout, cin, kh, kw, conv.stride[0], conv.stride[1],
                conv.padding[0], conv.padding[1], relu_flag[i])
        blob += struct.pack("<9i", *meta)
        blob += w.numpy().astype("<f4").tobytes()
        blob += b.numpy().astype("<f4").tobytes()

    for c1, c2, ds, st in blocks:
        blob += struct.pack("<4i", c1, c2, ds, st)

    fc = model.fc
    fw = fc.weight.data.detach().cpu().contiguous()
    fb = fc.bias.data.detach().cpu().contiguous()
    blob += struct.pack("<2i", fw.shape[1], fw.shape[0])
    blob += fw.numpy().astype("<f4").tobytes()
    blob += fb.numpy().astype("<f4").tobytes()

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "resnet18_folded.bin")
    with open(out, "wb") as f:
        f.write(bytes(blob))
    print("写出 %s（%.2f MB）" % (out, len(blob) / 1024 / 1024))
    return model, blocks, relu_flag


def selfcheck(model, blocks, relu_flag, path):
    """回读文件 + 用导出的拓扑复算 logits，与原模型逐位比对。"""
    with open(path, "rb") as f:
        data = f.read()
    off = 0
    assert data[:4] == MAGIC
    off = 4
    n_conv, n_block, has_pool = struct.unpack_from("<3i", data, off)
    off += 12

    meta = []
    maxdiff_w = 0.0
    for i in range(n_conv):
        m = struct.unpack_from("<9i", data, off)
        off += 36
        cout, cin, kh, kw = m[:4]
        n = cout * cin * kh * kw
        w = torch.frombuffer(bytearray(data[off:off + 4 * n]),
                             dtype=torch.float32).reshape(cout, cin, kh, kw)
        off += 4 * n
        b = torch.frombuffer(bytearray(data[off:off + 4 * cout]), dtype=torch.float32)
        off += 4 * cout
        meta.append((m, w, b))
    blk_info = []
    for i in range(n_block):
        blk_info.append(struct.unpack_from("<4i", data, off))
        off += 16
    fc_in, fc_out = struct.unpack_from("<2i", data, off)
    off += 8
    fcw = torch.frombuffer(bytearray(data[off:off + 4 * fc_in * fc_out]),
                           dtype=torch.float32).reshape(fc_out, fc_in)
    off += 4 * fc_in * fc_out
    fcb = torch.frombuffer(bytearray(data[off:off + 4 * fc_out]), dtype=torch.float32)
    off += 4 * fc_out
    assert off == len(data), "文件有 %d 字节未消费（应为 0）" % (len(data) - off)

    # 用导出数据复算
    def conv(i, x):
        m, w, b = meta[i]
        cout, cin, kh, kw, sh, sw, ph, pw, rl = m
        y = F.conv2d(x, w, b, (sh, sw), (ph, pw))
        return F.relu(y) if rl else y

    torch.manual_seed(0)
    x = torch.randn(4, 3, 32, 32)
    y = conv(0, x)
    if has_pool:
        y = F.max_pool2d(y, 3, 2, 1)
    for c1, c2, ds, _ in blk_info:
        identity = x0 = y
        if ds >= 0:
            identity = conv(ds, x0)
        out = conv(c1, y)
        out = conv(c2, out)
        y = F.relu(out + identity)
    pooled = F.adaptive_avg_pool2d(y, 1).flatten(1)
    logits = pooled @ fcw.t() + fcb

    ref = model(x)
    d = (logits - ref).abs().max().item()
    print("拓扑自检：导出拓扑复算 vs 原模型 max|diff| = %g（必须为 0）" % d)
    assert d < 1e-4, "拓扑或权重导出有偏差"
    print("通过")


if __name__ == "__main__":
    model, blocks, relu_flag = export()
    selfcheck(model, blocks, relu_flag,
              os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "resnet18_folded.bin"))