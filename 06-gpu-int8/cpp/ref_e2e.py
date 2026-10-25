# -*- coding: utf-8 -*-
"""端到端参考实现：纯 numpy 复现 GPU int8 引擎的**最终输出**。

⭐ 这个脚本存在的理由（判据的转变）：
   之前用「逐层比对 im2col / int32」来验证正确性，结果是：
     · 中间 col 的比对逻辑本身有 4 个 bug（编码溢出、batch 索引错、
       探针口径没跟上主线改动、双重量化）
     · 每次都要重新编译 probe，一轮下来烧掉十几轮迭代
   **中间张量的布局是实现细节，正确性应该以最终输出为准。**
   本脚本按同一套语义算一遍完整前向，导出参考 logits，
   再和 GPU 的 logits 逐元素比 —— 一次比对证明全部。

独立性（这是参考实现的价值所在）：
   · 不 import 任何 C++ 侧代码
   · 不用 PyTorch 的 conv / matmul / quantized算子做卷积与 GEMM
     （nn.functional.conv2d、torch._int_mm 都不用，避免"用被测对象的
      同源实现验证被测对象"）
   · im2col、GEMM、requant、maxpool、残差全部自己写

用到的只有 numpy（矩阵乘用 np.matmul，这是标准 BLAS，不算"抄实现"）。

语义要与 C++ 侧严格一致（见 ptq_quant.h）：
   权重：per-channel sym，scale = absmax/(2^(b-1)-1)，clamp[-(qmax+1), qmax]
   激活：**对称**（zp 恒 0），默认 AbsMax：scale = max(|lo|,|hi|)/(2^(a-1)-1)
        与 09-ptq-toolkit 校准一致；`--act-mode minmax` 可切回 (hi-lo) 口径
        对照 06 引擎的 `--act-minmax`。clamp 同上
   为什么是对称不是非对称：8bit 非对称的码空间 [0,255] 装不进 int8
   [-128,127]，详见 README 第一节。
"""
import argparse
import struct
import sys

import numpy as np

MAGIC = b"PTQW"
CALIB_MAGIC = b"PTQC"


# ---------------------------------------------------------------- 文件读取

def read_model(path):
    with open(path, "rb") as f:
        d = f.read()
    assert d[:4] == MAGIC, "模型 magic 不符"
    off = 4
    n_conv, n_block, stem_pool = struct.unpack_from("<3i", d, off)
    off += 12

    convs = []
    for _ in range(n_conv):
        cout, cin, kh, kw, sh, sw, ph, pw, relu = struct.unpack_from("<9i", d, off)
        off += 36
        n = cout * cin * kh * kw
        w = np.frombuffer(d, dtype="<f4", count=n, offset=off).reshape(cout, cin, kh, kw)
        off += 4 * n
        b = np.frombuffer(d, dtype="<f4", count=cout, offset=off)
        off += 4 * cout
        convs.append(dict(cout=cout, cin=cin, kh=kh, kw=kw, sh=sh, sw=sw,
                          ph=ph, pw=pw, relu=relu, w=np.array(w), b=np.array(b)))

    blocks = []
    for _ in range(n_block):
        c1, c2, ds, stride = struct.unpack_from("<4i", d, off)
        off += 16
        blocks.append((c1, c2, ds, stride))

    fc_in, fc_out = struct.unpack_from("<2i", d, off)
    off += 8
    fcw = np.frombuffer(d, dtype="<f4", count=fc_in * fc_out, offset=off).reshape(fc_out, fc_in)
    off += 4 * fc_in * fc_out
    fcb = np.frombuffer(d, dtype="<f4", count=fc_out, offset=off)
    off += 4 * fc_out
    assert off == len(d), "模型文件有 %d 字节未消费" % (len(d) - off)
    return convs, blocks, stem_pool, fcw, fcb


def read_calib(path):
    with open(path, "rb") as f:
        d = f.read()
    assert d[:4] == CALIB_MAGIC, "校准文件 magic 不符"
    n = struct.unpack_from("<i", d, 4)[0]
    ranges = []
    for i in range(n):
        lo, hi = struct.unpack_from("<2f", d, 8 + i * 8)
        ranges.append((lo, hi))
    return ranges


# ---------------------------------------------------------------- 量化

def quantize_weight_per_channel_sym(w, bits):
    """w: (Cout, Cin, kh, kw) → (int8 (Cout, K), scale (Cout,))

    与 ptq_quant.h::quantize_weight_per_channel_sym 同口径：
      scale = absmax / qmax  (clamp 1e-8)
      q     = round(w/scale).clamp(-(qmax+1), qmax)
    ⛔ 用 np.round（banker's rounding）会与 torch.round（half-away）差半码字，
       这里显式实现 half-away。
    """
    cout = w.shape[0]
    wf = w.reshape(cout, -1).astype(np.float64)
    qmax = 2 ** (bits - 1) - 1
    amax = np.abs(wf).max(axis=1)
    scale = np.maximum(amax / qmax, 1e-8)
    q = _round_half_away(wf / scale[:, None])
    q = np.clip(q, -(qmax + 1), qmax)
    return q.astype(np.int8), scale.astype(np.float32)


def _round_half_away(x):
    """half-away-from-zero，与 torch.round / ptq_quant.h::torch_round 一致。"""
    return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5))


def act_scale_from_range(lo, hi, bits, mode="absmax"):
    """对称激活量化参数，zp 恒 0。

    absmax（默认，与 09 校准 / 06 引擎默认一致）：scale = max(|lo|,|hi|)/qmax
    minmax（对照 06 的 --act-minmax）：scale = (hi-lo)/qmax
    两者只在范围跨零且两侧不对称时不同。
    """
    qmax = 2 ** (bits - 1) - 1
    if mode == "absmax":
        amax = max(abs(lo), abs(hi))
        return float(max(amax / qmax, 1e-8)), qmax
    return float(max((hi - lo) / qmax, 1e-8)), qmax


# ---------------------------------------------------------------- im2col + GEMM

def im2col(x, kh, kw, sh, sw, ph, pw, Kpad):
    """x: (N, C, H, W) → (N*OH*OW, Kpad)，L-major（(n,oh,ow) 在外、k 在内）。

    与 C++ kernel 的 row→(n,oh,ow) / k→(c,i,j) 分解顺序一致。
    用 stride_tricks 做纯视图展开，不循环不拷贝。

    ⚠️ Kpad 是硬要求：conv1 的 K = 3*7*7 = 147 不是 8 的倍数，
       int8 tensor core 要求 K % 8 == 0。末尾补零列（GEMM 里乘 0，不改结果）。
       —— 与 C++ kernel 里的 `if (k >= K) out = 0` 完全对应。
    """
    N, C, H, W = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    OH = (xp.shape[2] - kh) // sh + 1
    OW = (xp.shape[3] - kw) // sw + 1
    # (N,C,OH,OW,kh,kw) 的视图 —— k 的展平顺序天然是 (c,i,j)
    s = xp.strides
    view = np.lib.stride_tricks.as_strided(
        xp, shape=(N, C, OH, OW, kh, kw),
        strides=(s[0], s[1], s[2] * sh, s[3] * sw, s[2], s[3]), writeable=False)
    col = view.transpose(0, 2, 3, 1, 4, 5).reshape(N * OH * OW, C * kh * kw)

    K = C * kh * kw
    if Kpad != K:
        col = np.zeros((col.shape[0], Kpad), dtype=col.dtype)
        col[:, :K] = view.transpose(0, 2, 3, 1, 4, 5).reshape(col.shape[0], K)
    return col, OH, OW


def int8_conv(x, conv, wq_int8, wq_scale, ascale, aqmax):
    """一层的完整 int8 前向：量化激活 → im2col → int32 GEMM → per-channel 反量化。

    x: (N,C,H,W) float32
    返回 (N, Cout, OH, OW) float32
    """
    N, C, H, W = x.shape
    kh, kw = conv["kh"], conv["kw"]
    sh, sw, ph, pw = conv["sh"], conv["sw"], conv["ph"], conv["pw"]

    # 1) 量化激活（对称，zp=0）
    q = _round_half_away(x.astype(np.float64) / ascale)
    q = np.clip(q, -(aqmax + 1), aqmax).astype(np.int8)

    # 2) im2col（补零到 Kpad）
    K = C * kh * kw
    Kpad = wq_int8.shape[0]
    col, OH, OW = im2col(q, kh, kw, sh, sw, ph, pw, Kpad)

    # 3) int32 GEMM。wq_int8 已经是 (Kpad, Cout)
    acc = col.astype(np.int32) @ wq_int8.astype(np.int32)   # (M, Cout) int32

    # 4) per-channel 反量化 + bias（+ ReLU 由调用方决定）
    out = acc.astype(np.float32) * (wq_scale * ascale)[None, :]
    out += conv["b"][None, :]
    return out.reshape(N, OH, OW, -1).transpose(0, 3, 1, 2)


def relu(x):
    return np.maximum(x, 0.0)


def maxpool3x3s2p1(x):
    """(N,C,H,W) → (N,C,OH,OW)。只实现 stem 用到的那一种。"""
    N, C, H, W = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), constant_values=-np.inf)
    OH = (H + 2 - 3) // 2 + 1
    OW = (W + 2 - 3) // 2 + 1
    # 窗口视图 (N,C,OH,OW,3,3)
    s = xp.strides
    view = np.lib.stride_tricks.as_strided(
        xp, shape=(N, C, OH, OW, 3, 3),
        strides=(s[0], s[1], s[2] * 2, s[3] * 2, s[2], s[3]), writeable=False)
    return view.max(axis=(-1, -2))


# ---------------------------------------------------------------- 前向

def forward(x, convs, blocks, stem_pool, fcw, fcb, ranges, w_bits, a_bits,
            stop_after_block=None, trace=None, act_mode="absmax"):
    N = x.shape[0]
    # 预先量化所有权重（Cout, Cin*kh*kw）→ (Kpad, Cout)
    wq = []
    for c in convs:
        q, s = quantize_weight_per_channel_sym(c["w"], w_bits)
        K = q.shape[1]
        Kpad = (K + 7) // 8 * 8
        qt = np.zeros((Kpad, c["cout"]), dtype=np.int8)
        qt[:K, :] = q.T          # (K, Cout)
        wq.append((qt, s))
    act = [act_scale_from_range(lo, hi, a_bits, act_mode) for lo, hi in ranges]

    y = int8_conv(x, convs[0], wq[0][0], wq[0][1], *act[0])
    if convs[0]["relu"]:
        y = relu(y)
    if stem_pool:
        y = maxpool3x3s2p1(y)

    for bi, (c1, c2, ds, _stride) in enumerate(blocks):
        if ds >= 0:
            ident = int8_conv(y, convs[ds], wq[ds][0], wq[ds][1], *act[ds])
            if convs[ds]["relu"]:
                ident = relu(ident)
        else:
            ident = y
        t = int8_conv(y, convs[c1], wq[c1][0], wq[c1][1], *act[c1])
        if convs[c1]["relu"]:
            t = relu(t)
        t = int8_conv(t, convs[c2], wq[c2][0], wq[c2][1], *act[c2])
        if convs[c2]["relu"]:
            t = relu(t)
        y = relu(t + ident)
        if trace is not None:
            trace.append(y.copy())

        # ⭐ 二分定位用：跑到指定 block 就停，返回该处的激活。
        #   动机是端到端比对只告诉你「不一致」，不告诉你「哪一层开始错的」。
        #   逐层编 probe 找错层太慢（每层要重编译），
        #   而这里只需 --stop-after-block K 跑一次，比对 K+1/K 两次即可定位。
        if stop_after_block is not None and bi == stop_after_block:
            return y

    if stop_after_block is not None:
        return y

    pooled = y.mean(axis=(2, 3))          # (N, 512)
    return pooled @ fcw.T + fcb


# ---------------------------------------------------------------- 主流程

def read_test_batch(path, limit):
    """读 cifar10.cpp 用的同一种 raw 格式：每条 1 字节 label + 3072 字节像素。"""
    with open(path, "rb") as f:
        d = f.read()
    rec = 3073
    assert len(d) % rec == 0
    n = len(d) // rec
    n = min(n, limit)
    arr = np.frombuffer(d[:n * rec], dtype=np.uint8).reshape(n, rec)
    labels = arr[:, 0].astype(np.int64)
    px = arr[:, 1:].reshape(n, 3, 32, 32).astype(np.float32)

    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32) * 255.0
    std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32) * 255.0
    x = (px - mean[None, :, None, None]) / std[None, :, None, None]
    return x, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet18_folded.bin")
    ap.add_argument("--calib", default="calib_minmax.bin")
    ap.add_argument("--data", default="/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/data_raw/test_batch.raw")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--w-bits", type=int, default=8)
    ap.add_argument("--a-bits", type=int, default=8)
    ap.add_argument("--act-mode", choices=["absmax", "minmax"], default="absmax",
                    help="激活 scale 口径：默认 absmax（与引擎一致）；"
                         "引擎用 --act-minmax 时这里选 minmax")
    ap.add_argument("--chunk", type=int, default=250,
                    help="每次前向的样本数（numpy 内存占用与耗时都随此线性增长）")
    ap.add_argument("--out", required=True, help="参考 logits 输出路径")
    ap.add_argument("--stop-after-block", type=int, default=None,
                    help="跑到第 K 个 block 就停并导出该处激活（二分定位用）")
    ap.add_argument("--dump-mid", default=None,
                    help="前缀：逐 block 导出中间激活到 <prefix><k>.bin，"
                         "配合 GPU 侧的 locate_block 做分歧定位")
    args = ap.parse_args()

    convs, blocks, stem_pool, fcw, fcb = read_model(args.model)
    ranges = read_calib(args.calib)
    print("conv %d 层，block %d 个，stem_pool=%d" % (len(convs), len(blocks), stem_pool))

    limit = args.limit if args.limit > 0 else 10000
    x_all, labels = read_test_batch(args.data, limit)
    print("测试样本 %d 张" % x_all.shape[0])

    if args.dump_mid:
        # 逐 block 各跑一遍，只取该 block 之后的激活。
        # 8 个 block × 每次前向 —— 参考实现是 numpy，慢，但只在定位时用一次。
        for bi in range(len(blocks)):
            y = forward(x_all[:args.chunk], convs, blocks, stem_pool,
                        fcw, fcb, ranges, args.w_bits, args.a_bits,
                        stop_after_block=bi, act_mode=args.act_mode)
            path = "%s%d.bin" % (args.dump_mid, bi)
            with open(path, "wb") as f:
                f.write(b"PTQM")
                f.write(struct.pack("<4i", y.shape[0], y.shape[1], y.shape[2], y.shape[3]))
                f.write(np.ascontiguousarray(y, dtype=np.float32).tobytes())
            print("  导出 block %d → %s%s" % (bi, path, str(y.shape)))
        return

    outs = []
    for i in range(0, x_all.shape[0], args.chunk):
        outs.append(forward(x_all[i:i + args.chunk], convs, blocks, stem_pool,
                            fcw, fcb, ranges, args.w_bits, args.a_bits,
                            stop_after_block=args.stop_after_block,
                            act_mode=args.act_mode))
        print("  已算 %d/%d" % (min(i + args.chunk, x_all.shape[0]), x_all.shape[0]))
    logits = np.concatenate(outs, axis=0).astype(np.float32)

    if args.stop_after_block is not None:
        print("已停在 block %d，中间激活形状 %s" %
              (args.stop_after_block, logits.shape))
    else:
        pred = logits.argmax(axis=1)
        acc = float((pred == labels).mean())
        print("参考实现精度 %.2f%%（%d/%d）" % (100 * acc, int((pred == labels).sum()),
                                             x_all.shape[0]))

    with open(args.out, "wb") as f:
        f.write(b"PTQL")
        f.write(struct.pack("<2i", logits.shape[0], logits.shape[1]))
        f.write(logits.tobytes())
    print("已写出 %s（%s）" % (args.out, logits.shape))


if __name__ == "__main__":
    main()