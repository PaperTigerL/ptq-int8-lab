# -*- coding: utf-8 -*-
"""先看清数字是怎么被压到整数网格上的 —— 再去写 src/quant.py 的三个算子。

为什么要先跑这个
    填 `src/quant.py` 之前，先用 8 个数把量化流程**一步一步手动走一遍**，
    把每一步的中间数字全部打印出来。看完这四段，你就知道
    `quantize_per_tensor_sym` / `per_channel` / `per_tensor_asym`
    各自在解决什么问题、每一步为什么必须有。

这个脚本里没有留白，也不 import src.quant —— 它是**尺子**，不是答案。
    尺度上写死：8 个数、固定 bits、把公式显式摊开成一行行 print。
    你要做的是：把每段「流程」抽象成一个能处理任意张量、任意 bits、
    任意 dtype/device 的函数，并且 clamp 边界、处理全零输入。

跑：
    cd /mnt/d/WorkBuddy/ShowBro/lab/03-ptq-lab
    python tools/quant_by_hand.py
"""
import sys
import torch

torch.manual_seed(0)

# 刻意用 4bit：码字少，才能在屏幕上把每个码字都看清楚
BITS = 4
QMAX = 2 ** (BITS - 1) - 1          # 对称量化正方向最大码字 = 7
QMIN = -2 ** (BITS - 1)             # 负方向 = -8（注意不对称：8 和 7 不一样多）
QAS = 2 ** BITS - 1                 # 非对称的码字总数 = 15


def hr(title):
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def sqnr(ref, xq):
    """信号量化噪比，dB，越大越好。tests/ 里的判据用的就是这个量。"""
    mse = ((xq - ref) ** 2).mean().item()
    pwr = (ref ** 2).mean().item()
    return 10 * torch.log10(torch.tensor(pwr / max(mse, 1e-12))).item()


def show(x, q, xq, scale):
    """把一个张量的四行并排打出来：原值 / 码字 / 反量化 / 误差。"""
    print("  %-14s %-14s %-14s %-12s %-8s %-8s"
          % ("原值 x", "x/scale", "round 后", "反量化 xq", "误差", "码字"))
    print("  " + "-" * 70)
    for i in range(x.numel()):
        raw = x[i].item()
        ratio = x[i] / scale
        qi = q[i].item()
        out = xq[i].item()
        err = out - raw
        print("  %+14.6f %+14.4f %+14d %+14.6f %+8.4f %+8d"
              % (raw, ratio.item(), qi, out, err, qi))
    clamp_hit = (q == QMAX) | (q == QMIN)
    n_clamp = int(clamp_hit.sum())
    used = torch.unique(q).numel()
    print("  " + "-" * 70)
    print("  用了 %d 个不同码字（对称 %dbit 共 %d 个取值 [%d, %d]），"
          "取整后正好撞到边界被 clamp 的是 %d 个"
          % (used, BITS, QMAX - QMIN + 1, QMIN, QMAX, n_clamp))


# ==================================================================
# 第一段：对称 per-tensor —— 三个步骤，逐行打印
# ==================================================================
hr("第一段  对称 per-tensor：8 个数被压到 %d bit 的整数网格上" % BITS)

x = torch.tensor([0.37, -1.20, 0.05, 2.80, -0.62, 1.05, -2.40, 0.91])

print("\n[步骤 1] abs().max() —— 整张张量共用这一个数决定 scale")
amax = x.abs().max()
print("  x =", [round(v, 2) for v in x.tolist()])
print("  x.abs().max() = %.6f" % amax.item())

print("\n[步骤 2] scale = absmax / %d   （为什么是 %d 而不是 %d，见下面）" % (QMAX, QMAX, QAS))
scale = (amax / QMAX).clamp(min=1e-8)
print("  scale = %.6f / %d = %.8f" % (amax.item(), QMAX, scale.item()))
print("  校验：最大那个数除以 scale 应该正好等于 %d" % QMAX)
print("         %.6f / %.8f = %.4f" % (amax.item(), scale.item(),
                                        (amax / scale).item()))

print("\n[步骤 3] q = round(x / scale)，然后 clamp 到 [%d, %d]" % (QMIN, QMAX))
ratio = x / scale
q = torch.round(ratio)
print("  取整之后：  ", [int(v) for v in q.tolist()])
q = torch.clamp(q, QMIN, QMAX)
print("  clamp 之后：", [int(v) for v in q.tolist()])
print("  ^ 注意 %d 和 %d 不对称：负方向多一个码字。" % (QMIN, QMAX))
print("    因为对称量化的 0 必须落在网格上（round(0)=0），所以正负只能各占一半。")

print("\n[步骤 4] xq = q * scale —— 这一步叫「反量化 / fake quantization」")
xq = q * scale
show(x, q, xq, scale)

print("\n  MSE  = %.8f" % ((xq - x) ** 2).mean().item())
print("  SQNR = %.2f dB" % sqnr(x, xq))
print("\n  记住这条：fake quant 的输入输出都是浮点张量，")
print("  中间那一步压到整数网格再还原。PyTorch 里能量精度损失就靠这个。")


# ==================================================================
# 第二段：为什么权重必须 per-channel
# ==================================================================
hr("第二段  同上，但换成 4 个「通道」，其中通道 0 的量级只有通道 2 的 1/100")

w = torch.tensor([
    [0.10, -0.20,  0.30,  0.05],     # 通道 0：小
    [1.00, -1.50,  0.80,  1.20],
    [10.0, -12.0,  9.00, 11.0],     # 通道 2：大
    [0.50, -0.70,  0.60,  0.90],
])

print("\n[先看每通道的量级]")
for c in range(w.shape[0]):
    print("  通道 %d  absmax = %8.4f" % (c, w[c].abs().max().item()))

print("\n[per-tensor：整张共用一个 scale，被最大那个通道绑架]")
s_pt = (w.abs().max() / QMAX).clamp(min=1e-8)
q_pt = torch.clamp(torch.round(w / s_pt), QMIN, QMAX)
xq_pt = q_pt * s_pt
for c in range(w.shape[0]):
    nz = int((q_pt[c] != 0).sum())
    rel = ((xq_pt[c] - w[c]).abs().max() / w[c].abs().max()).item()
    print("  通道 %d  scale=%8.4f  非零码字 %d/4  相对误差 %.3f"
          % (c, s_pt.item(), nz, rel))
print("  ^ 通道 0 的四个数全是 0 —— 它的信息被完全抹掉了。")
print("    原因不是它太小，是 scale 被通道 2 的 absmax=%.1f 撑大了。" % w[2].abs().max().item())

print("\n[per-channel：每个通道自己算 scale]")
flat = w.reshape(w.shape[0], -1)
per_ch = flat.abs().max(dim=1).values / QMAX
per_ch = per_ch.clamp(min=1e-8)
shape = [-1] + [1] * (w.dim() - 1)
s_pc = per_ch.reshape(shape)
print("  每通道 scale =", [round(v, 4) for v in per_ch.tolist()])
q_pc = torch.clamp(torch.round(w / s_pc), QMIN, QMAX)
xq_pc = q_pc * s_pc
for c in range(w.shape[0]):
    nz = int((q_pc[c] != 0).sum())
    rel = ((xq_pc[c] - w[c]).abs().max() / w[c].abs().max()).item()
    print("  通道 %d  scale=%8.4f  非零码字 %d/4  相对误差 %.3f"
          % (c, per_ch[c].item(), nz, rel))
print("  ^ 通道 0 活下来了。tests/test_quant.py 里那条判据就是这一行。")

# ==================================================================
# 第三段：为什么激活要用非对称
# ==================================================================
hr("第三段  ReLU 之后全是非负数，为什么对称量化会白丢一半码字")

a = torch.rand(2000) ** 2 * 5.0
print("\n[数据] 非负，右偏")
print("  min = %.6f   max = %.6f   （min >= 0，说明负半边一个数据都没有）"
      % (a.min().item(), a.max().item()))

print("\n[对称量化] scale = absmax / %d，码字范围虽然是 [%d, %d]，" % (QMAX, QMIN, QMAX))
print("  但实际用到的码字全在 >= 0 的那一半 —— 负半边白白浪费")
s_sym = (a.abs().max() / QMAX).clamp(min=1e-8)
q_sym = torch.clamp(torch.round(a / s_sym), QMIN, QMAX)
xq_sym = q_sym * s_sym
mse_sym = ((xq_sym - a) ** 2).mean().item()
print("  实际用到的不同码字 = %d 个" % torch.unique(q_sym).numel())

print("\n[非对称量化] scale = (max-min) / %d，zero_point 把区间平移上去" % QAS)
lo, hi = a.min(), a.max()
s_as = ((hi - lo) / QAS).clamp(min=1e-8)
zp = torch.clamp(torch.round(-lo / s_as), 0, QAS)
print("  scale = (%.6f - %.6f) / %d = %.8f" % (hi.item(), lo.item(), QAS, s_as.item()))
print("  zero_point = round(-lo / scale) = %d" % zp.item())
print("  ^ zero_point 是整数，因为码字 0 必须代表 lo 那个值。")
q_as = torch.clamp(torch.round(a / s_as) + zp, 0, QAS)
xq_as = (q_as - zp) * s_as
mse_as = ((xq_as - a) ** 2).mean().item()
print("  实际用到的不同码字 = %d 个" % torch.unique(q_as).numel())

print("\n[对比]")
print("  %-14s %-12s %-14s %-10s" % ("方式", "码字利用率", "MSE", "SQNR"))
print("  %-14s %-12s %-14.8f %-10.2f"
      % ("对称", "%d/%d" % (torch.unique(q_sym).numel(), QAS + 1), mse_sym, sqnr(a, xq_sym)))
print("  %-14s %-12s %-14.8f %-10.2f"
      % ("非对称", "%d/%d" % (torch.unique(q_as).numel(), QAS + 1), mse_as, sqnr(a, xq_as)))
print("  SQNR 提升 %.2f dB —— 这就是 tests 里 e_asym > e_sym + 1.0 的来源"
      % (sqnr(a, xq_as) - sqnr(a, xq_sym)))


# ==================================================================
# 第四段：min/max 被离群点带偏 —— 为什么还需要 scale_mse / scale_entropy
# ==================================================================
hr("第四段  一个离群点能把 99.9% 的值挤进几个码字里")


def sym_quant(b, bits):
    """演示用的对称量化，和你要写的第一个函数同一条公式。"""
    s = (b.abs().max() / (2 ** (bits - 1) - 1)).clamp(min=1e-8)
    q = torch.clamp(torch.round(b / s), -2 ** (bits - 1), 2 ** (bits - 1) - 1)
    return q, q * s, s


b = torch.randn(20000)
q0, xq0, s0 = sym_quant(b, BITS)
print("\n[干净数据]")
print("  absmax = %.4f   scale = %.4f   SQNR = %.2f dB"
      % (b.abs().max().item(), s0.item(), sqnr(b, xq0)))

b[0] = 50.0
b[1] = -48.0
q1, xq1, s1 = sym_quant(b, BITS)
print("\n[加两个离群点 50.0 / -48.0 之后]")
print("  absmax = %.4f   scale = %.4f   SQNR = %.2f dB"
      % (b.abs().max().item(), s1.item(), sqnr(b, xq1)))
print("  用了 %d 个码字（4bit 对称共 %d 个取值）"
      % (torch.unique(q1).numel(), QMAX - QMIN + 1))

body = b[2:]
qb, xqb, sb = sym_quant(body, BITS)
print("\n  去掉离群点、只对剩下 %d 个数重新定 scale：" % body.numel())
print("    scale 从 %.4f 降到 %.4f" % (s1.item(), sb.item()))
print("    SQNR 从 %.2f dB 升到 %.2f dB" % (sqnr(b, xq1), sqnr(body, xqb)))
print("  ^ 这就是你要填的那几个函数要解决的问题：")
print("    前三个管「网格怎么定」，后两个管「离群点怎么绕开」。")


hr("你现在应该能回答这三个问题（答不出就别去写代码）")
print("""
  1. 为什么对称量化 clamp 的上界是 7 而不是 15？8 个码字去哪了？
     -> 因为 0 必须落在网格上，round(0) 恒等于 0，正负各占一半。

  2. per-tensor 的时候，通道 0 的四个数为什么全变成 0？
     -> scale 被通道 2 的 absmax 撑到 12.0/7，通道 0 的值除完不足半格，
        round 出来全是 0。信息不是「小到看不见」，是被别人挤没了。

  3. 非对称量化里 zero_point 起什么作用？为什么它必须是整数？
     -> 让码字 0 对应数据 lo，区间从 [lo,hi] 平移到 [0,15]，
        于是 [-min,0) 这半边原本浪费的码字也能用上。
        必须是整数，因为「码字」本身就是整数网格上的编号。

  答完这三个，再去填 src/quant.py 的 quantize_per_tensor_sym。
""")
