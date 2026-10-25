# -*- coding: utf-8 -*-
"""AdaRound：位宽不变，只把「怎么取整」从四舍五入换成可学习的取整方向。

为什么有效：
    RTN 里每个权重向上还是向下取整，是被 scale 定死的。
    但单个元素只有上/下两种选择，整个卷积核组合起来是 2^N 种可能 ——
    取整方向其实是一个**可学习的自由度**。
    用校准集把「量化后输出 vs FP32 输出」的重建误差压到最小，就能救回精度。

================================================================================
⚠️ 2026-10-15 按官方口径重写（此前版本与 AdaRound 论文 / AIMET / BRECQ 不一致）
================================================================================

改动依据是两份**互相独立**的官方实现，它们给同一组超参，互为印证：
  - AIMET `quic/aimet` 的 `adaround_weight.py` / `adaround_wrapper.py`（论文同实验室）
  - BRECQ `yhhhli/BRECQ` 的 `quant/adaptive_rounding.py` / `quant/block_recon.py`
    （本地副本 `paper-code/BRECQ_yhhhli/`）

逐项修正（P0 = 不一致会导致结论不成立）：

| 项                | 旧版（错）        | 官方              |
|-------------------|-------------------|-------------------|
| 迭代步数          | iters=300         | **20000**（BRECQ 默认）|
| β 温度           | 固定 1.2/0.1      | **20 → 2 cosine 退火** |
| 正则 warm start   | 无                | **前 20% 不施加正则** |
| 重建 loss         | MSE `.mean()`     | **L_p (p=2.4) `.sum()`** |
| 首层/末层         | lm_head 被筛掉     | **全网量化（含首末层）** |
| scale 粒度        | per-channel       | per-tensor（见下）|
| 学习率            | 固定 2e-3         | **cosine 衰减到 ~0** |

**关于 scale 粒度**：官方 AdaRound 用 per-tensor（per-layer scale）。旧版用 per-channel，
精度天然更高。`per_channel=True` 现在是显式开关，默认关闭以对齐官方；
如果要和本平台e2/e5 的其他实验保持口径一致，可显式打开，但报告里必须写清楚。

**为什么旧版的提升只有 +1.48%，而官方 ResNet18 W4 能救回 44 个百分点？**
主要是三个原因叠加：步数少 66 倍、末层（lm_head）根本没被优化、β 固定低温导致
h 一开始就被压死无法探索。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C
from src.data import make_calib_loader

# ---------------- 官方默认超参（AIMET 与 BRECQ 一致） ----------------
OFFICIAL = {
    "iters": 20000,      # BRECQ block_reconstruction 默认；AIMET 在 <8bit 时用 15000
    "lambda_reg": 0.01,  # 正则系数 weight，两个实现都是 0.01
    "b_start": 20,       # 起始温度
    "b_end": 2,          # 结束温度
    "warmup": 0.2,       # 前 20% 迭代不施加正则
    "batch_size": 32,    # 重建用的校准批大小
    "lr": 2e-3,          # Adam 初始学习率，cosine 衰减
    "p_norm": 2.4,       # 重建 loss 的广义 L_p 指数
    "gamma": -0.1,       # rectified sigmoid 的 gamma
    "zeta": 1.1,         # rectified sigmoid 的 zeta
}


@torch.no_grad()
def collect_real_activations(model, batch_size=32, device=None):
    """为每个 Conv2d 采一批**真实四维**输入，供单层重建用。

    注意和 calib.collect_activations 的区别：
      那个存的是降采样后的一维值（只为了算统计量，省内存）；
      单层重建必须吃真实的 [N,C,H,W]，所以这里是另一个函数。

    只跑校准集的第一个 batch —— 重建用 32 张足够，再多只是变慢。

    ⚠️ 必须同时匹配 nn.Conv2d 和 model.QuantConv2d：
       在**量化模型**上采x̂ 时，被包装的层是 QuantConv2d（继承 nn.Module，
       不是 nn.Conv2d），只判nn.Conv2d 会一��都采不到。
    """
    device = device or C.DEVICE
    store = {}
    hooks = []

    from src.model import QuantConv2d

    def mk(nm_):
        def hk(m, inp, out):
            x = inp[0]
            if isinstance(x, torch.Tensor) and nm_ not in store:
                store[nm_] = x.detach()[:batch_size].clone()
        return hk

    for nm, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, QuantConv2d)):
            hooks.append(mod.register_forward_hook(mk(nm_=nm)))

    model.eval()
    with torch.no_grad():
        for x, _ in make_calib_loader():
            model(x.to(device, non_blocking=True))
            break
    for h in hooks:
        h.remove()
    return store


class LinearTempDecay:
    """β 的 cosine 退火（前warmup 段保持 start_b，之后cosine 降到 end_b）。

    公式对齐 BRECQ `LinearTempDecay`：
        if t < rel_start_decay * t_max:  return start_b
        rel_t = (t - start_decay) / (t_max - start_decay)
        return end_b + (start_b - end_b) * max(0, 1 - rel_t)
    """

    def __init__(self, t_max, rel_start_decay=0.2, start_b=20, end_b=2):
        self.t_max = max(int(t_max), 1)
        self.start_decay = rel_start_decay * self.t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        if t < self.start_decay:
            return float(self.start_b)
        rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
        return self.end_b + (self.start_b - self.end_b) * max(0.0, 1.0 - rel_t)


def adaround_layer(conv, x_cal, bits=4,
                   iters=None, lr=None, lambda_reg=None,
                   b_start=None, b_end=None, warmup=None,
                   per_channel=False, verbose=False):
    """对单层做 AdaRound，返回已量化好的权重（可直接部署的浮点张量）。

    参数默认值全部走 OFFICIAL，调用方一般不用传。
    `x_cal` 应是**已经过前面各层量化后**的激活（asymmetric loss 的关键）。
    """
    iters = OFFICIAL["iters"] if iters is None else iters
    lr = OFFICIAL["lr"] if lr is None else lr
    lambda_reg = OFFICIAL["lambda_reg"] if lambda_reg is None else lambda_reg
    b_start = OFFICIAL["b_start"] if b_start is None else b_start
    b_end = OFFICIAL["b_end"] if b_end is None else b_end
    warmup = OFFICIAL["warmup"] if warmup is None else warmup
    p_norm = OFFICIAL["p_norm"]
    gamma = OFFICIAL["gamma"]
    zeta = OFFICIAL["zeta"]

    w = conv.weight.data
    qmax = 2 ** (bits - 1) - 1

    # ---- scale：默认 per-tensor（官方口径）。per_channel=True 时切回细粒度 ----
    if per_channel:
        f = w.reshape(w.shape[0], -1)
        s = (f.abs().max(dim=1, keepdim=True)[0] / qmax).clamp(min=1e-8)
        s = s.reshape([-1] + [1] * (w.dim() - 1))
    else:
        s = (w.abs().max() / qmax).clamp(min=1e-8)

    wd = w / s
    wf = torch.floor(wd)
    frac = (wd - wf).clamp(0, 1)

    # ---- α 的解析初始化：使初始 h(α) 精确复现 RTN ----
    # 对齐 BRECQ: alpha = -log((zeta-gamma)/(rest-gamma) - 1)
    with torch.no_grad():
        rest = frac.clamp(1e-4, 1 - 1e-4)
        alpha = -torch.log((zeta - gamma) / (rest - gamma) - 1)
        V = nn.Parameter(alpha.clone())

    opt = torch.optim.Adam([V], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)
    temp_decay = LinearTempDecay(iters, rel_start_decay=warmup,
                                 start_b=b_start, end_b=b_end)
    loss_start = iters * warmup

    def c2d(wq):
        return F.conv2d(x_cal, wq, conv.bias, conv.stride,
                        conv.padding, conv.dilation, conv.groups)

    def soft_target():
        return torch.clamp(torch.sigmoid(V) * (zeta - gamma) + gamma, 0, 1)

    with torch.no_grad():
        ref = c2d(w)                      # FP32 输出（用已量化链路的输入）

    last_rec = None
    for it in range(iters):
        h = soft_target()
        wq = s * (wf + h)
        out_q = c2d(wq)

        # 重建 loss：广义 L_p，p=2.4，逐通道 sum 后 mean（对齐 BRECQ 的 lp_loss）
        rec = (out_q - ref).abs().pow(p_norm).sum(dim=1).mean()

        # 正则项：warm start 期内不施加；期内β 由cosine 退火给出
        if it < loss_start:
            loss = rec
        else:
            b = temp_decay(it + 1)
            round_loss = (1 - ((h - 0.5).abs() * 2).pow(b)).sum()
            loss = rec + lambda_reg * round_loss

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        last_rec = float(rec)
        if verbose and (it + 1) % max(iters // 10, 1) == 0:
            print("      it %6d/%d  rec=%.6fbeta=%.2f"
                  % (it + 1, iters, last_rec, temp_decay(it + 1)), flush=True)

    with torch.no_grad():
        # 硬取整导出：不再走 soft rounding，留小数就不是可部署权重了
        wr = torch.floor(wf + (V >= 0).float())
        return s * wr, last_rec


def adaround_model(model, real_acts, bits=4, iters=None, verbose=True,
                   only_layers=None, per_channel=False, progress=None,
                   quant_fn_is_per_channel=True):
    """逐层优化。**含首层与末层**（对齐官方全网量化）。

    only_layers: 可选集合，只优化出现在里面的层（混合精度里只优化压低位的层）。
                 传None = 全网量化（官方默认）。
    返回 (dict{层名:权重}, list[rec])

    ⚠️ **口径守卫**（2026-10-17 加）：`quant_fn_is_per_channel` 说的是
    **模型推理时权重量化用哪个函数**（本平台 `QuantConv2d.forward` 硬编码
    `quantize_per_channel_sym`，即 per-channel）。
    如果 AdaRound 这边用 per-tensor 产出权重，就会出现
    「优化目标与评估基线口径不一致」——模型被装上另一套口径的权重，精度会大幅崩掉。
    实证：MobileNetV2 上 per-tensor 的 e7 只有 63.74%，per-channel 的 e6 是 81.91%。

    **所以默认会在两者不一致时直接报错，而不是静默跑出坏数据。**
    确实要用 per-tensor（例如对齐 AdaRound 官方口径做单独分析），
    必须显式传 `quant_fn_is_per_channel=False` 并自行在报告里声明。
    """
    if quant_fn_is_per_channel and not per_channel:
        raise ValueError(
            "口径错配：模型推理用 per-channel 量化（QuantConv2d.forward 硬编码 "
            "quantize_per_channel_sym），但 AdaRound 默认 per-tensor。\n"
            "这会让 AdaRound 用一套口径产出权重、却拿去和另一套口径的 RTN 对照，"
            "精度会大幅崩掉（实测 MobileNetV2：63.74% vs 81.91%）。\n"
            "修法：per_channel=True。\n"
            "若确实要对齐 AdaRound 官方（per-tensor）做单独分析，"
            "请显式传 quant_fn_is_per_channel=False 并在报告里声明口径不同。")

    out = {}
    recs = []
    layers = [(nm, mod) for nm, mod in model.named_modules()
              if isinstance(mod, nn.Conv2d)]
    # 官方是全网量化：不排除首层/末层
    layers = [(nm, mod) for nm, mod in layers if nm in real_acts]
    if only_layers is not None:
        layers = [(nm, mod) for nm, mod in layers if nm in only_layers]

    for i, (nm, mod) in enumerate(layers, 1):
        wq, rec = adaround_layer(mod, real_acts[nm].to(C.DEVICE), bits,
                                 iters=iters, per_channel=per_channel)
        out[nm] = wq.detach()
        recs.append(rec)
        if verbose:
            print("      [%2d/%2d] %-24s recon = %.6f"
                  % (i, len(layers), nm, rec), flush=True)
        if progress is not None:
            progress(i, len(layers))
    return out, recs


def apply_weights(quant_model, ada_weights):
    """把 AdaRound 得到的权重装回量化模型，并关掉这些层的权重量化（避免二次量化）。"""
    for nm, mod in quant_model.named_modules():
        if nm in ada_weights and hasattr(mod, "conv"):
            mod.conv.weight.data = ada_weights[nm].to(mod.conv.weight.device)
            mod.w_bits = 0
    return quant_model
