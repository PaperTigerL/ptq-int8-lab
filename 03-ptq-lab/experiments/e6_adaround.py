#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验 6：AdaRound —— 位宽不变，只换「怎么取整」。

RTN 的做法是 round(w / scale)，每个元素向上还是向下，是被 scale 定死的。
AdaRound 的洞察是：取整方向其实是一个**可学习的自由度**。
单个元素只有上/下两种选择，但整个卷积核组合起来是 2^N 种可能，
用校准集把「量化后输出 vs FP32 输出」的重建误差压到最小，就能救回精度。

================================================================================
⚠️ 2026-10-15 按官方口径重写
================================================================================
依据两份互相独立的官方实现（超参一致，互为印证）：
  - AIMET `quic/aimet`：论文同实验室的 AdaRound
  - BRECQ `yhhhli/BRECQ`：本地副本 `paper-code/BRECQ_yhhhli/`

**修掉的 5 个偏差（不修则结论不可信）**：

| 项 | 旧版 | 官方 |
|---|------|------|
| 迭代步数 | 300 | **20000** |
| β 温度 | 固定 1.2/0.1 | **20 → 2 cosine 退火** |
| 正则 warm start | 无 | **前 20% 不施加正则** |
| 重建 loss | MSE `.mean()` | **L_p (p=2.4) `.sum()`** |
| asymmetric loss | 否（FP 分支吃干净输入） | **是**（见下） |

**关于 asymmetric loss**：官方让 loss 衡量「误差在整条量化链路上的传播」——
FP 分支和量化分支吃**同一个已量化链路的输入 x̂**。
旧版用未量化的干净输入做参考，等于只测单层孤立误差，低估了误差传播。

**⚠️ 两处与官方不可比，必须在报告里声明**（不是 bug，是平台框架的既成事实）：
1. **lm_head（nn.Linear）从不参与量化** —— build_quant_model 只包装 nn.Conv2d。
   官方是全网量化含末层。**这会让本实验的精度偏乐观。**
2. **scale 粒度**：官方 AdaRound 用 per-tensor。本平台默认 per-tensor
   （--per-channel 可切到细粒度，但就不能声称对齐官方了）。
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

import config as C
import src.model as M
import src.eval as E
import src.calib as CB
import src.quant as Q
import src.adaround as AR
from src.data import make_calib_loader


def log(m):
    print(m, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=AR.OFFICIAL["iters"],
                    help="每层迭代步数（官方 20000）")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--per-tensor", action="store_true",
                    help="改用 per-tensor scale（AdaRound 官方口径，但与本平台其余实验不可比）")
    ap.add_argument("--official-baseline", action="store_true",
                    help="配 --per-tensor 使用：承认与平台其余实验口径不同")
    args = ap.parse_args()
    per_channel = not args.per_tensor
    if args.per_tensor and not args.official_baseline:
        log("[!] 你要跑 per-tensor（AdaRound 官方口径）。")
        log("    注意：本平台其余实验（e2/e5/e7）都用 per-channel，两边不可比。")
        log("    若确认要跑，请加 --official-baseline。")
        return 2

    BITS = args.bits
    log("=" * 72)
    log("实验 6 · AdaRound 取整优化（W%dA8，官方口径）" % BITS)
    log("=" * 72)
    log("设备     : %s" % C.device_info())
    log("量化算子 : %s" % Q.which_mode())
    log("超参     : iters=%d  λ=%.3g  β=(%g→%g)  warmup=%.0f%%  batch=%d  L_p=%.1f"
        % (args.iters, AR.OFFICIAL["lambda_reg"], AR.OFFICIAL["b_start"],
           AR.OFFICIAL["b_end"], AR.OFFICIAL["warmup"] * 100,
           AR.OFFICIAL["batch_size"], AR.OFFICIAL["p_norm"]))
    log("scale    : %s" % ("per-tensor（AdaRound 官方口径，与平台其余实验不可比）"
                             if not per_channel else "per-channel（与 e2/e5/e7 一致）"))
    log("")

    torch.manual_seed(C.CALIB_SEED)
    m = M.load_fp32(C.CKPT_MAIN).to(C.DEVICE)
    M.fold_all_bn(m)
    base = E.evaluate(m)
    log("[0] FP32 基线 : %.2f%%" % (base * 100))

    # ---- 激活校准统一 A8 ----
    log("")
    log("[1] 激活校准（A8）")
    store = CB.collect_activations(m, make_calib_loader())
    a_s, a_z = CB.compute_a_params(store, 8, strategy="minmax")

    # ---- RTN 对照 ----
    mq = M.build_quant_model(m, w_bits=BITS, a_bits=8,
                             a_scales=a_s, a_zeros=a_z).to(C.DEVICE)
    acc_rtn = E.evaluate(mq)
    del mq
    torch.cuda.empty_cache()
    log("    RTN W%dA8 : %.2f%%  (%+.2f%%)" % (
        BITS, acc_rtn * 100, (acc_rtn - base) * 100))

    # ---- ⭐ asymmetric loss 的关键：FP 参考分支必须走「已量化的前面各层」 ----
    # 先建一个 W{BITS}A8 的量化模型，用它跑前向、hook 采到的就是
    # 「误差已在链路中传播过」的激活 x̂。重建时 FP 分支与量化分支都吃它。
    log("")
    log("[2] 采集已量化链路的激活 x̂（asymmetric loss 的前提）")
    m_partial = M.build_quant_model(m, w_bits=BITS, a_bits=8,
                                    a_scales=a_s, a_zeros=a_z).to(C.DEVICE)
    x_hat = AR.collect_real_activations(m_partial, batch_size=AR.OFFICIAL["batch_size"])
    del m_partial
    torch.cuda.empty_cache()
    log("    已采集 %d 层的 x̂" % len(x_hat))

    # ---- AdaRound ----
    log("")
    log("[3] 逐层 AdaRound 优化（iters=%d，共 %d 层）..." % (args.iters, len(x_hat)))
    t0 = time.time()
    ada_w, recs = AR.adaround_model(
        m, x_hat, bits=BITS, iters=args.iters, verbose=True,
        per_channel=per_channel,
        quant_fn_is_per_channel=per_channel)
    dt = time.time() - t0
    log("    优化完成 %d 层，耗时 %.0fs（%.1fs/层），层均 recon = %.6f" % (
        len(ada_w), dt, dt / max(len(ada_w), 1),
        sum(recs) / max(len(recs), 1)))

    # ---- 组装：权重用 AdaRound 结果（不再二次量化），激活 A8 ----
    m_final = M.build_quant_model(m, w_bits=0, a_bits=8,
                                  a_scales=a_s, a_zeros=a_z).to(C.DEVICE)
    AR.apply_weights(m_final, ada_w)
    acc_ada = E.evaluate(m_final)
    log("")
    log("[4] AdaRound W%dA8 : %.2f%%  (%+.2f%%)" % (
        BITS, acc_ada * 100, (acc_ada - base) * 100))
    log("    相对 RTN 提升    : %+.2f%%" % ((acc_ada - acc_rtn) * 100))
    log("")
    log("    ⚠️ 口径声明：lm_head（nn.Linear）不参与量化，")
    log("       官方 AdaRound 是全网量化含末层，故本数字偏乐观，不可直接与官方对比。")

    E.save_result("e6_adaround", {
        "exp": "e6_adaround",
        "device": C.DEVICE,
        "quant_mode": Q.which_mode(),
        "bits": BITS,
        "official_aligned": True,
        "iters": args.iters,
        "lambda_reg": AR.OFFICIAL["lambda_reg"],
        "b_start": AR.OFFICIAL["b_start"],
        "b_end": AR.OFFICIAL["b_end"],
        "warmup": AR.OFFICIAL["warmup"],
        "p_norm": AR.OFFICIAL["p_norm"],
        "asymmetric_loss": True,
        "scale_granularity": "per_channel" if per_channel else "per_tensor",
        "lm_head_quantized": False,
        "fp32_acc": base,
        "rtn_acc": acc_rtn,
        "adaround_acc": acc_ada,
        "gain": acc_ada - acc_rtn,
        "avg_recon": sum(recs) / max(len(recs), 1),
        "elapsed_sec": dt,
    })
    log("")
    log("结果已存 results/")
    log("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
