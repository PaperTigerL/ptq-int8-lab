#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验 7：组合实验 —— 三条正交路线叠加。这是论文主表的雏形。

三条路线为什么正交：
  A. 激活定标策略   —— 改的是激活的 scale
  B. 混合精度       —— 改的是每层的位宽
  C. AdaRound       —— 改的是取整方向
三者作用在不同自由度上，互不冲突，可以叠加。

对照组（本实验的输出就是这张表）：
  1. W8A8  RTN                     参考上限
  2. W4A8  RTN                     最朴素基线
  3. W4A8  + AdaRound              单路线 C
  4. 混合 W4/W8 + RTN              单路线 B
  5. 混合 + AdaRound               B + C        ★ 主推
  6. 混合 + AdaRound + A4 entropy   A + B + C    极限压缩

外加两档补充：
  · 位宽档位对比（3/8、4/8、4/6），看哪档性价比最高
  · 校准集稳健性（换 3 个 seed 重跑第 5 项），报告精度波动范围
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import config as C
import src.model as M
import src.eval as E
import src.calib as CB
import src.sens as S
import src.adaround as AD
import src.quant as Q
from src.data import make_calib_loader


def log(m):
    print(m, flush=True)


def pick_keep(model, sens, max_avg_bits, low, high):
    """在「平均位宽 <= max_avg_bits」的约束下，选能保住的最多 8bit 层数。

    ⚠️ 2026-10-17 新增。旧版 e7 写死 KEEP=10（那是 e5 在 CIFAR-10 + ResNet-18 上的最优值）。
    实测翻车：CIFAR-100 + ResNet-18 上同样 KEEP=10，avg_bits 却飙到 6.28 ——
    **因为 avg_bits 是按参数量加权的，不是按层数**。敏感度分布一变，
    同样层数保护的参数量完全不同，混合精度就失去了压缩意义。

    `config.MIXED_MAX_AVG_BITS`（默认 5.0）这个约束此前在 e7 里完全没被使用。
    现在改为：**先扫约束，再定 KEEP**，保证四组之间位宽口径可比。
    """
    n = len(sens)
    best_k = 0
    best_bits = None
    for k in range(0, n + 1):
        per_layer = S.split_by_sensitivity(sens, k, low, high)
        ab = S.avg_bits(model, per_layer)
        if ab <= max_avg_bits + 1e-9:
            best_k = k
            best_bits = ab
        else:
            break
    if best_bits is None:
        # 连 k=0 都超限（理论上不该发生，除非默认位宽就太高）
        best_k, best_bits = 0, low
        per_layer = S.split_by_sensitivity(sens, 0, low, high)
        best_bits = S.avg_bits(model, per_layer)
    return best_k, best_bits


def main():
    LOW, HIGH = 4, 8

    log("=" * 74)
    log("实验 7 · 组合实验（混合精度 × AdaRound × 激活校准）")
    log("=" * 74)
    log("设备     : %s" % C.device_info())
    log("量化算子 : %s" % Q.which_mode())
    log("")

    torch.manual_seed(C.CALIB_SEED)
    m = M.load_fp32(C.CKPT_MAIN).to(C.DEVICE)

    torch.manual_seed(C.CALIB_SEED)
    m = M.load_fp32(C.CKPT_MAIN).to(C.DEVICE)
    M.fold_all_bn(m)
    base = E.evaluate(m)
    log("[0] FP32 基线 : %.2f%%" % (base * 100))

    # ---------- 敏感度 ----------
    log("")
    log("[1] 逐层敏感度扫描 ...")
    t0 = time.time()
    sens, _ = S.layer_sensitivity(m, bits=LOW, base_acc=base)
    # ⚠️ KEEP 不再写死：在 MIXED_MAX_AVG_BITS 约束下自动选（见 pick_keep 的说明）
    KEEP, avg_bits = pick_keep(m, sens, C.MIXED_MAX_AVG_BITS, LOW, HIGH)
    per_layer = S.split_by_sensitivity(sens, KEEP, LOW, HIGH)
    log("    完成 %.0fs；在平均位宽 <= %.1f 的约束下，自动选 KEEP = %d"
        % (time.time() - t0, C.MIXED_MAX_AVG_BITS, KEEP))
    log("    本次配置：保 %d 层 %dbit，其余 %dbit；平均权重位宽 = %.2f"
        % (KEEP, HIGH, LOW, avg_bits))

    # ---------- 激活校准 ----------
    log("")
    log("[2] 激活校准")
    store = CB.collect_activations(m, make_calib_loader())
    a8_s, a8_z = CB.compute_a_params(store, 8, strategy="minmax")
    a4_s, a4_z = CB.compute_a_params(store, 4, strategy="entropy")
    log("    A8 minmax / A4 entropy 均已完成（%d 层）" % len(a8_s))

    # ---------- AdaRound ----------
    # ⚠️ 2026-10-17 修正一个口径错配的 bug（此前两天的 e7 数据全部作废）：
    #   QuantConv2d.forward 里权重量化走的是 **quantize_per_channel_sym**（硬编码 per-channel），
    #   但 adaround_model 默认走 **per-tensor**。
    #   → AdaRound 用一套口径产出权重，却被拿去和另一套口径的 RTN 对照，
    #     模型等于被装上"口径不匹配的权重"，精度会大幅崩掉。
    #   实证：MobileNetV2 上 e6（--per-channel）得 81.91%，e7（默认 per-tensor）只有 63.74%。
    #   **修法：e7 必须显式传 per_channel=True，与 e2/e5/e6 全平台口径一致。**
    log("")
    log("[3] AdaRound 逐层优化（%d bit，全网 %d 层，per-channel 与全平台一致）..."
        % (LOW, 20))
    t0 = time.time()
    m_partial = M.build_quant_model(m, w_bits=LOW, a_bits=8,
                                    a_scales=a8_s, a_zeros=a8_z).to(C.DEVICE)
    real_acts = AD.collect_real_activations(m_partial, batch_size=AD.OFFICIAL["batch_size"])
    del m_partial
    torch.cuda.empty_cache()
    ada_all, mses = AD.adaround_model(m, real_acts, bits=LOW, verbose=True,
                                       per_channel=True)
    log("    %d 层完成，耗时 %.0fs（%.1fs/层），层均重建 = %.6f"
        % (len(ada_all), time.time() - t0,
           (time.time() - t0) / max(len(ada_all), 1),
           sum(mses) / max(len(mses), 1)))
    log("    口径：iters=%d（官方默认），per-channel=True，lm_head 不参与量化"
        % AD.OFFICIAL["iters"])

    low_only = {n for n, b in per_layer.items() if b == LOW}
    ada_low = {k: v for k, v in ada_all.items() if k in low_only}
    log("    其中 low-bit 层 %d 个（供配置 5/6 使用）" % len(ada_low))

    # ---------- 六个配置实测 ----------
    log("")
    log("[4] 六配置实测")
    log("    %-28s %9s %11s %10s" % ("配置", "accuracy", "相对FP32", "平均位宽"))
    log("    " + "-" * 62)

    results = {}

    def run(tag, w_bits, a_bits, per_layer_w, a_scales, a_zeros,
            ada=None, avg_bits_override=None):
        mq = M.build_quant_model(m, w_bits=w_bits, a_bits=a_bits,
                                 per_layer_w=per_layer_w,
                                 a_scales=a_scales, a_zeros=a_zeros).to(C.DEVICE)
        if ada:
            AD.apply_weights(mq, ada)
        acc = E.evaluate(mq)
        ab = avg_bits_override if avg_bits_override is not None else (
            S.avg_bits(m, per_layer_w) if per_layer_w else float(w_bits))
        results[tag] = {"acc": acc, "avg_bits": ab}
        log("    %-28s %8.2f%% %+10.2f%% %9.2f"
            % (tag, acc * 100, (acc - base) * 100, ab))
        del mq
        torch.cuda.empty_cache()
        return acc

    run("1. W8A8 RTN", 8, 8, None, a8_s, a8_z)
    run("2. W4A8 RTN", 4, 8, None, a8_s, a8_z)
    run("3. W4A8 + AdaRound", 4, 8, None, a8_s, a8_z, ada=ada_all)
    run("4. 混合 W4/W8 RTN", LOW, 8, per_layer, a8_s, a8_z)
    acc_bc = run("5. 混合 + AdaRound", LOW, 8, per_layer, a8_s, a8_z, ada=ada_low)
    run("6. 混合+AdaRound+A4 entropy", LOW, 4, per_layer, a4_s, a4_z, ada=ada_low)
    log("    " + "-" * 62)

    # ---------- 位宽档位对比（RTN，快速粗筛） ----------
    log("")
    log("[5] 位宽档位对比（RTN，用于判断下一步往哪档挖）")
    log("    %-16s %9s %10s" % ("(低位,高位)", "accuracy", "平均位宽"))
    log("    " + "-" * 40)
    ladder = []
    for lo, hi in [(3, 8), (4, 8), (4, 6), (6, 8)]:
        pl = S.split_by_sensitivity(sens, KEEP, lo, hi)
        ab = S.avg_bits(m, pl)
        mq = M.build_quant_model(m, w_bits=lo, a_bits=8,
                                 per_layer_w=pl,
                                 a_scales=a8_s, a_zeros=a8_z).to(C.DEVICE)
        acc = E.evaluate(mq)
        ladder.append({"low": lo, "high": hi, "acc": acc, "avg_bits": ab})
        log("    (%d,%d)%-11s %8.2f%% %9.2f" % (lo, hi, "", acc * 100, ab))
        del mq
        torch.cuda.empty_cache()
    log("    " + "-" * 40)

    # ---------- 校准集稳健性 ----------
    log("")
    log("[6] 校准集稳健性（换 seed 重跑第 5 项，看结论稳不稳）")
    robust = []
    for sd in (2026, 1, 42):
        torch.manual_seed(sd)
        ld = make_calib_loader(seed=sd)
        st = CB.collect_activations(m, ld)
        s8, z8 = CB.compute_a_params(st, 8, strategy="minmax")
        mq = M.build_quant_model(m, w_bits=LOW, a_bits=8,
                                 per_layer_w=per_layer,
                                 a_scales=s8, a_zeros=z8).to(C.DEVICE)
        AD.apply_weights(mq, ada_low)
        a = E.evaluate(mq)
        robust.append({"seed": sd, "acc": a})
        log("    calib seed %-6d %.2f%%" % (sd, a * 100))
        del mq
        torch.cuda.empty_cache()
    accs = [r["acc"] for r in robust]
    log("    波动范围 : %.2f%% ~ %.2f%%  (极差 %.2f%%)"
        % (min(accs) * 100, max(accs) * 100, (max(accs) - min(accs)) * 100))
    log("    注：测试集固定，所以这里量的是**校准集**带来的不确定性，")
    log("        不是测试集噪声（后者标准误约 0.33%，见手册 4.6）。")

    # ---------- 结论 ----------
    log("")
    log("[7] 结论")
    w4 = results["2. W4A8 RTN"]["acc"]
    w8 = results["1. W8A8 RTN"]["acc"]
    log("    最朴素基线 W4A8     : %.2f%%  (平均 4.00 bit)" % (w4 * 100))
    log("    组合方案 B+C        : %.2f%%  (平均 %.2f bit)" % (
        acc_bc * 100, results["5. 混合 + AdaRound"]["avg_bits"]))
    log("    提升                : %+.2f%%" % ((acc_bc - w4) * 100))
    log("")
    log("    相对 W8A8（8.00 bit）: %+.2f%%   <- 用一半的位宽换取这个差距"
        % ((acc_bc - w8) * 100))
    spread = (max(accs) - min(accs)) * 100
    if abs(acc_bc - w4) * 100 > max(0.33, spread):
        log("    判定：提升 %.2f%% 大于噪声尺度（校准波动 %.2f%%，测试集标准误 0.33%%），"
            % ((acc_bc - w4) * 100, spread))
        log("           **这个结论是稳的，可以写进论文。**")
    else:
        log("    判定：提升幅度未明显超过噪声尺度，需要更多验证才能下结论。")

    E.save_result("e7_combo", {
        "exp": "e7_combo",
        "device": C.DEVICE,
        "quant_mode": Q.which_mode(),
        "fp32_acc": base,
        "keep": KEEP, "low": LOW, "high": HIGH,
        "avg_bits": avg_bits,
        "ada_avg_recon_mse": sum(mses) / max(len(mses), 1),
        "results": results,
        "ladder": ladder,
        "robust": robust,
        "spread_pct": spread,
    })
    log("")
    log("结果已存 results/")
    log("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
