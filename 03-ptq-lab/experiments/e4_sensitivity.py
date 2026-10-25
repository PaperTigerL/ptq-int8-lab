#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验 4：逐层量化敏感度 —— 找出哪些层压不得。

做法：每次只把一层压到 W4，其余层全保持 FP32，测精度掉多少。
掉得多的就是硬骨头，混合精度里必须给它保高位宽。

这是整条链路上承上启下的一步：
  e2 告诉你「该在权重侧使劲」，
  e4 告诉你「具体是哪几个层」，
  e5 才能据此做出真正有效的混合精度配置。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import config as C
import src.model as M
import src.eval as E
import src.sens as S
import src.quant as Q


def log(m):
    print(m, flush=True)


def main():
    log("=" * 72)
    log("实验 4 · 逐层量化敏感度（每层单独压到 W4，其余保持 FP32）")
    log("=" * 72)
    log("设备     : %s" % C.device_info())
    log("量化算子 : %s" % Q.which_mode())
    log("")

    torch.manual_seed(C.CALIB_SEED)
    m = M.load_fp32(C.CKPT_MAIN).to(C.DEVICE)
    M.fold_all_bn(m)
    base = E.evaluate(m)
    log("[0] FP32 基线 : %.2f%%" % (base * 100))
    log("")

    log("[1] 逐层扫描（20 层，每层一次完整测试集推理）...")
    t0 = time.time()
    sens, base = S.layer_sensitivity(m, bits=4, base_acc=base)
    log("    扫描完成，耗时 %.0fs" % (time.time() - t0))
    log("")

    log("[2] 敏感度排序（delta 越大 = 越不能压）")
    log("    %-24s %10s %12s" % ("层名", "accuracy", "掉点"))
    log("    " + "-" * 48)
    for d in sens:
        log("    %-24s %9.2f%% %+11.2f%%" % (d["name"], d["acc"] * 100, -d["delta_pct"]))
    log("    " + "-" * 48)

    top = sens[-5:]
    bot = sens[:5]
    log("")
    log("[3] 结论")
    log("    最扛不住的 5 层（混合精度里保高位）:")
    for d in reversed(top):
        log("      %-24s 掉 %5.2f%%" % (d["name"], d["delta_pct"]))
    log("    最扛得住的 5 层（可以往死里压）:")
    for d in bot:
        log("      %-24s 掉 %5.2f%%" % (d["name"], d["delta_pct"]))

    spread = sens[-1]["delta_pct"] - sens[0]["delta_pct"]
    log("")
    log("    敏感度极差 = %.2f%%  <- 这个数越大，混合精度的收益空间越大" % spread)
    if spread < 0.5:
        log("    注意：极差太小，说明各层差不多，混合精度可能捞不到多少。")

    E.save_result("e4_sensitivity", {
        "exp": "e4_sensitivity",
        "device": C.DEVICE,
        "quant_mode": Q.which_mode(),
        "fp32_acc": base,
        "bits": 4,
        "spread_pct": spread,
        "sens": sens,
    })
    log("")
    log("结果已存 results/（e5 会读它）")
    log("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
