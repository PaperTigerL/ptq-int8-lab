#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验 3：激活校准策略对比 —— 同样位宽，换个定 scale 的方法能救回多少。

为什么值得单独做：
  min/max 定 scale 会被离群点带偏。一个异常大的激活值就能把 scale 拉大，
  导致剩下 99.9% 的值全挤在少数几个码字里，等于白丢精度。

在 A4 上比，不在 A8 上比 —— e2 已经证明 A8 只掉 0.05%，天花板太近看不出差别。

mse / entropy 两档会调用你在 quant.py 里填的 scale_mse / scale_entropy。
没填的话这两档会抛 TodoNotImplemented，其余档照常跑。
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
import src.quant as Q
from src.data import make_calib_loader
from src._todo import TodoNotImplemented


def log(m):
    print(m, flush=True)


def main():
    A_BITS = 4
    log("=" * 72)
    log("实验 3 · 激活校准策略对比（固定 W8，激活压到 A%d）" % A_BITS)
    log("=" * 72)
    log("设备     : %s" % C.device_info())
    log("量化算子 : %s" % Q.which_mode())
    log("")

    torch.manual_seed(C.CALIB_SEED)
    m = M.load_fp32(C.CKPT_MAIN).to(C.DEVICE)
    M.fold_all_bn(m)
    base = E.evaluate(m)
    log("[0] FP32 基线 : %.2f%%" % (base * 100))

    calib_loader = make_calib_loader()
    log("")
    log("[1] 采集校准激活（只采一次，四档策略共用）...")
    t0 = time.time()
    store = CB.collect_activations(m, calib_loader)
    log("    %d 层，%.1fs" % (len(store), time.time() - t0))

    STRATS = [("minmax", {}),
              ("percentile", {"percentile": 0.999}),
              ("mse", {}),
              ("entropy", {})]

    log("")
    log("[2] 各策略实测")
    log("    %-12s %10s %12s %10s" % ("策略", "accuracy", "相对FP32", "vs minmax"))
    log("    " + "-" * 48)

    res = {}
    for name, kw in STRATS:
        try:
            t1 = time.time()
            s, z = CB.compute_a_params(store, A_BITS, strategy=name, **kw)
            mq = M.build_quant_model(m, w_bits=8, a_bits=A_BITS,
                                     a_scales=s, a_zeros=z).to(C.DEVICE)
            acc = E.evaluate(mq)
            res[name] = acc
            extra = ""
            if "minmax" in res and name != "minmax":
                extra = "%+9.2f%%" % ((acc - res["minmax"]) * 100)
            log("    %-12s %9.2f%% %+11.2f%% %10s   (%.0fs)" % (
                name, acc * 100, (acc - base) * 100, extra, time.time() - t1))
            if acc < base - 0.10:
                log("       ⚠️ 该档掉点 %.2f%%，超过 10%% —— 这不是正常的校准差异。"
                    % ((base - acc) * 100))
                log("          大概率是某层的截断上界算得太小，把那层激活压没了。")
                log("          用 tools/quant_by_hand.py 或逐层 dump 截断上界定位。")
            del mq
            torch.cuda.empty_cache()
        except TodoNotImplemented as e:
            log("    %-12s 跳过 —— %s" % (name, e))
        except Exception as e:
            log("    %-12s 失败 —— %s: %s" % (name, type(e).__name__, e))

    log("    " + "-" * 48)

    log("")
    log("[3] 读法")
    if "minmax" in res and len(res) > 1:
        best = max(res.items(), key=lambda kv: kv[1])
        log("    最优策略 : %s  %.2f%%" % (best[0], best[1] * 100))
        log("    相对最朴素的 minmax 提升 %+.2f%%" % ((best[1] - res["minmax"]) * 100))
        log("    这个数就是「改校准策略」这条路的全部收益上限。")
        log("    如果它很小，说明瓶颈不在激活定标，别在这条路上耗时间。")
    log("")
    log("    提醒：本实验只改激活定标，权重固定 W8。")
    log("          真正的掉点大头在权重侧（见 e2），两者别混。")

    E.save_result("e3_act_calib", {
        "exp": "e3_act_calib",
        "device": C.DEVICE,
        "quant_mode": Q.which_mode(),
        "a_bits": A_BITS,
        "fp32_acc": base,
        "results": res,
    })
    log("")
    log("结果已存 results/")
    log("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
