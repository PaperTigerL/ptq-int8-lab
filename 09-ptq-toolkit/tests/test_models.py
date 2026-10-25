# -*- coding: utf-8 -*-
"""模型测试：每个模型的前向形状、参数量、拓扑自检。

覆盖三件事（M2 验收判据第 6 条）：
  1. 前向输出形状 = (batch, 10)
  2. 参数量落在预期量级（防止结构被改坏而不自知）
  3. **拓扑自检 max|diff| == 0** —— 两种格式都测：
       · ResNet-18 走 PTQW（blocks，与 06 引擎兼容）
       · 三个模型都走 PTQX（seq，通用格式）

为什么两种格式都要测：PTQX 是新写的，如果只在 ResNet-50/MobileNetV2 上跑通，
不能排除它在 ResNet-18 上也成立 —— 反之亦然。全测一遍才算覆盖。

用法：python tests/test_models.py
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from ptq_toolkit.export.exporter import export_weights      # noqa: E402
from ptq_toolkit.export.topoblocks import export_blocks     # noqa: E402
from ptq_toolkit.models import get_model, topo_kind       # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'✓' if cond else '✗'} {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILED.append(name)


# ---------------------------------------------------------------- 预期值
# 参数量按 CIFAR 适配版结构算，容差 ±5% —— 目的是发现「结构被改坏了」，
# 而不是卡死精确值（不同 PyTorch 版本的 BN 参数计数可能差几十个）。
# 参数量按各模型定义算，容差 ±5% —— 目的是发现「结构被改坏了」。
# ⚠️ MobileNetV2 在 M2r S1 从 t=1 重写为**标准 t=6**：
#    参数 0.18M → 2.24M，conv 35 → 52。这是为了能接 torchvision 权重。
EXPECTED = {
    "resnet18": dict(params=11.17e6, n_conv=20, n_block=8),
    "resnet50": dict(params=23.52e6, n_conv=53, n_block=16),
    # ⚠️ n_block 数的是**模型里的 InvertedResidual 实例**（17 个），
    #    而 topo_plan 有 18 条 —— 多出的一条是 features.18 投影层的
    #    「伪块」（_no_residual）。两者是不同口径，别混。
    "mobilenet_v2": dict(params=2.237e6, n_conv=52, n_block=17),
}


def plan_for(key):
    """取该模型的 PTQX 残差声明。"""
    if key == "resnet18":
        from ptq_toolkit.models import resnet18_plan
        return resnet18_plan.topo_plan()
    from ptq_toolkit.models import registry
    return registry.topo_plan(key)


def main():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 32, 32)
    print("模型测试：三模型 × 前向/参数/拓扑自检\n" + "-" * 62)

    for key, exp in EXPECTED.items():
        print(f"\n{key}")
        m = get_model(key).eval()

        # ---- 1) 前向形状 ----
        with torch.no_grad():
            y = m(x)
        check(f"前向输出 {tuple(y.shape)} == (2, 10)", tuple(y.shape) == (2, 10))

        # ---- 2) 参数量 & conv 数 ----
        n_param = sum(p.numel() for p in m.parameters())
        n_conv = sum(1 for _, mm in m.named_modules() if isinstance(mm, nn.Conv2d))
        rel = abs(n_param - exp["params"]) / exp["params"]
        check(f"参数量 {n_param / 1e6:.2f}M（预期 {exp['params'] / 1e6:.2f}M）",
              rel < 0.05, f"偏差 {rel * 100:.1f}%")
        check(f"conv 数 {n_conv} == {exp['n_conv']}", n_conv == exp["n_conv"])
        n_blk = sum(1 for _, mm in m.named_modules()
                    if type(mm).__name__ in ("BasicBlock", "InvertedResidual"))
        check(f"block 数 {n_blk} == {exp['n_block']}", n_blk == exp["n_block"],
              f"（BasicBlock/InvertedResidual 计数）")

        # 拓扑导出必须用**折叠后**的模型 —— 导出格式只含 conv 权重+bias，
        # 不含 BN 参数。参数量的检查在折叠前做（BN 也有参数）。
        from ptq_toolkit.pipeline import load_ptq_lab
        mf = load_ptq_lab()["fold_all_bn"](get_model(key)).eval()

        # ---- 3a) PTQ2 blocks 拓扑自检（三个模型都测）----
        try:
            r = export_blocks(mf, HERE.parent / "results" / f"_t_{key}_q2.bin",
                              x, plan_for(key))
            check(f"PTQ2 自检 max|diff| = {r['selfcheck_maxdiff']:.3g} == 0",
                  r["selfcheck_maxdiff"] < 1e-4,
                  f"conv={r['n_conv']} block={r['n_block']} "
                  f"max_convs/block={r['max_convs_per_block']}")
        except Exception as e:
            check("PTQ2 自检", False, str(e)[:110])

        # ---- 3b) PTQW blocks 拓扑自检（仅 ResNet-18，它用06 的格式）----
        if topo_kind(key) == "blocks":
            try:
                r2 = export_weights(mf, HERE.parent / "results" / f"_t_{key}_blk.bin", x)
                check(f"PTQW 自检 max|diff| = {r2['selfcheck_maxdiff']:.3g} == 0",
                      r2["selfcheck_maxdiff"] < 1e-4,
                      f"conv={r2['n_conv']} block={r2['n_block']}")
            except Exception as e:
                check("PTQW 自检", False, str(e)[:110])
        else:
            check("跳过 PTQW（该模型用 PTQX）", True)

    print("\n" + "-" * 62)
    if FAILED:
        print(f"✗ {len(FAILED)} 项失败：{', '.join(FAILED)}")
        return 1
    print("✓ 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
