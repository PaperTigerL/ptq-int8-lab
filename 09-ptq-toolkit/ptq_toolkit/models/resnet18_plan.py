# -*- coding: utf-8 -*-
"""ResNet-18 的 blocks 声明（blocks/PTQ2 导出用，也供 PTQW 兼容）。

basic block：conv1(3×3) → conv2(3×3) → + identity

⚠️ **layer1 没有 downsample** —— CIFAR 版 ResNet-18 的 stem 是
   3×3s1（不是 ImageNet 的 7×7s2），进layer1 时通道已是 64、stride 已是 1，
   所以不需要投影。实测全网只有 3 个 downsample（layer2/3/4 的 block 0）。
   给 layer1.0 也加会 KeyError。
"""
STAGES = {"layer1": 2, "layer2": 2, "layer3": 2, "layer4": 2}
# 有 downsample 的 stage（layer1 没有）
DS_STAGES = {"layer2", "layer3", "layer4"}


def topo_plan() -> list:
    plan = []
    for lname, n in STAGES.items():
        for bi in range(n):
            plan.append({
                "convs": [f"{lname}.{bi}.conv1", f"{lname}.{bi}.conv2"],
                "downsample": (f"{lname}.{bi}.downsample.0"
                               if (bi == 0 and lname in DS_STAGES) else None),
                "stride": (2 if (bi == 0 and lname in DS_STAGES) else 1),
            })
    return plan
