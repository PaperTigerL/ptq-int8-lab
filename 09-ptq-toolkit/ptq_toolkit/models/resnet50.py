# -*- coding: utf-8 -*-
"""ResNet-50（CIFAR-10 32×32 适配版）。

与 ImageNet 版的三处差别 —— 都是为了让 32×32 输入下不塌到 1×1：
  1. conv1 用 **3×3 stride 1**（ImageNet 是 7×7 stride 2）
  2. conv1 之后**不加 maxpool**（ImageNet 有 maxpool(3,3,2,1)）
     否则 32×32 → 16×16 → 8×8，到 layer2 就没东西可卷了
  3. fc 输入维度按实际特征图算，不硬编码 512

结构是 bottleneck（每个 block 三层conv）：
  conv1 1×1 reduce → conv2 3×3 → conv3 1×1 expand，残差相加后 ReLU。
  与 ResNet-18 的两层结构不同，所以**不能复用 06 的 blocks 拓扑格式**，
  走 PTQX 指令序列（见 export/exporter.py）。

本文件只负责「定义网络」。拓扑提取是通用代码，不在这里。
"""
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 4

    def __init__(self, cin, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, planes, 1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity)


class ResNet50(nn.Module):
    """CIFAR 适配的 ResNet-50。32×32 输入，输出 512 维特征。"""

    # stage 的 (层数, 输出通道倍数)，沿用 ImageNet 的 [3,4,6,3]
    STAGES = [(3, 64), (4, 128), (6, 256), (3, 512)]

    def __init__(self, num_classes: int = 10):
        super().__init__()
        # CIFAR 适配点 1：3×3 stride 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # CIFAR 适配点 2：stem 后**不加 maxpool**
        self.maxpool = nn.Identity()

        cin = 64
        blocks = []
        for n, planes in self.STAGES:
            for i in range(n):
                stride = 2 if (i == 0 and planes != 64) else 1
                planes_out = planes * BasicBlock.expansion
                down = None
                # 通道不变且 stride=1 时不需要 downsample；
                # ResNet-18 里「第一层也不需要」的直觉在这里不适用，
                # 因为 bottleneck 的 expansion=4，通道永远在变
                if stride != 1 or cin != planes_out:
                    down = nn.Sequential(
                        nn.Conv2d(cin, planes_out, 1, stride=stride, bias=False),
                        nn.BatchNorm2d(planes_out))
                blocks.append(BasicBlock(cin, planes, stride, down))
                cin = planes_out
        self.layer1 = nn.Sequential(*blocks[:3])
        self.layer2 = nn.Sequential(*blocks[3:7])
        self.layer3 = nn.Sequential(*blocks[7:13])
        self.layer4 = nn.Sequential(*blocks[13:16])

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


def build(num_classes: int = 10, **_) -> ResNet50:
    """registry 用的构造入口。额外 kwargs 忽略（保持 loader 签名统一）。"""
    return ResNet50(num_classes=num_classes)


def expected_topology():
    """预期拓扑规模，写进 test_models.py 做断言。

    stem:1 + 16 个 block × 3 conv + 4 个 downsample + fc:1
    """
    n_conv = 1 + sum(n * 3 for n, _ in STAGES) + 4
    return dict(n_conv=n_conv, n_block=sum(n for n, _ in STAGES),
                has_stem_pool=False, stem_conv="3x3s1")



def topo_plan() -> list:
    """残差块声明（blocks/PTQ2 导出用）。

    每个条目：
      convs       块内主支路的 conv 层名，**按执行顺序**
      downsample  identity 分支的 conv（None = 直连）
      stride      块的主支路第一个 conv 的 stride

    bottleneck 结构：conv1(1×1) → conv2(3×3) → conv3(1×1) → + identity

    ⚠️ 只有每个 stage 的**第一个 block** 有 downsample ——
       虽然 expansion=4，但`planes*4` 恰好等于下一个 block 的输入通道，
       通道不变且 stride=1 时不需要 downsample。实测全网只有 4 个。
    """
    plan = []
    n_per = {"layer1": 3, "layer2": 4, "layer3": 6, "layer4": 3}
    for lname, n in n_per.items():
        for bi in range(n):
            plan.append({
                "convs": [f"{lname}.{bi}.conv1",
                          f"{lname}.{bi}.conv2",
                          f"{lname}.{bi}.conv3"],
                "downsample": (f"{lname}.{bi}.downsample.0" if bi == 0
                               else None),
                "stride": 1 if bi == 0 and lname == "layer1" else (
                    2 if bi == 0 else 1),
            })
    return plan
