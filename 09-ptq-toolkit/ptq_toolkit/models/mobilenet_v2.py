# -*- coding: utf-8 -*-
"""MobileNetV2 —— **标准 torchvision 结构**（t=6，width_mult=1.0），CIFAR 3×3s1 stem。

与 torchvision `mobilenet_v2` 的**唯一差别**：
  `features.0.0` 的 stride 从 **2 改成 1**（CIFAR 是 32×32，不是 224×224）。
  其余 17 个 inverted residual 块、通道数、层名、BN 位置**完全一致** ——
  这是为了能用 torchvision 的 ImageNet 权重移植（见 tools/load_transfer.py）。

⚠️ 为什么必须改成标准结构（M2 的 t=1 版本为什么不行）：
   M2 那版用 t=1（无 expand conv）+ `classifier=nn.Linear`，
   与 torchvision 的 t=6 + `classifier=Sequential(Dropout, Linear)` 键集合完全不同
   （torchvision 的 stem 叫 `features.0.0`，t=1 版里没有这个键）。
   强行移植只会得到 KeyError，而不是一条能读懂的失败原因。
   代价：参数从 0.18M 变成约 3.5M —— 但换来的是**有出处的起点权重**，
   而不是随机初始化。对精度链路来说这个交换是必须的。

结构（每个 inverted residual 块，t=6 时三个 conv 全在）：
  conv 0 : expand    1×1   Cin → 6·Cin
  conv 1 : depthwise 3×3  6·Cin → 6·Cin，groups = 6·Cin
  conv 2 : project   1×1   6·Cin → Cout（**线性输出**，无 ReLU）
  skip    : 仅当 stride==1 且 Cin==Cout 时存在，加在 project 之前

前向：
  features.0 (stem) → features.1..17 (17 个 block) → features.18 (1×1 投影)
    → adaptive_avg_pool2d → flatten → classifier.1 (Linear 1280 → num_classes)
"""
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_divisible(v: float, divisor: int = 8) -> int:
    """torchvision 的 round-to-multiple-of-8。width_mult=1.0 时恒等，
    保留是为了与 torchvision 逐行对齐（将来改 width_mult 不出偏差）。"""
    if v < divisor:
        return int(v + 0.5)
    return int(v / divisor + 0.5) * divisor


class ConvBNReLU6(nn.Sequential):
    """Conv2d(bias=False) → BN → ReLU6 —— 与 torchvision 的
    `Conv2dNormActivation` 逐行同构。

    ⚠️ `act=False` 时**只放 Conv + BN 两层**，不是放个 `nn.Identity()` 占位：
       那样子层名会变成 `0/1/2`，而 torchvision 只有 `0/1`
       （`activation_layer=None` 时它压根不 append），
       键集合立刻错位一层 —— 实测症状是 missing 里出现
       `features.1.2.0.weight` 这种「本该有的 key 反而缺失」。
    """

    def __init__(self, inp: int, oup: int, kernel: int, stride: int,
                 groups: int = 1, act: bool = True):
        padding = (kernel - 1) // 2
        layers = [nn.Conv2d(inp, oup, kernel, stride, padding, groups=groups,
                            bias=False),
                  nn.BatchNorm2d(oup)]
        if act:
            layers.append(nn.ReLU6(inplace=True))
        super().__init__(*layers)


class InvertedResidual(nn.Module):
    """三个子模块（t=6 时全在），层名与 torchvision **逐字一致**：
      self.conv[0] = expand    Conv2dNormActivation 1×1 → ReLU6
      self.conv[1] = depthwise Conv2dNormActivation 3×3 stride，groups=hidden → ReLU6
      self.conv[2] = 裸 nn.Conv2d 1×1（**不是** Sequential 子类）
      self.conv[3] = 裸 nn.BatchNorm2d
    → state_dict 键：`features.N.conv.0.0.weight` / `.conv.1.0` /
      `.conv.2.weight` / `.conv.3.*`，与 torchvision 完全相同。

    ⚠️ project 必须是「裸 Conv2d + 裸 BN」而不能包进 ConvBNReLU6：
       那会把键变成 `features.N.conv.2.0.weight`（多一层），
       torchvision 里 project 的 conv 直接叫 `conv.2`、BN 叫 `conv.3`。
       实测症状：missing 里出现一堆 `features.N.2.0.weight`，
       unexpected 里出现 `features.N.conv.2.0.weight` —— 一眼看不出多在哪层。

    ⚠️ 本类必须是 nn.Module 而不是 nn.Sequential：
       子层统一叫 `conv`（torchvision 就是这样），由 forward 自己编排
       `expand → dw → project`，use_res 时加 identity。
    """

    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int):
        super().__init__()
        if stride not in (1, 2):
            raise ValueError(f"stride 只能是 1 或 2，得到 {stride}")
        self.stride = stride
        self.use_res = (stride == 1 and inp == oup)
        hidden_dim = int(round(inp * expand_ratio))

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU6(inp, hidden_dim, kernel=1, stride=1))
        layers.extend([
            ConvBNReLU6(hidden_dim, hidden_dim, kernel=3, stride=stride,
                        groups=hidden_dim),
            # pw-linear：裸 Conv + 裸 BN（无激活）—— 与 torchvision 一致
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        if self.use_res:
            return x + out
        return out


class MobileNetV2(nn.Module):
    """标准 MobileNetV2（t=6, width_mult=1.0），仅 stem stride 为 1（CIFAR）。"""

    # (t, 输出通道, 重复次数, stride) —— **torchvision inverted_residual_setting 原值**。
    # ⚠️ 第一项是 [1, 16, 1, 1]：**t=1**（没有 expand conv）！
    #    这是 MobileNetV2 的原版设计（论文 Table 1 第一行），不是笔误。
    #    写成全 6 会让 features.1 多一个 expand 1×1，键集合立刻错位
    #    （实测 size mismatch: 源 (32,1,3,3) vs 我的 (192,32,1,1)）。
    SETTINGS = [
        (1, 16, 1, 1),
        (6, 24, 2, 2),
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]
    EXPAND = 6              # 默认 t（仅作文档用途，实际按 SETTINGS 每项取）
    LAST_CHANNEL = 1280
    FIRST_CHANNEL = 32
    N_BLOCK = 17            # 1+2+3+4+3+3+1
    PROJ_IDX = 18           # features.18 = 1×1 投影

    def __init__(self, num_classes: int = 10, width_mult: float = 1.0):
        super().__init__()
        if width_mult != 1.0:
            raise NotImplementedError(
                f"width_mult={width_mult} 未实现（M2 只用 1.0）；"
                f"要支持得同步 _make_divisible 与 last_channel 的处理")

        first = _make_divisible(self.FIRST_CHANNEL * width_mult)
        last = _make_divisible(self.LAST_CHANNEL * max(1.0, width_mult))

        # CIFAR 适配点：stem 3×3 **stride 1**（torchvision 是 2）
        self.features = nn.Sequential()
        self.features.add_module("0", ConvBNReLU6(3, first, kernel=3,
                                                  stride=1))

        cin = first
        blk_idx = 1
        for t, cout, n, stride in self.SETTINGS:
            oup = _make_divisible(cout * width_mult)
            for i in range(n):
                s = stride if i == 0 else 1
                self.features.add_module(
                    str(blk_idx), InvertedResidual(cin, oup, s, expand_ratio=t))
                cin = oup
                blk_idx += 1

        # 1×1 投影到 last_channel（torchvision 的 features.18）
        self.features.add_module(
            str(self.PROJ_IDX), ConvBNReLU6(cin, last, kernel=1, stride=1))

        # 分类头：Sequential(Dropout, Linear) —— classifier.1 对应 torchvision
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)


def build(num_classes: int = 10, **_) -> MobileNetV2:
    """registry 用的构造入口。"""
    return MobileNetV2(num_classes=num_classes)


# ---------------------------------------------------------------- 拓扑声明

def topo_plan() -> List[dict]:
    """残差块声明（PTQ2 导出用）。

    每个 inverted residual 块是**三个 conv**（t=6 时全在）：
      conv 0 = expand    1×1
      conv 1 = depthwise 3×3
      conv 2 = project   1×1
    skip 仅当 `stride == 1 and inp == oup` 时存在。

    ⚠️ `features.0`（stem）**不在 plan 里** —— PTQ2 的 `build_blocks` 会把它
       识别为 stem（不在任何 block 中的第一个 conv）并先执行。

    ⚠️ `features.18`（1×1 投影到 1280）**必须显式进 plan**，
       用一个 `_no_residual` 的「伪块」表达：
       漏了它 replay 只会跑到 320 通道，fc 报
       "mat1 and mat2 shapes cannot be multiplied (2x64 and 512x10)"
       —— 错误信息完全指不到「少跑了一个 conv」。
       `build_blocks` 现在会拦住这种情况并列出漏掉的层名。
    """
    plan = []
    cin = MobileNetV2.FIRST_CHANNEL
    for si, (t, cout, n, stride) in enumerate(MobileNetV2.SETTINGS):
        for i in range(n):
            s = stride if i == 0 else 1
            use_res = (s == 1 and cin == cout)
            blk = 1 + sum(v[2] for v in MobileNetV2.SETTINGS[:si]) + i
            # ⚠️ 层名必须是 `_conv_list` 里的**真实 Conv2d 名**：
            #    ConvBNReLU6 是 Sequential，所以它的 conv 叫
            #    `features.N.conv.0.0`（比 state_dict 键多一层 `.0`）。
            #    写 `features.N.conv.0` 会与 state_dict 键一致但**匹配不到**
            #    `_conv_list` 的条目，于是被判成「plan 之外的 conv」而漏跑。
            # t=1 的块没有 expand —— 只有 dw + project 两个 conv
            if t == 1:
                cs = [f"features.{blk}.conv.0.0",   # dw 3×3
                      f"features.{blk}.conv.1"]     # project 1×1（裸 Conv2d）
            else:
                cs = [f"features.{blk}.conv.0.0",   # expand 1×1
                      f"features.{blk}.conv.1.0",   # dw 3×3
                      f"features.{blk}.conv.2"]     # project 1×1
            entry = {
                "convs": cs,
                "downsample": None,
                "stride": s,
                "_no_residual": (not use_res),
                # ⭐ PTQ2 激活语义（与真实前向逐处对齐，见 topoblocks 模块头）：
                #   project（块内最后一个 conv）是**线性输出**，不接激活；
                #   expand / dw 接 ReLU6（由顶层 _act 决定值为 2）。
                "_last_relu": False,
            }
            if use_res:
                # 残差块：x + project(x) 相加后**无**激活（不是 ReLU）。
                entry["_post_relu"] = False
            plan.append(entry)
            cin = cout

    # features.18：1×1 投影到 1280 —— 不是残差块，但必须执行。
    # 与 17 个 inverted residual 不同：它的末 conv **接 ReLU6**（ConvBNReLU6），
    # 所以**不设 _last_relu=False**（无残差块默认末 conv 激活）。
    plan.append({
        "convs": [f"features.{MobileNetV2.PROJ_IDX}.0"],  # ConvBNReLU6 的 conv
        "downsample": None,
        "stride": 1,
        "_no_residual": True,
    })
    # ⭐ 全模型激活类型 = ReLU6：plan 顶层（首项）带 _act，export_blocks 读出
    #   后传给 build_blocks；不写死模型名做 if/else。
    plan[0]["_act"] = "relu6"
    return plan


def expected_topology():
    """预期拓扑规模，写进 test_models.py 做断言。

    conv 数（t=1 首块只有 2 个，其余 16 块各 3 个，外加 stem 与末尾投影）：
      1 (stem) + 2 (features.1, t=1) + 16×3 (features.2..17) + 1 (features.18)
      = 52

    有残差的块数 = stride==1 且 Cin==Cout 的块。
    实测 10 个：features.2 的第二个块起、以及 stride=1 的 96 stage 全部 3 个。
    """
    n_block = sum(v[2] for v in MobileNetV2.SETTINGS)
    n_expand = sum(v[2] for v in MobileNetV2.SETTINGS[1:])   # 除首块外的块都有 expand
    n_conv = 1 + (n_block - n_expand) * 2 + n_expand * 3 + 1

    cin = MobileNetV2.FIRST_CHANNEL
    n_res = 0
    for _t, cout, n, stride in MobileNetV2.SETTINGS:
        for i in range(n):
            s = stride if i == 0 else 1
            if s == 1 and cin == cout:
                n_res += 1
            cin = cout
    return dict(n_conv=n_conv, n_block=n_block, has_stem_pool=False,
                stem_conv="3x3s1", n_residual=n_res,
                first_block_t=MobileNetV2.SETTINGS[0][0])
