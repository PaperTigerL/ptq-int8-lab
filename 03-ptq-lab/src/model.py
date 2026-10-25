# -*- coding: utf-8 -*-
"""模型加载、BN 折叠、量化模型构造。

两件事值得单独说清楚，因为它们是最容易做错、也最容易蒙混过关的地方。

一、BN 折叠是 PTQ 的必要前置，不是可选项
    推理时 BN 用的是训练期统计好的 running_mean / running_var，
    它已经退化成一个固定的仿射变换：z = A*y + B。
    既然是固定的线性变换，就可以等价地吸进前一层卷积的 weight / bias。
    不折叠的后果：卷积输出先被量化、再被 BN 缩放平移，
    量化网格和 BN 的统计量互相打架，掉点会凭空多出好几个百分点，
    而且你会误以为是「量化方法不行」—— 其实是前置步骤漏了。

二、激活量化参数必须离线校准，不能每次 forward 现算
    现算（动态量化）看着省事，但真实部署不是这么干的：
    chip 上不会为每个 batch 重算一次 scale。
    正确做法是用校准集跑一遍 FP32 前向，把每层激活的范围统计下来，
    固化成 a_scale / a_zero_point。也只有这么做，
    「minmax vs MSE vs 百分位 vs KL」这几种校准策略才比得出高下。
"""
import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, mobilenet_v2

import config as C
import src.quant as Q


# ---------------------------------------------------------------- 加载

def build_fp32(num_classes=None):
    """按 C.ARCH 建一个未训练模型（num_classes 由 C.NUM_CLASSES 推导）。

    ResNet-18 用 torchvision 默认 stem（7x7 stride2 + maxpool，11.18M 参数）。
    **这是有意与原平台保持一致的** —— 换了 stem 就换了模型，跨数据集对比会失去意义。
    """
    n = C.NUM_CLASSES if num_classes is None else num_classes
    if C.ARCH == "resnet18":
        return resnet18(num_classes=n)
    if C.ARCH == "mobilenetv2":
        return mobilenet_v2(num_classes=n)
    raise ValueError("未知模型 %r" % C.ARCH)


def load_fp32(ckpt=None, num_classes=None):
    """加载已训练好的基线权重（C.CKPT_MAIN），失败时给出明确原因。"""
    ckpt = ckpt or C.CKPT_MAIN
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            "缺基线权重：%s\n先跑 tools/train_baseline.py --dataset %s --arch %s"
            % (ckpt, C.DATASET, C.ARCH))
    model = build_fp32(num_classes)
    sd = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()
    return model


# ---------------------------------------------------------------- BN 折叠

def fold_bn_into_conv(conv, bn):
    """把 BatchNorm2d 的仿射变换吸进前面的 Conv2d。

    数学：BN(y) = gamma*(y-mean)/sqrt(var+eps) + beta = A*y + B
          A = gamma / sqrt(var+eps)
          B = beta - A*mean
    吸收后：W' = W * A（按 out_channel 广播），b' = b*A + B
    """
    assert isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d)
    gamma = bn.weight.data
    beta = bn.bias.data
    mean = bn.running_mean.data
    var = bn.running_var.data
    std = torch.sqrt(var + bn.eps)
    A = gamma / std
    B = beta - A * mean

    w = conv.weight.data
    w.mul_(A.reshape(-1, 1, 1, 1))
    if conv.bias is None:
        conv.bias = nn.Parameter(
            torch.zeros(w.shape[0], device=w.device, dtype=w.dtype))
    conv.bias.data.mul_(A)
    conv.bias.data.add_(B)
    return conv


def fold_all_bn(model):
    """递归遍历整棵树，把每对相邻的 (Conv2d, BatchNorm2d) 折叠，BN 换成正直通。

    ResNet-18 里 conv1/bn1、每个 BasicBlock 的 conv1/bn1 与 conv2/bn2、
    以及 downsample 里的 conv/bn，都会被折掉。
    残差相加发生在 BN 之后，所以两个分支各自折叠互不影响，数学上等价。
    """
    model.eval()

    children = list(model.named_children())
    for _, child in children:
        if len(list(child.children())) > 0:
            fold_all_bn(child)

    if isinstance(model, nn.Sequential):
        items = list(model)
        for i in range(len(items) - 1):
            if isinstance(items[i], nn.Conv2d) and isinstance(items[i + 1], nn.BatchNorm2d):
                fold_bn_into_conv(items[i], items[i + 1])
                model[i + 1] = nn.Identity()
        return model

    names = [n for n, _ in children]
    for i in range(len(names) - 1):
        a = getattr(model, names[i])
        b = getattr(model, names[i + 1])
        if isinstance(a, nn.Conv2d) and isinstance(b, nn.BatchNorm2d):
            fold_bn_into_conv(a, b)
            setattr(model, names[i + 1], nn.Identity())
    return model


def count_bn(model):
    return sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))


# ---------------------------------------------------------------- 量化包装

class QuantConv2d(nn.Module):
    """带量化功能的卷积包装。

    严格复用原 conv 的 stride / padding / dilation / groups ——
    硬编码 padding=1 是上一轮踩过的坑（ResNet 的 conv1 和 downsample
    的 padding 不一样，硬编码会让输出形状直接错掉）。

    w_bits=0 表示该层不量化（用于混合精度里保留敏感层）。
    激活量化用校准得到的固定 a_scale / a_zero（静态量化）。

    a_symmetric 选择激活的码空间：
        False（默认）：非对称，码字 [0, 2^bits - 1]，zero_point 平移。
        True        ：对称，码字 [-(2^(bits-1)), 2^(bits-1) - 1]，zp 恒为 0。
    ⚠️ 两者不能只靠改 scale 来切换：非对称的 clamp 下界是 0，
       直接灌对称 scale 会把所有负值截成 0，必须换整个 clamp 区间。
    """

    def __init__(self, conv, w_bits=0, a_bits=0, a_scale=None, a_zero=None,
                 a_symmetric=False):
        super().__init__()
        self.conv = conv
        self.w_bits = int(w_bits)
        self.a_bits = int(a_bits)
        self.a_symmetric = bool(a_symmetric)
        if a_scale is None:
            a_scale = torch.tensor(1.0)
        if a_zero is None:
            a_zero = torch.tensor(0.0)
        self.register_buffer("a_scale", torch.as_tensor(a_scale).float())
        self.register_buffer("a_zero", torch.as_tensor(a_zero).float())

    def forward(self, x):
        w = self.conv.weight
        if self.w_bits:
            w = Q.quantize_per_channel_sym(w, self.w_bits)
        if self.a_bits:
            s, z = self.a_scale, self.a_zero
            if self.a_symmetric:
                # 对称：0 必须落在网格上，上界是 2^(bits-1) - 1 而不是 2^bits - 1。
                # zp 恒为 0，所以 (q - z) * s 退化为 q * s。
                qmax = 2 ** (self.a_bits - 1) - 1
                qmin = -(qmax + 1)
                q = torch.round(x / s).clamp(qmin, qmax)
            else:
                qmax = 2 ** self.a_bits - 1
                q = (torch.round(x / s) + z).clamp(0, qmax)
            x = (q - z) * s
        return F.conv2d(x, w, self.conv.bias, self.conv.stride,
                        self.conv.padding, self.conv.dilation, self.conv.groups)


def conv_layer_names(model):
    return [n for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]


def _set_submodule(root, name, new_mod):
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_mod)


def build_quant_model(src, w_bits=0, a_bits=0,
                      per_layer_w=None, per_layer_a=None,
                      a_scales=None, a_zeros=None, a_symmetric=False):
    """构造量化模型。

    per_layer_w / per_layer_a : dict{层名: 位宽}，用于混合精度（e5）。
                                不出现在 dict 里的层用默认的 w_bits / a_bits。
    a_scales / a_zeros        : dict{层名: 张量}，来自 calibrate_activations()。
    a_symmetric               : 激活用对称码空间（见 QuantConv2d）。
                                默认 False，即本平台既有的非对称行为。
    """
    m = copy.deepcopy(src)
    m.eval()
    for nm, mod in list(m.named_modules()):
        if not isinstance(mod, nn.Conv2d):
            continue
        wb = (per_layer_w or {}).get(nm, w_bits)
        ab = (per_layer_a or {}).get(nm, a_bits)
        sc = (a_scales or {}).get(nm)
        zz = (a_zeros or {}).get(nm)
        _set_submodule(m, nm, QuantConv2d(mod, wb, ab, sc, zz, a_symmetric))
    return m
