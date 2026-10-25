# -*- coding: utf-8 -*-
"""数据加载与校准集构造。

校准集是 PTQ 的全部依据 —— 激活的量化范围只能从它上面统计出来。
所以校准集必须：
  1. 来自训练集（不是测试集，否则是泄露）
  2. 固定种子（否则每次跑结果都不同，没法比较）
  3. 不做随机增强（要的是真实分布，不是增强后的分布）

2026-10-15：按 C.DATASET 分发 CIFAR-10 / CIFAR-100。
**归一化参数跟着数据集走**（C.MEAN/C.STD），不能沿用 CIFAR-10 的值 ——
用错会让 FP32 基线本身就偏低，后面所有掉点数字都失真。
"""
import os
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

import config as C


def test_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(C.MEAN, C.STD),
    ])


def train_transform():
    """训练/校准用。校准时不该做随机增强，所以这里只给 ToTensor+Normalize。"""
    return T.Compose([
        T.ToTensor(),
        T.Normalize(C.MEAN, C.STD),
    ])


def _dataset_cls():
    """按 C.DATASET 返回 torchvision 数据集类。"""
    table = {
        "cifar10": torchvision.datasets.CIFAR10,
        "cifar100": torchvision.datasets.CIFAR100,
    }
    if C.DATASET not in table:
        raise ValueError("未知数据集 %r" % C.DATASET)
    return table[C.DATASET]


def load_datasets(download=False):
    """返回 (trainset, testset)。

    数据放在 paper-code/data/<dataset>/ 下。download=True 时缺失会自动下载
    （首次跑 CIFAR-100 需要；之后就不用联网了）。
    """
    cls = _dataset_cls()
    root = C.dataset_dir()
    trainset = cls(root, train=True, download=download, transform=train_transform())
    testset = cls(root, train=False, download=download, transform=test_transform())
    return trainset, testset


def ensure_dataset(download=True):
    """确保数据集在本地。缺失就下载。返回 (ok, 路径或原因)。"""
    root = C.dataset_dir()
    if os.path.isdir(root):
        return True, root
    if not download:
        return False, "缺 %s 且未允许下载" % root
    try:
        load_datasets(download=True)
        return True, root
    except Exception as exc:                       # 网络不通/磁盘满等
        return False, "下载失败：%s" % str(exc)[:120]


def make_calib_loader(size=None, seed=None):
    """从训练集按固定顺序切一段做校准集。

    注意用 Subset(range(n)) 而不是 RandomSampler —— 固定顺序才可复现。
    """
    size = size or C.CALIB_SIZE
    seed = seed or C.CALIB_SEED
    trainset, _ = load_datasets()
    g = torch.Generator()
    g.manual_seed(seed)
    # 用固定种子的置换取前 size 个，既不重复也可复现
    perm = torch.randperm(len(trainset), generator=g)[:size].tolist()
    sub = Subset(trainset, perm)
    return DataLoader(sub, batch_size=C.CALIB_BATCH, shuffle=False,
                      num_workers=C.NUM_WORKERS, pin_memory=True)


def make_test_loader(batch=None):
    _, testset = load_datasets()
    return DataLoader(testset, batch_size=(batch or C.EVAL_BATCH), shuffle=False,
                      num_workers=C.NUM_WORKERS, pin_memory=True)


def make_train_loader(batch=128):
    trainset, _ = load_datasets()
    return DataLoader(trainset, batch_size=batch, shuffle=False,
                      num_workers=C.NUM_WORKERS, pin_memory=True)
