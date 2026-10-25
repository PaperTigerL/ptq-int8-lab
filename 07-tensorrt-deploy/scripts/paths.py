"""路径与数据约定，集中放在一处——别在各脚本里重复硬编码。

两个环境变量可以覆盖默认值：
  SHOWBRO_ROOT  ShowBro 仓库根（默认 /mnt/d/WorkBuddy/ShowBro）
  TRT_WORKDIR   中间产物目录（默认 本项目的 work/）
"""
import os
from pathlib import Path

import numpy as np

SHOWBRO_ROOT = Path(os.environ.get("SHOWBRO_ROOT", "/mnt/d/WorkBuddy/ShowBro"))
HERE = Path(__file__).resolve().parent
TRT_WORKDIR = Path(os.environ.get("TRT_WORKDIR", HERE.parent / "work"))

# 数据：CIFAR-10 test_batch.raw，每条 3073 字节 = 1 字节 label + 3072 字节 CHW 像素
DATA = SHOWBRO_ROOT / "lab" / "05-ncnn-deploy" / "data_raw" / "test_batch.raw"
CKPT = SHOWBRO_ROOT / "paper-code" / "cifar_resnet18_50ep.pth"
PTQ_LAB = SHOWBRO_ROOT / "lab" / "03-ptq-lab"

REC = 3073

# 归一化口径：和 GPU/CPU 侧所有评测器一致 = (x/255 - mean) / std。
# 预先把 255 吸收进常量，省掉一次全量除法。
MEAN255 = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(3, 1, 1) * 255.0
INVSTD255 = (np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(3, 1, 1) * 255.0) ** -1


def load_raw(path=DATA):
    return np.fromfile(path, dtype=np.uint8)


def normalize(blk):
    """blk: (N, 3, 32, 32) float32 像素 -> 归一化后的 float32 NCHW。"""
    return (blk - MEAN255) * INVSTD255


def take_block(raw, start, count):
    """取 raw 里 [start, start+count) 条记录，返回像素/标签。

    返回 (pixels(N,3,32,32) float32 已归一化, labels(N,) uint8)
    """
    blk = raw[start * REC: (start + count) * REC].reshape(count, REC)
    px = blk[:, 1:].reshape(count, 3, 32, 32).astype(np.float32)
    return normalize(px), blk[:, 0]


def ensure_workdir():
    TRT_WORKDIR.mkdir(parents=True, exist_ok=True)
    return TRT_WORKDIR
