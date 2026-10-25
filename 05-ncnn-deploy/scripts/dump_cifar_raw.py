# -*- coding: utf-8 -*-
"""把 CIFAR-10 的 pickle 批次转成 C++ runner 能读的裸二进制。

磁盘上的真实格式（先弄清再动手，否则会白忙）
--------------------------------------------
`data/cifar-10-batches-py/` 下的 data_batch_* / test_batch **不是裸二进制**，
是 Python pickle（协议 2）序列化的 dict：
    {"batch_label": ..., "labels": [10000 个 int],
     "data": (10000, 3072) uint8, "filenames": [...]}
每行 3072 字节的排法是 [R 平面 1024][G 平面 1024][B 平面 1024]，
每个平面内 32x32 行优先。**没有任何 1 字节 label 前缀** —— label 在
`labels` 列表里单独放着。

所以 C++ 侧直接读这些文件会大小不整除（31035526 % 3073 != 0），
必须先转一层。

为什么用 torchvision 加载而不是自己 unpickle
------------------------------------------
ptq-lab 的训练/评测走的是 torchvision.datasets.CIFAR10。
这里必须走同一条路，才能保证「C++ 看到的图」和「PyTorch 看到的图」
是同一批像素。自己 unpickle 当然也能读对，但那就多了一处可能和
torchvision 不一致的实现 —— 而两边不一致时，精度差异是查不动的。

输出格式（每样本 3073 字节）
---------------------------
    [0]        label，1 字节
    [1..3072]  R 平面 1024 + G 平面 1024 + B 平面 1024
和 cifar10.h 里 kRecordSize / kPlaneSize 的定义严格对应。

用法（WSL）
----------
    ~/pyenv-torch/bin/python dump_cifar_raw.py                 # 只导测试集
    ~/pyenv-torch/bin/python dump_cifar_raw.py --train         # 连训练集一起导
"""
import argparse
import os

import torchvision

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
RAW_DIR = os.path.join(LAB, "data_raw")        # 产物目录
SHOWBRO = os.path.dirname(os.path.dirname(LAB))
CIFAR_ROOT = os.path.join(SHOWBRO, "paper-code", "data")

RECORD = 3073
PLANE = 1024


def dump_split(root, train, out_path):
    ds = torchvision.datasets.CIFAR10(root=root, train=train, download=False)
    data = ds.data          # (N, 32, 32, 3) uint8, HWC
    targets = ds.targets    # list[int]
    n = len(targets)
    assert data.shape == (n, 32, 32, 3), data.shape
    assert data.dtype.name == "uint8", data.dtype

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        for i in range(n):
            f.write(bytes([int(targets[i]) & 0xFF]))
            # HWC -> 三个平面。astype 一次拿 contiguous，避免逐字节 python 循环
            img = data[i]
            for c in range(3):
                plane = img[:, :, c]
                assert plane.shape == (32, 32) and plane.flags["C_CONTIGUOUS"]
                f.write(plane.tobytes())

    size = os.path.getsize(out_path)
    assert size == n * RECORD, (size, n * RECORD)
    print("  %-28s %6d 张  %10d 字节  (== %d x %d)"
          % (os.path.basename(out_path), n, size, n, RECORD))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=CIFAR_ROOT,
                    help="CIFAR-10 数据根目录（含 cifar-10-batches-py/）")
    ap.add_argument("--outdir", default=RAW_DIR)
    ap.add_argument("--train", action="store_true",
                    help="同时导出 5 个训练批次（默认只导测试集）")
    args = ap.parse_args()

    if not os.path.isdir(os.path.join(args.root, "cifar-10-batches-py")):
        raise SystemExit("找不到 %s/cifar-10-batches-py" % args.root)

    print("源      : %s" % args.root)
    print("输出目录: %s" % args.outdir)
    print("导出中  :")

    total = dump_split(args.root, False, os.path.join(args.outdir, "test_batch.raw"))

    if args.train:
        # 训练集有 5 个批次，各自导一个文件，和磁盘上的划分保持一致
        for b in range(1, 6):
            ds = torchvision.datasets.CIFAR10(root=args.root, train=True,
                                              download=False)
            # 上面那行拿到的是全部 50000 张；按 10000 切片对应 data_batch_1..5
            data, targets = ds.data, ds.targets
            sl = slice((b - 1) * 10000, b * 10000)
            sub_data, sub_targets = data[sl], targets[sl]
            out_path = os.path.join(args.outdir, "data_batch_%d.raw" % b)
            os.makedirs(args.outdir, exist_ok=True)
            with open(out_path, "wb") as f:
                for i in range(len(sub_targets)):
                    f.write(bytes([int(sub_targets[i]) & 0xFF]))
                    img = sub_data[i]
                    for c in range(3):
                        f.write(img[:, :, c].tobytes())
            size = os.path.getsize(out_path)
            assert size == 10000 * RECORD, (size, 10000 * RECORD)
            print("  %-28s %6d 张  %10d 字节" % (os.path.basename(out_path),
                                                10000, size))
            total += 10000

    print("完成，共 %d 张。" % total)


if __name__ == "__main__":
    main()
