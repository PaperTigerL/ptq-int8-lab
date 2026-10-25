# -*- coding: utf-8 -*-
"""把校准集导出成 ncnn2table 能吃的 .npy 文件。

为什么用 npy 而不是图片
----------------------
ncnn2table 有两条输入通路：
  type=0  读图片（jpg/png/bmp），靠 OpenCV 解码
  type=1  读 .npy，直接是 float32 张量

选 npy 的三个理由：
  1. **口径必须和 03-ptq-lab 完全一致**。那边用固定种子 2026 从训练集取前 1024 张，
     做 ToTensor + Normalize。图片通路会引入 JPEG/PNG 编解码误差和一次 0-255 往返，
     校准集就和 Python 侧不是同一批数据了 —— 那测出来的差异就说不清是引擎问题
     还是数据问题。
  2. OpenCV 在我们这版 ncnn 构建里没开（见 docs/），图片通路根本不可用。
  3. npy 是无损的，fp32 原样落盘。

npy 布局（已核对 ncnn 源码，别改）
---------------------------------
ncnn::Mat 是**平面式 CHW**：cstep ≈ w*h，通道是最外层连续块。
ncnn2table 的 read_npy() 要求 numpy shape 为 (c, h, w)，C 连续，float32。
所以 PyTorch 张量 .numpy() 后直接 np.save 就对，不需要转置。

用法（WSL）
----------
    ~/pyenv-torch/bin/python dump_calib_npy.py                # 默认 1024 张
    ~/pyenv-torch/bin/python dump_calib_npy.py --num 256      # 少量先试通路
    ~/pyenv-torch/bin/python dump_calib_npy.py --list calib.txt  # 指定 filelist 输出
"""
import argparse
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
CALIB_DIR = os.path.join(LAB, "calib")          # 放 .npy 和 filelist
SHOWBRO = os.path.dirname(os.path.dirname(LAB))
PTQLAB = os.path.join(SHOWBRO, "lab", "03-ptq-lab")

# 校准集口径必须和 03-ptq-lab 逐字一致 —— 换一个数，两边就不可比了。
CALIB_SIZE = 1024
CALIB_SEED = 2026
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=CALIB_SIZE,
                    help="导出多少张校准图（默认 1024，与 ptq-lab 相同）")
    ap.add_argument("--outdir", default=CALIB_DIR)
    ap.add_argument("--list", default=None,
                    help="filelist 输出路径，默认 <outdir>/filelist.txt")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    list_path = args.list or os.path.join(args.outdir, "filelist.txt")

    # 直接复用 ptq-lab 的 dataset_dir 逻辑，避免两处路径约定漂移
    sys_path = os.path.join(PTQLAB)
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    import config as C                      # ptq-lab 的集中配置
    import src.data as D                    # ptq-lab 的数据加载

    C.configure("cifar10", "resnet18")

    trainset, _ = D.load_datasets()
    g = torch.Generator()
    g.manual_seed(CALIB_SEED)
    perm = torch.randperm(len(trainset), generator=g)[:args.num].tolist()

    print("校准集 : 训练集前 %d 张（种子 %d，与 ptq-lab 同口径）"
          % (args.num, CALIB_SEED))
    print("输出   : %s" % args.outdir)

    paths = []
    for i, idx in enumerate(perm):
        img, _ = trainset[idx]              # 已 ToTensor + Normalize，(3,32,32) fp32
        assert img.shape == (3, 32, 32), img.shape
        assert img.dtype == torch.float32, img.dtype
        p = os.path.join(args.outdir, "calib_%05d.npy" % i)
        np.save(p, img.numpy())             # C 连续 CHW，ncnn 直接可用
        paths.append(p)

    with open(list_path, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(p + "\n")

    # ---- 自检：落盘的 npy 必须能原样读回，且布局是 CHW ----
    back = np.load(paths[0])
    assert back.shape == (3, 32, 32), back.shape
    assert back.dtype == np.float32, back.dtype
    assert back.flags["C_CONTIGUOUS"]
    ref = trainset[perm[0]][0].numpy()
    assert float(np.abs(back - ref).max()) == 0.0, "npy 往返有损"
    print("自检    : npy 往返无损，shape=%s dtype=%s C-contiguous"
          % (back.shape, back.dtype))
    print("filelist: %s （%d 行）" % (list_path, len(paths)))


if __name__ == "__main__":
    main()
