#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""基线训练：把 (数据集 × 模型) 组合训到收敛，作为量化实验的可信基线。

**为什么必须有这一步**：量化实验要谈"掉点"，只有在收敛的基线上掉点才是量化
引起的。若基线欠拟合，测出来的"掉点"里混着欠拟合噪声，分不清是谁造成的。
上一轮就踩过：3 epoch 的基线只有 58.09%，结果全歪。

**为什么要训多个基线**：跨数据集 / 跨模型的结论才站得住。
但也**不必全训**——见下方矩阵说明。

用法（WSL 内）:
    cd /mnt/d/WorkBuddy/ShowBro/lab/03-ptq-lab
    ~/pyenv-torch/bin/python tools/train_baseline.py --dataset cifar100 --arch resnet18
    ~/pyenv-torch/bin/python tools/train_baseline.py --dataset cifar10 --arch mobilenetv2

训练矩阵说明：
    |组合                | 是否需要训 | 用途 |
    |--------------------|-----------|------|
    | cifar10 + resnet18  | ❌ 已有    | 87.16%，原始基线 |
    | cifar100+ resnet18  | ✅        | 换数据集，验证结论是否跨数据集成立 |
    | cifar10 + mobilenetv2 | ✅      | 换模型，验证结论是否跨模型成立 |
    | cifar100+ mobilenetv2 | 可选     | 两者都换，最强但成本翻倍 |
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader

import config as C


def log(m):
    print(m, flush=True)


def build_loaders():
    """训练/测试数据增强。归一化参数跟着 C.MEAN/C.STD 走 —— 用错会让基线本身偏低。"""
    from torchvision.datasets import CIFAR10, CIFAR100
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(C.MEAN, C.STD),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(C.MEAN, C.STD),
    ])
    cls = {"cifar10": CIFAR10, "cifar100": CIFAR100}[C.DATASET]
    root = C.dataset_dir()
    tr = cls(root, train=True, download=True, transform=train_tf)
    te = cls(root, train=False, download=True, transform=test_tf)
    return (DataLoader(tr, batch_size=C.BATCH_SIZE, shuffle=True,
                        num_workers=C.NUM_WORKERS, pin_memory=True, drop_last=True),
            DataLoader(te, batch_size=C.EVAL_BATCH, shuffle=False,
                       num_workers=2, pin_memory=True),
            len(tr), len(te))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10", choices=list(C.DATASET_SPECS))
    ap.add_argument("--arch", default="resnet18", choices=list(C.ARCH_SPECS))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--tag", default="50ep", help="权重文件名后缀")
    ap.add_argument("--wd", type=float, default=5e-4,
                    help="weight decay（默认 5e-4）。过拟合时调到 5e-3")
    ap.add_argument("--ls", type=float, default=0.1,
                    help="label smoothing（默认 0.1）。过拟合时可调到 0.2")
    args = ap.parse_args()

    C.configure(args.dataset, args.arch)
    C.BATCH_SIZE = args.bs

    log("=" * 70)
    log("基线训练 · %s / %s" % (args.dataset, args.arch))
    log("=" * 70)
    if not torch.cuda.is_available():
        log("⚠️ 无 CUDA，将用 CPU 训练（会非常慢）")

    import src.model as M
    model = M.build_fp32().to(C.DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log("模型     : %.2fM params" % (n_params / 1e6))
    log("数据     : %s（归一化 mean[0]=%.4f std[0]=%.4f）" % (args.dataset, C.MEAN[0], C.STD[0]))
    log("")

    train_loader, test_loader, n_tr, n_te = build_loaders()
    log("训练集 %d / 测试集 %d" % (n_tr, n_te))
    log("超参     : epochs=%d lr=%.3f wd=%.1e ls=%.2f"
        % (args.epochs, args.lr, args.wd, args.ls))
    log("")

    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.wd, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=args.ls)

    @torch.no_grad()
    def evaluate():
        model.eval()
        c = n = 0
        for x, y in test_loader:
            x, y = x.to(C.DEVICE, non_blocking=True), y.to(C.DEVICE, non_blocking=True)
            c += (model(x).argmax(1) == y).sum().item()
            n += y.size(0)
        model.train()
        return c / n

    os.makedirs(C.CKPT_DIR, exist_ok=True)
    ckpt = os.path.join(C.CKPT_DIR,
                        "%s_%s_%s.pth" % (args.dataset, args.arch, args.tag))
    log("权重将保存到: %s" % ckpt)
    log("")
    log("  ep |  train  |   test   | lr | 耗时")
    log("-" * 52)

    history = []
    best = 0.0
    t_start = time.time()

    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        tot = corr = 0
        run_loss = 0.0
        for x, y in train_loader:
            x = x.to(C.DEVICE, non_blocking=True)
            y = y.to(C.DEVICE, non_blocking=True)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            tot += y.size(0)
            corr += (out.argmax(1) == y).sum().item()
            run_loss += loss.item() * y.size(0)
        sched.step()

        test_acc = evaluate()
        train_acc = corr / tot
        dt = time.time() - t0
        eta = (time.time() - t_start) / (ep + 1) * (args.epochs - ep - 1)

        log("  %2d | %6.2f%% | %6.2f%% | %.3f | %3.0fs  ETA %4.1fmin"
            % (ep + 1, train_acc * 100, test_acc * 100,
               opt.param_groups[0]["lr"], dt, eta / 60))

        history.append({"epoch": ep + 1, "train_acc": train_acc,
                        "test_acc": test_acc, "lr": opt.param_groups[0]["lr"]})

        if test_acc > best:
            best = test_acc
            torch.save(model.state_dict(), ckpt)
            log("       ↑ 新最优，已保存 (%.2f%%)" % (best * 100))

    log("")
    log("=" * 70)
    log("完成 · 最优测试精度 %.2f%% · 总耗时 %.1f 分钟"
        % (best * 100, (time.time() - t_start) / 60))
    log("权重: %s" % ckpt)
    log("=" * 70)

    hist_path = os.path.join(
        C.CKPT_DIR, "%s_%s_%s_history.json" % (args.dataset, args.arch, args.tag))
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump({"best": best, "args": vars(args), "params_m": n_params / 1e6,
                   "history": history}, f, indent=2)
    log("训练曲线: %s" % hist_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
