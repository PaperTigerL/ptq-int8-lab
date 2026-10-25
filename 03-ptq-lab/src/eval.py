# -*- coding: utf-8 -*-
"""评估与结果落盘。

一条纪律：实验结果必须落盘。
跑完只在终端打印等于没跑 —— 下周你想不起来当时是多少，
也没法把两次实验摆在一起比。所有结果进 results/。
"""
import csv
import json
import os
import time

import torch

import config as C
from src.data import make_test_loader
from src.resultio import append_csv_row, flatten_payload


@torch.no_grad()
def evaluate(model, loader=None, device=None, max_batches=None):
    """返回 top-1 准确率（0~1）。"""
    device = device or C.DEVICE
    loader = loader or make_test_loader()
    model.eval()
    correct = total = 0
    for bi, (x, y) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_detail(model, loader=None, device=None, max_batches=None):
    """带耗时和显存的完整评估。"""
    device = device or C.DEVICE
    loader = loader or make_test_loader()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    model.eval()
    correct = total = 0
    t0 = time.time()
    for bi, (x, y) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0

    peak = (torch.cuda.max_memory_allocated() / 1024 ** 2) if device == "cuda" else 0.0
    return {
        "acc": correct / max(total, 1),
        "correct": correct,
        "total": total,
        "seconds": round(dt, 2),
        "ms_per_image": round(dt * 1000 / max(total, 1), 3),
        "peak_mem_mb": round(peak, 1),
    }


def save_result(exp_name, payload, tag=None):
    """落盘到 results/<dataset>_<arch>/<exp>_<tag>.json，并往该目录 all.csv 追加一行。

    all.csv 是给你攒对比表用的（列对齐由 resultio 保证，可放心往 Excel 里贴）。

    ⚠️ **每个 (数据集 × 模型) 组合一个独立子目录**。这是 2026-10-17 参数化时加的：
    早期版本所有实验都写进 results/ 根目录，换模型跑完就和新数据混在一起，
    根本分不清哪行是哪组。**报告里必须能追溯到 (dataset, arch)。**
    """
    # 自动带上口径标记 —— 这样即使有人手工合并 csv 也不会丢失来源
    payload.setdefault("dataset", C.DATASET)
    payload.setdefault("arch", C.ARCH)
    payload.setdefault("config_tag", C.TAG)

    d = C.ensure_dirs()
    stamp = time.strftime("%m%d-%H%M%S")
    fname = "%s_%s.json" % (exp_name, tag or stamp)
    path = os.path.join(d, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    row = {"exp": exp_name, "tag": tag or stamp, "file": fname}
    row.update(flatten_payload(payload))
    append_csv_row(os.path.join(d, "all.csv"), row)
    return path
