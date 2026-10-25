# -*- coding: utf-8 -*-
"""独立复评：不信日志，直接把 .pth 加载进模型跑测试集。

用法（GPU 上 2000 张约 10–30 s；CPU 很慢，只建议 --limit 64 做冒烟）：
    PYTHONPATH=. /home/user/pyenv-torch/bin/python tools/eval_ckpt.py \
        --ckpt /mnt/d/WorkBuddy/ShowBro/paper-code/ckpts/ckpt_resnet50_ep25.pth \
        --arch resnet50 --limit 2000 --device cuda

输出一行 `RESULT ...`，含样本数、复评 top-1、ckpt 内记录值及差值。
全量 10000 张时容差 0.2 pp；子集按 3σ 二项抽样误差放宽（否则 64 张
的正常波动会被误判）。超容差退出码 2 —— 监工应把它当作
「声明与权重不一致」。
"""
import argparse
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLKIT = HERE.parent
sys.path.insert(0, str(TOOLKIT))
sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arch", default="resnet50",
                    choices=["resnet18", "resnet50", "mobilenet_v2"])
    ap.add_argument("--limit", type=int, default=0,
                    help="测试集上限；0=全量 10000。CPU 上请用 64")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    import torch
    from ptq_toolkit.models import get_model
    from ptq_toolkit.pipeline import get_loaders, load_trained_weights

    if str(args.device).startswith("cuda"):
        free_b, total_b = torch.cuda.mem_get_info()
        if free_b < 1500 * 2**20:
            print(f"[闸门] 显存可用 {free_b / 2**20:.0f} MiB < 1500 MiB"
                  f"（总 {total_b / 2**20:.0f}）—— 放弃，不抢 GPU",
                  file=sys.stderr)
            return 4

    ckpt_path = Path(args.ckpt)
    blob = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    meta = {k: blob.get(k) for k in
            ("epoch", "test_acc", "protocol", "target_epoch", "resumed_from")}
    model = get_model(args.arch)
    load_trained_weights(model, ckpt_path, device="cpu")
    model = model.to(args.device).eval()

    _, test_loader, _, _ = get_loaders(calib_n=1, eval_n=args.limit)
    if args.batch_size and args.batch_size != test_loader.batch_size:
        test_loader = torch.utils.data.DataLoader(
            test_loader.dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=0)
    vc = vt = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(args.device), y.to(args.device)
            vc += int((model(x).argmax(1) == y).sum())
            vt += y.numel()
    acc = 100.0 * vc / max(vt, 1)
    claimed = meta.get("test_acc")
    delta = (acc - float(claimed)) if claimed is not None else float("nan")
    if claimed is None:
        tol = 0.0
    elif vt >= 10000:
        tol = 0.2
    else:
        p = max(1e-6, min(1 - 1e-6, float(claimed) / 100.0))
        tol = max(0.2, 3.0 * math.sqrt(p * (1 - p) / max(vt, 1)) * 100)
    print(f"RESULT ckpt={ckpt_path.name} arch={args.arch} "
          f"epoch={meta.get('epoch')} n={vt} "
          f"acc={acc:.2f} claimed={claimed} delta={delta:+.2f}pp "
          f"tol={tol:.2f}pp protocol={meta.get('protocol')} "
          f"device={args.device}")
    if claimed is not None and abs(delta) > tol:
        print(f"[FAIL] 复评与 ckpt 内记录差 > 容差 {tol:.2f} pp",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
