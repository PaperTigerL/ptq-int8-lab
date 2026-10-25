# -*- coding: utf-8 -*-
"""续跑修复的有界验证（CPU，微型模型，不碰真实数据/GPU）。

覆盖：
1. 第 2 个 epoch 崩溃后，同命令续跑会从 ckpt 恢复权重/优化器/调度器，
   且权重回读 max|Δ| == 0（证明不是「只恢复 epoch 号」）。
2. 换协议（label_smoothing 0.1→0.2）会被闸门拒绝，不产生新 ckpt。
3. 同协议但换 lr（0.01→0.02）也会被拒绝。
4. --init-from 只加载权重开启新阶段，provenance 写进 ckpt/history。

跑法：/home/user/pyenv-torch/bin/python tests/test_resume.py
"""
import sys
import json
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS.parent))

import finetune                      # noqa: E402
import load_transfer                 # noqa: E402
import ptq_toolkit.models as models_mod      # noqa: E402
import ptq_toolkit.pipeline as pipe_mod      # noqa: E402


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        h = torch.relu(self.bn1(self.conv1(x)))
        return self.fc(h.mean(dim=(2, 3)))


class FakeDS:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


class FakeLoader:
    def __init__(self, n, bs, batches, count=False):
        self.dataset = FakeDS(n)
        self.bs = bs
        self.batches = batches
        self.count = count

    def __iter__(self):
        if self.count:
            CRASH["iters"] += 1
            if CRASH["armed"] and CRASH["iters"] == 2:
                raise RuntimeError("injected crash at start of epoch 2")
        for _ in range(self.batches):
            yield torch.randn(self.bs, 3, 8, 8), torch.randint(0, 10, (self.bs,))

    def __len__(self):
        return self.batches


CRASH = {"armed": True, "iters": 0}

models_mod.get_model = lambda arch: Tiny()
pipe_mod.load_ptq_lab = lambda: {}
pipe_mod.get_n_train = lambda: 50000
pipe_mod.get_train_loader = lambda bs, augment=True: FakeLoader(
    50000, bs, 2, count=True)
pipe_mod.get_loaders = lambda calib_n=128, eval_n=0: (
    None, FakeLoader(10000, 64, 1), None, 10000)
load_transfer.fetch_weights = lambda arch, verbose=False: {
    "state": {"conv1.weight": torch.randn(64, 3, 7, 7)}}

TMP = Path(tempfile.mkdtemp(prefix="ptq_resume_"))
finetune.CKPT_DIR = TMP
# 单测用 scratch：tiny 模型没有 ImageNet 权重；正式训练默认 imagenet。
BASE = ["--arch", "resnet50", "--batch-size", "64", "--device", "cpu",
        "--threads", "1", "--weights", "scratch"]


def run(extra):
    old = sys.argv
    sys.argv = ["finetune.py"] + BASE + extra
    try:
        return finetune.main()
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 1
    finally:
        sys.argv = old


def ckpath(ep, tag=""):
    stem = f"resnet50_{tag}" if tag else "resnet50"
    return TMP / f"ckpt_{stem}_ep{ep}.pth"


def main():
    print(f"[test] ckpt dir = {TMP}")
    print("\n=== 0. --resume 无同协议历史必须被拒 ===")
    rc = run(["--epochs", "2", "--resume"])
    assert rc == 6, rc
    assert not ckpath(1).exists()
    print("[test] --resume 无历史被拒，exit=6（首次训练不能带 --resume）")

    print("\n=== 1. 第 2 epoch 崩溃 ===")
    try:
        run(["--epochs", "2"])
        raise AssertionError("应当注入崩溃，但没有")
    except RuntimeError as e:
        assert "injected" in str(e), e
        print(f"[test] 已注入崩溃：{e}")
    assert ckpath(1).is_file() and not ckpath(2).exists()
    c1 = torch.load(ckpath(1), map_location="cpu", weights_only=False)
    assert c1["epoch"] == 1 and c1["target_epoch"] == 2, c1.keys()
    print(f"[test] ep1 已落盘，target_epoch={c1['target_epoch']}")

    print("\n=== 2. 同命令续跑 ===")
    CRASH["armed"] = False
    rc = run(["--epochs", "2"])
    assert rc == 0, rc
    c2 = torch.load(ckpath(2), map_location="cpu", weights_only=False)
    assert c2["resumed_from"] == "ckpt_resnet50_ep1.pth", c2["resumed_from"]
    assert c2["resume_max_abs_diff"] == 0.0, c2["resume_max_abs_diff"]
    hist = json.loads((TMP / "history_resnet50.json").read_text("utf-8"))
    assert [r["epoch"] for r in hist] == [1, 2], [r["epoch"] for r in hist]
    assert all(r["target_epoch"] == 2 for r in hist)
    print(f"[test] ep2 续跑成功：resumed_from={c2['resumed_from']} "
          f"max|Δ|={c2['resume_max_abs_diff']} history={[r['epoch'] for r in hist]}")

    print("\n=== 3. 跨协议续跑必须被拒 ===")
    rc = run(["--epochs", "3", "--label-smoothing", "0.2"])
    assert rc == 6, rc
    assert not ckpath(3).exists()
    print(f"[test] 跨协议被拒，exit={rc}，未产生 ep3")

    print("\n=== 3b. 同协议换 lr 也必须被拒 ===")
    rc = run(["--epochs", "2", "--lr", "0.02"])
    assert rc == 6, rc
    print(f"[test] 换 lr 被拒，exit={rc}")

    print("\n=== 4. --init-from 只加载权重开新段 ===")
    rc = run(["--epochs", "1", "--run-tag", "warm",
              "--init-from", str(ckpath(2))])
    assert rc == 0, rc
    cw = torch.load(ckpath(1, "warm"), map_location="cpu", weights_only=False)
    assert cw["init_from"].endswith("ckpt_resnet50_ep2.pth"), cw["init_from"]
    assert cw["init_epoch"] == 2 and cw["resumed_from"] is None
    assert cw["target_epoch"] == 1
    print(f"[test] 新段 ckpt={ckpath(1, 'warm').name} "
          f"init_from={Path(cw['init_from']).name} init_epoch={cw['init_epoch']}")

    print("\n=== 5. --weights imagenet 必须真的装进非头权重 ===")
    tiny = Tiny()
    src = {k: (torch.randn_like(v) if v.is_floating_point() else v.clone())
           for k, v in tiny.state_dict().items()}
    load_transfer.fetch_weights = lambda arch, verbose=False: {
        "state": src,
        "info": {"url": "fake://tiny", "sha256": "0" * 64}}
    # lr=0：不发生参数更新，才能用训练后的 ckpt 反证「初始化时确实装了」。
    # BN running stats 仍会在 train 模式更新，所以只比参数、跳过 buffer。
    rc = run(["--epochs", "1", "--weights", "imagenet", "--run-tag", "tf",
              "--lr", "0"])
    assert rc == 0, rc
    ck = torch.load(ckpath(1, "tf"), map_location="cpu", weights_only=False)
    assert ck["weights"] == "imagenet", ck["weights"]
    n_nonhead = sum(1 for k in src if not k.startswith("fc."))
    assert ck["n_transferred"] == n_nonhead, (ck["n_transferred"], n_nonhead)
    got = ck["model"]
    compared = 0
    for k, v in src.items():
        if k.startswith("fc.") or "running_" in k                 or k.endswith("num_batches_tracked"):
            continue
        assert torch.equal(got[k], v), (
            k, float((got[k].float() - v.float()).abs().max()))
        compared += 1
    print(f"[test] lr=0 下 {compared} 个非头参数逐元素与源一致"
          f"（n_transferred={ck['n_transferred']}/{n_nonhead}）；"
          f"head=fc.* 被重置（未加载）")

    print(f"\n[PASS] 7 项全部通过（ckpt 目录 {TMP}）")


if __name__ == "__main__":
    main()
