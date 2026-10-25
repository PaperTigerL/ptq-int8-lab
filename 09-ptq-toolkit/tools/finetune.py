# -*- coding: utf-8 -*-
"""ResNet-50 / MobileNetV2 在 CIFAR-10 上的微调（S1 第 2 件）。

用法
----
    # 2 epoch 冒烟（先跑这个：train/test 都要在涨，显存 < 3400 MiB）
    python tools/finetune.py --arch resnet50 --epochs 2 --batch-size 64

    # 冒烟通过后再跑满 20 epoch
    python tools/finetune.py --arch resnet50 --epochs 20 --batch-size 64 \
        --run-tag r50aug20

    # 断点续训：同协议历史 + ckpt 会自动恢复；--resume 强制要求「必须续跑」
    python tools/finetune.py --arch resnet50 --epochs 20 --batch-size 64 \
        --run-tag r50aug20 --resume

默认协议：全量 50k · batch 64 · lr 0.01 cosine · wd 5e-4 ·
训练集增强（RandomCrop(32,padding=4) + RandomHorizontalFlip）·
label_smoothing 0.1。改任一项都会体现在 history 的 `protocol` 字段里，
不同协议的记录**不可混着比**。

产物（paper-code/ckpts/，相对仓库根）
-----
    ckpt_<arch>_ep<N>.pth         每个 epoch 的 checkpoint（model/opt/epoch/acc）
    history_<arch>.json           逐 epoch 的 train/test acc、耗时、显存峰值
    SHA256SUMS                    history 与 ckpt 的 SHA256 清单

⚠️ 三个已知边界（写在这里，避免误读结果）
-----
1. **bn1 的 running stats 保持 ImageNet 原值，不重估也不重置。**
   实测三种处置在 CIFAR 上的 fp32 top-1（GPU，batch 128×20 重估）：
     ① 保留原 stats   16.56%   ← 最好
     ② 重估 bn1       12.03%   (-4.53 pp)
     ③ 重置 bn1        9.33%   (-7.23 pp)
   **反直觉但可解释**：bn1 的 stats 是在 224×224/7×7s2 上统计的，
   而 CIFAR 的输入分布完全不同 —— 重估只用了 20 个 batch（5120 张），
   统计量本身噪声大；且 16.56% 与 10% 的差距说明**三种都在随机猜附近**，
   bn1 根本不是瓶颈。真正的瓶颈是 conv1 中心裁剪丢了 59% L1 质量
   + fc 随机重置。所以保留原样，把吸收问题交给微调。

2. **微调前精度 ≈ 随机猜**（9.85% ~ 16.56%），这不是 bug：
   ImageNet 的 1000 类特征 + 裁剪过的 conv1 + 随机 fc，在 CIFAR 上
   不可能有可用精度。S1 的目标就是让它从「不可用」变「可用」。

3. **S1 未达标前，任何地方都不得写「精度达标」** —— 包括 report.md / README。
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
TOOLKIT = HERE.parent
REPO = TOOLKIT.parent.parent          # lab/09-ptq-toolkit → lab → ShowBro
sys.path.insert(0, str(TOOLKIT))
sys.path.insert(0, str(HERE))

CKPT_DIR = REPO / "paper-code" / "ckpts"


def sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet50",
                   choices=["resnet50", "mobilenet_v2"])
    ap.add_argument("--epochs", type=int, default=20,
                    help="目标总 epoch 数：跑完后到达的 epoch 编号。"
                         "续跑从已完成的下一 epoch 接着跑到它（默认 20）。")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="资源纪律：GPU 训练 batch <= 64")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1,
                    help="CrossEntropyLoss 的 label_smoothing；"
                         "抑制过拟合（train 99.9%% / test 84.5%% 那种）")
    ap.add_argument("--no-augment", dest="augment", action="store_false",
                    help="关掉训练集增强（默认开）。"
                         "⛔ 只在对照实验时用 —— 正式跑必须开")
    ap.add_argument("--weights", default="imagenet",
                    choices=["imagenet", "scratch"],
                    help="imagenet=torchvision ImageNet 权重移植（正式协议，"
                         "默认）；scratch=随机初始化（仅消融/单元测试，"
                         "⛔ 不能当 S1 迁移基线引用）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--conv1", default="A", choices=["A", "B", "C"],
                    help="conv1 移植方式：A 中心裁剪不缩放 / B 面积补偿 5.44x / "
                         "C 核空间求和池化。选哪个由 results/bn_conv1_compare.md "
                         "的对照实验决定")
    ap.add_argument("--calib-n", type=int, default=128,
                   help="⚠️ 与训练样本数无关：训练一律用全量 50k"
                        "（get_train_loader）。此参数只影响 test loader "
                        "的构建占位。")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=2,
                    help="torch.set_num_threads（资源纪律，默认 2）")
    ap.add_argument("--run-tag", default="",
                    help="实验段标识：history/ckpt 文件名加 _<tag>，"
                         "把不同协议或不同阶段的记录彻底分开")
    ap.add_argument("--init-from", default=None,
                    help="只加载该 .pth 的权重开启新阶段（优化器/调度器/epoch"
                         "全新，不是续跑）；必须配 --run-tag")
    ap.add_argument("--resume", action="store_true",
                    help="强制要求从同协议历史 + ckpt 续跑；没有可续的"
                         "历史/检查点就拒绝（默认行为是「有则续、无则从头」）。"
                         "给 run_finetune_full.sh 的 --resume 用。")
    ap.add_argument("--ckpt-dir", default=None,
                    help="checkpoint/history/SHA256SUMS 输出目录"
                         "（默认 paper-code/ckpts）。冒烟可用临时目录，"
                         "避免把废弃 ckpt 混进正式目录。")
    args = ap.parse_args()

    global CKPT_DIR
    if args.ckpt_dir:
        CKPT_DIR = Path(args.ckpt_dir).resolve()

    import torch
    import torch.nn as nn

    torch.set_num_threads(args.threads)
    from ptq_toolkit.models import get_model
    from ptq_toolkit.pipeline import (get_loaders, get_n_train,
                                      get_train_loader, load_ptq_lab)

    lab = load_ptq_lab()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 数据：训练用**全量** trainset，测试用全量 testset ----
    # ⛔ 曾经用 get_loaders(calib_n) 的第一个返回值当 train_loader，
    #    那其实是**校准集**（Subset(trainset, range(calib_n))），
    #    于是 n_train 永远是 calib_n（5120），跑出来的只是子集微调。
    #    现在训练专用 get_train_loader，并在开跑前硬校验 n_train。
    n_train_expected = get_n_train()
    train_loader = get_train_loader(args.batch_size, augment=args.augment)
    n_train = len(train_loader.dataset)
    _, test_loader, _, n_test = get_loaders(calib_n=args.calib_n, eval_n=0)

    if n_train != n_train_expected or n_train != 50000:
        print(f"\n[闸门] 训练集样本数 {n_train} != 50000"
              f"（期望 {n_train_expected}）—— 拒绝开跑。",
              file=sys.stderr)
        print("  说明拿到的是校准子集而不是全量训练集。", file=sys.stderr)
        raise SystemExit(4)
    print(f"[数据] 训练 {n_train} 张（全量）/ 测试 {n_test} 张，"
          f"batch={args.batch_size}，{len(train_loader)} batches/epoch")

    # ---- 模型：移植 ImageNet 权重（conv1 方式与 BN 处置由参数决定）----
    # conv1 的三种方式与 BN 三处置的对照见 tools/bn_reestimate.py，
    # 结果在 results/bn_conv1_compare.md。选定的起点由 --conv1 指定。
    #
    # ⛔ 真实踩过的坑（067b5ab 引入）：原实现是 model, _ = transplant(...)，
    #    重构时换成手写片段，却只把 conv1 搬进模型，**其余权重仍是随机初始化**。
    #    症状极隐蔽：能训练、精度也在涨（等于 from scratch），
    #    但结果根本不是「ImageNet 迁移 + 微调」，和 S1 的协议声明不符。
    #    现在整份非分类头 state_dict 逐张量加载 + 独立回读 torch.equal 校验；
    #    想显式跑 from scratch 只能写 --weights scratch（消融/单测）。
    from load_transfer import (HEAD_PREFIX, apply_conv1_mode, check_keys,
                               fetch_weights, set_bn_mode)
    transfer_meta = dict(weights=args.weights)
    model = get_model(args.arch)
    if args.weights == "imagenet":
        got = fetch_weights(args.arch, verbose=False)
        src, winfo = got["state"], got["info"]
        head = HEAD_PREFIX[args.arch]
        load_sd = {k: v for k, v in src.items() if not k.startswith(head)}
        if args.arch == "resnet50":
            conv1_info = apply_conv1_mode(model, src["conv1.weight"], args.conv1)
            # apply_conv1_mode 已按 A/B/C 改写 model.conv1；这里把改后的
            # 3×3 权重放回 load_sd，防止整体加载又把它覆盖成源 7×7。
            load_sd["conv1.weight"] = model.conv1.weight.detach().clone()
        else:
            conv1_info = dict(mode="none",
                              note="mobilenet_v2 stem 本就是 3×3，只改 stride 2→1")
        missing, unexpected = model.load_state_dict(load_sd, strict=False)
        check_keys(missing, unexpected, allowed_prefix=head)
        sd_now = model.state_dict()
        bad = [k for k, v in load_sd.items()
               if not torch.equal(sd_now[k].cpu(), v.cpu())]
        if bad:
            print(f"\n[闸门] ImageNet 移植后 {len(bad)} 个张量与源不一致："
                  f"{bad[:5]}", file=sys.stderr)
            raise SystemExit(5)
        transfer_meta = dict(
            weights="imagenet", weights_url=winfo["url"],
            weights_sha256=winfo["sha256"], n_transferred=len(load_sd),
            conv1_mode=args.conv1)
        print(f"[移植] {args.arch}：{len(load_sd)}/{len(src)} 个非分类头张量"
              f"来自 ImageNet（head={head}* 重置），逐元素回读一致")
    else:
        conv1_info = dict(mode="none")
        print("[移植] ⚠️ --weights scratch：随机初始化（仅消融/单测，"
              "不能当迁移基线引用）")
    bn_info = set_bn_mode(model, "keep")      # 微调起点：保留 ImageNet stats
    model = model.to(args.device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"[模型] {args.arch}，{n_param / 1e6:.2f}M 参数，"
          f"device={args.device}")

    # ---- 优化器 ----
    opt = torch.optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs)
    lossf = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # ---- protocol 字符串：把「这些数字是怎么跑出来的」写进记录本身 ----
    # ⛔ 吃过亏：history 里出现两条 "epoch: 1"，lr/随机性都不同，
    #    但 json 里分辨不出来。现在协议一变，protocol 字段就变，
    #    不同协议的记录**不可混着比**（尤其是 best epoch 不能跨协议取）。
    aug_tag = "aug" if args.augment else "noaug"
    init_tag = "imagenet" if args.weights == "imagenet" else "scratch"
    protocol = (f"cifar10-full-50k-{aug_tag}"
                f"-ls{args.label_smoothing:g}"
                f"-wd{args.weight_decay:g}-cos{args.epochs}-{init_tag}")

    if args.init_from and not args.run_tag:
        print("[闸门] --init-from 必须配 --run-tag：新阶段必须与原协议记录"
              "分开，否则 epoch 编号和 best 会混在一起。", file=sys.stderr)
        raise SystemExit(6)

    stem = f"{args.arch}_{args.run_tag}" if args.run_tag else args.arch
    hist_path = CKPT_DIR / f"history_{stem}.json"
    history = json.loads(hist_path.read_text(encoding="utf-8")) \
        if hist_path.exists() else []

    # ⛔ 只认同一协议的历史。不同协议的 epoch 编号彼此无关；跨协议取
    #    max 会让「从 ep27 接着跑」实际变成从零重训（真实踩过的坑）。
    same = [r for r in history if r.get("protocol") == protocol]
    ep0 = max((int(r["epoch"]) for r in same), default=0)

    if history and not same:
        other = sorted({r.get("protocol") for r in history})
        print(f"[闸门] {hist_path.name} 已有其它协议记录 {other}；"
              "为防 epoch 编号混用，请用 --run-tag 开一段新记录。",
              file=sys.stderr)
        raise SystemExit(6)
    if ep0 and args.init_from:
        print(f"[闸门] --init-from 不能接在已有同协议历史（ep{ep0}）上；"
              "请换一个 --run-tag。", file=sys.stderr)
        raise SystemExit(6)
    if args.resume and args.init_from:
        print("[闸门] --resume（严格续跑）不能与 --init-from（只载权重开新段）"
              "同时使用：二者语义相反。", file=sys.stderr)
        raise SystemExit(6)
    # --resume 的语义是「要求必须续跑」：没有同协议历史就拒绝。
    # 不传时保持原来的「有则续、无则从头」——这样 run_finetune_full.sh
    # 第一次执行不需要任何特殊参数，中断后重跑同一条命令即可续上。
    if args.resume and not ep0:
        print(f"[闸门] --resume 要求续跑，但 {hist_path.name} 里没有 "
              f"protocol={protocol!r} 的历史；"
              "首次训练请不要带 --resume。", file=sys.stderr)
        raise SystemExit(6)

    resume_ck = None
    resume_diff = None
    init_meta = {}
    if args.init_from:
        init_path = Path(args.init_from)
        if not init_path.is_file():
            print(f"[闸门] --init-from 文件不存在：{init_path}", file=sys.stderr)
            raise SystemExit(6)
        c0 = torch.load(init_path, map_location=args.device,
                        weights_only=False)
        model.load_state_dict(c0["model"], strict=True)
        init_meta = dict(init_from=str(init_path.resolve()),
                         init_epoch=c0.get("epoch"),
                         init_test_acc=c0.get("test_acc"),
                         init_protocol=c0.get("protocol"))
        print(f"[初始化] 仅加载权重 {init_path.name}（ep{c0.get('epoch')} "
              f"test={c0.get('test_acc')}% protocol={c0.get('protocol')}）；"
              "优化器/调度器/epoch 全部新起")
    elif ep0:
        resume_ck = CKPT_DIR / f"ckpt_{stem}_ep{ep0}.pth"
        if not resume_ck.is_file():
            print(f"[闸门] 历史到 ep{ep0} 但找不到 {resume_ck.name}；"
                  "拒绝从零重训冒充续跑。", file=sys.stderr)
            raise SystemExit(5)
        ck = torch.load(resume_ck, map_location=args.device,
                        weights_only=False)
        if ck.get("protocol") != protocol:
            print(f"[闸门] 检查点协议 {ck.get('protocol')!r} != "
                  f"当前 {protocol!r}", file=sys.stderr)
            raise SystemExit(6)
        if ck.get("target_epoch") is None:
            print(f"[闸门] {resume_ck.name} 是旧格式（无 target_epoch），"
                  "无法确认原定训练长度；请用 --init-from + --run-tag 开新段，"
                  "或用新 --run-tag 从头跑。", file=sys.stderr)
            raise SystemExit(6)
        if args.epochs != ck["target_epoch"]:
            print(f"[闸门] 续跑目标 {args.epochs} != 原目标 "
                  f"{ck['target_epoch']}；余弦调度会错位。要改目标请用 "
                  "--init-from + --run-tag 开新段。", file=sys.stderr)
            raise SystemExit(6)
        if float(ck.get("lr", args.lr)) != float(args.lr):
            print(f"[闸门] 续跑 lr={args.lr} != 检查点 lr={ck.get('lr')}；"
                  "换超参请用 --init-from + --run-tag 开新段。",
                  file=sys.stderr)
            raise SystemExit(6)
        if int(ck.get("batch_size", args.batch_size)) != int(args.batch_size):
            print(f"[闸门] 续跑 batch={args.batch_size} != 检查点 "
                  f"batch={ck.get('batch_size')}；请用 --batch-size "
                  f"{ck.get('batch_size')} 重试。", file=sys.stderr)
            raise SystemExit(6)
        if ck.get("conv1") != args.conv1:
            print(f"[闸门] 续跑 conv1={args.conv1} != 检查点 "
                  f"conv1={ck.get('conv1')}", file=sys.stderr)
            raise SystemExit(6)
        if ck.get("weights", "scratch") != args.weights:
            print(f"[闸门] 续跑 weights={args.weights} != 检查点 "
                  f"weights={ck.get('weights', 'scratch')}；"
                  "ImageNet 迁移与 from scratch 不可混着续。",
                  file=sys.stderr)
            raise SystemExit(6)
        if int(ck.get("n_train", n_train)) != n_train:
            print(f"[闸门] 续跑 n_train={n_train} != 检查点 "
                  f"n_train={ck.get('n_train')}", file=sys.stderr)
            raise SystemExit(6)
        model.load_state_dict(ck["model"], strict=True)
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        sd = model.state_dict()
        resume_diff = max(
            float((sd[k].detach() - v.to(args.device)).abs().max())
            for k, v in ck["model"].items())
        if resume_diff != 0.0:
            print(f"[闸门] 权重回读不一致 max|Δ|={resume_diff}",
                  file=sys.stderr)
            raise SystemExit(5)
        print(f"[续跑] 已从 {resume_ck.name} 恢复权重/优化器/调度器"
              f"（max|Δ|={resume_diff:g}）；ep{ep0 + 1} → ep{args.epochs}")
    else:
        print("[续跑] 无同协议历史，从 ep1 全新开始")

    if args.epochs <= ep0:
        print(f"[闸门] 目标 {args.epochs} 不大于已完成 ep{ep0}；无事可做。",
              file=sys.stderr)
        raise SystemExit(6)

    print(f"[训练] target_epochs={args.epochs} lr={args.lr} "
          f"batch={args.batch_size} wd={args.weight_decay} "
          f"ls={args.label_smoothing} augment={args.augment}")
    print(f"[协议] {protocol}")
    print("-" * 74)

    # OOM 守卫状态（见下面的 except 分支）
    cur_batch = args.batch_size
    oom_downgraded = False
    oom_downgrades = 0
    oom_log: List[Dict] = []

    for ep in range(ep0 + 1, args.epochs + 1):
        model.train()
        torch.cuda.reset_peak_memory_stats() if args.device == "cuda" else None
        t0 = time.time()
        run_loss, n_correct, n_total, n_batch = 0.0, 0, 0, 0
        nan_seen = False
        for x, y in train_loader:
            x, y = x.to(args.device), y.to(args.device)
            opt.zero_grad(set_to_none=True)
            try:
                out = model(x)
                loss = lossf(out, y)
                loss.backward()
                opt.step()
            except torch.cuda.OutOfMemoryError as oom:
                # ⚠️ OOM 守卫：捕获后自动降 batch 一次并记录，不让整轮白跑。
                #    4 GB 卡上 batch 128 + ResNet-50 已贴边（3767/4096 MiB）。
                #    只降一次 —— 连续降说明配置本身有问题，应该让人来看。
                if oom_downgraded:
                    print(f"\n[OOM] 第 {oom_downgrades + 1} 次 OOM"
                          f"（batch 已降到 {cur_batch}）—— 停止，不继续降。",
                          file=sys.stderr)
                    raise
                oom_downgrades += 1
                oom_downgraded = True
                new_bs = max(cur_batch // 2, 8)
                print(f"\n[OOM] batch={cur_batch} 显存不足 → 降到 {new_bs}",
                      file=sys.stderr)
                torch.cuda.empty_cache()
                train_loader = get_train_loader(new_bs, augment=args.augment)
                cur_batch = new_bs
                args.batch_size = new_bs
                opt = torch.optim.SGD(model.parameters(), lr=args.lr,
                                      momentum=args.momentum,
                                      weight_decay=args.weight_decay)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=args.epochs)
                oom_log.append(dict(epoch=ep, from_batch=new_bs * 2,
                                    to_batch=new_bs))
                continue
            if not torch.isfinite(loss):
                nan_seen = True
            run_loss += float(loss)
            n_correct += int((out.argmax(1) == y).sum())
            n_total += y.numel()
            n_batch += 1
        sched.step()
        dt = time.time() - t0

        train_acc = 100.0 * n_correct / max(n_total, 1)
        train_loss = run_loss / max(n_batch, 1)

        model.eval()
        with torch.no_grad():
            vc, vt = 0, 0
            for x, y in test_loader:
                x, y = x.to(args.device), y.to(args.device)
                pred = model(x).argmax(1)
                vc += int((pred == y).sum())
                vt += y.numel()
        test_acc = 100.0 * vc / max(vt, 1)

        peak = (torch.cuda.max_memory_allocated() / 2**20
                if args.device == "cuda" else float("nan"))

        rec = dict(epoch=ep, arch=args.arch, lr=args.lr,
                   batch_size=cur_batch,
                   train_acc=round(train_acc, 4),
                   train_loss=round(train_loss, 6),
                   test_acc=round(test_acc, 4),
                   seconds=round(dt, 2),
                   ms_per_img=round(dt * 1000 / max(n_total, 1), 4),
                   peak_mem_mib=round(peak, 1) if peak == peak else None,
                   nan=nan_seen,
                   # n_train 记**全量训练集大小**，不是本 epoch 实际迭代到的
                   # 样本数 —— 后者在 OOM 降 batch 后仍等于全集，
                   # 但若中途异常退出就会偏小，容易误读成「本来就用了子集」。
                   n_train=n_train, n_test=vt,
                   protocol=protocol,
                   conv1=args.conv1,
                   weight_decay=args.weight_decay,
                   label_smoothing=args.label_smoothing,
                   augment=args.augment,
                   oom_downgrades=oom_downgrades,
                   target_epoch=args.epochs,
                   run_tag=args.run_tag or None,
                   resumed_from=resume_ck.name if resume_ck else None,
                   resume_max_abs_diff=resume_diff,
                   **transfer_meta,
                   **init_meta)
        history.append(rec)

        print(f"ep{ep:>3} | train {train_acc:6.2f}% (loss {train_loss:.4f}) | "
              f"test {test_acc:6.2f}% | {dt:6.1f}s "
              f"({rec['ms_per_img']:.3f} ms/img) | "
              f"peak {peak:7.1f} MiB | {'NaN!' if nan_seen else 'finite'}")

        # ---- 每 epoch 存 checkpoint ----
        ck = CKPT_DIR / f"ckpt_{stem}_ep{ep}.pth"
        torch.save(dict(arch=args.arch, epoch=ep, model=model.state_dict(),
                        opt=opt.state_dict(), sched=sched.state_dict(),
                        train_acc=train_acc, test_acc=test_acc,
                        lr=args.lr, batch_size=cur_batch,
                        n_train=n_train, n_test=vt,
                        protocol=protocol,
                        conv1=args.conv1,
                        weight_decay=args.weight_decay,
                        label_smoothing=args.label_smoothing,
                        augment=args.augment,
                        target_epoch=args.epochs,
                        run_tag=args.run_tag or None,
                        resumed_from=resume_ck.name if resume_ck else None,
                        resume_max_abs_diff=resume_diff,
                        **transfer_meta,
                        **init_meta), ck)
        hist_path.write_text(json.dumps(history, indent=2,
                                        ensure_ascii=False), encoding="utf-8")

    # ---- SHA256 清单 ----
    sums = CKPT_DIR / "SHA256SUMS"
    lines = []
    for p in sorted(CKPT_DIR.glob("*")):
        if p.name == "SHA256SUMS" or not p.is_file():
            continue
        lines.append(f"{sha256_file(p)}  {p.name}")
    sums.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("-" * 74)
    print(f"[产物] {CKPT_DIR}")
    for p in sorted(CKPT_DIR.glob(f"*{args.arch}*")) + [sums]:
        print(f"       {p.name}  ({p.stat().st_size / 1024:.0f} KB)")
    print(f"[SHA256] {sums.name} 已写（{len(lines)} 个文件）")

    # ---- 冒烟判据 ----
    bad = [r for r in history if r.get("nan")]
    over = [r for r in history
            if r.get("peak_mem_mib") and r["peak_mem_mib"] > 3400]
    if bad:
        print("\n✗ 出现 NaN —— 立即停止，先查 loss/数据")
        return 2
    if over:
        print(f"\n✗ 显存峰值 {[r['peak_mem_mib'] for r in over]} MiB "
              f"超过 3400 MiB 上限")
        return 3
    print("\n✓ 冒烟通过：finite、显存 < 3400 MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
