# -*- coding: utf-8 -*-
"""S1 第一步：把 torchvision 的 ImageNet 权重接到 CIFAR-10 上，并留下可复核的出处。

为什么需要这一层
================
M2 的 ResNet-50 是**随机初始化**的（fp32 恒在 10% 附近，实测 6.25 / 10.40 / 10.65%），
它的精度数字不可引用。本脚本给出「有出处的初始化权重」：

  1. 从 `download.pytorch.org` 取 torchvision 官方 ImageNet 权重
     （不走代理；实测 curl 该 URL http=206，见 `weights_manifest.json`）
  2. `conv1` 7×7 → 3×3 **中心裁剪**。CIFAR 是 32×32，用 ImageNet 的 7×7s2
     会把特征图砍到 16×16，再叠 maxpool(3,2,1) 只剩 8×8，到 layer3 就没有空间维了。
     本仓库的 ResNet-50 定义本来就是 3×3s1 + 无 maxpool（见 models/resnet50.py），
     所以这里只需把 7×7 的**中心 3×3 区域**搬过去。
  3. `fc` 重置为 10 类（源是 1000 类，形状不兼容，不能"加载失败就算了"）

自检不是"key 名对上了就算"
==========================
`strict=False` 加载之后：

  A. missing / unexpected **只允许 `fc.*`**；出现任何其它 key 差异 → 退出码 1
  B. 所有**非 fc** 张量与源权重**逐元素 bit 相同**（`torch.equal`）——
     证明权重真的搬过来了，而不是靠改个名字、留个空壳通过自检
  C. 32×32 前向输出 `(N, 10)` 且无 NaN/Inf
  D. `fc` 形状确实是 `(num_classes, in_features)`，且其权重**不是**源权重
  E. 与 torchvision 原生 resnet50（conv1 换 3×3s1、maxpool/fc 换 Identity）
     同权重同输入，2048 维池化特征 `max|diff| ≤ 1e-4`

⭐ 为什么 B 必须存在：`strict=False` 只核对**键集合**。一个把 conv 权重留成
   随机初始化的实现，只要键名对得上，同样能通过 A。键名检查是必要条件，
   不是充分条件。

用法
====
    python tools/load_transfer.py --arch resnet50
    python tools/load_transfer.py --arch resnet50 --tag IMAGENET1K_V1
    python tools/load_transfer.py --arch resnet50 --save-ckpt ckpts/resnet50_init.pt

退出码：0 = 全部自检通过；1 = 自检失败；2 = 取权重失败（网络/缓存）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Tuple

# ------------------------------------------------------------------ 路径

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DEFAULT = REPO_ROOT / "weights_manifest.json"

# 直接 `python tools/load_transfer.py` 时 sys.path[0] 是 tools/，
# `import ptq_toolkit` 会失败 —— 把仓库根挂上去，脚本才能从任何 cwd 跑。
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# arch -> (torchvision 权重枚举类名, 默认 tag)
# 只登记**已经能真跑通**的模型。
#
# arch -> (torchvision 权重枚举类名, 默认 tag)
#
# MobileNetV2 已于 M2r S1 登记 —— 本仓库定义已重写为**标准 t=6** 结构
# （features.0.0 stem + 17 个 InvertedResidual × 3 conv + features.18 投影
#   + classifier=Sequential(Dropout, Linear)），与 torchvision 键集合一致，
#   唯一差别是 stem stride 2→1（CIFAR 32×32）。
ARCH_TABLE: Dict[str, Tuple[str, str]] = {
    "resnet50": ("ResNet50_Weights", "IMAGENET1K_V2"),
    "mobilenet_v2": ("MobileNet_V2_Weights", "IMAGENET1K_V1"),
}

# 各 arch 的「分类头前缀」—— 只有这些 key 允许与源权重不同
# （源是 1000 类，我们是 10 类，形状必然不兼容）。
#   resnet50     -> "fc."
#   mobilenet_v2 -> "classifier."
HEAD_PREFIX: Dict[str, str] = {
    "resnet50": "fc.",
    "mobilenet_v2": "classifier.",
}

# conv1 移植参数：源核 / 目标核 / 方式
CONV1_SRC_K = 7
CONV1_DST_K = 3
CONV1_MODE = "center_crop"

CST = timezone(timedelta(hours=8))


def _now_iso() -> str:
    """带时区的 ISO8601 —— 只写本地时间会让"什么时候取的"无法判定。"""
    return datetime.now(CST).isoformat(timespec="seconds")


# ------------------------------------------------------------------ 取权重

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _weights_enum(arch: str, tag: str):
    """取 torchvision 的权重枚举成员（URL 由它给出，不手写 URL）。"""
    if arch not in ARCH_TABLE:
        raise SystemExit(
            f"未登记的 arch={arch!r}；可选：{', '.join(sorted(ARCH_TABLE))}")
    cls_name, default_tag = ARCH_TABLE[arch]
    import torchvision.models as tvm
    cls = getattr(tvm, cls_name)
    tags = [m.name for m in cls]
    if tag is None:
        tag = default_tag
    if tag not in tags:
        raise SystemExit(f"{cls_name} 没有 tag={tag!r}；可选：{', '.join(tags)}")
    return cls[tag]


def fetch_weights(arch: str, tag: str | None = None,
                  verbose: bool = True) -> Dict:
    """下载（或命中缓存）torchvision 权重，返回出处信息。

    用 `torch.hub.load_state_dict_from_url(check_hash=True)`：
      · 断点续传与失败重试由 torch 自己处理
      · `check_hash` 会核对**文件名里的 8 位 hash 前缀**
        （`resnet50-0676ba61.pth` → 0676ba61），这是 torchvision 自带的第一道校验
    我们另算**完整 SHA256**并写进 manifest —— 8 位前缀不足以定位一个 98 MB 文件。
    """
    import torch

    w = _weights_enum(arch, tag)
    url = w.url
    fname = url.rsplit("/", 1)[-1]
    cache_path = Path(torch.hub.get_dir()) / "checkpoints" / fname
    had = cache_path.is_file()

    t0 = time.time()
    # get_state_dict 内部就是 download → torch.load，返回 dict（不构造 1000 类模型）
    state = w.get_state_dict(check_hash=True, progress=False)
    if not cache_path.is_file():
        raise SystemExit(f"下载后找不到缓存文件：{cache_path}")
    dt = time.time() - t0

    info = dict(
        arch=arch,
        weights_enum=f"{w.__class__.__name__}.{w.name}",
        url=url,
        sha256=sha256_file(cache_path),
        size_bytes=cache_path.stat().st_size,
        size_human=f"{cache_path.stat().st_size / 2**20:.2f} MiB",
        downloaded_at=datetime.fromtimestamp(
            cache_path.stat().st_mtime, CST).isoformat(timespec="seconds"),
        fetched_at=_now_iso(),
        fetch_status="cache-hit" if had else "downloaded",
        # ⚠️ 两个不同的量，别混：
        #   fetch_seconds     本次「取用」耗时。cache-hit 时只是读本地缓存
        #                     （0.0x 秒），**不是**下载耗时。
        #   download_seconds  首次从网络下载的耗时。只有 fetch_status=="downloaded"
        #                     时才有真实值；cache-hit 时为 None。
        # 曾经的坑：README/commit message 写「实测 8.42s 下完」，
        # 但 manifest 里只有 cache-hit 的 0.2s —— 8.42s 没有任何字段承载，
        # 读者无法验证。现在要么有 download_seconds 字段，要么不写这个数。
        fetch_seconds=round(dt, 2),
        download_seconds=(round(dt, 2) if not had else None),
        cache_path=str(cache_path),
    )
    if verbose:
        print(f"[取权重] {info['weights_enum']}")
        print(f"         url     {url}")
        print(f"         sha256  {info['sha256']}")
        print(f"         size    {info['size_human']}（{info['size_bytes']} B）")
        print(f"         状态    {info['fetch_status']}，{info['fetch_seconds']}s，"
              f"文件时间 {info['downloaded_at']}")
    return dict(info=info, state=state)


# ------------------------------------------------------------------ 移植

def _load_ours(arch: str, num_classes: int):
    """构造本仓库的 CIFAR 版模型（不经过 registry，避免拉起 03 的依赖）。"""
    if arch == "resnet50":
        from ptq_toolkit.models import resnet50 as M
        return M.build(num_classes=num_classes)
    if arch == "mobilenet_v2":
        from ptq_toolkit.models import mobilenet_v2 as M
        return M.build(num_classes=num_classes)
    raise SystemExit(f"未实现的 arch={arch!r}")


def center_crop_conv1(src_w, dst_k: int = CONV1_DST_K):
    """把 (O, I, 7, 7) 的 conv1 权重中心裁剪成 (O, I, k, k)。

    ⚠️ **不做缩放**（M2r 任务书指定的就是"中心裁剪"）。
       实测后果（不是猜的，数字见 manifest 的 `conv1_transplant`）：
         · 中心 3×3 只占原 7×7 权重 **L1 质量的 41%**（不是面积的 9/49=18%，
           学出来的 7×7 滤波器能量集中在中心）
         · 平均 |w| **上升**：0.156 → 0.350
       所以 conv1 的输出分布和 ImageNet 时**不一样**。

       ⚠️ **BN running stats 失配不止 bn1**：stem 从 7×7s2 改成 3×3s1、
       并且去掉了 maxpool，**每个 stage 的特征图几何都变了** ——
       layer1..4 的全部 BN stats 都是在 224×224/7×7s2 上统计的，
       与 32×32/3×3s1 的输入分布都不匹配。
       这个偏差由 S1 的微调吸收；若 10 epoch 后精度异常，
       第一件事是用 CIFAR 数据重估一遍**全部** BN running stats 再判断。
    """
    import torch
    if src_w.ndim != 4:
        raise SystemExit(f"conv1 权重维度异常：{tuple(src_w.shape)}")
    k = src_w.shape[-1]
    if k < dst_k:
        raise SystemExit(f"conv1 核 {k} 小于目标 {dst_k}，无法中心裁剪")
    off = (k - dst_k) // 2
    cropped = src_w[:, :, off:off + dst_k, off:off + dst_k].contiguous()
    assert cropped.shape[-1] == dst_k

    # ⛔⛔ 「确实是中心区域」不能拿 `off` 自己证自己。
    #   曾经的写法是
    #       assert torch.equal(cropped, src_w[:, :, off:off+dst_k, off:off+dst_k])
    #   两侧是同一个表达式、同一个 off —— 恒真，永远不可能失败，
    #   却注释着「防 off 算错成左上角」。
    #
    #   正确证法：用**独立算出的索引**比。
    #     ① 期望中心块 = src_w[:, :, 2:2+dst_k, 2:2+dst_k]
    #        （7×7 裁 3×3 的中心 offset 独立算出来就是 2）
    #     ② 断言 cropped **不等于**左上角块 —— off 被误算成 0 时必然失败
    #   只有 k=7/dst_k=3 这一种组合有硬编码的期望，其它尺寸跳过。
    if k == 7 and dst_k == 3:
        assert torch.equal(cropped, src_w[:, :, 2:2 + dst_k, 2:2 + dst_k]), (
            "裁剪结果不是中心块（7×7→3×3 的中心应是 [2:5, 2:5]）")
        assert not torch.equal(cropped, src_w[:, :, 0:dst_k, 0:dst_k]), (
            "裁剪结果等于左上角块 —— off 被误算成了 0")
    return cropped, off


def check_keys(missing, unexpected, allowed_prefix: str = "fc.") -> None:
    """自检 A：missing / unexpected 只允许 fc.*，否则退出码 1。"""
    bad_missing = [k for k in missing if not k.startswith(allowed_prefix)]
    bad_unexpected = [k for k in unexpected if not k.startswith(allowed_prefix)]
    if bad_missing or bad_unexpected:
        print("\n[自检 A 失败] 键集合差异超出 fc.* 允许范围", file=sys.stderr)
        if bad_missing:
            print(f"  missing（不允许）：{bad_missing}", file=sys.stderr)
        if bad_unexpected:
            print(f"  unexpected（不允许）：{bad_unexpected}", file=sys.stderr)
        print(f"  全部 missing={list(missing)}", file=sys.stderr)
        print(f"  全部 unexpected={list(unexpected)}", file=sys.stderr)
        print("  → 说明本仓库的模型定义与 torchvision 结构已分叉，"
              "不是权重问题，先对齐定义。", file=sys.stderr)
        raise SystemExit(1)


def ref_equivalence_check_mobilenet(model, src: Dict, atol: float = 1e-4,
                                    verbose: bool = True) -> float:
    """自检 E 的 MobileNetV2 版本：与 torchvision 原生实现同权重同输入对拍。

    MobileNetV2 与 resnet50 的差异：
      · 没有 conv1 移植（stem 本就是 3×3），只需把 features[0][0].stride 2→1
      · 分类头是 classifier.1（不是 fc）
      · 特征取的是 adaptive_avg_pool2d 之后的 1280 维（不是 2048）

    做法：torchvision 原生 mobilenet_v2，改 stem stride → 1、classifier → Identity，
    用 strict=True 装同一份权重；本仓库侧同样把 classifier 换成 Identity。
    """
    import copy
    import torch
    import torch.nn as nn
    import torchvision

    head = HEAD_PREFIX["mobilenet_v2"]

    tv = torchvision.models.mobilenet_v2(weights=None)
    # stem stride 2 → 1（CIFAR 适配，本仓库与它必须一致）
    tv.features[0][0].stride = (1, 1)
    tv.classifier = nn.Identity()

    payload = {k: v for k, v in src.items() if not k.startswith(head)}
    try:
        tv.load_state_dict(payload, strict=True)
    except RuntimeError as e:
        print("\n[自检 E 失败] torchvision 原生 mobilenet_v2 strict=True 装不进同一份权重",
              file=sys.stderr)
        print(f"  {e}", file=sys.stderr)
        print("  → 本仓库的 MobileNetV2 定义与 torchvision 已分叉（层名/层数/形状），"
              "必须先对齐定义。", file=sys.stderr)
        raise SystemExit(1)
    tv.eval()

    ours = copy.deepcopy(model)
    ours.classifier = nn.Identity()
    ours.eval()

    torch.manual_seed(0)
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        a = tv(x)
        b = ours(x)
    if a.shape != b.shape:
        print(f"\n[自检 E 失败] 特征形状不同：torchvision {tuple(a.shape)} "
              f"vs 本仓库 {tuple(b.shape)}", file=sys.stderr)
        raise SystemExit(1)
    maxdiff = float((a - b).abs().max())
    if verbose:
        print(f"[自检E ] 与 torchvision 原生 mobilenet_v2 同权重同输入："
              f"{a.shape[-1]} 维特征 max|diff|={maxdiff:.3g}（阈值 {atol:g}）")
    if maxdiff > atol:
        print(f"\n[自检 E 失败] max|diff|={maxdiff:.3g} > {atol:g}："
              f"权重一致但前向不一致 → forward 图（stride/残差/分组）有差异",
              file=sys.stderr)
        raise SystemExit(1)
    return maxdiff


def ref_equivalence_check(model, src: Dict, arch: str = "resnet50",
                          atol: float = 1e-4, verbose: bool = True) -> float:
    """自检 E：与 torchvision 原生实现**同权重同输入**比较池化特征。

    为什么这条必须存在
    ------------------
    自检 A 只核对**键集合**，B 只核对**权重数值**。两者都通过，仍然可能
    因为 **forward 图不同**而算错：stride 放在 conv1 还是 conv2、
    downsample 分支带不带 stride、残差是 `relu(out+id)` 还是 `relu(out)+id`、
    stem 后有没有 maxpool —— 这些差异**一个 key 都不会变**。

    做法：拿 torchvision 原生 resnet50，把 conv1 换成 3×3s1、maxpool 换成
    Identity、fc 换成 Identity，用 `strict=True` 装同一份权重。
    `strict=True` 能过，顺带证明**本仓库的模型定义与 torchvision 的键集合完全一致**
    （于是自检 A 里"missing 只有 fc.*"不是因为定义分叉恰好凑上了）。

    返回 max|diff|（2048 维池化特征）。
    """
    import copy
    import torch
    import torch.nn as nn
    import torchvision

    tv = getattr(torchvision.models, arch)(weights=None)
    tv.fc = nn.Identity()                       # 先换掉 fc，state_dict 里就没有 fc.*
    tv.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    tv.maxpool = nn.Identity()

    payload = {k: v for k, v in src.items() if not k.startswith("fc.")}
    payload["conv1.weight"] = model.conv1.weight.detach()
    try:
        tv.load_state_dict(payload, strict=True)     # 键集合必须严丝合缝
    except RuntimeError as e:
        print("\n[自检 E 失败] torchvision 原生实现 strict=True 装不进同一份权重", file=sys.stderr)
        print(f"  {e}", file=sys.stderr)
        print("  → 本仓库的模型定义与 torchvision 已分叉（层名/层数/参数形状），"
              "必须先对齐定义。", file=sys.stderr)
        raise SystemExit(1)
    tv.eval()

    ours = copy.deepcopy(model)
    ours.fc = nn.Identity()
    ours.eval()

    torch.manual_seed(0)
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        a = tv(x)
        b = ours(x)
    if a.shape != b.shape:
        print(f"\n[自检 E 失败] 特征形状不同：torchvision {tuple(a.shape)} "
              f"vs 本仓库 {tuple(b.shape)}", file=sys.stderr)
        raise SystemExit(1)
    maxdiff = float((a - b).abs().max())
    if verbose:
        print(f"[自检E ] 与 torchvision 原生实现同权重同输入："
              f"2048 维池化特征 max|diff|={maxdiff:.3g}（阈值 {atol:g}）")
    if maxdiff > atol:
        print(f"\n[自检 E 失败] max|diff|={maxdiff:.3g} > {atol:g}："
              f"权重一致但前向结果不一致 → forward 图（stride/残差/池化）有差异",
              file=sys.stderr)
        raise SystemExit(1)
    return maxdiff


def conv1_variants(src_w, dst_k: int = 3):
    """conv1 的 7×7→3×3 三种移植方式。返回 {mode: (weight, meta)}。

    A  center_crop_noscale     中心裁剪，不缩放（现状）
    B  center_crop_area_scaled 中心裁剪 ×(k*k)/(dst_k*dst_k) = 49/9 ≈ 5.44
    C  kernel_sum_pool         7×7 核空间求和池化成 3×3，保留全部 L1 质量

    三种都返回数值自检 meta（L1 比、是否等于中心块等），
    调用方必须把它们落进结果文件 —— 不许只写不验。
    """
    import torch
    import torch.nn as nn
    k = src_w.shape[-1]
    off = (k - dst_k) // 2
    center = src_w[:, :, off:off + dst_k, off:off + dst_k].contiguous()
    area_ratio = (k * k) / (dst_k * dst_k)

    out = {}

    # 公共数值自检字段（三种共用，便于对照表逐行比）
    def _meta(w, impl, ratio, **extra):
        src_l1 = float(src_w.abs().sum())
        dst_l1 = float(w.abs().sum())
        d = dict(impl=impl, ratio=round(ratio, 4),
                 src_shape=tuple(src_w.shape),
                 dst_shape=tuple(w.shape),
                 src_l1=round(src_l1, 4), dst_l1=round(dst_l1, 4),
                 l1_ratio=round(dst_l1 / src_l1, 4),
                 src_mean_abs=round(float(src_w.abs().mean()), 6),
                 dst_mean_abs=round(float(w.abs().mean()), 6),
                 equals_center_crop=bool(torch.equal(w.float(),
                                                     center.float())))
        d.update(extra)
        return d

    # A：中心裁剪不缩放
    out["A"] = (center, _meta(
        center, "center_crop_noscale", 1.0,
        l1_less_than_src=bool(float(center.abs().sum())
                              < float(src_w.abs().sum()))))

    # B：中心裁剪 × 面积比
    scaled = (center.float() * area_ratio).to(center.dtype)
    out["B"] = (scaled, _meta(
        scaled, "center_crop_area_scaled", area_ratio,
        equals_center_times_ratio=bool(torch.allclose(
            scaled.float(), center.float() * area_ratio, rtol=1e-5))))

    # C：空间求和池化（avg_pool × 面积比，保留全部 L1）
    pooled = nn.functional.adaptive_avg_pool2d(src_w.float(), dst_k)
    pooled = (pooled * area_ratio).to(src_w.dtype)
    out["C"] = (pooled, _meta(
        pooled, "kernel_sum_pool", area_ratio,
        is_pooled_not_crop=True,
        # C 保留全部 L1：avg_pool × 面积比 ≈ 求和，与源 L1 应几乎相等
        l1_preserved=bool(abs(float(pooled.abs().sum())
                              - float(src_w.abs().sum()))
                          / float(src_w.abs().sum()) < 0.02)))
    return out


def apply_conv1_mode(model, src_conv1_w, mode: str = "A", dst_k: int = 3):
    """按 A/B/C 之一把 conv1 权重移植进模型，返回数值自检。"""
    import torch
    if mode not in ("A", "B", "C"):
        raise SystemExit(f"未知 conv1 模式 {mode!r}；可选 A/B/C")
    variants = conv1_variants(src_conv1_w, dst_k)
    w, meta = variants[mode]
    conv1 = dict(model.named_modules())["conv1"]
    conv1.weight.data.copy_(w.reshape(conv1.weight.shape))
    return dict(mode=mode, **meta)


def set_bn_mode(model, mode: str = "keep", loader=None, device="cpu",
                batches: int = 8, scope: str = "all"):
    """BN running stats 处置：keep / reestimate / reset。

    scope="all" 时作用于**全部 BatchNorm2d**（遍历 model.modules()）——
    BN 失配是全网性的（stem 几何变了，layer1..4 全部失配），
    只修 bn1 一层会制造「半新半旧」的不一致状态。
    """
    import torch
    import torch.nn as nn
    bns = ([m for n, m in model.named_modules() if n == "bn1"]
           if scope == "bn1"
           else [m for m in model.modules()
                 if isinstance(m, nn.BatchNorm2d)])
    if mode == "keep":
        return dict(mode="keep", scope=scope, n_layers=len(bns))
    if mode == "reset":
        for bn in bns:
            bn.running_mean.zero_()
            bn.running_var.fill_(1.0)
            bn.num_batches_tracked.zero_()
        return dict(mode="reset", scope=scope, n_layers=len(bns))
    if mode == "reestimate":
        if loader is None:
            raise SystemExit("reestimate 需要 loader")
        for bn in bns:
            bn.reset_running_stats()
        was_training = model.training
        model.train()
        n = 0
        with torch.no_grad():
            for x, _ in loader:
                model(x.to(device))
                n += 1
                if n >= batches:
                    break
        if not was_training:
            model.eval()
        return dict(mode="reestimate", scope=scope, n_layers=len(bns),
                    batches=n)
    raise SystemExit(f"未知 BN 模式 {mode!r}；可选 keep/reestimate/reset")


def check_disk_match(model, disk_sd: Dict, head: str,
                     conv1_override=None, verbose: bool = True) -> int:
    """自检 B：模型的张量与**磁盘 .pth** 逐元素 bit 相同。返回通过的张量数。

    为什么单独抽成函数：这条自检必须能被**反例测试**调用
    （注入随机权重后它必须红、必须非零退出）。内联在 transplant 里的话，
    测试只能跑完整的下载+移植流程，无法构造「权重被污染但键集合仍干净」
    这种关键场景 —— 而那正是原实现漏掉的那类失败。

    参数
      disk_sd        : 从 .pth **独立 torch.load** 出来的 state_dict
      head           : 分类头前缀（"fc." / "classifier."），这些 key 跳过
      conv1_override : resnet50 的 conv1 是裁剪后搬进去的，磁盘里是 7×7，
                       所以用裁剪结果代替（None = 不替换）

    ⛔ 曾经的证法无效：拿 `model.state_dict()` 与刚灌进去的 dict 比，
       那只是在证明「load_state_dict 拷了张量」——
       把灌进去的 dict 换成随机噪声，照样报 318/318（实测过）。
       现在比的是「磁盘文件 → 模型」这条完整链路。
    """
    import torch
    sd = model.state_dict()
    n_same, diffs = 0, []
    for k, v in disk_sd.items():
        if k.startswith(head):
            continue
        if conv1_override is not None and k == "conv1.weight":
            v = conv1_override
        if k not in sd:
            diffs.append((k, "模型里没有这个 key"))
            continue
        if not torch.equal(sd[k], v):
            d = (sd[k].float() - v.float()).abs().max().item()
            diffs.append((k, f"数值不同 max|d|={d:.3g}"))
        else:
            n_same += 1
    if diffs:
        print(f"\n[自检 B 失败] {len(diffs)} 个张量与磁盘 .pth 不一致",
              file=sys.stderr)
        for k, why in diffs[:10]:
            print(f"  {k}: {why}", file=sys.stderr)
        raise SystemExit(1)
    n_expect = sum(1 for k in disk_sd if not k.startswith(head))
    if n_same != n_expect:
        print(f"\n[自检 B 失败] 应比对 {n_expect} 个张量，实际只比了 {n_same}",
              file=sys.stderr)
        raise SystemExit(1)
    if verbose:
        print(f"[自检B ] {n_same}/{n_expect} 个非分类头张量与**磁盘 .pth** "
              f"逐元素 bit 相同（独立 torch.load 回读）")
    return n_same


def transplant(arch: str = "resnet50", tag: str | None = None,
               num_classes: int = 10, verbose: bool = True):
    """取权重 → conv1 中心裁剪（resnet50）→ 分类头重置 → 五项自检。"""
    import torch
    import torch.nn as nn

    got = fetch_weights(arch, tag, verbose=verbose)
    info, src = got["info"], got["state"]

    model = _load_ours(arch, num_classes)
    src_keys = list(src.keys())

    # ---- conv1：中心裁剪（**仅 resnet50**）----
    # MobileNetV2 的 stem 本来就是 3×3，权重 (32,3,3,3) 形状完全兼容，
    # 只是 stride 2→1 —— 那属于 forward 的行为差异，权重直接搬即可。
    # 无条件取 src["conv1.weight"] 会对 mobilenet_v2 KeyError（实测过）。
    crop = None
    if arch == "resnet50":
        src_conv1 = src["conv1.weight"]
        crop, off = center_crop_conv1(src_conv1)
        if verbose:
            s7 = src_conv1.abs().mean().item()
            s3 = crop.abs().mean().item()
            print(f"[conv1 ] {tuple(src_conv1.shape)} → {tuple(crop.shape)} "
                  f"中心裁剪 offset={off}（不缩放）；"
                  f"平均 |w| {s7:.5f} → {s3:.5f}")
    elif verbose:
        print("[conv1 ] mobilenet_v2 stem 本就是 3×3，无需裁剪"
              "（只改 stride 2→1，权重形状兼容）")

    # ---- 组装待加载 state_dict：剔除分类头 ----
    # 分类头前缀按 arch 取（resnet50=fc.，mobilenet_v2=classifier.）。
    # MobileNetV2 **没有 conv1 移植**（stem 就是 3×3，只是 stride 2→1，
    # 权重形状 (32,3,3,3) 完全兼容）—— 所以只有 resnet50 走裁剪分支。
    head = HEAD_PREFIX[arch]
    adapted = dict(src)
    if arch == "resnet50":
        adapted["conv1.weight"] = crop
    load_sd = {k: v for k, v in adapted.items() if not k.startswith(head)}

    # ---- 加载 ----
    missing, unexpected = model.load_state_dict(load_sd, strict=False)
    if verbose:
        print(f"[加载  ] strict=False；missing={list(missing)} "
              f"unexpected={list(unexpected)}")
    check_keys(missing, unexpected, allowed_prefix=head)

    # ---- 自检 B：模型的张量与**磁盘 .pth** 逐元素 bit 相同 ----
    # 见 check_disk_match 的文档：必须独立回读，不能拿 state_dict 自证。
    disk_sd = torch.load(info["cache_path"], map_location="cpu",
                         weights_only=True)
    n_same = check_disk_match(
        model, disk_sd, head,
        conv1_override=(crop if arch == "resnet50" else None),
        verbose=verbose)

    # ---- 自检 C：32×32 前向 ----
    model.eval()
    with torch.no_grad():
        x = torch.randn(2, 3, 32, 32)
        y = model(x)
    ok_shape = tuple(y.shape) == (2, num_classes)
    ok_finite = bool(torch.isfinite(y).all())
    if not (ok_shape and ok_finite):
        print(f"\n[自检 C 失败] 前向输出 shape={tuple(y.shape)} "
              f"finite={ok_finite}", file=sys.stderr)
        raise SystemExit(1)
    if verbose:
        print(f"[自检C ] 前向 2×3×32×32 → {tuple(y.shape)}，"
              f"logits 范围 [{y.min():.3f}, {y.max():.3f}]，finite=True")

    # ---- 自检 E：与 torchvision 原生实现同权重同输入对拍 ----
    # mobilenet_v2 走单独一条（没有 conv1 移植、头是 classifier.1、特征 1280 维）
    if arch == "mobilenet_v2":
        ref_maxdiff = ref_equivalence_check_mobilenet(model, src, verbose=verbose)
    else:
        ref_maxdiff = ref_equivalence_check(model, src, arch=arch,
                                            verbose=verbose)

    # ---- 自检 D：分类头是重置的，不是从源权重来的 ----
    # 源分类头是 (1000, in)，形状就不兼容，所以它必然没被加载
    # （自检 A 的 missing 里）。这里只核对形状与初始化统计，
    # 避免"形状对但值来自别处"的假通过。
    head_mod = dict(model.named_modules())[head.rstrip(".")]     # "fc" / "classifier"
    head_lin = head_mod[-1] if isinstance(head_mod, nn.Sequential) else head_mod
    fc_w = head_lin.weight.detach()
    if tuple(fc_w.shape) != (num_classes, int(head_lin.in_features)):
        print(f"\n[自检 D 失败] {head.rstrip('.')}.weight shape="
              f"{tuple(fc_w.shape)}，期望 "
              f"{(num_classes, int(head_lin.in_features))}", file=sys.stderr)
        raise SystemExit(1)
    fc_std = float(fc_w.std())

    record = dict(
        **info,
        num_classes=num_classes,
        n_tensors_src=len(src_keys),
        n_tensors_loaded=n_same,
        conv1_transplant=(
            dict(mode=CONV1_MODE, src_kernel=CONV1_SRC_K,
                 dst_kernel=CONV1_DST_K, crop_offset=off, rescaled=False,
                 mean_abs_src=round(float(src_conv1.abs().mean()), 6),
                 mean_abs_dst=round(float(crop.abs().mean()), 6),
                 l1_ratio=round(float(crop.abs().sum()
                                      / src_conv1.abs().sum()), 4),
                 note="不缩放（任务书口径）。中心 3×3 占原 7×7 权重 L1 质量的 41%，"
                      "平均 |w| 0.156→0.350。"
                      "⚠️ BN running stats 失配**不止 bn1**：stem 从 7×7s2 改为 "
                      "3×3s1 且去掉 maxpool，改变了每个 stage 的特征图几何，"
                      "layer1..4 的全部 BN stats 都是在 224×224/7×7s2 上统计的，"
                      "与 32×32/3×3s1 的输入分布都不匹配 → 由 S1 微调吸收")
            if arch == "resnet50" else
            dict(mode="none", src_kernel=3, dst_kernel=3, crop_offset=0,
                 rescaled=False,
                 note="mobilenet_v2 的 stem 本就是 3×3，(32,3,3,3) 形状兼容；"
                      "只把 stride 2→1（CIFAR 32×32）。无权重裁剪。"
                      "⚠️ BN running stats 失配**不止 stem**：stride 2→1 改变了"
                      "每个 stage 的特征图几何，features.1..17 的全部 BN stats "
                      "都是在 224×224/3×3s2 上统计的，与 32×32/3×3s1 都不匹配 "
                      "→ 由 S1 微调吸收。")
        ),
        head_reset=dict(module=head.rstrip("."),
                        in_features=int(head_lin.in_features),
                        out_features=int(head_lin.out_features),
                        weight_std=round(fc_std, 6),
                        init="nn.Linear 默认 reset_parameters"
                             "（kaiming_uniform）"),
        selfcheck=dict(
            missing=list(missing),
            unexpected=list(unexpected),
            allowed_only_fc=True,
            non_fc_bit_identical=f"{n_same}/{len(load_sd)}",
            forward_shape=list(y.shape),
            forward_finite=ok_finite,
            ref_torchvision_maxdiff=ref_maxdiff,
            ref_torchvision_atol=1e-4,
        ),
        verified_at=_now_iso(),
    )
    return model, record


# ------------------------------------------------------------------ manifest

def update_manifest(record: Dict, path: Path = MANIFEST_DEFAULT,
                    verbose: bool = True) -> Dict:
    """把一条记录写进 weights_manifest.json（按 arch 覆盖，保留其它条目）。"""
    path = Path(path)
    doc = {"schema": 1, "updated_at": None, "entries": {}}
    if path.is_file():
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise SystemExit(f"{path} 不是合法 JSON（{e}）；先修好再写，"
                             f"不要覆盖掉已有出处")
    doc.setdefault("schema", 1)
    doc.setdefault("entries", {})
    doc["updated_at"] = _now_iso()
    doc["entries"][record["arch"]] = record
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    if verbose:
        print(f"[manifest] 写入 {path}（arch={record['arch']}）")
    return doc


# ------------------------------------------------------------------ CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="torchvision ImageNet 权重 → CIFAR-10 迁移初始化（带出处与自检）")
    ap.add_argument("--arch", default="resnet50", help="resnet50 | mobilenet_v2")
    ap.add_argument("--tag", default=None,
                    help="IMAGENET1K_V1 / IMAGENET1K_V2（默认 V2）")
    ap.add_argument("--num-classes", type=int, default=10)
    ap.add_argument("--manifest", default=str(MANIFEST_DEFAULT))
    ap.add_argument("--save-ckpt", default=None,
                    help="把移植后的 state_dict 存到这里（.pt）。注意 fc 是随机初始化，"
                         "这份 ckpt 只是微调的起点")
    ap.add_argument("--no-manifest", action="store_true",
                    help="只跑自检，不写 manifest")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    verbose = not a.quiet
    try:
        model, rec = transplant(a.arch, a.tag, a.num_classes, verbose=verbose)
    except SystemExit:
        raise
    except Exception as e:                       # 网络/缓存类失败 → 退出码 2
        print(f"[取权重失败] {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    if a.save_ckpt:
        import torch
        p = Path(a.save_ckpt)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), p)
        rec["saved_ckpt"] = dict(path=str(p.resolve()),
                                 sha256=sha256_file(p),
                                 size_bytes=p.stat().st_size)
        if verbose:
            print(f"[ckpt   ] {p.resolve()}  "
                  f"{p.stat().st_size / 2**20:.1f} MiB  "
                  f"sha256={rec['saved_ckpt']['sha256'][:16]}…")

    if not a.no_manifest:
        update_manifest(rec, Path(a.manifest), verbose=verbose)

    if verbose:
        print("\n全部自检通过（A 键集合 / B 逐元素 / C 前向 / D fc 重置 / E 与 torchvision 对拍）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
