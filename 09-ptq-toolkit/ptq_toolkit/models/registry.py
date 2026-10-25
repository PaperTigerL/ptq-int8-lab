# -*- coding: utf-8 -*-
"""模型 registry —— `--model` 的唯一入口。

**加模型 = 在 REGISTRY 里加一条**，pipeline / export / report 都不改。

⭐ 拓扑格式的分工（为什么 ResNet18 与另两个模型走不同格式）：
   `PTQW`（06 的格式）：`blocks = (c1, c2, ds, stride)`，
       只能表达「两层 conv + 一个可选 downsample 残差」——ResNet-18 专用。
       **保持不动**，因为 06 的引擎二进制要能直接读它。

   `PTQX`（通用）：指令序列 `CONV / ADD / MUL / POOL / MEAN`，
       能表达任意拓扑 —— bottleneck（ResNet-50，三层 conv）、
       可选 skip 的 inverted residual（MobileNetV2）、SE 的乘法等。

   所以 registry 里记 `topo_kind`，export 层据此选格式。
"""
from dataclasses import dataclass
from typing import Callable, Dict, List

import torch.nn as nn


@dataclass(frozen=True)
class ModelSpec:
    """一个模型的完整描述。"""
    key: str                  # CLI 里的名字，如 "resnet50"
    display: str              # 报告显示的名字
    build: Callable           # () -> nn.Module
    topo_kind: str            # "blocks"（PTQW）或 "seq"（PTQX）
    source: str               # 模型定义出处（README 要引用）
    topo_plan_fn: Callable | None = None   # () -> 残差声明（PTQX 用）
    weights_note: str = ""    # 权重来源说明


# ---------------------------------------------------------------- 三个模型

def _resnet18_build(num_classes: int = 10, **_) -> nn.Module:
    """ResNet-18 复用 03-ptq-lab 的定义 —— 它是量化语义的 ground truth，
    自己重定义一份会让精度口径与 03/06/07 分叉。"""
    import sys
    from pathlib import Path
    for p in Path(__file__).resolve().parents:
        src = p / "03-ptq-lab"
        if (src / "src" / "model.py").is_file():
            for d in (src, src / "src"):
                if str(d) not in sys.path:
                    sys.path.insert(0, str(d))
            break
    import model as M03
    return M03.load_fp32(num_classes=num_classes)


def _resnet50_build(num_classes: int = 10, **_) -> nn.Module:
    from . import resnet50
    return resnet50.build(num_classes)


def _mobilenet_build(num_classes: int = 10, **_) -> nn.Module:
    from . import mobilenet_v2
    return mobilenet_v2.build(num_classes)


REGISTRY: Dict[str, ModelSpec] = {
    "resnet18": ModelSpec(
        key="resnet18", display="ResNet-18",
        build=_resnet18_build, topo_kind="blocks",
        source="03-ptq-lab/src/model.py（量化语义 ground truth）",
        topo_plan_fn=lambda: __import__(
            "ptq_toolkit.models.resnet18_plan",
            fromlist=["topo_plan"]).topo_plan(),
        weights_note="50 ep 训练，fp32 top-1 = 87.16%"),
    "resnet50": ModelSpec(
        key="resnet50", display="ResNet-50",
        build=_resnet50_build, topo_kind="seq",
        source="ptq_toolkit/models/resnet50.py",
        topo_plan_fn=lambda: __import__(
            "ptq_toolkit.models.resnet50", fromlist=["topo_plan"]).topo_plan(),
        weights_note="本项目定义（CIFAR 适配：3×3s1 stem、无 maxpool），随机初始化"),
    "mobilenet_v2": ModelSpec(
        key="mobilenet_v2", display="MobileNetV2",
        build=_mobilenet_build, topo_kind="seq",
        source="ptq_toolkit/models/mobilenet_v2.py",
        topo_plan_fn=lambda: __import__(
            "ptq_toolkit.models.mobilenet_v2",
            fromlist=["topo_plan"]).topo_plan(),
        weights_note="本项目定义（t=1，CIFAR 适配：3×3s1 stem），随机初始化"),
}


def list_models() -> List[ModelSpec]:
    return [REGISTRY[k] for k in sorted(REGISTRY)]


def get_model(key: str, num_classes: int = 10, **kw) -> nn.Module:
    """按名字构造模型。名字不存在时列出全部可选名 —— 报错要说清怎么改。"""
    if key not in REGISTRY:
        raise KeyError(f"未知模型 {key!r}；可选：{', '.join(sorted(REGISTRY))}")
    return REGISTRY[key].build(num_classes=num_classes, **kw)


def topo_kind(key: str) -> str:
    """该模型该用哪种拓扑格式。"""
    if key not in REGISTRY:
        raise KeyError(f"未知模型 {key!r}；可选：{', '.join(sorted(REGISTRY))}")
    return REGISTRY[key].topo_kind


def topo_plan(key: str) -> list:
    """该模型的残差声明（只有 PTQX 模型需要）。

    ⭐ 由**模型自己声明**，不是通用层猜的 ——
       自动推断试了 6 轮都错位，见 export/toposeq.py 的模块注释。
    """
    if key not in REGISTRY:
        raise KeyError(f"未知模型 {key!r}；可选：{', '.join(sorted(REGISTRY))}")
    fn = REGISTRY[key].topo_plan_fn
    return fn() if fn else []
