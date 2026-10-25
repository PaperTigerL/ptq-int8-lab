# -*- coding: utf-8 -*-
"""W4 候选配置：per-group / per-channel / 混合精度。

为什么需要这一层：
  · W4 per-channel 在 MobileNetV2 上掉 5.22 pp（M2 实测），
    其中 depthwise 与首尾层的动态范围差异最大；
  · 把所有层都压到 W4 不是唯一选项，常见做法是**保护敏感层**：
    首/末 conv、downsample、depthwise 保留 W8，其余 W4。
  · 本模块只负责「每个 conv 用哪个位宽/粒度」的**声明**，
    真正的量化与导出在 pipeline / quant.ptq 里完成。

⛔ 本模块的候选是否能达标，必须用引擎在 BOSS 的 GPU 窗口实测。
   这里不跑精度、不猜结论 —— 只把候选配置固定下来，保证每次导出可复现。
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch.nn as nn


@dataclass(frozen=True)
class LayerQuantSpec:
    """单层权重配置。"""
    bits: int
    scheme: str = "per_channel"     # per_channel / per_group
    group_size: int = 0             # 仅 per_group 使用

    def __post_init__(self):
        if self.bits not in (4, 8):
            raise ValueError(f"bits 只支持 4/8，收到 {self.bits}")
        if self.scheme not in ("per_channel", "per_group"):
            raise ValueError(
                f"scheme 只支持 per_channel/per_group，收到 {self.scheme!r}")
        if self.scheme == "per_group" and self.group_size <= 0:
            raise ValueError("per_group 必须给正的 group_size")

    def to_dict(self) -> Dict:
        d = {"bits": self.bits, "scheme": self.scheme}
        if self.scheme == "per_group":
            d["group_size"] = self.group_size
        return d


@dataclass(frozen=True)
class Candidate:
    """一个完整的候选配置。

    protect_* 为 True 时，命中的 conv 改用 `protected`（默认 W8 per-channel）。
    """
    key: str
    display: str
    default: LayerQuantSpec
    protect_first: bool = False
    protect_last: bool = False
    protect_downsample: bool = False
    protect_depthwise: bool = False
    protected: LayerQuantSpec = LayerQuantSpec(8, "per_channel", 0)
    # 显式 per-layer W8 覆盖（来自 I2 敏感度计划或 CLI/JSON）。
    # 这些层用 protected（W8 per-channel），其余用 default；不做规则推断。
    w8_layers: Tuple[str, ...] = ()
    note: str = ""

    def to_dict(self) -> Dict:
        return {
            "key": self.key,
            "display": self.display,
            "default": self.default.to_dict(),
            "protect_first": self.protect_first,
            "protect_last": self.protect_last,
            "protect_downsample": self.protect_downsample,
            "protect_depthwise": self.protect_depthwise,
            "protected": self.protected.to_dict(),
            "w8_layers": list(self.w8_layers),
            "note": self.note,
        }


# 候选集合。顺序固定，导出/报告都按这个顺序，便于横向对比。
CANDIDATES: Dict[str, Candidate] = {
    "w8a8": Candidate(
        key="w8a8", display="W8A8 per-channel（基线）",
        default=LayerQuantSpec(8, "per_channel"),
        note="与既有 06/09 口径一致；用于验证新导出链路不引入偏差。"),
    "w4a8_ch": Candidate(
        key="w4a8_ch", display="W4A8 per-channel",
        default=LayerQuantSpec(4, "per_channel"),
        note="M2 已测：R50 掉 3.22 pp / MV2 掉 5.22 pp，均超 3 pp 预算。"),
    "w4a8_g32": Candidate(
        key="w4a8_g32", display="W4A8 per-group (group_size=32)",
        default=LayerQuantSpec(4, "per_group", 32),
        note="组粒度沿展平 K 切；K<32 的层退化为 per-channel。"),
    "w4a8_mixed_g32": Candidate(
        key="w4a8_mixed_g32",
        display="W4A8 per-group 32 + 敏感层 W8",
        default=LayerQuantSpec(4, "per_group", 32),
        protect_first=True, protect_last=True,
        protect_downsample=True, protect_depthwise=True,
        protected=LayerQuantSpec(8, "per_channel"),
        note="首/末 conv、downsample、depthwise 保留 W8 per-channel；"
             "其余 W4 per-group 32。"),
}


def candidate_keys() -> List[str]:
    return list(CANDIDATES)


def get_candidate(key: str) -> Candidate:
    if key not in CANDIDATES:
        raise KeyError(
            f"未知候选 {key!r}；可选：{', '.join(CANDIDATES)}")
    return CANDIDATES[key]


def _is_depthwise(m: nn.Conv2d) -> bool:
    """真 depthwise：groups == in_channels 且 groups > 1。

    ⛔ 不能只看 groups>1：一般分组卷积（groups=2..C/2）也会命中，
       把整层抬到 W8 会改变候选的语义。
    """
    return m.groups > 1 and m.groups == m.in_channels


def build_layer_plan(model: nn.Module,
                     candidate: Candidate) -> "Dict[str, LayerQuantSpec]":
    """按候选规则给每个 Conv2d 分配配置（保持 named_modules 定义顺序）。

    首/末 conv 用**定义顺序**判定，与导出/PTQC 的 conv 顺序同源；
    downsample 按层名含 "downsample" 判定（R50 的 four projections）。
    """
    convs = [(n, m) for n, m in model.named_modules()
             if isinstance(m, nn.Conv2d)]
    if not convs:
        raise RuntimeError("模型里没有 Conv2d，无法生成量化计划")
    first_name, last_name = convs[0][0], convs[-1][0]

    plan: Dict[str, LayerQuantSpec] = {}
    explicit = set(candidate.w8_layers)
    for name, m in convs:
        spec = candidate.default
        hit = (
            name in explicit
            or (candidate.protect_first and name == first_name)
            or (candidate.protect_last and name == last_name)
            or (candidate.protect_downsample and "downsample" in name)
            or (candidate.protect_depthwise and _is_depthwise(m))
        )
        plan[name] = candidate.protected if hit else spec
    unknown = explicit - set(plan)
    if unknown:
        raise ValueError(
            f"显式 W8 层里有 {len(unknown)} 个不是 Conv2d 层名："
            f"{sorted(unknown)[:6]}{' ...' if len(unknown) > 6 else ''}")
    return plan


def plan_summary(plan: Dict[str, LayerQuantSpec]) -> Dict:
    """统计每种配置的层数，并列出被保护层名（用于导出报告）。"""
    counts: Dict[Tuple, int] = {}
    protected = []
    for name, spec in plan.items():
        key = (spec.bits, spec.scheme, spec.group_size)
        counts[key] = counts.get(key, 0) + 1
    # 被保护层 = 配置与默认不同的层；由调用方传入默认值时不强求，
    # 这里只按「W8 per-channel」列出，够报告定位。
    for name, spec in plan.items():
        if spec.bits == 8 and spec.scheme == "per_channel":
            protected.append(name)
    rows = []
    for (bits, scheme, gs), n in sorted(counts.items()):
        rows.append({"bits": bits, "scheme": scheme,
                     "group_size": gs, "layers": n})
    return {"counts": rows, "w8_layers": protected,
            "n_layers": len(plan)}


def plan_layer_list(plan: Dict[str, LayerQuantSpec]) -> List[Dict]:
    """把 plan（保持 named_modules 顺序）展开成可写 JSON 的逐层列表。"""
    return [dict(name=name, **spec.to_dict()) for name, spec in plan.items()]


def explicit_plan(model: nn.Module, w8_layers,
                  w4_group_size: int = 32,
                  w4_scheme: str = "per_group") -> Dict[str, LayerQuantSpec]:
    """**显式 per-layer 覆盖**：名单内 W8 per-channel，其余 W4。

    这是 I2 推荐计划（top4_all：4 层 W8 + 49 层 W4）的落地入口，
    与 CANDIDATES 的「按规则保护」不同 —— 名单由外部证据给定，不做推断。

    校验：
      · 名单里每个名字都必须是模型里真实存在的 Conv2d（拼错直接报错，
        静默忽略会让少数一层保持 W4，掉点却找不出原因）；
      · 名字允许重复（去重），不允许非字符串/空串。
    """
    if isinstance(w8_layers, str):
        w8_layers = [s.strip() for s in w8_layers.split(",") if s.strip()]
    names = [n for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    name_set = set(names)
    w8_list: List[str] = []
    for n in w8_layers:
        if not isinstance(n, str) or not n:
            raise ValueError(f"W8 层名必须是非空字符串，收到 {n!r}")
        if n not in name_set:
            raise ValueError(f"W8 层 {n!r} 不是模型的 Conv2d 层名")
        if n not in w8_list:
            w8_list.append(n)
    w4 = LayerQuantSpec(4, w4_scheme, w4_group_size)
    w8 = LayerQuantSpec(8, "per_channel")
    w8_set = set(w8_list)
    return {n: (w8 if n in w8_set else w4) for n in names}


def make_explicit_candidate(key: str, w8_layers,
                            w4_group_size: int = 32,
                            w4_scheme: str = "per_group",
                            display: Optional[str] = None,
                            note: str = "") -> Candidate:
    """由显式 W8 名单构造一个 Candidate（供 export/eval 复用同一套流程）。"""
    w8 = tuple(w8_layers)
    return Candidate(
        key=key,
        display=display or f"显式 per-layer 混合精度（W8 {len(w8)} 层 / 其余 W4）",
        default=LayerQuantSpec(4, w4_scheme, w4_group_size),
        protected=LayerQuantSpec(8, "per_channel", 0),
        w8_layers=w8,
        note=note,
    )


def w8_layers_from_json(path, plan_name: Optional[str] = None):
    """从 I2 敏感度 JSON（或简单 `{"w8_layers": [...]}`）读 W8 层名单。

    返回 (w8_layers, meta)。I2 文件里有 `plans` 列表时：
      · plan_name 为 None → 用 `recommended.plan`；
      · 指定 plan_name → 从 plans 里精确匹配，找不到直接报错（不退回默认）。
    这样命令里写错计划名不会静默跑成别的计划。
    """
    p = Path(path)
    doc = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(doc, list):
        doc = {"w8_layers": doc}
    if not isinstance(doc, dict):
        raise ValueError(f"{p}: 期望 JSON 对象或数组，收到 {type(doc).__name__}")

    rec_doc = doc.get("recommended") or doc.get("recommendation") or {}
    rec: Dict = {}
    plans = doc.get("plans")
    if isinstance(plans, list) and plans:
        by_name = {x.get("name"): x for x in plans if isinstance(x, dict)}
        name = plan_name or rec_doc.get("plan")
        if name is None:
            name = min(plans, key=lambda x: (x.get("n_w8", 1 << 30),
                                             x.get("drop_pp", 0.0))).get("name")
        if name not in by_name:
            raise KeyError(
                f"{p}: 没有计划 {name!r}；可选：{', '.join(sorted(by_name))}")
        rec = by_name[name]
        w8 = rec.get("w8_layers")
        meta = dict(source=str(p), plan=name,
                    n_w8=rec.get("n_w8"), n_w4=rec.get("n_w4"),
                    drop_pp=rec.get("drop_pp"), acc=rec.get("acc"),
                    measured_on=doc.get("measured_on"),
                    selection_basis=rec_doc.get("selection_basis"))
    else:
        w8 = doc.get("w8_layers")
        meta = dict(source=str(p), plan=plan_name or "custom")
        if plan_name is not None and plan_name not in (None, "custom"):
            # 简单 JSON 没有计划表：只允许 key 显式匹配
            if doc.get("plan") not in (None, plan_name):
                raise KeyError(
                    f"{p}: 文件里的 plan={doc.get('plan')!r} 与请求 "
                    f"{plan_name!r} 不一致")

    if w8 is None and plan_name is None and rec_doc.get("w8_layers"):
        w8 = rec_doc["w8_layers"]
        meta = dict(source=str(p), plan=rec_doc.get("plan") or "custom",
                    n_w8=rec_doc.get("n_w8"), n_w4=rec_doc.get("n_w4"),
                    drop_pp=rec_doc.get("drop_pp"), acc=rec_doc.get("acc"),
                    measured_on=doc.get("measured_on"),
                    selection_basis=rec_doc.get("selection_basis"))
    # W4 粒度随计划变化（I3 的 MV2 计划用 per-group16，不是 I2 的 32）。
    # 把计划记录里的 group_size/scheme 一并带出去，让 CLI 的 --w4-group
    # 默认值能跟随证据，而不是默默退回 32。
    for field in ("w4_group_size", "w4_scheme"):
        if meta.get(field) is None:
            meta[field] = rec.get(field, doc.get(field))

    if not isinstance(w8, list):
        raise ValueError(f"{p}: w8_layers 必须是数组")
    out: List[str] = []
    for n in w8:
        if not isinstance(n, str) or not n:
            raise ValueError(f"{p}: w8_layers 里有非法层名 {n!r}")
        if n not in out:
            out.append(n)
    if not out:
        raise ValueError(f"{p}: w8_layers 为空 —— 全 W4 不需要混合精度计划")
    meta["w8_layers"] = list(out)
    return out, meta
