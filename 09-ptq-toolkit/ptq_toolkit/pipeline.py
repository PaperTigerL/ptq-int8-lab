# -*- coding: utf-8 -*-
"""M1 主流水线：PyTorch 模型 + 校准数据 → PTQ → 导出 → 基准 → 报告。

**复用 03-ptq-lab 的三件套**（不重造）：
  · `fold_all_bn`    BN 折叠 —— 06/07 的所有测量都基于折叠后的权重
  · `evaluate`       精度评测 —— 保证与 03/06/07 同一口径
  · `calibrate`      激活 range —— 与 06 的 export_calib.py 同源

自己实现的只有两件：
  · `ptq.quantize_per_channel_sym`  可部署的 per-channel 权重量化
  · `exporter.export_weights`        权重 + 拓扑导出（格式对齐 06）

⚠️ 关于精度的口径声明（本工具的已知偏差，golden test 会遇到）：
   本模块的 `quant_acc` 是 **权重-only fake-quant** 的精度，
   **不含激活量化**——激活量化在 C++ 引擎侧才真正生效。
   所以这里的 quant_acc 会**高于**真实 W8A8 引擎的精度
   （W8 权重损失约 0.1 pp，W4 约 2 pp）。
   golden test 对齐的是**引擎侧的数字**，不是这里的。
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------- 03 的复用

def _ptq_lab_dir() -> Path:
    """定位 03-ptq-lab（相对本文件，保证可移植）。"""
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "03-ptq-lab"
        if (cand / "src" / "model.py").is_file():
            return cand
    raise FileNotFoundError(
        "找不到 03-ptq-lab（PTQ 语义的 ground truth）。"
        "本工具依赖它做 BN 折叠、校准与精度评测，不重复实现——"
        "重复实现会导致精度口径与 03/06/07 不一致。")


def _ensure_ptq_on_path() -> None:
    """把 03 的两个目录都挂上。

    ⚠️ 两个都要：`03-ptq-lab/`（放 config.py）与 `03-ptq-lab/src/`（放 model.py）。
       只加 src 会报 `ModuleNotFoundError: No module named 'config'` ——
       这是 03 自己的布局决定的，不是笔误。
    """
    import sys
    root = _ptq_lab_dir()
    for d in (root, root / "src"):
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))


def load_ptq_lab():
    """加载 03-ptq-lab 的关键函数。"""
    _ensure_ptq_on_path()
    import calib as _calib          # noqa: E402
    import eval as _eval            # noqa: E402
    import model as _model          # noqa: E402
    return dict(fold_all_bn=_model.fold_all_bn,
                load_fp32=_model.load_fp32,
                evaluate=_eval.evaluate,
                calibrate=_calib.calibrate,
                root=str(_ptq_lab_dir()))


# ---------------------------------------------------------------- 数据

def get_loaders(calib_n: int = 512, eval_n: int = 0):
    """取训练集（校准用）与测试集（评测用）。

    口径与 03/06/07 一致：归一化 (x/255 - mean)/std。

    ⚠️ eval_n > 0 时只取测试集的**前 eval_n 张**。
       M2 必须用它：ResNet-50 在 CPU 上跑全量 10000 张要约 20 小时
       （实测单 batch 256 张 7.3 s），而它/MobileNetV2 是**随机初始化**的，
       精度恒在 10% 附近（全量与子集没有区别），全量评测纯属浪费。
       ResNet-18 是训练过的权重，**必须全量评测**（eval_n=0），
       否则它的 87% 与别人不可比。
    """
    _ensure_ptq_on_path()
    import config as C               # noqa: E402
    import data as D                 # noqa: E402

    _, testset = D.load_datasets()
    if eval_n and eval_n < len(testset):
        testset = torch.utils.data.Subset(testset, list(range(eval_n)))
    test = torch.utils.data.DataLoader(
        testset, batch_size=C.EVAL_BATCH, shuffle=False, num_workers=0)

    trainset, _ = D.load_datasets()
    idx = list(range(min(calib_n, len(trainset))))
    calib = torch.utils.data.DataLoader(
        torch.utils.data.Subset(trainset, idx),
        batch_size=C.CALIB_BATCH, shuffle=False, num_workers=0)
    return calib, test, len(trainset), len(testset)


def get_train_loader(batch_size: int = 64, shuffle: bool = True,
                     num_workers: int = 2, pin_memory: bool = True,
                     augment: bool = False):
    """**训练专用**的全量训练集 DataLoader。

    augment=True 时给训练集加随机增强（RandomCrop(32, padding=4) +
    RandomHorizontalFlip），**仅训练集**。

    ⛔ 增强绝不能进校准集与测试集：
       校准是「统计激活分布」—— 随机裁剪会改变分布，量化参数就错了；
       评测是「测精度」—— 增强会让 top-1 与 03/06/07 的口径不可比。
       所以这里只改**本函数自己拿到的那个 trainset 对象**；
       `get_loaders()` 内部另调一次 `load_datasets()`，拿到的是独立对象，
       不受影响。

    增强必须**插在 03 的 transform 之前**：03 的 `train_transform()` 是
    `Compose([ToTensor(), Normalize])`，而 RandomCrop/RandomHorizontalFlip
    吃的是 PIL 图 —— 插到 ToTensor 之后就无效了（且不会报错，静默失效）。

    ⛔⛔ 为什么必须有这个函数（一个真实的协议缺陷）：
       `get_loaders(calib_n)` 的第一个返回值是
       `Subset(trainset, range(calib_n))` —— 那是**校准集**，
       不是训练集。曾经的 finetune.py 直接把它当 train_loader 用，
       于是 `n_train` 永远是 calib_n（5120），
       而 history 里记的 n_train=5120 看起来还「挺合理」，
       结果跑出来的 90.06% 只是**子集微调**的中间结果。

       症状之所以隐蔽：校准集也是从训练集前 N 张取的、
       batch 也是对的、精度也在涨 —— 只有 n_train 一个字段暴露它。

    本函数只用全量 trainset，且调用方必须校验 n_train == 50000
    （见 finetune.py 的 SystemExit 闸门）。
    """
    _ensure_ptq_on_path()
    import data as D                 # noqa: E402

    trainset, _ = D.load_datasets()
    if augment:
        import torchvision.transforms as T
        base = trainset.transform
        tail = list(base.transforms) if isinstance(base, T.Compose) \
            else [base]
        trainset.transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            *tail,                   # 03 的 ToTensor + Normalize 必须留在最后
        ])
    return torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0)


def get_n_train() -> int:
    """训练集全量样本数（CIFAR-10 应为 50000）。用于开跑前校验。"""
    _ensure_ptq_on_path()
    import data as D                 # noqa: E402
    trainset, _ = D.load_datasets()
    return len(trainset)


# ---------------------------------------------------------------- 显存保护

def gpu_mem_guard(need_bytes: int, device: str = "cuda") -> None:
    """device=cuda 时检查显存是否够用；不够就报错退出。

    M2 验收判据第 7 条要求「能用 nvidia-smi 前后对比证明没有我们的计算进程」，
    所以本工具**默认 device=cpu**，这个检查只在显式要上 GPU 时才触发。
    留在代码里是为了以后接GPU 引擎时不再踩 OOM。
    """
    if not str(device).startswith("cuda"):
        return
    try:
        import torch
        free_b, total_b = torch.cuda.mem_get_info()
    except Exception:
        return                                  # 拿不到就放行
    if free_b < need_bytes + total_b // 10:       # 留 10% 余量
        raise RuntimeError(
            f"显存不足：需要约 {need_bytes / 2**20:.0f} MB，"
            f"可用 {free_b / 2**20:.0f} MB / 总 {total_b / 2**20:.0f} MB。"
            f"请减小 --calib-n 或改用 --device cpu。")


# ---------------------------------------------------------------- 主流程

@dataclass
class BuildResult:
    """`ptq build` 的产物。"""
    w_bits: int
    a_bits: int
    fp32_acc: float
    quant_acc: float
    exported: Dict = field(default_factory=dict)
    quant_cfg: Dict = field(default_factory=dict)
    ptqc: Dict = field(default_factory=dict)
    act_ranges: Dict = field(default_factory=dict)
    n_calib_batches: int = 0
    elapsed_s: float = 0.0
    model_key: str = "resnet18"
    model_name: str = "ResNet-18"
    topo_kind: str = "blocks"

    @property
    def acc_drop(self) -> float:
        return self.fp32_acc - self.quant_acc

    def to_dict(self) -> Dict:
        return dict(model=self.model_key, model_name=self.model_name,
                    topo_kind=self.topo_kind,
                    w_bits=self.w_bits, a_bits=self.a_bits,
                    fp32_acc=self.fp32_acc, quant_acc=self.quant_acc,
                    acc_drop=self.acc_drop,
                    exported=self.exported, quant_cfg=self.quant_cfg,
                    ptqc=self.ptqc,
                    n_calib_batches=self.n_calib_batches,
                    elapsed_s=round(self.elapsed_s, 2))


@dataclass
class CandidateExport:
    """`ptq export` 的单个候选产物。"""
    model_key: str
    model_name: str
    candidate: str
    display: str
    n_conv: int
    n_block: int
    exported: Dict = field(default_factory=dict)
    quant_cfg: Dict = field(default_factory=dict)
    ptqc: Dict = field(default_factory=dict)
    plan: Dict = field(default_factory=dict)
    layer_plan: Dict = field(default_factory=dict)
    manifest: Dict = field(default_factory=dict)
    selfcheck: Dict = field(default_factory=dict)
    ptq2_info: Dict = field(default_factory=dict)
    elapsed_s: float = 0.0

    def to_dict(self) -> Dict:
        return dict(model=self.model_key, model_name=self.model_name,
                    candidate=self.candidate, display=self.display,
                    n_conv=self.n_conv, n_block=self.n_block,
                    exported=self.exported, quant_cfg=self.quant_cfg,
                    ptqc=self.ptqc, plan=self.plan,
                    layer_plan=self.layer_plan,
                    ptq2_info=self.ptq2_info,
                    elapsed_s=round(self.elapsed_s, 2),
                    accuracy="未评测（本阶段 CPU-only，只做导出校验）")


def export_candidates(model_key: str,
                      weights=None,
                      candidate_keys=None,
                      calib_n: int = 512,
                      a_bits: int = 8,
                      out_dir="results/w4_candidates",
                      device: str = "cpu",
                      verbose: bool = True,
                      w8_layers=None,
                      w4_group_size: int = 32,
                      w4_scheme: str = "per_group",
                      plan_name=None,
                      plan_meta: Dict | None = None) -> List[CandidateExport]:
    """按候选配置导出 PTQ2 + PTQC（**不跑精度**）。

    为什么单独一个入口而不复用 `build()`：
      build 会跑 fp32 / 量化后精度（R50 全量 CPU 要 20 小时），
      本阶段只要「权重 + 校准 + 拓扑 + 文件校验」。
      校准只跑 512 张，是 PTQC 的唯一来源，不做任何评测。

    ⭐ 权重导出的是 **QDQ 后的 fp32**：
       PTQ2 的 conv 权重字段是 fp32，没有 bits/scale 元数据。
       所以 W4/per-group 的实现方式是「先量化再反量化」，写出的权重
       已落在 W4 网格上；06 引擎再按 int8 per-channel 重编码时，
       int8 的 127 级远细于 W4 的 7 级，能忠实保留 W4 网格。
       代价：文件仍是 fp32 大小，**W4 只改变数值分布，不改变存储/算力**。
       真正的 W4 存储压缩需要扩展 PTQ2 格式（本阶段不做，报告里列为未验证项）。

    ⭐ fc 不量化：PTQ2 的 fc 是 fp32 字段，06 引擎走 fp32 cuBLAS。
       若把 fc 也 QDQ，等于在引擎里测「量化后的 fc」，与部署路径不符。

    ⭐ 显式 per-layer 覆盖：传 `w8_layers=[...]` 时按名单生成混合精度计划
       （名单内 W8 per-channel，其余 W4 per-group `w4_group_size`），
       用于落地 I2 的 top4_all。此时只导出一个候选，并额外写出
       `layer_plan_*.json` + `manifest.json`，导出后按字节回读自检
       （`export/mixed_check.py`）；自检不过直接抛异常，不产出未校验的包。
    """
    import copy
    import time
    from types import SimpleNamespace

    from .export import exporter, mixed_check, ptqc, topoblocks
    from .export.ptq2_info import read_ptq2_info
    from .models import get_model, list_models, topo_kind, topo_plan
    from .quant.mixed import (build_layer_plan, candidate_keys as all_keys,
                              get_candidate, make_explicit_candidate,
                              plan_layer_list, plan_summary)
    from .quant.ptq import ActQuantSpec, quantize_weight_sym

    if topo_kind(model_key) == "blocks":
        raise ValueError(
            f"{model_key} 走 PTQW/blocks 格式，不属于本候选导出（PTQ2）。"
            f"可选：resnet50 / mobilenet_v2")

    spec = next(s for s in list_models() if s.key == model_key)
    explicit = w8_layers is not None
    if explicit:
        if candidate_keys is not None:
            raise ValueError("w8_layers 与 candidate_keys 只能二选一")
        key = (plan_meta or {}).get("plan") or plan_name or "explicit"
        cands = [make_explicit_candidate(
            key, w8_layers, w4_group_size, w4_scheme=w4_scheme,
            note=(plan_meta or {}).get("selection_basis") or "")]
    else:
        if candidate_keys is None:
            candidate_keys = all_keys()
        if isinstance(candidate_keys, str):
            candidate_keys = [k.strip() for k in candidate_keys.split(",")
                              if k.strip()]
        cands = [get_candidate(k) for k in candidate_keys]

    t_all = time.time()
    lab = load_ptq_lab()
    model = get_model(model_key)

    if weights:
        n_sd, miss, unexp = load_trained_weights(model, weights)
        if miss or unexp:
            raise RuntimeError(
                f"加载 {weights} 后键集合不干净：missing={miss} "
                f"unexpected={unexp}")
        if verbose:
            print(f"加载训练权重 {weights}（{n_sd} 个张量，键集合干净）")

    calib, _test, n_train, _n_test = get_loaders(calib_n=calib_n, eval_n=1)
    model = lab["fold_all_bn"](model.to(device).eval()).to(device).eval()
    if verbose:
        print(f"模型 {spec.display}（{model_key}）· 校准集 {calib_n} 张"
              f"（训练集共 {n_train}）· 只导出，不评测")

    a_scales, a_zeros, act_store = lab["calibrate"](
        model, calib, a_bits, strategy="minmax", symmetric=True, device=device)
    act_specs = {k: ActQuantSpec(scale=float(a_scales[k]),
                                 zero=int(a_zeros[k]), bits=a_bits)
                 for k in a_scales}
    if verbose:
        print(f"激活校准完成：{len(a_scales)} 层（PTQC 写原始 minmax）")

    xb, _ = next(iter(calib))
    x0 = xb[:2].to(device)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results: List[CandidateExport] = []

    for cand in cands:
        t0 = time.time()
        m = copy.deepcopy(model)
        plan = build_layer_plan(m, cand)
        quant = {}
        for name, mod in m.named_modules():
            if not isinstance(mod, nn.Conv2d):
                continue
            cfg = plan[name]
            w = mod.weight.detach().cpu().numpy()
            q = quantize_weight_sym(w, cfg.bits, cfg.scheme, cfg.group_size)
            quant[name] = q
            new_w = torch.from_numpy(q.dequant()).to(
                mod.weight.device, mod.weight.dtype)
            mod.weight.data.copy_(new_w.reshape(mod.weight.shape))

        stem = f"{model_key}_{cand.key}"
        exp = topoblocks.export_blocks(
            m, out / f"{stem}.ptq2.bin", x0, topo_plan(model_key))
        pq = ptqc.write_ptqc_for_model(
            out / f"calib_{stem}.ptqc", m, act_store,
            n_expected=exp["n_conv"])

        # 导出校验：按 PTQ2 布局完整重读，终点必须等于文件长度。
        info = read_ptq2_info(exp["path"])
        if info["n_conv"] != exp["n_conv"]:
            raise RuntimeError(
                f"{stem}: PTQ2 重读 n_conv={info['n_conv']} != 导出 "
                f"{exp['n_conv']}")
        if info["n_conv"] != pq["n"]:
            raise RuntimeError(
                f"{stem}: PTQC n={pq['n']} != PTQ2 n_conv={info['n_conv']}")
        if info["n_conv"] != len(plan):
            raise RuntimeError(
                f"{stem}: PTQ2 conv 数 {info['n_conv']} != 量化计划层数 "
                f"{len(plan)}")

        summary = plan_summary(plan)
        layer_plan = plan_layer_list(plan)
        w8_names = [e["name"] for e in layer_plan if int(e["bits"]) == 8]
        plan_path = out / f"layer_plan_{stem}.json"
        mixed_check.write_layer_plan(
            plan_path, model_key, cand.key, layer_plan,
            plan_meta={**(plan_meta or {}), "candidate": cand.to_dict()})

        cfg = exporter.write_quant_config(
            SimpleNamespace(w_bits=cand.default.bits, a_bits=a_bits,
                            weights=quant, acts=act_specs),
            out / f"quant_config_{stem}.json",
            extra={
                "model": model_key, "model_name": spec.display,
                "topo_kind": spec.topo_kind,
                "candidate": cand.to_dict(),
                "plan_summary": summary,
                "layer_plan": {"path": str(plan_path), "n": len(layer_plan),
                               "w8_layers": w8_names},
                "calib_n": calib_n,
                "ptqc": {"path": pq["path"], "n": pq["n"]},
                "fc": "fp32（PTQ2 的 fc 不量化；本导出只 QDQ conv）",
                "note": "权重已 QDQ 到 W4/W8 网格；PTQ2 不携带 bits/scale，"
                        "per-layer bits/per-group scale 记录在 quant_config "
                        "与 layer_plan。精度未评测。",
            })

        rec = CandidateExport(
            model_key=model_key, model_name=spec.display,
            candidate=cand.key, display=cand.display,
            n_conv=exp["n_conv"], n_block=exp["n_block"],
            exported=exp, quant_cfg=cfg, ptqc=pq, plan=summary,
            layer_plan={"path": str(plan_path), "n": len(layer_plan),
                        "counts": summary["counts"],
                        "w8_layers": w8_names},
            ptq2_info={k: info[k] for k in
                       ("bytes", "n_conv", "n_block", "ncp", "stem_pool",
                        "n_grouped", "n_depthwise", "n_relu1", "n_relu6",
                        "n_relu0", "n_residual1", "n_residual2",
                        "n_residual0")},
            elapsed_s=time.time() - t0,
        )
        build_path = out / f"build_{stem}.json"
        build_path.write_text(
            json.dumps(rec.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8")

        if explicit:
            mixed_check.write_manifest(
                out, model=model_key, plan_key=cand.key,
                layer_plan=layer_plan,
                files={"weights": exp["path"], "calib": pq["path"],
                       "quant_config": cfg["path"],
                       "layer_plan": str(plan_path),
                       "build": str(build_path)},
                exported=exp, source_weights=weights,
                plan_meta=plan_meta,
                topology_maxdiff=exp["selfcheck_maxdiff"])
            rep = mixed_check.check_package(out)
            (out / "selfcheck.json").write_text(
                json.dumps(rep, indent=2, ensure_ascii=False),
                encoding="utf-8")
            rec.manifest = {"path": str(out / "manifest.json"),
                            "n_w8": rep.get("n_w8"),
                            "n_w4": rep.get("n_w4"),
                            "files": (json.loads(
                                (out / "manifest.json").read_text(
                                    encoding="utf-8"))["files"])}
            rec.selfcheck = rep
            if not rep["ok"]:
                bad = [c for c in rep["checks"] if not c["ok"]]
                raise RuntimeError(
                    f"{stem}: 混合精度包自检失败（{len(bad)} 项）："
                    + "; ".join(f"{c['name']}={c['detail']}"
                                for c in bad[:6]))

        results.append(rec)
        if verbose:
            print(f"  [{cand.key}] {exp['path']} "
                  f"（{exp['n_conv']} conv / {exp['n_block']} block / "
                  f"{exp['bytes'] / 1e6:.1f} MB，自检 "
                  f"max|diff|={exp['selfcheck_maxdiff']:.3g}）")
            print(f"           PTQC {pq['n']} 层 / {pq['bytes']} B；"
                  f"W8 层 {len(w8_names)}/{len(layer_plan)}"
                  + ("；混合精度包自检 " + ("PASS" if rec.selfcheck.get("ok")
                                            else "FAIL") if explicit else ""))
        del m, quant

    if verbose:
        print(f"全部候选导出完成：{len(results)} 个，"
              f"{time.time() - t_all:.1f} s → {out}")
    return results


@dataclass
class CandidateEval:
    """`ptq eval` 的单个候选精度结果。"""
    model_key: str
    model_name: str
    candidate: str
    display: str
    eval_n: int
    n_test: int
    fp32_acc: float
    quant_acc: float
    budget_pp: float = 3.0
    plan: Dict = field(default_factory=dict)
    layer_plan: Dict = field(default_factory=dict)
    elapsed_s: float = 0.0

    @property
    def acc_drop(self) -> float:
        return self.fp32_acc - self.quant_acc

    @property
    def passed(self) -> bool:
        return self.acc_drop <= self.budget_pp

    def to_dict(self) -> Dict:
        return dict(model=self.model_key, model_name=self.model_name,
                    candidate=self.candidate, display=self.display,
                    eval_n=self.eval_n, n_test=self.n_test,
                    fp32_acc=round(self.fp32_acc, 4),
                    quant_acc=round(self.quant_acc, 4),
                    acc_drop=round(self.acc_drop, 4),
                    budget_pp=self.budget_pp,
                    passed=bool(self.passed),
                    plan=self.plan,
                    layer_plan=self.layer_plan,
                    elapsed_s=round(self.elapsed_s, 2))


def eval_candidates(model_key: str,
                    weights,
                    candidate_keys=None,
                    eval_n: int = 2000,
                    budget_pp: float = 3.0,
                    out_dir="results/w4_candidates",
                    device: str = "cpu",
                    verbose: bool = True,
                    w8_layers=None,
                    w4_group_size: int = 32,
                    w4_scheme: str = "per_group",
                    plan_name=None,
                    plan_meta: Dict | None = None) -> Dict:
    """按 M2 协议做**权重-only** CPU 精度评测，返回含逐候选结果的 dict。

    协议（与 M2 对齐）：
      加载训练权重 → BN 折叠 → 同一个 test loader → fp32 基线
      → 每个候选只对 Conv2d 做 QDQ（fc 保持 fp32）→ 评测。

    ⛔ 不要再按 `--w-bits 4` 二次量化：W4 候选的权重已经是 QDQ 到
       W4/W8 网格的 fp32，这里只复现导出时的同一份数值。
    ⛔ 本函数只量化权重，**不含激活量化**；引擎侧真实 W8A8 仍需 GPU。
    """
    import copy
    import time

    from .models import get_model, list_models, topo_kind
    from .quant.mixed import (build_layer_plan, candidate_keys as all_keys,
                              get_candidate, make_explicit_candidate,
                              plan_layer_list, plan_summary)
    from .quant.ptq import quantize_weight_sym

    if topo_kind(model_key) == "blocks":
        raise ValueError(
            f"{model_key} 走 PTQW/blocks 格式，不属于 W4 候选评测。"
            f"可选：resnet50 / mobilenet_v2")
    if not weights:
        raise ValueError("eval_candidates 必须提供训练权重，否则精度无意义")

    spec = next(s for s in list_models() if s.key == model_key)
    if w8_layers is not None:
        if candidate_keys is not None:
            raise ValueError("w8_layers 与 candidate_keys 只能二选一")
        key = (plan_meta or {}).get("plan") or plan_name or "explicit"
        cands = [make_explicit_candidate(key, w8_layers, w4_group_size,
                                         w4_scheme=w4_scheme)]
    else:
        if candidate_keys is None:
            candidate_keys = all_keys()
        if isinstance(candidate_keys, str):
            candidate_keys = [k.strip() for k in candidate_keys.split(",")
                              if k.strip()]
        cands = [get_candidate(k) for k in candidate_keys]

    t_all = time.time()
    lab = load_ptq_lab()
    model = get_model(model_key)
    n_sd, miss, unexp = load_trained_weights(model, weights)
    if miss or unexp:
        raise RuntimeError(
            f"加载 {weights} 后键集合不干净：missing={miss} "
            f"unexpected={unexp}")

    # calib_n=1：评测不需要校准集，但复用同一 loader 构造口径。
    _calib, test, _n_train, n_test = get_loaders(calib_n=1, eval_n=eval_n)
    model = lab["fold_all_bn"](model.to(device).eval()).to(device).eval()
    if verbose:
        print(f"模型 {spec.display}（{model_key}）· 训练权重 {n_sd} 个张量 · "
              f"测试集 {n_test} 张 · device={device}")
        print("协议：训练权重 → BN 折叠 → 同一 test loader → fp32 基线 → "
              "逐候选 Conv2d QDQ（fc fp32）")

    t0 = time.time()
    fp32_acc = 100.0 * float(lab["evaluate"](model, test, device=device))
    if verbose:
        print(f"fp32 基线 {fp32_acc:.2f}%（{time.time() - t0:.1f} s）")

    rows = []
    for cand in cands:
        t0 = time.time()
        m = copy.deepcopy(model)
        plan = build_layer_plan(m, cand)
        for name, mod in m.named_modules():
            if not isinstance(mod, nn.Conv2d):
                continue
            cfg = plan[name]
            w = mod.weight.detach().cpu().numpy()
            q = quantize_weight_sym(w, cfg.bits, cfg.scheme, cfg.group_size)
            mod.weight.data.copy_(
                torch.from_numpy(q.dequant()).to(
                    mod.weight.device, mod.weight.dtype).reshape(mod.weight.shape))
        acc = 100.0 * float(lab["evaluate"](m, test, device=device))
        layer_plan = plan_layer_list(plan)
        rec = CandidateEval(
            model_key=model_key, model_name=spec.display,
            candidate=cand.key, display=cand.display,
            eval_n=eval_n, n_test=n_test,
            fp32_acc=fp32_acc, quant_acc=acc, budget_pp=budget_pp,
            plan=plan_summary(plan),
            layer_plan={"n": len(layer_plan),
                        "w8_layers": [e["name"] for e in layer_plan
                                      if int(e["bits"]) == 8],
                        "counts": plan_summary(plan)["counts"]},
            elapsed_s=time.time() - t0)
        rows.append(rec)
        if verbose:
            verdict = "PASS" if rec.passed else "FAIL"
            print(f"  [{cand.key:<16}] {acc:6.2f}%  掉点 {rec.acc_drop:5.2f} pp  "
                  f"{verdict}（{rec.elapsed_s:.1f} s）")
        del m

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"n{eval_n}" if eval_n else "full"
    doc = dict(
        model=model_key, model_name=spec.display,
        weights=str(weights), eval_n=eval_n, n_test=n_test,
        budget_pp=budget_pp, device=device,
        protocol="训练权重→BN 折叠→同一 test loader→fp32 基线→"
                 "逐候选只对 Conv2d 做 QDQ（fc 保持 fp32）→evaluate",
        note="权重-only，未含激活量化；引擎侧真实 W8A8 仍需 GPU 验证。"
             "本评测不再按 --w-bits 4 二次量化。",
        fp32_acc=round(fp32_acc, 4),
        candidates=[r.to_dict() for r in rows],
        elapsed_s=round(time.time() - t_all, 2),
    )
    path = out / f"eval_{model_key}_{tag}.json"
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    doc["path"] = str(path)
    if verbose:
        print(f"评测结果 → {path}（{time.time() - t_all:.1f} s）")
    return doc


def load_trained_weights(model, ckpt, device="cpu"):
    """把 tools/finetune.py 产出的 checkpoint 加载进模型。

    支持两种封装：
      · {"model": state_dict, ...}   —— finetune.py 的格式
      · 裸 state_dict               —— torch.save(model.state_dict())
    返回 (n_loaded, missing, unexpected)。

    ⚠️ 必须**折叠 BN 之前**加载：checkpoint 是带 BN 的模型存的，
       折叠后键集合会变（bn.* 消失、conv.weight/bias 被改写）。
    """
    import torch
    sd = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    if isinstance(sd, dict):
        for k in ("model", "state_dict", "net"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return len(sd), list(missing), list(unexpected)


def build(model_key: str = "resnet18",
          w_bits: int = 8, a_bits: int = 8,
          calib_n: int = 512,
          eval_n: int = 0,
          weights=None,
          out_dir="results",
          device: str = "cpu",
          verbose: bool = True) -> BuildResult:
    """build：折叠 → 校准 → 量化 → 导出 → 测权重-only 精度。

    ⭐ 模型由 registry 决定（`--model`），pipeline 里**没有 if/else** ——
       加模型 = 在 registry 加一条。拓扑格式（PTQW / PTQ2）也由 registry
       的 `topo_kind` 决定。
    """
    import time
    t0 = time.time()
    lab = load_ptq_lab()
    from .quant.ptq import quantize_per_channel_sym, act_spec_from_range
    from .export import exporter, ptqc, topoblocks
    from .models import get_model, list_models, topo_kind, topo_plan

    spec = next(s for s in list_models() if s.key == model_key)

    # ---- 1) 取模型（fp32，未折叠）----
    model = get_model(model_key)

    # ---- 1b) 可选：加载训练好的权重（在折叠 BN 之前）----
    if weights:
        n_sd, miss, unexp = load_trained_weights(model, weights)
        if miss or unexp:
            raise RuntimeError(
                f"加载 {weights} 后键集合不干净：missing={miss} "
                f"unexpected={unexp}。\n"
                f"权重与模型定义不配套 —— 检查是不是拿错了 ckpt"
                f"（尤其注意不同仓库的同名实现，键名可能一样但语义不同）。")
        if verbose:
            print(f"加载训练权重 {weights}（{n_sd} 个张量，键集合干净）")
    # ResNet-18 是训练过的权重 → 全量评测（与 03/06/07 可比）；
    # 其余是随机初始化 → 子集足够（精度恒在10% 附近）。
    if not eval_n and model_key != "resnet18":
        eval_n = 2000
    calib, test, n_train, n_test = get_loaders(calib_n=calib_n, eval_n=eval_n)
    if verbose:
        print(f"模型 {spec.display}（{model_key}）· 拓扑 {spec.topo_kind}")
        print(f"  定义：{spec.source}")
        print(f"  权重：{spec.weights_note}")
        note = "" if n_test >= 10000 else (
            f"，**只取前 {n_test} 张**（随机初始化模型，全量无意义且耗时 20 小时）")
        print(f"校准集 {calib_n} 张（训练集共 {n_train}）"
              f"，测试集 {n_test} 张{note}")

    # ---- 2) fp32 基线精度（折叠前后应当一致，先量折叠前）----
    model = model.to(device).eval()
    # 03 的 evaluate 返回 0~1 的 float（不是百分数），这里统一乘 100，
    # 让本工具内部与报告里的「%」口径一致。
    fp32_acc = 100.0 * float(lab["evaluate"](model, test, device=device))
    if verbose:
        print(f"fp32 基线精度（折叠前）{fp32_acc:.2f}%")

    # ---- 3) BN 折叠 ----
    model = lab["fold_all_bn"](model).to(device).eval()
    fp32_acc_folded = 100.0 * float(lab["evaluate"](model, test, device=device))
    if verbose:
        print(f"BN 折叠后精度 {fp32_acc_folded:.2f}%（应与折叠前基本一致）")
    # 用折叠后的数做基线 —— 06/07 的所有测量都基于折叠权重
    fp32_acc = fp32_acc_folded

    # 显存保护：校准阶段会前向 batch_size 张图
    gpu_mem_guard(256 * 2**20, device)

    # ---- 4) 激活校准 ----
    #
    # ⚠️ 必须传 `symmetric=True`：
    #     03 的 calibrate 默认是**非对称**（qmax = 2^bits-1，带 zero_point），
    #     而 06/07 的 C++ 引擎与 TensorRT int8 通路只支持**对称**
    #     （ncnn 的 src/layer/quantize.cpp 把码字截在 ±127 且没有 zero_point；
    #     int8 tensor core 的 GEMM 也没有 zp 偏移）。
    #     用默认的非对称参数去喂对称引擎，码字会越界 127 溢出成负数，
    #     精度直接掉到随机猜水平（这个坑 06 踩过，见其 ptq_quant.h 注释）。
    a_scales, a_zeros, act_store = lab["calibrate"](
        model, calib, a_bits, strategy="minmax", symmetric=True, device=device)
    if verbose:
        print(f"激活校准完成：{len(a_scales)} 层（对称码空间，zp 恒为 0）")
    act_ranges = {k: {"scale": float(a_scales[k]),
                      "zero": int(a_zeros[k]),
                      "qmax": (1 << (a_bits - 1)) - 1}
                  for k in a_scales}

    # ---- 5) 权重 per-channel 量化（就地 fake-quant，便于量精度）----
    quant = {}
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            w = m.weight.detach().cpu().numpy()
            q = quantize_per_channel_sym(w, w_bits)
            quant[name] = q
            new_w = torch.from_numpy(q.dequant()).to(
                m.weight.device, m.weight.dtype)
            m.weight.data.copy_(new_w.reshape(m.weight.shape))

    quant_acc = 100.0 * float(lab["evaluate"](model, test, device=device))
    if verbose:
        print(f"W{w_bits} 权重-only 量化后精度 {quant_acc:.2f}%"
              f"（掉 {fp32_acc - quant_acc:.2f} pp；注意这是权重-only，"
              f"真实 W{w_bits}A{a_bits} 引擎精度会更低）")

    # ---- 6) 导出（按 registry 声明的拓扑格式分派）----
    out = Path(out_dir)
    x0, _ = next(iter(test))
    x0 = x0.to(device)
    # ⚠️ 文件名必须带**模型 + 位宽**：否则 build 三个模型 × 两个位宽
    #    会互相覆盖（实测踩过：build W8 再 build W4 后只剩 W4 的产物）。
    stem = f"{model_key}_folded_w{w_bits}a{a_bits}"

    if topo_kind(model_key) == "blocks":
        # ResNet-18：用 06 的 PTQW 格式，导出文件能被 06 的引擎二进制直接读
        exp = exporter.export_weights(model, out / f"{stem}.bin", x0)
    else:
        # ResNet-50 / MobileNetV2：用 PTQ2（blocks 推广版，支持 N 层 conv
        # + 可选 downsample + 可选残差 + depthwise groups）
        exp = topoblocks.export_blocks(model, out / f"{stem}.bin", x0,
                                       topo_plan(model_key))

    # ---- 6b) PTQC 校准文件（06 load_calib 直接消费）----
    # 写**原始 minmax**，不是 09 对称校准的 [-t,t]：06 引擎的
    # tensor_quant_from_range 用 (hi-lo)/127 且 zp=0，与 09 的
    # max(|lo|,|hi|)/127 在 lo 明显为负时不等价。口径差见 ptqc.py 模块头。
    pq = ptqc.write_ptqc_for_model(
        out / f"calib_{stem}.ptqc", model, act_store,
        n_expected=exp["n_conv"])

    from .quant.ptq import PTQResult, ActQuantSpec
    qr = PTQResult(w_bits=w_bits, a_bits=a_bits, weights=quant,
                   acts={k: ActQuantSpec(scale=v["scale"], zero=v["zero"],
                                         bits=a_bits)
                         for k, v in act_ranges.items()},
                   fp32_acc=fp32_acc, quant_acc=quant_acc)
    # ⚠️ 量化参数文件名也要带模型名 —— 三个模型同名会互相覆盖
    cfg = exporter.write_quant_config(
        qr, out / f"quant_config_{model_key}_w{w_bits}a{a_bits}.json",
        extra=dict(model=model_key, model_name=spec.display,
                   topo_kind=spec.topo_kind, calib_n=calib_n,
                   note="权重-only 精度；激活量化在引擎侧生效"))
    if verbose:
        print(f"导出 {exp['path']}（{exp['n_conv']} conv / {exp['n_block']} block / "
              f"{exp['bytes'] / 1e6:.1f} MB，拓扑自检 max|diff|="
              f"{exp['selfcheck_maxdiff']:.3g}）")
        print(f"导出 {cfg['path']}（{cfg['n_weights']} 权重 + "
              f"{cfg['n_acts']} 激活的量化参数）")
        print(f"导出 {pq['path']}（PTQC {pq['n_conv']} 层，"
              f"{pq['bytes']} B，层数与 {Path(exp['path']).name} 一致）")

    return BuildResult(w_bits=w_bits, a_bits=a_bits,
                       fp32_acc=fp32_acc, quant_acc=quant_acc,
                       exported=exp, quant_cfg=cfg, ptqc=pq,
                       act_ranges=act_ranges,
                       model_key=model_key, model_name=spec.display,
                       topo_kind=spec.topo_kind,
                       n_calib_batches=len(calib),
                       elapsed_s=time.time() - t0)
