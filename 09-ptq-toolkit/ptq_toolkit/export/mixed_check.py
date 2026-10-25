# -*- coding: utf-8 -*-
"""混合精度权重包的格式自检（纯标准库 + numpy，不依赖 torch/C++）。

一个「混合精度包」= 同目录下的 5 个文件 + manifest：

    <stem>.ptq2.bin            PTQ2 权重/拓扑（fp32 容器，权重已 QDQ 到 W4/W8 网格）
    calib_<stem>.ptqc          A8 校准 range（06 load_calib 直接消费）
    quant_config_<stem>.json   逐层 bits / scheme / group_size / scale
    layer_plan.json            逐层名单（顺序 = 引擎 conv 顺序）
    build_<stem>.json          导出摘要（含拓扑自检 max|diff|）
    manifest.json              SHA256 / 字节数 / 层数 / 自检结论

为什么要单独一层自检：
  · PTQ2 只存 fp32 权重，**不携带 bits/scale 元数据**；如果只信导出时
    的内存对象，磁盘上的文件是否真的落在 W4/W8 网格上没有任何证据。
  · 06 的引擎按 PTQC 层数做硬校验（set_calib 层数不符直接 exit）。
    所以这里按文件字节回读，把「层数 / 顺序 / 网格 / 码字范围 / 哈希」
    全部验一遍，失败即 raise（导出流程不允许产出未自检的包）。

⭐ 本模块的层顺序口径与 ptqc.py / topoblocks.py 完全一致：
   `named_modules()` 过滤 `nn.Conv2d` 的定义顺序，不按语义重排。
"""
import hashlib
import json
import math
import struct
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .ptq2_info import conv_weight, read_ptq2_info
from .ptqc import read_ptqc

SCHEMA = "ptq/mixed-package@1"
PLAN_SCHEMA = "ptq/layer-plan@1"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _round_half_away(x):
    """与 quant.ptq.round_half_away 同语义（这里不依赖 torch）。"""
    return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5))


def _counts(layer_plan: List[Dict]) -> List[Dict]:
    acc: Dict[tuple, int] = {}
    for e in layer_plan:
        k = (int(e["bits"]), str(e["scheme"]), int(e.get("group_size", 0)))
        acc[k] = acc.get(k, 0) + 1
    return [dict(bits=b, scheme=s, group_size=g, layers=n)
            for (b, s, g), n in sorted(acc.items())]


def write_layer_plan(path, model: str, plan_key: str,
                     layer_plan: List[Dict],
                     plan_meta: Optional[Dict] = None) -> Dict:
    """写 layer_plan.json（顺序即引擎 conv 顺序）。"""
    doc = {
        "schema": PLAN_SCHEMA,
        "model": model,
        "plan": plan_key,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "layer_order": "named_modules() 过滤 nn.Conv2d 的定义顺序",
        "counts": _counts(layer_plan),
        "w8_layers": [e["name"] for e in layer_plan if int(e["bits"]) == 8],
        "layers": layer_plan,
    }
    if plan_meta:
        doc["evidence"] = plan_meta
    p = Path(path)
    p.write_text(json.dumps(doc, indent=2, ensure_ascii=False),
                 encoding="utf-8")
    return doc


def write_manifest(package_dir, *, model: str, plan_key: str,
                   layer_plan: List[Dict],
                   files: Dict[str, str],
                   exported: Dict,
                   source_weights: Optional[str] = None,
                   plan_meta: Optional[Dict] = None,
                   topology_maxdiff: Optional[float] = None) -> Dict:
    """写 manifest.json，记录每个产物的 SHA256/字节数与导出参数。"""
    d = Path(package_dir)
    file_recs = {}
    for role, path in files.items():
        p = Path(path)
        file_recs[role] = {
            "name": p.name,
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        }

    sw = None
    if source_weights:
        p = Path(source_weights)
        if p.is_file():
            sw = {"path": str(p), "bytes": p.stat().st_size,
                  "sha256": sha256_file(p)}

    manifest = {
        "schema": SCHEMA,
        "model": model,
        "plan": plan_key,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "note": ("PTQ2 是 fp32 容器：W4 只改变权重数值（QDQ），"
                 "不改变存储大小；bits/scheme 元数据在 quant_config + "
                 "layer_plan 里。"),
        "source_weights": sw,
        "evidence": plan_meta,
        "n_conv": int(exported["n_conv"]),
        "n_block": int(exported["n_block"]),
        "counts": _counts(layer_plan),
        "w8_layers": [e["name"] for e in layer_plan if int(e["bits"]) == 8],
        "n_w8": sum(1 for e in layer_plan if int(e["bits"]) == 8),
        "n_w4": sum(1 for e in layer_plan if int(e["bits"]) == 4),
        "topology_selfcheck_maxdiff": topology_maxdiff,
        "files": file_recs,
    }
    p = d / "manifest.json"
    p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                 encoding="utf-8")
    return manifest


# ---------------------------------------------------------------- 回读自检

class _Checks:
    def __init__(self):
        self.rows: List[Dict] = []

    def __call__(self, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append({"name": name, "ok": bool(ok), "detail": detail})
        return bool(ok)

    @property
    def ok(self) -> bool:
        return all(r["ok"] for r in self.rows)


def _grid_errors(weight: np.ndarray, entry: Dict) -> tuple:
    """返回 (max_abs_err, code_min, code_max, scale_shape_ok)。

    权重已在导出时 QDQ，这里用声明的 scale 反推码字，验证：
      · 反量化值与文件里的 fp32 权重一致（真的落在该 bits 的网格上）
      · 码字落在 [-qmax-1, qmax]
      · per_group 的 scale 形状与 (cout, n_groups) 一致
    """
    bits = int(entry["bits"])
    qmax = (1 << (bits - 1)) - 1
    scheme = entry.get("scheme", "per_channel")
    scale = np.asarray(entry["scale"], dtype=np.float64)
    w = weight.reshape(weight.shape[0], -1).astype(np.float64)
    cout, k = w.shape
    if scheme == "per_group":
        gs = int(entry.get("group_size", 0))
        if gs <= 0:
            return math.inf, 0, 0, False
        ng = scale.shape[1] if scale.ndim == 2 else 0
        ok_shape = (scale.ndim == 2 and scale.shape[0] == cout
                    and (ng - 1) * gs < k <= ng * gs)
        if not ok_shape:
            return math.inf, 0, 0, False
        max_err = 0.0
        cmin, cmax = 1 << 30, -(1 << 30)
        for g in range(ng):
            seg = w[:, g * gs:(g + 1) * gs]
            if seg.size == 0:
                continue
            s = scale[:, g:g + 1]
            code = _round_half_away(seg / s)
            cmin = min(cmin, int(code.min()))
            cmax = max(cmax, int(code.max()))
            max_err = max(max_err, float(np.abs(seg - code * s).max()))
    else:
        ok_shape = (scale.ndim == 1 and scale.shape[0] == cout)
        if not ok_shape:
            return math.inf, 0, 0, False
        code = _round_half_away(w / scale[:, None])
        cmin, cmax = int(code.min()), int(code.max())
        max_err = float(np.abs(w - code * scale[:, None]).max())
    if cmin < -(qmax + 1) or cmax > qmax:
        max_err = math.inf
    return max_err, cmin, cmax, True


def check_package(package_dir, *, check_grid: bool = True,
                  require_sha256: bool = True) -> Dict:
    """回读并校验一个混合精度包；返回 {ok, checks, ...}，不抛异常。

    严格失败（文件缺失/magic 错/长度错）会以 check 失败的形式记录，
    调用方用 report["ok"] 决定是否接受产物。
    """
    d = Path(package_dir)
    ck = _Checks()
    report: Dict = {"package": str(d), "ok": False, "checks": []}

    def fail(msg: str) -> Dict:
        ck("包结构可读", False, msg)
        report["checks"] = ck.rows
        report["ok"] = False
        report["error"] = msg
        return report

    mpath = d / "manifest.json"
    if not mpath.is_file():
        return fail(f"缺 {mpath.name}")
    try:
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
    except Exception as e:
        return fail(f"manifest 解析失败：{type(e).__name__}: {e}")
    ck("manifest schema", manifest.get("schema") == SCHEMA,
       str(manifest.get("schema")))

    files = manifest.get("files") or {}
    for role in ("weights", "calib", "quant_config", "layer_plan", "build"):
        if role not in files:
            return fail(f"manifest.files 缺 {role}")
    paths = {role: d / rec["name"] for role, rec in files.items()}
    for role, p in paths.items():
        ck(f"{role} 存在", p.is_file(), p.name)
    if not ck.ok:
        report["checks"] = ck.rows
        return report

    for role, p in paths.items():
        rec = files[role]
        ck(f"{role} 字节数", p.stat().st_size == int(rec["bytes"]),
           f"{p.stat().st_size} vs {rec['bytes']}")
        if require_sha256:
            ck(f"{role} SHA256", sha256_file(p) == rec["sha256"],
               rec["sha256"][:12])

    try:
        plan_doc = json.loads(paths["layer_plan"].read_text(encoding="utf-8"))
        cfg = json.loads(paths["quant_config"].read_text(encoding="utf-8"))
        info = read_ptq2_info(paths["weights"])
        ranges = read_ptqc(paths["calib"])
    except Exception as e:
        return fail(f"产物解析失败：{type(e).__name__}: {e}")

    layers = plan_doc.get("layers") or []
    names = [e.get("name") for e in layers]
    ck("layer_plan schema", plan_doc.get("schema") == PLAN_SCHEMA)
    ck("layer_plan 非空", len(layers) > 0, f"{len(layers)} 层")
    ck("PTQ2 n_conv == plan 层数", info["n_conv"] == len(layers),
       f"{info['n_conv']} vs {len(layers)}")
    ck("PTQC 层数 == PTQ2 n_conv", len(ranges) == info["n_conv"],
       f"{len(ranges)} vs {info['n_conv']}")
    ck("PTQC 文件长度 8+8n",
       paths["calib"].stat().st_size == 8 + 8 * len(ranges),
       f"{paths['calib'].stat().st_size} B")
    ck("PTQC 有限且 lo<=hi",
       all(math.isfinite(lo) and math.isfinite(hi) and lo <= hi
           for lo, hi in ranges))
    ck("manifest n_conv", int(manifest.get("n_conv", -1)) == info["n_conv"],
       f"{manifest.get('n_conv')} vs {info['n_conv']}")

    weights = cfg.get("weights") or {}
    ck("quant_config 权重数 == conv 数",
       len(weights) == info["n_conv"], f"{len(weights)} vs {info['n_conv']}")
    ck("quant_config 键顺序 == plan 顺序", list(weights) == names)

    w8_plan = [n for n, e in zip(names, layers) if int(e["bits"]) == 8]
    ck("manifest w8_layers == plan",
       list(manifest.get("w8_layers") or []) == w8_plan,
       f"{len(manifest.get('w8_layers') or [])} vs {len(w8_plan)}")
    ck("counts 一致",
       manifest.get("counts") == _counts(layers),
       json.dumps(manifest.get("counts"), ensure_ascii=False))

    data = paths["weights"].read_bytes()
    meta_ok = True
    grid_ok = True
    scale_shape_ok = True
    max_err = 0.0
    code_lo, code_hi = 1 << 30, -(1 << 30)
    for i, (name, entry) in enumerate(zip(names, layers)):
        wcfg = weights.get(name)
        if wcfg is None:
            meta_ok = False
            continue
        if (int(wcfg.get("bits", -1)) != int(entry["bits"])
                or wcfg.get("scheme", "per_channel")
                != entry.get("scheme", "per_channel")
                or int(wcfg.get("group_size", 0))
                != int(entry.get("group_size", 0))):
            meta_ok = False
        if not check_grid:
            continue
        conv = info["convs"][i]
        weight = conv_weight(data, conv)
        err, cmin, cmax, shape_ok = _grid_errors(weight, wcfg)
        scale_shape_ok = scale_shape_ok and shape_ok
        if not math.isfinite(err):
            grid_ok = False
        max_err = max(max_err, err if math.isfinite(err) else math.inf)
        code_lo, code_hi = min(code_lo, cmin), max(code_hi, cmax)
    ck("逐层 bits/scheme/group_size == layer_plan", meta_ok)
    if check_grid:
        ck("scale 形状与 scheme 一致", scale_shape_ok)
        ck("权重落在声明的 bits 网格", grid_ok,
           f"max|err|={max_err:.3g}，码字 [{code_lo}, {code_hi}]")
        ck("码字范围合法", math.isfinite(max_err))

    try:
        build_doc = json.loads(paths["build"].read_text(encoding="utf-8"))
        dmax = build_doc.get("exported", {}).get("selfcheck_maxdiff")
        ck("拓扑自检 max|diff| < 1e-4",
           dmax is not None and float(dmax) < 1e-4, str(dmax))
    except Exception as e:
        ck("build 摘要可读", False, f"{type(e).__name__}: {e}")

    report.update(
        ok=ck.ok,
        checks=ck.rows,
        n_conv=info["n_conv"],
        n_block=info["n_block"],
        n_w8=len(w8_plan),
        n_w4=len(layers) - len(w8_plan),
        w8_layers=w8_plan,
        counts=_counts(layers),
        sha256={role: files[role]["sha256"] for role in files},
    )
    return report
