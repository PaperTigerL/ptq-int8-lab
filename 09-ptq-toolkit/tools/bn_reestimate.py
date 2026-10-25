# -*- coding: utf-8 -*-
"""BN running stats 与 conv1 移植方式的一次性对照实验（S1 第 3 件）。

回答两个问题
-----------
Q1  BN running stats 失配该怎么处置？（保留 / 用 CIFAR 重估 / 重置）
    —— 失配是**全网性**的：stem 从 7×7s2 改成 3×3s1 且去掉 maxpool，
    每个 stage 的特征图几何都变了，layer1..4 的 BN stats 全部失配。
    所以本脚本默认重估**所有 BatchNorm2d**，不是只 bn1。

Q2  conv1 的 7×7 → 3×3 该用哪种移植方式？
    A  中心裁剪不缩放（现状）
    B  中心裁剪后按面积补偿 ×(7×7)/(3×3) = 5.44
    C  7×7 核**空间求和池化**成 3×3（保留全部 L1 质量）

输出
----
  results/bn_conv1_compare.md    人读表：conv1 三组 × BN 三处置 → fp32 top-1
  results/bn_conv1_compare.json  机读版

资源纪律（负责人明确要求）
------------------------
  · 设备默认 cuda（GPU 空闲时）；CPU 上跑必须 nice -n 19
  · torch.set_num_threads(2)   —— 不占满整机
  · batches 默认 8             —— 重估是统计量，8 批足够看趋势
  · 最后一次用**全量 10000 张**评测

用法
----
    python tools/bn_reestimate.py --full            # 3 × 3 = 9 个组合
    python tools/bn_reestimate.py --conv1 A --bn keep,reset
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
TOOLKIT = HERE.parent
sys.path.insert(0, str(TOOLKIT))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

# ------------------------------------------------------------------ conv1 移植 / BN 处置
# 实现统一放在 load_transfer.py（conv1_variants / apply_conv1_mode / set_bn_mode），
# 这里只做实验编排。⛔ 不要在两份文件里各写一遍 conv1 逻辑 ——
# 曾经就是这么分叉的：bn_reestimate 与 transplant 各有一份 center_crop，
# 改一处另一处不跟着变。

from load_transfer import apply_conv1_mode, set_bn_mode   # noqa: E402

def _sha(t) -> str:
    """张量的 sha256（前 16 位）—— 每组配置都有，便于复核「这行确实是这个权重」。"""
    import hashlib
    return hashlib.sha256(
        t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


CONV1_MODES = ("A", "B", "C")
BN_MODES = ("keep", "reestimate", "reset")
RESULT_MD = TOOLKIT / "results" / "bn_conv1_compare.md"
RESULT_JSON = TOOLKIT / "results" / "bn_conv1_compare.json"

# 每种 conv1 模式**独有**的那条数值自检项（公共项另有 L1 比）。
# 落表时必须带上 —— 否则三种模式在表上看起来只是个字母区别，
# 无从判断「这一行真的是按该模式算出来的权重」。
INVARIANT_KEY = {
    "A": "l1_less_than_src",
    "B": "equals_center_times_ratio",
    "C": "l1_preserved",
}


def apply_conv1(model, src_conv1_w, mode: str):
    """把 conv1 按 A/B/C 移植进模型，返回数值自检 dict。"""
    return apply_conv1_mode(model, src_conv1_w, mode)


# ------------------------------------------------------------------ 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet50")
    ap.add_argument("--conv1", default="ABC", choices=["A", "B", "C", "ABC"],
                    help="conv1 移植方式；ABC = 三组都跑")
    ap.add_argument("--bn", default="keep,reestimate,reset",
                    help="BN 处置，逗号分隔")
    ap.add_argument("--bn-scope", default="all", choices=["all", "bn1"],
                    help="重估/重置的范围（all = 全部 BatchNorm2d）")
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--eval-n", type=int, default=0,
                    help="评测样本上限（0 = 全量 10000）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    if args.device == "cpu":
        print("⚠️ 在 CPU 上跑 —— 请用 nice -n 19 启动，"
              "并考虑减小 --batches / --eval-n")

    from load_transfer import fetch_weights
    from ptq_toolkit.models import get_model
    from ptq_toolkit.pipeline import get_loaders, load_ptq_lab

    lab = load_ptq_lab()
    got = fetch_weights(args.arch, verbose=False)
    src = got["state"]
    src_conv1 = src["conv1.weight"]
    cache_path = got["info"]["cache_path"]

    # 磁盘 .pth 独立回读（不与 src 共用对象），顺带验一道
    disk = torch.load(cache_path, map_location="cpu", weights_only=True)
    disk_bit_same = bool(torch.equal(src_conv1, disk["conv1.weight"]))

    conv1_modes = list(CONV1_MODES) if args.conv1 == "ABC" else [args.conv1]
    bn_modes = [x.strip() for x in args.bn.split(",") if x.strip()]
    for b in bn_modes:
        if b not in BN_MODES:
            raise SystemExit(f"未知 BN 处置 {b!r}；可选 {BN_MODES}")

    n_reest = args.batches * 64
    _, test, _, n_test = get_loaders(calib_n=n_reest, eval_n=args.eval_n)
    calib, _, _, _ = get_loaders(calib_n=n_reest, eval_n=0)

    print("=" * 78)
    print(f"conv1 × BN 对照（arch={args.arch}, device={args.device}, "
          f"threads={args.threads}, batches={args.batches}）")
    print(f"  conv1: {conv1_modes}   BN: {bn_modes} (scope={args.bn_scope})")
    print(f"  评测 {n_test} 张；磁盘/内存 conv1 逐元素相同={disk_bit_same}")
    print("=" * 78)

    rows: List[Dict] = []
    for cm in conv1_modes:
        for bm in bn_modes:
            model = get_model(args.arch)
            sc = apply_conv1(model, src_conv1, cm)
            conv1_sha = _sha(dict(model.named_modules())["conv1"].weight)
            model = model.to(args.device).eval()

            if bm == "keep":
                bn_info = set_bn_mode(model, "keep", scope=args.bn_scope)
            elif bm == "reestimate":
                bn_info = set_bn_mode(model, "reestimate", loader=calib,
                                      device=args.device,
                                      batches=args.batches,
                                      scope=args.bn_scope)
            else:
                bn_info = set_bn_mode(model, "reset", scope=args.bn_scope)

            acc = 100.0 * float(lab["evaluate"](model, test,
                                                device=args.device))
            peak = (torch.cuda.max_memory_allocated() / 2**20
                    if args.device == "cuda" else None)

            rows.append(dict(
                conv1=cm, conv1_impl=sc["impl"], bn=bm,
                bn_scope=args.bn_scope, fp32_acc=round(acc, 4),
                n_eval=n_test, batches=args.batches,
                conv1_sha=conv1_sha,
                conv1_selfcheck=sc, bn_info=bn_info,
                peak_mem_mib=round(peak, 1) if peak else None))
            print(f"  conv1={cm} ({sc.get('impl', sc['mode']):<24}) bn={bm:<11} "
                  f"→ fp32 top-1 = {acc:6.2f}%  [l1_ratio={sc.get('l1_ratio', 0):.3f}, "
                  f"conv1 sha={conv1_sha}]")

    # ---- 落盘 ----
    # 自检：同一 conv1 模式在各 BN 处置下的权重必须逐位相同。
    # 若不等，说明后续 BN 处置动到了 conv1 —— 那是 bug，不是实验结论。
    # 这条必须落进结果文件，不能只在心里过一遍。
    by_mode: Dict[str, set] = {}
    for r in rows:
        by_mode.setdefault(r["conv1"], set()).add(r["conv1_sha"])
    inconsistent = sorted(m for m, v in by_mode.items() if len(v) > 1)
    sha_consistent = not inconsistent
    if not sha_consistent:
        print(f"⚠️ conv1 权重在以下模式下被后续处置改动了：{inconsistent}"
              f" —— 这些行的精度不可横向比较")

    RESULT_MD.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(arch=args.arch, device=args.device, threads=args.threads,
                   batches=args.batches, bn_scope=args.bn_scope,
                   n_eval=n_test, conv1_modes=conv1_modes, bn_modes=bn_modes,
                   src_conv1_shape=tuple(src_conv1.shape),
                   disk_conv1_bit_identical=disk_bit_same,
                   conv1_sha_consistent=sha_consistent,
                   conv1_sha_inconsistent_modes=inconsistent,
                   rows=rows)
    RESULT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                           encoding="utf-8")

    lines = [
        "# conv1 移植方式 × BN 处置 对照实验", "",
        f"arch=`{args.arch}` · device=`{args.device}` · "
        f"threads=`{args.threads}` · batches=`{args.batches}` · "
        f"BN scope=`{args.bn_scope}` · 评测 `{n_test}` 张", "",
        "> BN running stats 失配是**全网性**的（stem 7×7s2→3×3s1 且去掉 maxpool，",
        "> 每个 stage 的特征图几何都变），所以 `bn_scope=all` 重估的是**全部**",
        "> BatchNorm2d，不是只 bn1。", "",
        "## fp32 top-1", "",
        "| conv1 | 实现 | L1 保留 | BN 处置 | fp32 top-1 | conv1 sha256 | 特有自检 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        inv_key = INVARIANT_KEY[r["conv1"]]
        lines.append(f"| {r['conv1']} | `{r['conv1_impl']}` | "
                     f"{r['conv1_selfcheck']['l1_ratio']:.3f} | {r['bn']} | "
                     f"**{r['fp32_acc']:.2f}%** | `{r['conv1_sha']}` | "
                     f"{inv_key}=`{r['conv1_selfcheck'][inv_key]}` |")
    lines += ["", "## conv1 数值自检", "",
              "| 方式 | src shape | dst shape | src mean\\|w\\| | dst mean\\|w\\| | "
              "src L1 | dst L1 | L1 比 | 等于中心块 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        sc = r["conv1_selfcheck"]
        lines.append(
            f"| {r['conv1']} | {sc.get('src_shape', ())} | {sc.get('dst_shape', ())} | "
            f"{sc.get('src_mean_abs', 0):.4f} | {sc.get('dst_mean_abs', 0):.4f} | "
            f"{sc.get('src_l1', 0):.1f} | {sc.get('dst_l1', 0):.1f} | {sc.get('l1_ratio', 0):.3f} | "
            f"{sc.get('equals_center_crop', None)} |")
    lines += ["", f"磁盘 .pth 与内存 src 的 conv1 逐元素相同："
                  f"`{disk_bit_same}`", "",
              f"同一 conv1 模式在各 BN 处置下权重逐位相同：`{sha_consistent}`"
              + (f"（⚠️ 异常模式 {inconsistent}）" if inconsistent else ""),
              "—— 若此项为 False，说明 BN 处置过程动到了 conv1，"
              "这些行的精度不可横向比较。", ""]
    RESULT_MD.write_text("\n".join(lines), encoding="utf-8")

    print()
    print(f"已写 {RESULT_MD}")
    print(f"已写 {RESULT_JSON}")
    best = max(rows, key=lambda r: r["fp32_acc"])
    print()
    print("=" * 78)
    print(f"最高：conv1={best['conv1']} + bn={best['bn']} → "
          f"{best['fp32_acc']:.2f}%")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
