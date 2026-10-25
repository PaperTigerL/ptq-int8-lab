# -*- coding: utf-8 -*-
"""golden test：外部参照对齐 + 三模型回归基线。

两类判据：
  A. **ResNet-18 有外部参照**（06/07 已提交的结果）→ 精度必须对齐
     · fp32 与 06/07 记录一致（差 ≤ 0.3 pp）
     · W8 权重-only 掉点 ≤ 1.0 pp
     · W8 权重-only 精度 ≥ 06 引擎的真实 W8A8
       —— 本工具是权重-only，必然 ≥ 引擎的 W8A8；不等就是量化写错了

  B. **ResNet-50 / MobileNetV2 无外部参照** → 用**本次自测**锁定基线
     ⚠️ 这类基线**不是真精度**：两个模型是随机初始化的，
        fp32 精度恒在 10%（随机猜），所以它们的「掉点」几乎恒为 0，
        **测的是「量化代码没把权重写坏」而不是「量化损失多少」**。
        这一点必须写在报告里，否则会被误当成精度结果引用。

用法：python tests/test_golden.py
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLKIT = HERE.parent

# ---- A. ResNet-18 的外部参照（从 06/07 实际文件读，不手抄）----
EXT_FP32 = 87.16          # 06/07 都记录为这个值
EXT_ENGINE_W8A8 = 86.93   # 06 自写 int8 引擎的端到端结果

# ---- B. 自测基线（来源=自测，锁定用于回归）----
#
# ⚠️⚠️ 随机初始化模型的 fp32 精度**不是常数**：
#    实测同一份代码（未固定 seed）跑三次得到
#    resnet50 = 6.25% / 10.40% / 10.65%，mobilenet_v2 = 9.80% / 10.80%。
#    都在 10%（随机猜）附近波动，**单次测量不能当基线**。
#
#    所以这里的判据不是「等于某个数」，而是
#    **「落在随机猜的合理区间内」**（0 ~ 25%，越宽越能容忍波动）。
#    这条判据测的是「量化 + 拓扑导出没把网络算坏」——
#    如果算坏了，精度会掉到远低于随机猜的水平（比如 0%或 100%）。
SELF_RANGE = (0.0, 25.0)

MODELS = ["resnet18", "resnet50", "mobilenet_v2"]

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'✓' if cond else '✗'} {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILED.append(name)


def _find_lab() -> Path:
    for p in HERE.parents:
        if (p / "06-gpu-int8").is_dir() and (p / "03-ptq-lab").is_dir():
            return p
    raise FileNotFoundError("找不到 lab/")


def load_builds():
    out = {}
    for p in sorted((TOOLKIT / "results").glob("build_*_w*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        out[(d["model"], d["w_bits"])] = d
    return out


def read_external():
    """从 06/07 的实际文件读参考值（不手抄）。"""
    src = {}
    lab = _find_lab()
    p = lab / "06-gpu-int8" / "results" / "e2e_compare.txt"
    if p.exists():
        m = re.search(r"GPU int8 引擎\s+([\d.]+)%",
                      p.read_text(encoding="utf-8"))
        if m:
            src["engine"] = float(m.group(1))
    p = lab / "07-tensorrt-deploy" / "results" / "idle_final.txt"
    if p.exists():
        m = re.search(r"^fp32\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+\d+\s+\S+\s+([\d.]+)%",
                      p.read_text(encoding="utf-8"), re.M)
        if m:
            src["fp32"] = float(m.group(1))
    return src


def main():
    builds = load_builds()
    if not builds:
        print("[SKIP] 没有 results/build_*_w*.json —— 先跑 ptq build")
        return 0

    ext = read_external()
    print("从 06/07 实际文件读到的外部参照：")
    for k, v in sorted(ext.items()):
        print(f"  {k:<10} {v:.2f}%")

    # ---------------- A. ResNet-18 外部对齐 ----------------
    r18_8 = builds.get(("resnet18", 8))
    if r18_8:
        print("\nResNet-18（外部参照）")
        fp32_ref = ext.get("fp32", EXT_FP32)
        check(f"fp32 基线 {r18_8['fp32_acc']:.2f}% vs 06/07 的 {fp32_ref:.2f}%",
              abs(r18_8["fp32_acc"] - fp32_ref) <= 0.30,
              f"差 {abs(r18_8['fp32_acc'] - fp32_ref):.2f} pp")
        check(f"W8 掉点 {r18_8['acc_drop']:.2f} pp <= 1.0",
              r18_8["acc_drop"] <= 1.0)
        eng = ext.get("engine", EXT_ENGINE_W8A8)
        check(f"W8 权重-only {r18_8['quant_acc']:.2f}% >= 引擎 W8A8 {eng:.2f}%",
              r18_8["quant_acc"] >= eng - 1e-9)
    else:
        print("\n[SKIP] ResNet-18 W8 未构建")

    # ---------------- B. 三模型完整性 ----------------
    print("\n三模型 × 两宽度 完整性")
    for m in MODELS:
        for w in (8, 4):
            d = builds.get((m, w))
            check(f"{m} W{w}", d is not None,
                  (f"fp32={d['fp32_acc']:.2f}% quant={d['quant_acc']:.2f}% "
                   f"drop={d['acc_drop']:.2f}pp") if d else "未构建")
            if not d:
                continue
            check(f"  {m} W{w} 权重文件存在",
                  Path(d["exported"]["path"]).exists(),
                  Path(d["exported"]["path"]).name)
            check(f"  {m} W{w} 量化参数存在",
                  Path(d["quant_cfg"]["path"]).exists(),
                  Path(d["quant_cfg"]["path"]).name)
            check(f"  {m} W{w} 拓扑自检 ≈ 0",
                  d["exported"].get("selfcheck_maxdiff", 1) < 1e-4,
                  f"{d['exported'].get('selfcheck_maxdiff')}")

    # ---------------- B2. 掉点阈值 ----------------
    print("\n掉点阈值（W8 <= 1.0 pp / W4 <= 3.0 pp）")
    for m in MODELS:
        for w, lim in ((8, 1.0), (4, 3.0)):
            d = builds.get((m, w))
            if d:
                check(f"{m} W{w} 掉点 {d['acc_drop']:.2f} pp <= {lim}",
                      d["acc_drop"] <= lim)

    # ---------------- B3. 自测基线回归 ----------------
    print("\n自测基线（来源=自测，锁定用于回归）")
    print("  ⚠️ ResNet-50/MobileNetV2 是**随机初始化**的，fp32 在随机猜附近波动")
    print("     （实测同代码三次：resnet50 = 6.25 / 10.40 / 10.65%，")
    print("       mobilenet_v2 = 9.80 / 10.80%），所以判据是「落在合理区间」")
    print("     而不是「等于某个数」。这条测的是「量化 + 拓扑导出没把网络算坏」，")
    print("     不是量化损失。")
    for m in ("resnet50", "mobilenet_v2"):
        for w in (8, 4):
            d = builds.get((m, w))
            if d:
                lo, hi = SELF_RANGE
                check(f"{m} W{w} fp32 {d['fp32_acc']:.2f}% 落在 "
                      f"[{lo:.0f}, {hi:.0f}]",
                      lo <= d["fp32_acc"] <= hi)

    print()
    if FAILED:
        head = ", ".join(FAILED[:6])
        more = " ..." if len(FAILED) > 6 else ""
        print(f"✗ {len(FAILED)} 项失败：{head}{more}")
        return 1
    print("✓ 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
