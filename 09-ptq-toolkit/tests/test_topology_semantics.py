# -*- coding: utf-8 -*-
"""PTQ2 导出拓扑的「激活语义」自检（M2 修复的回归闸门）。

⚠️ 为什么现有 test_models.py 测不出这里的问题：
   它用随机初始化 + `torch.randn` 输入，MV2 逐层激活从 0.44 衰减到 1e-7，
   自检 max|diff| = 1.58e-08 也「通过」—— 因为激活值小到 relu(±1e-7) 的差别
   都在容差内，ReLU6 的截断（>6 才生效）从不触发。
   于是「project 该不该接激活 / 相加后接不接激活 / ReLU 还是 ReLU6」这三处
   语义错误全部被掩盖（results/_randomprobe.py 复现）。

本测试用**放大输入**（×10）逼出 ReLU6 截断，并直接断言 recs/blocks 的标志位，
再叠一层训练权重下的导出级数值自检。三种语义缺一不可：
  1. 17 个 inverted residual 块的 project（末 conv）**无激活**（relu=0）；
  2. 残差块 x + conv(x) **相加后无激活**（residual=2）；
  3. MV2 激活是 **ReLU6**（clamp 6，relu=2），不是 ReLU。

用法：python tests/test_topology_semantics.py
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from ptq_toolkit.export.topoblocks import build_blocks, export_blocks   # noqa: E402
from ptq_toolkit.models import get_model, registry                      # noqa: E402
from ptq_toolkit.models.mobilenet_v2 import MobileNetV2                 # noqa: E402
from ptq_toolkit.pipeline import load_ptq_lab                           # noqa: E402

FAILED = []
TMP = HERE.parent / "results"


def check(name, cond, detail=""):
    print(f"  {'✓' if cond else '✗'} {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILED.append(name)


def expected_use_res():
    """按 SETTINGS 重算 17 个块各自的 use_res（stride==1 且 Cin==Cout）。"""
    res, cin = [], MobileNetV2.FIRST_CHANNEL
    for _t, cout, n, stride in MobileNetV2.SETTINGS:
        for i in range(n):
            s = stride if i == 0 else 1
            res.append(s == 1 and cin == cout)
            cin = cout
    return res


def folded(key):
    return load_ptq_lab()["fold_all_bn"](get_model(key)).eval()


# ---------------------------------------------------------------- 1) 结构断言

def test_mv2_structure():
    print("\n[mv2] 结构：relu/residual 标志位")
    plan = registry.topo_plan("mobilenet_v2")
    m = folded("mobilenet_v2")
    recs, blocks, _sp, stem_idx = build_blocks(m, plan, False, act="relu6")

    check("stem conv relu==2（ReLU6）", recs[stem_idx].relu == 2,
          f"实际 {recs[stem_idx].relu}")

    use_res = expected_use_res()
    check("blocks 数 == 18（17 块 + features.18 伪块）", len(blocks) == 18,
          f"实际 {len(blocks)}")

    ok_last, ok_mid, ok_res = True, True, True
    for bi in range(17):
        blk = blocks[bi]
        # 末 conv（project）无激活
        if recs[blk.convs[-1]].relu != 0:
            ok_last = False
        # 非末 conv（expand / dw）接 ReLU6
        for ci in blk.convs[:-1]:
            if recs[ci].relu != 2:
                ok_mid = False
        # residual 编码：有 skip→2，无 skip→0
        want = 2 if use_res[bi] else 0
        if blk.residual != want:
            ok_res = False
    check("17 块 project（末 conv）relu==0", ok_last)
    check("17 块中间 conv（expand/dw）relu==2", ok_mid)
    check("17 块 residual==2（有 skip）或 0（无 skip）", ok_res)

    # features.18 伪块：非残差，末 conv 接 ReLU6
    proj = blocks[17]
    check("features.18 residual==0 且末 conv relu==2",
          proj.residual == 0 and recs[proj.convs[-1]].relu == 2,
          f"residual={proj.residual} relu={recs[proj.convs[-1]].relu}")


def test_r50_structure():
    print("\n[r50] 结构：默认 relu 语义保持不变")
    plan = registry.topo_plan("resnet50")
    m = folded("resnet50")
    recs, blocks, _sp, stem_idx = build_blocks(m, plan, False, act="relu")

    check("stem relu==1（ReLU）", recs[stem_idx].relu == 1)
    all_res1 = all(b.residual == 1 for b in blocks)
    ok_last = all(recs[b.convs[-1]].relu == 0 for b in blocks)
    ok_mid = all(recs[c].relu == 1 for b in blocks for c in b.convs[:-1])
    ok_ds = all(recs[b.ds].relu == 0 for b in blocks if b.ds >= 0)
    check("16 块 residual==1（相加后 ReLU）", all_res1)
    check("16 块末 conv relu==0（激活在相加后）", ok_last)
    check("16 块中间 conv relu==1", ok_mid)
    check("downsample conv relu==0（残差侧不激活）", ok_ds)


# ---------------------------------------------------------------- 2) 数值自检

def _mv2_selfcheck(device):
    x = torch.randn(2, 3, 32, 32, device=device) * 10.0   # 放大触发 ReLU6 截断
    m = folded("mobilenet_v2").to(device)
    r = export_blocks(m, TMP / "_sem_mv2_q2.bin", x,
                      registry.topo_plan("mobilenet_v2"), strict=True)
    return r["selfcheck_maxdiff"]


def test_mv2_numeric():
    print("\n[mv2] 数值自检（放大输入，逼出 ReLU6/相加后语义）")
    d = _mv2_selfcheck("cpu")
    check(f"CPU export_blocks(strict) max|diff| = {d:.3g} < 1e-4", d < 1e-4)

    if torch.cuda.is_available():
        d = _mv2_selfcheck("cuda")
        check(f"CUDA export_blocks(strict) max|diff| = {d:.3g} < 1e-4", d < 1e-4)
    else:
        print("  · CUDA 自检：SKIP（无 GPU；设备放置修复由此覆盖）")


def test_r50_numeric():
    print("\n[r50] CPU 自检回归（应仍为 0）")
    x = torch.randn(2, 3, 32, 32)
    m = folded("resnet50")
    r = export_blocks(m, TMP / "_sem_r50_q2.bin", x,
                      registry.topo_plan("resnet50"), strict=True)
    check(f"R50 max|diff| = {r['selfcheck_maxdiff']:.3g} == 0",
          r["selfcheck_maxdiff"] < 1e-4)


def main():
    torch.manual_seed(0)
    print("PTQ2 激活语义自检\n" + "-" * 62)
    test_mv2_structure()
    test_r50_structure()
    test_mv2_numeric()
    test_r50_numeric()
    print("\n" + "-" * 62)
    if FAILED:
        print(f"✗ {len(FAILED)} 项失败：{', '.join(FAILED)}")
        return 1
    print("✓ 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
