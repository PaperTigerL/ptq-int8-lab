# -*- coding: utf-8 -*-
"""权重移植自检的**负向测试** —— 证明这道闸门真的会拦人。

为什么需要它
============
一条"从来没失败过"的自检，等于没有自检。`tools/load_transfer.py` 的 A/B/E 三条
自检如果只跑通过路径，就无法区分「检查生效」与「检查写错成永远通过」。
本文件专门构造**应当被拦下**的输入：

  1. 改名一个非 fc 的 key            → 自检 A 必须 SystemExit(1)
  2. 多出一个非 fc 的 unexpected key  → 自检 A 必须 SystemExit(1)
  3. 只有 fc.* 差异                   → 自检 A **必须放行**（否则闸门过紧，正常流程走不通）
  4. conv1 中心裁剪的窗口位置          → 必须是 offset=2 的中心，不是左上角
  5. 把 bn1 权重重扰动                → 自检 E 必须 SystemExit(1)
  6. 把权重换成随机噪声（键集合仍干净）→ 自检 B 必须 SystemExit(1)
     —— 这一条是「自检 B 不能自证」的证明：旧实现拿 model.state_dict()
        与刚灌进去的 dict 比，注入随机噪声照样全绿。

不下载权重：正向对拍把模型自己的 state_dict 当 src 用（权重值无所谓，
检查的是"同一份权重下两套 forward 是否等价"）。

用法：python tests/test_load_transfer.py
"""
import contextlib
import io
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from tools.load_transfer import (                            # noqa: E402
    center_crop_conv1, check_disk_match, check_keys,
    ref_equivalence_check)

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'✓' if cond else '✗'} {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILED.append(name)


@contextmanager
def expect_exit(name, code=1):
    """断言这段代码以 SystemExit(code) 结束。"""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stderr(buf):
            yield
    except SystemExit as e:
        check(name, e.code == code, f"退出码 {e.code}")
    else:
        check(name, False, "没有退出 —— 闸门失效")


def main():
    torch.manual_seed(0)
    print("权重移植自检 · 负向测试\n" + "-" * 62)

    from ptq_toolkit.models.resnet50 import build as build50
    m = build50(num_classes=10).eval()
    sd = m.state_dict()

    # ---- 1) 非 fc 的 missing 必须被拦 ----
    print("\n[1] 改一个非 fc 的 key")
    bad = dict(sd)
    bad["layer1.0.conv1x.weight"] = bad.pop("layer1.0.conv1.weight")
    missing, unexpected = m.load_state_dict(bad, strict=False)
    with expect_exit("missing 含非 fc key → 退出码 1"):
        check_keys(missing, unexpected)

    # ---- 2) 非 fc 的 unexpected 必须被拦 ----
    print("\n[2] 多一个非 fc 的 key")
    bad2 = dict(sd)
    bad2["ghost.weight"] = torch.zeros(1)
    missing2, unexpected2 = m.load_state_dict(bad2, strict=False)
    with expect_exit("unexpected 含非 fc key → 退出码 1"):
        check_keys(missing2, unexpected2)

    # ---- 3) 只有 fc.* 差异必须放行 ----
    print("\n[3] 只有 fc.* 差异")
    ok = {k: v for k, v in sd.items() if not k.startswith("fc.")}
    missing3, unexpected3 = m.load_state_dict(ok, strict=False)
    try:
        check_keys(missing3, unexpected3)
        check("missing 只有 fc.* → 放行", True,
              f"missing={list(missing3)} unexpected={list(unexpected3)}")
    except SystemExit:
        check("missing 只有 fc.* → 放行", False, "被误拦")

    # ---- 4) conv1 中心裁剪窗口 ----
    print("\n[4] conv1 中心裁剪窗口")
    src = torch.arange(64 * 3 * 7 * 7, dtype=torch.float32).reshape(64, 3, 7, 7)
    crop, off = center_crop_conv1(src, 3)
    check("形状 (64,3,3,3)", tuple(crop.shape) == (64, 3, 3, 3), str(tuple(crop.shape)))
    check("offset == 2", off == 2, f"off={off}")
    check("等于 src[:,:,2:5,2:5]（不是左上角）",
          torch.equal(crop, src[:, :, 2:5, 2:5]))
    check("不等于左上角 src[:,:,0:3,0:3]",
          not torch.equal(crop, src[:, :, 0:3, 0:3]))
    # 若实现里偷偷乘了 49/9 之类的补偿因子，L1 质量会超过源权重
    check("不缩放：L1 质量小于源（未乘补偿因子）",
          float(crop.abs().sum()) < float(src.abs().sum()),
          f"{float(crop.abs().sum()):.0f} < {float(src.abs().sum()):.0f}")

    # ---- 6) 自检 B 必须能抓到「权重被污染但键集合干净」----
    print("\n[6] 自检 B 的反例：注入随机权重后必须红")
    #
    # 这一条针对的是一个**真实的失效模式**：
    #   旧的自检 B 拿 `model.state_dict()` 与刚灌进去的 load_sd 比 ——
    #   那只是在证明 load_state_dict 拷了张量。把 load_sd 换成随机噪声，
    #   键集合一个不变，照样报 318/318。
    # 现在自检 B 独立回读磁盘 .pth 再与模型比，注入噪声必须红。
    from ptq_toolkit.models import get_model
    from ptq_toolkit.models import registry

    m6 = get_model("resnet50").eval()
    disk = {k: v.clone() for k, v in m6.state_dict().items()}
    # 只扰动一个非 fc 的 conv 权重，键集合完全不变
    disk["layer1.0.conv1.weight"] = torch.randn_like(
        disk["layer1.0.conv1.weight"])
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            check_disk_match(m6, disk, "fc.", verbose=False)
        check("注入随机权重 → 自检 B 必须失败", False, "竟然放行了（自证循环？）")
    except SystemExit as e:
        check("注入随机权重 → 自检 B 必须失败", e.code == 1, f"exit={e.code}")

    # 对照：**还原** disk 后必须放行（否则闸门过紧，正常流程走不通）
    # ⚠️ 上一用例已经把 disk["layer1.0.conv1.weight"] 换成噪声，
    #    这里必须重新 clone 一份干净的，否则对照组必然红。
    disk_clean = {k: v.clone() for k, v in m6.state_dict().items()}
    try:
        n = check_disk_match(m6, disk_clean, "fc.", verbose=False)
        check("不扰动 → 自检 B 放行", n > 0, f"{n} 个张量全部一致")
    except SystemExit:
        check("不扰动 → 自检 B 放行", False, "被误拦")

    # ---- 5) 结构对拍必须能抓到 forward 图差异 ----
    print("\n[5] 自检 E 的失败路径（把 bn1 权重重扰动）")
    src5 = m.state_dict()
    with contextlib.redirect_stdout(io.StringIO()):
        d = ref_equivalence_check(m, src5, verbose=False)
    check("未扰动时 max|diff| == 0", d == 0.0, f"max|diff|={d:.3g}")

    m2 = build50(num_classes=10).eval()
    m2.load_state_dict(src5)
    with torch.no_grad():
        m2.bn1.weight.mul_(1.5)            # 只改权重，key 集合完全不变
    with expect_exit("bn1 被扰动 → 退出码 1"):
        with contextlib.redirect_stdout(io.StringIO()):
            ref_equivalence_check(m2, src5, verbose=False)

    print("\n" + "-" * 62)
    if FAILED:
        print(f"✗ {len(FAILED)} 项失败：{', '.join(FAILED)}")
        return 1
    print("✓ 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
