# -*- coding: utf-8 -*-
"""探针：确认训练增强**真的生效了**，而且**没漏进**校准/评测。

为什么要有这个探针
------------------
`get_train_loader(augment=True)` 是把 RandomCrop / RandomHorizontalFlip
**插到 03 的 transform 之前**实现的。03 的 `train_transform()` 是
`Compose([ToTensor(), Normalize])` —— 而这两个增强吃的是 **PIL 图**。

一旦顺序插错（插到 ToTensor 之后），增强会**静默失效**：
不报错、不崩、形状也对，只是永远返回原图。
唯一的外部症状是「train acc 比无增强时低」—— 但增强本来就会让 train acc
变低，所以靠精度根本分辨不出「生效了」还是「插错了」。

所以必须直接验三件事：
  1. transform 链里**有** RandomCrop / RandomHorizontalFlip
  2. 它们排在 **ToTensor 之前**
  3. 同一张图取两次**不一样**（随机性真的在起作用）

第 4 条是反向的：校准/评测的 transform **必须没有**这两个增强。

用法（CPU，不碰 GPU）
--------------------
    python tests/check_augment.py
    # 退出码 0 = 全通过；非 0 = 有失败项
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLKIT = HERE.parent
sys.path.insert(0, str(TOOLKIT))

import torch  # noqa: E402
import torchvision.transforms as T  # noqa: E402


def _names(tf):
    """把一个 transform（可能是 Compose）摊平成类名列表。"""
    if isinstance(tf, T.Compose):
        return [type(t).__name__ for t in tf.transforms]
    return [type(tf).__name__]


def main():
    from ptq_toolkit.pipeline import get_loaders, get_train_loader

    fails = []

    def check(name, ok, detail=""):
        print(f"  {'✓' if ok else '✗'} {name}" + (f"  {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    print("== 训练集 augment=True ==")
    aug_loader = get_train_loader(batch_size=4, augment=True, num_workers=0)
    aug_ds = aug_loader.dataset
    names = _names(aug_ds.transform)
    print(f"  transform = {names}")

    check("含 RandomCrop", "RandomCrop" in names)
    check("含 RandomHorizontalFlip", "RandomHorizontalFlip" in names)

    # 顺序：增强必须在 ToTensor 之前，否则吃不到 PIL 图 → 静默失效
    i_to = names.index("ToTensor") if "ToTensor" in names else None
    i_rc = names.index("RandomCrop") if "RandomCrop" in names else None
    i_hf = names.index("RandomHorizontalFlip") if "RandomHorizontalFlip" in names \
        else None
    check("ToTensor 在链里", i_to is not None)
    if i_to is not None:
        check("RandomCrop 在 ToTensor 之前", i_rc is not None and i_rc < i_to,
              f"(idx {i_rc} < {i_to})")
        check("RandomHorizontalFlip 在 ToTensor 之前",
              i_hf is not None and i_hf < i_to, f"(idx {i_hf} < {i_to})")
        check("Normalize 在最后", names[-1] == "Normalize", f"末尾={names[-1]}")

    # 随机性真的在起作用：同一 index 取两次应不同
    a = aug_ds[0][0]
    b = aug_ds[0][0]
    check("同一 index 两次取样不同（随机性生效）",
          not torch.equal(a, b))

    print("== 训练集 augment=False（默认） ==")
    plain = get_train_loader(batch_size=4, num_workers=0).dataset
    pn = _names(plain.transform)
    print(f"  transform = {pn}")
    check("默认无 RandomCrop", "RandomCrop" not in pn)
    check("默认无 RandomHorizontalFlip", "RandomHorizontalFlip" not in pn)

    print("== 校准 / 评测 必须无增强 ==")
    calib, test, _, _ = get_loaders(calib_n=128, eval_n=0)
    for tag, loader in (("校准集", calib), ("测试集", test)):
        ds = loader.dataset
        # 校准集是 Subset，transform 在其 .dataset 上
        inner = ds.dataset if hasattr(ds, "dataset") else ds
        tn = _names(inner.transform)
        print(f"  {tag} transform = {tn}")
        check(f"{tag} 无 RandomCrop", "RandomCrop" not in tn)
        check(f"{tag} 无 RandomHorizontalFlip",
              "RandomHorizontalFlip" not in tn)

    print()
    if fails:
        print(f"✗ {len(fails)} 项失败：{fails}")
        return 1
    print("✓ 全部通过：增强只作用于训练集，且排在 ToTensor 之前")
    return 0


if __name__ == "__main__":
    sys.exit(main())
