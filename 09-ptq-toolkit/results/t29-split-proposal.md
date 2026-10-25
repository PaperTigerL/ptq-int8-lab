# T-20261021-29 · 45k/5k train/val split 生成器草案（纯函数 · 未执行）

- 作者：ptq
- 日期：2026-10-21 22:05 (+08:00)
- 性质：**只写草案文档，未执行、未落盘、不改任何训练代码**。
- 背景：T-04 评估方案（`proposals/ptq-eval-plan.md §2`）已给出确定性分层算法与
  `split_sha256=2329a983…`，但 **09-ptq-toolkit 现有工具中没有任何 split 生成器**
  （`tools/` 下只有 `finetune.py / eval_ckpt.py / audit_ckpts.py / load_transfer.py /
  bn_reestimate.py`，无 `make_split.py` / `check_split.py`；全仓对 `split` 的命中仅
  `load_transfer.py:141` 的 `rsplit("/",1)` 字符串拆分，属误匹配）。故本单裁决为「未落盘」，
  本文给出可评审、可复现的生成脚本草案。

---

## 1. 口径来源（为什么需要一个与 test 无关的 val）

- `tools/finetune.py:129` 硬校验 `n_train == 50000`，每个 epoch 在全量 10k test 上评估
  （`finetune.py:338-346`）→ 历史 `84.65%` 是 **best-on-test**。
- 项目已复核口径（T-04 方案 §2.5 硬闸门 #4/#6、§2.7 点 2）：**选模/选起点规则必须预先登记，
  且不得由 test 值决定**；`test_eval_count` 必须为 1，>1 即判「test 被用于选模」。
- 因此需要一块 **与 test 无关的 val（5k）**：训练期只在 val 上选 best/last，test 只在最终评一次。
- 注意（T-04 §2.7 点 1）：现有 ep25/ep26 训练时已见过全部 50k，**不能**拿它们在这 5k val 上做
  任何「干净」比较；45k 臂必须从头训练（同 ImageNet 初始化、同协议），不能用 `--init-from`。

## 2. 纯函数生成脚本草案（`tools/make_split.py`，本单不落盘、不执行）

固定 seed、确定性分层、只依赖标签，输出 manifest + 自检 + SHA256。

```python
#!/usr/bin/env python3
"""确定性 45k/5k 分层切分生成器（草案，纯函数；不访问 GPU/网络，只读 data_batch_*.）。
用法（批准后）：python tools/make_split.py --data-dir <cifar-10-batches-py> --out <split.json>
"""
import argparse, hashlib, json
import numpy as np
import pickle

SEED = 20261021
VAL_PER_CLASS = 500
N_CLASSES = 10
DATA_SHA256 = {  # 固化数据指纹，任一不符即 abort（防切分建立在被改动过的数据上）
    "data_batch_1": "54636561a3ce25bd3e19253c6b0d8538147b0ae398331ac4a2d86c6d987368cd",
    "data_batch_2": "766b2cef9fbc745cf056b3152224f7cf77163b330ea9a15f9392beb8b89bc5a8",
    "data_batch_3": "0f00d98ebfb30b3ec0ad19f9756dc2630b89003e10525f5e148445e82aa6a1f9",
    "data_batch_4": "3f7bb240661948b8f4d53e36ec720d8306f5668bd0071dcb4e6c947f78e9682b",
    "data_batch_5": "d91802434d8376bbaeeadf58a737e3a1b12ac839077e931237e0dcd43adcb154",
    "test_batch":   "f53d8d457504f7cff4ea9e021afcf0e0ad8e24a91f3fc42091b8adef61157831",
    "batches.meta": "f962466ef690d46b226450fb9aadc74ba4bc64a76aa526b5827fe4bc5c7125cb",
}


def load_train_labels(data_dir):
    """按 batch_1..5 顺序拼接标签，返回全局 0..49999 的 label 数组。"""
    labels = []
    for b in range(1, 6):
        with open(f"{data_dir}/data_batch_{b}", "rb") as f:
            d = pickle.load(f, encoding="bytes")
        labels.extend(d[b"labels"])
    return np.asarray(labels, dtype=np.int64)


def make_split(labels, seed=SEED, val_per_class=VAL_PER_CLASS):
    """确定性分层：类顺序固定 0..9；每类 rng.permutation，前 val_per_class 进 val。
    train/val 各自升序排序后返回（升序保证哈希与迭代顺序无关、可复现）。"""
    rng = np.random.default_rng(seed)
    train_idx, val_idx, per_class = [], [], {}
    for c in range(N_CLASSES):
        idx_c = np.flatnonzero(labels == c)          # 升序
        perm = rng.permutation(idx_c)
        v = sorted(perm[:val_per_class].tolist())
        t = sorted(perm[val_per_class:].tolist())
        val_idx.extend(v)
        train_idx.extend(t)
        per_class[c] = [len(t), len(v)]
    train_idx.sort()
    val_idx.sort()
    return train_idx, val_idx, per_class


def canonical_hash(seed, val_per_class, train_idx, val_idx):
    blob = json.dumps({"seed": seed, "val_per_class": val_per_class,
                       "train_idx": train_idx, "val_idx": val_idx},
                      sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest(), blob


def selfcheck(labels, train_idx, val_idx, per_class):
    """四条硬闸门，任一失败即抛异常（调用方据此 abort）：
    ① 45k/5k；② 每类 4500/500；③ train∩val=∅ 且并集=0..49999；④ 每类成员与标签一致。"""
    assert len(train_idx) == 45000 and len(val_idx) == 5000, "① 数量不对"
    assert all(per_class[c] == [4500, 500] for c in range(N_CLASSES)), "② 分层不均"
    ts, vs = set(train_idx), set(val_idx)
    assert ts.isdisjoint(vs), "③ train/val 有交集"
    assert ts | vs == set(range(50000)), "③ 并集非 0..49999"
    for c in range(N_CLASSES):
        expect = np.flatnonzero(labels == c)
        got = np.array(sorted([i for i in train_idx if i in set(expect.tolist())]
                              + [i for i in val_idx if i in set(expect.tolist())]))
        assert len(got) == len(expect), "④ 类成员丢失"
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # 数据指纹自检（在切分前，确保标签来自未改动的原版数据）
    for name, want in DATA_SHA256.items():
        h = hashlib.sha256(open(f"{args.data_dir}/{name}", "rb").read()).hexdigest()
        assert h == want, f"数据指纹不符: {name} {h[:8]}!={want[:8]}"

    labels = load_train_labels(args.data_dir)
    train_idx, val_idx, per_class = make_split(labels)
    selfcheck(labels, train_idx, val_idx, per_class)
    split_sha, _ = canonical_hash(SEED, VAL_PER_CLASS, train_idx, val_idx)

    # 期望值：与 T-04 §2.1 只读计算结果对齐；若不一致说明数据/算法漂移，立即停
    EXPECT_SHA = "2329a983869a2de1c5c428a5bc33e5729a74dd21e7f9e43f9188b9823cdf0a45"
    assert split_sha == EXPECT_SHA, f"split_sha 漂移: {split_sha} != {EXPECT_SHA}"

    manifest = {
        "split_id": f"seed{SEED}_val{len(val_idx)}",
        "seed": SEED, "val_per_class": VAL_PER_CLASS,
        "n_train": len(train_idx), "n_val": len(val_idx),
        "per_class": per_class,
        "train_idx": train_idx, "val_idx": val_idx,
        "dataset_sha256": DATA_SHA256,   # 注意：不写 test 索引，test 永远为官方 10k
        "split_sha256": split_sha,
        "created_at": None,              # 落盘时由执行方填 ISO 时间；哈希输入不含此字段
    }
    with open(args.out, "w") as f:
        json.dump(manifest, f)
    print("OK", split_sha, "->", args.out)


if __name__ == "__main__":
    main()
```

## 3. 校验脚本草案（`tools/check_split.py`，独立回读，不重算选择）

```python
# 只读校验：重新用同一 seed/算法算一遍，比对 manifest 内的 train_idx/val_idx/split_sha256。
# 断言：① 数量 45k/5k；② 每类 4500/500；③ 无交集、并集=0..49999；④ 重算 split_sha == 记录值；
#       ⑤ manifest.dataset_sha256 与磁盘实际数据指纹一致。任一失败 exit≠0。
```

## 4. 输出 manifest 规范

| 字段 | 说明 |
|---|---|
| `split_id` | `seed20261021_val5000`（进 ckpt/history/eval 三处元数据，保证可追溯） |
| `seed` | `20261021`（训练前写死；事后改 seed 使切分失效，见 T-04 §2.7 点 4） |
| `val_per_class` | `500` |
| `n_train / n_val` | `45000 / 5000` |
| `per_class` | `{0:[4500,500], …, 9:[4500,500]}` |
| `train_idx / val_idx` | 升序全局索引（0..49999） |
| `dataset_sha256` | 7 个 CIFAR-10 文件指纹（见 §2） |
| `split_sha256` | 规范化 JSON（sort_keys, `separators=(',',':')`, 不含 created_at）的 SHA256 |
| `created_at` | 仅记录，**不参与哈希** |

- 落盘路径（**本单未创建**）：`lab/09-ptq-toolkit/results/split/split_seed20261021_val5000.json`。
- 不写 test 索引：test 永远保持官方 10k，只在最终评一次。

## 5. 确定性锚点（供执行后核对）

- `n_train=45000  n_val=5000`，每类 `[4500, 500]`。
- `split_sha256 = 2329a983869a2de1c5c428a5bc33e5729a74dd21e7f9e43f9188b9823cdf0a45`
  （来自 T-04 方案 §2.1 的只读计算；执行方必须在 `make_split` 里 assert 命中，否则视为数据/算法漂移）。

## 6. 落地前提（本单不执行，仅列出让评审知道还差什么）

1. 负责人批准改码：新增 `tools/make_split.py`、`tools/check_split.py`，并给
   `ptq_toolkit/pipeline.py` 加 `get_split_loaders`、`tools/finetune.py` 加 `--split-json`
   （校准只从 45k train 取，训练期不碰 test），`tools/eval_ckpt.py` 加 `--eval-role`。
2. 上述代码改动 + `n_train==50000` 闸门改为「等于 split train 数」，属训练代码变更，需单独派工与批准。
3. 生成/校验为 CPU、<1 s；45k 训练需 GPU 窗口（BOSS 批准）。

## 7. 风险

- **污染不可逆**：若先看了 test 再定 seed/选模规则，切分即失去意义；seed 与选择规则（建议 last）
  必须在训练前登记。
- **旧权重不能当干净 val 基线**：ep25/ep26 见过全部 50k，只能作历史 best/last 参考，5k val 上的
  任何数都不算干净（T-04 §2.7 点 1）。
- **5k 抽样噪声**：85% 附近 5k val 的 3σ ≈ 1.5 pp；0.5 pp 差异不足以选模，须与 last 并报。
