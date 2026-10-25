# CIFAR-100 基线问题诊断与重训方案

> 2026-10-17
> 这是「投稿前必做」清单里的第一项。

---

## 问题

两个 CIFAR-100 基线都偏低：

| 组合 | best test | ResNet-18 在 CIFAR-100 的正常水平 |
|---|---|---|
| CIFAR-100 + ResNet-18 | 60.56% | **75%+** |
| CIFAR-100 + MobileNetV2 | 55.91% | 参考：MobileNetV2 CIFAR-100 约 65-70% |

**相对比较仍然有效**（所有掉点、敏感度都在同一基线上测），
但**绝对精度不可与文献对比**。

---

## ⚠️ 诊断：两个组合的病因**不同**

我一开始笼统地写"过拟合严重"，**这个判断只对 ResNet-18 成立**。

### CIFAR-100 + ResNet-18 —— 过拟合为主

```
ep 46  train 94.04%  test 60.38%  gap +33.66
ep 50  train 96.34%  test 60.56%  gap +35.78
```

**gap 35.78 pp**，train 已经 96%+ 而 test 停在 60%。
典型过拟合。test 虽仍在微涨（60.38→60.56），但已接近平台期。

**修法**：
- weight decay `5e-4 → 5e-3`（10 倍）
- label smoothing `0.1 → 0.2`
- epoch `50 → 100`（有正则后需要更多 epoch 收敛）

### CIFAR-100 + MobileNetV2 —— 欠拟合为主

```
ep 46  train 61.33%  test 55.22%  gap +6.11
ep 50  train 64.22%  test 55.91%  gap +8.31
```

**gap 仅 8.31 pp**，且 train 只有 64%——**模型还没拟合上**。
**这是训练不足，不是过拟合。**

**修法**：
- **只延长 epoch**（`50 → 150`）
- **正则保持默认**（wd 5e-4 / ls 0.1）——对欠拟合加正则反而更差

---

## 为什么必须分开处理

| | ResNet-18 | MobileNetV2 |
|---|---|---|
| 病因 | 过拟合（gap 35.78） | 欠拟合（gap 8.31） |
| train acc | 96.34%（已拟合） | 64.22%（未拟合） |
| 该加正则？ | ✅ 加（wd↑ ls↑） | ❌ 不能加 |
| 该延 epoch？ | ✅ 到 100 | ✅ 到 150 |

**用同一套参数重训两个组合，至少有一个会更差。**

这是诊断的价值——**看一眼 train/test gap 和 train 绝对值，就能判断是过拟合还是欠拟合**：
- gap 大 + train 高 → 过拟合 → 加正则
- gap 小 + train 低 → 欠拟合 → 延 epoch / 提容量

---

## 重训命令

```bash
cd /mnt/d/WorkBuddy/ShowBro/lab/03-ptq-lab

# 治过拟合
~/pyenv-torch/bin/python tools/train_baseline.py \
    --dataset cifar100 --arch resnet18 --epochs 100 \
    --wd 5e-3 --ls 0.2 --tag reg

# 治欠拟合
~/pyenv-torch/bin/python tools/train_baseline.py \
    --dataset cifar100 --arch mobilenetv2 --epochs 150 \
    --wd 5e-4 --ls 0.1 --tag long
```

（`--wd` / `--ls` 是 2026-10-17 新加的，此前硬编码无法按组合区分。）

---

## 重训后必须做的事

1. **检查新基线**：ResNet-18 应显著高于 60.56%（目标 70%+），
   MobileNetV2 应高于 55.91%（目标 62%+）
2. **重跑该组合的全部 e1-e7** —— 换了基线，所有数字都变了
3. **更新四组横向对比表** 和 README 第一章
4. **重新核对回收率** —— 基线上去了，W4A8 掉点和 AdaRound 收益都会变，
   「回收率按数据集聚类」这个结论需要重新验证

> ⚠️ **第 4 点最重要**：核心发现建立在当前这组基线上。
> 基线变了，结论要重新算——不能假设它自动成立。

---

## 当前状态

- [ ] CIFAR-100 + ResNet-18 重训（治过拟合）
- [ ] CIFAR-100 + MobileNetV2 重训（治欠拟合）
- [ ] 重跑两组的 e1-e7
- [ ] 更新四组横向对比表
- [ ] 重新核对核心发现

**在完成前，论文里所有 CIFAR-100 的绝对精度都不可用。**
