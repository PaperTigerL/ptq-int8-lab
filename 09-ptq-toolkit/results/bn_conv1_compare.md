# conv1 移植方式 × BN 处置 对照实验

arch=`resnet50` · device=`cuda` · threads=`2` · batches=`8` · BN scope=`all` · 评测 `10000` 张

> BN running stats 失配是**全网性**的（stem 7×7s2→3×3s1 且去掉 maxpool，
> 每个 stage 的特征图几何都变），所以 `bn_scope=all` 重估的是**全部**
> BatchNorm2d，不是只 bn1。

## fp32 top-1

| conv1 | 实现 | L1 保留 | BN 处置 | fp32 top-1 |
|---|---|---|---|---|
| A | `center_crop_noscale` | 0.412 | keep | **10.00%** |
| A | `center_crop_noscale` | 0.412 | reestimate | **10.00%** |
| A | `center_crop_noscale` | 0.412 | reset | **10.06%** |
| B | `center_crop_area_scaled` | 2.241 | keep | **10.90%** |
| B | `center_crop_area_scaled` | 2.241 | reestimate | **10.00%** |
| B | `center_crop_area_scaled` | 2.241 | reset | **10.00%** |
| C | `kernel_sum_pool` | 0.531 | keep | **9.61%** |
| C | `kernel_sum_pool` | 0.531 | reestimate | **10.00%** |
| C | `kernel_sum_pool` | 0.531 | reset | **10.39%** |

## conv1 数值自检

| 方式 | src shape | dst shape | src mean\|w\| | dst mean\|w\| | src L1 | dst L1 | L1 比 | 等于中心块 |
|---|---|---|---|---|---|---|---|---|
| A | (64, 3, 7, 7) | (64, 3, 3, 3) | 0.1563 | 0.3502 | 1470.2 | 605.2 | 0.412 | True |
| A | (64, 3, 7, 7) | (64, 3, 3, 3) | 0.1563 | 0.3502 | 1470.2 | 605.2 | 0.412 | True |
| A | (64, 3, 7, 7) | (64, 3, 3, 3) | 0.1563 | 0.3502 | 1470.2 | 605.2 | 0.412 | True |
| B | (64, 3, 7, 7) | (64, 3, 3, 3) | 0.1563 | 1.9068 | 1470.2 | 3294.9 | 2.241 | False |
| B | (64, 3, 7, 7) | (64, 3, 3, 3) | 0.1563 | 1.9068 | 1470.2 | 3294.9 | 2.241 | False |
| B | (64, 3, 7, 7) | (64, 3, 3, 3) | 0.1563 | 1.9068 | 1470.2 | 3294.9 | 2.241 | False |
| C | (64, 3, 7, 7) | (64, 3, 3, 3) | 0.1563 | 0.4517 | 1470.2 | 780.5 | 0.531 | False |
| C | (64, 3, 7, 7) | (64, 3, 3, 3) | 0.1563 | 0.4517 | 1470.2 | 780.5 | 0.531 | False |
| C | (64, 3, 7, 7) | (64, 3, 3, 3) | 0.1563 | 0.4517 | 1470.2 | 780.5 | 0.531 | False |

磁盘 .pth 与内存 src 的 conv1 逐元素相同：`True`
