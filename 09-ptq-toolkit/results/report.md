# PTQ 工具链报告（M2 · 三模型 × 两宽度）

> 本报告由 `ptq report` 生成。所有基准数字是**对 06/07 已有结果的引用**，
> 不是本次重测 —— GPU 让给用户，且重测会引入机器负载这个变量。

## 一、量化构建（三模型 × W8/W4）

| 模型 | 拓扑 | 宽度 | fp32 基线 | 量化后 | 掉点 | 导出 | 自检 |
|---|---|---|---|---|---|---|---|
| MobileNetV2 | seq | W4A8 | 92.87% | 87.65% | 5.22 pp | `results/m2_accept/mobilenet_v2_folded_w4a8.bin`（52 conv / 18 block / 8.9 MB） | 0 |
| MobileNetV2 | seq | W8A8 | 92.87% | 92.91% | −0.04 pp | `results/m2_accept/mobilenet_v2_folded_w8a8.bin`（52 conv / 18 block / 8.9 MB） | 0 |
| ResNet-18 | blocks | W4A8 | 87.16% | 85.30% | 1.86 pp | `resnet18_folded_w4a8.bin`（20 conv / 8 block / 44.7 MB） | 0 |
| ResNet-18 | blocks | W8A8 | 87.16% | 87.08% | 0.08 pp | `resnet18_folded_w8a8.bin`（20 conv / 8 block / 44.7 MB） | 0 |
| ResNet-50 | seq | W4A8 | 91.85% | 88.63% | 3.22 pp | `results/m2_accept/resnet50_folded_w4a8.bin`（53 conv / 16 block / 94.0 MB） | 0 |
| ResNet-50 | seq | W8A8 | 91.85% | 91.85% | 0.00 pp | `results/m2_accept/resnet50_folded_w8a8.bin`（53 conv / 16 block / 94.0 MB） | 0 |

> **2026-10-22 M2 验收（训练权重，全量 10000 张）**：ResNet-50 ← `ckpt_resnet50_r50aug20_ep18.pth`（best 91.85%），MobileNetV2 ← `ckpt_mobilenet_v2_mv2aug20_ep19.pth`（best 92.87%），协议 `cifar10-full-50k-aug-ls0.1-wd0.0005-cos20` + ImageNet 权重移植。
> 4 份 `.bin` 均为 CUDA 导出、拓扑自检 max|diff|=0（2026-10-22 10:07 修复 replay 权重设备放置与 MV2 激活语义后重跑；修复前 MV2 max|diff| 4.7–5.6、R50 只能 CPU 回退）。导出修复不影响权重-only 精度，表中数字与 09:55 验收一致。
> 旧的随机初始化数字（6.25 / 10.65 / 9.80 / 10.80%）作废。

> ⚠️ **「量化后精度」是权重-only 的 fake-quant，不含激活量化**——
> 激活量化在 C++ 引擎侧才真正生效，所以真实 W8A8 引擎精度会更低。
> W4 的掉点尤其要按「权重-only」读：真实 W4A8 会更低。

> `自检`列是导出时的拓扑自检 max|diff|，必须为 0（或 fp32 舍入级 1e-5）。

## 二、基准（引用 06/07 已有结果）

| 路线 | ms/img (median) | 加速 | top-1 | 出处 |
|---|---|---|---|---|
| ncnn CPU fp32 | 11.6000 | 1.0× | — | `05-ncnn-deploy/README.md:473` |
| ncnn CPU int8 | 3.4000 | 3.4× | — | `04-libtorch-cuda/README.md:72` |
| libtorch C++ | 0.0784 | 148× | 87.16% | `04-libtorch-cuda/README.md:74` |
| ncnn CUDA fork | 14.8900 | 0.8× | 88.00% | `05-ncnn-deploy/cuda/README.md:12` |
| TRT fp32 | 0.0276 | 420× | 87.16% | `07-tensorrt-deploy/results/idle_final.txt` |
| TRT fp16 | 0.0149 | 778× | 87.18% | `07-tensorrt-deploy/results/idle_final.txt` |
| TRT int8 | 0.0189 | 614× | 87.22% | `07-tensorrt-deploy/results/idle_final.txt` |

### 自写 CUDA int8 引擎（06）的端到端结果

| 项 | 值 |
|---|---|
| GPU int8 引擎精度（全量 10000 张）| 86.93% |
| W6A8 精度 | 86.91% |
| W4A8 精度 | 85.52% |

出处：`06-gpu-int8/results/e2e_compare.txt`

### TensorRT 三精度（门控口径）

| 配置 | min | median | util 区间 | 精度 | 加速(median) |
|---|---|---|---|---|---|
| TRT fp32 | 0.0274 | 0.0276 | 11-66% | 87.16% | 420× |
| TRT fp16 | 0.0138 | 0.0149 | 26-42% | 87.18% | 779× |
| TRT int8 | 0.0181 | 0.0189 | 11-56% | 87.22% | 614× |

出处：`07-tensorrt-deploy/results/idle_final.txt`

## 三、数据出处

- `summary_csv`：`../lab/08-report/results/summary.csv`
- `idle_final`：`../lab/07-tensorrt-deploy/results/idle_final.txt`
- `engine_e2e`：`06-gpu-int8/results/e2e_compare.txt`

全部为**已有结果的引用**，非本次重测

## 四、S1 权重状态（⚠️ 手写补充，非生成内容）

> ⚠️ **本节是 2026-10-21T10:35+08:00 的快照，已被 2026-10-22 的训练权重验收取代；当前数字见第一节表。**

> 本节是**人工补写**的，不在 `ptq report` 的生成范围
> （生成器：`ptq_toolkit/report/render.py`）。下次执行 `ptq report`
> 会把本节**整段冲掉** —— 要让它持久，得把内容搬进 `render.py`。

（本段为历史快照）当时表里的 ResNet-50 / MobileNetV2 精度（≈10%）是
**随机初始化**下的数字，不是可用结果。S1 的路线是
「torchvision ImageNet 权重移植 + CIFAR 微调」；快照当时只完成了 R50 第一段。

**核查时间 2026-10-21T10:35+08:00**：当前无训练进程
（`nvidia-smi` 无 python 计算进程，GPU 1% / 1086 MiB 为系统占用）。
下列数字均由 `tools/audit_ckpts.py` 从 `.pth` 独立回读，
证据 `results/ckpt_chain_audit.log`；唯一可引用的精度另用
`tools/eval_ckpt.py` 全量复评（`results/eval_ep25_full10k.log`）。

| 项 | 真实状态 |
|---|---|
| ResNet-50 主候选 | **已完成一段连续 10 epoch（ep17–26）**：`protocol=cifar10-full-50k`（无增强）、`n_train=50000`、conv1=A、lr=0.01、batch=64。**best = ep25 84.65%**，最后 ep26 84.52%。可引用检查点：`paper-code/ckpts/ckpt_resnet50_ep25.pth` |
| 独立全量复评 | `ckpt_resnet50_ep25.pth` 在**全量 10000 张**上复评 = **84.65%**，与 ckpt 记录差 **0.00 pp**（GPU，`results/eval_ep25_full10k.log`）。这是本表唯一经过全量复评的数字；ep26 84.52% 仅有训练日志，尚未复评 |
| ResNet-50 早期段 | ep12–16 是另一段连续 5 epoch（best ep16 82.32%）；ep11 是单 epoch 段。**不能与 ep17–26 合并称「训了 22 epoch」** |
| aug 协议 ep27–31 | **全部作废**：ep27 与 ep29 都是重新初始化（fc 余弦 −0.0011 / +0.0006），各自只跑了 2–3 epoch；best ep30 66.51%。与 84.65% 不可比，不得引用 |
| 续跑缺陷 | **已修复（2026-10-21T10:40+08:00）**：`finetune.py` 现在加载权重+优化器+调度器，回读 `max|Δ|==0` 才继续；跨协议/缺 ckpt/目标或超参不一致均拒绝。证据 `results/resume_fix_test.log`（CPU 微型模型 5 项全过，约 7 s） |
| MobileNetV2 全量微调 | **仍未跑**（无 `ckpt_mobilenet_v2_ep*.pth`、无训练日志）；预计时长未实测，必须先跑 1–2 epoch 冒烟取 `seconds/epoch` 再外推 |
| conv1 × BN 九组对照 | **已出结论**：九组全部 9.61~10.90%（chance）→ 不微调怎么处置都没用，**微调必需**，conv1 选 **A**（中心裁剪不缩放） |
| 自检 B | 已改为从磁盘 `.pth` **独立回读**（不再是自证循环），并有随机噪声反例（`tests/test_load_transfer.py` 第 6 项） |

#### 续跑语义（修复后，引用命令前先看这里）

- `--epochs N` = **目标总 epoch 数**（跑完到达 epN），不再表示「再跑 N 个」。
- 同协议历史 + 对应 ckpt + `target_epoch` 一致 → 自动恢复权重/优化器/调度器，
  从 ep+1 跑到 epN；回读不一致直接退出。
- 跨协议、缺 ckpt、旧格式（无 `target_epoch`）→ 拒绝续跑，要求
  `--run-tag` 开新记录，或 `--init-from <pth>` **只加载权重**开新阶段。
- ⛔ 旧 ckpt（ep11–31）都没有 `target_epoch`，**不能严格续跑**。
  要用 84.65% 的 ep25 继续训练，只能走 `--init-from`（新优化器/新调度器），
  并在结果里写明这是第二段，不能把 epoch 号接成 26、27……

#### 下一步（历史快照 · 已作废）

> ⚠️ 以下三条是 2026-10-21 的记录：R50 / MV2 的 20ep 训练已于
> 2026-10-22 完成，当前数字见第一节表；**不要再当作待办执行**。

1. R50 第二段：`--init-from ckpt_resnet50_ep25.pth --run-tag r50aug10 --epochs 10
   --lr 0.003`（增强 + ls0.1），预计 45–80 min，峰值约 1893.7 MiB。
2. MobileNetV2：先 1 epoch 冒烟（时长未实测，先取 `seconds/epoch`），
   再定 20 epoch 窗口。
3. 只有 fp32 达到验收线后，才重建 W8/W4 并跑全量 10000 张；
   当前的 ≈10% 量化表必须整表作废。

## 五、混合精度（I2 落地，2026-10-23）

> 本节为 M2 收尾后**人工补写**，不在 `ptq report` 的生成范围内。
> 生成器（`ptq_toolkit/report/render.py`）下次运行会冲掉本节。

**依据**：`results/i2/layer_sensitivity_resnet50.{csv,json}`（CPU-only，
64 张粗排 53 层 + 512 张复核 12 层 + 512 张真实组合评测）推荐
**top4_all**：W8 4 层（`conv1` / `layer1.0.conv1` / `layer1.0.conv3` /
`layer1.1.conv1`）+ W4 per-group32 49 层，512 张 in-sample 掉点 **0.000 pp**。

**实现**：`ptq export --plan-json ... --plan-name top4_all` 支持显式
per-layer 覆盖；导出后由 `export/mixed_check.py` 按磁盘字节回读自检
（PTQ2 完整解析 / PTQC 层数 8+8n / 逐层 bits-scheme-scale / 权重网格 /
5 个产物 SHA256 / 拓扑自检）。06 的 C++ 代码未改。

**包**：`results/mixed_precision/resnet50_top4_all/`（94.0 MB PTQ2 +
432 B PTQC + 24 MB 逐层量化配置 + layer_plan + manifest + selfcheck）。
自检 **32/32 PASS**，拓扑 `max|diff| = 0`，源 ckpt
`ckpt_resnet50_r50aug20_ep18.pth`（SHA256 `b16cc5aa…137b3`）。

**掉点对比**（同一 test 前 N 张，CPU 前向，仅权重量化，fc fp32）：

| 配置 | 512 张 | 掉点 | 2000 张 | 掉点 | W8 层 |
|---|---|---|---|---|---|
| fp32（R50 ep18） | 92.77% | — | 92.05% | — | — |
| 全 W4 per-channel（M2 口径） | 89.45% | 3.32 pp | — | — | 0 |
| 全 W4 per-group32（I2 口径） | 90.04% | 2.73 pp | — | — | 0 |
| **top4_all** | **92.19%** | **0.59 pp** | **91.55%** | **0.50 pp** | **4** |
| 对照：structural 混合 | — | — | 91.50% | 0.55 pp | 6 |

出处：`results/mixed_precision/accuracy_allw4_vs_top4_all_n512.json`、
`results/mixed_precision/resnet50_top4_all/eval_resnet50_n2000.json`。

> 跨语言抽查：06 的 CPU probe（`ptq_model_probe`，只读）读取本包 rc=0
> （53 conv / 16 block，PTQC n=53）；证据
> `results/mixed_precision/resnet50_top4_all/probe_06_cpu.txt`。C++ 未改。
>
> ⚠️ 口径：这里只量化权重，**不含激活量化**；I2 的 0.0 pp 是 512 张
> 校准子集 + A8 fake-quant 的 in-sample 结果，**不可与上表直接相减**。
> PTQ2 是 fp32 容器：W4 只改变数值分布，不改存储大小/算力。
> 06 引擎的真实 W4A8 精度仍需 GPU + C++ 侧支持，未验证。

## 六、MobileNetV2 混合精度（I3 收口，2026-10-25）

> 与第五节同：本节人工补写，`ptq report` 重新生成会冲掉。

**问题**：第五节只给了 R50 的可用计划；MV2 沿用的 top4_all 实测
1.95 pp @512 / 2.15 pp @2000，**超 1 pp 预算**，必须重做选层。

**三条根因**（`tools/mv2_mixed_study.py` 可复跑，证据在 `results/i3/`）：

1. **口径错位**：I2 的掉点是训练集前 512 张 in-sample + A8 fake-quant
   （fp32 = 97.85%），而验收口径是测试集前 512 张、只量化权重
   （fp32 = 91.80%）。两套数字不能相减。
2. **单层敏感度是噪声级**：52 层里只有 `features.1.conv.0.0`（2.15 pp）与
   `features.1.conv.1`（1.37 pp @g32）超过 512 张的 0.195 pp 分辨率，其余
   全在 ±0.6 pp；I2 的 64 张粗排分辨率 1.56 pp，等于随机选层。
3. **per-group32 是 MV2 上最差的 W4 粒度，且与权重 MSE 反向**：全 W4 g32
   rel-MSE 0.0095 却掉 11.75 pp @2000；per-channel rel-MSE 0.0164 只掉
   7.65 pp。权重 MSE 不能当 MV2 的精度代理。

**方案**：W4 **per-channel**（28 层）+ **W8 保护 24 层**。层序按 512 张上
「单层降到 W4」的边际 CE 伤害降序；k 取最小整数使 512 ≤1.0 pp、
2000 ≤1.5 pp、单次全量 10k ≤1.5 pp **同时**成立。

| 配置 | 512 | 2000 | 全量 10k | W8 层 | 有效比特/权重 |
|---|---|---|---|---|---|
| fp32 | 91.80% | 92.20% | 92.20% | — | 32 |
| 全 W4 per-channel（M2 口径） | −5.86 | −7.65 | — | 0 | 4.25 |
| 全 W4 per-group32（I2 口径） | −9.77 | −11.75 | — | 0 | 5.08 |
| 全 W8 per-channel | −0.20 | −0.10 | — | 52 | 8.25 |
| top4_all（第一轮） | −1.95 | −2.15 | — | 4 | 5.09 |
| 备用 `w4g16_w8top21` | −0.39 | −1.10 | −1.08 | 21 | 6.74 |
| **`w4pc_w8top24`（收口）** | **−0.39** | **−0.75** | **−0.92** | **24** | **4.99** |

**结论**：1.95 → **0.39 pp @512**、2.15 → **0.75 pp @2000**、全量 10k
**0.92 pp**；有效比特从 5.09 降到 4.99，不是拿体积换精度。
代价是 W8 层从 4 涨到 24（**只有 28/52 层停在 W4**）—— MV2 在 W4 下很脆，
这是本轮实测的硬结论。

**产物**：`results/mixed_precision/mobilenet_v2_w4pc_w8top24/`
（52 conv / 18 block、PTQC 424 B、自检 32/32、拓扑 max|diff| = 0；
`eval_mobilenet_v2_n{512,2000}.json` 为 `ptq eval` 官方口径）。
计划与复核证据：`results/i3/plan_mobilenet_v2_w4pc_w8top24.json`
（含 `verified[]`：2000 与 10k 两条独立实测，层名必须一致才允许合并）。
验收：`python tests/test_mv2_plan_i3.py`（断言 512 张 ≤1.0 pp）。

**剩余风险**：

- 24/52 层 W8、有效比特 4.99，压 W4 比例的空间已被 W4 掉点吃掉；
  再进一步需要先解释 per-group 在 MV2 上的异常（本轮只观测、未定位机制）。
- 只量化权重，不含激活量化；06 引擎的真实 W4A8 仍需 GPU 侧支持。
- 512 子集普遍比 2000/10k 乐观 ~0.3–0.7 pp，所以选层后的复核必须 ≥2000 张；
  本轮的 k 就是被 2000/10k 卡上去的（k=11 在 512 上已达标，2000 是 1.40）。

