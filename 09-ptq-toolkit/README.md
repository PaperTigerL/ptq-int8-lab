# 09-ptq-toolkit · PTQ 工具链

一条命令从 **PyTorch 模型 + 校准数据**走到 **per-channel PTQ → 导出权重/拓扑 → 基准 → 报告**。

当前里程碑：**M2（多模型 + W4A8，端到端跑通）**。
M3 TRT 对比、M4 docs+CI 之后逐步开。

> **S1 已收尾（ResNet-50 / MobileNetV2 的可用权重，2026-10-22）**
> · ResNet-50 全量微调**已完成**：20ep best = ep18 **91.85%**（全量 10k）
> · MobileNetV2 全量微调**已完成**：20ep best = ep19 **92.87%**；
> W8 权重-only 92.91%（−0.04 pp）、W4 权重-only 87.65%（5.22 pp）
> · conv1×BN 九组对照**已出结论**：不微调全部 ≈10%（chance），
> 微调必需，conv1 选 **A**
> · 自检 B 已改为磁盘**独立回读**，并有随机噪声反例
> 细节见下方「S1：ResNet-50 / MobileNetV2 的可用权重」章节。

---

## 10 分钟上手

```bash
cd /mnt/d/WorkBuddy/ShowBro/lab/09-ptq-toolkit

# 一键跑通（三模型 × W8/W4 → bench → report → 验收脚本）
bash run_all.sh
```

只用 CPU，不碰 GPU。约 10–15 分钟。

> ⚠️ `run_all.sh` 用**默认/随机初始化权重**，只验证「代码路径能跑通」；
> 它会把 root 的 `results/build_*.json` 刷成 ≈10% 的旧口径。
> **M2 的真实精度数字在 `results/m2_accept/`**（训练权重 + `--weights` 复现，
> 命令见下），不要用 run_all 生成的 report 当 M2 结果。

### 分步跑

```bash
source ~/pyenv-torch/bin/activate

# 1) 量化 + 导出（折叠 BN、校准激活、per-channel 量化、写 .bin 与量化参数）
#    --model all = 依次跑 registry 里注册的全部模型
python -m ptq_toolkit build --model all --w-bits 8 --calib-n 128
python -m ptq_toolkit build --model all --w-bits 4 --calib-n 128

# 单个模型
python -m ptq_toolkit build --model resnet50 --w-bits 8 --calib-n 128

# 1b) 混合精度（I2 推荐计划 top4_all：4 层 W8 + 49 层 W4）
python -m ptq_toolkit export --model resnet50 \
  --weights /mnt/d/WorkBuddy/ShowBro/paper-code/ckpts/ckpt_resnet50_r50aug20_ep18.pth \
  --plan-json results/i2/layer_sensitivity_resnet50.json --plan-name top4_all \
  --calib-n 512 --device cpu --out-dir results/mixed_precision/resnet50_top4_all
python -m ptq_toolkit mixed-check results/mixed_precision/resnet50_top4_all

# 2) 基准（**引用** 06/07 已有结果，不重新测 GPU —— 见下方「为什么不重测」）
python -m ptq_toolkit bench

# 3) 报告（Markdown + JSON）
python -m ptq_toolkit report

# 4) 验收脚本
python tests/test_quant_ops.py          # 量化口径与 03 逐位对照
python tests/test_models.py             # 三模型：前向形状 / 参数量 / 拓扑自检
python tests/test_topology_semantics.py # 激活语义：结构断言 + ×10 输入（MV2 ReLU6 / 残差）
python tests/check_binary_compat.py     # ResNet-18 的 .bin 与 06 引擎二进制格式一致
python tests/test_golden.py             # 精度对齐 06/07 + 三模型回归基线
```

### 只想验证不想重跑

```bash
SKIP_BUILD=1 bash run_all.sh
```

---

## M2 实测结果（三模型 × W8/W4）

| 模型 | 参数 | 拓扑 | 宽度 | fp32 基线 | 量化后 | 掉点 | 自检 |
|---|---|---|---|---|---|---|---|
| ResNet-18 | 11.18 M | blocks | W8A8 | **87.16%** | 87.08% | 0.08 pp | 0 |
| ResNet-18 | 11.18 M | blocks | W4A8 | 87.16% | 85.30% | 1.86 pp | 0 |
| ResNet-50 | 23.52 M | seq | W8A8 | **91.85%** | 91.85% | 0.00 pp | 0 |
| ResNet-50 | 23.52 M | seq | W4A8 | 91.85% | 88.63% | **3.22 pp ✗** | 0 |
| MobileNetV2 | 2.24 M | seq | W8A8 | **92.87%** | 92.91% | −0.04 pp | 0 |
| MobileNetV2 | 2.24 M | seq | W4A8 | 92.87% | 87.65% | **5.22 pp ✗** | 0 |

每个数字都可溯源到 `results/m2_accept/build_<model>_w<bits>.json`。
root 的 `results/build_<model>_w<bits>.json` 是训练权重接入前的随机初始化旧产物，
**不要再引用**；`test_golden.py` 仍读 root 文件（ResNet-18 外部对齐有效，
R50/MV2 段只是「量化+导出没把网络算坏」的旧回归，真精度以本表为准）。

> **MobileNetV2 独立复核（2026-10-23）**：ckpt SHA256 与
> `paper-code/ckpts/SHA256SUMS` 一致（`fc11f14…a64ff72`）；
> 用 `ptq eval` 在前 2000 张上复跑 fp32 **92.85%** / W8 **92.95%** /
> W4 **87.80%**，与全量 10k 的 92.87 / 92.91 / 87.65 差 0.02–0.15 pp
> （`results/m2_accept/eval_mobilenet_v2_n2000.json`）。
> 数字没有重跑 10k —— 按纪律复用已有全量结果，只做抽样对账。

### 训练权重验收（2026-10-22，全量 10k）

- R50 ← `D:/WorkBuddy/ShowBro/paper-code/ckpts/ckpt_resnet50_r50aug20_ep18.pth`（20ep best 91.85%）
- MV2 ← `D:/WorkBuddy/ShowBro/paper-code/ckpts/ckpt_mobilenet_v2_mv2aug20_ep19.pth`（20ep best 92.87%）
- 协议：`cifar10-full-50k-aug-ls0.1-wd0.0005-cos20` + ImageNet 权重移植；fp32 基线为同一 test set 全量 10k。
- 判定：**W8 PASS**（R50 0.00 pp / MV2 −0.04 pp，门槛 ≤1pp）；**W4 FAIL**（R50 3.22 pp / MV2 5.22 pp，门槛 ≤3pp）。
- 产物：R50 / MV2 的 W8 与 W4 共 4 份（`results/m2_accept/`，CUDA 导出，拓扑自检 max|diff| 均为 0）：R50 53 conv / 16 block / 94.0 MB，MV2 52 conv / 18 block / 8.9 MB。
- 导出修复（2026-10-22 10:07）：replay 权重跟随输入设备；PTQ2 conv `relu` 扩为 0/1/2（无激活 / ReLU / ReLU6）、block `residual` 扩为 0/1/2（无残差 / 相加后 ReLU / 相加后无激活），MV2 按真实前向声明。回归测试：`tests/test_topology_semantics.py`（结构断言 + ×10 输入；原 `test_models.py` 随机初始化下是假阴性）。

复现（WSL，`/home/user/pyenv-torch/bin/python`，`PYTHONPATH=.`）：

```bash
python -m ptq_toolkit build --model resnet50 --weights /mnt/d/WorkBuddy/ShowBro/paper-code/ckpts/ckpt_resnet50_r50aug20_ep18.pth --w-bits 8 --a-bits 8 --calib-n 512 --eval-n 10000 --device cuda --out-dir results/m2_accept
python -m ptq_toolkit build --model resnet50 --weights /mnt/d/WorkBuddy/ShowBro/paper-code/ckpts/ckpt_resnet50_r50aug20_ep18.pth --w-bits 4 --a-bits 8 --calib-n 512 --eval-n 10000 --device cuda --out-dir results/m2_accept
python -m ptq_toolkit build --model mobilenet_v2 --weights /mnt/d/WorkBuddy/ShowBro/paper-code/ckpts/ckpt_mobilenet_v2_mv2aug20_ep19.pth --w-bits 8 --a-bits 8 --calib-n 512 --eval-n 10000 --device cuda --out-dir results/m2_accept
python -m ptq_toolkit build --model mobilenet_v2 --weights /mnt/d/WorkBuddy/ShowBro/paper-code/ckpts/ckpt_mobilenet_v2_mv2aug20_ep19.pth --w-bits 4 --a-bits 8 --calib-n 512 --eval-n 10000 --device cuda --out-dir results/m2_accept
```

### ⚠️ 三个必须知道的口径

**1. 「量化后精度」是权重-only 的 fake-quant，不含激活量化。**
激活量化在 C++ 引擎侧才真正生效，所以真实 W8A8 引擎精度会**更低**
（对照：ResNet-18 权重-only 87.08% vs 06 引擎真实 W8A8 86.93%）。

**2. ResNet-50 / MobileNetV2 的随机初始化数字已作废**，上表已换成 2026-10-22 训练权重实测
（R50 91.85% / MV2 92.87%，全量 10k）。历史随机权重下的 ≈10% 数字仅用于验证
「量化 + 拓扑导出没把网络算坏」，不代表量化损失。

**3. 三个模型均为全量 10000 张评测**（2026-10-22 起）。CPU 全量单次约 10–12 分钟/模型（含校准与导出），GPU 更快；
早先「另两个只取前 2000 张」的口径已不再使用。

## 混合精度（I2 结论落地 · 2026-10-23）

**动机**：M2 的 W4A8 是全网 per-channel W4，R50 掉 3.22 pp / MV2 掉 5.22 pp，
都超 3 pp 预算。I2（`tools/layer_sensitivity.py`，CPU-only）用
「64 张粗排 53 层 → 512 张复核 12 层 → 512 张真实组合评测」选出敏感层，
推荐计划 **top4_all：4 层 W8 + 49 层 W4**（W8 = `conv1` /
`layer1.0.conv1` / `layer1.0.conv3` / `layer1.1.conv1`，
其余 W4 per-group 32），I2 的 512 张 in-sample 掉点为 **0.000 pp**。
证据：`results/i2/layer_sensitivity_resnet50.{csv,json}` + 完整日志。

### 导出 + 自检（Python 侧，不改 06 的 C++）

```bash
# 1) 按 I2 推荐计划导出一个混合精度包（只校准，不评测）
python -m ptq_toolkit export --model resnet50 \
  --weights /mnt/d/WorkBuddy/ShowBro/paper-code/ckpts/ckpt_resnet50_r50aug20_ep18.pth \
  --plan-json results/i2/layer_sensitivity_resnet50.json --plan-name top4_all \
  --calib-n 512 --device cpu --out-dir results/mixed_precision/resnet50_top4_all

# 2) 按磁盘字节回读自检（manifest / PTQ2 / PTQC / 逐层 bits 网格 / SHA256）
python -m ptq_toolkit mixed-check results/mixed_precision/resnet50_top4_all
```

- 包路径：`results/mixed_precision/resnet50_top4_all/`
  （`.ptq2.bin` 94.0 MB、`.ptqc` 432 B、`quant_config_*.json` 24 MB、
  `layer_plan_*.json`、`build_*.json`、`manifest.json`、`selfcheck.json`）。
  大文件按仓库纪律不入库（`.gitignore`），**manifest 里的 SHA256/字节数
  就是重建校验的锚点**；重跑上面两条命令即可恢复到逐字节一致（同机同权重）。
- 自检 **32 项全过**：PTQ2 完整解析 53 conv / 16 block / 终点=文件长度；
  PTQC 53 层 / 432 B / lo≤hi；`quant_config` 53 层顺序与 layer_plan 一致；
  每一层权重都落在声明的 W4/W8 网格上（码字范围合法）；
  5 个产物 SHA256 与 manifest 一致；拓扑自检 `max|diff| = 0`。
- 逐层配置：`quant_config_*.json` 的每个 `weights[name]` 带
  `bits/scheme/group_size/scale`；`layer_plan_*.json` 按引擎 conv 顺序
  列出 53 层（**不按语义重排**）。
- **PTQ2 是 fp32 容器**：W4 只通过 QDQ 改变数值分布，不改存储大小/算力；
  真正的 W4 存储压缩需要扩展格式（未做）。C++ 引擎侧（06）本轮**未改**。
- 跨语言抽查：06 的 CPU probe（`ptq_model_probe`）只读该包并 **rc=0**
  （53 conv / 16 block，PTQC n=53 / 432 B），证据
  `results/mixed_precision/resnet50_top4_all/probe_06_cpu.txt`。

### 掉点对比（同一模型 / 同一 test 前 N 张 / CPU 前向 / 仅权重量化）

| 配置 | 512 张 | 掉点 | 2000 张 | 掉点 | W8 层数 |
|---|---|---|---|---|---|
| fp32（R50 ep18） | 92.77% | — | 92.05% | — | — |
| 全 W4 per-channel（M2 口径） | 89.45% | 3.32 pp | — | — | 0 |
| 全 W4 per-group32（I2 口径） | 90.04% | 2.73 pp | — | — | 0 |
| **top4_all（4 W8 + 49 W4）** | **92.19%** | **0.59 pp** | **91.55%** | **0.50 pp** | **4** |
| 对照：structural 混合（6 W8 + 47 W4） | — | — | 91.50% | 0.55 pp | 6 |

> 512 张数字来自 `tests/test_mixed_precision.py`（`--skip-eval` 可跳过），
> 2000 张来自 `ptq eval`（`results/mixed_precision/resnet50_top4_all/
> eval_resnet50_n2000.json`）。两者都不是全量 10k；按纪律不重复跑 10k，
> 引用全量请用 M2 表（全 W4 per-channel）与此处的 top4_all 对照。
>
> ⚠️ 口径：这里只量化权重（fc fp32），**不含激活量化**；I2 的 0.0 pp 是
> **512 张校准子集 + A8 fake-quant** 的 in-sample 结果，两者不可直接相减。
> 06 引擎的真实 W4A8 精度仍需 GPU + C++ 侧支持，未验证。

### MobileNetV2 10ep 迁移权重（2026-10-23）

- 权重：`paper-code/ckpts/ckpt_mobilenet_v2_mv2img10_ep9.pth`
  （10 epoch / cos10 / ImageNet 移植；全量 10k test **92.56%**；
  SHA256 `62d72539…e74f3e`）。
- W8A8：92.20 → **92.10%**（2000 张，-0.10 pp，PASS）。
- 全 W4 per-channel：84.55%（-7.65 pp）；全 W4 per-group32：80.45%
  （-11.75 pp）——**单档 W4 不达标**，这正是 top4_all 要解决的问题。
- **top4_all（第一轮：4 W8 + 48 W4 per-group32）：90.05%（-2.15 pp，2000 张）**；
  512 张为 -1.95 pp，**超 1 pp 预算**，已由下面的 I3 计划取代。
  W8 层：`features.1.conv.0.0` / `features.1.conv.1` /
  `features.2.conv.2` / `features.3.conv.2`。
  包：`results/mixed_precision/mobilenet_v2_top4_all/`（52 conv / 18 block；
  PTQC 424 B；32/32 自检；max|diff| = 0）；复跑 `tests/test_mv2_mixed.py`。

#### 第二轮（I3，2026-10-25）：W4 per-channel + W8 保护 24 层

**为什么第一轮压不下去**（三条都能用 `tools/mv2_mixed_study.py` 复跑）：

1. **口径错位**：I2 的掉点是在**训练集前 512 张 in-sample + A8 fake-quant**
   上量的（其 fp32 = 97.85%），而 top4_all 的验收口径是**测试集前 512 张、
   只量化权重**（fp32 = 91.80%）。两套数字不能相减 —— I2 的 2.73 pp 与
   top4_all 的 1.95 pp 根本不是同一件事。
2. **单层敏感度在 MV2 上是噪声级**：52 层里只有 `features.1.conv.0.0`
   （2.15 pp）与 `features.1.conv.1`（1.37 pp @per-group32）超过 512 张的
   0.195 pp 分辨率，其余全在 ±0.6 pp 内。I2 的 64 张粗排分辨率 1.56 pp，
   用它排序等于随机选层 —— 所以「再加几层 W8」自然压不下去。
3. **per-group32 是 MV2 上最差的 W4 粒度，且与权重 MSE 反向**：全 W4 g32 的
   权重 rel-MSE 0.0095（比 per-channel 的 0.0164 低 42%），端到端却掉
   11.75 pp（per-channel 只掉 7.65 pp）。**权重 MSE 不能当 MV2 的精度代理**。

**选层规则**：层序 = 512 张上「全 W8、只把一层降到 W4」的边际**交叉熵**伤害
降序（512 张的 top-1 分辨率不够，CE 连续且单调，只用来排序）；k = 取最小 k，
使 **512 ≤1.0 pp、2000 ≤1.5 pp、单次全量 10k ≤1.5 pp** 三个口径同时成立。

| 计划 | W4 粒度 | W8 层 | 512 | 2000 | 全量 10k | 有效比特/权重 |
|---|---|---|---|---|---|---|
| fp32 基线 | — | — | 91.80% | 92.20% | 92.20% | 32 |
| 全 W4 per-channel（M2 口径） | per-channel | 0 | −5.86 | −7.65 | — | 4.25 |
| 全 W4 per-group16 | per-group16 | 0 | −3.91 | −6.50 | — | 6.05 |
| 全 W4 per-group32（I2 口径） | per-group32 | 0 | −9.77 | −11.75 | — | 5.08 |
| 全 W8 per-channel | per-channel | 52 | −0.20 | −0.10 | — | 8.25 |
| top4_all（第一轮） | per-group32 | 4 | −1.95 | −2.15 | — | 5.09 |
| 备用 `w4g16_w8top21` | per-group16 | 21 | −0.39 | −1.10 | −1.08 | 6.74 |
| **推荐 `w4pc_w8top24`** | **per-channel** | **24** | **−0.39** | **−0.75** | **−0.92** | **4.99** |

> 掉点单位 pp；「有效比特/权重」= (Σ bits×权重数 + 32×scale 数 + 8×zero 数)
> / 权重总数。这是唯一能横向比较 per-channel 与 per-group 的量：per-group16
> 的 scale 元数据本身就要 2 bit/权重（这套 MV2-CIFAR 的 K 普遍 ≤960，更贵的
> 组粒度更吃亏）。表里 512 列对齐的是同一 test 前缀，不是训练集。

**结论**：**1.95 → 0.39 pp @512、2.15 → 0.75 pp @2000、全量 10k 0.92 pp**，
同时有效比特比 top4_all 还低一点（4.99 vs 5.09）—— 不是拿体积换精度。
代价是 W8 层从 4 涨到 24：**MV2 在 W4 下很脆，1 pp 预算只能让 28/52 层停在
W4**，这是本轮实测的硬结论，也是 report 里保留的剩余风险。

**不对称量化测过但不采用**（`grid` 模式的 `*_asym` 档）：per-group32 asym
6.45 pp（比对称好 5.3 pp），但 per-channel asym 10.15 pp（比对称**差**
2.5 pp），收益方向不稳定；而且 06 引擎只实现对称 int8，进包要改格式。
数据在 `results/i3/w4_scheme_grid_n2000.json`。

**包**：`results/mixed_precision/mobilenet_v2_w4pc_w8top24/`（52 conv /
18 block、PTQC 424 B、W8 24 / W4 28、自检 32/32、拓扑 max|diff| = 0；
`eval_mobilenet_v2_n{512,2000}.json` 是 `ptq eval` 官方口径）。复跑四步：

```bash
python -m ptq_toolkit export --model mobilenet_v2 --weights <ckpt> --plan-json results/i3/plan_mobilenet_v2_w4pc_w8top24.json --plan-name w4pc_w8top24 --calib-n 512 --device cpu --out-dir results/mixed_precision/mobilenet_v2_w4pc_w8top24
python -m ptq_toolkit mixed-check results/mixed_precision/mobilenet_v2_w4pc_w8top24
python -m ptq_toolkit eval --model mobilenet_v2 --weights <ckpt> --plan-json results/i3/plan_mobilenet_v2_w4pc_w8top24.json --plan-name w4pc_w8top24 --eval-n 2000 --budget-pp 1.5 --device cpu --out-dir results/mixed_precision/mobilenet_v2_w4pc_w8top24
python tests/test_mv2_plan_i3.py
```

> `--w4-scheme` / `--w4-group` 现在默认跟随 `--plan-json` 里声明的粒度
> （本计划是 per-channel），不再静默退回 per-group32；per-channel 时
> group_size 归 0，否则 layer_plan 与 quant_config 会不一致。

**测试**：`python tests/test_mixed_precision.py`（R50：计划/覆盖/包自检/CLI +
全 W4 vs top4_all 掉点对比）与 `python tests/test_mv2_plan_i3.py`
（MV2：计划自洽/粒度跟随/包自检/512 张实测，断言 ≤1.0 pp；各约 1–2 分钟，CPU-only）。

### 拓扑格式：两种并存

| 格式 | 用在哪 | 表达能力 | 兼容性 |
|---|---|---|---|
| `PTQW` | ResNet-18 | 2 层 conv + 可选 downsample | **06 的引擎二进制可直接读** |
| `PTQ2` | ResNet-50 / MobileNetV2 | **N 层** conv + 可选 downsample + **可选残差** + **depthwise groups** | 新格式，需引擎侧支持 |

⭐ **为什么不用「通用指令序列」（PTQX）**：M2 中途设计过 PTQX
（模型声明残差 + 栈机指令 `CONV/ADD/SAVE/POOL/MEAN/FC`），
试图自动推断残差位置。实测迭代 20+ 轮仍不可靠：
ResNet-50 上自检 `max|diff| = 0.03`（relu 标志回填的边界难穷尽），
MobileNetV2 因「可选 skip」直接崩。

最终回退到**扩展已验证的 blocks 格式** —— M1 已用它与 06 的引擎
逐位对齐过（11,172,032 参数），推广到 N 层只多一个维度。
**能用已验证的机制扩展，就不要换新机制。**

### 验收（9 条脚本，全过）

| 脚本 | 检查 |
|---|---|
| `test_quant_ops.py` | 量化口径与 03 逐位一致（W8/W4 max\|diff\| 均为 0） |
| `test_models.py` | 三模型：前向形状 / 参数量（±5%）/ **拓扑自检 max\|diff\| ≈ 0** |
| `test_topology_semantics.py` | MV2 激活语义：project 无激活 / 残差相加后无激活 / ReLU6；结构断言 + ×10 输入数值自检 |
| `check_binary_compat.py` | ResNet-18 的 `.bin` 与 06 引擎格式一致（20 conv / 8 block / 11,172,032 参数） |
| `test_golden.py` | ResNet-18 外部对齐（fp32 差 0.00 pp）+ 三模型完整性 + 掉点阈值 + 区间基线 |
| `test_export_ptqc.py` | PTQC 格式往返 + PTQ2/PTQC 层数配对 |
| `test_w4_candidates.py` | per-group 逐组参考实现 / QDQ 网格 / 真实产物 / 06 CPU probe |
| `test_mixed_precision.py` | R50：I2 计划、per-layer 覆盖、包自检、全 W4 vs top4_all 掉点 |
| `test_mv2_plan_i3.py` | MV2：I3 计划自洽、W4 粒度跟随计划（per-channel）、包自检、512 张 ≤1.0 pp |

## 三条命令在做什么

### `ptq build`

```
加载 fp32 模型（03 的 load_fp32）
  → 测 fp32 基线精度
  → BN 折叠（03 的 fold_all_bn）—— 06/07 的所有测量都基于折叠权重
  → 再测一次，确认折叠无损
  → 激活校准（03 的 calibrate，**symmetric=True**）
  → per-channel 权重量化（自己的 ptq.quantize_per_channel_sym）
  → 就地 fake-quant，再测量化后精度
  → 导出 权重+拓扑（自己的 exporter，格式对齐 06 的 PTQW）
  → 导出 量化参数 JSON
```

**复用 03 的三件套**（`fold_all_bn` / `evaluate` / `calibrate`），不重造：
BN 折叠、校准、精度评测的口径必须与 03/06/07 一致，
重造一套会让精度数字无法互相比较。

### `ptq bench`

从 `06/07/08` 的 results 文件里**读**已有基准，写成 `results/bench.json`。

### `ptq report`

把 build + bench 汇总成 `results/report.md` 与 `results/report.json`。

---

## ⭐ 为什么不重新测 GPU

两个理由，第二个更重要：

1. GPU 让给用户（不占用游戏）。
2. **重测会引入「机器负载」这个变量，让 golden test 不可复现。**

   08-report 已经吃过这个亏：同一份代码在 idle 与占用两种状态下
   差距可达 **21 倍**（`idle_summary.txt` 里 util 1% 时 0.0274 ms/img，
   util 98~100% 时 0.5867）。在这种机器上测出来的速度数字，
   下次复现时对不上是必然的。

所以 M1 的 bench 是**引用**而不是**测量**。真正自测的基准留给 M3
（那时会有自己的引擎与统一的门控脚本）。

---

## 设计要点

### 量化口径三处（与 03 逐位一致）

| 项 | 值 | 写错的后果 |
|---|---|---|
| round | **half-away-from-zero** | numpy 默认是 banker's，差一个量化等级 |
| clamp | `[-(qmax+1), qmax]` | 负侧少一个码字 |
| scale 分母 | `qmax = 2^(bits-1)-1`（127 / 7） | 用 255 会让 scale 差 2 倍 |

`tests/test_quant_ops.py` 把这三条锁死，并**与 03 的实现真对照**
（不是重述规格）。

### 模型走 registry，pipeline 里没有 if/else

`--model` 的唯一入口是 `ptq_toolkit/models/registry.py`：
加模型 = 加一条 `ModelSpec`（build 函数 + topo_kind + 残差声明）。
pipeline / export / report 都不感知具体模型。

### 残差拓扑由**模型声明**，不是通用层猜

`topo_plan()` 写在每个模型文件里（`after_conv` / `convs` / `downsample`）。
这不是妥协 —— 网络结构本来就是拓扑信息，
让模型作者写清楚比让通用提取器猜可靠得多。

### 激活必须用对称码空间

03 的 `calibrate` 默认**非对称**（带 zero_point），而 06/07 的引擎只支持对称
（ncnn 的 `float2int8()` 把码字截在 ±127 且没有 zp；int8 tensor core
的 GEMM 也没有 zp 偏移）。所以 build 里**显式传 `symmetric=True`**。

用非对称参数喂对称引擎，码字会越过 127 溢出成负数，精度直接掉到随机猜水平
（06 的 `ptq_quant.h` 注释记了这个坑）。

### 拓扑必须显式导出

ResNet-18 不是纯串联。C++ 侧若按 conv 索引顺序串联，**residual 分支会被整个丢掉** ——
实测症状是单层校验全绿（max|diff| = 9.5e-7）而全网精度 7.6%。

所以导出里带 relu 标志（谁在 add 之前）与 block 结构（c1/c2/ds/stride），
并在导出时**用同一份拓扑复算一遍 logits 做自检**。

### 二进制格式与 06 一致

`magic "PTQW"`，字段顺序与 `06-gpu-int8/cpp/export_weights.py` 逐字节相同，
所以 09 导出的文件能直接喂给 06 的引擎二进制。格式各自演化的话，
golden test 就失去意义。

---

## 目录

```
09-ptq-toolkit/
├── ptq_toolkit/
│   ├── cli.py              ptq build/export/eval/ptqc/mixed-check/bench/report
│   ├── pipeline.py         主流程（复用 03 的 fold/eval/calibrate）
│   ├── models/                  模型定义 + registry
│   │   ├── registry.py          --model 的唯一入口
│   │   ├── resnet50.py          CIFAR 适配版 + topo_plan
│   │   ├── mobilenet_v2.py      t=6（torchvision 对齐），CIFAR 适配版
│   │   └── resnet18_plan.py     ResNet-18 的残差声明
│   ├── quant/
│   │   ├── ptq.py               per-channel / per-group PTQ 核心
│   │   └── mixed.py             W4 候选 + 显式 per-layer 覆盖（I2 计划）
│   ├── export/
│   │   ├── exporter.py          PTQW 权重 + 拓扑导出
│   │   ├── topoblocks.py        PTQ2（N 层 conv / 残差 / depthwise）
│   │   ├── ptqc.py              PTQC 校准文件（与 06 load_calib 同布局）
│   │   ├── ptq2_info.py         PTQ2 结构化回读
│   │   └── mixed_check.py       混合精度包磁盘自检（哈希/层数/网格）
│   ├── bench/collect.py    读 06/07 已有基准
│   └── report/render.py    Markdown / JSON 报告
├── tests/
│   ├── test_quant_ops.py        量化口径自测（与 03 逐位对照）
│   ├── test_models.py           三模型：前向/参数/拓扑自检
│   ├── test_topology_semantics.py 激活语义回归（结构断言 + ×10 输入）
│   ├── test_load_transfer.py    权重移植五条自检的**负向**测试（证明闸门真会拦）
│   ├── test_export_ptqc.py      PTQC 格式往返 + 层数配对
│   ├── test_w4_candidates.py    W4 候选 / QDQ 网格 / 06 probe 跨语言
│   ├── test_mixed_precision.py  R50：I2 计划 + 包自检 + 全 W4 vs top4_all
│   ├── test_mv2_plan_i3.py      MV2：I3 计划 + 粒度跟随 + 包自检 + 512 张 ≤1.0pp
│   ├── check_binary_compat.py   ResNet-18 的 .bin 与 06 引擎格式兼容
│   └── test_golden.py           外部对齐 + 三模型回归基线
├── tools/
│   ├── load_transfer.py         torchvision ImageNet → CIFAR-10 迁移初始化
│   ├── finetune.py              CIFAR 全量微调（--weights imagenet / 续跑闸门）
│   ├── layer_sensitivity.py     I2 逐层敏感度（CPU-only，64 粗排 + 512 复核）
│   └── mv2_mixed_study.py       I3：W4 粒度网格 / 边际伤害 / 累进曲线 / 计划生成
├── weights_manifest.json   权重出处（URL / SHA256 / 大小 / 取用时间），入库
├── results/               产物（build_*.json / *.bin / bench.json / report.md /
│                           m2_accept/ / i2/ / i3/ / w4_candidates/ / mixed_precision/）
├── run_all.sh             一键跑通
└── README.md
```

---

## 已知限制

- **ResNet-50 / MobileNetV2 已换成 20ep 训练权重**（2026-10-22），
  root 里旧的「随机初始化 ≈10%」build json 只保留作历史，不再引用。
  M2 验收数字在 `results/m2_accept/`，入口见上表。
- **三个模型均为全量 10000 张评测**（2026-10-22 起；早先 R50/MV2 只取前 2000 张）。
- **PTQ2 格式需要 C++ 引擎侧支持**（PTQW 是 06 现成的）。
- **MV2 的 W4 余量很薄**：要 ≤1 pp 只能让 28/52 层停在 W4（24 层 W8），
  有效比特 4.99（全 W8 是 8.25）。想要更激进的 W4 比例，必须先解决
  per-group 在 MV2 上的异常掉点（见 I3 证据）。
- 依赖 `../03-ptq-lab` 的模型定义与评测口径（BN 折叠 / 校准 / evaluate）。
- **量化后精度是权重-only 的**，不含激活量化。真实 W8A8 引擎精度更低
  （06 的引擎 W8A8 是 86.93%，本工具的权重-only 是 87.08%）。
  这是**预期的**——激活量化在引擎侧生效。
- **bench 是引用不是测量**（见上）。
- 导出的是 fp32 权重 + 量化参数，**没有导出 int8 码字**。
  C++ 引擎侧要做「加载 → 反量化 → 前向」；
  要直接喂 int8 权重给 tensor core，需要 M2/M3 补上。

---

## M2 已完成 / 待办

### ✅ 已完成（三模型 × W8/W4 端到端跑通，四条验收全过）

- 三个模型的 CIFAR 适配版 + registry（`--model` 唯一入口，pipeline 无 if/else）
- 两种拓扑格式并存：`PTQW`（ResNet-18，06 引擎兼容）/ `PTQ2`（N 层 conv + 可选残差 + depthwise groups）
- W8A8 与 W4A8 全链路：折叠 → 校准 → per-channel 量化 → 导出 → 精度
- 八条验收：`test_quant_ops` / `test_models` / `test_topology_semantics` / `check_binary_compat` / `test_golden` / `test_export_ptqc` / `test_w4_candidates` / `test_mixed_precision`
- 报告数字可溯源到 `results/build_<model>_w<bits>.json`

**2026-10-22 训练权重验收**：R50/MV2 已用 20ep 训练权重全量实测，W8 PASS、W4 FAIL（3.22 / 5.22 pp）。
导出侧两个 bug 已修（2026-10-22 10:07）：CUDA 设备放置（`torch.from_numpy` 造 CPU 权重喂 CUDA 输入）
与 MV2 的 PTQ2 激活语义（project 无激活、残差相加后无激活、ReLU6）。修复后 4 次 CUDA build 全部自检 0、
4/4 产物落盘；权重-only 精度与修复前一致。回归见 `tests/test_topology_semantics.py`。

## S1：ResNet-50 / MobileNetV2 的可用权重（已完成，2026-10-22）

S1 给两个新模型一个有出处的起点：**torchvision ImageNet 权重移植 + CIFAR 全量微调**。
最终可引用权重：R50 `r50aug20/ep18` 91.85%、MV2 `mv2aug20/ep19` 92.87%（均全量 10k）。

### 权重移植（已完成）

```bash
cd /mnt/d/WorkBuddy/ShowBro/lab/09-ptq-toolkit
source ~/pyenv-torch/bin/activate

# ResNet-50：conv1 7×7→3×3 中心裁剪（offset=2，不缩放）+ fc 重置 10 类
python tools/load_transfer.py --arch resnet50

# MobileNetV2：stem 本就是 3×3，只把 stride 2→1；classifier.1 重置
python tools/load_transfer.py --arch mobilenet_v2
```

五条自检任一失败即非零退出：A 键集合只允许分类头差异 ·
B 非头张量逐元素 bit 相同 · C 32×32 前向 finite · D 分类头形状与重置统计 ·
E 与 torchvision 原生实现同权重同输入对拍。负向测试见
`tests/test_load_transfer.py`（改非 fc key / 加 ghost key / 扰动 bn1 均 exit=1）。

**自检 B 已改成「独立回读」**：原先是拿 `load_state_dict` 刚灌进去的那个
dict 来比 —— 等于在证明「`load_state_dict` 把张量拷进去了」，自证循环。
现在 B 从磁盘 `.pth` **重新 `torch.load` 一份**再逐张量比对
（`bn_reestimate.py` 里的 `disk_conv1_bit_identical` 就是这个机制）。

配套 `tests/test_load_transfer.py` 第 6 项的**随机反例**：把
`layer1.0.conv1.weight` 替换成 `torch.randn_like` 噪声（键集合仍然干净、
形状也对得上），此时 B **必须** `SystemExit(1)`。
若这条反例是绿的但正常情况是红的，就说明比对又退化成了自证。

| arch | missing | unexpected | B 逐元素 bit | E 对拍 max\|diff\| |
|---|---|---|---|---|
| resnet50 | `fc.weight/bias` | `[]` | 318/318 | 0（2048 维特征） |
| mobilenet_v2 | `classifier.1.weight/bias` | `[]` | 312/312 | 0（1280 维特征） |

**MobileNetV2 为接权重把定义从 t=1 重写成了标准 t=6**（torchvision 原值，
第一项 `[1,16,1,1]` 没有 expand conv），参数 0.18M → 2.24M。
代价换来的是键集合与 torchvision 逐字一致 —— t=1 版根本搬不过去。

### ⛔ 一个「键名匹配但语义不同」的陷阱（实测，差点误用）

`paper-code/ckpts/cifar10_mobilenetv2_50ep.pth`（2.271M 参数、314 键、
`features.0.0.weight` 开头、`classifier.1` 头）加载到本仓库的 t=6 定义上，
`missing=[] unexpected=[]` **看起来完美匹配**，但 fp32 top-1 只有 **38.90%**，
而它自己的 history 记 best = **83.58%** —— 差 44.68 pp。

根因：那份权重来自 `BRECQ_yhhhli/models/mobilenetv2.py`，它的 `InvertedResidual`
把 conv 与 BN **合并成了带 BN 的 Conv2d**，所以 `features.N.conv.3.weight`
在 BRECQ 里是 **depthwise 权重** `(C,1,3,3)`，而 torchvision/本仓库里
`features.N.conv.3.weight` 是 **BatchNorm 的 weight** `(C,)`。

**键名相同、张量语义完全不同。** `load_state_dict(strict=False)` 只校验键名与形状，
形状恰好都能塞进去（BN 的 (C,) 与 dw 的 (C,1,3,3) 在逐层比对时才暴露），
于是拿到一份「能加载但算错」的权重。

**处置：弃用该 pth。** MobileNetV2 与 ResNet-50 走同一条路 ——
torchvision ImageNet 权重移植 + CIFAR 微调。
教训写进 `load_transfer.py` 的自检 E：**只比对键集合和数值不够，
必须与一个已知正确的实现对拍前向**（E 就是干这个的）。

### CIFAR 微调（ResNet-50 / MobileNetV2 均已完成 20ep）

```bash
cd /mnt/d/WorkBuddy/ShowBro/lab/09-ptq-toolkit

# 一键脚本（nohup + --wait；协议：全量 50k / batch64 / lr0.01 cosine /
# wd 5e-4 / label_smoothing 0.1 / RandomCrop+HFlip / ImageNet 移植）
ARCH=mobilenet_v2 TAG=mv2aug20 EPOCHS=20 WEIGHTS=imagenet \
  LOG=results/finetune_mv2_full.log bash run_finetune_full.sh --wait

ARCH=resnet50 TAG=r50aug20 EPOCHS=20 WEIGHTS=imagenet \
  LOG=results/finetune_r50_aug20.log bash run_finetune_full.sh --wait
```

等价的直接命令（**MobileNetV2 实际使用的就是这条**）：

```bash
PYTHONPATH=. nice -n 19 /home/user/pyenv-torch/bin/python -u tools/finetune.py \
  --arch mobilenet_v2 --run-tag mv2aug20 --epochs 20 \
  --batch-size 64 --lr 0.01 --weight-decay 5e-4 --label-smoothing 0.1 \
  --weights imagenet --device cuda --num-workers 2 \
  > results/finetune_mv2_full.log 2>&1
```

实测（CPU 结论不适用；这是 GPU 训练，训练前后 `nvidia-smi` 留档在 `results/`）：
MV2 20ep 每 epoch 31–34 s、峰值 **429.7 MiB**，总耗时约 11 min；
test 78.38%（ep1）→ 92.87%（ep19，best）→ 92.85%（ep20）。
R50 20ep best ep18 = **91.85%**（每 epoch 约 4.5 min，峰值 1893.7 MiB）。

> `--weights imagenet` 是正式协议：整份非分类头 state_dict 逐张量加载并回读校验。
> 早先实现只搬 conv1，其余仍是随机初始化（等于 from scratch）——
> 该 bug 已修（commit `5f0b93d`），`--weights scratch` 只留给消融/单测。
> 两个训练不能同时占 GPU，否则耗时与显存数字不可用（`08-report` 实测负载影响）。

产物在 `paper-code/ckpts/`：`ckpt_<arch>[_<tag>]_ep<N>.pth`（每 epoch 一个）+
`history_<arch>[_<tag>].json`（逐 epoch 的 train/test acc、耗时、显存峰值）+
`SHA256SUMS`。冒烟判据写在脚本末尾：出现 NaN → exit 2，显存 > 3400 MiB → exit 3。

#### 历史：ResNet-50 无增强第一段（ep17–26，已结束）

> 这一段只作为过程记录：之后按正式协议（增强 + ls0.1 + wd5e-4）跑了
> `r50aug20` 20ep，best **ep18 91.85%**，才是当前引用值。
> 下表 84.65% 不要与 91.85% 混用（协议不同，不可比）。

数据源 `paper-code/ckpts/history_resnet50.json`，`protocol=cifar10-full-50k` ·
`n_train=50000` · `conv1=A` · `lr=0.01` · `batch=64` · 峰值 1893.7 MiB ·
无 NaN · 无 OOM 降级。检查点链由 `tools/audit_ckpts.py` 独立回读
（`results/ckpt_chain_audit.log`）：

| epoch | test |
|---|---|
| 17 | 57.18% |
| 18 | 71.18% |
| 19 | 76.24% |
| 20 | 78.41% |
| 21 | 80.18% |
| 22 | 81.15% |
| 23 | 82.85% |
| 24 | 84.19% |
| **25** | **84.65%** ← 该段最佳 |
| 26 | 84.52% |

- 这是**一段连续的 10 epoch**（fc 权重余弦 17→18..25→26 均 ≥0.977），
  该段 best = ep25 **84.65%**、检查点 `ckpt_resnet50_ep25.pth`
  （已被 `r50aug20/ep18 91.85%` 取代，仅作过程记录）。
- 独立复评（全量 10000 张，GPU）：
  `python tools/eval_ckpt.py --ckpt .../ckpt_resnet50_ep25.pth --arch resnet50
  --limit 0 --device cuda` → `n=10000 acc=84.65 delta=+0.00pp`
  （`results/eval_ep25_full10k.log`）。
- ep12–16 是另一段连续 5 epoch（best 82.32%），ep11 是单 epoch 段；
  **不能合并称「训了 22 epoch」**。
- ep27–31（aug 协议）全部作废：ep27、ep29 都是重新初始化，
  各自只跑了 2–3 epoch，best 66.51%，与 84.65% 不可比。

> ✅ **续跑缺陷已修复（2026-10-21）。** `finetune.py` 现在会加载
> 权重 + 优化器 + 调度器，并校验回读 `max|Δ| == 0`；跨协议、缺 ckpt、
> 目标/超参不一致都会拒绝。`--epochs` 现表示「目标总 epoch 数」。
> 旧 ckpt（无 `target_epoch`）不能严格续跑；继续训练请用
> `--init-from <pth> --run-tag <新段>`，结果里必须写明是第二段。
> 验证：`results/resume_fix_test.log`（CPU 微型模型 5 项，约 7 s）。
> 审计：`python tools/audit_ckpts.py --dir <ckpt_dir> --arch resnet50`。

### BN running stats 失配的范围与处置（已决策，不留给微调）

**失配范围不止 bn1**：stem 从 7×7s2 改成 3×3s1 且去掉 maxpool，
**每个 stage 的特征图几何都变了** —— layer1..4 的**全部** BN stats
都是在 224×224/7×7s2 上统计的，与 32×32/3×3s1 都不匹配。
MobileNetV2 同理（stride 2→1 影响 features.1..17 的所有 BN）。

实测三种处置（GPU，5120 张重估，只动 bn1 一层）：

| 处置 | fp32 top-1 | 相对保留 |
|---|---|---|
| **① 保留原 stats** | **16.56%** | — |
| ② 用 CIFAR 重估 bn1 | 12.03% | −4.53 pp |
| ③ 重置为 mean=0/var=1 | 9.33% | −7.23 pp |

**决策：保留原值（①），不做局部重估、不做局部重置。**

三条理由：
1. **三者都在随机猜附近**（9~17%）—— 说明 BN stats 根本不是当前的瓶颈。
   真正的瓶颈是 conv1 中心裁剪丢了 59% 的 L1 质量 + fc 随机重置。
2. **只重估 bn1 一层是治标不治本**。失配是全网性的，单层重估会让 bn1 适应
   新分布而其后的层仍按旧几何统计 —— 制造一个「半新半旧」的不一致状态，
   比重估前更难推理。上面 ② 掉 4.53 pp 正是这个原因。
3. **要重估就该全网一起重估**，而那本质上是「再跑一次前向统计」，
   与直接微调的代价相同（微调还会同时修正 conv1 裁剪的损失），
   所以没有必要单做一步。

工具：`tools/bn_reestimate.py`（`--mode compare/reestimate/reset`，
默认作用于 bn1；要全网重估把 `bn1_of()` 换成遍历所有 BN）。

#### 九组对照：conv1 A/B/C × BN keep/reestimate/reset（`bn_scope=all`，全量 10000 张）

上一张表是「只动 bn1」的口径；这一版按失配的真实范围改成
`bn_scope=all`（重估/重置作用于**全部** BatchNorm2d），且评测用**全量 10000 张**：

| conv1 | 实现 | L1 保留 | keep | reestimate | reset |
|---|---|---|---|---|---|
| A | `center_crop_noscale` | 0.412 | 10.00% | 10.00% | 10.06% |
| B | `center_crop_area_scaled` | 2.241 | 10.90% | 10.00% | 10.00% |
| C | `kernel_sum_pool` | 0.531 | 9.61% | 10.00% | 10.39% |

原始产物：`results/bn_conv1_compare.md` / `.json`。

**结论 —— 九组全部落在 chance 附近（9.61~10.90%）：**

1. **不微调就全都是 10%。** CIFAR-10 十类随机猜正是 10%。这说明
   conv1 怎么移植、BN stats 怎么处置，**在微调之前都不是瓶颈**。
2. **微调是必需的，不是可选项。** BN 三种处置之间最多 ±0.4 pp 的抖动，
   而微调把 57.18% → 81.15%（同一 protocol）。不要再往 BN 处置上投时间。
3. **conv1 选 A（中心裁剪，不缩放）。** B 名义最高 10.90%、A 是 10.00%，
   但在 chance 附近这点差不具可解释性 —— 10 类 × 10000 张，一个标准误约
   0.3 pp，且九组里 keep/reestimate/reset 的排序在三种 conv1 下并不一致
   （A 是 reset 最高，B 是 keep 最高），说明这就是噪声。
   A 是唯一 `equals_center_crop=True`、不引入人为缩放的模式，
   L1 保留 0.412 与 C 的 0.531 同量级。选 B/C 等于拿噪声当依据。

> ⚠️ **两张表不是同一实验，数字不可直接相减。**
> 上一张：5120 张重估 + **只动 bn1**（保留原 stats 16.56%）；
> 本张：全网 `scope=all` + 评测 10000 张（A/keep 10.00%）。
> 评测样本数也不同。上一张保留在此，是因为它记录了「单层重估会更差」
> 这个发现；**引用结论请以本张为准**。

### ⏳ 待办（按优先级）

| # | 事项 | 说明 |
|---|---|---|
| 1 | ~~训练 ResNet-50 / MobileNetV2~~ | **已完成（2026-10-22）**：R50 20ep best 91.85%、MV2 20ep best 92.87%，均为全量 10k + ImageNet 移植 + 正式增强协议。训完的权重已接入 `ptq build`；M2 表见上。 |
| 2 | **C++ 引擎支持 PTQ2** | `PTQW` 已被 06 的引擎验证；`PTQ2`（N 层 conv / 可选残差 / depthwise groups）需要引擎侧加解析代码，**否则 ResNet-50/MobileNetV2 的导出文件没有消费者** |
| 3 | **导出 int8 码字** | 当前导出 fp32 权重 + 量化参数，引擎侧要做「加载 → 反量化 → 前向」。要直接喂 tensor core 需导出 `(K, N)` 布局的 int8 |
| 4 | **`--w-bits` 与 `--a-bits` 解耦** | 现在 `a_bits` 实际只影响 clamp 区间，激活仍是 fp32 fake-quant。要真实 W4A8 需在 PyTorch 侧也做激活量化 |
| 5 | **校准集敏感性扫描** | M2 用了 `--calib-n 128`；量化语义对校准量通常不敏感，但没实测过 |
| 6 | **GPU 基准** | 全程 device=cpu。要在 GPU 上测需等 GPU 可用时，且必须加门控（08-report 实测占用与空闲差 21 倍） |

### 🔴 引用这两个模型的精度前必须知道

```
ResNet-50 / MobileNetV2 现在引用的是 2026-10-22 的 20ep 训练权重：
  resnet50     fp32 91.85%  W8 91.85%（0.00 pp）  W4 88.63%（3.22 pp）
  mobilenet_v2 fp32 92.87%  W8 92.91%（-0.04 pp） W4 87.65%（5.22 pp）
root 下 build_<model>_w*.json 里 6.25 / 10.65 / 9.80 / 10.80% 是
**训练权重接入前**的随机初始化旧产物，只作历史，不要再引用。
「量化后」仍是权重-only fake-quant，不含激活量化；真实引擎 W8A8 更低。
```
