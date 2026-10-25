# push-20261025 / repro —— 独立复现 int8 c1x1（含 fp32 / fp16 同口径补测）

目的只有一个：**07 原轮的 0.00744 ms/img / 87.22% 必须能被别人从 ONNX 重跑出来**，
否则它只是「某次跑出来的数」，不能当交付数字。

- 复现脚本：**repro_int8.sh**（int8 主线）、**repro_fp32_fp16.sh**（补主表另外两行）
- 对账脚本：**compare_repro.py**（判据写死在文件里，不许事后看数字改标准）
- 抖动探针：**fp16_variance.sh** / **int8_variance.sh** / **int8_ab.sh** / **clock_trace.sh** / **env_check.py**
- 原始日志：**repro_int8.txt**、**repro_fp32_fp16.txt**、**fp16_variance.txt**、**int8_variance.txt**、**int8_ab.txt**、**clock_trace.txt**（根 .gitignore 忽略 *.log，所以用 .txt）
- 机读结果：**bench.jsonl** / **bench_fp16v.jsonl** / **bench_int8v.jsonl** / **bench_ab.jsonl** / **bench_drift.jsonl**

## 一、怎么跑（一条命令）

```
cd /mnt/d/WorkBuddy/ShowBro/lab/07-tensorrt-deploy
bash results/push-20261025/repro/repro_int8.sh        # int8：导出→折叠→量化→冷构建→测→对账
bash results/push-20261025/repro/repro_fp32_fp16.sh   # 主表另外两行
```

脚本内部就是 07 的 **scripts/push_\***，没有另写一套测量代码：

```
export_onnx.py fp32                    → repro/work/resnet18_cifar10_fp32.onnx
push_optimize_onnx.py --verify-ort     → repro/resnet18_cifar10_fp32_c1x1.onnx
push_quantize_variant.py               → repro/resnet18_cifar10_int8_c1x1.onnx
push_one.sh（= 等空闲 + flock -x /tmp/ptq-gpu.lock + push_bench.py）
```

口径与原轮逐项一致：**batch=512 opt=3 tactic=default workspace=1GB static skip=1000 preboost=100**；
精度在同一次 dataset pass 上算（9000 张，跳过前 1000 张校准集）；再单独报 steady。

**与 07 原轮唯一的差别**：所有 ONNX 在 **repro/** 下重新生成、**repro/engines/** 是空目录
→ 一定走冷构建，不可能悄悄复用 07 的 engine 缓存。

## 二、结果与对账

原轮基准（**results/push-20261025/bench.jsonl** 的 **int8_c1x1_pb**）：
**dataset 0.00744 / steady 0.00790 / 精度 87.22%**。

| 复现档 | dataset ms/img | Δ | steady ms/img | Δ | 精度 | Δ | 冷构建 | 判定 |
|---|---|---|---|---|---|---|---|---|
| **repro_int8_l2000**（2000 张） | 0.00731 | −1.75% | 0.00780 | −1.27% | 86.80% | −0.42 pp | 247 s | PASS |
| **repro_int8_full9000**（9000 张） | **0.00725** | **−2.55%** | **0.00781** | **−1.14%** | **87.22%** | **+0.00 pp** | 复用 | PASS |

判据：**|Δlatency| ≤ 3%**（dataset 与 steady 各自）、**|Δacc| ≤ 0.5 pp**。
**两条都过**，而且复现值比原轮略快（−2.55%），也就是说 07 报的 0.00744 是**偏保守**的。

换算成倍数（基线 ncnn CPU fp32 = 11.6 ms/img）：
**11.6 / 0.00725 = 1600×**（dataset 中位）/ **11.6 / 0.00781 = 1485×**（steady）。

### 2.1 图是逐字节可复现的，只有 engine 不保证

| 文件 | 本轮 sha256 | 07 原轮 | 一致？ |
|---|---|---|---|
| **resnet18_cifar10_fp32_c1x1.onnx** | **bc8ef4e6…** | **bc8ef4e6…** | ✅ 逐字节相同 |
| **resnet18_cifar10_int8_c1x1.onnx** | **cf421f5c…** | **cf421f5c…** | ✅ 逐字节相同 |
| **resnet18_cifar10_fp16_c1x1.onnx** | **dbd1205e…** | **dbd1205e…** | ✅ 逐字节相同 |
| int8 engine（**…_78b88e1614.trt**） | **b463c574…** | **f78b02ec…**（原轮） | ❌ 不同，且**不要求**相同 |

三张 ONNX 的哈希与原轮完全一致，说明「导出 → 3×3 折 1×1 → 1000 张 Entropy 校准 → QDQ」
这条链是确定性的，包括 **strip_int32_bias_dq** 的每一步；int8 图大小也一致（4991236 B）。

**engine 不同是预期的**：TRT 的 tactic 搜索不保证逐字节可复现。本轮 int8 冷构建 247 s，
原轮记录了 679 s —— 构建时间差 2.7 倍，选到的 tactic 不完全一样。
所以对账看的是**数字**，不是 engine 哈希；图哈希一致才是「语义没走样」的硬证据。

### 2.2 精度不是「对齐出来的」

- 9000 张（与原轮同一批图、同样的 skip 1000）：**87.22%，与原轮完全相同**。
- 2000 张档 86.80%：样本量只有 1/4.5，标准误约 ±0.75 pp，差 −0.42 pp 属正常抽样波动，
  **不是精度退化**；同一个 engine 换回 9000 张就回到 87.22%。

### 2.3 逐层 profile 也对得上

原轮 README §6 说「折叠后 int8 最大剩余热点是 **/layer4/layer4.0/conv1**（2×2→1×1、stride 2
的 3×3），0.86 ms / 22%」。本轮 profile（9000 张档，profiler 有额外开销，只看占比）：

| 层 | ms/batch | 占比 |
|---|---|---|
| **/layer4/layer4.0/conv1**（2×2→1×1, s2） | 0.844 | **21.2%** |
| **/conv1/conv1** | 0.345 | 8.7% |
| **/layer1/layer1.0/conv2**、**layer1.1/conv2** | 0.248 / 0.247 | 6.2% / 6.2% |
| **input_QuantizeLinear** | 0.099 | 2.5% |
| 合计（profiler 口径） | 3.983 | 100% |

21.2% ≈ 原轮报的 22%，热点位置一致。

### 2.4 主表另外两行：fp32 与 fp16 也从 ONNX 重建

| 配置 | dataset ms/img | steady ms/img | 精度 | 冷构建 | 说明 |
|---|---|---|---|---|---|
| fp32（1×1 折叠，opt3） | **0.02477** | 0.02743 | 87.16% | 16 s | 468× / steady 423× |
| fp16（1×1 折叠，opt5+DIRECT_IO） | 0.01068–0.01112 | 0.01222–0.01256 | 87.18% | 80–105 s | 1043–1086×，见 2.5 |

fp32 是本轮唯一「新测」的行（07 原轮的 0.0276 是**未折叠**的 base 图）。
fp16 则出现了构建抖动，单独写在下节。

### 2.5 构建抖动：fp16 三次重建里有 1 次选到差 tactic

同一张 fp16 图（**dbd1205e**，逐字节相同）、同一配置（opt5 + DIRECT_IO），共 4 个 engine：

| 构建 | 冷构建 | dataset 中位 | steady | 相对原轮 |
|---|---|---|---|---|
| 07 原轮（**…_d0ebc6f4f8**） | 76.6 s | 0.01068 | 0.01222 | — |
| 复现 A（**…_1c2d4767fb**） | 105 s | 0.01261（首测 0.01382） | 0.01375–0.01415 | **+18~29% / +12~16%** |
| 复现 B（**…_fp16_rebuild_1**） | 100 s | 0.01108 | 0.01256 | +3.7% / +2.8% |
| 复现 C（**…_fp16_rebuild_2**） | 80 s | 0.01112 | 0.01253 | +4.1% / +2.5% |

**结论**：fp16 这条线 3 个好构建落在 0.0107–0.0111，1 个差构建落在 0.0126–0.0138
（差 12–29%），构建耗时看不出好坏（差的 105 s，好的 80–100 s）。
int8 没有这个问题：三个独立构建（679 s / 247 s / 241 s）都在 0.00725–0.00744（±2.6%）。

> 顺带排除了一个「能用 cache 修」的可能：**TRT 11.4 的 python 绑定**
> （**timing_cache_probe.py**）确实没有 **deserialize_timing_cache**，
> **Runtime** 里带「iming」的属性为空 —— 所以复现脚本无法靠固定 tactic cache 消除这个抖动，
> 只能靠「多次构建 + 多次测量取中位」验收。

### 2.6 时段抖动：同一份 engine 在「冷机/热机」窗口能差 15–26%

这一条比构建抖动更麻烦，也是最容易被忽略的：**同一个 engine 文件，隔两分钟再测就不一样**。

| 时间窗 | 被测 engine | dataset | steady | 负载下的 SM clock / 温度 |
|---|---|---|---|---|
| 15:42（刚空闲了几分钟） | 07 原轮 int8 **f78b02ec** | **0.00734** | 0.00787 | ~1905 MHz / 60–70 ℃ |
| 15:43（一分钟后，同场 A/B） | **同一份 f78b02ec** | 0.0084–0.0093 | 0.0088–0.0091 | **1710–1770 MHz** / 76 ℃、50.4 W |
| 15:44（同窗口，**clock_trace.sh**） | **同一份 f78b02ec** | 0.0086–0.0091 | 0.00912 | 负载中位 **1770 MHz**（min 1710）/ 76 ℃ |

**同一份 engine 在一分钟内从 0.00734 翻到 0.0093（+26%）**，只要换到同场的另一个 engine
（重建的 78b88e16）也一样慢（0.0087–0.0092）—— 所以不是 engine 的问题，是**这个窗口**的问题。

**clock_trace.csv** 是决定性证据：慢窗口里 SM clock 在负载下中位 **1770 MHz**（min 1710），
功率峰值 **50.4 W** 正好是这颗 1650 Ti 的 **50 W 上限**（**nvidia-smi -q** 的
**Current Power Limit: 50.00 W**），温度 76 ℃。快窗口则是 1905 MHz / 69 ℃。

机制最可能是**50 W 功耗墙 + 温度**（preboost 顶 clock 的同时也在加热），
但在这台 WSL 上不能完全排除外部争用 —— 无论哪种，**做法一样**：
绝对数字必须同场 A/B，不能跨窗口比。

**所以验收纪律是**（已写进脚本）：

1. **同场 A/B**：每次复现都顺手把 07 原轮 engine（**--engine-path**）在同一个窗口里重测一遍，
   用「相对原轮的比值」判断，而不是拿绝对数字跨窗口比。
2. **至少 3 次取中位**，单次数字不采信（**int8_noise.sh**）。
3. **记录 clock / 温度 / 功率**（**push_bench.py** 本来每次就打印）。

## 三、复现的边界（不要过度解读）

1. **engine 二进制不复现**，只复现数字（±3%，且要在同一温度/clock 窗口内比较）。
2. **同一份 engine 也会因热状态差 15–26%**（§2.6）——这台笔记本不是恒频平台。
3. **这不是「同一进程内的第二次测量」**：独立进程、冷构建、从原始 ONNX 开始。
4. **不做任何调参**：没有为了凑 0.00744 去改 batch / 换 opt level / 改测量次数。
   复现值偏快 2.55% 也照实写，没有往回找。
5. **不覆盖 07 既有结果**：**bench.jsonl**、**results/push-20261025/engines/**、**final.txt**
   一律没动，本轮只往 **repro/** 里写。
6. fp16 的 1086× 只在「好构建 + 冷机窗口」成立，差构建/热机会掉到 839–920×（§2.5、§2.6）。
   **int8 的 1559× 是稳的那条**（3 个构建 0.00725–0.00744）。

## 四、文件

| 文件 | 内容 |
|---|---|
| **repro_int8.sh** / **repro_fp32_fp16.sh** | 复现入口（调 07 的 push_* 脚本） |
| **compare_repro.py** | 对账脚本，判据写死（±3% / ±0.5 pp），不通过返回非 0 |
| **fp16_variance.sh** / **int8_variance.sh** | 构建抖动探针（同一张图重建多次） |
| **int8_noise.sh** | 测量噪声探针（同一 engine 连测 3 次，结果进 bench_int8v.jsonl） |
| **int8_ab.sh** | 同场 A/B：07 原轮 engine vs 复现 engine 交替测 |
| **clock_trace.sh** / **clock_trace.csv** | 边跑边采 clock / 温度 / 功率（0.25 s 一次） |
| **env_check.py** | 与 TRT 无关的绝对算力基准（mm / _int_mm / memcpy） |
| **timing_cache_probe.py** | 确认 TRT 11.4 python 绑定读不回 timing cache |
| **repro_int8.txt** 等 6 份 .txt | 完整原始日志（env、每轮 dataset、clock、profile、JSON 行） |
| **bench*.jsonl** | 每档一行 JSON（tag 前缀区分：repro_ / fp16_ / int8_ / ab_ / clock_） |
| **sha256*.txt** | 图、engine、脚本哈希 + git HEAD |
| **env.txt** | 环境快照（torch / TRT / onnx / ORT 版本、GPU、HEAD） |
| **work/** **engines/** | 中间产物（.gitignore 忽略：*.onnx / *.trt / *.cache） |
