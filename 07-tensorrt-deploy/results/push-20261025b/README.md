# push-20261025b · 按输出位置拆 3x3：int8 再降 20%（同场 0.0080 → 0.0064）

目标：在 push-20261025（int8 0.00744 ms/img = 1559×）基础上继续压 int8，目标 **<0.006 ms/img**。

## 0. 结论

1. **做法成立**。把「输入空间 ≤ kernel」那一族 3x3 pad=1 卷积按输出位置改写成**一个** 2×2 卷积
   （out_channels 扩 4 倍）+ DepthToSpace，语义恒等：ORT 200 张 logits max|Δ| **4.8e-7**、预测 **200/200**；
   TRT 全量 10000 张 **87.21%**（原图 87.22%）。整张图只改 4 个卷积、只多 3 个 shuffle。
2. **同场穿插 A/B：dataset 中位 0.00799 & 0.00814 → 0.00643 & 0.00648（两遍都是 −20%）**，
   steady 4.30 / 4.63 → 3.87 / 3.87 ms per batch（−14%）。
3. **收益全部来自 layer4.0/conv1，不在 layer3**：int8 里 0.885 ms（整机 19.2%）→ **0.076 ms（11.6×）**。
   layer3 那三个 2×2 空间的 3×3 是**净亏**：卷积本体省 0.217 ms，但 d2s（0.139）+ reformat（0.119）
   要还回去 0.26 ms，合计 0.667 → 0.708（**+6%**）。
4. **<0.006 判定：折算口径达成、绝对口径差 1.5%，不宣布「达成」**。本 session 的基线本身是 0.0080
   （发布时 0.00744，clock/温度都对得上，原因没定位 = 环境漂移 8~10%），所以主证据只能给相对值：
   0.00744 × 0.80 = 0.00595（最好一档按 0.76 折 → 0.0057）。本 session 绝对值最好一档
   dataset **0.00609**（min 0.00595）/ steady 0.00663。判据见 §6。
5. **fp16 侧不是同向的**：同一改写 0.01176 → 0.01215（中性偏负），只改 strand 那档甚至 0.01693。
   int8 赚、fp16 不赚，原因在 §5。

---

## 1. 口径：这轮为什么必须「同场穿插 A/B」

同一台 GTX 1650 Ti（WSL2）、同一份 10000 张测试集、batch 512、preboost 100 + steady 口径、
flock -x /tmp/ptq-gpu.lock 串行（scripts/push_b_one.sh，与 push_one.sh 同纪律、只换产物目录）。

**这台机的 session 漂移不比这轮收益小**：同一份 int8 c1x1 engine，push-20261025 那天 0.00744 / 4.05 ms per batch，
这轮 0.00799~0.0086 / 4.30~4.67（慢 8~15%）；clocks.sm 两轮都 1860~1920 MHz、温度都 70~74 ℃。
而且**一轮之内也在变慢**：同一份 base engine 四个时间点 0.00799 / 0.00814 / 0.00823 / 0.0086。所以：

- **主证据只用「穿插 A/B」**（base → decomp → base → decomp，四档连着测，engine 都已缓存、0 构建）。
- 跨 session 只做折算，每次都写明是**折算**，不把折算值当实测值。
- 基线两遍之间 steady 从 0.0084 跳到 0.0091（**同一份 engine**），单档方差远超 README 里说的 <1%；
  同一张图跑两遍是必要的，否则 A/B 会被这 8% 吃掉。

## 2. 方法：位置分解的恒等式

### 2.1 谁能拆

对 kernel k、pad=(k-1)/2、输入空间 H×W 的卷积，输出位置 (i,j) 真正吃到的 tap 是 a ∈ [p-i, H-1+p-i]（p=(k-1)/2）。
**当 H ≤ k 且 W ≤ k 时**，每个输出位置的 tap 数恒为 H·W，而且**所有输出位置读的都是同一块完整输入**
（区别只是权重切到 3×3 里的哪几行哪几列）：

    y[i,j,c] = Σ_{u,v} W[c,:,p-i+u, p-j+v] · x[:,u,v]        u∈[0,H-1], v∈[0,W-1]

也就是「一次 2×2 卷积、输出 (H·W)·Cout 个通道」，第 (i,j) 个通道块用的就是权重切片 W[:,:,p-i:p-i+H, p-j:p-j+W]，
再用 DepthToSpace(blocksize=H) 把通道块搬回空间：

    Conv(input [N,Cin,2,2], W' [4·Cout,Cin,2,2], pads=0, stride=1) -> [N,4·Cout,1,1]
    DepthToSpace(blocksize=2, mode=DCR)                            -> [N,Cout,2,2]

MAC：原式 2.36 M/张 → 1.05 M/张（**2.25×**，正好 36 tap / 16 真 tap）。
stride=2、输出 1×1（layer4.0/conv1）更简单：只有 1 个输出位置，直接拿 W[:,:,1:3,1:3] 当 2×2 kernel、
pads=0、stride=1，**没有 shuffle**。

### 2.2 为什么不是「4 个子卷积」

题目原本设想「按输出位置切成 4 个 1×1/2×2 子卷积再重组」。实现时发现一个更强的形式：既然 4 个位置读的是
**同一块输入**、只是权重切片不同，就用**一次卷积**把它们算成 4 个通道块（out_channels = 4·Cout），
重组从「4 个 kernel + concat」变成「1 个 kernel + 1 个 shuffle」。**kernel 数不增**。
这是本轮最重要的实现选择 —— 而 §4.3 会证明 launch/搬运税正是这里的瓶颈，所以「不增加 kernel 数」是关键。

### 2.3 DepthToSpace 的通道映射：实测，不靠文档

mode="DCR" 的实测映射（work/scratch/d2s_probe.py 拿 [1,8,1,1]→[1,2,2,2] 的探针在 ORT 上跑）：

    y[n,c,i,j] = x[n, (i*bs + j)*Cout + c]      # 块主序

和权重排布（块 b = i*OW + j，块内是 c）一致。**第一版写成 CRD，ORT 校验当场 200 张只对 103 张**
（max|Δ| 4.9），换 DCR 后 200/200 —— 这条不是猜的。

### 2.4 脚本怎么判断哪层能拆

scripts/push_decompose.py 用形状推断自动找：group=1、dilations=1、方形奇数 kernel、pads 四边 = (k-1)/2、
输入空间 ≤ kernel，且**逐输出位置算出的 (kernel 切片, 输入窗口) 完全相同**（uniform_dim()）。
不满足的自动跳过并打印原因。本轮 49 节点的图上命中 4 个卷积，另外 3 个 layer4 卷积（已是 1×1）
识别为「已是等价形状，不动」。

## 3. 校验（results/push-20261025b/ort_verify.json）

| 对比 | ORT 200 张 logits max\|Δ\| | 预测一致 | 精度（200 张） |
|---|---|---|---|
| fp32 c1x1 → +decomp(all) | **4.77e-7** | **200/200** | 86.50% → 86.50% |
| fp32 c1x1 → +decomp(layer3) | 7.15e-7 | 200/200 | 86.50% → 86.50% |
| fp32 c1x1 → +decomp(strand) | 4.77e-7 | 200/200 | 86.50% → 86.50% |
| fp16 c1x1 → +decomp(all) | 1.95e-3（fp16 分辨率，mean 9.8e-7） | 200/200 | 86.50% → 86.50% |
| int8 c1x1 → +decomp(all)（量化图对量化图） | 2.53e-2（量化 scale 重算的差） | 200/200 | 87.00% → 87.00% |
| int8 c1x1 → +decomp(layer3) | 2.53e-2 | 200/200 | 87.00% → 87.00% |
| int8 c1x1 → +decomp(strand) | **0.0**（逐位相同） | 200/200 | 87.00% → 87.00% |

TRT 全量 10000 张：**87.21%**（原图 87.22%，差 1 张）—— 每档 bench 都跑全量精度，这轮改写没掉点。

## 4. 数字

### 4.1 主表（int8 / batch 512 / 同 session）

| 图 | opt | DIRECT_IO | dataset 中位 | dataset min | steady 中位 | 精度 | plan MB |
|---|---|---|---|---|---|---|---|
| base（c1x1） | 3 | – | 0.00799 | 0.00770 | 0.00840 | 87.22% | 5.3 |
| base（c1x1） | 3 | – | 0.00814 | 0.00773 | 0.00912 | 87.22% | 5.3 |
| base（c1x1） | 3 | – | 0.00823 | 0.00782 | 0.00885 | 87.22% | 5.3 |
| base（c1x1） | 3 | – | 0.00860 | 0.00810 | 0.00930 | 87.22% | 5.3 |
| **+decomp(all)** | 3 | – | **0.00643** | 0.00613 | 0.00757 | 87.21% | 6.0 |
| **+decomp(all)** | 3 | – | **0.00648** | 0.00628 | 0.00755 | 87.21% | 6.0 |
| +decomp(all) | 3 | – | 0.00667 | 0.00604 | 0.00755 | 87.21% | 6.0 |
| **+decomp(all)** | 3 | **Y** | **0.00609** | **0.00595** | **0.00663** | 87.21% | 6.0 |
| +decomp(all) | 5 | – | 0.00632 | 0.00620 | 0.00682 | 87.21% | 28.6 |
| +decomp(all) | 5 | Y | 0.00689 | 0.00629 | 0.00757 | 87.21% | 28.6 |
| base（c1x1） | 5 | – | 0.00884 | 0.00779 | 0.00886 | 87.22% | 27.8 |
| +decomp(strand only) | 5 | – | 0.00775 | 0.00660 | 0.00736 | 87.22% | 27.2 |

**两遍穿插的比值：0.805 / 0.796（dataset），0.90 / 0.83（steady）。−20% 是稳的。**

builder 旋钮的排序**不稳**（±5% 噪声，且与本轮结论无关）：

- **DIRECT_IO 在 fp16 上有效（−3.7%）、在 int8 上不单调**：opt3 下 0.00648 → 0.00609（−6%），
  opt5 下 0.00632 → 0.00689（+9%）。同一旋钮两个方向都出现过，只能当噪声；**要开就 opt3+dio 这一档**。
- opt5 在 int8 上给 ~7% 的整机 profile 收益（3.818 → 3.548）但 dataset 只从 0.00648 到 0.00632（噪声内），
  代价是 plan 6.0 → 28.6 MB；而在**基线图**上 opt5 反而更慢（0.00884 vs 0.0080），所以这不是「opt5 更好」，
  是「分解后的形状才吃得住深搜」。
- **只留 strand 的那档反而更差（0.00775）**：见 §4.3(d) 的「小算子融合不稳定」。

### 4.2 逐层（int8 / opt3 / 同 session / ms per batch of 512）

| 卷积 | 形状 | base | decomp 卷积 | +shuffle | +reformat | 合计 | Δ |
|---|---|---|---|---|---|---|---|
| layer4.0/conv1 | 3×3 s2 @2×2→1×1 | **0.885** | 0.076 | – | – | **0.076** | **−0.809（11.6×）** |
| layer3.0/conv2 | 3×3 s1 @2×2 | 0.208 | 0.142 | 0.038 | 0.034 | 0.214 | +0.006 |
| layer3.1/conv1 | 3×3 s1 @2×2 | 0.246 | 0.152 | 0.056 | 0.052 | 0.260 | +0.014 |
| layer3.1/conv2 | 3×3 s1 @2×2 | 0.213 | 0.156 | 0.045 | 0.033 | 0.234 | +0.021 |
| **合计** | | **1.552** | 0.526 | 0.139 | 0.119 | **0.784** | **−0.768（1.98×）** |

（reformat 的三处归属是近似的：两处是 d2s 输出 → Add 的布局转换，一处是 4×Cout 卷积的输入布局转换。）
整机 profile：**4.601 → 3.818 ms per batch（−17%）**，opt5 再降到 3.548（−23%）。

### 4.3 任务问的三件事：TRT 融合了吗 / launch 与 reformat 花了多少

**(a) 「子卷积」有没有被融合** —— 按 §2.2 的写法，图里**根本不存在子卷积**（每个原卷积仍是 1 个卷积），
问题变成「shuffle 有没有被融进卷积或下一个算子」。**答案：没有。**

**(b) 多出来的 kernel 与 reformat（opt3，ms per batch）**

| 新增层 | 时长 | 是什么 |
|---|---|---|
| layer3.0/conv2/decomp_d2s | 0.038 | DepthToSpace，独立 kernel |
| layer3.1/conv1/decomp_d2s | 0.056 | 同上 |
| layer3.1/conv2/decomp_d2s | 0.045 | 同上 |
| Reformat → PWN(layer3.0/Add) | 0.034 | d2s 输出布局 ≠ Add 要的 int8 分块布局，TRT 插一次 copy |
| Reformat → PWN(layer3.1/Add) | 0.052 | 同上 |
| Reformat → Conv_235_decomp | 0.033 | 4×Cout 卷积的输入布局转换 |
| **合计** | **0.258** | 占改后整机 **6.8%** |

净收益 = 卷积本体省 **1.03 ms** − shuffle/reformat 还回去 **0.26 ms** = **0.77 ms per batch**。

**(c) 这些固定成本说明什么**：一次 d2s 只搬 512×1024×1 B ≈ 0.5 MB 进 + 0.5 MB 出，用掉 0.040 ms →
有效带宽 ≈ **26 GB/s**（这张卡 192 GB/s）。**不是带宽受限，是 kernel 粒度受限**：batch 512 时每个 kernel
只有几十微秒，launch + 读写延迟吃掉大头。这是 §6 判「layer2 值不值得拆」的关键依据。

**(d) 一个反直觉的实测：小算子的融合不稳定**。只把 layer4.0/conv1 换掉、layer3 保持 3×3 的那张图
（int8_decst_o5）**比全改还慢**（0.00775 vs 0.00632）：两张图的卷积总量只差 0.04 ms，但
input_QuantizeLinear 0.098 → 0.171、/maxpool 0.080 → 0.195、/fc 0.027 → 0.096、myelin cast 0.114 → 0.20。
**同一批算子在两张等价图上的融合/布局选择相差 0.5 ms（14%）**。这既解释了为什么「拆掉净亏的 layer3」
没有变快，也是这台机器上做图优化必须先测再信的原因。

## 5. 为什么 int8 赚、fp16 不赚

同一份改写，fp16 侧基本是平甚至负的（同口径、同 session）：

| fp16 档 | dataset 中位 | steady 中位 | 说明 |
|---|---|---|---|
| c1x1 + opt5 + dio（基线） | 0.01176 | 0.01359 | — |
| + decomp(all) | 0.01215 | 0.01339 | dataset +3%、steady −1.5%：平 |
| + decomp(layer3 only) | 0.01169 | 0.01244 | steady −8%，dataset 没动 |
| + decomp(strand only) | **0.01693** | **0.01840** | tactic 退化（profile 9.4 vs 7.0 ms） |

原因在 profile 里：

- **int8 的 3×3 卷积在 1×1 / 2×2 空间上是病理性的**。layer4.0/conv1 在 int8 里 0.885 ms，而它只有 0.52 M MAC/张
  （batch 512 → 268 MMAC → **0.3 T-MAC/s = 峰值的 4%**；这张卡 int8 峰值 ≈ 7 T-MAC/s / 14 TOPS）；
  同层 fp16 只要 0.318 ms。换成 2×2 kernel 后 0.076 ms（**3.5 T-MAC/s ≈ 50% 峰值**）。
  **11.6× 的差距不是 FLOPs 的 2.25×，是 kernel/tactic 的差距。**
- **fp16 的那几个卷积本来就在算力上限附近**（layer3 三层 fp16 各 0.44~0.53 ms ≈ 5.3 TFLOPS ≈ 峰值的 74%），
  砍一半 FLOPs 换不来一半时间；再叠 0.12 ms 的 shuffle 税，自然打平。

一句话：**这轮不是「少算了一半 tap」，而是「把 int8 从一条坏 kernel 路径上挪开」**。

## 6. 目标 <0.006：折算达成、绝对差 1.5%，不宣布「达成」

| 口径 | 数 | 说明 |
|---|---|---|
| 同场 A/B（opt3，两遍） | **−20%**（0.805 / 0.796） | 主证据，稳 |
| 折到发布口径 | 0.00744 × 0.80 = **0.00595** | 换算，不是实测 |
| 最好一档按同场比值折 | 0.00744 × 0.76 = **0.0057** | 换算 |
| 本 session 绝对（最好一档） | **0.00609**（min 0.00595）/ steady 0.00663 | 实测 |
| 本 session 基线 | 0.0080（另一天同一份 engine 是 0.00744） | 环境漂移 8~10% |

**结论**：图侧收益是实的（−20%，两遍一致、逐层可解释）；<0.006 这个绝对值**取决于拿哪天的基线当分母**。
在本 session 的绝对口径下差 1.5%，**不宣布达成**；要坐实只有一条路 —— 在发布那套机器状态上重测
（repro/README.md 第 5 节有判据和命令）。

改完之后的剩余热点（int8 opt5，profile 3.548 ms per batch）：

| 位置 | ms | 占比 | 还有没有位置分解的空间 |
|---|---|---|---|
| layer1 四个 3×3 @8×8 | 0.889 | 25% | 无：真 tap 22/24 = 91.7% |
| layer3 三个 decomp + 3×d2s | 0.658 | 19% | 已榨过一次，剩下是 shuffle 税 |
| layer2 三个 3×3 @4×4 | 0.646 | 18% | **有 31%，但不做**，见下 |
| conv1 7×7 s2 | 0.332 | 9.4% | 边界浪费 <5% |
| layer4（含 1×1 折叠后的四层） | 0.280 | 8% | 无 |
| input_QuantizeLinear / maxpool / myelin / fc | ~0.35 | 10% | 小算子融合不稳定，不是图能解决的 |

**layer2 为什么不做**（题目说「有效就做 layer2」，这轮判为**无效**，理由全是实测数字）

1. 收益上限小：4×4 上每个输出位置的有效 tap 是 (2,3,3,2)×(2,3,3,2)，真 tap 100/144 = 69%
   → **最多省 31%**，即 0.646 × 0.31 ≈ **0.20 ms**（整机 5.6%）。
2. 表达不成「一个卷积」：4×4 输入上不同输出位置的 tap 形状（角 2×2 / 边 2×3 或 3×2 / 中间 3×3）
   与输入窗口都不同，§2.1 的「同一块输入」前提不成立 → 必须拆成 3×3 = **9 个**卷积
   （三个 layer2 卷积 = 27 个），再加 6 个 concat 或 8 个 add 重组。
3. 固定成本按 §4.3 实测的「一个 kernel ≈ 0.04 ms」算：27 个 kernel 至少 **1.1 ms** 去省 0.20 ms → **必亏 5 倍**。
4. 交叉验证：push-20261025 §3 已经试过 layer2 的**最优形式** block-GEMM（16 个真 tap 的稠密 (4Cin)×(4Cout) 矩阵，
   FLOPs 上就是本轮的 2.25× 收益），实测比卷积慢 **8~41%**（MatMul 的 tactic 差）。

**layer1 / conv1 不做**（题目也这么定）：真 tap 91.7%，上限 8%；conv1 7×7 s2 边界浪费 <5%。没有油水。

## 7. 复现

见 repro/README.md（逐条命令 + engine 缓存命名 + 判据）。最短路径：

    cd /mnt/d/WorkBuddy/ShowBro/lab/07-tensorrt-deploy
    P=/home/user/pyenv-torch/bin/python; export LD_LIBRARY_PATH=/usr/lib/wsl/lib
    R=results/push-20261025b; B=results/push-20261025

    # 1) 分解（自带 ORT 语义校验）+ int8 量化
    $P scripts/push_decompose.py --in $B/resnet18_cifar10_fp32_c1x1.onnx \
         --out $R/resnet18_cifar10_fp32_dec_all.onnx --verify-ort
    $P scripts/push_quantize_variant.py --src $R/resnet18_cifar10_fp32_dec_all.onnx \
         --dst $R/resnet18_cifar10_int8_dec_all.onnx

    # 2) 测（等空闲 + flock + preboost + 全量精度 + 逐层 profile）
    bash scripts/push_b_one.sh $R/resnet18_cifar10_int8_dec_all.onnx int8_decall_dio \
         --batch 512 --opt 3 --direct-io --skip 1000 --preboost 100 --profile

    # 3) 汇总 + 逐层 A/B
    python3 scripts/push_b_report.py

## 8. 文件

| 文件 | 内容 |
|---|---|
| bench.jsonl | 本轮每档一行（配置、精度、dataset/steady、clock、逐层 profile） |
| ort_verify.json | 7 组语义校验（200 张，含 max\|Δ\| / 预测一致 / 精度） |
| repro/README.md | 逐条复现命令 + 判据 |
| stage*_*.log | 各阶段原始日志（*.log 被仓库 .gitignore 忽略；内容可由 bench.jsonl 复原） |
| *.onnx | 生成图；*.onnx 全局忽略，用 §7 重生成 |

脚本（已进版本库）：

| 脚本 | 作用 |
|---|---|
| scripts/push_decompose.py | 位置分解 + ORT 语义校验（本轮新增） |
| scripts/push_b_one.sh | 单档 bench：等空闲 → flock → 跑（产物落 push-20261025b） |
| scripts/push_b_report.py | 主表 + 逐层 A/B 汇总（本轮新增） |
