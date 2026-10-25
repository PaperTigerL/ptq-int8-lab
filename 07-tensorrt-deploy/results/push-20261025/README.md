# push-20261025 · 把 TRT 延迟再压一档

目标：int8 ≤ 0.016 ms/img、fp16 ≤ 0.013 ms/img（README 原值 0.0189 / 0.0149）。
结论：**两条都到**，int8 还超了一倍。

| 精度 | 图 | batch | dataset ms/img（median / min，clock 预热） | steady ms/img | 精度 | vs ncnn CPU fp32 (11.6) | 目标 |
|---|---|---|---|---|---|---|---|
| fp16 | base | 512 | 0.01304 / 0.01270 | 0.01445 | 87.18% | 889× | — |
| **fp16** | **+1x1 折叠，opt5 + DIRECT_IO** | 512 | **0.01068 / 0.01065** | **0.01222** | **87.18%** | **1086×** | ✅ ≤0.013 |
| int8 | base | 512 | 0.01775 / 0.01730 | 0.01694 | 87.22% | 654× | — |
| **int8** | **+1x1 折叠** | 512 | **0.00744 / 0.00734** | **0.00790** | **87.22%** | **1559×** | ✅ ≤0.016 |

> **绝对数字别跨机器引用**，同场 A/B 才是结论：fp16 0.01304 → **0.01068**（−18%，含 opt5+DIRECT_IO），
> 只换图（opt3）是 0.01109（−15%）；int8 0.01775 → 0.00744（−58%）。
> 数据里还留着不预热的原始数字
> （fp16 0.0214 / int8 0.0215），那才是这台机器"空闲但 clock 没起来"时能测到的值。

---

## 1. 先修测量：这台卡的 clock 会掉

同一份 fp16 engine、同一份数据，本轮第一轮就测出 0.0149 和 0.0214 两个数（**1.44 倍**）。
`nvidia-smi` 给出答案：

```text
clocks.current.sm = 1350 MHz   clocks.max.sm = 2100 MHz
```

空闲时 GPU 只有 **1350 MHz**（上限的 64%），跑起来才升到 1755–1920。
而 `trt_eval.py` 的逐批协议是「H2D → execute → sync → CPU argmax」，
每批之间有 5–10 ms 的 python 间隙，GPU **每批都要重新爬 clock**，所以 ms/img 被系统性抬高。

旧 `idle_final.txt` 能测到 0.0138，是因为桌面上那些 Electron 窗口（WorkBuddy/豆包/WXWork）
一直在给 GPU 底噪，clock 一直是高的 —— **"等 GPU 空闲"反而把 clock 等低了**。
这是本轮最反直觉的一条。

处理方法（两条都保留在产物里）：

1. **`--preboost N`**：每轮 dataset 测量前先背靠背跑 N 批（默认 100），把 clock 顶上去再开测。
   只预热硬件时钟，不跳过任何计算、不换数据。
2. **steady 口径**：同址 buffer 背靠背 enqueue，CUDA event 掐总时间 / 批数，方差 <1%。
   量的是**持续吞吐**（受功耗/温度限制，通常比预热后的短突发慢一点点）。

每次测量都打印 `clocks.sm / temp / power`，`clocks_steady` 里能看到 1455 → 1860 MHz 的爬升。

---

## 2. 优化一（主力）：1x1 空间上的 3x3 卷积，白算 9 倍

**发现**：ResNet-18 在 32×32 输入下，layer4 的空间尺寸已经是 **1×1**，卷积却还是 3×3、pad=1。
3×3 窗口里只有中心 tap 碰到真实像素，另外 8 个 tap 乘的都是 padding 零。
这三层（layer4.0/conv2、layer4.1/conv1、layer4.1/conv2）占 fp16 总时间 16%、int8 的 **47%**。

**改法**（`scripts/push_optimize_onnx.py`）：权重切中心 tap `W[:,:,1,1]`、kernel 改 1×1、pad 清零。
恒等式：`sum_{a,b} W[a,b] x_pad[a,b] = W[1,1] x`（x_pad 只有中心非零）。

**验证**（不只看速度）：

- ORT CPU、200 张真实图片：logits 最大差 **2.4e-7**，预测 **200/200 一致**，精度 86.50% → 86.50%
- TRT 全量 10000 张：fp16 87.18%（= base）、int8 87.22%（= base）
- plan：fp16 22.5 → 9.9 MB，int8 11.6 → 5.3 MB

**逐层效果**（fp16，batch 512，steady 8.44 → 6.45 ms/batch）：

| 层 | base | +1x1 折叠 | |
|---|---|---|---|
| layer4.1/conv2 | 0.512 ms | 0.135 ms | −74% |
| layer4.0/conv2 | 0.465 ms | 0.114 ms | −75% |
| layer4.1/conv1 | 0.394 ms | 0.101 ms | −74% |

int8 侧更夸张（那三层在 int8 里是 2.364 / 2.160 / 1.731 ms，见 §6），折叠后整机
10.45 → 4.08 ms/batch（**−61%**）。

---

## 3. 优化二（失败，但有信息量）：2x2 空间上的 3x3 → block-GEMM

layer3 的 3 个卷积是「2×2 输入、3×3 pad=1、2×2 输出」：每个输出像素的窗口覆盖整个 2×2 输入，
36 个 tap 里只有 16 个是真的。理论上可写成 `Y = X_flat @ M`（M 是 (Cout·4)×(Cin·4) 的稠密块矩阵），
FLOPs 少 **2.25×**。`scripts/push_blockgemm.py` 实现了它，语义校验同样通过
（ORT 200 张：logits 最大差 4.8e-7、预测 200/200 一致）。

**但 TRT 上它是亏的**：

| 图 | batch 512 steady |
|---|---|
| fp16 + 1x1 折叠（opt5） | **0.01238** |
| + block-GEMM（opt3） | 0.01746（+41%） |
| + block-GEMM（opt5） | 0.01338（+8%） |

profile 显示 3 个 MatMul 各 1.43 ms（opt3）：512×1024×1024 的 GEMM 只有 **0.75 TFLOPS**，
而同样 FLOPs 的卷积能跑 4.8 TFLOPS。**瓶颈不在 FLOPs，在 TRT 对这个 GEMM 形状的 tactic**。
opt5 能把 MatMul 压到 ~0.55 ms，但仍追不上卷积。结论：这条优化在 TRT 上不成立，
脚本与数据保留，不并进主图。

---

## 4. builder 旋钮扫描（fp16 + 1x1 折叠图，batch 512）

| 配置 | dataset median | dataset min | steady | 相对最优 |
|---|---|---|---|---|
| **opt5 + DIRECT_IO（最终主配置）** | **0.01068** | **0.01065** | **0.01222** | 1.00× |
| opt5 | 0.01109 | 0.01089 | 0.01238 | +3.8% |
| opt3 | 0.01138 | 0.01107 | 0.01267 | +6.6% |
| opt2 | 0.01150 | 0.01131 | 0.01290 | +7.7% |
| opt5 + workspace 0.5 GB | 0.01122 | 0.01079 | 0.01225 | +5.1% |
| opt3 + CUDA Graph | 0.01119 | 0.01107 | 0.01261（graph 0.01267） | +4.8%（graph 本身 0.5%） |
| opt0 | 0.02687 | 0.02602 | 0.02810 | +152% |
| opt3 + `max_aux_streams=2` | 0.01381 | 0.01339 | 0.01438 | +29% |
| opt5 + tactic = 只开 JIT | 0.09234 | 0.07964 | 0.09411 | +765% |

要点：

- **opt level 是最大旋钮**：opt0 直接 2.4× 慢；opt3→opt5 再快 2.6%。默认 3 不是最优。
  对 GEMM 型图（block-GEMM）opt 级别差 30%，因为 GEMM 的 tactic 质量对搜索深度更敏感。
- **tactic source 只剩两个源**（TRT 11.4 把 cuDNN/cuBLAS 源删了）：`EDGE_MASK_CONVOLUTIONS` 和
  `JIT_CONVOLUTIONS`。只开 JIT 会让卷积掉到 8× 慢 —— 说明这颗卡的好卷积 kernel 全在
  EDGE_MASK 那一支，**两个源必须都开（默认值）**。
- **DIRECT_IO 有效**：`cfg.set_flag(trt.BuilderFlag.DIRECT_IO)` 读回 True，接口仍是
  NCHW 连续（`get_tensor_strides = (3072,1024,32,1)`），产物正确；dataset 0.01109 → **0.01068**（−3.7%），
  steady 0.01238 → 0.01222。大概率是省掉了输入 NCHW→NHWC 的 reformat copy
  （base 图里这一项单独占 0.19–0.96 ms），但没有对 dio engine 单独 profile 证实。
- **workspace** 0.5 GB vs 1 GB 没有可测差异（4 GB 卡，2 GB 的档没跑：会和 batch 512 的
  activation 抢内存）。
- **CUDA Graph 仍然无收益**：0.01267 vs 0.01261（0.5%，噪声内），与 README「三·五」的三次实测一致。
  batch=512 时每批 6.4 ms，launch 开销被摊薄到看不见。
- **多流是负收益**：`max_aux_streams=2` 让 TRT 把部分 kernel 拆到辅助流，同步开销 > 并行收益
  （+25%）。手工双流切 batch 在算力受限的图上同理，没做。
- **timing cache**：TRT 11.4 的 python 绑定 **没有** `Runtime.deserialize_timing_cache`
  （只剩 `ITimingCache` / `TimingCacheKey` / `TimingCacheValue` 三个符号）。
  `config.get_timing_cache().serialize()` 写得出 125 KB 的 cache 文件，但**读不回来**，
  跨进程复用只能走 C++/trtexec。所以本轮每档构建都是冷启动（fp16 ~90–180 s，int8 最多 680 s）。
  它只影响构建时间、不影响运行延迟，因此没有把它的收益算进任何 ms/img。

---

## 5. batch 大小

| batch | dataset median | dataset min | steady | 说明 |
|---|---|---|---|---|
| 512（opt5 + DIRECT_IO） | **0.01068** | 0.01065 | **0.01222** | 主口径，最快 |
| 1024（opt5） | 0.01068 | 0.01063 | 0.01226 | 与 512 持平 |
| 2048（opt5） | 0.01084 | 0.01077 | 0.01228 | 略差；4 GB 卡到这儿没收益 |

batch 512–1024 是平台区，2048 不再收益（与 README 里「1024 反而比 512 慢」的老结论方向一致，
只是这轮差距在噪声内）。**主配置取 batch 512**：显存占用小、尾批不用特殊处理。

---

## 6. int8：原来为什么比 fp16 慢，折叠后为什么反超

旧结论「小模型 + 小分辨率 + 中低端 GPU，fp16 优于 int8」这次定位到了**具体原因**，不是普遍规律。

int8 base 的逐层 profile（13.4 ms/batch）：

| 层 | int8 | 同层 fp16 | |
|---|---|---|---|
| layer4.0/conv2（3x3@1x1） | 2.364 ms | 0.465 ms | int8 反而慢 5× |
| layer4.1/conv1（3x3@1x1） | 2.160 ms | 0.394 ms | 慢 5.5× |
| layer4.1/conv2（3x3@1x1） | 1.731 ms | 0.512 ms | 慢 3.4× |
| layer2.0/conv2（4x4） | 0.168 ms | 0.389 ms | **int8 快 2.3×** |
| layer3.x（2x2） | 0.195–0.28 ms | 0.42 ms | **int8 快 ~1.7×** |

int8 在**正常空间尺寸**的卷积上都是赚的（约 2×），问题只出在那 3 个 1×1 空间的卷积：
int8 kernel 要为每层做 NHWC/reformat，输出又只有 1 个像素，固定开销吃掉全部收益。
1×1 折叠把这三层变成 512×512 的 1×1 卷积（等价 GEMM），int8 直接反超：

| int8 | ms/batch | ms/img（steady） |
|---|---|---|
| base | 10.45 | 0.02040 |
| +1x1 折叠 | **4.08** | **0.00790** |

精度：折叠前 87.22% / 折叠后 87.22%（fp32 87.16%，同精度）。

折叠后 int8 最大剩余热点是 `/layer4/layer4.0/conv1`（2×2→1×1、stride 2 的 3×3）：
0.86 ms / 22%，同样是 padding 浪费（4 真 tap / 9），但 block-GEMM 那条路已证明不划算。

---

## 7. int8 权重 + fp16 激活（混跑）：能跑，但数字是假的

动机：int8 图里未量化的算子会回落到 **fp32**（比 fp16 慢一倍），边界层在拖后腿。
做法：拿 **fp16 图**，只把 Conv/MatMul 换成 int8 QDQ，其余算子保持 fp16
（`scripts/push_make_mixed.py`；ORT 用 fp16 校准数据量化 fp16 图，再 strip int32 bias）。
另做了 `--exclude-layer4` 变体。

**结果：steady 0.01133 ms/img（比 fp16 还快 8%），精度 9.94% —— 随机猜。**

这就是 README「三·五」里那个坑的翻版：只看速度会以为混跑成功了。
好消息是 bench 每档都强制跑全量精度，所以它当场被抓住。
根因（未完全定位，证据充分）：ORT 在 fp16 图上量化出的 Q/DQ **scale 是 float16**，
而 ONNX 规范要求 int8 Q/DQ 的 scale 是 float32；TRT 很可能把 fp16 scale 当坏值处理了。
即使修好，也不划算：外层 fp16 的 3 个 block-GEMM MatMul（0.70 ms×3）+ layer4.0/conv1（0.40 ms）
注定让它比纯 int8（0.00790）慢。

**结论：混跑这条路在本模型上不成立**，纯 int8（+1x1 折叠）才是最优。

---

## 8. 融合与 plugin：TRT 已经做了，不需要自己写

profile 的层名直接证明 TRT 已经融合了 Conv+BN+ReLU 和 Conv+Add+ReLU：

```text
/layer3/layer3.1/conv2/Conv + /layer3/layer3.1/Add + /layer3/layer3.1/relu_1/Relu   0.425 ms
/layer1/layer1.0/conv1/Conv + /layer1/layer1.0/relu/Relu                            0.409 ms
```

- BN 在 torch.onnx.export 时就被折进 Conv（图里 0 个 BatchNorm 节点）
- Conv+ReLU / Conv+Add+ReLU 被合成一个 kernel（层名里能看到拼接）
- 常数折叠在 `do_constant_folding=True` 已做（49 节点 = 20 conv + 17 relu + 8 add + 1 gemm + 1 pool + 1 gap + 1 flatten）

所以「写 plugin 做 Conv+ReLU+Add」没有意义：TRT 已经这么做了；自研 plugin 在 1×1 空间
那种访存受限的形状上也不会比现成 kernel 快。稀疏化（2:4）Turing 不支持。

---

## 9. 瓶颈在哪、还能往哪走

fp16 现在是 30.7 M MAC/张（折叠前 37.0 M）。batch 512 稳态 **6.26 ms**（0.01222 × 512）：
**31.4 GFLOP / 6.26 ms = 5.02 TFLOPS**，1650 Ti 在 1755 MHz 下的 fp16 峰值约 7.2 TFLOPS
（1024 core × 1755 × 2 fp16x2 × 2 FMA）→ **约 70%**。int8 的 7.9 TOPS / 约 14 TOPS 峰值 ≈ 56%。

还能做的（按性价比）：

1. **layer3 的 padding 浪费**（2.25×）—— 需要 TRT 有高效的 (Cout·4)×(Cin·4) GEMM tactic，实测没有。
2. **conv1 7x7 s2 + maxpool**：0.6 + 0.13 ms，加输入 reformat 0.2–0.5 ms，是固定成本。
3. **fp8 / 稀疏**：Turing 都不支持。
4. **把 input 量化/转换挪到 host**：不诚实（把 0.2–0.9 ms 挪出计时窗口），没做。

**一句话**：fp16 已经贴着这颗卡的算力上限（~70%），int8 还有一点空间（56%），
剩下的大头是 TRT 对 1×1 空间形状的 kernel 选择，不是图能解决的。

---

## 10. 复现

```bash
cd /mnt/d/WorkBuddy/ShowBro/lab/07-tensorrt-deploy
P=~/pyenv-torch/bin/python
export LD_LIBRARY_PATH=/usr/lib/wsl/lib

# 1) 底稿 ONNX（fp32 给量化做底，fp16 直接建 engine）
$P scripts/export_onnx.py all

# 2) 语义等价图优化：1x1 空间 3x3 -> 1x1（自带 ORT 数值校验）
$P scripts/push_optimize_onnx.py --in work/resnet18_cifar10_fp16.onnx \
    --out results/push-20261025/resnet18_cifar10_fp16_c1x1.onnx
$P scripts/push_optimize_onnx.py --in work/resnet18_cifar10_fp32.onnx \
    --out results/push-20261025/resnet18_cifar10_fp32_c1x1.onnx --verify-ort

# 3) int8：对折叠后的 fp32 图做 QDQ 量化（与 quantize_int8.py 完全同口径）
$P scripts/push_quantize_variant.py \
    --src results/push-20261025/resnet18_cifar10_fp32_c1x1.onnx \
    --dst results/push-20261025/resnet18_cifar10_int8_c1x1.onnx

# 4) 测（等空闲 + flock 锁 + clock 预热 + 全量精度）
bash scripts/push_one.sh results/push-20261025/resnet18_cifar10_fp16_c1x1.onnx \
    512 fp16_c1x1 --opt 5 --preboost 100 --graph
bash scripts/push_one.sh results/push-20261025/resnet18_cifar10_int8_c1x1.onnx \
    512 int8_c1x1 --skip 1000 --preboost 100
```

`scripts/push_one.sh` 封装了纪律：等 GPU 空闲 → `flock -x /tmp/ptq-gpu.lock` → 跑 → 释放。
结果 JSON 追加到 `bench.jsonl`，`scripts/push_summary.py` 汇总成 `sweep.txt`。

## 11. 文件

| 文件 | 内容 |
|---|---|
| `bench.jsonl` | 每档一行 JSON（配置、精度、dataset/steady/graph、clock、逐层 profile） |
| `sweep.txt` | 由 `push_summary.py` 生成的汇总表 |
| `phase0/phaseA/phaseB/phaseC.log` | 各阶段原始日志（仓库根 `.gitignore` 忽略 `*.log`，只留在本地；同内容可从 `bench.jsonl` 复原） |
| `*.onnx` | 生成图；仓库 `.gitignore` 里 `*.onnx` 全局忽略，用 §10 的命令重生成 |
