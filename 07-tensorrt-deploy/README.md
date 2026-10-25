# 07 · TensorRT 部署（fp16 / int8）

把 `03-ptq-lab` 的 ResNet-18 CIFAR-10 模型搬到 TensorRT，
目标是回答一件事：**在不作弊的前提下，这条路能比 ncnn CPU fp32（11.6 ms/img）快多少。**

**结论（2026-10-25 口径，protocol = preboost+steady）：int8 batch 512 = 0.00744 ms/img（1559×），
fp16 = 0.01068 ms/img（1086×）**，精度 87.22% / 87.18%，与 fp32 基线同精度。
关键是把 layer4「1×1 空间上的 3×3 卷积」折叠成 1×1，同时不再「等 GPU 空闲」（那会把 clock 等低）。

> **旧口径（idle 门控，2026-10-20 测）**：fp16 0.0149 ms/img = 778×、int8 0.0189 = 614×。
> 同一块卡、同一批 engine，差的是 clock 状态与折叠后的图。**两套口径并存**，
> 完整对照见 `../08-report/README.md` 第二节；本目录第二节的两张表都标了「旧口径」。
> 早期版本还写过「0.0128 ≈ 906×」—— 那是二十轮的**最小值**且 GPU 被占用，最不可信。

> **2026-10-25 第二轮（`results/push-20261025b/`）**：同一类 padding 浪费的另外两个形状 ——
> 把「输入空间 ≤ kernel」那族 3x3 pad=1 卷积按输出位置改写成**一个** 2×2 卷积（out_channels 扩 4 倍）
> + DepthToSpace，语义恒等（ORT 200 张 max|Δ| 4.8e-7、预测 200/200；TRT 全量 87.21%）。
> **int8 同场穿插 A/B：dataset 中位 0.00799 / 0.00814 → 0.00643 / 0.00648（−20%）**，steady −14%。
> 收益**几乎全部来自 layer4.0/conv1**（int8 0.885 → **0.076 ms，11.6×**）；layer3 那三层是**净亏**
> （shuffle 0.139 + TRT 插的 reformat 0.119 > 卷积省下的 0.217）。折到本轮口径 0.00744 × 0.80 ≈
> **0.00595 ms/img ≈ 1950×**（本 session 绝对值最好一档 0.00609，基线自身漂移 8~10%）。
> **layer2 判为不做**：真 tap 100/144，上限只省 0.20 ms，却要 9 个 kernel/卷积（27 个）→ 必亏。
> 详见 `results/push-20261025b/README.md`。

## 〇、2026-10-25 复盘：1x1 空间上的 3x3 是白算（int8 0.0189 → 0.0074，fp16 0.0149 → 0.0107）

> 完整数据、脚本、逐层 profile：**`results/push-20261025/README.md`**。
> 下面所有数字都来自同一次 session、同一套口径（口径见本节末尾与第二节）。

**发现**：ResNet-18 在 32×32 输入下，layer4 的空间尺寸已经是 **1×1**，但卷积还是 3×3、pad=1 ——
3×3 窗口里 8 个 tap 乘的是 padding 零，**白算 9 倍**。这三层在 fp16 里占 16% 时间，
在 int8 里占 **47%**（int8 kernel 在 1×1 空间上还要额外做 NHWC reformat，比 fp16 还慢 4–5 倍）。

**改法**：在 ONNX 层把这三层的权重切成中心 tap、kernel 改 1×1、pad 清零（数学恒等）。
ORT 校验 200 张：logits 最大差 2.4e-7、预测 200/200 一致；TRT 全量 10000 张精度不变。

| | base | +1x1 折叠 | 提升 |
|---|---|---|---|
| fp16 batch=512（steady） | 0.01648 | **0.01222**（opt5+DIRECT_IO） | −26% |
| fp16 batch=512（dataset median，clock 预热） | 0.01304 | **0.01068**（opt5+DIRECT_IO） | −18% |
| int8 batch=512（steady） | 0.02040 | **0.00797** | −61% |
| int8 batch=512（dataset median，clock 预热） | 0.01775 | **0.00744** | −58% |
| 精度 | fp16 87.18% / int8 87.22% | 同 | 不变 |

> **独立复现（另一轮，冷构建、从原始 ONNX 开始）**：int8 **0.00725 ms/img（1600×）/ 87.22%**
> —— 比原轮还快 2.55%，图（ONNX）逐字节一致；fp16 重建 3 次落在 0.0107–0.0111，
> 其中 1 次选到差 tactic 掉到 0.0126–0.0138（构建抖动，不是图或口径问题）。
> 命令、日志、SHA、两口径对账：`results/push-20261025/repro/`。

**顺手修掉的测量坑**：这台 1650 Ti 空闲只跑 1350 MHz（上限 2100）。逐批协议每批之间有
5–10 ms 的 python 间隙，GPU 每批都要重新爬 clock —— 同一份 engine 能测出 0.0149 和 0.0214。
现在每档开测前先背靠背预热（`scripts/push_one.sh --preboost 100`），并同时报 steady
（背靠背 enqueue）口径；每次打印 clock/温度，两个数字能不能比一眼可查。
**「等 GPU 空闲」反而把 clock 等低了**，这是本轮最反直觉的一条。

**试过但没收益的**：

- CUDA Graph 复测 0.01267 vs 0.01261（0.5%，噪声内），与「三·五」结论一致
- layer3「2x2 空间上的 3x3」改 block-GEMM（FLOPs 少 2.25×）：语义对，但 TRT 的 GEMM tactic
  只有 0.75 TFLOPS（卷积 4.8），opt3 下反而慢 38%、opt5 下仍慢 5.6%（`scripts/push_blockgemm.py`）
- aux streams（`max_aux_streams=2`）：0.01381 vs 0.01109，**更慢**，不要开
- plugin 不用写：TRT 已经把 Conv+ReLU+Add 合成一个 kernel（逐层 profile 的层名可证）

**现在的上限**：fp16 稳态 6.2 ms/batch ≈ 4.9 TFLOPS，约为 1755 MHz 下峰值的 70%；
int8 约 55%。剩下的大头是 TRT 对 1×1 空间形状的 kernel 选择，不是图能解决的。

---

## 一、先说测量纪律（不看这段，后面的数字会被你读错）

这台机器**常年有别进程占着 GPU**。同一份 engine，同一份数据：

| 测量时 GPU 状态 | 同一份 fp16 engine |
|---|---|
| 空闲 | 0.0138 ms/img |
| 被别的进程占着 | 0.76 ms/img |

**差 55 倍。** 所以本项目规定：

1. 测之前先 `${SHOWBRO_ROOT}/lab/07-tensorrt-deploy/scripts/wait_idle.py`
   等 `utilization.gpu < 5%`
2. 必须 `--repeat N` 多轮取最小，单轮数字没有意义
3. `trt_eval.py` 每轮都会打印当时的 GPU 占用，让污染可追溯

跑分脚本里那条 `⚠️ 波动` 告警不是装饰——出现它就说明这轮数据在抓时间段，
需要重测。

---

## 二、结果

数据见 `results/final.txt`（20 轮取最小）和 `results/sweep.txt`（batch 扫描）。

**旧口径（idle 门控，每配置 5 轮）：`results/idle_final.txt`** ——
这是 2026-10-20 的测量，**clock 未预热**。新口径（preboost+steady）见 §〇 与 08-report §二。

| 精度 / engine | batch | ms/img（中位） | ms/img（min） | 10000 张 | 精度 | vs ncnn CPU fp32 |
|---|---|---|---|---|---|---|
| fp32 | 512 | 0.0276 | 0.0274 | 0.28 s | 87.16% | 420× |
| **fp16** | **512** | **0.0149** | 0.0138 | **0.15 s** | **87.18%** | **778×** |
| int8 QDQ | 512 | 0.0189 | 0.0181 | 0.19 s | 87.22% | 614× |

下面是**被占用时**的历史数据（`results/final.txt`，batch 扫描），保留作对照：

| 精度 / engine | batch | ms/img（最优） | 同批中位 | 10000 张 | 精度 | vs ncnn CPU fp32 |
|---|---|---|---|---|---|---|
| fp32 | 512 | 0.0353 | 0.0465 | 0.35 s | 87.16% | 329× |
| fp16 | 512 | 0.0128 | 0.0196 | 0.13 s | 87.18% | 906× |
| fp16 | 2048 | 0.0131 | 0.0172 | 0.13 s | 87.18% | 885× |
| fp16 | 4096 | 0.0141 | 0.0403 | 0.14 s | 87.18% | 823× |
| int8 QDQ | 512 | 0.0170 | 0.0646 | 0.17 s | 87.22% | 682× |

### ⚠️ 关于口径：min 不是性能指标，median 才是

`results/final.txt` 跑的时候 GPU 被别的进程占着（`nvidia-smi` util 常年 40–99%），
所以 fp16 二十轮从 0.0128抖到 0.0273（**2.1 倍**），int8 从 0.0170 抖到 0.1976（**11.6 倍**）。

**方向性纠正（原来写反了）**：被别的进程占着只会让数字**偏慢**——
抢 SM、抢显存带宽、上下文切换。所以「被占时的数字」是**下界**，
空闲测出的 min 才是性能**上限**。原文写「这些是上界」是错的。

那906× 怎么来的？它= 11.6 / 0.0128，用的是二十轮的**最小值**。
同一份数据的中位数是 0.0196 ms/img，对应 **592×**。**用最小值当性能指标是自欺。**

**本目录以`results/idle_final.txt` 为准**：

| 口径 | 用途 |
|---|---|
| **min** | 性能上限（硬件能力的乐观值） |
| **median** | **可信的代表值**（对外报数用这个） |

`idle_final.txt` 由 `idle_final.sh` 生成，判据如下。

### 门控：`util <= 20%` 连续 3 次，最多等 60s，等不到也照常测

**为什么阈值是 20% 而不是 5%**：本机**空闲时** util 就有 10~18%——
桌面合成器 + 常驻 Electron 窗口（WorkBuddy / 豆包 / WXWork / DingTalk /
B站，约 790 MiB）撑着。它们是**图形进程**，`nvidia-smi` 的 compute-apps
里查不到，但让 `utilization.gpu` 常年有底噪。而**真实计算负载时**
util 稳定在 95~100%。20% 恰好落在两者之间。

**曾试过「相对基线」（`util <= baseline+8`），实测失败**：
采样 10 次得到 `[95,97,96,96,87,100,36,0,100,100]`——
min 恰好采到一次 0%，把基线拉到 0，相对判据塌成 `util<=8%`，比不做还糟；
而 median 是 100，判据变成 `util<=108`，等于没有门控。
**两头都会坏，说明「估基线」这个思路本身不对**：底噪不是稳定常数，
猜它不如直接用物理上有意义的绝对阈值。

**等不到也照常测**（跳过 = 白跑）。事后由 `summarize_idle.py` **按 util 分档**：
`util<=20%` 的轮次为「低档」，其 median 才是可信值；`util>20%` 的单列为
「被占用时的下界」，不混进主结论。

**util 必须在轮次开跑前采样**：原来只在轮次结束后采，那时本进程已跑完，
采到的是别的进程的负载 —— 结果出现「util 66% 的轮次比 util 11% 的更快」
这种自相矛盾的数据。现在开跑前/跑完后各采一次，用开跑前的值分档。

可比性靠「同一次测量、同一套口径」保证，绝对值不要跨机器引用。

batch 不是越大越好：1024 反而比 512 慢（同一份 `sweep.txt` 里 b1024 最优 0.0272），
因为这颗 4 GB 卡到这个尺寸开始有显存压力。512～2048 都是平台区。

### 横向对比（同一模型、同一份测试集、同一精度口径）

| 方案 | ms/img（中位） | 全量耗时 | vs ncnn CPU fp32 | 口径 |
|---|---|---|---|---|
| ncnn CPU fp32（起点） | 11.6 | 116 s | 1× | CPU 4 线程 |
| ncnn CPU int8 4 线程 | 3.4 | 34 s | 3.4× | CPU 4 线程 |
| PyTorch CPU batch=128 | 0.15 | 1.5 s | 77× | batch128 |
| libtorch C++ GPU batch=512 | 0.0784 | 0.78 s | 148× | batch512 |
| TensorRT fp32（1×1 折叠） | 0.02477 | 0.25 s | 468× | **preboost+steady** |
| **TensorRT fp16（1×1 折叠）** | **0.01068** | **0.11 s** | **1086×** | **preboost+steady** |
| **TensorRT int8（1×1 折叠）** | **0.00744** | 0.067 s（9000 张） | **1559×** | **preboost+steady** |

（TensorRT 三行是**新口径**：batch 512、preboost 100、dataset 中位；steady 值与
逐层 profile 见 `results/push-20261025/README.md`。**旧口径** 778×/614×/420×
见 `../08-report/README.md` §4.2 与 `results/idle_final.txt` —— 两套口径并存，
不要混着引用。完整七路线对照见 `../08-report/README.md`。）

精度：fp32 87.16% / fp16 87.18% / int8 87.22% —— **都在误差范围内，是同精度，不是近似**。

---

## 三、为什么不走 ncnn / 为什么最终是 TRT

1. **ncnn 主线没有 CUDA 后端**：`ncnn/src/layer/cuda/` 不存在，`src/CMakeLists.txt`
   grep 不到 `NCNN_CUDA`，主线 GPU 路径只有 Vulkan。
   **但社区 fork（`atanmarko/ncnn-with-cuda`）是有的，而且我真的把它编译、跑通、测完了** ——
   完整记录在 [`../05-ncnn-deploy/cuda/README.md`](../05-ncnn-deploy/cuda/README.md)。
   结论：修掉「CUDA 层建了却没被调用」的 bug 之后 fp32 GPU 精度 88.00%，
   但**天花板只有 0.8×**（比 CPU 基线还慢），int8 CUDA 路径精度 9.00% 是坏的。
   根因是框架层：ncnn 的 `Mat` 没有 batch 维，换 Vulkan 也一样
   （而且这台机器 Vulkan 根本不可用，见那份 README 里的探测记录）。
2. **libtorch C++ 已经到 148×**，但它的 kernel 是 PyTorch 现成的，
   没有算子融合级别的优化空间。
3. **TRT 的三个增量**：conv+BN+ReLU 融合、per-conv kernel 调优、
   fp16/int8 tensor core。实测把 0.0784 压到 0.0128，约 **6 倍**。

---

## 三·五、CUDA Graph：试过了，没有收益（而且差点被假数字骗）

思路：ResNet-18 在 32×32 上**层多而每层极小**，怀疑 CPU 侧 launch 开销占比高，
用 CUDA Graph 把整批推理固化成一次 launch 应该能压下去。

**结论：没有收益。** 全量 10000 张、三次独立测量：

| 批次 | graph | 非 graph | graph 提速 |
|---|---|---|---|
| 2048 张 ×3 轮（GPU 76%） | 0.0124 | 0.0123 | **0.99×** |
| 10000 张 ×5 轮（GPU 33%） | 0.0132 | 0.0131 | **0.99×** |
| 10000 张 ×5 轮（GPU 88%，graph 先跑） | 0.0270 | 0.0125 | 0.46×（噪声） |

**没有一次显示 graph 更快。** 原因很好懂：batch=512 已经把 launch 开销摊薄到
看不见了，graph 抹掉的那点收益淹没在噪声里。CUDA Graph 对小模型**大 batch**
没意义，只对"小 batch / 需要极低单批延迟"的场景有价值。

### ⚠️ 中间踩到一个非常危险的坑

第一版用 `relaxed` 捕获模式，TRT 不报错，torch 只给一句 warning：

```
UserWarning: The CUDA Graph is empty.
  graph round 1: 0.0016 ms/img  精度 9.67%    ← "7258×"
```

**快 23 倍，全是假的。** relaxed 模式把错误吞了，抓到的是**空 graph**，
`replay()` 什么都没做，`d_out` 根本没被写，读到上一次的残留 —— 精度 9.67% 就是随机猜。

所以现在 `capture_graph()` 强制加了一步**输出一致性校验**：
plain 跑一次存基准 → 捕获 → `replay()` → 逐位比对，对不上就判这个模式失败、换下一个。
**任何 graph 优化都必须先验证输出一致，只看速度会直接被骗。**

另外顺手修掉一个 TRT 自己警告的问题：
`Using default stream in enqueueV3() may lead to performance issues due to
additional calls to cudaStreamSynchronize()`。
现在推理一律跑在专用 `torch.cuda.Stream()` 上，捕获和 replay 都用同一个流。
（`execute_async_v3` 显式指定流，`execute_v2` 走 TRT 自己的默认流，不一定落在捕获流上。）

脚本：`scripts/trt_eval_graph.py`，`final_graph.sh`，结果 `results/cudagraph.txt`。

---

## 四、⭐ TRT 11.4 的 python 绑定缺了一大堆枚举

这一节是本项目最花时间的地方，**别再用旧版 API 的写法**。

```python
>>> list(trt.BuilderFlag.__members__.keys())
['DEBUG', 'GPU_FALLBACK', 'REFIT', 'DISABLE_TIMING_CACHE', 'EDITABLE_TIMING_CACHE',
 'TF32', 'SPARSE_WEIGHTS', 'SAFETY_SCOPE', 'DIRECT_IO', 'VERSION_COMPATIBLE', ...]
```

**没有 `FP16`，没有 `INT8`，没有 `OBEY_PRECISION_CONSTRAINTS`。**
而且绕不过去：

```python
cfg.set_flag(trt.BuilderFlag(0))   # 想设 kFP16
cfg.flags                           # -> 64，没变
cfg.flags = 65                      # 直接写位
cfg.flags                           # -> 64，还是没变
```

getter 和 setter 都是坏的。枚举在这里是位下标（`DEBUG=2` → mask `1<<2`），
但 pybind 那条路径对未注册值不生效。
`TacticSource` 同样只剩 `EDGE_MASK_CONVOLUTIONS` / `JIT_CONVOLUTIONS`。

**结论：精度只能在图里表达。**

- fp16 → 导出 Float16 typed 的 ONNX（见 `export_onnx.py`）
- int8 → Q/DQ 节点（见 `quantize_int8.py`）

两条路都不需要任何 builder flag。

### 验证精度的唯一可靠办法

**不要看 plan 大小。** fp16 和 fp32 两版 engine 都是 44.9 MB，跑分也只差 1.28 倍，
看起来"fp16 生效了"——但 `inspect_engine.py` 一打，逐层全是
`volta_scudnn_128x64_relu_medium_nn_v1`，**全是 fp32 kernel**。

```bash
python scripts/inspect_engine.py work/resnet18_cifar10_fp16_b2048.trt 2048
```

fp16 生效的 engine（`inspect_engine.py` 实测输出）：

```text
/conv1/Conv + /relu/Relu                    turing_fp16x2_hcudnn_fp16x2_128x64_relu_medium_nn_v1
/layer1/layer1.0/conv1/Conv + ...relu/Relu  turing_fp16x2_hcudnn_fp16x2_128x64_relu_medium_nn_v1
/layer2/layer2.0/conv2/Conv                 volta_fp16x2_hcudnn_fp16x2_128x64_relu_small_nn_v1
tactic 关键字分布: {('fp16','turing'): 13, ('fp16','volta'): 7, ...}
```

int8 真跑起来时是这些：

```text
/conv1/Conv + PWN(/relu/Relu)               volta_int8x4_icudnn_int8x4_128x64_relu_xregs_large_nn_v1
/layer4/layer4.1/conv2/Conv + ...           sm75_xmma_fprop_implicit_gemm_interleaved_i8i8_i8i32_f32_...
```

---

## 五、int8 QDQ 的三个必踩的坑

按 ncnn 那套语义（`uint8` 非对称激活 + `int8` per-channel 对称权重）走到哪一步炸为止：

### 坑 1 · TRT 在 GPU 上不支持非对称量化

```
[E] onnxOpImporters.cpp:1931 Assertion failed: shiftIsAllZeros(zeroPoint):
    Non-zero zero point is not supported.
```

→ 激活必须从 `QUInt8` + 非零 zero_point 改成 **`QInt8` 对称**
（`ActivationSymmetric=True`）。这一点和 ncnn 的口径直接冲突，不能照搬。

### 坑 2 · Gemm 的 bias 会被量化成 int32，TRT 不收

```
[E] ITensor::getDimensions: Error Code 4: API Usage Error
    (fc.bias_DequantizeLinear: input has type Int32 but must have type
     FP8, FP4, Int4, Int8, or UInt8)
```

→ `nodes_to_exclude=["/fc/Gemm"]`。这一层是 512→10，算力占比可以忽略。

### 坑 3 · Conv 的 bias 也是 int32

改成对称量化之后，ORT 把 20 个 Conv 的 bias 也量化成 `int32`（per-channel scale），
TRT 同样拒收。→ `strip_int32_bias_dq()`：把这类 DequantizeLinear 离线算成
浮点常量（scale 和 bias 等长，逐元素相乘）。

```text
strip int32 bias DQ: 移除 20 节点 / 新增 20 浮点常量 / 清理 60 无用 initializer
```

### 量化语义对照表

| | ncnn / ptq-lab | 本目录（TRT） |
|---|---|---|
| 权重 | per-channel int8 对称 | per-channel int8 对称 ✓ 一致 |
| 激活 | per-tensor **uint8 非对称** | per-tensor **int8 对称**（被迫） |
| 校准 | KL 散度 | Entropy（KL）✓ 一致 |

### int8 反而比 fp16 慢，为什么

这颗 GTX 1650 Ti 上，**瓶颈是访存不是算力**。int8 省下的算力，抵不过
每层多出来的 requantize / reformat 通路：

- QDQ 图 TRT 侧 382 个节点 → 建出来的 engine 27 层里混着
  `Reformatting CopyNode` 和每层的 scale 反量化
- 这些额外的 pointwise 通路在 batch 512 的中间 tensor 上是几十 MB 级的读写

所以：**小模型 + 小分辨率 + 中低端 GPU，fp16 优于 int8**。
这个结论不能直接外推到大模型或 A100/H100——那里算力才是瓶颈。

---

## 六、复现步骤

```bash
cd /mnt/d/WorkBuddy/ShowBro/lab/07-tensorrt-deploy
P=~/pyenv-torch/bin/python           # ⚠️ 不是 /usr/bin/python3
export LD_LIBRARY_PATH=/usr/lib/wsl/lib

# 1) 导出两种 ONNX（fp32 动态 batch 是 PTQ 的底稿，fp16 直接给 TRT）
$P scripts/export_onnx.py all

# 2) INT8：ORT 静态 PTQ → QDQ 图 + int32 bias 后处理
$P scripts/quantize_int8.py          # 校准 1000 张 + Entropy

# 3) 评测（先等 GPU 空闲）
$P scripts/wait_idle.py
$P scripts/trt_eval.py --model fp16 --batch 2048 --repeat 5
$P scripts/trt_eval.py --model int8 --batch 2048 --skip 1000 --repeat 5

# 4) 全配置扫描，结果写进 results/sweep.txt
bash sweep.sh 8
```

产物默认落在 `work/`（可用环境变量 `TRT_WORKDIR` 覆盖），已在 `.gitignore` 里忽略。

自检（`smoke.sh`，真的是从空目录跑一遍导出→量化，交付前刚验过）：

```bash
bash smoke.sh
# 期望：fp32 44.7 MB / fp16 22.4 MB / int8 11.3 MB，
#       最后一行 strip int32 bias DQ: 移除 20 节点
```

### `--skip 1000` 为什么不能省

校准集用的是 `test_batch.raw` 的前 1000 张。评测时如果不跳过这 1000 张，
等于**在自己标定过的样本上测精度**——数字会虚高，这是自欺。
int8 的全部评测都必须带 `--skip 1000`。

---

## 七、坑（操作层面的）

| 现象 | 原因 | 处理 |
|---|---|---|
| `ctx.set_binding_shape` 不存在 | TRT 11 改了 API | 用 `ctx.set_input_shape(name, shape)` |
| `plan is None` / quantitative illegal | ONNX 节点问题 | 看 `parser.get_error(i)` 的完整输出 |
| 多轮复测第 2 轮起精度掉到 51.76% | `set_input_shape` 给尾部批次设过 shape 没恢复 | 每轮开头重置成 `batch_size` |
| ` BuilderFlag.FP16` AttributeError | 见第四节 | 靠图里的精度，不用 flag |
| 跑分离散到几十倍 | GPU 被别的进程占 | 见第一节 |

---

## 八、还没做的

- **稀疏化（2:4）**：Turing 不支持，Ampere 起才有 → 这台机器做不了
- **更大的 batch**：受 4 GB 显存限制，5120 已经接近上限且不再收益
- ~~**CUDA Graph**~~ → **已做完，无收益**，见「三·五、CUDA Graph」一节。
  三次测量 0.99× / 0.99× / 0.46×，没有一次更快；batch=512 已经把 launch 开销摊薄了。
  顺带记录了一个危险的假象：relaxed 捕获模式会静默产出**空 graph**，
  看起来 7258×，实际精度是随机猜。
- **TRT 的 C++ 部署**：目前是 python engine + torch tensor 做 buffer
  （这台机器没装 pycuda）。要脱离 torch 需要用 C++ API 重写同一套流程。
  如果真要走 CUDA Graph 的 C++ 版（`cudaStreamBeginCapture` + `enqueueV3`），
  这是唯一还没试的路 —— 但基于 python 侧 0.99× 的实测，预期收益也不乐观。
