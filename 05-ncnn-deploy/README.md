# 05 · ncnn 部署线 —— PTQ 结论能不能活着走出 Python

> ⚠️ **本页命令默认在 WSL Ubuntu 里执行**，工作目录 `/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy`。
> Windows 侧只是数据源（`D:/WorkBuddy/ShowBro/...`）。
>
> 📌 **动手前先读完「一、这条线在解决什么问题」**。不知道要验证什么就跑命令，
> 跑出来的数字没有意义。

## 导航

| 章节 | 讲什么 |
|---|---|
| **一、这条线在解决什么问题** | **⭐ 为什么从 PyTorch 实验转向 C++ 推理引擎** |
| 二、为什么是 ncnn | 选型理由 + 备选对比 |
| 三、阶段 1：已完成 | **主表、结论、意外发现** |
| 四、阶段 2：PTQ 配置迁移 | **桥已打通且三策略忠实：Python 86.99/87.00/86.02% → C++ 87.22/87.02/86.02%** |
| **四·五、阶段 3：sub-8bit 已打通** | **P0 完成：ncnn 跑出 W4A4 78.42%（ptq 78.61%）。非对称激活留 P1** |
| 五、目录与一键复现 | 怎么跑 |
| 六、踩过的坑 | 11 个，都钉在代码里了 |
| 七、三条纪律 | 不可违反 |

---

## 一、这条线在解决什么问题

### 1.1 03-ptq-lab 留下的问题

`03-ptq-lab` 跑了七组实验，拿到四组场景下的 PTQ 精度数据（W4A8 / 混合 4-8bit /
AdaRound）。**这些数字全部是在 PyTorch 里、用自己手写的量化算子测出来的。**

问题在于：**模型最终要跑在 C++ 推理引擎里，不是跑在 PyTorch 里。**
PyTorch 的 `quantize_per_tensor_sym` 和 ncnn 的 INT8 是两套独立实现：

| 差异源 | 说明 |
|---|---|
| 校准方式 | PyTorch 端用 minmax / percentile / entropy，ncnn 用 KL 散度 |
| scale 粒度 | per-tensor / per-channel 的支持程度和默认值不同 |
| 算子融合 | ncnn 会把相邻层的量化边界合并（fused requantize），PyTorch 侧逐层独立 |
| 数值累加 | PyTorch 用 float32 累加，ncnn 用 int32 累加，饱和行为有差异 |

所以「PyTorch 里 W4A8 掉 1.88pp」这个结论，
**不能直接等同于「部署到 ncnn 后掉 1.88pp」**。差多少？没人系统测过。

### 1.2 这条线要回答的问题

> **同一份量化配置，在 PyTorch 仿真和 C++ 推理引擎里，精度差多少？差的原因是什么？**

| 子问题 | 状态 | 产出 |
|---|---|---|
| Q1: ncnn 原生 INT8 vs PyTorch W8A8，差多少？ | ✅ **已答：0.03pp** | 主表 + 归因 |
| Q2: ptq-lab 的量化配置能不能搬到 ncnn？ | ✅ **已答：W8A8 能，差 0.01pp**；4bit 不能（框架无 int4 路径） | 见阶段 2 |
| Q3: 搬过去之后，PyTorch 的结论还成立吗？ | ✅ **三个激活定标策略全部成立**（落差 ≤0.23pp）。此前 percentile/entropy 有 3~17pp 落差，机制已定位并修复 | 见阶段 2 §3.6 |

### 1.3 为什么这件事有价值

1. **工程层（保底价值）**：PyTorch → ONNX → ncnn 的完整部署链路自己打通，
   含 C++ 推理 runner、校准集构造、量化表。AI Infra 岗位的直接证据。
2. **对比层（中坚价值）**：「Python 仿真 vs C++ 引擎」的差异是部署侧的系统性误差，
   公开数据很少。它告诉你「论文数字离真实部署有多远」。
3. **桥接层（上限价值）**：把 ptq-lab 的逐层配置搬进 ncnn，等于做出
   「研究实验配置 → 生产部署配置」的转换器。

### 1.4 明确的边界（先想清楚不做什么）

- ❌ 不重新发明量化算法（03-ptq-lab 已论证：改进空间只有 ~0.25pp）。
- ❌ 不追求 CIFAR-100 的绝对精度（基线重训是冻存状态）。
- ❌ 不碰 GPU 部署。ncnn 的 Vulkan 路径关闭，只测 CPU —— 数字更稳、可复现。
- ❌ **不魔改 ncnn 内核**（除非阶段 2 选择那条路，见阶段 2 的说明）。

---

## 二、为什么是 ncnn

| 候选 | 优势 | 劣势 | 结论 |
|---|---|---|---|
| **ncnn** | 腾讯开源、MIT、无外部依赖、`ncnn2int8` 开箱即用、**量化表是纯文本可读可改** | 主要是 ARM/x86 移动场景 | ✅ **选它** |
| ONNX Runtime | 生态大 | 依赖重；INT8 走 QDQ，和手写 scale 对不齐 | ❌ |
| TensorRT | GPU 最快 | 要 NVIDIA + 闭源 | ❌ |
| MNN | 类似 ncnn | 生态文档弱 | 备选 |

**选 ncnn 的核心理由是第三点**：量化表是纯文本，可以手工编辑 ——
这正是阶段 2 做配置迁移的前提。

---

## 三、阶段 1：ncnn INT8 基线（已完成）

> 2026-10-18 实测，`scripts/run_pipeline.sh all` 一键复现，单线程，全量 10000 张。
> 模型：CIFAR-10 + ResNet-18，FP32 87.16%，权重 `paper-code/cifar_resnet18_50ep.pth`。
> ⚠️ **延迟是机器空闲时测的**，复现前先确认没视频/编译/训练在跑（见 3.6）。精度不受影响。

### 3.1 主表

| # | 口径 | 权重 | 激活 | 在哪测 | 精度 | ms/张 | 大小 |
|---|---|---|---|---|---|---|---|
| 1 | FP32 基线 | fp32 | 不量化 | PyTorch | **87.16%** | — | 44.8 MB |
| 2 | FP32 基线 | fp32 | 不量化 | ncnn | **87.15%** | 12.55 | 43.7 MB |
| 3 | W8A8 / INT8 | int8 per-channel | int8 | PyTorch（ptq-lab e7） | **87.08%** | — | — |
| 4 | W8A8 / INT8 | int8 per-channel | int8 | ncnn（KL 校准表） | **87.05%** | 6.32 | 10.9 MB |
| 5 | 「INT8」不传表 | **fp16** | 不量化 | ncnn | 87.16% | 11.99 | 21.8 MB |

### 3.2 三条结论

| 对照 | 差 | 结论 |
|---|---|---|
| 行2 vs 行1 | **-0.01 pp** | ONNX 导出 + ncnn 转换 + C++ 预处理，整条链路没有引入可测误差 |
| 行4 vs 行3 | **-0.03 pp** | **W8A8 下 Python 仿真与 C++ 引擎高度一致**（远小于 CIFAR-10 测试集 0.33pp 标准误） |
| 行4 vs 行2 | -0.10 pp | ncnn 自己 INT8 掉 0.10pp，和 PyTorch W8A8 掉的 0.08pp 同量级 |

**阶段 1 的核心答案：在这条链路上，「PyTorch 里测的 W8A8 精度」可以直接当作
「部署到 ncnn 后的精度」用，系统偏差约 0.03pp。**

### 3.3 意外发现一：`ncnn2int8` 不传校准表 ≠ INT8

`ncnn2int8 [inparam] [inbin] [outparam] [outbin] [calibration table]`
—— 第 5 个参数可选。**不传也成功退出、也产出 `_int8` 后缀的文件**，但：

| 观察 | 传 KL 表 | 不传表 |
|---|---|---|
| bin 大小 | 10.9 MB（FP32 的 1/4） | 21.8 MB（FP32 的 1/2） |
| Convolution 的 `8=`（int8_scale_term） | 2 / 102 | **缺省 = 0** |
| 耗时 | 6.88 ms/张 | 11.99 ms/张 |
| 精度 | 87.05% | 87.16% |

`int8_scale_term`（param 第 8 号参数，语义从 `src/layer/convolution.cpp` 读出）：

| 值 | 含义 |
|---|---|
| `0` | **没有任何 int8 scale**。权重按 fp16 存，激活不量化 |
| `1`–`100` | 权重 int8 + 输入激活 int8，输出 blob 仍 fp32 |
| `> 100`（如 102） | 上面全部，再加**输出 blob int8**（fused requantize） |

不传表 → `int8_scale_term = 0` → fp16 权重、激活不量化。
**bin 只有 FP32 一半、速度和 FP32 一样、精度和 FP32 一样，因为它根本不是 INT8。**

> **这条对所有用 ncnn 做量化的人都成立**：不传校准表跑一遍 `ncnn2int8`
> 然后报「我们的 INT8 模型没掉点」，是在汇报一个假结果。

### 3.4 意外发现二：fused requantize 改变了量化边界

传表后 ncnn2int8 日志大量出现 `fuse_requantize /layerX/conv1 /layerX/conv2`。
对应到 param：`/conv1/Conv` 是 `8=2`（输出 fp32），
`/layer1/layer1.0/conv1/Conv` 是 `8=102`（输出也是 int8）。

ResNet 的 BasicBlock 是 `conv → relu → conv → add → relu`。第二个 conv 的输出
直接进残差 add，若是 int8，add 就能在 int8 域做，省两次 requantize。

**这是「引擎侧融合」和「Python 侧仿真」的第一个结构性差异**：ptq-lab 的
`QuantConv2d` 逐层独立，ncnn 会合并相邻层的量化边界。
**阶段 2 搬逐层配置时，不能直接照搬层名。**

### 3.5 意外发现三：小模型上多线程是负优化

同一份 INT8 模型、同样 2000 张：

| 线程 | ms/张 | 张/秒 |
|---|---|---|
| 1 | 6.56 | 152.4 |
| 4 | 4.63 | 215.8 |
| 12 | **19.84** | 50.4 |

12 线程比单线程慢 3 倍。32×32 图上每层计算量小，OpenMP 进出并行区的固定开销
盖过并行收益。**`run_pipeline.sh` 默认 `NCNN_THREADS=1`。**
换大图（224×224）结论会反过来。

### 3.6 意外发现四：延迟数字在负载机器上不可复现

2026-10-18 复测时发现：**同一份 INT8 模型、同样的命令，耗时是主表的 2 倍。**

| 测量时间 | 机器状态 | INT8 ms/张 |
|---|---|---|
| 主表（早前） | 空闲 | **6.32** |
| 复测 10000 张 | B 站视频 + Qoder + WorkBuddy 在跑 | 12.87 |
| 复测 2000 张 run1 | 同上 | 10.24 |
| 复测 2000 张 run2 | 同上 | 8.54 |

连续两次差 20%，整体是空闲值的 1.4–2 倍。查了 WSL 内负载（load 0.72，无重进程）、
电源计划（高性能，非降频）—— **慢在 Windows 侧的其它进程抢 CPU**。

**结论（方法论，比数字本身重要）**：

1. **精度不受影响**。复测 87.15% / 87.05%，与主表一字不差。精度是确定性的，
   延迟不是。
2. **延迟必须绑定机器状态才有意义**。本项目的延迟数字全部是「机器空闲时」测得，
   复现时先确认没有视频播放 / 编译 / 训练在跑。
3. **`ms/张` 的口径**：含 `make_input` 归一化 + `create_extractor` + 前向，
   不含模型加载（加载在构造函数里，计时外）。是「端到端单张推理」口径，
   不是「纯前向计算」口径 —— 报数字时要说清。

---

## 四、阶段 2：PTQ 配置迁移进 ncnn（桥已打通）

### 4.1 核心结论：能迁，不用改 ncnn

原以为必须改框架才能把 ptq-lab 的配置搬进来。**这个判断是错的。**
查证发现 ncnn 的量化表里存的是具体 scale 数值，而 `ncnn2int8`
对表是**照单执行、一个 scale 都不重算**（源码 + 实测双重证实，
见 `docs/阶段2-PTQ配置迁移到ncnn.md`）。

| 激活定标策略 | Python 侧（ptq-lab） | C++ 侧（ncnn 桥） | 差 |
|---|---|---|---|
| minmax | 86.99% | **87.22%** | **+0.23 pp** |
| percentile | 87.00% | **87.02%** | +0.02 pp |
| entropy | 86.02% | **86.02%** | **0.00 pp** |
| （参照）ncnn 自己的 KL 校准 | — | 87.05% | — |

> ⚠️ 上表是 **2026-10-18 修好码空间换算之后**的数字。
> 修正前三策略是 87.17% / **83.88%** / **68.88%** —— 桥一直把非对称 255 码的
> scale 原样灌进对称 ±127 码的引擎，注入值大了 255/127 倍，
> 三个策略里只有 minmax 扛得住那份溢出。
> 完整定位过程（三层证据 + 源码 + 因果回验）见 `docs/阶段2-PTQ配置迁移到ncnn.md` §3.6。

> ⚠️ **跑 ptq-lab 必须加 `PTQLAB_REFERENCE=0`**。默认是 1，会用
> `reference/quant_ref.py` 顶替 `src/quant.py` 的实现，而 reference 的
> `scale_entropy` 缺截断约束——实测把 conv1 的激活截掉 84.89%，
> entropy 两边都只有 38%。言瑞自己的实现（测试 15/15 全过）修好了它。
> 完整证据链见 `docs/阶段2-PTQ配置迁移到ncnn.md` §3。

**桥是忠实的**（minmax 差 0.18pp）。三个结论和两个坑详见阶段 2 文档。

### 4.2 4bit 不可行 —— 已实测确认

原计划是把 ptq-lab e7 的「10 层保 8bit、其余 4bit」搬进 ncnn。**先做了可行性
验证，结论是不行：**

| 证据 | 说明 |
|---|---|
| `src/layer/convolution.cpp` / `innerproduct.cpp` | 无任何 int4 / 4bit / nf4 代码路径 |
| `src/modelbin.h:22` | 注释写了 `4 = weight block quantize int4` / `6 = int6` / `8 = int8` |
| `src/modelbin.cpp` `ModelBinFromDataReader::load` | 只认 5 个 magic tag：fp16 `0x01306B47`、bf16 `0x01348B83`、int8 `0x000D4B38`、fp32+scale `0x0002C056`、legacy 256 项索引表。**没有 block quantize 的分支** |
| `src/layer/*.cpp` | 没有任何层用 `load(w, 4)` 这类调用 |
| `tools/quantize/ncnn2int8.cpp` | 没有位宽参数 |
| `tools/quantize/ncnn2table.cpp` | `method` 只有 kl/aciq/eq/percentile，无位宽选项 |

`modelbin.h` 里的 int4/int6 是**文档里有、实现里没有**的格式（可能是给外部
fork 或未来版本留的接口）。**stock ncnn 表达不了 4bit。**

### 4.2 可行的替代路线：INT8 + FP16 逐层回退（已实测）

`ncnn2int8` 的 `quantize_convolution()` 第 169-171 行：

```cpp
std::map<std::string, ncnn::Mat>::iterator iter_data =
    blob_int8scale_table.find(layers[i]->name);
if (iter_data == blob_int8scale_table.end())
    continue;          // ← 层不在表里 → 跳过量化，保持 fp16
```

**从校准表里删掉某层的两个条目（`<层名>` 和 `<层名>_param_0`），该层就保持 FP16。**
实测（删 `/conv1/Conv`）：

| 模型 | conv1 | 精度 | ms/张 | bin |
|---|---|---|---|---|
| 全 INT8 | int8（`8=102`） | 87.05% | 6.32 | 10948 KB |
| conv1 保 FP16 | 无 `8=` 参数 | **87.13%** | 7.72 | 10957 KB |

**保护最敏感的一层就找回 0.08pp，代价是慢 22%。**

这就是 ncnn 版的混合精度，而且和 ptq-lab 的敏感度分析（e4）天然衔接：
e4 已经算出了「哪些层最不能量化」，把 top-K 从表里删掉就是。

### 4.3 下一步：路线已定，不再是待选

阶段 2 的核心问题（配置能不能搬）已经 answered。剩下的按优先级：

| 事项 | 内容 | 状态 |
|---|---|---|
| **A. 敏感层保护** | 用 e4 敏感度排序，把 top-K 层保成 FP16，扫 K=0/1/2/4/8/12 | ✅ **已完成，结论是"不划算"** |
| **C. 码空间换算**（已完成） | 桥漏了 asym 255 码 → sym ±127 码的换算，导致非 minmax 策略掉 3~17pp。**已定位 + 已修**（`make_table.py` 的 `SYM_CODE_FIX = 127/255`），修完三策略落差 ≤0.23pp | ✅ **已完成** |
| **B. 重新审 e7 的 entropy 结论** | ptq-lab 默认跑 reference 实现，其 `scale_entropy` 缺截断约束，conv1 激活被截 84.89%。言瑞自己的实现（15/15 测试通过）修好了：W8A8+entropy 从 38.19% → 86.02%。e7 结果 6「混合+AdaRound+A4 entropy」= 85.45% 用的是坏口径，**必须用 `PTQLAB_REFERENCE=0` 重跑 e7** | 待做 —— ⚠️ **这是 Python 研究侧的活，用户已叫停那条线，做之前先问** |
| ~~给 ncnn 加 int4~~ | ~~自己写 block quantize magic tag + int4 解包 + GEMM kernel~~ | **已放弃**：2-4 周改框架，且 W8A8 的桥已经打通，边际价值不抵成本 |

### A 的结果（负面，但很干净）

| K（保护层数） | 精度 | ms/张 | bin/KB | 仍量化 |
|---|---|---|---|---|
| 0 | 87.17% | 8.27 | 10948 | 20 |
| 1 | **87.20%** | 9.12 | 11523 | 19 |
| 2 | 87.18% | 8.57 | 11530 | 18 |
| 4 | 87.15% | 14.35 | 14982 | 16 |
| 8 | 87.13% | 13.22 | 18578 | 12 |
| 12 | 87.05% | 13.77 | 21180 | 8 |

**精度全程平坦（波动 0.15pp，远小于 0.33pp 标准误），体积却单调涨 93%。**
原因是 W8A8 已经追平 FP32（87.17% vs 87.15%），**没有精度余量可回收**。
Pareto 前沿就是 K=0 一个点。

⚠️ 只在 CIFAR-10 + ResNet-18 这个组合上成立，别外推成"混合精度没用"。
详见 `docs/阶段2b-敏感层保护Pareto.md`。

### C 的结果：机制定位，并修掉了桥的一个真 bug

之前 percentile 掉 3.12pp、entropy 掉 17.14pp，一直没解释。**现在有答案了：
不是引擎的问题，是桥少做了一步换算。**

ptq-lab 的激活是**非对称 255 码**（`[0,255]` + zero_point），
ncnn 的 int8 是**对称 ±127 码、无 zero_point**（`src/layer/quantize.cpp:30-36`
的 `float2int8` 截在 ±127）。桥注入 `1/a_scale = 255/量程`，
而 ncnn 的表示区间被钉在以 0 为中心的 `[-127/s, +127/s]`——
**每层只能表示到量程的一半，桥自带 2 倍溢出。**

三组证据（`scripts/overflow_probe.py` 一次跑完）：

| 证据 | 做法 | 结果 |
|---|---|---|
| A 溢出倍数 | 纯读 json 比量程 | percentile 中位 1.99× / 最坏 2.58×；entropy 3.06× / 4.81× |
| B 溢出容忍度 | minmax 表注入 scale × f，跑 C++ | f≤2 全在 86.90~87.22%（噪声级），**f=4→80.37%**，f=8→30.97% |
| C 因果回验 | 只改码空间换算这一处（修正前基准） | percentile 83.88%→**87.03%**；entropy 68.88%→**86.02%** |

**B 的形状是关键**：缩小是免费的，`f` 从 1 缩到 0.25 只掉 0.32pp（噪声级）；
放大是致命的，放大 4 倍直接掉 6.85pp。这是饱和截断的签名，不是舍入误差的。
（修正前用同一脚本扫出的曲线与之**正好错开一格 2×**，
等于从实测上确认了修正量恰好是 255/127。）

修复：`make_table.py` 注入时乘 `SYM_CODE_FIX = 127/255`
（`--no-sym-fix` 可复现旧行为）。修完端到端重跑，entropy 的 C++ 86.02%
和 Python 侧**一字不差**。

详见 `docs/阶段2-PTQ配置迁移到ncnn.md` §3.6。

**诚实的边界**：目前搬进去的是 **W8A8**（8bit 权重 + 8bit 激活）。
ptq-lab 的 W4A8 / 混合 4-8bit **搬不了**——不是工具问题，是 stock ncnn
没有 int4 代码路径。所以不能说「我把混合精度搬过来了」，
能说的是「W8A8 配置完整迁移，且研究侧校准方法在引擎里可 A/B」。

---

## 五、目录与一键复现

```
05-ncnn-deploy/
├── README.md            ← 本文件
├── docs/
│   ├── 阶段1-ncnn-INT8基线.md       ← 阶段 1 完整记录（含 10 个坑）
│   ├── 阶段2-PTQ配置迁移到ncnn.md   ← 阶段 2：桥怎么通的、坑、结论
│   ├── 阶段2b-敏感层保护Pareto.md   ← 阶段 2b：混合精度值不值（负面结果）
│   └── 阶段3-扩展ncnn量化语义-设计.md ← 阶段 3：sub-8bit + 非对称激活（设计稿）
│   └── 阶段3b-P0实测-int4数值语义.md  ← 阶段 3b：P0 实测，W4A4 78.42%
├── scripts/
│   ├── run_pipeline.sh   ← 一条龙：raw → export → convert → calib → int8 → eval
│   ├── export_onnx.py    ← PyTorch → ONNX（opset 11，带 onnxruntime 自检）
│   ├── dump_cifar_raw.py ← pickle → 裸二进制（C++ runner 的数据源）
│   ├── dump_calib_npy.py ← 校准集 → npy（ncnn2table 的输入）
│   ├── make_table.py     ← ptq-lab 的 scale json → ncnn 量化表（阶段 2 的桥）
│   ├── rebuild_ptq_tables.sh ← 三策略重建表 + 端到端评测（一条出全表）
│   ├── pareto_scan.py    ← 敏感层保护 K 扫描（阶段 2b，支持 --keep-fp16）
│   ├── overflow_probe.py ← 激活 scale 溢出探针（阶段 2 C：A/B/C 三组证据）
│   ├── bench.py          ← **模型体检**：一条命令跑完全部模型并判定回归
│   └── bench_models.json ← 体检清单（期望精度 + 容差，改这里不用改代码）
├── cpp/                  ← C++ runner（CMake，3 个模块）
│   ├── cifar10.h/.cpp    ← 只读 CIFAR-10 裸二进制
│   ├── evaluator.h/.cpp  ← 只做「模型 × 数据集 → 精度/延迟」
│   └── main.cpp          ← 只解析参数
├── results/              ← 扫描结果 json（小，进 git）
├── models/               ← onnx / param / bin（大文件不进 git）
├── tables/               ← 量化表（纯文本，进 git）
├── calib/                ← 校准集 npy（不进 git）
└── data_raw/             ← 裸二进制测试集（不进 git）
```

### 模型体检（改完 ncnn 必跑）

改量化通路之后，要确认 4 个模型都没回归。手工跑 4 次 eval + 肉眼比对有两个问题：
慢（全量 5–10 分钟）、会漏（78.42 → 78.19 用眼睛看不出来）。所以做成了带断言的命令：

```bash
python3 scripts/bench.py --quick 1000      # 快速回归，约 35 秒
python3 scripts/bench.py                   # 全量 10000 张，出正式数字
python3 scripts/bench.py --only W4A4-asym  # 只跑一个
```

输出（全量 10000 张，单线程，5m57s）：

```
tag                     acc   expect        Δ   ms/img     判定
--------------------------------------------------------------
FP32                 87.15%   87.16%   -0.01    16.08   PASS
W8A8                 87.05%   87.05%   -0.00     6.60   PASS
W4A4-sym             78.42%   78.42%   +0.00     6.65   PASS
W4A4-asym            83.92%   83.92%   +0.00     6.67   PASS
--------------------------------------------------------------
全部通过
```

**Δ 全部 ≤ 0.01pp** —— 说明期望值配置准确、模型结果可复现、工具解析无误。

**超容差即 FAIL，退出码非 0**，可以直接放进脚本或 CI。`--quick` 精度会抖 ±1pp，
判定自动放宽到 ±1.2pp；全量模式容差 ±0.2pp。

期望值不硬编码在代码里，改 `scripts/bench_models.json` 即可。

### 环境准备（一次性）

```bash
# WSL 里。ncnn 需要 protobuf 才能编 onnx2ncnn，sudo 要密码所以装到用户目录
git clone --depth 1 -b v3.21.12 https://github.com/protocolbuffers/protobuf.git ~/ncnn-build/protobuf
cd ~/ncnn-build/protobuf && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/local -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_BUILD_SHARED_LIBS=OFF
make -j12 && make install

# ncnn 本体。注意 tools/CMakeLists.txt 要手动补一行 add_subdirectory(onnx)
git clone --depth 1 https://github.com/Tencent/ncnn.git ~/ncnn-build/ncnn
cd ~/ncnn-build/ncnn && sed -i 's|^add_subdirectory(darknet)|add_subdirectory(darknet)\nadd_subdirectory(onnx)|' tools/CMakeLists.txt
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DNCNN_BUILD_TOOLS=ON -DNCNN_BUILD_EXAMPLES=OFF \
      -DNCNN_VULKAN=OFF -DProtobuf_INCLUDE_DIR=$HOME/local/include \
      -DProtobuf_LIBRARY=$HOME/local/lib/libprotobuf.a \
      -DProtobuf_PROTOC_EXECUTABLE=$HOME/local/bin/protoc
make -j12

# C++ runner
cd /mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cpp && mkdir build && cd build && cmake .. && make -j12
```

### 一键跑

```bash
cd /mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy
./scripts/run_pipeline.sh all        # 全流程，约 4 分钟
./scripts/run_pipeline.sh eval       # 只测精度（已有模型时）

./scripts/rebuild_ptq_tables.sh      # 三策略重建表 + 端到端评测（阶段 2）
python3 scripts/overflow_probe.py --skip-run   # 溢出倍数，秒出，不跑引擎
```

---

## 六、踩过的坑（都已在代码/脚本里钉住）

| # | 坑 | 现象 | 根因 | 现在怎么防 |
|---|---|---|---|---|
| 1 | CIFAR-10 磁盘文件不是裸二进制 | 「文件大小 31035526 不是 3073 的整数倍」 | `cifar-10-batches-py/` 是 Python pickle 的 dict，label 在 `labels` 列表里 | `dump_cifar_raw.py` 先转一层，用 torchvision 加载保证同源 |
| 2 | `ncnn2table` 报 `expect 1 shapes, but got 0` | 表生成不出来 | shape **不从 param 的 Input 层读**，必须命令行 `shape=[w,h,c]`（`ncnn2table.cpp:2403`） | 脚本固定带 `shape=[32,32,3]` |
| 3 | 给 param 的 Input 层补 `0=32 1=32 2=3` 没用 | 照样报错 | 见 #2，补错了地方 | param 保持原样 |
| 4 | `onnx2ncnn` 编不出来 | `tools/onnx/` 不在构建里 | 这个版本 ncnn 把 `tools/onnx` 从 `tools/CMakeLists.txt` 摘掉了（官方转向 pnnx） | 手动 `add_subdirectory(onnx)`；protobuf 源码装 `~/local` |
| 5 | `undefined reference to GOMP_parallel` | 链接失败 | ncnn 用 OpenMP，只链 pthread 不够 | CMakeLists 里 `find_package(OpenMP REQUIRED)` |
| 6 | **复用 Extractor 后精度崩到 9.95%** | 2000 张只对 199 张，耗时 0.002ms/张 | `Extractor::extract()` 只在目标 blob `dims == 0` 时才跑前向（`net.cpp:3254`），复用直接返回上次结果；`input()` 不清其他 blob | 每张图新建 Extractor，`evaluator.cpp` 写了长注释 |
| 7 | 多线程反而慢 3 倍 | 见 3.5 | OpenMP 并行区开销 | 默认单线程 |
| 8 | 复测延迟是主表 2 倍 | 同命令同模型，12.87 vs 6.32 ms/张 | **不是代码问题**：Windows 侧其它进程抢 CPU（B 站视频 + 3 个 IDE 进程）。精度一字不差，只有延迟变 | 延迟数字绑定「机器空闲」状态才可比；口径说明见 3.6 |
| 9 | **ptq scale 灌进 ncnn 表，精度掉到 10%** | 不报错、能跑、速度正常，只有精度是垃圾 | **两边 scale 约定互为倒数**：ptq 是 `absmax/127`（除数），ncnn 是 `127/absmax`（乘数，`ncnn2table.cpp:580`） | `make_table.py` 写入前取倒数；文档 §2 有完整证据链 |
| 10 | PyTorch→ncnn 层名映射错 | 断言拦下，未产出废表 | `layer1.0` 在 ncnn 是两级目录；`downsample.0` 会被展开成 `downsample/downsample.0` | 映射结果全部对照 param 断言，有一个不在就中止 |
| 11 | **漏了 asym→sym 的码空间换算** | minmax 忠实（87.17%），percentile 掉 3.12pp、entropy 掉 17.14pp；不报错、能跑、只有精度不对 | ptq 激活是非对称 **255 码**，ncnn int8 是**对称 ±127 码且无 zp**（`quantize.cpp:30-36`）。注入 `255/量程` 后 ncnn 只能表示到量程一半，**桥自带 2× 溢出** | `make_table.py` 注入时乘 `SYM_CODE_FIX = 127/255`；`--no-sym-fix` 可复现旧行为。定位过程与三组证据见阶段 2 文档 §3.6 |

> 坑 #11 和 #9 是同一类：**两侧约定不一致**。
> #9 是"除数 vs 乘数"，#11 是"255 码 vs ±127 码"。
> 以后再往表里灌任何数值，先问一句"这个数在对方那儿的语义是什么"。

---

## 七、三条纪律

1. **FP32 不对齐就不要测 INT8**。ncnn FP32 和 PyTorch FP32 差超 0.1pp，
   说明 ONNX 导出或算子转换就有问题，此时 INT8 的数字全是噪声。

2. **每个数字都要能说出「和哪个 Python 数字比」**。孤立的 C++ 精度没有意义。
   当前对照关系见 3.1 主表，新增口径先补进行 2/3 再跑。

3. **框架能力先验证再动手**。4bit 这件事就是教训 ——
   先花半小时读源码确认「ncnn 有没有这个能力」，
   比写三百行转换代码最后发现框架不支持要划算得多。

---

## 八、ncnn + CUDA 路线（已跑通，结论：走不到 500×）

见 [`cuda/README.md`](cuda/README.md)。

一句话总结：

- 修掉了「CUDA 层建了却没被调用」这个真 bug（开关是 Extractor 里塞 Mat 还是 CudaMat，
  `opt.use_cuda_compute` 是个没人读的死开关）。修完 fp32 GPU 精度 88.00%，GPU 是真的在跑。
- 但实测天花板是 **0.8×**（唯一精度正确的配置，14.89 ms/img，比 CPU 基线的 11.6 还慢）；
  显存一复用能到 1.5×，但精度掉到 12%（随机猜）。
- int8 的 CUDA 路径结果是错的（9.00%），CPU 通路同一模型是 87%。
- 原因：fork 的卷积是手写三重循环（无 cuDNN/TensorCore），ncnn 的 Mat 没有 batch 维，
  且显存复用会让精度单调劣化。

**500× 的目标在 `../07-tensorrt-deploy/` 已经达成，而且两种口径下都达成**：

| 口径 | fp16 | int8 | 说明 |
|---|---|---|---|
| 新（2026-10-25） | **0.01068 ms/img = 1086×** | **0.00744 = 1559×** | preboost(100)+steady，clock 1755–1920 MHz |
| 旧（2026-10-20） | 0.0149 = 778× | 0.0189 = 614× | idle 门控，clock 1345–1455 MHz（**偏低**） |

两者差的不是代码而是 GPU clock 状态 + 1×1 折叠。完整七路线同口径对照、以及
「500× 依赖哪些条件」见 `../08-report/README.md`（§二 口径、§2.6 达成条件）。
