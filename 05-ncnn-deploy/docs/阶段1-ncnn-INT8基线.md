# 阶段 1 · ncnn INT8 基线 —— Python 仿真与 C++ 引擎差多少

> 2026-10-18 实测。模型：CIFAR-10 + ResNet-18（FP32 87.16%，权重 `paper-code/cifar_resnet18_50ep.pth`）。
> 全部数字由 `scripts/run_pipeline.sh all` 一键复现，单线程，全量 10000 张测试图。
> ⚠️ **延迟数字是「机器空闲时」测的**，复现前先确认没有视频播放 / 编译 / 训练在跑，
> 否则延迟不可比（实测负载下同样模型慢 1.4–2 倍，见 4.5）。精度不受影响。

## 1. 主表

| # | 口径 | 权重 | 激活 | 在哪测 | 精度 | ms/张 | 模型大小 |
|---|---|---|---|---|---|---|---|
| 1 | FP32 基线 | fp32 | 不量化 | **PyTorch** | **87.16%** | — | 44.8 MB |
| 2 | FP32 基线 | fp32 | 不量化 | **ncnn** | **87.15%** | 12.55 | 43.7 MB |
| 3 | W8A8 / INT8 | int8 per-channel | int8 | **PyTorch**（03-ptq-lab e7） | **87.08%** | — | — |
| 4 | W8A8 / INT8 | int8 per-channel | int8 | **ncnn**（KL 校准表） | **87.05%** | 6.32 | 10.9 MB |
| 5 | 「INT8」不传表 | **fp16** | 不量化 | ncnn | 87.16% | 11.99 | 21.8 MB |

三条对照关系，每条都指向一个明确结论：

| 对照 | 差 | 结论 |
|---|---|---|
| 行2 vs 行1 | **-0.01 pp** | ONNX 导出 + ncnn 转换 + C++ 预处理，整条链路没有引入可测误差 |
| 行4 vs 行3 | **-0.03 pp** | **W8A8 下 Python 仿真与 C++ 引擎高度一致**（远小于 CIFAR-10 测试集 0.33pp 的标准误） |
| 行4 vs 行2 | -0.10 pp | ncnn 自己的 INT8 掉了 0.10pp，和 PyTorch W8A8 掉的 0.08pp 同一量级 |

**阶段 1 的核心答案：在这条链路上，「PyTorch 里测的 W8A8 精度」可以直接当作
「部署到 ncnn 后的精度」用，系统偏差约 0.03pp。**

## 2. 意外发现：`ncnn2int8` 不传校准表 ≠ INT8

行 5 是这条线上最值得记下来的一条，因为它**不报错，只会静默给你一个错的模型**。

`ncnn2int8` 的用法是：

```
ncnn2int8 [inparam] [inbin] [outparam] [outbin] [calibration table]
```

第 5 个参数是可选的。不传，它照样成功退出、照样产出 `_int8` 后缀的 param/bin。
但看产出：

| 观察 | 传 KL 表 | 不传表 |
|---|---|---|
| bin 大小 | 10.9 MB（FP32 的 1/4） | 21.8 MB（FP32 的 1/2） |
| Convolution 的 `8=`（int8_scale_term） | 2 / 102 | **缺省，即 0** |
| 耗时 | 6.32 ms/张 | 11.99 ms/张 |
| 精度 | 87.05% | 87.16% |

`int8_scale_term`（param 第 8 号参数）的语义，从 `src/layer/convolution.cpp` 读出来的：

| 值 | 含义 |
|---|---|
| `0` | **没有任何 int8 scale**。权重按 fp16 存，激活不量化 |
| `1`–`100` | 权重 int8 + 输入激活 int8，输出 blob 仍是 fp32 |
| `> 100`（如 102） | 上面全部，再加**输出 blob int8**（fused requantize） |

不传表 → `int8_scale_term = 0` → 权重 fp16、激活不量化。
**bin 只有 FP32 的一半（fp16 权重）、速度和 FP32 一样、精度和 FP32 一样**，
因为它根本不是 INT8 推理。名字叫 `_int8` 纯属误导。

**这条对所有用 ncnn 做量化的人都成立**：不传校准表跑一遍 `ncnn2int8` 然后报
「我们的 INT8 模型精度 87.16% 没有掉点」，是在汇报一个假结果。

## 3. `int8_scale_term > 100` 与 fused requantize

传了 KL 表之后，ncnn2int8 的日志里大量出现：

```
fuse_requantize /layer2/layer2.1/conv1/Conv /layer2/layer2.1/conv2/Conv
```

对应到 param 上，`/conv1/Conv` 是 `8=2`（输出 fp32），
而 `/layer1/layer1.0/conv1/Conv` 是 `8=102`（输出也是 int8）。

含义：ResNet 的 BasicBlock 是 `conv → relu → conv → add → relu`。
第二个 conv 的输出直接进残差 `add`，如果它是 int8，add 就能在 int8 域做，
省掉两次 requantize。ncnn2int8 自动挑出了这些位置。

**这就是「引擎侧融合」和「Python 侧仿真」的第一个结构性差异**：
PyTorch 的 `QuantConv2d` 是逐层独立的（`e7` 里每层自己算 a_scale / a_zero），
ncnn 会把相邻层的量化边界合并。阶段 2 做配置迁移时，**逐层位宽分配不能直接
照搬层名**，要先搞清楚哪些层被 ncnn 合并了。

## 4. 线程数：小模型上多线程是负优化

同一份 INT8 模型、同样 2000 张图：

| 线程 | ms/张 | 张/秒 |
|---|---|---|
| 1 | 6.56 | 152.4 |
| 4 | 4.63 | 215.8 |
| 12 | **19.84** | 50.4 |

12 线程比单线程慢 3 倍。原因很直接：32×32 的图上每层的计算量很小，
OpenMP 每次进出并行区的固定开销（线程唤醒、barrier 同步）盖过了并行收益，
线程数一多还会互相抢。

**所以 `run_pipeline.sh` 默认 `NCNN_THREADS=1`。**
这不是懒，是这个模型规模下的最优解。换大图（224×224）结论会反过来。

## 4.5 延迟数字在负载机器上不可复现（2026-10-18 复测发现）

复测阶段 1 主表时发现：**同一份 INT8 模型、同样的命令，耗时是主表的 2 倍。**

| 测量时间 | 机器状态 | INT8 ms/张 |
|---|---|---|
| 主表（早前） | 空闲 | **6.32** |
| 复测 10000 张 | B 站视频 + Qoder + WorkBuddy 在跑 | 12.87 |
| 复测 2000 张 run1 | 同上 | 10.24 |
| 复测 2000 张 run2 | 同上 | 8.54 |

排查过程：WSL 内 `uptime` load 0.72、`free` 余 15 GB、`ps` 无重进程；
`powercfg /getactivescheme` 是高性能（非降频）。
**慢在 Windows 侧的其它进程抢 CPU** —— B 站视频播放 + 三个 IDE 进程。

结论（按重要性排）：

1. **精度不受影响，延迟受影响**。复测 87.15% / 87.05%，与主表一字不差。
   前向是确定性计算；延迟取决于CPU 有没有被别人占。
2. **延迟必须绑定机器状态才有意义**。本项目所有延迟数字都是「机器空闲时」测得。
   复现前先确认没有视频播放 / 编译 / 训练在跑，否则数字没有可比性。
3. **`ms/张` 的口径**：计时循环含 `make_input`（归一化）+ `create_extractor`
   + 前向；不含模型加载（加载在 `Evaluator` 构造函数里，计时起点之前）。
   这是「端到端单张推理」口径，不是「纯前向计算」口径。

## 5. 踩到的坑（都已在代码/脚本里钉住）

| # | 坑 | 现象 | 根因 | 现在怎么防 |
|---|---|---|---|---|
| 1 | CIFAR-10 磁盘文件不是裸二进制 | 报「文件大小 31035526 不是 3073 的整数倍」 | `cifar-10-batches-py/` 是 Python pickle 的 dict，label 在 `labels` 列表里，不在记录前缀 | `dump_cifar_raw.py` 先转一层，且用 torchvision 加载保证和 PyTorch 同源 |
| 2 | `ncnn2table` 报 `expect 1 shapes, but got 0` | 表生成不出来 | shape **不从 param 的 Input 层读**，必须命令行 `shape=[w,h,c]` 传（`ncnn2table.cpp:2403`） | 脚本里固定带 `shape=[32,32,3]` |
| 3 | 给 param 的 Input 层补 `0=32 1=32 2=3` 没用 | 同上，照样报错 | 见 #2，补错了地方 | param 保持原样，没打补丁 |
| 4 | `onnx2ncnn` 编不出来 | `tools/onnx/` 不在构建里 | 这个版本 ncnn 已把 `tools/onnx` 从 `tools/CMakeLists.txt` 摘掉（官方转向 pnnx） | 手动 `add_subdirectory(onnx)`；另需 protobuf（sudo 要密码，源码装到 `~/local`） |
| 5 | `undefined reference to GOMP_parallel` | 链接失败 | ncnn 用 OpenMP，只链 pthread 不够 | CMakeLists 里 `find_package(OpenMP REQUIRED)` |
| 6 | **复用 Extractor 后精度崩到 9.95%** | 2000 张只对 199 张，耗时 0.002ms/张 | `Extractor::extract()` 只在目标 blob `dims == 0` 时才跑前向（`net.cpp:3254`），复用直接返回上一次结果；`input()` 不清其他 blob | 每张图新建 Extractor，`evaluator.cpp` 里写了长注释钉住 |
| 7 | 多线程反而慢 3 倍 | 见第 4 节 | OpenMP 并行区开销 | 默认单线程 |
| 8 | 复测延迟是主表 2 倍 | 同命令同模型，12.87 vs 6.32 ms/张 | **不是代码问题**：Windows 侧其它进程抢 CPU（B 站视频 + 3 个 IDE 进程）。精度一字不差，只有延迟变 | 延迟数字绑定「机器空闲」状态才可比；口径说明见 4.5 |

## 6. 还没做的事（阶段 2 的输入）

- ✅ **4bit 可行性已验证，结论是不行**（2026-10-18 补）。
  `src/modelbin.h:22` 的注释确实声明了 `4 = weight block quantize int4`，
  但 `ModelBinFromDataReader::load` 只认 5 个 magic tag（fp16 / bf16 / int8 /
  fp32+scale / legacy 256 项索引表），**没有 block quantize 的分支**；
  没有任何层调用 `load(w, 4)`；`ncnn2int8` 没有位宽参数；
  `ncnn2table` 的 `method` 只有 kl/aciq/eq/percentile。
  **stock ncnn 表达不了 4bit**，这是「文档里有、实现里没有」的格式。
  完整证据链见 README 4.1，替代路线见 README 4.2。
- `pnnx` 路径没试。官方文档已推荐 pnnx 替代 onnx2ncnn，但它要 libtorch，
  构建成本高一个量级。当前 onnx2ncnn 对 ResNet-18 完全够用，暂时不投入。
- 其他校准方法（`method=aciq/eq/percentile`）没测。KL 已够对齐 Python 侧，
  等阶段 2 需要精细控制 scale 时再补。
