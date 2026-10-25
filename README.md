# PTQ int8 部署实验室（代码 + 核心文档）

把 CIFAR ResNet / MobileNet 做完训练后量化（PTQ）、导出、C++/TensorRT 部署，并记录**可复现**的精度/延迟、roofline 与负结果。

> 本仓库只保留**代码 + 核心思路文档**；逐轮实验日志、中间结果 JSON、benchmark 原始数据、模型权重/ONNX/engine 均已剔除（避免仓库膨胀）。

## 主结果（GTX 1650 Ti，CIFAR-10，10000 张）

| 路径 | 精度 | 延迟 | 相对 ncnn CPU fp32（11.6 ms/img） |
|---|---|---|---|
| TensorRT int8 | 87.22% | **0.00744 ms/img** | **1559×** |
| TensorRT fp16 | 87.18% | 0.01068 | 1086× |
| TensorRT fp32 | 87.16% | 0.02477 | 468× |
| 自研 C++/CUDA int8（按层混合） | 87.01% | 0.0248 | 468× |

- 位置分解（3×3 pad=1 → 2×2 conv + DepthToSpace）：int8 同场 **−20%**（0.0080→0.0064，最好 0.00609）
- MobileNetV2 混合精度：全量 10k 掉点 **0.92pp**（W4 per-channel + 24 层 W8）
- 口径与限制见下文与 `ROOFLINE-int8.md` / `PTQ-RETRO.md`

## 核心思路（先读这三份）

1. **`PTQ-RETRO.md`**——自研引擎 vs TensorRT 的复盘：四个根因、十条负结果、三条可迁移教训
2. **`ROOFLINE-int8.md`**——DP4A 理论上限（15.34 TOPS）、实测差距、Amdahl 分解、clock/功耗墙测量协议
3. **`08-report/README.md`**——同口径主表、三张图、可溯源自检

两个关键发现：

- 32×32 输入的 ResNet-18，layer4 空间已是 1×1，却仍挂 3×3 pad=1 卷积 → 8/9 乘法在乘零；切中心 tap 改 1×1（语义恒等，logits 差 2.4e-7），int8 0.0189→0.00744
- 输入空间 ≤ kernel 的卷积按输出位置改写成「一个 2×2 conv（out_channels×4）+ DepthToSpace」，FLOPs −2.25×、kernel 数不增；收益集中在 `layer4.0/conv1`（11.6×），layer3 因 shuffle 税净亏

## 目录（只列保留内容）

| 路径 | 内容 |
|---|---|
| `03-ptq-lab/` | PTQ 算法原型：per-channel/per-group、percentile/entropy/MSE 校准、评测 |
| `04-libtorch-cuda/` | libtorch C++ CUDA 推理线 |
| `05-ncnn-deploy/` | ncnn CPU/CUDA 部署与验收 harness（含 `cuda/` fork 集成） |
| `06-gpu-int8/` | 自研 C++/CUDA int8 引擎：im2col/GEMM/按层混合/roofline 探针 |
| `07-tensorrt-deploy/` | TensorRT 构建/评测/位置分解/复现脚本；`results/push-20261025*/README.md` 是实验记录 |
| `08-report/` | 汇总报告 + `figs/` 三张图 |
| `09-ptq-toolkit/` | 一条命令工具链（build/bench/report）、混合精度包、I2/I3 选层 |
| `PTQ-RETRO.md` / `ROOFLINE-int8.md` | 复盘与 roofline |

## 复现前提

- Linux/WSL2 + NVIDIA GPU；TensorRT 11.4、CUDA 12.4、PyTorch 2.6
- 模型权重、ONNX、engine、数据**未入库**；各目录 README / 脚本注释给出生成方式
- 第三方组件与内核库遵循各自许可；本仓库代码 MIT（见 LICENSE）

## 边界（必须一起读）

- 基线是本机 ncnn CPU fp32 11.6 ms/img，**不可跨机器比较**
- int8 延迟 skip 前 1000 张校准样本；只计 forward
- 仅 CIFAR-10 小模型；未验证 ImageNet / Transformer / LLM
- 1559× 是发布口径；位置分解 <0.006 为“折算/待发布态复测”，未宣布绝对达成
- 测量：空闲 clock 1350MHz、预热 100 次会撞 50W 功耗墙；只信同场 A/B 比值
