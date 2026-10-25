# PTQ int8 部署实验室（CIFAR ResNet / MobileNet）

把 PyTorch 模型做完训练后量化（PTQ）、导出、C++/TensorRT 部署，并给出**可复现**的精度/延迟证据、roofline 分析与负结果记录。

> 定位：**部署优化与测量方法学**的工程仓库，不是新模型/新算法论文。所有倍数是相对本机 ncnn CPU fp32 基线（11.6 ms/img, CIFAR-10 10000 张）的估算。

## 主结果

| 路径 | 精度 | 延迟（ms/img） | 相对基线 |
|---|---|---|---|
| TensorRT int8（发布口径） | 87.22% | **0.00744** | **1559×** |
| TensorRT fp16 | 87.18% | 0.01068 | 1086× |
| TensorRT fp32 | 87.16% | 0.02477 | 468× |
| 自研 C++/CUDA int8（按层混合 implicit/cublas） | 87.01% | 0.0248 | 468× |

补充：
- 位置分解（3×3 pad=1 → 2×2 conv + DepthToSpace）在 int8 上同场 **−20%**（0.0080→0.0064，最好 0.00609；待发布态复测）
- MobileNetV2 混合精度：全量 10k 掉点 **0.92pp**（W4 per-channel + 24 层 W8，有效比特 4.99）
- ResNet-50 top4_all 混合精度：512 张 0.59pp / 2000 张 0.50pp

## 三个值得看的技术点

1. **结构性省功 > 调 kernel**：32×32 输入的 ResNet-18，layer4 空间已是 1×1，却仍挂 3×3 pad=1 卷积——8/9 的乘法在乘 padding 零。ONNX 层切中心 tap 改成 1×1，语义恒等（logits 差 2.4e-7），int8 从 0.0189 降到 0.00744。
2. **位置分解**：输入空间 ≤ kernel 的那族卷积，按输出位置改写成“一个 2×2 卷积（out_channels 扩 4 倍）+ DepthToSpace”，FLOPs 少 2.25×，kernel 数不增；收益集中在 `layer4.0/conv1`（11.6×），layer3 因 shuffle/reformat 净亏。
3. **测量协议**：本机空闲 clock 只有 1350MHz（上限 2100），逐批测试会低估 1.4×；预热 100 次又会撞 50W 功耗墙。结论必须用同场穿插 A/B + 多次中位 + clock 采样。

## 目录

| 路径 | 内容 |
|---|---|
| `03-ptq-lab/` | PTQ 算法原型（per-channel/per-group、percentile/entropy/MSE 校准）、PyTorch 评测 |
| `04-libtorch-cuda/` | libtorch C++ CUDA 推理线 |
| `05-ncnn-deploy/` | ncnn CPU/CUDA 部署与验收 harness |
| `06-gpu-int8/` | 自研 C++/CUDA int8 引擎（im2col/GEMM/ref_e2e/roofline/按层混合） |
| `07-tensorrt-deploy/` | TensorRT 构建/评测/位置分解/复现脚本 |
| `08-report/` | 同口径汇总表、图、可溯源自检 |
| `09-ptq-toolkit/` | 一条命令的工具链（build/bench/report、混合精度包、自检、I2/I3 选层） |
| `ROOFLINE-int8.md` | int8 DP4A 理论上限、实测差距、Amdahl 分解 |
| `PTQ-RETRO.md` | 自研引擎 vs TensorRT 的复盘与负结果 |

## 复现前提

- Linux/WSL2 + NVIDIA GPU（本项目在 GTX 1650 Ti 上测）；TensorRT 11.4、CUDA 12.4、PyTorch 2.6
- 模型权重、ONNX、TensorRT engine、训练数据**未入库**（体积/许可原因）；各目录 README 给出生成命令
- TensorRT/CUDA/ncnn 属各自厂商许可；本仓库代码 MIT

## 已知边界

- 仅 CIFAR-10 小模型；未验证 ImageNet/Transformer/LLM
- int8 延迟 skip 前 1000 张校准样本；只计 forward；基线不可跨机器比较
- 位置分解的 −20% 是同 session 相对值，绝对 <0.006 ms/img 待发布态复测
- 自研引擎（06）仍慢于 TensorRT 约 3.3×，保留作研究对照

## 许可

MIT（见 LICENSE）。第三方组件/内核库遵循各自许可。
