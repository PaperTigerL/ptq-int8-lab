# 编译与运行

环境：WSL Ubuntu，GTX 1650 Ti（Turing，支持 int8 tensor core）。

## 一次性准备

```bash
# 1) CUDA toolkit 装到用户目录（sudo 要密码，runfile 支持 --toolkitpath）
cd ~
curl -L -o cuda_12.4.1.run https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run
sh cuda_12.4.1.run --toolkit --toolkitpath=/home/user/cuda --silent --override --no-man-page
/home/user/cuda/bin/nvcc --version   # 应输出 V12.4.131

# 2) libtorch C++（CUDA 版）
cd ~
curl -L -o libtorch.zip "https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.4.0+cu121.zip"
unzip -q libtorch.zip   # 解出 ~/libtorch
```

## 导出 TorchScript 模型

```bash
cd /mnt/d/WorkBuddy/ShowBro/lab/03-ptq-lab
~/pyenv-torch/bin/python ../04-libtorch-cuda/export_ts.py
# 产出 resnet18_cifar10.pt（约 43 MB），并自检 max diff = 0.0
```

## 编译

```bash
cd /mnt/d/WorkBuddy/ShowBro/lab/04-libtorch-cuda/cpp

g++ -O2 -std=c++17 gpu_eval.cpp cifar10.cpp -o gpu_eval \
    -I. \
    -I/home/user/libtorch/include \
    -I/home/user/libtorch/include/torch/csrc/api/include \
    -L/home/user/libtorch/lib \
    -Wl,--no-as-needed \
    -ltorch -ltorch_cuda -lc10 -lc10_cuda -ltorch_cpu \
    -Wl,-rpath,/home/user/libtorch/lib \
    -fopenmp
```

### ⚠️ 两个必须注意的编译/运行细节

1. **`-Wl,--no-as-needed` 不能省**。默认 `--as-needed` 会把 `libtorch_cuda.so` 剔掉
   （因为 `torch::cuda::is_available()` 的符号经由 `libtorch_cpu.so` 转发），
   结果 `ldd` 里看不到 `libtorch_cuda.so`，运行时 `cuda available: 0`，
   而且**没有任何报错信息**——只是静默退到 CPU。

2. **运行要带 `LD_LIBRARY_PATH=/usr/lib/wsl/lib`**。WSL 的 NVIDIA 驱动库
   `libcuda.so.1` 在 `/usr/lib/wsl/lib`，不在默认搜索路径里。

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/libtorch/lib ./gpu_eval
```

## 运行

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/libtorch/lib ./gpu_eval \
    resnet18_cifar10.pt \
    /mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/data_raw/test_batch.raw \
    --batch 256            # 或 --limit 2000 快速验证，--cpu 对照
```

## 实测结果（ResNet-18 CIFAR-10，全量 10000 张，fp32）

| 方案 | ms/img | 10000 张耗时 | vs v1 |
|---|---|---|---|
| ncnn CPU fp32（v1） | 11.6 | 116 s | 1× |
| ncnn CPU int8 4 线程 | 3.4 | 34 s | 3.4× |
| PyTorch CPU batch=128 | 0.15 | 1.5 s | 77× |
| **libtorch GPU batch=512** | **0.0784** | **0.78 s** | **148×** |

**精度 87.16%，与 ncnn CPU / PyTorch 逐位一致**——同精度，不是近似。

GPU 首次运行含预热（0.117 ms/img），稳定后 0.078–0.082 ms/img。
表中取稳定值。数据准备+上传只占 0.018 s（2.3%），瓶颈在推理本身。

### batch 大小扫描（全量 10000 张）

| batch | ms/img |
|---|---|
| 64 | 0.40 |
| 256 | 0.117 |
| **512** | **0.078–0.082** |
| 1024 | 0.113 |

512 是这台机器的拐点，再大反而变慢（显存压力 + 调度）。

## 后续：TensorRT 路线

把同样的模型走 TensorRT（见 `../07-tensorrt-deploy/README.md`），
**fp16 batch=512 做到 0.01068 ms/img = 1086×、int8 = 0.00744 ms/img = 1559×**
（2026-10-25 口径：preboost+steady，先把 layer4「1×1 空间上的 3×3」折叠成 1×1），
精度 87.18% / 87.22%。本目录的 148× 是那条路的起点，不是终点。

> **口径说明（两个都留着，别混用）**：
> - **旧口径**（idle 门控，2026-10-20）：fp16 0.0149（778×）/ int8 0.0189（614×）。
>   当时以为「等 GPU 空闲再测」最干净，实际把 clock 等回 1350 MHz（上限 2100），
>   **数字偏慢**；再往前那版「0.0128 ≈ 906×」用的是二十轮的**最小值**，更不可信（中位只有 592×）。
> - **新口径**（2026-10-25）：preboost(100) + steady，clock 1755–1920 MHz，
>   并且图把 8/9 乘零的卷积折掉。fp16 → 1086×、int8 → 1559×。
>
> 完整两口径对照见 `../08-report/README.md` 第二节（§2.3 是同一份 engine 的 A/B）。

七条路线的同口径对照见 **`../08-report/README.md`**。

## 冒烟测试

```bash
cd /mnt/d/WorkBuddy/ShowBro/lab/04-libtorch-cuda/cpp
g++ -O2 -std=c++17 cuda_smoke.cpp -o cuda_smoke <同样的链接参数>
LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/libtorch/lib ./cuda_smoke
# 应输出 cuda available: 1 / device count: 1
```

`cuda_smoke` 先跑，确认 CUDA 通路再跑 `gpu_eval`，否则会在错误的地基上调试。
