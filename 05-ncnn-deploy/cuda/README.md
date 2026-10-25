# 05-ncnn-deploy/cuda —— ncnn + CUDA 路线（已跑通，结论：这条路走不到 500×）

ResNet-18 / CIFAR-10，10000 张测试集子集，GTX 1650 Ti（WSL + CUDA 12.4）。
基线沿用全项目口径：**ncnn CPU fp32 = 11.6 ms/img = 1×**（04-libtorch-cuda README）。

---

## 一、结论先说

| 配置 | 精度 | ms/img | 加速比 | 能不能用 |
|---|---|---|---|---|
| ncnn CUDA **fp32**，显存不复用 | **88.00%** | **14.89** | **0.8×** | 精度对，**比 CPU 还慢** |
| ncnn CUDA fp32，显存每张复用 | 12.00% | 7.71 | 1.5× | ❌ 精度是随机猜 |
| ncnn CUDA **int8**，显存不复用 | **9.00%** | 39.14 | 0.3× | ❌ int8 CUDA 路径是坏的 |
| ncnn CPU fp32（基线） | 87.15% | 11.6 | 1× | — |
| libtorch C++ GPU batch=512 | 87.16% | 0.0784 | 148× | ✅ |
| **TensorRT fp16 batch=512** | **87.18%** | **0.0149** | **778×** | ✅ |

**这条路的天花板是 1.5×（而且那个 1.5× 的精度是错的），精度正确的配置只有 0.8×。
距离 500× 差两个数量级，不是"再优化一下"能补的。**

样本数只能用 200：唯一精度正确的配置（显存不复用）每图吃掉约 1.3 MB 且从不释放，
1000 张就把 256 MB arena 撑爆（实测报 `arena 溢出`）。这是这条路又一个硬限制。

---

## 二、这一轮真正修掉的东西：CUDA 层根本没被调用

一开始 `LOG_LAYERS` 日志里 0 个 cuda 层，全都是 `convolution.cpp(110)`、
`quantize.cpp(45)` 这种 CPU 源文件。但同一时刻 `create_layer("Convolution")->support_cuda`
明明是 1。层建成了 CUDA 层，执行却走了基类 CPU 的 `forward`。

根因（`src/net.cpp` / `src/layer.cpp` 读出来的）：

```
Extractor::input (const char*, const CudaMat&)  ->  Net::forward_layer(idx, blob_mats_gpu, opt)   // GPU
Extractor::input (const char*, const Mat&)      ->  Net::forward_layer(idx, blob_mats,      opt)   // CPU
```

**开关是"你塞进 Extractor 的是 Mat 还是 CudaMat"，跟 `opt.use_cuda_compute` 无关。**
而且 `use_cuda_compute` 在 `option.h:107` 只有一行声明，整个 `src/` 没有任何一处读它 ——
它是个死开关。设它等于没设。

修法：派生一个 `NetCUDA : public ncnn::Net`，把 protected 的
`forward_layer(int, std::vector<CudaMat>&, const Option&)` 提到 public，自己驱动推理循环
（顺带避开 `Extractor::extract` 每张图内部都 `cudaDeviceSynchronize()` 一次的开销），
输入输出全程走 `CudaMat`。

改完之后 fp32 GPU 精度 88.00%，和 CPU 一致 —— GPU 通路是真的在跑了。

---

## 三、这条路为什么到不了 500×（四条，都是实测/读码得出的，不是推测）

### 1. fork 的 CUDA kernel 是手写朴素实现，没有 cuDNN

`src/layer/cuda/convolution_cuda.cu` 里全是这种三重循环：

```cpp
for (int input_channel = 0; input_channel < a_info.c; input_channel++)
    for (int k = 0; k < product_info.maxk; k++)
        sum += sptr[gpu_space_offset[k]] * shared_kptr[...];
```

整个 `.cu` 里搜不到 `cudnn`、`cublas`、`im2col`、`wmma`。没有 implicit GEMM、没有
TensorCore、没有共享内存分块的矩阵乘。这是 2020 年的一个研究性质 fork
（atanmarko/ncnn-with-cuda），ncnn 上游至今**没有 CUDA 后端**——主线 GPU 只有 Vulkan，
而 WSL 里只有 nouveau ICD，用不了。

同一个卷积，TensorRT 走的是 cuDNN 的 `turing_fp16x2_hcudnn_fp16x2_*` tactic。
这两者的差距不是一个数量级能形容的。

### 2. ncnn 的 Mat 没有 batch 维

`src/mat.h` 的 `Mat` 只有 `w / h / c`（`CudaMat` 同样），没有 `d`，也没有 batch。
所以图上没法做 batch —— 每张图都要重新走一遍 20 个 conv + 8 个 binaryop + 8 个 split
共几十个 kernel。小图上 launch 开销直接吃掉一切。

对比：libtorch 的 148× 和 TensorRT 的 778× 都是 batch=512 打出来的。

### 3. 显存一复用，精度就崩

这是本轮最花时间的一条。三种分配器实测：

| 分配器 | ms/img | 精度 |
|---|---|---|
| 裸 `CudaAllocator`（每次 cudaMalloc） | 126.0 | 88.00% ✅ |
| fork 的 `CudaPoolAllocator`（benchncnn_gpu 同款） | 5.47 | 19.00% ❌ |
| 自研单调 arena + 每张回退 | 6.83 | 10.50% ❌ |
| 自研单调 arena **不回退** | 32.65 | 88.00% ✅ |

一开始怀疑是"arena 回退把惰性常驻显存冲掉了"，加了 `--no-reset` 验证：精度确实回到
88%，所以假设成立一半。但接着扫 `--reset-every` 得到一条**单调曲线**：

| reset-every | 1 | 2 | 4 | 8 | 32 | ∞（不复用） |
|---|---|---|---|---|---|---|
| 精度 | 10.5% | 16.0% | 20.5% | 24.0% | 43.5% | **88.0%** |

不是"某个 bug 在某处"，是**复用得越狠、错得越多**。这个 fork 的 `CudaMat` 引用计数
（`mat.h` 的 `release()` 无条件把 `data=0; refcount=0`，`create()` 在形状相同且
allocator 相同时直接复用旧缓冲、完全不看 refcount）跟显存回收配合不上。

`--no-reset` 是唯一正确的配置，代价是显存单调增长 → 跑不了全量，而且反而更慢
（14.89 ms/img 对复用的 7.7 ms/img）。

### 4. int8 的 CUDA 路径结果是错的

`--int8` 在任何配置下都是 **9.00%**（随机猜测水平）。同一个 `resnet18_cifar10_int8_old.*`
模型走 CPU 通路是 87%。也就是说：

- 模型本身没问题（CPU 上对的）
- 驱动框架没问题（fp32 GPU 88% 是对的）
- **是 `Convolution_cuda::forward_int8` → `gpu_convolution_cuda_forward_int8` 这条链路坏了**

`ncnn_cuda_cmp.cpp` 可以逐层打印 CPU/GPU 的 RMS 做定位，但**这个 bug 修好了也没意义**：
int8 不复用显存时是 39.1 ms/img，比 fp32 的 14.89 还慢 2.6 倍。没有修的动力。

---

## 四、文件清单

| 文件 | 作用 |
|---|---|
| `ncnn_cuda_eval.cpp` | 主评测器。含 `NetCUDA` 派生类、`BumpCudaAllocator` 自研 arena、`--no-reset` / `--reset-every` 诊断开关 |
| `ncnn_cuda_cmp.cpp` | 逐层对比工具：同一张图分别走 CPU(Mat) 和 GPU(CudaMat)，打印每个 blob 的 RMS，定位第一个分叉的层 |
| `patch_ncnn_cuda.py` | 给 fork 打补丁：两处 `throw std::runtime_error` 换成 `return -1`（库整体 `-fno-exceptions`，编译不过） |
| `_rebuild.sh` | 干净重建 ncnn CUDA 库。**必须带 `-DLOG_LAYERS=OFF`** |
| `build.sh` | 编译评测器 |
| `run.sh` | 跑评测 |
| `final.sh` | 三组最终实测，结果落到 `results/` |
| `_sweep_reset.sh` | 扫 `--reset-every`，复现上面那条精度/复用曲线 |
| `_cmp.sh` | 编译并跑逐层对比 |
| `_probe.sh` / `probe_gpu.cpp` | GPU 探针：设备数、`support_cuda` |
| `_mk_int8_model.sh` | 用 fork 自带的 `ncnn2int8` 生成 fork 格式的 int8 模型 |
| `results/` | `cuda_fp32_noreset.txt` / `cuda_fp32_reuse.txt` / `cuda_int8_noreset.txt` |
| `analyze_logits.py` | logits 逐位比对器（`max|Δ|` + 预测一致数），被多个实验脚本调用 |

### 实验归档（2026-10-25 清点）

| 目录 | 内容 | 进库 |
|---|---|---|
| `experiments/20261025/` | h024–h093 那轮扫描的 144 个一次性脚本 / 补丁 / 变体评测器源码，按家族分（`ab/ check/ harness/ patch/ probe/ run/`）。清单见 `experiments/20261025/MANIFEST.md` | ✅ |
| `cuda/` 根下的 58 个评测器二进制 | `ncnn_cuda_eval_*` / `ncnn_cuda_dump_logits_*` / `ncnn_cuda_profile_layers_*` 等，都是 `build.sh` 的产物，可重建；其中 55 个是这次清点新纳入 ignore 的，3 个（`ncnn_cuda_eval` / `_log` / `_cmp`）原本就在 `.gitignore` 里 | ❌ `.gitignore` |
| `work/` | 各轮日志 / 逐层转储 / fork 源码快照 / verify 快照及其构建日志（605 个文件，72 MB），**保留在磁盘，不删** | ❌ `.gitignore` |

`results-h021/` 的 8 份验收证据照旧跟踪；其中 nvidia-smi 快照在 `work/results-h021/`。

## 五、复现

```bash
# 1) 重建库（WSL 里跑；注意必须走 `wsl.exe bash -lc 'bash <脚本>'`，
#    直接把 WSL 路径当 Windows 路径传给 wsl.exe 会被 Git Bash 拼坏）
bash /mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda/_rebuild.sh

# 2) 编译 + 跑
bash /mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda/build.sh
bash /mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda/final.sh
```

## 六、踩过的坑（按被坑的顺序）

1. **`opt.use_cuda_compute` 是死开关** —— 只有声明，没人读。真正的开关是 Mat vs CudaMat。
2. **`opt.blob_cuda_allocator` 默认不初始化** —— `option.cpp` 里一行都没有，是空
   shared_ptr，第一个 `create()` 就段错误。必须在 `load_param` 之前赋值
   （`benchncnn.cpp:337` 是这么写的）。
3. **`_loglayers.sh` 重跑 cmake 时漏了 `-DNCNN_CUDA=ON`** —— `CMakeLists.txt:17-20`
   会把 `LOG_LAYERS` 拼进全局 `CMAKE_CXX_FLAGS`，于是整个库每层 forward 都 fprintf 一行。
   带这个库测出来的延迟没有任何意义，必须 `_rebuild.sh` 干净重编。
4. **WSL 里 `cudaMalloc` 是毫秒级的** —— 裸分配器把 fp32 GPU 拖到 126 ms/img，
   比 CPU 还慢 10 倍。
5. **fork 的池化分配器会算错** —— 见第三节第 3 条。
6. **arena 水位线不能打在 0** —— 但这不是唯一原因，见 `--reset-every` 曲线。
7. **`Net::forward_layer` 是 protected** —— 派生类 `using` 提到 public 才能自己驱动循环。
8. **`ncnn::create_gpu_instance()` 只存在于 `NCNN_VULKAN` 分支** —— CUDA 侧要用
   `try_initialize_cuda_gpu_instances()`。
9. **链接要两个库** —— `-Wl,--start-group -lncnn -lncnn_cuda_lib -Wl,--end-group`。
   `libncnn.a` 是宿主代码，`libncnn_cuda_lib.a` 是 nvcc 编出来的 .cu kernel，少一个就
   undefined reference。
10. **老 fork 不认新版 `ncnn2int8` 的 int8 权重格式** —— 会 segfault
    （`ModelBin read index_array failed`）。必须用 fork 自带的 `ncnn2int8` 重新生成
    `resnet18_cifar10_int8_old.*`。
11. **WSL 内联命令会吃掉 `$VAR`** —— 循环变量、`PATH` 全变空。所有命令一律写成 `.sh`
    文件再执行。

---

## 七、那 500× 在哪

在 `../07-tensorrt-deploy/`：TensorRT fp16 batch=512，**0.0149 ms/img = 778×**，
精度 87.18%，与 fp32 基线同精度。

同样是 GPU、同样这张 1650 Ti，差别在于 TensorRT 用的是 cuDNN 调好的 tactic +
真 batch + fp16 TensorCore；而这个 ncnn fork 用的是手写三重循环 + 无 batch +
一复用显存就出错的 CudaMat。

### 那"换 ncnn 的 Vulkan 后端"行不行？

**本机不可行，而且就算可行也解决不了根本问题。** 这条我原本是凭印象写的，已实测推翻，
探测结果见 `results/vulkan_probe.md`：

| 环境 | 结论 |
|---|---|
| WSL | ❌ `/usr/share/vulkan/icd.d/` 只有 asahi / gfxstream / intel / lvp / nouveau / radeon / virtio，**没有 NVIDIA ICD**。WSL 的 GPU 直通只给了 CUDA |
| Windows | ❌ 有 `vulkan-1.dll` loader、驱动 616.92，但 DriverStore 里**没有 `nv-vk64.dll`**，注册表 `Khronos\Vulkan\Drivers` **不存在**，没装 Vulkan SDK，`glslangValidator` 也没有（ncnn 要靠它把 shader 编成 SPIR-V） |

**更关键的是：瓶颈不在 CUDA 还是 Vulkan。** ncnn 的 `Mat` 只有 `w/h/c`，
图里没有 batch 维 —— 这是框架层面的限制，换哪个 GPU 后端都一样，一张一张跑。
libtorch 的 148× 和 TensorRT 的 778× 全是 batch=512 打出来的红利。

所以正确的判断不是"换 Vulkan"，而是：

> 要吃 GPU 的吞吐，就得用**有 batch 维**的推理引擎（TensorRT / libtorch / ONNXRuntime）。
> ncnn 的定位是移动端**单张低延迟**，不是吞吐 —— 用它去比 batch=512 的吞吐，
> 从选框架那一步就输了。
