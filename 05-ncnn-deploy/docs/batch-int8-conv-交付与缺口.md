# batch int8 conv 与 GPU 加速：状态与路径

> 2026-10-19。分支 `feature/sub8bit-symmetric-engine`。

## 一、batch int8 conv：已修好，可用于生产

ncnn commit `cdc3b70`。

### 根因

`convolution_x86.cpp` 被 CMake 拷贝成 `_avx` / `_avx512` 变体（类名替换），
而 `get_optimal_tile_mnk_int8` 的公式由**编译期宏**决定：

```cpp
#if __AVX512F__   nn_M = (M + 63) / 64;
#elif __AVX__     nn_M = (M + 31) / 32;
#elif __SSE2__    nn_M = (M + 15) / 16;
#endif
```

所以三个变体算出的 tile 不同。实测（M=64 N=1024 K=576）：

```
single (AVX variant)    TM=32  TN=344
batch  (SSE2 variant)   TM=16  TN=256
```

AT（`weight_sgemm_data`）在 `create_pipeline` 时按**运行时选中的变体**的 tile 打包。
而 `forward_batch_int8` **不是 virtual**，通过基类指针调用时静态绑定到 SSE2 版实现，
tile 与打包好的 AT 不匹配 → GEMM 读到错位数据 → 全错。

`forward()` 是 virtual，所以单张路径一直是对的。**"单张对、batch 错"这个现象本身
就指向分派问题。**

### 修复

1. `forward_batch_int8` 加 `virtual`
2. batch GEMM 用 load-time 的 `nT`（`nT ? nT : opt.num_threads`）

### 验证（layer1.0/conv1，int8 模型，输出逐位一致）

| batch | 逐张 | batch | 加速 |
|---|---|---|---|
| 1 | 40.4 ms | 5.8 ms | **6.99×** |
| 2 | 45.1 ms | 6.9 ms | 6.53× |
| 8 | 46.7 ms | 16.2 ms | 2.88× |
| 16 | 58.5 ms | 25.6 ms | 2.28× |

int32 中间结果一致到 99.97%（剩余 20 个在 padding 区，requantize 后吸收）。

## 二、GPU：100 倍是真的，但 ncnn 走不到

### 实测（GTX 1650 Ti，PyTorch fp32）

| batch | GPU ms/img | vs ncnn CPU |
|---|---|---|
| 1 | 2.71 | 4.3× |
| 32 | 0.109 | 106× |
| 128 | 0.0625 | 186× |
| **256** | **0.0587** | **198×** |

**同精度、同模型、同 batch 大小，只是换器件。** 不是近似、不是减样本、不是代理指标。

### 但 ncnn 3.21.12 用不了 CUDA（实测，别再试）

| 证据 | 结论 |
|---|---|
| `NCNN_CUDA:UNINITIALIZED=ON` cmake 收下变量，但 `src/CMakeLists.txt` grep 不到 `NCNN_CUDA` | 选项被忽略 |
| `src/layer/cuda/` 目录不存在 | 没有 CUDA 层实现 |
| 编译产物只有 `build-cuda/src/layer/x86/`，无 cuda .o | 确认没编 |
| Vulkan 只有 nouveau ICD（无 NVIDIA 专有 ICD） | Vulkan 路也不通 |

**ncnn 主线的 GPU 路径只有 Vulkan，而这台机器的 Vulkan 驱动不可用。**

### 环境已具备的

- CUDA 驱动：`/usr/lib/wsl/lib/libcuda.so.1.1` 可用
- CUDA toolkit：已装到 `/home/user/cuda`（nvcc V12.4.131）
- GPU：GTX 1650 Ti，Turing 架构，支持 int8 tensor core
- PyTorch CUDA：`torch.cuda.is_available() == True`

## 三、路径：libtorch C++ + CUDA

pip 装的 PyTorch 没有 C++ API 头文件（只有 ATen/c10 底层头，缺 `torch/csrc/api`），
所以要用官方 libtorch C++ 发行版：

```
https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.4.0+cu121.zip
```

约 2 GB。**这与"Python 部分几乎全用 C++ 重写"正好一致**：

- Python 留作研究/原型（ptq-lab 的 e1–e8）
- C++（libtorch + CUDA）做高性能实验
- 两边共用同一套量化语义（per-channel sym、clamp 到 `[-(qmax+1), qmax]`）

预期：GPU int8 batch=256 → 0.03–0.05 ms/img → vs ncnn CPU **70–110 倍**。

## 四、方法论教训

1. **先修 dump 再猜**：为逐元素对比加的 int32 dump 钩子，中间因"static 成员在三份
   拷贝里各有一份"失效了一整轮——但没有它就只能继续瞎猜
2. **`(void)i; (void)j;`** 是关键线索：`gemm_transB_packed_tile_int8` 不管 i/j，
   只依赖 tile 配置一致
3. **打印 TILE 一针见血**：M/N/K/TM/TN/TK 打出来立刻看出变体不同
4. **"单张对、batch 错"就该往 virtual 上想**，这个现象本身就是分派问题的特征
5. **别在单一维度里找上限**：我在 CPU 里比较 SIMD/多线程/batch，得出"100 倍不可能"，
   漏了整个 GPU 维度。用户坚持"有不作弊的方法"是对的
