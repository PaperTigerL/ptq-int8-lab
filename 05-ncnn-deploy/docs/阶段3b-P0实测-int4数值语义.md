# 阶段 3b · P0+P2a 实测：ncnn 打通 sub-8bit 数值与存储通路

> 2026-10-18 实测。承接 `阶段3-扩展ncnn量化语义-设计.md`。
> ncnn 改动在分支 `feature/sub8bit-symmetric-engine`
> （commit `d21994b` 数值语义、`d9a0db2` 存储压缩）。
> 上层管线与模型在 `lab` 主仓 commit `9b99429`。

---

## 0. 结论

**ncnn 现在能跑真正的 4bit 码空间，并且权重按 2 值/byte 存储——体积减半、精度不变。**

| | ptq-lab（PyTorch） | ncnn（本次） | 差 |
|---|---|---|---|
| W4A8 对称 | 85.36% | **85.36%** | **0.00pp** |
| W4A4 对称 | 78.61% | **78.42%** | 0.19pp（噪声内） |

| 格式 | 体积 | 相对 fp32 |
|---|---|---|
| fp32 | 44.71 MB | 1.00× |
| int8 | 11.21 MB | 3.99× |
| **int4 打包** | **5.63 MB** | **7.94×** |

W8A8 回归：**87.05%**，与 `阶段1-ncnn-INT8基线.md` 记录**完全一致**——`bits=8` 路径未受任何影响。

---

## 1. 为什么加个"位宽参数"不够（这张表是核心证据）

| # | 权重量化 | 激活表 scale | bits | packing | 精度 | 说明 |
|---|---|---|---|---|---|---|
| 1 | W4 | A8（127/t） | 8 | on | 85.36% | 基线，对齐 ptq W4A8 |
| 2 | W4 | A4（7/t） | 8 | on | 78.44% | **scale 单独就决定了码字**，clamp 无关 |
| 3 | W4 | A4（7/t） | 4 | **on** | **10.00%** | ✗ pack8 下数据被搅乱 |
| 4 | W4 | A4（7/t） | 4 | **off** | **78.42%** | ✓ 正确 int4 |
| 5 | W4 | A8（127/t） | **4** | off | **10.00%** | ✗ scale 与位宽不匹配 |

两个 10.00%（随机猜水平）分别来自两个**不同的**错误，都必须避开：

### 坑 1：pack8 不能复用（行 3）

`quantize_to_int8()` 内部 `create_layer(Quantize)` 拿到的是 `Quantize_x86`，它重写 forward 走 SIMD，硬编码 ±127，**完全不看基类的 `bits`**。

回落到基类 `Quantize::forward` 后，问题变成：**基类只处理 `elempack==1` 的连续 blob**，不认 pack8 的通道布局。硬走 pack8，数据顺序全错 → 10%。

处理：`Quantize_x86`/`Requantize_x86` 在 `bits != 8` 时回落基类，**同时由评测器检测到 sub-8bit 后关闭 packing**（`evaluator.cpp::param_has_sub8bit`）。代价是慢（7.7 → 14.6 ms/张），int4 SIMD 留 P2。

> 这条也解释了为什么"只加个 bits 参数"不成立：**kernel 的数据布局假设和位宽是绑定的**。

### 坑 2：scale 与位宽必须成对（行 5）

A8 表的 scale 是 `127/t`，`bits=4` 把码字截到 ±7。于是 `x=t` 时 `q=127` 被截成 7，**几乎全部激活撞在上界**。等效于"用放大 18 倍的尺子量再砍掉一半刻度"，信息全毁。

反过来看行 2：A4 表的 scale 是 `7/t`，`x=t` 时 `q=7`，正好落在 `bits=4` 的边界内。**所以行 2 和行 4 结果几乎相同（78.44 vs 78.42）——clamp 在 scale 匹配时根本不起作用。**

> 这条是"位宽不是孤立参数"的直接证据：**位宽、scale、kernel 三者必须一起改，单独动任何一个都会崩。**
> 这正好回答了一开始的质疑——"加 int4 是不是只是加个位宽"：不是。

---

## 2. 改了什么（ncnn 侧，13 个文件 / +100 −26）

| 文件 | 改动 |
|---|---|
| `src/layer/quantize.{h,cpp}` | 新增 param id 1 = `bits`，clamp 改为 `[-(2^(b-1)-1), 2^(b-1)-1]` |
| `src/layer/requantize.{h,cpp}` | 新增 param id 5 = `bits`，同上 |
| `src/layer/convolution.{h,cpp}` | 新增 param id 20 = `int8_bits`，透传给激活量化与输出 requantize |
| `src/mat.{h,cpp}` | `quantize_to_int8` / `requantize_from_int32_to_int8` 加 `bits` 默认参数 |
| `src/layer/x86/quantize_x86.cpp` | `bits != 8` 回落基类 |
| `src/layer/x86/requantize_x86.cpp` | 同上 |
| `src/layer/x86/convolution_x86.cpp` | 两处调用点透传 `int8_bits` |
| `tools/modelwriter.h` | Convolution 分支写出 `20=` |
| `tools/quantize/ncnn2int8.cpp` | 可选尾部参数 `--bits N` |

**故意没改**：`convolution.cpp:145`（权重量化）。那条路径只在权重仍是 fp32 时才走；sub-8bit 的权重是"值落在 4bit 网格上、仍用 int8 容器存"，压到 ±127 正是我们要的。

**全部向后兼容**：`bits` 默认 8，`bits==8` 时不进入任何新分支。

---

## 3. 上层管线怎么配合

| 环节 | 改动 |
|---|---|
| `03-ptq-lab/tools/dump_scales.py` | `--symmetric`，导出 `scales_*_sym.json` |
| `03-ptq-lab/src/calib.py` | `calibrate` 透传 `symmetric` |
| `05-ncnn-deploy/scripts/export_onnx.py` | `--w-bits 4`：fold BN 后逐通道量化，值存 fp32 |
| `05-ncnn-deploy/scripts/make_table.py` | `--a-symmetric`：注入 `1/a_scale`，**不做** 127/255 换算 |
| `05-ncnn-deploy/cpp/evaluator.cpp` | 检测 `20<8` 后关闭 packing |

`make_table` 那条要注意：`127/255` 换算是给"非对称 255 码"用的。对称的 `a_scale` 已经是网格步长，直接取倒数。**套错会差 2 倍**——这是阶段 2 埋下的伏笔，这里必须分叉。

---

## 4. 复现

```bash
cd /mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy

# 1) W4 权重 → onnx（fold BN + 量化到 4bit 网格）
~/pyenv-torch/bin/python scripts/export_onnx.py --w-bits 4
/home/user/ncnn-build/ncnn/build/tools/onnx/onnx2ncnn \
    models/resnet18_cifar10_w4.onnx models/resnet18_cifar10_w4.param models/resnet18_cifar10_w4.bin
/home/user/ncnn-build/ncnn/build/tools/ncnnoptimize \
    models/resnet18_cifar10_w4.param models/resnet18_cifar10_w4.bin \
    models/resnet18_cifar10_w4_opt.param models/resnet18_cifar10_w4_opt.bin 0

# 2) A4 对称配置 + 表
cd ../03-ptq-lab
~/pyenv-torch/bin/python tools/dump_scales.py --w-bits 4 --a-bits 4 --symmetric
cd ../05-ncnn-deploy
~/pyenv-torch/bin/python scripts/make_table.py \
    --scales ../03-ptq-lab/results/cifar10_resnet18/scales_w4a4_minmax_sym.json \
    --param models/resnet18_cifar10_w4_opt.param \
    --baseline tables/resnet18_cifar10_w4_kl.table \
    --out tables/resnet18_cifar10_w4_a4sym.table

# 3) 量化 + 评测
/home/user/ncnn-build/ncnn/build/tools/quantize/ncnn2int8 \
    models/resnet18_cifar10_w4_opt.param models/resnet18_cifar10_w4_opt.bin \
    models/resnet18_cifar10_w4_int8.param models/resnet18_cifar10_w4_int8.bin \
    tables/resnet18_cifar10_w4_a4sym.table --bits 4
./cpp/build/cifar10_eval --param models/resnet18_cifar10_w4_int8.param \
    --bin models/resnet18_cifar10_w4_int8.bin --data data_raw/test_batch.raw \
    --threads 1 --tag "W4A4-int4"
```

---

## 5. 没有做的（明确边界）

| 项 | 状态 | 说明 |
|---|---|---|
| int4 **存储压缩** | ✅ **已做（P2a）** | 权重 2 值/byte，bin 11.21MB → **5.63MB**，精度不变 |
| int4 **SIMD kernel** | ❌ 未做 | 现在走标量，10–15 ms/张（int8 是 6.5 ms/张） |
| **非对称激活** | ❌ 未做 | P1 内容：Quantize 带 zp + bias 折叠 + padding 填 z_in |
| 混合位宽调度 | ❌ 未做 | P3 内容 |
| 其他后端（arm/riscv/…） | 未验证 | 只改了 x86；其余后端不传 bits 即保持 8 |

**所以现在的定位是**：ncnn 具备 sub-8bit 的**完整存储 + 数值通路**，体积减半且精度与 PyTorch 对齐；**速度没有收益**（标量路径），那属于 P2b。

---

## 5.5 存储压缩（P2a）：权重 2 值/byte

| 格式 | 体积 | 相对 fp32 | 精度 |
|---|---|---|---|
| fp32 | 44.71 MB | 1.00× | 87.16% |
| int8 | 11.21 MB | 3.99× | 87.05% |
| **int4 打包** | **5.63 MB** | **7.94×** | **78.42%** |

**做法**：打包存盘，`load_model` 时解包成 int8。kernel 一行没改——它吃的仍是 `signed char` 数组。所以这是**纯存储格式变更**，运行时内存不变。

格式：每字节两个 4bit 补码，低 nibble = 偶数索引，高 nibble = 奇数索引；`weight_data_size`（param id 6）仍是**解包后**的元素数。

### 这里踩了三个坑，都值得记住

**坑 A：权重量化的位宽必须和打包一致**

`quantize_to_int8` 默认 `bits=8`，权重被压到 ±127。之后按 4bit 打包只取低 4 位，**高 4 位被静默丢弃** → 精度 13.99%。
修正：权重量化也传 `int8_bits`。表里注入的 `w_scale` 本来就是 `7/absmax`，所以 `q` 天然落在 `[-7,7]`，打包无损。

**坑 B：3D Mat 的 `cstep` 有对齐 padding**

`quantize_to_int8` 输出的是 3D Mat `(maxk, num_input, num_output)`。ncnn 的 Mat 每通道按 16 字节对齐，`cstep >= w*h*d`。直接 `const signed char* p = mat; p[i]` 线性遍历会**跨过 padding 读到未初始化内存**。

- `conv1`：`w*h = 49×3 = 147` → 补齐到 160 → 读到垃圾（值域里冒出 −128）
- 其余层：`w*h = 576`，本来就是 16 的倍数 → 没有 padding → 看不出问题

**所以只有 conv1 坏**，极容易误判成"conv1 特殊"。修正：用 `Mat::channel(c)` 逐通道拷贝成连续数组。

**坑 C：`ConvolutionDepthWise` 还没有 `int8_bits`**

`ncnn2int8` 的打包只加在 `quantize_convolution`。DepthWise 走原 int8 路径，不受影响（MobileNet 类模型暂不支持 sub-8bit）。

---

## 5.6 速度：packing 开回来之后，追平 int8

| 模型 | ms/张 | 精度 |
|---|---|---|
| int8（packing on） | 5.64 | 87.05% |
| int4 非对称（packing **off**，标量） | 14.74 | 83.92% |
| **int4 非对称（packing on）** | **6.16** | **83.92%** |

**2.4× 提升，精度一分没变。**

关键认知：**慢的原因不是 int8 GEMM**。4bit 的值放进 int8 寄存器，
`int8×int8→int32` 累加数学上完全正确。真正的瓶颈是基类
`Quantize`/`Requantize` 硬编码 `elempack=1`，逼我们关掉 packing，
连带把 conv 也降级成标量路径。

两处改动就解决：

1. 基类 dims 3/4 保持输入 packing，每通道元素数改为 `w*h*d*elempack`
2. `Requantize` 的 per-channel 索引改成 `q*elempack+e`——
   packing 是**跨通道交错**，第 e 个 lane 属于原始通道 `q*elempack+e`。
   按 `q` 索引会让 8 个 lane 里 7 个取错 scale（精度直接掉到 19.58%）

**所以不需要自己重写 int8 kernel。** 剩下的优化空间只有"int4 权重常驻内存
+ 融合解包"，那是**内存**收益而非速度收益。

---

## 5.7 非对称激活（P1）：把浪费的半个码空间拿回来

**结果：78.42% → 83.92%，体积不变（5.63MB）。** ptq-lab 参考值是 83.98%，差 0.06pp。

| 配置 | 精度 | 体积 |
|---|---|---|
| FP32 | 87.16% | 44.71 MB |
| W4A4 对称 | 78.42% | 5.63 MB |
| **W4A4 非对称** | **83.92%** | **5.63 MB** |
| ptq-lab W4A4 对称 | 78.61% | — |
| ptq-lab W4A4 非对称 | 83.98% | — |

### 码空间

```
对称    q ∈ [-(2^(b-1)-1), 2^(b-1)-1]   4bit 只有 8 个有效码（0..7），负半轴全废
非对称  q ∈ [0, 2^b-1] + zero_point     4bit 有 16 个码全部用上
```

ReLU 之后激活全非负，对称量化把负半轴那一半码字空着——这就是 5.50pp 的来源。

### 关键：zero_point 不进 kernel

```
y = s_x·s_w·( Σ q_x·q_w − zp·Σ q_w ) + b
  = Σ q_x·q_w · scale_in  +  ( b − zp·Σ q_w·scale_in )
                                  └──── 离线算一次，折进 bias ────┘
```

`scale_in = 1/(bottom_scale × weight_scale)`，`Σ q_w` 是每输出通道的权重和。
`ncnn2int8` 对每个输出通道算一次，**运行时零开销，conv kernel 一行没改**。

### 四处配套改动

| 位置 | 改动 |
|---|---|
| `Quantize` | param 2=`asymmetric`、3=`zero_point`；非对称 `q = round(x·s) + zp`，clamp `[0, 2^b-1]` |
| `Convolution` | param 21=`asymmetric`、22=`zp_in`、23=`zp_out` |
| `ncnn2int8` | 从表里读 `<层名>_zp` → 算 `Σ q_w` → 改 bias → 设 `pad_value = zp` |
| fused requantize | `conv1->zp_out = conv2->zp_in`，让融合后两层的零点接得上 |

**padding 为什么必须填 zp**：`pad_value` 是 float，对 int8 blob 会被 `static_cast<signed char>`。padding 位置在真实语义里是"零"，量化后零对应的码字正是 `zp`——填 0 等于填了一个非零的数。

### 一个前提条件

非对称模式**要求该层有 bias**（折叠项无处放就报错）。ResNet 的 conv 都有 bias；bias-free 网络需要改成给 Requantize 单独加一个 bias 项。

### 为什么这次不能只改 scale

`--a-asymmetric` 时注入的是 `1/a_scale`，**不做 127/255 换算**。因为 ptq 的 `a_scale = (hi-lo)/(2^b-1)` 已经是"非对称码空间步长"，ncnn 要的乘数就是它的倒数。**只有「ptq 的 asym 塞进 ncnn 的对称码空间」才需要 127/255**——那是阶段 2 的场景，两者不能混。

---

## 6. 一句话（给简历/答辩）

> 在 ncnn 上实现 sub-8bit 量化码空间（`Quantize`/`Requantize` 位宽参数 + `Convolution` 透传 + 工具链 `--bits`），实测 W4A4 对称精度 78.42%，与 PyTorch 参考实现 78.61% 相差 0.19pp；并定位出两处必须同时满足的约束——pack8 布局假设、scale 与位宽的配对关系——违反任一条精度即跌至随机水平。
