# int8 Conv / GEMM 规格（06 引擎口径）

- 版本：2026-10-22
- 参考实现：`ptq_toolkit/ref_int8.py`
- 单测：`tests/test_int8_ref.py`
- 对应实现：`06-gpu-int8/cpp/include/ptq_quant.h`、
  `06-gpu-int8/cpp/int8_kernels.cu`、`06-gpu-int8/cpp/int8_engine.cuh`

本文件锁死 **CPU 参考侧**的数值语义。任何一处口径漂移都必须同时改参考实现与
单测；06 引擎代码本单不改。

---

## 0. 目的与边界

- 引擎实际执行的是 **对称码空间**：权重 per-channel、激活 per-tensor、
  `zero_point = 0`。本文第 2.2 节是生产口径。
- 第 2.3 节的非对称 zero-point 是**参考语义**，用于把两种口径的差异做成
  可执行、可断言的测试；06 的 `_int_mm` 路径没有 zero-point 偏移，
  **不能**用非对称口径跑引擎。
- 全程 CPU/NumPy，不涉及 CUDA、训练或 06 代码改动。

---

## 1. 张量与数据类型

| 名称 | 形状 | 类型 | 说明 |
|---|---|---|---|
| 输入 `x` | `[N, C, H, W]` | float32 | NCHW，连续 |
| 权重 `w` | `[Cout, Cin/groups, kh, kw]` | float32 | PyTorch Conv2d 布局 |
| 偏置 `bias` | `[Cout]` | float32 | 反量化后相加，不量化 |
| 激活码字 | 同 `x` | int8（对称）/ uint8（非对称参考） | 见第 2 节 |
| 权重码字 | 同 `w` | int8 | 对称 |
| 累加器 | `[N, Cout, OH, OW]` | int32 | 仅在参考实现内部 |
| 输出 `y` | `[N, Cout, OH, OW]` | float32 | 反量化 + bias + 激活 |

- 舍入统一为 **half-away-from-zero**：
  `round(v) = sign(v) * floor(|v| + 0.5)`。
  不能用 `np.round` / `rintf` 的 half-to-even。
- 权重 scale 为 `[Cout]`，激活 scale 为标量。

---

## 2. 量化口径

### 2.1 对称权重量化（生产口径）

对每个输出通道 `co`，把权重展平成 `K = Cin/groups * kh * kw`：

```
qmax_w = 2^(bits-1) - 1          # W8: 127, W4: 7
s_w[co] = max_k |w[co, k]| / qmax_w
code  = clamp(round(w[co,k] / s_w[co]), -(qmax_w+1), qmax_w)
dequant = code * s_w[co]
```

- 负侧多一个码字 `-(qmax_w+1)`（W8 为 -128，W4 为 -8），保证 0 在网格上。
- 全零通道：`s_w = 1e-8`，码字全 0。
- **分母是 qmax_w（127/7），不是 255/15**。用 `2^bits-1` 会让 scale 大一倍。
- 本参考实现只做 per-channel；per-group W4 属于 I2 的候选方案，
  不在本文件的 conv/gemm 语义内。

### 2.2 对称激活：AbsMax（生产口径）

由校准范围 `(lo, hi)` 推 per-tensor scale：

```
qmax_a = 2^(bits-1) - 1          # 8 bit: 127
s_a = max(|lo|, |hi|) / qmax_a
zp  = 0
code = clamp(round(x / s_a), -(qmax_a+1), qmax_a)
dequant = code * s_a
```

- **跨零范围** `lo < 0 < hi`：AbsMax 取两侧绝对值较大者，正负都能表示；
  `hi` 映射到 `+127`，`lo` 映射到 `>= -127`（不会裁剪）。
- `-(qmax_a+1) = -128` 是对称码空间的负侧保护位；在 AbsMax 下正常数据
  不会用到它（因为 `|x| <= max(|lo|,|hi|)`）。
- `MinMax = (hi-lo)/qmax_a` **不是**默认口径，只保留作对照：
  它把整段 `[lo, hi]` 铺满 255 个码字，跨零不对称时 `max(|lo|,|hi|)` 一侧
  的分辨率反而更粗。以 `lo=-2, hi=4` 为例：
  AbsMax `4/127 = 0.031496`，MinMax `6/127 = 0.047244`。

### 2.3 非对称激活：zero-point（仅参考语义）

```
qmax_a = 2^bits - 1              # 8 bit: 255
s_a  = (hi - lo) / qmax_a
zp   = clamp(round(-lo / s_a), 0, qmax_a)
code = clamp(round(x / s_a) + zp, 0, qmax_a)
dequant = (code - zp) * s_a
```

- 码空间是 `[0, 255]`（256 个码字）；与对称的 `[-128, 127]` 不同。
- 真实值 0 的码字是 `zp`，不是 0。
- 反量化必须先减 `zp`；GEMM 的整数累加同样要减：
  `acc = Σ (code_a - zp_a) * code_w`。
- 以 `lo=-2, hi=4` 为例：`s_a = 6/255 = 0.023529`，`zp = 85`；
  同一组端点码字为 `[0, 85, 255]`，而对称 AbsMax 为 `[-64, 0, 127]`。
- **padding 的坑**：卷积窗口越界补的是**真实 0**。非对称路径必须补
  `code = zp`；补字面 0 会让反量化值是 `-zp*s_a`，边界元素整体偏移。
  对称路径 `zp=0`，所以补码字 0 恰好等于补真实 0。

### 2.4 两种口径不能混用

| 项目 | 对称 AbsMax（引擎） | 非对称 zp（参考） |
|---|---|---|
| 码空间 | `[-128, 127]` | `[0, 255]` |
| zero_point | 恒 0 | `round(-lo/s_a)` |
| scale 分母 | `qmax=127` | `2^bits-1=255` |
| 0 的码字 | 0 | `zp` |
| padding 码字 | 0 | `zp` |
| GEMM 修正 | 无 | 累加前减 `zp` |
| 06 引擎 | 支持 | **不支持** |

同一层混用两套 scale/码字会产生整体偏移或 2 倍分辨率误差，必须显式选择。

---

## 3. Conv2d 规格

### 3.1 输出形状

dilation 固定 1：

```
OH = floor((H + 2*ph - kh) / sh) + 1
OW = floor((W + 2*pw - kw) / sw) + 1
```

`stride`、`padding` 支持 int 或 `(h, w)`。输出必须 `OH > 0, OW > 0`。

### 3.2 分组

- 校验：`groups >= 1`、`C % groups == 0`、`Cout % groups == 0`、
  `Cin/groups == w.shape[1]`。
- 第 `g` 组使用输入通道 `[g*Cin/groups, (g+1)*Cin/groups)`，
  输出通道 `[g*Cout/groups, (g+1)*Cout/groups)`。
- `groups == C == Cout` 且 `Cin/groups == 1` 是 depthwise，走同一语义；
  引擎另有专用 kernel，但数值结果必须与通用分组路径一致。

### 3.3 im2col 顺序与 Kpad

```
k = (c * kh + i) * kw + j          # c 为组内通道，i/j 为核内坐标
K = (Cin/groups) * kh * kw
Kpad = ceil(K / 8) * 8             # 引擎为 int8 GEMM 对齐
```

- `k >= K` 的补位写码字 0（乘 0，不影响累加）。
- 权重按同一顺序展平：`w[co].reshape(-1)[k]`。
- padding 位置写 `zp`（对称即 0），语义是真实 0，不是「把 0.0 量化后的值」；
  在对称路径两者恰好相同，在非对称路径不同。

### 3.4 整数累加与反量化

```
acc[n,co,oh,ow] = Σ_{c,i,j} (code_a[n, c, ih, iw] - zp_a) * code_w[co, c, i, j]
ih = oh*sh + i - ph
iw = ow*sw + j - pw
y = acc * (s_a * s_w[co]) + bias[co]
```

- 累加用 int32，必须精确；反量化/加 bias 用 float32。
- 输出布局与 PyTorch 一致：`[N, Cout, OH, OW]`，最内层 `ow`。

---

## 4. 激活与残差

### 4.1 ReLU / ReLU6

反量化后、fp32 域执行：

```
ReLU(x)  = max(x, 0)
ReLU6(x) = clamp(x, 0, 6)
```

边界语义（单测锁定）：

| 输入 | -1 | -1e-6 | 0 | 6 | 6+1e-6 | 100 |
|---|---|---|---|---|---|---|
| ReLU6 | 0 | 0 | 0 | 6 | 6 | 6 |

### 4.2 残差

- 主支路最后一个 conv **不在 conv 内做激活**；等 identity 相加后再做后处理。
- ResNet（引擎 `residual=1`）：`y = ReLU(main + identity)`。
- MobileNetV2（引擎 `residual=2`）：`y = main + identity`，add 后不激活。
- 激活顺序写反（先 ReLU 再 add）会改变负半轴结果，单测显式覆盖。

---

## 5. GEMM 规格

```
A: [M, K] 激活码字（int8 或 uint8）
W: [N, K] 权重码字（int8），每行一个输出通道
acc[M,N] = (A - zp_a) @ W^T          # int32
Y[M,N]   = acc * (s_a * s_w[N]) + bias[N]
```

- 权重按 `[N, K]` 行主序，与 conv 的 `[Cout, K]` 展平一致。
- `Kpad` 补零列不改变结果（零列乘任何码字仍为 0）。
- GEMM 不含 padding/stride/groups 语义，这些在 im2col 阶段完成。

---

## 6. 参考实现与单测映射

参考 API（`ptq_toolkit/ref_int8.py`）：

| API | 作用 |
|---|---|
| `SymActQuant.from_range(lo, hi, bits, mode="absmax")` | 对称 AbsMax（`mode="minmax"` 仅对照） |
| `AsymActQuant.from_range(lo, hi, bits)` | 非对称 zp 参考语义 |
| `quantize_weight_per_channel_sym(w, bits)` | W8/W4 per-channel 权重 |
| `conv2d_int8(...)` | 零填充/步长/分组/depthwise + int32 累加 |
| `gemm_int8(...)` | int32 GEMM + per-channel 反量化 |
| `add_residual(...)` / `relu6(...)` | 残差与 ReLU6 后处理 |

单测覆盖（`tests/test_int8_ref.py`，63 项断言）：

| 验收点 | 测试 |
|---|---|
| 对称 AbsMax vs 非对称 zp：scale、码空间、零点、码字 | `test_activation_modes` |
| 跨零范围无裁剪、误差 ≤0.5×scale | `test_activation_modes` |
| half-away 舍入与码字裁剪 | `test_rounding_and_clamp` |
| W8/W4 per-channel scale、全零通道、码字范围 | `test_weight_quantization` |
| 零填充 / 步长 / groups=1,2 / depthwise | `test_conv_variants` |
| padding 补真实 0（非对称补 zp） | `test_padding_is_real_zero` |
| GEMM int32 累加与反量化 | `test_gemm` |
| 残差 add 顺序、ReLU6 边界 | `test_residual_relu6` |
| groups/类型/范围校验 | `test_validation` |

运行：

```bash
cd /mnt/d/WorkBuddy/ShowBro/lab/09-ptq-toolkit
PYTHONPATH=. /home/user/pyenv-torch/bin/python tests/test_int8_ref.py
# ALL PASS
```

---

## 7. 与 06 引擎的已知差异

1. 06 的 `quant_im2col_kernel` 当前把 `zp` 加进 `round(x/scale)` 后又直接
   clamp 到 int8 范围，没有减 `zp`；由于引擎 `zp` 恒为 0，数值上与本文
   2.2 节一致。**不要把非对称参数传给 06**，否则会得到整体偏移。
2. 06 的 `Kpad` 对齐到 8；参考实现不做 K 对齐，但 K 维补零不改变结果。
3. 06 的 depthwise 使用专用 kernel；本文要求其数值与通用分组路径一致，
   参考实现用通用路径覆盖。
4. 06 的 bias 在 requant 阶段以 fp32 相加；本文同口径。

---

## 8. 结论

- 生产路径只有一种：**对称 AbsMax + per-channel 对称权重 + int32 累加 +
  fp32 反量化/bias + 明确的 ReLU/ReLU6/残差顺序**。
- 非对称 zp 作为参考语义保留，用于验证「补 zp / 减 zp / 码空间」三处
  容易混淆的细节；不得进入 06 引擎。
- 63 项断言全部通过后才允许作为 I2 敏感度实验的算子基线。
