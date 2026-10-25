# 阶段 3 · 扩展 ncnn 的量化语义（int4 位宽 × 非对称激活）

> 状态：**设计定稿，未开工**
> 前置：阶段 2（PTQ 配置迁移进 ncnn）已完成，桥打通、三策略忠实
> 判据实验：`03-ptq-lab/experiments/e8_sym_gap.py`（commit `9ca6fab`）

---

## 0. 结论先行

| 问题 | 答案 |
|---|---|
| 要不要做 | **做**。判据 +5.37pp，远超阈值 |
| 做什么 | 给 ncnn 加 **sub-8bit 通路**，且该通路**必须原生支持非对称激活** |
| 为什么不是"只加位宽" | 位宽与非对称是同一件事的两面：8bit 下丢 zp 免费（−0.11pp），4bit 下是悬崖（−5.37pp） |
| 改造面 | **3 处**做非对称 + **4 处**做 int4，**不需要动 eltwise** |
| 工作量 | 6–8 周（P0–P3 四阶段，见 §4） |

### 判据数据（CIFAR-10 + ResNet-18，minmax 校准，10000 张）

| 配置 | asym（非对称） | sym（对称） | 差值 |
|---|---|---|---|
| W8A8 | 87.08% | 87.19% | **−0.11pp**（噪声内） |
| W4A8 | 85.34% | 85.36% | −0.02pp |
| **W4A4** | **83.98%** | **78.61%** | **+5.37pp** |

**读法**：8bit 下对称量化的 SQNR（≈43 dB）远超模型所需，丢一半码字无所谓；4bit 下对称只剩 ≈20 dB，跌破悬崖，同样一半码字变成 5.37pp。

**这解释了**：为什么 TFLite/LiteRT 规范与 US 12430542 专利敢放心把 zp 折进 bias（8bit 下确实免费），为什么 ncnn 至今只有对称 int8（8bit 下够用）。**该结论只在 8bit 成立。**

---

## 1. 现状勘察（全部为源码实证）

| 环节 | ncnn 现状 | 源码位置 |
|---|---|---|
| 输入量化 | 对称，**无 zero_point** | `mat.cpp:2444` 只传 `scale_data`；`quantize.cpp:29-36` `float2int8` 截在 ±127 |
| 码空间 | `[-127/s, +127/s]`，0 恒在网格上 | `src/layer/quantize.cpp:29` |
| padding 填充 | `pad_value`（默认 0），对 int8 blob 转 `signed char` | `convolution.cpp:390` `copy_make_border(..., BORDER_CONSTANT, pad_value, ...)` |
| int32 累加 | **真整数流水线**，不回落 fp32 | `convolution_x86.cpp:1117` `convolution_packed_int8 → top_blob_int32` |
| 输出 requantize | **已带 `bias_data` 参数** | `mat.cpp:2468`；`requantize.cpp:88-99` `v = acc*scale_in + bias` |
| fused requantize 条件 | **仅 Conv/ConvDW → Conv/ConvDW 直连** | `ncnn2int8.cpp:900-916`（中间不能有任何其他层） |
| 残差 add | **无 int8 通路** | `eltwise.cpp` 全文无 `elemsize`/`int8` 分支；`ncnn2int8.cpp` 不含 `Eltwise` |
| sub-8bit | **不存在** | `ncnn2table.cpp:557-577` 的 `quant_6bit` 只是 int8 Winograd F43 防溢出 hack，权重仍存 int8 |

### 1.1 关键收窄：残差不用改

原计划要处理"残差 add 两端 (scale, zp) 不同"。勘察后**这个担心不成立**：

- fused requantize 只允许 Conv→Conv 直连，中间有 `Eltwise` 就断链
- `ncnn2int8` 根本不把 `Eltwise` 纳入量化，它两侧的 conv 输出保持 fp32

所以 **ResNet 的残差天然在 fp32 域完成**，非对称在残差处免费。`eltwise.cpp` 一行都不用碰。这砍掉了原估计的 2–3 周。

---

## 2. 非对称激活落地方案（3 处）

### 2.1 数学：zp 折叠到哪里

记输入 `x_real = (q_x − z_x)·s_x`，权重对称 `w_real = q_w·s_w`。卷积输出（先忽略 bias）：

```
y = Σ (q_x − z_x)·s_x · q_w·s_w
  = s_x·s_w · ( Σ q_x·q_w  −  z_x·Σ q_w )
         ↑ int32 累加器        ↑ 编译期常量
```

`Σ q_w` 是每个输出通道的权重和，**离线可算**。于是：

```
bias_requantize = bias_orig + ( − z_x · Σ q_w ) · s_x · s_w
```

而 `Requantize` 层已有 `bias` 参数、公式正是 `v = acc·scale_in + bias`。**因此折叠零运行时开销，且不需要给 Requantize 加任何新参数。**

### 2.2 三处改动

| # | 改动 | 具体做法 | 难度 |
|---|---|---|---|
| 1 | `Quantize` 层支持 zp | 新增 param 携带 `z_in`，`forward` 里 `q = round(x/s) + z_in`，clamp 到 `[0, 2^bits−1]` | 低 |
| 2 | `ncnn2int8` 算 zp 并折叠 | 从表里读 `z_in` → 算 `Σ q_w` → 写 `bias_data`；同时把 conv 的 `pad_value` 设为 `z_in` | 中（要动工具链） |
| 3 | `make_padding` 填 `z_in` | **无需改代码结构**。`pad_value` 本来就是 conv 的 param，ncnn2int8 设值即可 | 低 |

> ⚠️ 第 3 处的坑：`pad_value` 是 `float` 成员，对 `elemsize==1` 的 blob 走 `static_cast<signed char>`。必须保证 `z_in` 是整数且落在 `[-128,127]`，否则静默截断。要加断言。

### 2.3 与专利的边界（写论文时必须守住）

上述三处**正是 US 12430542 权利要求逐条覆盖的内容**（bias 折叠、zero-padding → zero point padding、add 的 BiasC）。

**因此它们只能作为"实现手段"写在系统描述里，绝不能当作 contribution。** 我们的贡献是下一节那条边界发现：

> 折叠在 8bit 上免费、在 sub-8bit 上失效（−5.37pp）。工业规范与专利都在 8bit 上制定，未覆盖该失效边界。

---

## 3. int4 落地方案（4 处）

| # | 改动 | 说明 |
|---|---|---|
| 1 | 权重打包存储 | 现为 `signed char`/权重。int4 需 2 权重/byte。建议新增 `weight_data_int4` + `int4_scale_term`，与 int8 并存而非替换 |
| 2 | int4 kernel | P0 先做参考实现（解包 → int8 域累加），P1 再做 x86 AVX2 / ARM NEON。可复用 `convolution_packed_int8` 的 im2col 骨架 |
| 3 | 位宽语义进 param | 现 `int8_scale_term` 只有 0/1/2/101/102 五个值。需扩展为可表达位宽（建议新增独立 param，不破坏既有语义） |
| 4 | 混合位宽调度 | 按 ptq-lab e5 的 `per_layer_w` 分配，落到 param 的每层位宽字段 |

**不要一上来写 SIMD。** 参考实现跑通、精度对齐 ptq-lab 之后，再优化性能——否则精度和性能两个 bug 会混在一起，无法定位。

---

## 4. 分阶段与验收

| 阶段 | 内容 | 验收标准 | 预估 |
|---|---|---|---|
| **P0** | int4 参考实现（纯 C++，无 SIMD）+ ncnn2int8 支持 int4 表 | ncnn int4 对称跑出 **78.61%**（复现 ptq-lab sym） | 1.5 周 |
| **P1** | 非对称激活三处改动 ① ② ③ | ncnn int4 非对称跑出 **83.98%**（复现 ptq-lab asym） | 1.5 周 |
| **P2** | x86 AVX2 int4 kernel（或 ARM NEON） | 相对 fp32 有可测加速比，精度与 P1 一致 | 2–3 周 |
| **P3** | 混合位宽调度 + 端到端评测 | 在同平均位宽下优于单一位宽（对照 e5 的 `MIXED_MAX_AVG_BITS=5.0`） | 1.5 周 |

**P0/P1 的验收数字是硬指标**：78.61% 和 83.98% 是 ptq-lab 已实测的 ground truth。ncnn 复现不出来就是实现有 bug，不接受"差不多"。

---

## 5. 已知风险

| 风险 | 影响 | 应对 |
|---|---|---|
| `pad_value` 静默截断 | padding 填错值，精度下降但不报错 | 加断言；用单层 diff 测试对齐 |
| fused requantize 与非对称叠加 | 折叠项在 `int8_scale_term>100` 路径下要重算 | P1 只在 `≤100` 路径验证，P2 再扩 |
| int4 打包的内存对齐 | 读越界或 kernel 崩溃 | 先按 32 元素对齐填充 |
| x86 AVX2 无 int4 原生指令 | 只能用 int8 指令模拟，加速比受限 | 预期加速来自**内存带宽**（权重体积减半）而非算力；实测后如实报告 |
| 基础侧：SIMD 不熟 | P2 可能超期 | 允许 P2 降级为"仅参考实现 + 理论分析"，不与 P0/P1 绑死 |

---

## 6. 复现命令

```bash
# 判据实验（本设计的数据来源）
cd /mnt/d/WorkBuddy/ShowBro/lab/03-ptq-lab
~/pyenv-torch/bin/python experiments/e8_sym_gap.py --mode sqnr   # 分钟级
~/pyenv-torch/bin/python experiments/e8_sym_gap.py --mode e2e    # 小时级

# 现状基线（阶段 2 的桥）
cd ../05-ncnn-deploy
bash scripts/rebuild_ptq_tables.sh
```

---

## 7. 一句话定位（给简历/答辩用）

> 在对称-only 的端侧推理引擎 ncnn 上打通 sub-8bit 量化，实测并定位了"zero_point 折叠在 8bit 免费、在 4bit 失效 5.37pp"的边界，据此实现非对称激活通路（bias 折叠 + padding 零点 + requantize 协同）与混合位宽调度。

**不能写成**："我提出了非对称量化" / "我发明了 zp 折叠" —— 均为 prior art。
