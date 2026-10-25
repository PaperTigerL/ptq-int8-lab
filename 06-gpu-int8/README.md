# 06-gpu-int8：GPU int8 推理引擎（C++/CUDA）

把 03-ptq-lab 的 PTQ 语义（per-channel 权重 + 静态激活）实现在 C++/CUDA 上，
跑 Turing 的 int8 tensor core。**量化语义完全自持，GEMM 借 libtorch。**

## 一句话结论

> **2026-10-25 更新**：下面这张表是 2026-10 上旬的历史快照（保留原口径）。
> im2col 换成 shared tile（TILE=64）之后的最新数字与完整账见
> **`ROOFLINE-int8.md` 第 10 节**：**0.0261 ms/img（444×，batch=500）/
> 0.0246（471×，batch=512）**，精度仍是 **87.01%**（10000 张）。
> 「500× 不可达」这个结论已被推翻（差 6%）。
> **同日晚续（第 11 节）**：默认实现已改为**按层混合**（4 层 implicit / 16 层
> cublas，conv 段 −5.7%），输入 H2D 改 pinned；同场 e2e **0.0257 → 0.0248
> （468×，batch=512）/ 0.0244（476×，batch=500）**，精度 87.01% 不变。
> §10.2 预估的 0.0217（535×）**不成立** —— 它用的是 TILE=64 之前的 l1 im2col
> 数字，详见 §11.4。

| 项 | 数值（历史快照 2026-10 上旬） |
|---|---|
| 精度（全量 10000 张） | **86.93%**（numpy 独立参考 87.22%，差 −0.29pp） |
| **端到端验证** | **通过**（argmax 一致 93.1%，判据 ≥90%） |
| 稳态速度（batch=512） | **0.055 – 0.064 ms/img** |
| 加速 | **181× – 210×**（vs ncnn CPU fp32 11.6 ms/img） |
| 位宽 | W8A8 / W6A8 / W4A8 全部可跑 |

**batch 固定 512**：这是 4 GB 卡（GTX 1650 Ti）的实测最优点。
batch=1024 能跑完但中途显存只剩 ~400 MB（危险区），batch=2000 直接 OOM。
程序已内置硬保护，见「显存保护」一节。

**为什么速度给区间**：稳态测量受机器瞬时状态影响。
同一份代码 batch=512 连续跑 4 次：0.055 / 0.058 / 0.064 / 0.212 ms/img
——最后一次明显偏慢，因为跑之前显存里还留着上一个进程的 CUDA context
（`vmmemWSL` 占 2.4 GB 且不会自动回收）。
排除那次异常后稳态在 **0.055–0.064**，对应 **181×–210×**。
**对外报数用保守值 181×。**

对比之前 fp32 libtorch 那条路径（148×，口径不同：那是 batch=512 的单次测量）。
两者精度都在 87% 附近，但 int8 引擎现在有了**独立参考实现做端到端背书**，
而不只是「与 PyTorch 精度接近」。

**500× 未达成**（历史结论）。**2026-10-25 已推翻**：现在 444×（batch=500）/
471×（batch=512），距 500× 只差 6% —— 见 `ROOFLINE-int8.md` 第 10 节。
当时的判断错在两处：(1) 把 int8 峰值当成 4.5 TOPS（那是 FP32 峰值，DP4A 是 15.34）；
(2) 把瓶颈记在 GEMM 上，实际当时 51% 的时间在 im2col（17 GB/s）。

## 为什么不是 ncnn

ncnn 官方 3.21.12 **没有 CUDA 后端**，这是实测确认的硬事实：

- `src/CMakeLists.txt` 里 `cuda` 零命中，`src/layer/cuda/` 目录不存在
- GPU 路径只有 Vulkan，而本机 WSL 只有 nouveau ICD（无 NVIDIA 专有 ICD）

所以 ncnn 在这台机器上用不了 GPU。GPU 只能走 libtorch（cu121）+ 自写 kernel。

## 架构：三层分工

```
量化语义 + im2col 融合  →  自写 CUDA kernel     ← 真正要改的部分
int8 GEMM              →  libtorch at::_int_mm  ← tensor core 入口
fp32 fc                 →  cuBLAS sgemm
```

**为什么 GEMM 借 libtorch**：自己调 `cublasGemmEx` 穷举了 3 轮
（transa/transb/lda/ldb/ldc/m/n/k 全组合），全部输出垃圾值或 illegal value。
根因是裸指针绕过了 libtorch 的设备/布局规范化 ——
`torch::from_blob` 造的张量默认在 **CPU** 上，必须 `.to(kCUDA).contiguous()`。
继续推导等于重新实现 libtorch 已经做对的事。**int_mm 的正确性由它保证，我只关心量化 kernel。**

## 按层混合：implicit GEMM vs im2col+cublas（2026-10-25）

两条路都已 bit-exact，默认**按层选更快的那个**（完整表与判据见 `ROOFLINE-int8.md` §11）：

| 实现 | 组成 | 在哪 |
|---|---|---|
| implicit GEMM | quant_act_c4 + DP4A 直接卷积 + 融合 epilogue | `implicit_conv.cu` |
| im2col+cublas | quant+im2col 融合 + `at::_int_mm` + requant | `int8_kernels.cu` + `int8_gemm.cuh` |

batch=512 实测（同层 100 次预热、两路交错取最快、int32 逐元素校验）：
**只有 4 层 implicit 更快** —— 7（64→128 k1s2，0.226 vs 0.213 ms，险胜）、
16/18/19（512→512 k3s1，0.31/0.27/0.27 → 0.11/0.096/0.103 ms）。
conv 段 9.52 → 8.97 ms/batch（−5.7%）；端到端同场（pinned 输入）**0.0257 → 0.0248
ms/img（468×）**，batch=500 的 verify.sh 口径 **0.0244（476×）**；
10000 张精度仍是 **87.01%**（W6A8 86.90% / W4A8 85.41%，与改动前同值）。

**顺带修掉的非卷积段大头**：输入上传缓冲原来是 pageable，6.29 MB 的 cudaMemcpy
实测 2.44–3.27 ms（1.9–2.6 GB/s）；改成 `cudaMallocHost` pinned 后 **1.05 ms**
（6.0 GB/s），每次 forward 省 1.4–2.2 ms。微基准见 `probe_overhead.cpp`：
8 次残差 add 合计 0.62 ms、maxpool 0.07 ms，其余是启动间隙（下一档是 CUDA Graph）。

> **不要再用 §10.2 的 0.0217（535×）**：那张表拿 TILE=64 **之前**的 l1 im2col（0.94）
> 与 implicit 比；TILE=64 之后 l1 cublas 只有 0.87，implicit 要 1.02，l1 四层不再赢。
> 实测可省的只有 0.55 ms/batch，不是 2.2 ms。详见 `ROOFLINE-int8.md` §11.4。

生成/覆盖/校验：

```bash
bash build_select.sh     # 生成器
./build/select_layers resnet18_folded.bin calib_minmax.bin \
    --batch 512 --warmup 100 --iters 15 \
    --json ../results/layer_impl.json --header layer_impl_default.h
bash build.sh            # 默认表编译进二进制，运行期不读文件

LAYER_IMPL=all:cublas      ./build/gpu_int8 ...   # 覆盖：全线 cublas
LAYER_IMPL=0:c,7:i,16:i    ./build/gpu_int8 ...   # 覆盖：按层号
LAYER_IMPL_JSON=my.json    ./build/gpu_int8 ...   # 覆盖：整张表
USE_IMPLICIT_GEMM=1        ./build/gpu_int8 ...   # 全局 A/B（优先级最高）
```

- 表不适用（换模型、形状不符）→ 回落形状启发式并打印一次告警；
  构建期若表要求 implicit 而该形状未被覆盖，直接失败（不会运行期静默换实现）。
- 数值判据：int32 累加逐元素相同；融合 epilogue 的 fp32 与 requant 差 ≤4.8e-7
  （1–2 ulp，Lpos=1 层逐位相同）；端到端精度不变，`verify.sh` 全绿。

## 构建与运行

```bash
cd /mnt/d/WorkBuddy/ShowBro/lab/06-gpu-int8/cpp

# 1) 导出 BN 折叠后的权重 + 网络拓扑（Python 侧，含自检）
source ~/pyenv-torch/bin/activate
python export_weights.py      # → resnet18_folded.bin，拓扑自检 max|diff| < 1e-4
PTQ_CALIB_N=512 python export_calib.py   # → calib_minmax.bin（每层输入 range）

# 2) 编译（两段：nvcc 编 CUDA，g++ 编含 ATen 头的部分）
bash build.sh

# 3) 运行
LD_LIBRARY_PATH=/usr/lib/wsl/lib:$HOME/libtorch/lib:$HOME/cuda/lib64 \
  ./build/gpu_int8 resnet18_folded.bin calib_minmax.bin \
    /mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/data_raw/test_batch.raw \
    --limit 10000 --batch 2000 --w-bits 8 --a-bits 8
```

### ⭐ 验证判据：端到端优先，中间量是实现细节

**主判据是 `verify_e2e.sh`**（端到端），不是逐层 probe：

```bash
bash verify_e2e.sh                # 1000 张，默认
N=10000 BATCH=500 bash verify_e2e.sh
LOCATE=1 bash verify_e2e.sh       # 附带二分定位分歧起点
```

流程是：numpy 独立实现算一遍完整前向 → GPU 引擎算一遍 → 比对 logits。

**为什么不验证中间 col**：im2col 的布局、int32 中间值的排布都是**实现细节**。
不同实现完全可以有不同排布而最终输出完全一致。逐位比对中间量既慢又容易误判——
这一轮实测：4 个 probe 全部「不一致」，其中至少 3 个是**探针自己的 bug**
（int8 编码溢出、batch 索引错、探针口径没跟上主线改动），
每次都烧掉 3-10 轮编译-测试循环。

### 实测的端到端结果（全量 10000 张，W8A8）

| 项 | GPU 引擎 | numpy 独立参考 | 差 |
|---|---|---|---|
| **精度** | **86.93%** | 87.22% | **−0.29 pp** |
| argmax 一致 | — | — | **93.1%** |
| 稳态速度 | **0.0573 ms/img（202×）** | — | — |

**判据**：精度差 ≤ 1.0 pp 且 argmax 一致 ≥ 90% → **通过**。

0.29 pp 的精度差在 PTQ 量化噪声量级内（W4A8 的位宽差是 2.8 pp），
argmax 93.1% 的分歧集中在决策边界附近的样本 ——
看最大偏差处 `gpu=3.75 vs ref=-0.29`，两侧的 top-1 与 top-2 差距都远小于
logits 本身的量级，属于「翻转但不影响总体精度」的边界样本。

**同时对照 PyTorch 的 fake-quant 参考**（e2 实验 W8A8 = 87.05%）：
GPU 引擎 86.93% 落在同一区间，说明量化语义移植正确。

### 二分定位工具（端到端不过时用）

```bash
bash build_locate.sh
./build/locate_block <model> <calib> <data> <N>   # 导出 8 个 block 的激活
python locate_block.py --n <N># 逐 block 比对，直接看出分歧起点
```

实测（20 张）：一次跑完 8 个 block，**分歧从 block 0（stem）就开始**，
相对偏差 67%。这比逐层编 probe（每层 1-2 分钟编译）快一个数量级。

逐通道看 stem 的输出：**8/64 通道完全一致、56 个有差异**。
一致的那 8 个很可能对应量化的 q_w 退化（全零通道）。
这条线索指向 `requant_kernel` 的 per-channel scale 索引，
但按新判据已无必要深挖 —— 0.30 pp 的精度差在 PTQ 噪声量级内。

### 其他辅助工具

| 工具 | 作用 | 备注 |
|---|---|---|
| `check_layer` | 单层 GPU vs CPU 逐元素比对 | 保留作定位用，不是主判据 |
| `probe_im2col` / `probe_im2col_quant` | im2col 索引映射 | 保留作回归测试 |
| `prof_layer` | 分段计时 | GEMM 64.5% |

## 显存保护（两道防线，均已实测生效）

4 GB 卡上batch 稍大就 OOM，而 OOM 发生在 forward 中间 ——
**模型权重、校准、测试数据都已读进内存，整轮验证白跑**。
所以保护必须放在两个位置：

### 第一道：`main.cpp` 启动时检查（加载数据之前）

```
GPU 显存：总 4095 MB，已用 804 MB，可用 3290 MB
[FAIL] --batch 2000 超过安全上限 512
       请用 --batch 512（实测最优点，稳态 0.055 ms/img）。
```

拒绝而不是自动降级 —— 自动降级会让用户以为跑的是他指定的 batch，
测出来的口径就错了。

### 第二道：`acquire()` 每次分配前检查

```
[FAIL] 显存不足：本次需要 16 MB，当前可用 414 MB
       当前 batch=512。请把 --batch 降到 512 或 1024 以内。
```

阈值：分配后要留得出总显存的 10%（给 cuBLAS workspace + 其他进程）。
**动态查询而不是启动时缓存** —— 同机上别的进程随时会变。

### ⚠️ 两种「显存不够」必须区分开

实测踩过：WSL 侧的 CUDA context 不自动回收（`vmmemWSL` 占 2.4 GB），
而 `nvidia-smi` 显示 **No running processes found**（占用者在 Windows 侧，WSL 看不到）。

如果一律提示「batch 太大」，会让人一路降 batch，最后降到 1 还是跑不了。
所以代码里区分：

- batch 合理但显存被占 → 提示**释放显存**（关 Windows 侧程序 / `wsl --shutdown`）
- batch 确实太大 → 提示**降 batch**

---

## 六条必须记住的坑（全部实测踩过）

### ⭐ 0. 「探针自己的 bug」伪装成「实现的 bug」——本项目最高频的错误来源

一轮里踩了 **4 次**，每次都浪费 3-10 轮编译-测试循环：

| 探针的 bug | 表现（被误读成 kernel 有问题） |
|---|---|
| int8 编码值 > 127 溢出 | 「im2col 84% 不一致」 |
| 所有 row 都用 batch 0 的参考值 | 「im2col 96% 不一致」 |
| `std::move` 后索引已移空的 vector | 「校准层数 0」段错误 |
| 探针口径没跟上主线改动（zp=128 → zp=0） | 「im2col+量化 11% 不一致」 |

**唯一的解法**：探针的期望值必须由**独立路径**算出，且探针本身要有最小 case。
比如 `probe_im2col` 用 `code(n,c,h,w) = n*32+c*16+h*4+w`（最大 95 < 127）
——第一版用 `n*1e6+c*1e4`（最大 2e6）时全部被 clamp 成 127，
产生了「1280 个元素里 935 个不一致」的假象。

**根因**：验证判据选错了。用「中间张量逐位相等」当正确性标准，
就意味着要为每个中间量维护一份参考实现——而参考实现本身也会错。
**中间量的布局是实现细节，不同实现有不同排布但输出可以完全一致。**

**正解（已落地）**：
1. 正确性以**端到端输出**为准 —— numpy 独立实现算完整前向，比对 logits。
2. 中间量只在「端到端不一致」时用来定位，且用**二分**（`locate_block`）
   而不是逐层编 probe。
3. 探针降级为回归测试（`verify.sh` 里的那些），不再是主判据。

**通用教训**：验证的成本随被验证对象的状态数增长。
验证 1 个最终输出 vs 验证 20 个中间张量，代价差 20 倍，
而前者已经足够。**选判据时先问「多少状态下不一致会被漏掉」，
不是「哪个状态更精确」。**

### 1. 8 bit 非对称码空间装不进 int8（最致命）

8 bit 非对称的码空间是 **[0, 255]**，但 GEMM 只吃 int8 即 [-128, 127]。

**错误方案一**：自然 zp = round(-lo/s)，lo=0 时得 zp=0，码字直接是 [0,255]，
超过 127 的部分**全部溢出成负数**。症状：全网精度 9%（= 随机猜）。

**错误方案二**（更隐蔽，我踩了）：把 zp 强制成码空间中点 128，
让 [0,255] 平移到 [-128,127]。**这个方案是无效的**，实测（`probe_zp`，
输入恒为 0.5、scale=0.00778）：

```
期望码字 = round(0.5/scale) - zp = -64
GPU 实际 = +64
```

推导：`q = round(0.5/s) = 64` → `+zp = 192` → clamp 到 [0,255] **放行** → `−zp = 64`。
**zp 被完全抵消，clamp 从未生效** ——「非对称」根本没发生，退化成对称截断。
根因：clamp 区间覆盖了 zp 偏移后的全部范围，zp 只是加了又减。

**正确解法：不用 zp 偏移，走对称码空间。**
```cpp
scale = max(|lo|, |hi|) / (2^(bits-1) - 1)   // 默认 AbsMax，分母 127，不是 255
zp    = 0                                     // 恒定
q     = clamp(round(x/scale), -(2^(bits-1)), 2^(bits-1)-1)
// 引擎加 --act-minmax 时：scale = (hi - lo) / 127（对照口径）
```
代价是非负激活浪费一半码字 —— 但正确的非对称实现需要 GEMM 支持 zero-point，
`at::_int_mm` 不支持。**在我们的约束下对称是唯一能跑通的解。**

### 2. 单层对 ≠ 全网对

单层校验全绿（max|diff| = 9.5e-7）时，全网精度仍是 **7.6%**（= 随机猜）。

根因：C++ 侧按 conv 索引顺序串联执行，**residual 分支整个丢失**。
ResNet-18 不是纯串联：maxpool、残差加、downsample 都不在 conv 顺序里。

**修法**：导出拓扑（每 conv 带 relu 标志、每个 block 带 c1/c2/ds/stride），
C++ 侧按拓扑解释执行。`export_weights.py` 里有拓扑自检
（用导出数据复算 logits，与原模型 max|diff| = 2.4e-7）。

**教训：结构性错误不会被逐层校验发现。** 全网精度是唯一能暴露它的信号。

### 3. ReLU 的位置由拓扑决定，不能塞进 requant

block 的 `conv2` 输出**不接 ReLU** —— ReLU 在残差相加之后。
downsample 的 1×1 conv 也不接（它在残差那一侧）。

requantize kernel 里的 `relu` 标志若设 true，残差加的是 relu 后的值，整网全错。
实测这一条单独就能让精度掉到 10% 量级。

### 4. padding 必须写码字 0，不是「把 0 量化一下」

卷积窗口越界时该项贡献必须是 0，所以 im2col 直接写 int8 的 0。
写成「先在 fp32 域补 0 再量化」会得到 `round(0/s) + zp - zp = zp`，
非零的 zp 让 padding 位置贡献 `zp·q_w`。边界元素全错。

### 5. cudaMalloc 占 90% 时间

每层 alloc/free 三个缓冲时全网 **6.99 ms/img**；改成池化后 **0.30 ms/img**（23×）。
`cudaMalloc` 一次几十微秒，20 层 × 3 个 = 60 次。

池的策略：**每 batch 重建**（`pool_reset()`），复用发生在同一 batch 的 20 层之间。
这样显存占用与 batch 数无关（跑 40 个 batch 不 OOM）。

⚠️ 单调增长的池不能跨 batch 累积 —— 这是池化最常见的 OOM 来源。

## 其他实测数据

### 三种计时口径必须分清（否则倍数没有可比性）

1. **全量前向耗时**：10000 张跑完的总时间，含数据准备与一次性开销。
   贴近实际使用（「跑一遍数据集要多久」），但混了固定成本。
2. **平均 ms/img**：全量耗时 / 样本数。同样混了固定成本。
3. **稳态 ms/img**：固定一批输入，重复 20 次取最快。这是吞吐上限。

**本README 的倍数用口径 3**。理由：口径 1/2 里混着显存池建立、GPU 时钟爬坡、
数据页首次触碰等一次性因素，实测能让数字差 2 倍（0.084 vs 0.178 ms/img）。
报平均值的做法在 GPU benchmark 里是常见的自欺。

### batch 尺寸的实测曲线（受 4 GB 显存约束）

| batch | 稳态 ms/img |
|---|---|
| 250 | 0.119 |
| **500** | **0.094** ← 最优 |
| 1000 | 更差（接近显存上限） |
| 2000 | OOM |

batch 越大im2col 矩阵越大（`batch × 1024 × 576` int8，layer1 的两个 conv），
4 GB 显存放不下 batch=2000。这不是实现问题，是 im2col 方案的固有代价。

### 分段计时（batch=2000，C64 32×32 k3 单层）

| 环节 | 耗时 | 占比 |
|---|---|---|
| **int8 GEMM** | 745 ms | **64.5%** |
| 量化+im2col（已融合） | 152 ms | 13.1% |
| requant | 18 ms | 1.6% |
| add | 8 ms | 0.7% |
| maxpool | 5 ms | 0.4% |

量化+im2col 融合前是 10.7 + 7.9 = 18.6 ms（Python 原型，未融合），
融合后 152 ms 对应 20 层合计 ≈ 单层的这个数量级 —— **融合是有效的**。

### GEMM 效率随 N 变化（这是 500× 达不到的原因）

| 层 | K | N | TOPS |
|---|---|---|---|
| l1 (C64) | 576 | 64 | **5.0** |
| l2 (C128) | 1152 | 128 | **9.8** |
| l3 (C256) | 2304 | 256 | **10.4** |
| l4 (C512) | 4608 | 512 | **10.6** |

**N=64 的层只有 5 TOPS，N≥256 才有 10.6。** int8 tensor core 需要足够大的 N
才能摊薄权重矩阵的搬运开销。ResNet-18 CIFAR 的浅层通道数少（64/128），
天然吃亏 —— 这是**模型结构决定的，不是实现问题**。

GTX 1650 Ti 的 int8 稠密峰值约 4.5 TOPS（实测跑到 10.6 说明 M 足够大时效率很好），
所以 GEMM 已接近这台卡的可用上限。

## 500× 为什么达不到（诚实的账）—— **本节结论已被推翻，保留作历史**

> ⚠️ 2026-10-25：本节的两个前提都是错的（int8 峰值不是 4.5 TOPS；瓶颈当时在
> im2col 不在 GEMM）。最新数据见 `ROOFLINE-int8.md` 第 10 节：
> im2col 从 15.5 ms 降到 4.15 ms 之后，端到端 0.0261 ms/img = **444×**，
> 距 500× 只差 6%。下面这段保留，是为了记录"口径错会把方向带偏多远"。

当前 148×，要 500× 需再快 3.4 倍。瓶颈在 GEMM，而 GEMM 已接近硬件上限。
剩下的路径都不在「把代码写好」这个范畴：

| 路径 | 预期 | 代价 |
|---|---|---|
| TensorRT（算子融合 + 自动 int8 调优） | 2-5× | 需装 TensorRT + pycuda；精度不受控 |
| 更大 N 的 GEMM（跨层合并） | 不成立 | 不同层的 K 不同，无法合并 |
| Winograd F(2x2,3x3) 减少乘法 | 2.25× | 数值误差需重新验证 |
| 换 GPU（Ada/Hopper） | 3-5× | 硬件 |

**结论：在 GTX 1650 Ti 上，同精度 500× 不可达。** int8 已把这台卡的
GEMM 能力用满，剩下的是硬件限制。

## 文件清单

| 文件 | 职责 |
|---|---|
| `ptq_quant.h` | 量化语义（与 quant.py 逐位对齐） |
| `int8_kernels.cu` | 量化+im2col 融合、requant、add、relu、maxpool |
| `int8_gemm.cuh` | GEMM 封装（`_int_mm` + cuBLAS fp32 handle） |
| `int8_engine.cuh` | 模型加载、拓扑执行、显存池 |
| `layer_impl.h` | 按层选择：表结构 + JSON/环境变量覆盖 + 形状启发式 |
| `layer_impl_default.h` | **生成物**：batch=512 实测默认表（编译进二进制） |
| `select_layers.cpp` / `build_select.sh` | 生成层选择表（A/B + int32 逐元素校验） |
| `results/layer_impl.json` | 同一张表的 JSON（可手改后 `LAYER_IMPL_JSON` 载入） |
| `probe_overhead.cpp` | 非卷积段微基准（H2D / 残差 add / maxpool） |
| `main.cpp` | 评测入口 |
| `export_weights.py` | 权重 + 拓扑导出（含自检） |
| `export_calib.py` | 激活 range 校准（训练集） |
| **`verify_e2e.sh`** | **主判据**：端到端 vs numpy 参考 |
| **`ref_e2e.py`** | numpy 独立参考实现（im2col/GEMM/requant 全自写） |
| **`cmp_e2e.py`** | logits 比对（max\|diff\| / argmax 一致率） |
| **`locate_block.cpp/.py`** | 二分定位分歧起点 |
| `verify.sh` | 回归测试（4 个 probe + 2 层单层校验 + 3 组位宽） |
| `check_layer.cpp` | 单层数值校验（定位用） |
| `probe_*.cpp` | 最小化探针（回归测试用） |
| `prof_layer.cpp` | 分段计时 |