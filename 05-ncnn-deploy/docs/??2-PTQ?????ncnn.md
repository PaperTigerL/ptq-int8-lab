# 阶段 2 · PTQ 配置迁移进 ncnn —— 桥通了

> 2026-10-18 实测。这是 05-ncnn-deploy 从「C++ 侧基线」走向
> 「PTQ + C++ 合体」的一步，也是整条线一开始就要解决的那个问题：
> **ptq-lab 在 PyTorch 里算出的量化配置，能不能原样跑进 C++ 引擎？**

## 0. 先说结论

**能，而且不用改 ncnn 一行代码。**

| 激活定标策略 | Python 侧（ptq-lab） | C++ 侧（ncnn 桥） | 差 |
|---|---|---|---|
| minmax | 86.99% | **87.17%** | **+0.18 pp** |
| percentile | 87.00% | **83.88%** | -3.12 pp |
| entropy | 86.02% | **68.88%** | -17.14 pp |
| （参照）ncnn 自己的 KL 校准 | — | 87.05% | — |
| （参照）FP32 基线 | 87.16% | 87.15% | -0.01 pp |

> ⚠️ 上表是**言瑞自己实现**的口径（`PTQLAB_REFERENCE=0`）。
> 最初一版用 reference 实现跑，其中 `scale_entropy` 缺截断约束、
> 把 conv1 的激活截掉 84.89%，entropy 两边都只有 38%。
> 详见 §3 —— **这是本次最重要的发现，直接影响 e7 的结果 6。**

三个可复用的结论：

1. **桥是忠实的**：minmax 口径下 Python 86.99% → C++ 87.17%，
   差 0.18pp，小于 CIFAR-10 测试集 0.33pp 的标准误。
   「ptq-lab 里测的 W8A8 精度」现在可以名正言顺地当作
   「同一份配置部署到 ncnn 后的精度」用。
2. **entropy 在这个模型上并不占优**：修正后 86.02%，比 minmax（86.99%）
   差约 1pp。以前以为它"灾难性崩盘"，其实是实现 bug；
   修完之后它是"平平无奇"。
3. **asym → sym 的转换曾经不是免费的，代价随校准策略变化**：
   minmax 几乎无损，percentile 掉 3.12pp，entropy 掉 17.14pp。
   **机制已定位并修好（§3.6）**：不是引擎的问题，是桥少做了一步码空间换算
   —— 非对称 255 码的 scale 被原样灌进了对称 ±127 码的引擎，
   注入值大了 255/127 倍。修正后三个策略的落差全部消失
   （+0.23 / +0.02 / 0.00 pp）。

## 1. 机制：ncnn2int8 是纯执行器

这条链是从源码逐层确认的，不是猜的：

| 环节 | 位置 | 事实 |
|---|---|---|
| 解析表 | `tools/quantize/ncnn2int8.cpp:37` `read_int8scale_table` | key 带 `_param_` 后缀 → 权重 scale 表；否则 → 激活 scale 表 |
| 取用 | `ncnn2int8.cpp:246` `quantize_convolution` | `weight_data_int8_scales = iter->second`，**直接取自表** |
| 执行 | `src/mat.cpp:2420` `quantize_to_int8` | 把 scale 当 Quantize 层的权重加载（`ModelBinFromMatArray`）后跑前向 |
| 写回 | 同上 | `weight_data_int8_scales` / `bottom_blob_int8_scales` 原样写进 bin |

**KL 校准只发生在 `ncnn2table` 生成表那一步。表一旦生成，
`ncnn2int8` 一个 scale 都不重算。**

### 1.1 决定性实验：改表里的值，精度真的会变

推断得再漂亮也要实测。把 conv1 的权重 scale 乘 2 灌进表里：

| 模型 | conv1 权重 scale | 精度 |
|---|---|---|
| 基线（KL 表原值） | 506.113 | 87.05% |
| 探针（scale × 2） | 1012.227 | **80.53%** |

掉了 6.52pp。如果 ncnn2int8 是把表当「这层要量化」的开关、
然后自己用 KL 重算，改表里的数值应该毫无影响。
它实打实用了我们的数。

## 2. 最大的坑：两边的 scale 约定互为倒数

这是本次唯一一个让产出直接变废品的错误，值得单独一节。

| 侧 | 公式 | 语义 |
|---|---|---|
| ptq-lab | `scale = absmax / 127`（`src/quant.py:211`） | **除数**：`w_q = round(w / scale)` |
| ncnn | `scale = 127 / absmax`（`ncnn2table.cpp:580`） | **乘数**：ncnn 侧反着用 |

第一次直接把 ptq 的 scale 写进 ncnn 表，**精度 87.05% → 10.00%**
（十个类随机猜）。现象极具误导性：不报错、模型能跑、速度快、
bin 大小正常——只有精度是垃圾。

修复就一行：写入前取倒数。修复后 conv1 权重 scale
注入值 506.113329，KL 值 506.1133（权重上 minmax 和 KL 都用 absmax，
所以几乎一致，这本身就是个交叉验证）。

## 3. 意外发现：reference 实现的 entropy 校准是坏的（2026-10-18 下午修正）

桥跑出 entropy 只有 37.63%，第一反应是桥错了。对照之后发现桥是忠实的，
问题在 ptq-lab 侧——但**根因不是"entropy 这个方法不行"，而是"跑的一直是
坏的那份实现"**。

### 3.1 怎么发现的

`inspect.getsource(Q.scale_entropy)` 显示的代码和 `src/quant.py` 里
**不是同一份**。同一个文件 MD5 一致、`grep` 和 `open().read()` 都说没有
`_sym_qmax`，唯独 `getsource` 说有。打印 `__code__.co_filename` 才真相大白：

```
co_filename = .../03-ptq-lab/reference/quant_ref.py
```

`src/quant.py` 末尾有个开关：

```python
if C.USE_REFERENCE:
    _ref = _load_reference()
    for _n in _TODO_NAMES:            # 5 个函数，含 scale_entropy
        globals()[_n] = getattr(_ref, _n)
```

而 `config.py:175`：

```python
USE_REFERENCE = os.environ.get("PTQLAB_REFERENCE", "1") == "1"   # 默认开
```

**所以至今所有实验跑的都是 `reference/quant_ref.py` 的实现，
`src/quant.py` 里言瑞自己写的那份从未被实验用过。**

### 3.2 两份实现的差别

| | reference（一直在跑） | 言瑞的实现（`src/quant.py`） |
|---|---|---|
| 截断约束 | **没有** | `ENTROPY_MAX_CLIP = 0.01`，超限跳过 |
| 候选下界 | 硬编码 `amax * 0.1` | `min(amax*CALIB_SHRINK, 分位数)` |
| 精确零处理 | `nz = xa[xa>0]` 排除 | 不排除 |
| 全超限时 | `best_c` 默认 `amax` | 降级到 `amax/qmax` |

reference 版没有截断约束，纯 KL 最小化就会把阈值压到候选区下界
（`amax*0.1`），**conv1 上实测截掉 84.89% 的激活**。
言瑞版有约束，实测只截 0.70%，合规。

### 3.3 修正后的数字

`PTQLAB_REFERENCE=0`（用言瑞的实现，测试 15/15 全过）：

| 策略 | reference 口径 | 言瑞的实现口径 | 变化 |
|---|---|---|---|
| minmax | 87.11% | 86.99% | -0.12pp |
| percentile | 86.96% | 87.00% | +0.04pp |
| **entropy** | **38.19%** | **86.02%** | **+47.83pp** |

**entropy 从"灾难性崩盘"回到正常区间**（但仍比 minmax 差约 1pp，
说明 entropy 在这个模型上并不占优，只是以前被实现 bug 埋了）。

### 3.4 ⚠️ 连带影响：e7 的结果 6 要用新口径重审

e7 的「混合 + AdaRound + A4 entropy」= 85.45%，`avg_bits = 4.306`，
是在 reference 口径下跑出来的。**这个数字的激活定标部分是坏的**
（虽然 85.45% 不是 38%，因为混合精度 + AdaRound 另有权衡，
但定标口径确实被动过）。要拿它当论文主表数据，必须用
`PTQLAB_REFERENCE=0` 重跑一遍 e7。

**已记为待办，本次未处理。**

### 3.5 桥修正前的全表（言瑞实现口径，旧行为）

| 激活定标 | Python | C++（修正前） | 差 |
|---|---|---|---|
| minmax | 86.99% | **87.17%** | +0.18pp |
| percentile | 87.00% | **83.88%** | -3.12pp |
| entropy | 86.02% | **68.88%** | -17.14pp |
| （参照）ncnn 自带 KL | — | 87.05% | — |

这张表是**历史记录**。当时的困惑是「同一座桥，为什么只有 minmax 忠实」。

走过的弯路（都记下来，省得以后重走）：

- 假设「丢 zero_point 的代价」——**已被证伪**：两种 dump 的 zp 分布
  完全相同（19/20 层 zp=0，conv1 zp=123），代价却差 15 倍。
- 假设「`src/__pycache__` 字节码陈旧」——**已被证伪**：删掉 .pyc 重编，
  行为一字不差。真凶是 `USE_REFERENCE` 开关（§3）。
- 观察「entropy 的激活 scale 比 minmax 细，Python 扛得住 C++ 扛不住」
  ——方向对了，但没说到点子上。真正的机制在 §3.6。

> 顺带排除一个错误猜测：我一度以为 `src/__pycache__/quant*.pyc` 是陈旧
> 字节码（它的 mtime 比 .py 新、且删掉后 `inspect` 输出没变）。
> 实际上 .pyc 是好的，删掉再重编行为完全一致——问题一直在
> `USE_REFERENCE` 这个开关上，不在缓存上。

### 3.6 机制已定位，并修掉了桥的一个真 bug：码空间换算没做

**一句话**：ptq-lab 的激活量化是**非对称 255 码**（`[0,255]` + zero_point），
ncnn 的 int8 是**对称 ±127 码、且没有 zero_point**。
`make_table.py` 之前把 `1/a_scale = 255/量程` 原样灌进 ncnn 的表，
而 ncnn 的表示区间被钉死在以 0 为中心的 `[-127/s, +127/s]`——
**等于每层只能表示到校准量程的一半，桥本身就自带 2 倍溢出。**

minmax 扛得住（超出部分在分布尾巴上，损失小于噪声），
percentile / entropy 又把量程各自截掉 2~3 倍，溢出叠加上去，
尾巴被截掉的比例从"可忽略"变成"显著"，精度就塌了。

#### 机制链条（每一环都有源码）

| 环节 | 位置 | 事实 |
|---|---|---|
| ptq 激活量程 | `src/calib.py:87,104` | `qmax = 2^a_bits - 1 = 255`，`scale = (hi-lo)/255` |
| 注入 ncnn | `scripts/make_table.py:215`（修正前） | `a_scale = 1.0 / entry["a_scale"]` = `255/(hi-lo)` |
| ncnn 的截断 | `src/layer/quantize.cpp:30-36` | `float2int8(v)`：`>127 → 127`，`<-127 → -127` |
| ncnn 无 zero_point | Convolution 的 int8 激活只有一个 scale | 表示区间只能是 `[-127/s, +127/s]`，**中心被钉在 0** |

所以每层实际能表示的激活上界是 `127/s = 127×(hi-lo)/255 ≈ (hi-lo)/2`。
要让策略自己选的阈值 `hi` 恰好落在 ±127 上，注入值必须是
`127/(hi-lo)`，也就是 `255/(hi-lo)` 的 **127/255**。

> 注意这不是「ncnn 算错了」。表是我们给的，ncnn 照单执行（§1 已证实）。
> 给错了就是我们的锅。

#### 证据 A：各策略的溢出倍数（纯读 json，秒出）

`scripts/overflow_probe.py` 的 A 部分，倍数 = `a_scale_minmax / a_scale_策略`
= 真实量程 / 策略截后的量程：

| 策略 | 中位数 | 最坏 | 溢出层数 | 最坏层 |
|---|---|---|---|---|
| minmax | 1.00 | 1.00 | —（基准） | — |
| percentile | **1.99** | **2.58** | 20/20 | `layer2.1.conv2` |
| entropy | **3.06** | **4.81** | 19/20 | `layer2.1.conv2` |

叠上桥自带的 2×，三者在 ncnn 里的实际截点分别是 **量程/2、量程/4、量程/6**。

#### 证据 B：溢出容忍度（minmax 表，注入 scale × f，真跑 C++）

下表是**修正后**的口径（f=1 即修正后的桥）：

| f | ncnn 的实际截点 | 精度 |
|---|---|---|
| 0.25 | 4× 量程 | 86.90% |
| 0.498 | 2× 量程 | 87.03% |
| **1.0** | **量程（= 策略自己选的阈值，修正后基线）** | **87.22%** |
| 2.0 | 量程/2（**= 修正前的行为**） | 87.12% |
| 4.0 | 量程/4 | **80.37%** |
| 8.0 | 量程/8 | **30.97%** |

**缩小是免费的，放大是致命的**——这是饱和截断（clipping）的签名，
不是舍入误差的签名。`f ≤ 2` 全在 86.90~87.22% 之间（跨度 0.32pp，
和 0.33pp 标准误同量级，即噪声），`f = 4` 直接塌到 80.37%。
拐点极陡：从"免费"到"崩掉"只隔一格。

**顺带拿到一个交叉验证**：修正前用同一个脚本扫出来的是
`f=1 → 87.17%`、`f=2 → 80.25%`、`f=4 → 30.61%`、`f=8 → 11.99%`。
两条曲线**正好错开一格（2×）**——新的 f=4 对应旧的 f=2，新的 f=8 对应旧的 f=4。
这是对「修正量恰好是 255/127 倍」最直接的确认，不是靠推导，是靠曲线平移量。

#### 证据 C：因果回验（只改码空间换算这一处）

**修正前**跑的那组，是决定性的：

| 策略 | C++ 原样 | 注入 scale × 127/255 | 变化 | Python 侧 |
|---|---|---|---|---|
| percentile | 83.88% | **87.03%** | +3.15pp | 87.00% |
| entropy | 68.88% | **86.02%** | +17.14pp | 86.02% |

另一组对照（各策略按自己的最坏溢出倍数缩小，彻底消除溢出）：
percentile 83.88% → 87.10%，entropy 68.88% → 87.10%。
**两条路都回到 87% 附近，机制没有第二种解释。**

**修正后再缩 2×**（检查是不是已经在平台上了）：

| 策略 | 修正后 | 再 × 127/255 | 变化 |
|---|---|---|---|
| percentile | 87.02% | 87.18% | +0.16pp |
| entropy | 86.02% | 87.15% | +1.13pp |

读法：修正后已经站在平台边缘（再缩只有 0.16pp 的余量，属噪声）。
entropy 剩下的 1.13pp **不是桥的错，是它自己选的**——entropy 把量程
截得比 minmax 紧 3 倍，被截掉的那部分本来就要付钱。
Python 侧同样付这笔钱（entropy 86.02% vs minmax 86.99%，差 0.97pp），
**两边付得一样多，这才是"忠"的意思。**

> 复现提示：`overflow_probe.py` 内部调 `make_table.py`，所以基线跟着桥走。
> 想复现「修正前」那两行，给它传 `--no-sym-fix`。

#### 修复

`make_table.py` 新增常量 `SYM_CODE_FIX = 127.0 / 255.0`，注入时乘上去：

```python
a_scale = [SYM_CODE_FIX / entry["a_scale"]]
```

`--no-sym-fix` 可以关掉它、复现修正前的旧行为（用于对照，正常别用）。

**修复后的桥（端到端重跑，`scripts/rebuild_ptq_tables.sh`）：**

| 激活定标 | Python | C++（修正后） | 差 |
|---|---|---|---|
| minmax | 86.99% | **87.22%** | +0.23pp |
| percentile | 87.00% | **87.02%** | +0.02pp |
| entropy | 86.02% | **86.02%** | **0.00pp** |
| （参照）ncnn 自带 KL | — | 87.05% | — |

**三个策略全部忠实。** entropy 的 86.02% 和 Python 侧一字不差——
这是整座桥最强的单点证据：同一份配置、两套独立实现、同一个数字。

#### 两条诚实的边界

1. **修的是"量程对齐"，不是"分辨率对齐"**。非对称 256 码换成对称后
   （且 zp 丢弃、负半边浪费）实际只用了约 128 个码字，等于掉 1 bit。
   实测这 1 bit 在 CIFAR-10 + ResNet-18 上不值钱（87.22% vs 86.99%），
   **但不能外推**——换更深的模型或更难的 dataset，这 1 bit 可能就不免费了。
2. **percentile / entropy 在 Python 侧"更好"的优势没有在 C++ 侧复现**。
   修正后三者排序和 Python 一致（minmax ≈ percentile > entropy），
   说明落差确实是桥的问题而非策略本身。但 0.02 / 0.00pp 这种级别的吻合
   有运气成分，别当成"桥零误差"的证据。

## 4. 工具与口径

### 4.1 两个工具，分置两侧

```
03-ptq-lab/tools/dump_scales.py        ← 研究侧：算 scale，落 json
05-ncnn-deploy/scripts/make_table.py   ← 部署侧：json → ncnn table
```

分置是刻意的：**scale 的计算公式只能有一个真相源**。
`make_table.py` 只做格式转换和量纲换算，不重复任何量化公式。

量纲换算有两处，都写在常量/注释里，不是散落的魔数：

| 换算 | 系数 | 由来 |
|---|---|---|
| 除数 → 乘数 | `1 / scale` | ncnn 存 `127/absmax`，ptq 存 `absmax/127`（§2） |
| **asym 255 码 → sym ±127 码** | `SYM_CODE_FIX = 127/255` | ncnn `float2int8` 截在 ±127 且无 zp（**§3.6**） |

第二处是 2026-10-18 才补上的，之前漏了它，桥对所有非 minmax 策略都不忠实。

`dump_scales.py` 的关键设计是**自检**：它重算一遍权重 scale，
再和 `Q.quantize_per_channel_sym` 的输出逐元素比对，
最大差必须是 0（实测 0.00e+00）。这个断言挂在写文件之前——
公式抄错时产出的 json 全是废的，不能让它流到下一步。

### 4.2 层名映射：PyTorch → ncnn

| PyTorch | ncnn |
|---|---|
| `conv1` | `/conv1/Conv` |
| `layer1.0.conv1` | `/layer1/layer1.0/conv1/Conv` |
| `layer2.0.downsample.0` | `/layer2/layer2.0/downsample/downsample.0/Conv` |
| `fc` | `/fc/Gemm` |

两个容易错的地方，都踩过：

1. `layer1.0` 这个「stage.block」在 ncnn 里是**两级目录**
   （`/layer1/layer1.0/`），不能简单 `replace('.', '/')`。
2. 多段模块路径会被 onnx2ncnn 展开成「目录/全名」两级：
   `downsample.0` → `downsample/downsample.0`。

映射结果**全部对照 param 文件断言**，有一个不在就中止。
第一次跑就是被这个断言拦下来的（当时规则漏了第 2 条）。

### 4.3 没覆盖的层：明确回退，不静默混口径

ptq-lab 只量化 Conv2d，`fc/Gemm` 不在范围内。
`make_table.py` 对这一层回退到 KL 基线值，并打印：

```
[回退] /fc/Gemm    用 KL 基线值（ptq 未覆盖此层）
⚠️ 上面 1 层用的是 KL 值而非 ptq 值，报数字时不能算作「ptq 配置完整迁移」。
```

## 5. 复现

```bash
# 研究侧：算 scale 并落 json（WSL，需要 torch）
# ⚠️ PTQLAB_REFERENCE=0 必须加！默认是 1，会用 reference/quant_ref.py
#    顶替 src/quant.py 的实现，而 reference 的 scale_entropy 缺截断约束
#    （详见 §3）。不加这个变量，entropy 的数字是废的。
cd /mnt/d/WorkBuddy/ShowBro/lab/03-ptq-lab
PTQLAB_REFERENCE=0 ~/pyenv-torch/bin/python tools/dump_scales.py --strategy minmax

# 部署侧：json → ncnn table → int8 模型
cd /mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy
python3 scripts/make_table.py \
    --scales /mnt/d/WorkBuddy/ShowBro/lab/03-ptq-lab/results/cifar10_resnet18/scales_w8a8_minmax.json \
    --param models/resnet18_cifar10_opt.param \
    --baseline tables/resnet18_cifar10_kl.table \
    --out tables/resnet18_cifar10_ptq_minmax.table

~/ncnn-build/ncnn/build/tools/quantize/ncnn2int8 \
    models/resnet18_cifar10_opt.param models/resnet18_cifar10_opt.bin \
    models/ptq_minmax.param models/ptq_minmax.bin \
    tables/resnet18_cifar10_ptq_minmax.table

./cpp/build/cifar10_eval --param models/ptq_minmax.param --bin models/ptq_minmax.bin \
    --data data_raw/test_batch.raw --threads 1 --tag ptq-minmax

# Python 侧同口径对照（同样要加 PTQLAB_REFERENCE=0）
cd /mnt/d/WorkBuddy/ShowBro/lab/03-ptq-lab
PTQLAB_REFERENCE=0 ~/pyenv-torch/bin/python tools/w8a8_python_ref.py --strategy minmax
```

三策略一起重建 + 端到端评测（推荐，一条出全表）：

```bash
cd /mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy
./scripts/rebuild_ptq_tables.sh                  # minmax / percentile / entropy
NO_SYM_FIX=1 ./scripts/rebuild_ptq_tables.sh     # 复现修正前的旧行为（对照用）
```

溢出探针（§3.6 的三组证据一次跑完）：

```bash
python3 scripts/overflow_probe.py --factors 0.25,0.498,1,2,4,8 --causal-shrink 2.0079
python3 scripts/overflow_probe.py --skip-run    # 只算溢出倍数，不跑引擎，秒出
```

⚠️ `make_table.py` 只用标准库，但在 WSL 里要用 `python3` 跑——
用 Windows 的 python.exe 会解析不了 `/mnt/d/...` 路径（踩过）。

## 6. 还没做

- **4bit 仍然搬不了**。ncnn 无 int4 路径（源码已证实），
  「W4A8 混合精度迁移」这件事在 stock ncnn 上无解，需要另想载体。
- **fused requantize 的层名冲突未处理**。ncnn 会把 BasicBlock 里
  第二个 conv 的输出也量化（`8=102`），而 ptq-lab 是逐层独立的。
  本次迁移能成，是因为只换了 scale 数值、没换量化边界；
  真要搬逐层位宽分配时，这个问题会正面撞上。
- ~~**percentile / entropy 掉点的归因未定位**~~
  → **已定位并修复，见 §3.6**。原因是桥漏了 asym 255 码 → sym ±127 码
  的换算；修正后三策略落差全部消失。
- **asym 的 zero_point 仍然丢掉了**，这是引擎能力边界，不是桥的问题。
  ncnn 的 Convolution int8 激活只有一个 scale、没有 zp，所以非对称配置
  迁过来必然退化为对称（负半边码字浪费，约掉 1 bit）。
  实测这 1 bit 在 CIFAR-10 + ResNet-18 上免费（§3.6 边界 1），
  **但没有在别的模型上验证过，别外推**。
- **负半边浪费的 1 bit 能不能捡回来**：理论上可以在 `make_table.py` 里
  对 `lo < 0` 的层改用 `127/max(|lo|,|hi|)` 而不是 `127/(hi-lo)`，
  本模型 19/20 层 lo≈0 所以没差别。**未做，也没有动机做**
  （前提是本模型那 1 bit 确实免费）。
