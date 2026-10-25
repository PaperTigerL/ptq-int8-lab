# experiments/20261025 —— ncnn CUDA（h024–h093）那轮扫描的脚本归档

这里的 **144 个文件**是 2026-10-23 ~ 10-25 在 `05-ncnn-deploy/cuda/` 上做的一轮实验留下的：
问的是「ncnn CUDA fork 能不能靠 kernel 改写 / 开关把 int8 压下去」，逐个开关做穿插 A/B。
这轮不是一个要发布的功能节点，所以补丁和驱动脚本没有留在主目录，按家族收在这里 ——
**一个都没删**。

路线的总结论没有变，仍在 [`../../README.md`](../../README.md)：**这条路走不到 500×**，
唯一精度正确的配置是 0.8×，已经停止投入。本目录保存的只是「当时是怎么问的」；
哪些问题被采纳、哪些被证伪，进过正式记录的部分在 `../../README.md` 和对应实验的 `results-h0XX` 里。

---

## 一、清点口径（2026-10-25 收尾）

| 东西 | 数量 | 去哪儿 | 为什么 |
|---|---|---|---|
| 一次性 A/B 驱动、构建/校验脚本、补丁、探针、变体评测器源码 | **144** | **本目录**（`ab/ check/ harness/ patch/ probe/ run/` 六个家族） | 一次实验用完就扔的驱动；留着为可追溯，不参与构建 |
| 编译出来的评测器二进制 | **58**（这次新纳入 ignore 55 个，另 3 个原本就在 `.gitignore` 里） | **留在 `cuda/` 根**，已进 `cuda/.gitignore` | `build.sh` 的 `EVAL_BIN` 默认就落在这里，`run_*/verify_*` 按名字直接调；挪走主目录的验收路径就断了 |
| 原始日志 / 逐层转储 / verify 快照 / fork 源码快照 | **605 个文件（72 MB）** | `cuda/work/`（已 gitignore，**没删**） | 每次重跑都会新建时间戳目录，进库只是噪声 |
| 仍有复用价值的 3 个 | **3** | **留在 `cuda/` 根** | `run_h021.sh`（h021 验收驱动，产物 `results-h021/` 已入库）、`verify_final_smw.sh`（bit-exact 复验）、`analyze_logits.py`（被多个脚本调用的 logits 比对器） |

`cuda/work/` 的结构（都在磁盘上，只是不进库）：

```
work/
├── sweeps/       # results-h019 / h024..h093 —— 496 个文件，每轮实验的日志与转储
├── archive/      # h090/ h093/ —— 30 个文件，ncnn fork 的源码快照与 h015 补丁
├── results/      # 73 个文件，原先散在 cuda/results/ 下的原始 stdout、verify 快照及其构建/运行日志
└── results-h021/ # 6 个文件，h021 验收时前后拍的 nvidia-smi 快照
```

进库的 `cuda/results/` 只保留它原本跟踪的 4 份（`cuda_*_noreset/reuse.txt`、`vulkan_probe.md`）。
`results-h021/` 的 8 份验收证据照旧跟踪，只有 nvidia-smi 快照进了 `work/`。

## 二、怎么读

一轮实验在文件上通常是四件套，用编号（h0XX）串起来：

| 家族 | 在这一轮里的角色 |
|---|---|
| `patch/*.py` | **改什么**：给 fork 的 `src/layer/cuda/*.cu` / `*.cpp` 打补丁，加一个开关或换一种 kernel |
| `check/*.sh` | **对不对**：构建变体 → 与 baseline 逐位比 logits / 精度 → 有时兼做崩溃隔离 |
| `ab/*.py` | **快多少**：同进程 8 轮交替换臂（base/变体/…/变体），报中位与配对 CI，而不是两次单跑相减 |
| `run/*.sh` | **一条龙的驱动**：把上面三步串起来，输出落到 `results-h0XX` |
| `probe/*.py` | 侧路：nm 符号化、逐节点 profile 汇总、误差仿真 |
| `harness/*` | 上面这些脚本要编译的一次性评测器/插桩器源码（从 tracked 的 `ncnn_cuda_eval_2000c.cpp` 派生） |

实验编号和 `work/sweeps/results-h0XX` 一一对应：脚本开头若写着 `H-0XX`，它那轮的原始输出就在
`work/sweeps/results-h0XX/`（例如 `check/check_conv_smw.sh` → `results-h090/`）。
逐文件的首行说明见 [`MANIFEST.md`](MANIFEST.md)。

## 三、这轮问过的问题（按编号）

| 编号 | 问的是什么 | 主要脚本 |
|---|---|---|
| h024 | implicit 路径 vs 优化版（smem 权重、pool padding 融合）的稳态交替 A/B | `ab/ab_h024.py` `check/build_ab_h024.sh` `check/regress_h024.sh` |
| h025–h027 | 逐版基线与「fused pad+pack」的交替 A/B（每版换一个基线 commit） | `ab/ab_h025..h027.py` |
| h028 | Conv 激活融合前后的 A/B | `ab/ab_h028.py` `patch/patch_fuse_act.py` |
| h029 | maxpool valid-padding 融合的数值校验 | `run/run_pool_tests.sh` `patch/patch_pool_fuse.py` |
| h032 | 三个 opt-in 开关（pool/binop/relu）进程级交替 A/B | `ab/ab_switches.py` `run/run_ab_three.sh` |
| h035 / h039 | lazy 权重上传（opt-in）：数值校验 + 512 转储崩溃的隔离 | `patch/patch_lazy_w8.py` `check/check_lazy_w8*.sh` `run/run_trace_4b_isolation.sh` |
| h048 / h055 | CUDA Graph（capture/replay）：API 计数、默认开、512/batch64 崩溃定位 | `probe/add_graph_*.py` `harness/graph_shim.c` `check/gdb_off_crash.sh` |
| h061 | 从 644efc3 到 HEAD 的累计账与逐优化开关的分解 | `ab/ab_h061.py` |
| h065 | 2 像素/warp 的隐式卷积：dispatch 取证 + A/B | `patch/patch_conv2px.py` `check/check_conv2px*.sh` |
| h071 | 图形态逐节点 profile（cumulative 前缀 → 边际 kernel 成本） | `probe/map_nodeprof*.py` `run/run_nodeprof*.sh` |
| h074 | quantize 融进 pack kernel：数值校验 + 默认翻开的复验 + 逐节点 profile | `patch/patch_fuse_quant_pack.py` `check/check_quantpack*.sh` |
| h079 | 去 padding 打包（border-aware conv）：构建 + 数值 + 逐节点 profile | `patch/patch_conv_border.py` `run/run_border_profile.sh` |
| h084 | int8 Winograd F(2×2,3×3) 在真实层形状上的误差仿真 | `probe/winograd_int8_error.py` |
| h086 | flat warp 映射 / ILP（2 条 dp4a 链）两种 kernel | `patch/patch_conv_flat.py` `patch/patch_conv_ilp.py` `ab/ab_flat.py` `ab/ab_ilp.py` |
| h090 | 分块共享内存权重暂存（smem staging） | `patch/patch_conv_smw.py` `check/check_conv_smw.sh` `ab/ab_smw.py` |
| h093 | h015 的补丁快照（CudaMat 复用 + batch 入口），源码副本在 `work/archive/h093/` | `../h093/`（已移入 work） |

## 四、复现须知

- 脚本里写死了 `/mnt/d/WorkBuddy/ShowBro/lab/...` 与 `/home/user/ncnn-cuda`（WSL + CUDA 12.4）。
  Windows 侧只负责编辑，跑仍然要进 WSL。
- 二进制要自己重建（它们不进库）：

  ```bash
  cd /mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda
  NCNN_BUILD_DIR=/home/user/ncnn-cuda/build-ptq \
    EVAL_SRC=$PWD/ncnn_cuda_eval_2000c.cpp EVAL_BIN=$PWD/ncnn_cuda_eval_2000c \
    bash build.sh
  ```

  变体评测器用 `experiments/20261025/harness/*.cpp` 当 `EVAL_SRC`。
- **这些是历史记录，不保证在当前 HEAD 上原样跑通**：fork 源码、默认开关、二进制名一直在变，
  `patch/*.py` 里的锚点字符串（`ARG_OLD` / `BLOCK_OLD`）对应的是当时的源码行。
  要重跑先看 `check/*.sh` 里的构建命令，再决定要不要重放补丁。
- 这轮的全部实验都在同一台 GTX 1650 Ti 上，且**没有一条能突破 `../../README.md` 里那四条结构性问题**。
