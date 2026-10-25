# MANIFEST —— experiments/20261025 逐文件清单

> 说明列由脚本从每个文件的**首行文档/注释**机械抽取，不是手写摘要；
> 少数脚本开头是实现行（如 `set -uo pipefail`），说明列照实显示。
> 家族含义、为什么归档、结论见 `README.md`。

## ab/ —— 穿插 A/B 计时驱动（8 轮交替、同进程换臂、报中位与配对 CI）

| 文件 | 首行说明 |
|---|---|
| `ab/ab_binop_512.py` | A/B: 9561414 baseline vs fused maxpool padding, 512 images. |
| `ab/ab_border.py` | H-079: 8-round A/B, border-aware conv (unpadded pack) off vs on. |
| `ab/ab_conv2px.py` | H-065: 8-round A/B, conv2px off vs on (same binary). |
| `ab/ab_default_flip.py` | 8-round interleaved A/B: a82d8d7 (fusion off) vs default-on fusion binary. |
| `ab/ab_flat.py` | H-086: 8-round A/B, flat warp mapping off vs on. |
| `ab/ab_graph_default_on.py` | H-055: 8-round A/B, graph default-on vs explicit NCNN_CUDA_GRAPH=0. |
| `ab/ab_graph.py` | 8-round interleaved A/B: normal forward vs captured CUDA Graph replay. |
| `ab/ab_h024.py` | Steady-state alternating A/B timing of old vs optimized implicit path. |
| `ab/ab_h025.py` | Steady-state alternating A/B: e053c57 vs fused pad+pack (T-20261024-05). |
| `ab/ab_h026.py` | Steady-state alternating A/B: d45b49e vs fused pad+pack (T-20261024-06). |
| `ab/ab_h027.py` | Steady-state alternating A/B: 6ea514a vs fused pad+pack (T-20261024-07). |
| `ab/ab_h028.py` | A/B: 644efc3 baseline vs Conv activation fusion, 512 images only. |
| `ab/ab_h061.py` | H-061: cumulative accounting 644efc3 -> HEAD plus per-optimization toggles. |
| `ab/ab_ilp.py` | H-086: 8-round A/B, ILP (2 accumulator chains) off vs on. |
| `ab/ab_ipgemv.py` | 8-round interleaved A/B: a82d8d7 (fusion off) vs default-on fusion binary. |
| `ab/ab_lazy_w8.py` | 8-round interleaved A/B: cf8fc38 binary vs lazy raw-weight upload (env on). |
| `ab/ab_pool_512.py` | A/B: 9561414 baseline vs fused maxpool padding, 512 images. |
| `ab/ab_quantpack.py` | H-074: 8-round A/B, fused quant+pack off vs on (same binary). |
| `ab/ab_smw.py` | H-090: 8-round interleaved A/B, shared-memory weight staging off vs on. |
| `ab/ab_switches.py` | Interleaved A/B for the three opt-in switches (process-level alternation). |
| `ab/ab_sync_512.py` | A/B: per-image sync vs deferred sync (--sync-every 8), 512 images. |
| `ab/rebase_ab.py` | import io |

## check/ —— 单次实验的构建 + 数值校验 + 崩溃隔离脚本

| 文件 | 首行说明 |
|---|---|
| `check/build_ab_h024.sh` | Build old (8918acf) and smem variants of the eval binary for A/B timing. |
| `check/check_ab_crash.sh` | H-055: reproduce A/B crash: run off and on once each at 512. |
| `check/check_chunks_h024.sh` | Per-chunk accuracy cross-check after the pooling optimization. |
| `check/check_conv_border.sh` | H-079: build border-aware conv and verify numerics. |
| `check/check_conv_border2.sh` | H-079: build + verify (patches already applied). |
| `check/check_conv_flat.sh` | H-086: build flat-warp kernel, verify numerics, prep A/B binary. |
| `check/check_conv_ilp.sh` | H-086: build ILP kernel, verify numerics, run A/B. |
| `check/check_conv_smw.sh` | H-090: build shared-memory weight staging kernel, resource report, verify numerics. |
| `check/check_conv2px_perf.sh` | H-065: confirm the 2px kernel is used, and compare per-layer conv time on/off. |
| `check/check_conv2px.sh` | H-065: build candidate binaries and verify 2px kernel logits/top1/top5. |
| `check/check_fb_small.sh` | H-039: fallback dump at smaller limits (128 crashes even with lazy off). |
| `check/check_graph_default_on.sh` | H-055: default-on verification: with shim (graph), explicit off, shim missing |
| `check/check_graph_off.sh` | H-048: default-off (env unset) must take the ordinary path unchanged. |
| `check/check_lazy_w8_2.sh` | H-039: finish 512 with lazy on; also prove default path of the new binary is |
| `check/check_lazy_w8.sh` | H-039: logits / top1 / top5 checks for lazy weight upload (opt-in). |
| `check/check_quantpack_default.sh` | H-074: flip default, rebuild, re-verify numerics (env unset = fused). |
| `check/check_quantpack.sh` | H-074: build fused quant+pack and verify numerics. |
| `check/check_switch_logits.sh` | set -uo pipefail |
| `check/compare_regs.sh` | set -u |
| `check/control_512_dump.sh` | H-039: is the 512 dump crash specific to lazy, or pre-existing/flaky? |
| `check/dump_check.sh` | Reproduce the dump-logits crash at different limits. |
| `check/dump_diag.sh` | set -uo pipefail |
| `check/dump_diag2.sh` | set -uo pipefail |
| `check/dump_single_batch.sh` | set -uo pipefail |
| `check/flaky_check.sh` | H-055: estimate flakiness of the 512 crash for old vs graph(with off) binaries. |
| `check/gdb_off_crash.sh` | H-055: gdb backtrace for graph-off 512/batch64 heap corruption. |
| `check/isolate_512_crash.sh` | H-055: isolate the 512/warmup30 crash: old binary vs graph binary off, warmup 20/30. |
| `check/isolate_dump_crash.sh` | H-039: narrow the fallback+dump crash at limit 128. |
| `check/isolate_lazy_fallback.sh` | H-039: isolate fallback crash: baseline vs lazy binary, lazy env on/off. |
| `check/locate_off_crash.sh` | H-055: where does graph-off 512/batch64 crash; does another batch work? |
| `check/prof_binop_ab.sh` | set -uo pipefail |
| `check/regress_h024.py` | H-20261024-004 single-command regression for ncnn CUDA implicit int8. |
| `check/regress_h024.sh` | Single command: 2000-image x 4-config regression vs H-20261023-021 baseline. |
| `check/scale_test_h026.sh` | Scale-cache invalidation test: same process, mutate layer-1 bottom scale. |
| `check/try_512_batches.sh` | H-039: 512 dump crash appears at ~108 images regardless of binary; try other batch sizes. |
| `check/try_512_single.sh` | H-039: 512 logits in a single formal batch (multi-batch dump copy is known-crashy). |

## harness/ —— 一次性评测器/插桩器源码（从 tracked 的 `ncnn_cuda_eval_2000c.cpp` 派生的变体）

| 文件 | 首行说明 |
|---|---|
| `harness/graph_shim.c` | /* H-048: redirect kernel launches onto a capture stream for CUDA Graph. |
| `harness/make_ab_harness.py` | Turn the eval harness copy into a same-process interleaved A/B benchmark. |
| `harness/make_profile_harness.py` | Replace the measurement tail of ncnn_cuda_profile_layers.cpp with a |
| `harness/memcpy_trace.c` | /* LD_PRELOAD interposer: aggregate cudaMemcpy* calls by kind, bytes and call site. |
| `harness/ncnn_cpu_eval.cpp` | include <cstdio> |
| `harness/ncnn_cuda_ab_harness.cpp` | ncnn + CUDA 评测器（ResNet-18 CIFAR-10）。 |
| `harness/ncnn_cuda_dump_logits.cpp` | ncnn + CUDA 评测器（ResNet-18 CIFAR-10）。 |
| `harness/ncnn_cuda_eval_2000.cpp` | ncnn + CUDA 评测器（ResNet-18 CIFAR-10）。 |
| `harness/ncnn_cuda_eval_attrib.cpp` | ncnn + CUDA 评测器（ResNet-18 CIFAR-10）。 |
| `harness/ncnn_cuda_eval_final.cpp` | ncnn + CUDA 评测器（H-20261023-021 验收版，ResNet-18 CIFAR-10）。 |
| `harness/ncnn_cuda_graph_eval.cpp` | ncnn + CUDA 评测器（ResNet-18 CIFAR-10）。 |
| `harness/ncnn_cuda_profile_layers.cpp` | ncnn + CUDA 评测器（ResNet-18 CIFAR-10）。 |
| `harness/ncnn_cuda_scale_test.cpp` | ncnn + CUDA 评测器（ResNet-18 CIFAR-10）。 |

## patch/ —— 给 ncnn fork 打的一次性源码补丁（每个对应一个开关或 kernels 变体）

| 文件 | 首行说明 |
|---|---|
| `patch/fix_border_act.py` | import io |
| `patch/fix_fuse_launchers.py` | Add the missing activation_params parameter to the two launcher definitions. |
| `patch/fix_graph_print.py` | import io |
| `patch/fix_scale_cache_loadmodel.py` | One-off: add the missing cache invalidation to the second load_model overload. |
| `patch/flip_quantpack_default.py` | H-074: flip fused quant+pack to default-on (NCNN_CUDA_FUSE_QUANT_PACK=0 disables). |
| `patch/make_act_params.py` | Create Clip and LeakyReLU param variants for activation-fusion tests. |
| `patch/make_clip_cuda.py` | Add a minimal CUDA Clip layer so unfused Clip references can run. |
| `patch/patch_attrib.py` | Add sync / host-forward timers to an attribution copy of the eval harness. |
| `patch/patch_avgpool.py` | Replace global average pooling with a lock-free one-thread-per-channel kernel. |
| `patch/patch_binaryop_debug.py` | Add an env-guarded debug line for the BinaryOp vector path. |
| `patch/patch_binaryop_default_off.py` | Make BinaryOp float4 opt-in: NCNN_CUDA_BINARYOP_VEC=1 enables it. |
| `patch/patch_binaryop_vec4.py` | Vectorize the same-shape contiguous BinaryOp path with float4. |
| `patch/patch_conv_border.py` | H-079: drop padded pack (border write amplification) by making the conv |
| `patch/patch_conv_flat.py` | H-086: flat warp mapping for the implicit int8 conv. |
| `patch/patch_conv_ilp.py` | H-086: ILP variant of the implicit conv (2 independent dp4a chains). |
| `patch/patch_conv_smw.py` | H-090: per-block shared-memory weight staging for the implicit int8 conv. |
| `patch/patch_conv2px.py` | H-065: add a 2-pixels-per-warp implicit int8 conv kernel for 3x3 stride1. |
| `patch/patch_fuse_act.py` | Fuse ReLU(slope=0)/Clip into the producing int8 Convolution epilogue. |
| `patch/patch_fuse_default_off.py` | Make the activation fusion opt-in: NCNN_CUDA_FUSE_ACT=1 enables it. |
| `patch/patch_fuse_default_on.py` | Flip ReLU/Clip epilogue fusion to default-on; NCNN_CUDA_FUSE_ACT=0 disables. |
| `patch/patch_fuse_quant_pack.py` | H-074: fuse float->int8 quantize into the pack kernel (opt-in). |
| `patch/patch_fused_pack.py` | Fuse constant-zero padding into the implicit int8 pack kernel. |
| `patch/patch_graph_default_on.py` | H-055: make CUDA Graph default-on in the eval harness, explicit off with |
| `patch/patch_ip_debug.py` | Temporary debug print of the int8 InnerProduct shapes. |
| `patch/patch_ip_gemv.py` | Fused int8 InnerProduct GEMV: one warp per output, dp4a, fused dequant. |
| `patch/patch_kprof.py` | Temporary KPROF instrumentation: per-call GPU time of the two scale |
| `patch/patch_kprof2.py` | Temporary KPROF2: per-call GPU time of fused pack, plain pack and conv. |
| `patch/patch_kprof3.py` | Temporary KPROF3: host time of the scale H2D upload block + upload/skip counts. |
| `patch/patch_lazy_w8.py` | H-039: make raw int8 weight H2D lazy (opt-in NCNN_CUDA_LAZY_W8=1). |
| `patch/patch_nodeprof_cum.py` | H-071: switch node profiling to cumulative-prefix timing. |
| `patch/patch_pool_fuse.py` | Fuse valid-padding into the max-pooling kernel. |
| `patch/patch_px4.py` | Replace the implicit int8 CUDA kernel with a 4-pixel-per-lane version. |
| `patch/patch_scale_cache.py` | Cache the last uploaded int8 scales per convolution layer. |
| `patch/patch_smem.py` | Stage implicit-conv weights per tap in shared memory. |
| `patch/patch_splitk_debug.py` | Add an env-guarded split decision log (NCNN_CUDA_SPLIT_DEBUG). |
| `patch/patch_splitk_default_off.py` | Default split-K off (rollback after no reliable gain); keep it opt-in. |
| `patch/patch_splitk_env.py` | Add split-K tuning knobs: NCNN_CUDA_SPLIT_MAX and NCNN_CUDA_SPLIT_MIN_K. |
| `patch/patch_splitk.py` | Split-K for the implicit int8 convolution (small feature maps). |
| `patch/patch_sync_every.py` | Add an opt-in --sync-every N to the eval/dump harnesses. |
| `patch/patch_unroll.py` | Unroll the implicit conv C4 loop 4-wide to expose memory-level parallelism. |

## probe/ —— 探针与日志分析脚本（nm 符号化、逐节点 profile 汇总）

| 文件 | 首行说明 |
|---|---|
| `probe/add_conv2px_debug.py` | H-065: temporary env-gated debug print to prove the 2px dispatch fires. |
| `probe/add_graph_dump.py` | H-048: optional logits dump from the graph path (env NCNN_CUDA_GRAPH_DUMP). |
| `probe/add_graph_loop.py` | H-048: replace the 50-iter replay probe with the formal graph measurement loop. |
| `probe/add_graph_probe.py` | Insert the H-048 CUDA Graph capture probe into the harness copy. |
| `probe/add_nodeprof.py` | H-071: per-node timing of the captured graph (graph-form kernel profile). |
| `probe/analyze_smw.py` | H-090: combine the two independent 8-round batches and size the per-layer shared tile. |
| `probe/find_residual_quant.py` | import subprocess |
| `probe/map_memcpy_sites.py` | Map interposer caller offsets (module-relative) to nearest symbol via nm. |
| `probe/map_nodeprof_cum.py` | H-071: aggregate cumulative-prefix graph profile by kernel symbol. |
| `probe/map_nodeprof.py` | H-071: map graph node offsets to symbols and rank kernel-only time. |
| `probe/summarize_memcpy_trace.py` | Summarize interposer trace: group by (kind, bytes) and by symbol. |
| `probe/winograd_int8_error.py` | H-084: int8 Winograd F(2x2,3x3) error simulation on the real layer shapes. |

## run/ —— 一次跑完某个实验的驱动脚本（构建 → 跑 → 汇总）

| 文件 | 首行说明 |
|---|---|
| `run/run_ab_three.sh` | set -uo pipefail |
| `run/run_binop_tests.sh` | set -uo pipefail |
| `run/run_border_profile.sh` | H-079: graph node profile with border-aware conv (env on). |
| `run/run_conv2px_ab.sh` | H-065: rebuild eval, prove dispatch, then run 8-round A/B. |
| `run/run_default_flip_tests.sh` | set -uo pipefail |
| `run/run_fuse_perf.sh` | set -uo pipefail |
| `run/run_fuse_tests.sh` | Minimal activation-fusion tests: baseline, ReLU, Clip, LeakyReLU, no-act. |
| `run/run_graph_eval.sh` | H-048: build graph eval, run 128 (with logits dump) and 512. |
| `run/run_graph_nsys.sh` | H-048: nsys API counts, normal vs graph. |
| `run/run_graph_probe.sh` | H-048: build the CUDA Graph probe and run it once on 8 images. |
| `run/run_ipgemv_tests.sh` | set -uo pipefail |
| `run/run_memcpy_trace_lazy.sh` | H-039: trace memcpy with lazy weight upload enabled. |
| `run/run_memcpy_trace.sh` | H-039: run 128-image eval under LD_PRELOAD memcpy interposer. |
| `run/run_nodeprof_cum.sh` | H-071: cumulative-prefix node profile (marginal kernel costs in graph form). |
| `run/run_nodeprof.sh` | H-071: graph-form per-node kernel profile. |
| `run/run_nsys_default_on.sh` | H-055: nsys API counts for graph default-on (env NCNN_CUDA_GRAPH unset). |
| `run/run_pool_tests.sh` | set -uo pipefail |
| `run/run_quantpack_default_profile.sh` | H-074: final graph-form node profile with the fused kernel as default. |
| `run/run_quantpack_nodeprof.sh` | H-074: graph-form node count + kernel mix, fused vs not. |
| `run/run_sync_tests.sh` | set -uo pipefail |
| `run/run_trace_4b_isolation.sh` | H-039: isolate when 4-byte scalar H2D happens (warmup vs formal batch). |

---

合计 **144** 个文件（6 个家族）。
