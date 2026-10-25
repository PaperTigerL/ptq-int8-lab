# -*- coding: utf-8 -*-
"""实测 TRT 11.4 在这台机器上能不能拿逐层 profiler。

**实测结论（2026-10-20）：不能。**
即使在**同一进程里 build engine** 并设了
`config.profiling_verbosity = DETAILED`，
`ctx.profiler` 仍然是 `None`（TRT 11.4.0.106+ WSL, CUDA 12.4）。

所以 08-report 的「每层 TOPS」图不用 TRT profile，
改用 06-gpu-int8 用 `torch._int_mm` 实测的数据（results/gemm_tops.csv）。

这个脚本保留下来是为了**别让下一个人再花 6 分钟 build 一遍**。
想复现结论：跑本脚本，看 [A] 那段输出。

用法：
    export LD_LIBRARY_PATH=/usr/lib/wsl/lib
    ~/pyenv-torch/bin/python probe_trt_profiler.py
"""
import os
import sys

import numpy as np
import tensorrt as trt
import torch

WORK = os.environ.get("TRT_WORKDIR", "/mnt/d/WorkBuddy/_scratch/libtorch_cuda")
LOGGER = trt.Logger(trt.Logger.WARNING)
BATCH = 512


def probe(tag, engine):
    ctx = engine.create_execution_context()
    print(f"[{tag}]")
    print(f"  ctx.profiler存在: {ctx.profiler is not None}")
    for attr in ("profiler", "profile_stream", "engine_host_time"):
        print(f"  engine.{attr}: {hasattr(engine, attr)}")
    if ctx.profiler is None:
        return False

    iname = engine.get_tensor_name(0)
    dt = engine.get_tensor_dtype(iname)
    tdt = torch.float16 if dt == trt.DataType.HALF else torch.float32
    d_in = torch.empty(BATCH * 3 * 32 * 32, dtype=tdt, device="cuda")
    d_out = torch.empty(BATCH * 10, dtype=torch.float32, device="cuda")
    bindings = [int(d_in.data_ptr()), int(d_out.data_ptr())]
    ctx.set_input_shape(iname, (BATCH, 3, 32, 32))
    d_in.fill_(0.5)

    for _ in range(3):
        ctx.execute_v2(bindings)
    torch.cuda.synchronize()
    n = len(ctx.profiler)
    print(f"  执行 3 轮后层数: {n}")
    if n == 0:
        return False
    info = ctx.profiler.get_layer(0)
    print(f"  层0: {info.name[:60]}")
    print(f"  stats 键数: {len(info.stats) if info.stats else 0}")
    if info.stats:
        k = list(info.stats.keys())[0]
        print(f"  样例统计字段: {list(info.stats[k].keys())}")
    return True


def main():
    onnx = os.path.join(WORK, "resnet18_cifar10_int8.onnx")
    if not os.path.exists(onnx):
        print(f"缺 {onnx}")
        return 2

    # ---- A: 本进程 build + DETAILED ----
    print("=== A: 本进程 build，config.profiling_verbosity = DETAILED ===")
    builder = trt.Builder(LOGGER)
    net = builder.create_network()
    p = trt.OnnxParser(net, LOGGER)
    with open(onnx, "rb") as f:
        if not p.parse(f.read()):
            print("ONNX 解析失败")
            return 2
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    print(f"  设完 profiling_verbosity = {cfg.profiling_verbosity}")
    prof = builder.create_optimization_profile()
    iname = net.get_input(0).name
    prof.set_shape(iname, (1, 3, 32, 32), (BATCH, 3, 32, 32), (BATCH, 3, 32, 32))
    cfg.add_optimization_profile(prof)
    print("  build ...")
    blob = builder.build_serialized_network(net, cfg)
    if blob is None:
        print("  build 失败")
        return 2
    eng = trt.Runtime(LOGGER).deserialize_cuda_engine(blob)
    ok_a = probe("A", eng)
    # 注意：builder.build 后要用 builder.deserialize，而不是 Runtime().deserialize
    if ok_a:
        with open(os.path.join(WORK, "probe_int8_b512.trt"), "wb") as f:
            f.write(bytes(blob))
        print("\n结论：profiling 可用，但必须用 **builder.deserialize_cuda_engine(blob)**，"
              "不能 Runtime().deserialize()")
        print("（Runtime().deserialize 拿到的是无 profiler 的执行上下文）")
        return 0

    print("\n结论：builder.deserialize 也不行，TRT 11.4 的逐层 profiler 不可用")
    return 1


if __name__ == "__main__":
    sys.exit(main())
