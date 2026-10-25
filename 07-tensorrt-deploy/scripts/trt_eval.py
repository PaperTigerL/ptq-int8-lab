"""TensorRT 评测：动态 batch + optimization profile。

口径（别改，改了数字就不可比）：
  - 只计 forward + synchronize，不含数据准备
  - 归一化 = (x/255 - mean)/std，与 ncnn / libtorch 侧评测器完全一致
  - 全量 10000 张（int8 用 --skip 跳过校准集那部分）

本机注意事项：
  - buffer 用 torch 的 CUDA tensor（这台机器没装 pycuda），省依赖
  - python 解释器 ~/pyenv-torch/bin/python（torch 2.6.0+cu124, tensorrt 11.4.0.106）
  - 运行必须带 LD_LIBRARY_PATH=/usr/lib/wsl/lib

用法:
    python trt_eval.py --model fp16 --batch 3072
    python trt_eval.py --model int8 --batch 3072 --skip 1000
    python trt_eval.py --model fp32 --batch 512
"""
import argparse
import time

import numpy as np
import tensorrt as trt
import torch

from paths import TRT_WORKDIR, ensure_workdir, take_block, load_raw

REC = 3073
LOGGER = trt.Logger(trt.Logger.WARNING)

NP_DTYPE = {trt.DataType.FLOAT: np.float32, trt.DataType.HALF: np.float16}
TC_DTYPE = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16}


def model_path(name):
    ensure_workdir()
    return TRT_WORKDIR / f"resnet18_cifar10_{name}.onnx"


def build_engine(onnx_path, batch, rebuild=False):
    plan_path = TRT_WORKDIR / f"{onnx_path.stem}_b{batch}.trt"

    if plan_path.exists() and not rebuild:
        print("复用 engine:", plan_path.name)
        with open(plan_path, "rb") as f:
            return trt.Runtime(LOGGER).deserialize_cuda_engine(f.read())

    builder = trt.Builder(LOGGER)
    network = builder.create_network()
    parser = trt.OnnxParser(network, LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("parse error:", parser.get_error(i))
            raise SystemExit("ONNX 解析失败")
    in_name = network.get_input(0).name
    print(f"ONNX {onnx_path.name}: {network.num_layers} layers, input={in_name}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    # optimization profile 的 opt 点就是实际 batch，让 tactic 就着这个 shape 调
    prof = builder.create_optimization_profile()
    prof.set_shape(in_name, (1, 3, 32, 32), (batch, 3, 32, 32), (batch, 3, 32, 32))
    config.add_optimization_profile(prof)

    print("build engine ...")
    t0 = time.time()
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise SystemExit("engine 构建失败")
    print(f"built in {time.time() - t0:.1f}s, {plan.nbytes / 1e6:.1f} MB")
    with open(plan_path, "wb") as f:
        f.write(bytes(plan))
    return trt.Runtime(LOGGER).deserialize_cuda_engine(plan)


def gpu_state():
    """返回 (显存占用 MiB, 利用率 %)，拿不到就返回 (-1, -1)。

    ⚠️ 采样时机很要紧（2026-10-20 实测发现）：
       原来只在**轮次结束后**采样，那时本进程已经跑完，
       nvidia-smi 采到的是**别的进程**的负载 ——
       结果出现「util 66% 的轮次反而比 util 11% 的轮次更快」
       这种自相矛盾的数据（fp32: 0.0276 vs 0.0330）。

       现在改为：轮次**开始前**和**结束后**各采一次，
       用结束时减去开始时的差值不好算（噪声大），
       所以输出时把两个值都打出来，由 summarize_idle.py 用
       **开始前的 util** 做分档（那才是该轮测量时的外部负载）。
    """
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        used, util = [int(x) for x in out.split(",")]
        return used, util
    except Exception:
        return -1, -1


def run_once(ctx, engine, raw, skip, n, batch_size,
             in_name, ndt, d_in, d_out, bindings, host_in, timed=True):
    """跑完一遍数据集。返回 (准确率, 毫秒/张)。"""
    correct = 0
    processed = skip
    end = skip + n
    total_ms = 0.0
    # 每轮开头必须恢复 batch_size 的 shape：上一轮尾部改过 shape 不恢复的话，
    # 下一轮整批会按尾部的 shape 跑，输出全是垃圾（实测精度掉到 51.76%）
    cur_shape = batch_size
    ctx.set_input_shape(in_name, (batch_size, 3, 32, 32))

    while processed < end:
        bs = min(batch_size, end - processed)
        if bs != cur_shape:
            ctx.set_input_shape(in_name, (bs, 3, 32, 32))
            cur_shape = bs
        px, labels = take_block(raw, processed, bs)
        src = torch.from_numpy(np.ascontiguousarray(px.astype(ndt))).view(-1)
        host_in[: src.numel()].copy_(src)
        d_in[: src.numel()].copy_(host_in[: src.numel()], non_blocking=True)

        torch.cuda.synchronize()       # 数据就位才开始计时
        t0 = time.perf_counter()
        ctx.execute_v2(bindings)
        torch.cuda.synchronize()
        total_ms += (time.perf_counter() - t0) * 1000

        pred = d_out[: bs * 10].view(bs, 10).float().argmax(1).cpu().numpy()
        correct += int((pred == labels).sum())
        processed += bs

    return 100.0 * correct / n, (total_ms / n if timed else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fp16", choices=["fp32", "fp16", "int8"])
    ap.add_argument("--batch", type=int, default=3072)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip", type=int, default=0,
                    help="跳过前 N 张：int8 要跳过校准集，否则精度虚高")
    ap.add_argument("--repeat", type=int, default=3,
                    help="重复跑几遍取最优（这台机器常被别的进程占着，单次不可信）")
    args = ap.parse_args()

    onnx_path = model_path(args.model)
    if not onnx_path.exists():
        raise SystemExit(f"先跑 export_onnx.py / quantize_int8.py，缺 {onnx_path}")

    batch_size = args.batch
    engine = build_engine(onnx_path, batch_size, args.rebuild)
    ctx = engine.create_execution_context()

    in_name, out_name = engine.get_tensor_name(0), engine.get_tensor_name(1)
    dt, dt_o = engine.get_tensor_dtype(in_name), engine.get_tensor_dtype(out_name)
    ndt, tdt = NP_DTYPE[dt], TC_DTYPE[dt]
    print(f"engine: in={in_name} {dt} | out={out_name} {dt_o}")

    raw = load_raw()
    total = raw.size // REC
    n = total - args.skip
    if args.limit:
        n = min(n, args.limit)

    max_el = batch_size * 3 * 32 * 32
    d_in = torch.empty(max_el, dtype=tdt, device="cuda")
    d_out = torch.empty(batch_size * 10, dtype=TC_DTYPE.get(dt_o, torch.float32), device="cuda")
    bindings = [int(d_in.data_ptr()), int(d_out.data_ptr())]
    host_in = torch.empty(max_el, dtype=tdt).pin_memory()
    ctx.set_input_shape(in_name, (batch_size, 3, 32, 32))
    for _ in range(3):                 # warmup
        ctx.execute_v2(bindings)
    torch.cuda.synchronize()

    used, util = gpu_state()
    print(f"样本 {n} (skip {args.skip}), batch {batch_size} | GPU 开跑前 {used} MiB / {util}%")
    if util > 20:
        print("  ⚠️ GPU 正被别的进程占着，这一轮数字会被污染，建议等空闲再测")

    correct, acc = 0, 0.0
    measurements = []
    for r in range(args.repeat):
        # ⚠️ 轮次**开始前**采样：此刻的 util 反映的是「本进程之外的负载」，
        #   才是该轮测量结果是否可信的依据。结束后采样已经晚了。
        u0, ul0 = gpu_state()
        a, ms = run_once(ctx, engine, raw, args.skip, n, batch_size,
                         in_name, ndt, d_in, d_out, bindings, host_in)
        acc = a
        measurements.append(ms)
        u1, ul1 = gpu_state()
        # 打印开始前的值（分档依据）+ 结束后的值（供参考）
        print(f"  round {r + 1}: {ms:.4f} ms/img  精度 {a:.2f}%  "
              f"(开跑前 GPU {u0} MiB / {ul0}%)")
        print(f"            (跑完后 GPU {u1} MiB / {ul1}%)")

    best = min(measurements)
    measurements.sort()
    mid = measurements[len(measurements) // 2]
    print("----------------------------------------")
    print(f"精度 {acc:.2f}%")
    print(f"最优 {best:.4f} ms/img | 中位 {mid:.4f} | 全量 10000 张约 {best * 10:.2f} s")
    if max(measurements) > best * 1.3:
        print(f"  ⚠️ 波动 {best:.4f} ~ {max(measurements):.4f}，机器忙，取最优值")


if __name__ == "__main__":
    main()
