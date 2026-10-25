"""TensorRT 评测：CUDA Graph 版本。

为什么要单独写一个：
    trt_eval.py 每批都调一次 ctx.execute_v2()，ResNet-18 在 32×32 上**层多而每层极小**
    （20 个 conv + 一堆 elementwise），每层的 kernel 都很短，CPU 侧 launch 开销占比很高。
    CUDA Graph 把整批推理固化成一个 graph，之后每次只要一次 graph launch，
    把这部分开销整个抹掉 —— 这是生产环境部署小模型的标准做法。

口径必须和 trt_eval.py 完全一致，否则数字不可比：
  - 只计推理（graph replay）+ synchronize，不含 H2D/D2H、不含数据准备
  - 归一化 = (x/255 - mean)/std
  - 全量 10000 张，多轮取最优，每轮打印 GPU 占用

⚠️ CUDA Graph 的硬约束：捕获期间 shape、指针地址、流都不能变。
   所以：
     - H2D 必须在 replay 之前单独做（pinned memory + non_blocking）
     - 尾部不足一个 batch 的那一批 shape 不同，必须单独走 execute_v2，不能进 graph
     - d_in / d_out 的指针全程固定（torch tensor 不复分配）

用法:
    python trt_eval_graph.py --model fp16 --batch 512
    python trt_eval_graph.py --model fp16 --batch 512 --compare   # 同时跑非 graph 版做对照
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


def gpu_state():
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

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

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


def capture_graph(ctx, d_in, d_out, bs, host_in, raw, start, ndt, comp,
                  mode=None, warmup=5):
    """把一次推理固化成 CUDA Graph，并**验证它真的抓到了东西**。

    返回 (graph, 捕获模式)；抓不到或验证不过就抛 SystemExit。

    ⚠️⚠️ 这里有个非常危险的坑，实测踩到过：
        `relaxed` 捕获模式下 TRT 不报错，torch 只给一句 warning
        "The CUDA Graph is empty"，然后 replay 什么都不做 ——
        结果是 **0.0016 ms/img = 7258×，精度 9.67%（随机猜）**。
        快 23 倍，全是假的。d_out 根本没被写，读到的是上一次的残留。
        所以**任何 graph 优化都必须先验证输出一致**，只看速度会直接被骗。

    为什么用 execute_async_v3 而不是 execute_v2：
        execute_v2 走 TRT 自己的默认流，不一定落在 torch 的捕获流上；
        execute_async_v3 显式指定流，才能保证 kernel 被记进 graph。

    两种模式都试，但**先信 global**：relaxed 太容易静默产出空 graph。
    """
    # 用一批真实数据填好输入，后面 plain / graph 两次都用这同一份
    px, _ = take_block(raw, start, bs)
    src = torch.from_numpy(np.ascontiguousarray(px.astype(ndt))).view(-1)
    host_in[: src.numel()].copy_(src)
    d_in[: src.numel()].copy_(host_in[: src.numel()], non_blocking=True)
    torch.cuda.synchronize()

    # 旁路流 warmup：TRT 的惰性初始化（tactic / 显存池）不能被捕获，先跑掉
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            ctx.execute_async_v3(side.cuda_stream)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    # 基准输出（plain，非捕获），跑在专用流上
    with torch.cuda.stream(comp):
        ctx.execute_async_v3(comp.cuda_stream)
    torch.cuda.synchronize()
    ref = d_out[: bs * 10].clone()

    modes = [mode] if mode else ["global", "thread_local", "relaxed"]
    last_err = None
    for m in modes:
        try:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=comp, capture_error_mode=m):
                ctx.execute_async_v3(comp.cuda_stream)
            torch.cuda.synchronize()

            # 关键一步：验证 graph 真的抓到了 kernel
            d_out[: bs * 10].zero_()
            with torch.cuda.stream(comp):
                g.replay()
            torch.cuda.synchronize()
            got = d_out[: bs * 10]
            if not torch.equal(ref, got):
                diff = (ref.float() - got.float()).abs().max().item()
                print(f"  捕获模式 {m}: graph 输出与 plain 不一致（最大差 {diff:.4g}），判失败")
                del g
                continue
            return g, m
        except RuntimeError as e:
            last_err = e
            print(f"  捕获模式 {m} 抛异常: {str(e).splitlines()[0]}")
            torch.cuda.synchronize()

    raise SystemExit(f"没有一种捕获模式能拿到正确的 graph，最后错误: {last_err}")


def run_once_graph(ctx, g, raw, skip, n, batch_size, in_name, ndt,
                   d_in, d_out, host_in, comp):
    """用 graph replay 跑完一遍。返回 (准确率, ms/张, 尾部批次数)。"""
    correct = 0
    processed = skip
    end = skip + n
    total_ms = 0.0
    tail_batches = 0

    # 每轮开头恢复 batch_size：上一轮尾部改过 shape 不恢复会让输出全变垃圾
    # （trt_eval.py 里踩过，精度掉到 51.76%）
    cur_shape = batch_size
    ctx.set_input_shape(in_name, (batch_size, 3, 32, 32))

    # CUDA event 掐时间：graph replay 只有一次 launch，CPU 侧 perf_counter 的
    # 精度已经不够看了，必须用 event
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)

    while processed < end:
        bs = min(batch_size, end - processed)
        px, labels = take_block(raw, processed, bs)
        src = torch.from_numpy(np.ascontiguousarray(px.astype(ndt))).view(-1)
        host_in[: src.numel()].copy_(src)
        d_in[: src.numel()].copy_(host_in[: src.numel()], non_blocking=True)

        # H2D 在默认流上做完了，专用流要等它
        comp.wait_stream(torch.cuda.current_stream())

        if bs == batch_size:
            torch.cuda.synchronize()   # 数据就位才开始计时
            with torch.cuda.stream(comp):
                ev0.record()
                g.replay()
                ev1.record()
            torch.cuda.synchronize()
            total_ms += ev0.elapsed_time(ev1)
        else:
            # 尾部不足一批：shape 变了，graph 不能用，退回普通执行
            tail_batches += 1
            if bs != cur_shape:
                ctx.set_input_shape(in_name, (bs, 3, 32, 32))
                cur_shape = bs
            torch.cuda.synchronize()
            with torch.cuda.stream(comp):
                ev0.record()
                ctx.execute_async_v3(comp.cuda_stream)
                ev1.record()
            torch.cuda.synchronize()
            total_ms += ev0.elapsed_time(ev1)

        pred = d_out[: bs * 10].view(bs, 10).float().argmax(1).cpu().numpy()
        correct += int((pred == labels).sum())
        processed += bs

    return 100.0 * correct / n, total_ms / n, tail_batches


def run_once_plain(ctx, raw, skip, n, batch_size, in_name, ndt,
                   d_in, d_out, host_in, comp):
    """非 graph 对照版，逻辑与 trt_eval.py 的 run_once 一致。"""
    correct = 0
    processed = skip
    end = skip + n
    total_ms = 0.0
    cur_shape = batch_size
    ctx.set_input_shape(in_name, (batch_size, 3, 32, 32))
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)

    while processed < end:
        bs = min(batch_size, end - processed)
        if bs != cur_shape:
            ctx.set_input_shape(in_name, (bs, 3, 32, 32))
            cur_shape = bs
        px, labels = take_block(raw, processed, bs)
        src = torch.from_numpy(np.ascontiguousarray(px.astype(ndt))).view(-1)
        host_in[: src.numel()].copy_(src)
        d_in[: src.numel()].copy_(host_in[: src.numel()], non_blocking=True)
        comp.wait_stream(torch.cuda.current_stream())
        torch.cuda.synchronize()
        with torch.cuda.stream(comp):
            ev0.record()
            ctx.execute_async_v3(comp.cuda_stream)
            ev1.record()
        torch.cuda.synchronize()
        total_ms += ev0.elapsed_time(ev1)
        pred = d_out[: bs * 10].view(bs, 10).float().argmax(1).cpu().numpy()
        correct += int((pred == labels).sum())
        processed += bs

    return 100.0 * correct / n, total_ms / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fp16", choices=["fp32", "fp16", "int8"])
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--compare", action="store_true",
                    help="同场再跑一遍非 graph 版做对照")
    ap.add_argument("--warmup", type=int, default=5,
                    help="graph 捕获前在旁路流上跑几次")
    ap.add_argument("--capture-mode", default=None,
                    choices=["global", "thread_local", "relaxed"],
                    help="CUDA Graph 捕获模式，不给就依次降级尝试")
    args = ap.parse_args()

    ensure_workdir()
    onnx_path = TRT_WORKDIR / f"resnet18_cifar10_{args.model}.onnx"
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
    d_out = torch.empty(batch_size * 10,
                        dtype=TC_DTYPE.get(dt_o, torch.float32), device="cuda")
    bindings = [int(d_in.data_ptr()), int(d_out.data_ptr())]
    host_in = torch.empty(max_el, dtype=tdt).pin_memory()
    ctx.set_input_shape(in_name, (batch_size, 3, 32, 32))
    ctx.set_tensor_address(in_name, int(d_in.data_ptr()))
    ctx.set_tensor_address(out_name, int(d_out.data_ptr()))

    # ⚠️ TRT 自己警告过：用默认流会让它额外调 cudaStreamSynchronize 保序，
    #    所以推理一律跑在专用流上（capture 和 replay 都用同一个）。
    comp = torch.cuda.Stream()

    print("捕获 CUDA Graph（并验证输出与 plain 一致）...")
    g, used_mode = capture_graph(ctx, d_in, d_out, batch_size, host_in, raw,
                                 args.skip, ndt, comp,
                                 mode=args.capture_mode, warmup=args.warmup)
    print(f"捕获成功并验证通过，模式 {used_mode}")

    used, util = gpu_state()
    print(f"样本 {n} (skip {args.skip}), batch {batch_size} | GPU 开跑前 {used} MiB / {util}%")
    if util > 20:
        print("  ⚠️ GPU 正被别的进程占着，这一轮数字会被污染")

    base = 11.6  # ncnn CPU fp32 基线 ms/img = 1x
    acc, tails, ms_list = 0.0, 0, []
    for r in range(args.repeat):
        a, ms, tb = run_once_graph(ctx, g, raw, args.skip, n, batch_size,
                                   in_name, ndt, d_in, d_out, host_in, comp)
        acc, tails = a, tb
        ms_list.append(ms)
        u, ul = gpu_state()
        print(f"  graph round {r + 1}: {ms:.4f} ms/img  精度 {a:.2f}%  (GPU {u} MiB / {ul}%)")

    best = min(ms_list)
    print("----------------------------------------")
    print(f"CUDA Graph: 精度 {acc:.2f} | 最优 {best:.4f} ms/img -> {base / best:.0f}x"
          f" | 尾部非整批 {tails} 次")

    if args.compare:
        p_list = []
        for r in range(args.repeat):
            a, ms = run_once_plain(ctx, raw, args.skip, n, batch_size,
                                   in_name, ndt, d_in, d_out, host_in, comp)
            p_list.append(ms)
            u, ul = gpu_state()
            print(f"  plain round {r + 1}: {ms:.4f} ms/img  精度 {a:.2f}%  (GPU {u} MiB / {ul}%)")
        pbest = min(p_list)
        print("----------------------------------------")
        print(f"非 graph:   最优 {pbest:.4f} ms/img -> {base / pbest:.0f}x")
        print(f"graph 提速: {pbest / best:.2f}x  ({pbest:.4f} -> {best:.4f} ms/img)")


if __name__ == "__main__":
    main()
