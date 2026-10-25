# -*- coding: utf-8 -*-
"""push-20261025：可配置的 TRT 延迟测试台（与既有 trt_eval.py 口径兼容）。

与 trt_eval.py 的区别：把 builder 的所有开关暴露成参数，engine 按配置缓存，
可选 CUDA Graph / 逐层 profile，输出 JSON 便于批量扫描。

口径（与 README 一致，不改）：
  - 只计 forward（CUDA event 掐 GPU 时间），不含 H2D / 数据准备
  - 归一化 (x/255 - mean)/std
  - dataset 模式 = 老协议（逐批 H2D→sync→execute→sync，全量 10000 张，跳过 skip）
  - steady 模式 = 同址 buffer 背靠背 enqueue，量吞吐上限
  - 精度在同一遍 dataset 里算；int8 必须 --skip 1000

用法示例：
  python push_bench.py --onnx work/resnet18_cifar10_fp16.onnx --batch 512 --tag fp16_b512
  python push_bench.py --onnx ... --opt 5 --tactic jit --workspace-gb 1 --graph --profile
"""
import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import load_raw, take_block

LOG = trt.Logger(trt.Logger.WARNING)
NP_DTYPE = {trt.DataType.FLOAT: np.float32, trt.DataType.HALF: np.float16}
TC_DTYPE = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16}
TACTIC_MASK = {"default": 3, "edge": 1, "jit": 2}   # bit0=EDGE_MASK, bit1=JIT
BASE_MS = 11.6                                      # ncnn CPU fp32 基线


def gpu_state():
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        used, util = [int(x) for x in out.split(",")]
        return used, util
    except Exception:
        return -1, -1


def gpu_clocks():
    """时钟/温度。这台 1650 Ti 空闲跑 1350 MHz、上限 2100 MHz：
    空闲态每批之间 GPU 会掉回低 clock，逐批协议的 ms/img 因此被系统性抬高。
    记录时钟才知道两个数字能不能比。"""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        vals = [x.strip() for x in out.split(",")]
        return dict(sm=vals[0], sm_max=vals[1], temp=vals[2], power=vals[3])
    except Exception:
        return {}


def _deser_timing_cache(path):
    if not path or not Path(path).exists():
        return None
    data = Path(path).read_bytes()
    rt = trt.Runtime(LOG)
    for attr in ("deserialize_timing_cache",):
        fn = getattr(rt, attr, None)
        if fn:
            try:
                return fn(data)
            except Exception as e:
                print("timing cache deserialize failed:", e)
    return None


def build_engine(onnx_path, batch, engine_path, args):
    if engine_path.exists() and not args.rebuild:
        with open(engine_path, "rb") as f:
            return trt.Runtime(LOG).deserialize_cuda_engine(f.read()), 0.0, engine_path.stat().st_size / 1e6

    builder = trt.Builder(LOG)
    network = builder.create_network()
    parser = trt.OnnxParser(network, LOG)
    if not parser.parse(Path(onnx_path).read_bytes()):
        for i in range(parser.num_errors):
            print("parse error:", parser.get_error(i))
        raise SystemExit("ONNX 解析失败")
    in_name = network.get_input(0).name

    cfg = builder.create_builder_config()
    cfg.builder_optimization_level = args.opt
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_gb * (1 << 30)))
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    cfg.set_tactic_sources(TACTIC_MASK[args.tactic])
    if args.max_aux_streams is not None:
        cfg.max_aux_streams = args.max_aux_streams
    if args.tiling is not None:
        cfg.tiling_optimization_level = args.tiling
    if args.avg_timing is not None:
        cfg.avg_timing_iterations = args.avg_timing
    if args.direct_io:
        try:
            cfg.set_flag(trt.BuilderFlag.DIRECT_IO)
            print("DIRECT_IO ->", cfg.get_flag(trt.BuilderFlag.DIRECT_IO))
        except Exception as e:
            print("DIRECT_IO set failed:", e)

    prof = builder.create_optimization_profile()
    lo = batch if args.static else 1
    prof.set_shape(in_name, (lo, 3, 32, 32), (batch, 3, 32, 32), (batch, 3, 32, 32))
    cfg.add_optimization_profile(prof)

    cache = _deser_timing_cache(args.timing_cache)
    if cache is not None:
        cfg.set_timing_cache(cache, ignore_mismatch=True)
    elif args.timing_cache:
        cfg.set_timing_cache(cfg.create_timing_cache(b""), ignore_mismatch=False)

    t0 = time.time()
    plan = builder.build_serialized_network(network, cfg)
    build_s = time.time() - t0
    if plan is None:
        raise SystemExit("engine 构建失败")
    if args.timing_cache:
        try:
            c = cfg.get_timing_cache()
            if c is not None:
                Path(args.timing_cache).write_bytes(bytes(c.serialize()))
        except Exception as e:
            print("timing cache save failed:", e)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(plan))
    return trt.Runtime(LOG).deserialize_cuda_engine(plan), build_s, plan.nbytes / 1e6


def engine_name(onnx_path, args):
    st = Path(onnx_path).stat()
    key = f"{Path(onnx_path).name}:{st.st_size}:{int(st.st_mtime)}:b{args.batch}:o{args.opt}" \
          f":t{args.tactic}:w{args.workspace_gb}:a{args.max_aux_streams}:s{int(args.static)}:x{args.tiling}" \
          f":d{int(args.direct_io)}:g{args.avg_timing}"
    h = hashlib.sha1(key.encode()).hexdigest()[:10]
    return f"{Path(onnx_path).stem}__b{args.batch}_o{args.opt}_{args.tactic}_w{args.workspace_gb}" \
           f"_a{args.max_aux_streams}_x{args.tiling}_d{int(args.direct_io)}_{h}.trt"


def run_dataset_pass(ctx, bs, raw, skip, n, d_in, d_out, host_in, ndt, stream, limit=0):
    """老协议：逐批 H2D → sync → execute → sync，只累计 execute 的 GPU 时间。"""
    correct = 0
    processed = skip
    end = skip + n
    if limit:
        end = skip + min(n, limit)
    total_ms = 0.0
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    pxbuf = np.empty((bs, 3, 32, 32), dtype=np.float32)
    while processed < end:
        b = min(bs, end - processed)
        px, labels = take_block(raw, processed, b)
        pxbuf[:b] = px
        if b < bs:                      # 尾部补齐，保持静态 shape
            pxbuf[b:] = pxbuf[b - 1]
        src = torch.from_numpy(np.ascontiguousarray(pxbuf.astype(ndt))).view(-1)
        host_in[: src.numel()].copy_(src)
        d_in[: src.numel()].copy_(host_in[: src.numel()], non_blocking=True)
        stream.wait_stream(torch.cuda.current_stream())
        ev0.record(stream)
        with torch.cuda.stream(stream):
            ctx.execute_async_v3(stream.cuda_stream)
        ev1.record(stream)
        torch.cuda.synchronize()
        total_ms += ev0.elapsed_time(ev1)
        pred = d_out[: bs * 10].view(bs, 10).float().argmax(1).cpu().numpy()[:b]
        correct += int((pred == labels[:b]).sum())
        processed += b
    return 100.0 * correct / (end - skip), total_ms / (end - skip)


def bench_iters(fn, iters, stream, repeats):
    """背靠背 enqueue，event 掐总时间 / iters。返回每批 ms 列表。"""
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    out = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        ev0.record(stream)
        with torch.cuda.stream(stream):
            for _ in range(iters):
                fn()
        ev1.record(stream)
        torch.cuda.synchronize()
        out.append(ev0.elapsed_time(ev1) / iters)
    return out


class Prof(trt.IProfiler):
    def __init__(self):
        super().__init__()
        self.acc = {}

    def report_layer_time(self, layer_name, ms):
        self.acc[layer_name] = self.acc.get(layer_name, 0.0) + float(ms)


def layer_profile(ctx, iters=10):
    """在**独立的 ctx** 上跑 profile：给计时用的 ctx 挂 profiler 会引入额外开销。"""
    prof = Prof()
    ctx.profiler = prof
    stream = torch.cuda.current_stream().cuda_stream
    for _ in range(3):
        ctx.execute_async_v3(stream)
    torch.cuda.synchronize()
    prof.acc.clear()
    for _ in range(iters):
        ctx.execute_async_v3(stream)
    torch.cuda.synchronize()
    rows = sorted(((k, v / iters) for k, v in prof.acc.items()), key=lambda kv: -kv[1])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--opt", type=int, default=3)
    ap.add_argument("--tactic", default="default", choices=list(TACTIC_MASK))
    ap.add_argument("--workspace-gb", type=float, default=1.0)
    ap.add_argument("--max-aux-streams", type=int, default=None)
    ap.add_argument("--tiling", type=int, default=None)
    ap.add_argument("--avg-timing", type=int, default=None)
    ap.add_argument("--direct-io", action="store_true")
    ap.add_argument("--static", action="store_true", default=True)
    ap.add_argument("--dynamic", dest="static", action="store_false")
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--preboost", type=int, default=0,
                    help="每轮 dataset 测量前先跑多少次背靠背（顶 clock）")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--no-accuracy", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--engine-path", default=None,
                    help="直接用已有 .trt（A/B 对照用），跳过 build")
    ap.add_argument("--engine-dir", default="work/push-20261025/engines")
    ap.add_argument("--timing-cache", default="work/push-20261025/timing.cache")
    ap.add_argument("--jsonl", default="results/push-20261025/bench.jsonl")
    args = ap.parse_args()

    t_start = time.time()
    used, util0 = gpu_state()
    eng_dir = Path(args.engine_dir)
    eng_dir.mkdir(parents=True, exist_ok=True)
    if args.timing_cache:
        Path(args.timing_cache).parent.mkdir(parents=True, exist_ok=True)
    if args.engine_path:
        eng_path = Path(args.engine_path)
        with open(eng_path, "rb") as f:
            engine = trt.Runtime(LOG).deserialize_cuda_engine(f.read())
        build_s, plan_mb = 0.0, eng_path.stat().st_size / 1e6
        print("使用已有 engine:", eng_path)
    else:
        eng_path = eng_dir / engine_name(args.onnx, args)
        engine, build_s, plan_mb = build_engine(args.onnx, args.batch, eng_path, args)
    ctx = engine.create_execution_context()
    in_name, out_name = engine.get_tensor_name(0), engine.get_tensor_name(1)
    dt, dt_o = engine.get_tensor_dtype(in_name), engine.get_tensor_dtype(out_name)
    ndt, tdt, tdt_o = NP_DTYPE[dt], TC_DTYPE[dt], TC_DTYPE[dt_o]

    bs = args.batch
    d_in = torch.empty(bs * 3 * 32 * 32, dtype=tdt, device="cuda")
    d_out = torch.empty(bs * 10, dtype=tdt_o, device="cuda")
    host_in = torch.empty(bs * 3 * 32 * 32, dtype=tdt).pin_memory()
    ctx.set_input_shape(in_name, (bs, 3, 32, 32))
    try:
        print("in strides:", ctx.get_tensor_strides(in_name), "shape:", ctx.get_tensor_shape(in_name))
    except Exception as e:
        print("stride probe failed:", e)
    ctx.set_tensor_address(in_name, int(d_in.data_ptr()))
    ctx.set_tensor_address(out_name, int(d_out.data_ptr()))
    stream = torch.cuda.Stream()

    raw = load_raw()
    total = raw.size // 3073
    n = total - args.skip

    # 填充一份真实输入，供 steady / graph 用
    px, _ = take_block(raw, args.skip, bs)
    src = torch.from_numpy(np.ascontiguousarray(px.astype(ndt))).view(-1)
    host_in[: src.numel()].copy_(src)
    d_in[: src.numel()].copy_(host_in[: src.numel()], non_blocking=True)
    torch.cuda.synchronize()

    for _ in range(10):     # warmup（惰性初始化不能进 graph）
        ctx.execute_async_v3(stream.cuda_stream)
    torch.cuda.synchronize()

    rec = dict(tag=args.tag or Path(args.onnx).stem, onnx=str(args.onnx), batch=bs,
               opt=args.opt, tactic=args.tactic, workspace_gb=args.workspace_gb,
               aux=args.max_aux_streams, tiling=args.tiling, static=bool(args.static),
               direct_io=bool(args.direct_io), avg_timing=args.avg_timing,
               build_s=round(build_s, 1), plan_mb=round(plan_mb, 1),
               engine=eng_path.name, gpu_util_before=util0, gpu_mem_before=used,
               dtype=str(dt), out_dtype=str(dt_o))

    # 精度 + 老协议延迟
    acc = None
    ds = []
    if not args.no_accuracy:
        for r in range(args.repeats):
            # 可选 pre-boost：逐批协议每批之间有 python 间隙，GPU 会掉回低 clock，
            # 测出来的 ms/img 被这个环境效应抬高。跑一段背靠背把 clock 顶上去，
            # 再立刻开测，能显著减小「同一份 engine 一会儿 0.0149 一会儿 0.021」的漂移。
            if args.preboost:
                for _ in range(args.preboost):
                    ctx.execute_async_v3(stream.cuda_stream)
                torch.cuda.synchronize()
            c0 = gpu_clocks()
            a, ms = run_dataset_pass(ctx, bs, raw, args.skip, n, d_in, d_out, host_in, ndt,
                                     stream, limit=args.limit)
            acc = a
            ds.append(ms)
            print(f"  dataset round {r+1}: {ms:.4f} ms/img  精度 {a:.2f}%")
        rec.setdefault("clocks", []).append(c0)
        rec.update(acc=round(acc, 2), dataset_ms=[round(x, 5) for x in ds],
                   dataset_median=round(statistics.median(ds), 5),
                   dataset_min=round(min(ds), 5),
                   dataset_max=round(max(ds), 5))

    # 背靠背吞吐
    cst0 = gpu_clocks()
    st = bench_iters(lambda: ctx.execute_async_v3(stream.cuda_stream), args.iters, stream, args.repeats)
    rec["clocks_steady"] = [cst0, gpu_clocks()]
    print("  steady clocks:", cst0, "->", rec["clocks_steady"][1])
    st_img = [x / bs for x in st]
    rec.update(steady_ms_batch=[round(x, 4) for x in st],
               steady_median=round(statistics.median(st_img), 5),
               steady_min=round(min(st_img), 5))
    print(f"  steady: {statistics.median(st_img):.5f} ms/img (min {min(st_img):.5f})")

    # CUDA Graph
    if args.graph:
        try:
            ref = None
            d_out.zero_()
            with torch.cuda.stream(stream):
                ctx.execute_async_v3(stream.cuda_stream)
            torch.cuda.synchronize()
            ref = d_out.clone()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=stream):
                ctx.execute_async_v3(stream.cuda_stream)
            torch.cuda.synchronize()
            d_out.zero_()
            with torch.cuda.stream(stream):
                g.replay()
            torch.cuda.synchronize()
            if not torch.equal(ref, d_out):
                diff = (ref.float() - d_out.float()).abs().max().item()
                print(f"  ⚠️ graph 输出不一致（max diff {diff:.4g}），判定失败，不采信速度")
                rec["graph"] = "MISMATCH"
            else:
                gms = bench_iters(lambda: g.replay(), args.iters, stream, args.repeats)
                gimg = [x / bs for x in gms]
                rec.update(graph_median=round(statistics.median(gimg), 5),
                           graph_min=round(min(gimg), 5), graph="OK")
                print(f"  graph: {statistics.median(gimg):.5f} ms/img (min {min(gimg):.5f}) "
                      f"[已验证输出一致]")
        except Exception as e:
            rec["graph"] = f"FAIL: {str(e).splitlines()[0][:120]}"
            print("  graph 失败:", str(e).splitlines()[0][:160])

    # 逐层 profile 放最后：给 ctx 挂上 profiler 后 enqueue 会多一次回调，
    # 不能让它污染前面的计时。用计时同一个 ctx，避免第二个 workspace 在 4GB 卡上 OOM。
    if args.profile:
        rows = layer_profile(ctx, iters=10)
        tot = sum(v for _, v in rows)
        rec["profile"] = [[k, round(v, 3)] for k, v in rows]
        print(f"-- layer profile (sum {tot:.3f} ms/batch, profiler 有开销，仅看占比) --")
        for k, v in rows[:14]:
            print(f"   {k[:58]:58s} {v:8.3f} ms {100*v/tot:5.1f}%")
        rec["profile_sum_ms"] = round(tot, 3)

    rec["wall_s"] = round(time.time() - t_start, 1)
    u1, ul1 = gpu_state()
    rec["gpu_util_after"] = ul1
    # 汇总行
    best = rec.get("dataset_median")
    if best:
        rec["x_median"] = round(BASE_MS / best)
        print(f"== {rec['tag']} b{bs} opt{args.opt} {args.tactic} ws{args.workspace_gb}: "
              f"dataset {rec['dataset_min']:.4f}/{rec['dataset_median']:.4f} "
              f"(min/median) ms/img = {BASE_MS/rec['dataset_min']:.0f}x/{BASE_MS/best:.0f}x, "
              f"acc {rec.get('acc')}%, build {build_s:.0f}s, plan {plan_mb:.1f}MB")
    print("JSON " + json.dumps(rec, ensure_ascii=False))
    if args.jsonl:
        Path(args.jsonl).parent.mkdir(parents=True, exist_ok=True)
        with open(args.jsonl, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
