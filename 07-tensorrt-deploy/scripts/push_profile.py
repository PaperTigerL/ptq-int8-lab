# -*- coding: utf-8 -*-
"""Profile an existing TRT engine layer by layer (GPU; run under flock)."""
import argparse, sys, time
from pathlib import Path
import numpy as np, tensorrt as trt, torch

sys.path.insert(0, "/mnt/d/WorkBuddy/ShowBro/lab/07-tensorrt-deploy/scripts")
from paths import load_raw, take_block

LOG = trt.Logger(trt.Logger.WARNING)
NP = {trt.DataType.FLOAT: np.float32, trt.DataType.HALF: np.float16}
TC = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16}

class Prof(trt.IProfiler):
    def __init__(self):
        super().__init__()
        self.acc = {}
    def report_layer_time(self, layer_name, ms):
        self.acc[layer_name] = self.acc.get(layer_name, 0.0) + float(ms)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--top", type=int, default=100)
    a = ap.parse_args()
    with open(a.engine, "rb") as f:
        eng = trt.Runtime(LOG).deserialize_cuda_engine(f.read())
    ctx = eng.create_execution_context()
    prof = Prof()
    ctx.profiler = prof
    in_name, out_name = eng.get_tensor_name(0), eng.get_tensor_name(1)
    ndt = NP[eng.get_tensor_dtype(in_name)]
    raw = load_raw()
    px, _ = take_block(raw, a.skip, a.batch)
    d_in = torch.from_numpy(np.ascontiguousarray(px.astype(ndt))).cuda()
    d_out = torch.empty(a.batch * 10, dtype=torch.float32, device="cuda")
    ctx.set_input_shape(in_name, (a.batch, 3, 32, 32))
    ctx.set_tensor_address(in_name, d_in.data_ptr())
    ctx.set_tensor_address(out_name, d_out.data_ptr())
    for _ in range(5):
        ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    prof.acc.clear()
    ev0 = torch.cuda.Event(True); ev1 = torch.cuda.Event(True)
    ev0.record()
    for _ in range(a.iters):
        ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    ev1.record(); torch.cuda.synchronize()
    total = ev0.elapsed_time(ev1) / a.iters
    rows = sorted(prof.acc.items(), key=lambda kv: -kv[1])
    s = sum(prof.acc.values())
    print(f"engine={Path(a.engine).name} batch={a.batch} wall={total:.3f} ms/batch "
          f"({total/a.batch*1000:.4f} ms/img) profiled_sum={s/a.iters:.3f} ms")
    print(f"{'layer':60s} {'ms':>8s} {'%':>6s}")
    for k, v in rows[:a.top]:
        ms = v / a.iters
        print(f"{k[:60]:60s} {ms:8.3f} {100*ms/total:6.1f}")

if __name__ == "__main__":
    main()
