# -*- coding: utf-8 -*-
"""绝对算力对照：这件卡「现在」还能跑出多少 TFLOPS。

用途：区分两种完全不同的解释
  (a) engine 选到的 tactic 变差  → 只有那个 engine 慢，别的 engine 不受影响
  (b) 机器状态变了（clock/功耗/温控/争用）→ **所有** engine 一起慢，且绝对算力也掉
2026-10-25 复现时两个现象同时出现过，所以必须拿一个与 TRT 无关的基准来判。
"""
import time

import torch

dev = "cuda"
print("GPU:", torch.cuda.get_device_name(0))


def timed(fn, iters, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters


def bench_fp16(n=4096, iters=30):
    a = torch.randn(n, n, device=dev, dtype=torch.float16)
    b = torch.randn(n, n, device=dev, dtype=torch.float16)
    dt = timed(lambda: torch.mm(a, b), iters)
    flops = 2 * n ** 3
    return flops / dt / 1e12, dt


def bench_int8(n=4096, iters=30):
    a = torch.randint(-128, 127, (n, n), device=dev, dtype=torch.int8)
    b = torch.randint(-128, 127, (n, n), device=dev, dtype=torch.int8)
    dt = timed(lambda: torch._int_mm(a, b), iters)
    flops = 2 * n ** 3
    return flops / dt / 1e12, dt


def bench_bandwidth(iters=30):
    x = torch.empty(256 * 1024 * 1024 // 4, device=dev, dtype=torch.float32)  # 256 MB
    y = torch.empty_like(x)
    dt = timed(lambda: y.copy_(x), iters)
    return 2 * x.numel() * 4 / dt / 1e9, dt          # 读+写 GB/s


tf16, dt16 = bench_fp16()
ti8, dti8 = bench_int8()
bw, dtbw = bench_bandwidth()
print("fp16 matmul  4096^3: %6.1f ms  -> %.2f TFLOPS" % (dt16 * 1e3, tf16))
print("int8 _int_mm 4096^3: %6.1f ms  -> %.2f TOPS" % (dti8 * 1e3, ti8))
print("memcpy 256MB       : %6.1f ms  -> %.0f GB/s" % (dtbw * 1e3, bw))
