#!/usr/bin/env python3
"""Add an opt-in --sync-every N to the eval/dump harnesses.

Default 1 keeps the exact current behaviour.  N>1 defers the per-image
cudaDeviceSynchronize to every N images; the blob arena stays monotonic
(never reuses addresses, the known-safe mode), and only the workspace arena is
rewound after the deferred sync (N=8 keeps it ~30MB < 64MB).
"""
import io

PATHS = [
    "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda/ncnn_cuda_eval_2000c.cpp",
    "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda/ncnn_cuda_dump_logits.cpp",
]

ARG_OLD = "    int reset_every = 1;"
ARG_NEW = "    int reset_every = 1;\n    int sync_every = 1;"

PARSE_OLD = "        else if (!strcmp(argv[i], \"--reset-every\") && i + 1 < argc) reset_every = atoi(argv[++i]);"
PARSE_NEW = """        else if (!strcmp(argv[i], "--reset-every") && i + 1 < argc) reset_every = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sync-every") && i + 1 < argc) sync_every = atoi(argv[++i]);"""

BLOCK_OLD = """        if (!no_reset)
        {
            cudaDeviceSynchronize();
            if (img_seq % reset_every == 0)
            {
                // h015 语义：reset 在有活块时自动 defer。blob 每张图都有上一张的
                // blob 句柄未释放，因此实际不回退（used 单调增长 = 无地址复用）；
                // workspace 的临时块在 forward 内已释放，回退成功（used 保持 ~4MB）。
                g_blob_arena.reset();
                g_work_arena.reset();
            }
        }"""

BLOCK_NEW = """        if (!no_reset)
        {
            if (sync_every <= 1)
            {
                cudaDeviceSynchronize();
                if (img_seq % reset_every == 0)
                {
                    // h015 语义：reset 在有活块时自动 defer。blob 每张图都有上一张的
                    // blob 句柄未释放，因此实际不回退（used 单调增长 = 无地址复用）；
                    // workspace 的临时块在 forward 内已释放，回退成功（used 保持 ~4MB）。
                    g_blob_arena.reset();
                    g_work_arena.reset();
                }
            }
            else if (img_seq % sync_every == 0)
            {
                // 延迟同步（opt-in）：每 N 张才等一次 GPU；blob arena 保持单调
                // （绝不地址复用），同步后只回收已无活块的 workspace。
                cudaDeviceSynchronize();
                g_work_arena.reset();
            }
        }"""


def patch(path):
    with io.open(path, "r", encoding="utf-8") as f:
        text = f.read()
    for old, new in ((ARG_OLD, ARG_NEW), (PARSE_OLD, PARSE_NEW), (BLOCK_OLD, BLOCK_NEW)):
        if old not in text:
            raise SystemExit("marker missing in %s: %r" % (path, old[:60]))
        text = text.replace(old, new, 1)
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print("patched", path)


for p in PATHS:
    patch(p)
