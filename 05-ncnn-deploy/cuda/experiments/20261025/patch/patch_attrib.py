#!/usr/bin/env python3
"""Add sync / host-forward timers to an attribution copy of the eval harness."""
import io

PATH = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda/ncnn_cuda_eval_attrib.cpp"

GLOBALS_OLD = "static const double kBaselineMsPerImg = 11.6;"
GLOBALS_NEW = """static const double kBaselineMsPerImg = 11.6;

static double g_attrib_sync_ms = 0.0;
static long g_attrib_sync_n = 0;
static double g_attrib_host_fwd_ms = 0.0;
static long g_attrib_fwd_n = 0;"""

SYNC_OLD = """        if (!no_reset)
        {
            cudaDeviceSynchronize();"""
SYNC_NEW = """        if (!no_reset)
        {
            auto ts0 = std::chrono::steady_clock::now();
            cudaDeviceSynchronize();
            g_attrib_sync_ms += ms_of(ts0, std::chrono::steady_clock::now());
            g_attrib_sync_n++;"""

FWD_OLD = """            if (net.forward_batch(blob_mats_gpu, in_idx, out_idx, out_producer,
                                  cu_in, cnt, cu_out, ioalloc, clear_blobs) != 0)
                fatal("forward_batch 失败");"""
FWD_NEW = """            auto hf0 = std::chrono::steady_clock::now();
            const int fwd_ret = net.forward_batch(blob_mats_gpu, in_idx, out_idx, out_producer,
                                                  cu_in, cnt, cu_out, ioalloc, clear_blobs);
            g_attrib_host_fwd_ms += ms_of(hf0, std::chrono::steady_clock::now());
            g_attrib_fwd_n++;
            if (fwd_ret != 0)
                fatal("forward_batch 失败");"""

PRINT_OLD = """    printf("arena_used_mb blob=%.1f work=%.1f io=%.1f\\n",
           g_blob_arena.bytes_used() / 1048576.0,
           g_work_arena.bytes_used() / 1048576.0,
           g_io_arena.bytes_used() / 1048576.0);
    return 0;"""
PRINT_NEW = """    printf("arena_used_mb blob=%.1f work=%.1f io=%.1f\\n",
           g_blob_arena.bytes_used() / 1048576.0,
           g_work_arena.bytes_used() / 1048576.0,
           g_io_arena.bytes_used() / 1048576.0);
    printf("ATTRIB sync_ms_per_img=%.4f sync_n=%ld host_fwd_ms_per_img=%.4f enqueue_ms_per_img=%.4f fwd_n=%ld\\n",
           g_attrib_sync_ms / n, g_attrib_sync_n, g_attrib_host_fwd_ms / n,
           (g_attrib_host_fwd_ms - g_attrib_sync_ms) / n, g_attrib_fwd_n);
    return 0;"""


def repl(old, new):
    with io.open(PATH, "r", encoding="utf-8") as f:
        text = f.read()
    if old not in text:
        raise SystemExit("marker missing: %r" % old[:60])
    text = text.replace(old, new, 1)
    with io.open(PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


repl(GLOBALS_OLD, GLOBALS_NEW)
repl(SYNC_OLD, SYNC_NEW)
repl(FWD_OLD, FWD_NEW)
repl(PRINT_OLD, PRINT_NEW)
print("attrib harness patched")
