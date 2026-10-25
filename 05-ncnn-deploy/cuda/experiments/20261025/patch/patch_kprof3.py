#!/usr/bin/env python3
"""Temporary KPROF3: host time of the scale H2D upload block + upload/skip counts.
Guarded by NCNN_KPROF=1; reverted before the real commit."""
import io

PATH = "/home/user/ncnn-cuda/src/layer/cuda/convolution_cuda.cpp"

GLOBALS = r'''
// ---- temporary KPROF3 (NCNN_KPROF=1), removed before commit ----
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace {

const bool g_kprof3_on = std::getenv("NCNN_KPROF") != nullptr;
double g_kprof3_upload_ms = 0.0;
long g_kprof3_upload_n = 0;
long g_kprof3_skip_n = 0;

struct KProf3Host
{
    bool on;
    std::chrono::steady_clock::time_point t0;
    KProf3Host(bool on_) : on(on_)
    {
        if (on) t0 = std::chrono::steady_clock::now();
    }
    ~KProf3Host()
    {
        if (!on) return;
        g_kprof3_upload_ms += std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  std::chrono::steady_clock::now() - t0).count() / 1e6;
        g_kprof3_upload_n++;
    }
};

void g_kprof3_dump()
{
    if (!g_kprof3_on) return;
    printf("KPROF3 scale_upload_ms=%.3f uploads=%ld skips=%ld\n",
           g_kprof3_upload_ms, g_kprof3_upload_n, g_kprof3_skip_n);
}

struct KProf3AtExit
{
    KProf3AtExit()
    {
        if (g_kprof3_on) atexit(g_kprof3_dump);
    }
} g_kprof3_atexit;

} // namespace
// ---- end KPROF3 ----
'''

OLD = """    if (!int8_scale_cache_valid
        || std::memcmp(&cached_bottom_blob_int8_scale, &bottom_blob_int8_scale, sizeof(float)) != 0
        || std::memcmp(&cached_top_blob_int8_scale, &top_blob_int8_scale, sizeof(float)) != 0)
    {
        checkCudaErrors(cudaMemcpy(gpu_bottom_blob_int8_scale, &bottom_blob_int8_scale,
                                   sizeof(float), cudaMemcpyHostToDevice));"""
NEW = """    const bool scale_need_upload = !int8_scale_cache_valid
        || std::memcmp(&cached_bottom_blob_int8_scale, &bottom_blob_int8_scale, sizeof(float)) != 0
        || std::memcmp(&cached_top_blob_int8_scale, &top_blob_int8_scale, sizeof(float)) != 0;
    if (scale_need_upload)
    {
        KProf3Host kp3(g_kprof3_on);
        checkCudaErrors(cudaMemcpy(gpu_bottom_blob_int8_scale, &bottom_blob_int8_scale,
                                   sizeof(float), cudaMemcpyHostToDevice));"""

OLD_TAIL = """        cached_top_blob_int8_scale = top_blob_int8_scale;
        int8_scale_cache_valid = true;
    }"""
NEW_TAIL = """        cached_top_blob_int8_scale = top_blob_int8_scale;
        int8_scale_cache_valid = true;
    }
    else
    {
        g_kprof3_skip_n++;
    }"""


def main():
    with io.open(PATH, "r", encoding="utf-8") as f:
        text = f.read()
    if "g_kprof3_on" in text:
        raise SystemExit("already instrumented")
    pos = text.index("namespace ncnn {") + len("namespace ncnn {")
    text = text[:pos] + GLOBALS + text[pos:]
    for old, new in ((OLD, NEW), (OLD_TAIL, NEW_TAIL)):
        if old not in text:
            raise SystemExit("marker missing: %r" % old[:60])
        text = text.replace(old, new, 1)
    with io.open(PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print("instrumented scale cache")


if __name__ == "__main__":
    main()
