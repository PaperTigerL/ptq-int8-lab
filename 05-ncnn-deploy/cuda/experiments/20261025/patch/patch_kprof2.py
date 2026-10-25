#!/usr/bin/env python3
"""Temporary KPROF2: per-call GPU time of fused pack, plain pack and conv.
Guarded by NCNN_KPROF=1; reverted after measuring the fused build."""
import io

PATH = "/home/user/ncnn-cuda/src/layer/cuda/convolution_cuda.cpp"

GLOBALS = r'''
// ---- temporary KPROF2 instrumentation (NCNN_KPROF=1), removed before commit ----
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace {

const bool g_kprof_on = std::getenv("NCNN_KPROF") != nullptr;
double g_kprof_pack_fused_ms = 0.0, g_kprof_pack_plain_ms = 0.0, g_kprof_conv_ms = 0.0;
long g_kprof_npack_fused = 0, g_kprof_npack_plain = 0, g_kprof_nconv = 0;

struct KProfEvent
{
    bool on;
    cudaEvent_t a, b;
    double* acc;
    long* cnt;
    KProfEvent(bool on_, double* acc_, long* cnt_) : on(on_), acc(acc_), cnt(cnt_)
    {
        if (!on) return;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        cudaEventRecord(a);
    }
    ~KProfEvent()
    {
        if (!on) return;
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, a, b);
        *acc += ms;
        (*cnt)++;
        cudaEventDestroy(a);
        cudaEventDestroy(b);
    }
};

void g_kprof_dump()
{
    if (!g_kprof_on) return;
    printf("KPROF2 pack_fused_ms=%.2f n=%ld | pack_plain_ms=%.2f n=%ld | conv_ms=%.2f n=%ld\n",
           g_kprof_pack_fused_ms, g_kprof_npack_fused, g_kprof_pack_plain_ms,
           g_kprof_npack_plain, g_kprof_conv_ms, g_kprof_nconv);
}

struct KProfAtExit
{
    KProfAtExit()
    {
        if (g_kprof_on) atexit(g_kprof_dump);
    }
} g_kprof_atexit;

} // namespace

// ---- end KPROF2 ----
'''

FUSED_PACK_OLD = """            implicit_pack_padded_int8_c4_launch(
                static_cast<const signed char*>(bottom_blob_unbordered.get_craw_data()),
                (int)bottom_blob_unbordered.cstep, C, H, W,
                pad_top, pad_left, pad_bottom, pad_right,
                static_cast<signed char*>(gpu_implicit_pack.get_raw_data()), PH, PW);"""
FUSED_PACK_NEW = """            {
                KProfEvent kp(g_kprof_on, &g_kprof_pack_fused_ms, &g_kprof_npack_fused);
                implicit_pack_padded_int8_c4_launch(
                    static_cast<const signed char*>(bottom_blob_unbordered.get_craw_data()),
                    (int)bottom_blob_unbordered.cstep, C, H, W,
                    pad_top, pad_left, pad_bottom, pad_right,
                    static_cast<signed char*>(gpu_implicit_pack.get_raw_data()), PH, PW);
            }"""

PLAIN_PACK_OLD = """            implicit_pack_int8_c4_launch(
                static_cast<const signed char*>(bottom_blob_bordered.get_craw_data()),
                (int)bottom_blob_bordered.cstep, C, bh, bw,
                static_cast<signed char*>(gpu_implicit_pack.get_raw_data()));"""
PLAIN_PACK_NEW = """            {
                KProfEvent kp(g_kprof_on, &g_kprof_pack_plain_ms, &g_kprof_npack_plain);
                implicit_pack_int8_c4_launch(
                    static_cast<const signed char*>(bottom_blob_bordered.get_craw_data()),
                    (int)bottom_blob_bordered.cstep, C, bh, bw,
                    static_cast<signed char*>(gpu_implicit_pack.get_raw_data()));
            }"""

CONV_OLD = """            implicit_conv_int8_launch(
                static_cast<const signed char*>(gpu_implicit_pack.get_craw_data()),
                C, PH, PW,"""
CONV_NEW = """            {
            KProfEvent kp(g_kprof_on, &g_kprof_conv_ms, &g_kprof_nconv);
            implicit_conv_int8_launch(
                static_cast<const signed char*>(gpu_implicit_pack.get_craw_data()),
                C, PH, PW,"""
CONV_CLOSE_OLD = """                use_int8_requantize, activation_type,
                top_blob.get_raw_data(), (int)top_blob.cstep);
            return 0;
        }
    }

    CudaMat bottom_blob_bordered;"""
CONV_CLOSE_NEW = """                use_int8_requantize, activation_type,
                top_blob.get_raw_data(), (int)top_blob.cstep);
            }
            return 0;
        }
    }

    CudaMat bottom_blob_bordered;"""

CONV_PLAIN_OLD = """            implicit_conv_int8_launch(
                static_cast<const signed char*>(gpu_implicit_pack.get_craw_data()),
                C, bh, bw,"""
CONV_PLAIN_NEW = """            {
            KProfEvent kp(g_kprof_on, &g_kprof_conv_ms, &g_kprof_nconv);
            implicit_conv_int8_launch(
                static_cast<const signed char*>(gpu_implicit_pack.get_craw_data()),
                C, bh, bw,"""
CONV_PLAIN_CLOSE_OLD = """                use_int8_requantize, activation_type,
                top_blob.get_raw_data(), (int)top_blob.cstep);

            return 0;"""
CONV_PLAIN_CLOSE_NEW = """                use_int8_requantize, activation_type,
                top_blob.get_raw_data(), (int)top_blob.cstep);
            }

            return 0;"""


def main():
    with io.open(PATH, "r", encoding="utf-8") as f:
        text = f.read()
    if "g_kprof_on" in text:
        raise SystemExit("already instrumented")
    pos = text.index("namespace ncnn {") + len("namespace ncnn {")
    text = text[:pos] + GLOBALS + text[pos:]
    for old, new in ((FUSED_PACK_OLD, FUSED_PACK_NEW), (PLAIN_PACK_OLD, PLAIN_PACK_NEW),
                     (CONV_OLD, CONV_NEW), (CONV_CLOSE_OLD, CONV_CLOSE_NEW),
                     (CONV_PLAIN_OLD, CONV_PLAIN_NEW), (CONV_PLAIN_CLOSE_OLD, CONV_PLAIN_CLOSE_NEW)):
        if old not in text:
            raise SystemExit("marker missing: %r" % old[:70])
        text = text.replace(old, new, 1)
    with io.open(PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print("instrumented fused build")


if __name__ == "__main__":
    main()
