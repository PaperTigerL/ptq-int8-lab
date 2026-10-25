#!/usr/bin/env python3
"""Temporary KPROF instrumentation: per-call GPU time of the two scale
memcpys (host), make_padding kernel, pack kernel and implicit conv kernel.
Guarded by NCNN_KPROF=1; revert before committing the real optimization."""
import io

PATH = "/home/user/ncnn-cuda/src/layer/cuda/convolution_cuda.cpp"

GLOBALS = r'''
// ---- temporary KPROF instrumentation (NCNN_KPROF=1), removed before commit ----
#include <chrono>

namespace {

const bool g_kprof_on = std::getenv("NCNN_KPROF") != nullptr;
double g_kprof_scale_host_ms = 0.0, g_kprof_pad_ms = 0.0, g_kprof_pack_ms = 0.0, g_kprof_conv_ms = 0.0;
long g_kprof_nscale = 0, g_kprof_npad = 0, g_kprof_npack = 0, g_kprof_nconv = 0;

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

struct KProfHost
{
    bool on;
    std::chrono::steady_clock::time_point t0;
    double* acc;
    long* cnt;
    KProfHost(bool on_, double* acc_, long* cnt_) : on(on_), acc(acc_), cnt(cnt_)
    {
        if (on) t0 = std::chrono::steady_clock::now();
    }
    ~KProfHost()
    {
        if (!on) return;
        *acc += std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - t0).count() / 1e6;
        (*cnt)++;
    }
};

void g_kprof_dump()
{
    if (!g_kprof_on) return;
    printf("KPROF scale_host_ms=%.2f n=%ld | pad_ms=%.2f n=%ld | pack_ms=%.2f n=%ld | conv_ms=%.2f n=%ld\n",
           g_kprof_scale_host_ms, g_kprof_nscale, g_kprof_pad_ms, g_kprof_npad,
           g_kprof_pack_ms, g_kprof_npack, g_kprof_conv_ms, g_kprof_nconv);
}

struct KProfAtExit
{
    KProfAtExit()
    {
        if (g_kprof_on) atexit(g_kprof_dump);
    }
} g_kprof_atexit;

} // namespace

// ---- end KPROF instrumentation ----
'''

OLD_SCALE = """    checkCudaErrors(cudaMemcpy(gpu_bottom_blob_int8_scale, &bottom_blob_int8_scale,
                               sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(gpu_top_blob_int8_scale, &top_blob_int8_scale,
                               sizeof(float), cudaMemcpyHostToDevice));"""
NEW_SCALE = """    {
        KProfHost kp(g_kprof_on, &g_kprof_scale_host_ms, &g_kprof_nscale);
        checkCudaErrors(cudaMemcpy(gpu_bottom_blob_int8_scale, &bottom_blob_int8_scale,
                                   sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(gpu_top_blob_int8_scale, &top_blob_int8_scale,
                                   sizeof(float), cudaMemcpyHostToDevice));
    }"""

OLD_PAD = """    CudaMat bottom_blob_bordered;
    make_padding(bottom_blob_unbordered, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())"""
NEW_PAD = """    CudaMat bottom_blob_bordered;
    {
        KProfEvent kp(g_kprof_on, &g_kprof_pad_ms, &g_kprof_npad);
        make_padding(bottom_blob_unbordered, bottom_blob_bordered, opt);
    }
    if (bottom_blob_bordered.empty())"""

OLD_PACK = """            implicit_pack_int8_c4_launch(
                static_cast<const signed char*>(bottom_blob_bordered.get_craw_data()),
                (int)bottom_blob_bordered.cstep, C, bh, bw,
                static_cast<signed char*>(gpu_implicit_pack.get_raw_data()));"""
NEW_PACK = """            {
                KProfEvent kp(g_kprof_on, &g_kprof_pack_ms, &g_kprof_npack);
                implicit_pack_int8_c4_launch(
                    static_cast<const signed char*>(bottom_blob_bordered.get_craw_data()),
                    (int)bottom_blob_bordered.cstep, C, bh, bw,
                    static_cast<signed char*>(gpu_implicit_pack.get_raw_data()));
            }"""

OLD_CONV = """            implicit_conv_int8_launch(
                static_cast<const signed char*>(gpu_implicit_pack.get_craw_data()),
                C, bh, bw,
                static_cast<const signed char*>(gpu_weight_data_implicit.get_craw_data()),
                static_cast<const float*>(gpu_weight_data_int8_scales.get_craw_data()),
                bias_term ? static_cast<const float*>(gpu_bias_data.get_craw_data()) : nullptr,
                gpu_bottom_blob_int8_scale, gpu_top_blob_int8_scale,
                num_output, kernel_w, kernel_h, dilation_w, dilation_h,
                stride_w, stride_h, outw, outh,
                use_int8_requantize, activation_type,
                top_blob.get_raw_data(), (int)top_blob.cstep);"""
NEW_CONV = """            {
                KProfEvent kp(g_kprof_on, &g_kprof_conv_ms, &g_kprof_nconv);
                implicit_conv_int8_launch(
                    static_cast<const signed char*>(gpu_implicit_pack.get_craw_data()),
                    C, bh, bw,
                    static_cast<const signed char*>(gpu_weight_data_implicit.get_craw_data()),
                    static_cast<const float*>(gpu_weight_data_int8_scales.get_craw_data()),
                    bias_term ? static_cast<const float*>(gpu_bias_data.get_craw_data()) : nullptr,
                    gpu_bottom_blob_int8_scale, gpu_top_blob_int8_scale,
                    num_output, kernel_w, kernel_h, dilation_w, dilation_h,
                    stride_w, stride_h, outw, outh,
                    use_int8_requantize, activation_type,
                    top_blob.get_raw_data(), (int)top_blob.cast<p><cast>());
            }"""


def main():
    with io.open(PATH, "r", encoding="utf-8") as f:
        text = f.read()
    if "g_kprof_on" in text:
        raise SystemExit("already instrumented")
    anchor = "namespace ncnn {"
    pos = text.index(anchor) + len(anchor)
    text = text[:pos] + GLOBALS + text[pos:]
    for old, new in ((OLD_SCALE, NEW_SCALE), (OLD_PAD, NEW_PAD),
                     (OLD_PACK, NEW_PACK)):
        if old not in text:
            raise SystemExit("marker missing:\n" + old[:80])
        text = text.replace(old, new, 1)
    # conv call has a cstep expression we must keep verbatim
    old_conv = OLD_CONV.replace("top_blob.get_raw_data(), (int)top_blob.cstep);",
                                "top_blob.get_raw_data(), (int)top_blob.cstep);")
    if old_conv not in text:
        raise SystemExit("conv marker missing")
    new_conv = """            {
                KProfEvent kp(g_kprof_on, &g_kprof_conv_ms, &g_kprof_nconv);
                implicit_conv_int8_launch(
                    static_cast<const signed char*>(gpu_implicit_pack.get_craw_data()),
                    C, bh, bw,
                    static_cast<const signed char*>(gpu_weight_data_implicit.get_craw_data()),
                    static_cast<const float*>(gpu_weight_data_int8_scales.get_craw_data()),
                    bias_term ? static_cast<const float*>(gpu_bias_data.get_craw_data()) : nullptr,
                    gpu_bottom_blob_int8_scale, gpu_top_blob_int8_scale,
                    num_output, kernel_w, kernel_h, dilation_w, dilation_h,
                    stride_w, stride_h, outw, outh,
                    use_int8_requantize, activation_type,
                    top_blob.get_raw_data(), (int)top_blob.cstep);
            }"""
    text = text.replace(old_conv, new_conv, 1)
    with io.open(PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print("instrumented", PATH)


if __name__ == "__main__":
    main()
