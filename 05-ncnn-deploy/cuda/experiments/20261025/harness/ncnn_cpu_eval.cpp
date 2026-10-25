#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include "net.h"
#include "cifar10.h"

static const float kMean[3] = {0.4914f, 0.4822f, 0.4465f};
static const float kStd[3] = {0.2470f, 0.2435f, 0.2616f};
static inline void fill_input(ncnn::Mat& m, const uint8_t* pixels)
{
    const int plane = cifar10::kWidth * cifar10::kHeight;
    for (int c = 0; c < 3; c++)
    {
        const float scale = 1.0f / (255.0f * kStd[c]);
        const float bias = -kMean[c] / kStd[c];
        float* dst = m.channel(c);
        const uint8_t* src = pixels + c * plane;
        for (int i = 0; i < plane; i++) dst[i] = (float)src[i] * scale + bias;
    }
}
int main(int argc, char** argv)
{
    if (argc < 4) { fprintf(stderr, "usage: %s <param> <bin> <raw> [--int8|--fp32] [--limit N] [--threads N]\n", argv[0]); return 2; }
    bool int8 = true; int limit = 100, threads = 4;
    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "--int8")) int8 = true;
        else if (!strcmp(argv[i], "--fp32")) int8 = false;
        else if (!strcmp(argv[i], "--limit") && i + 1 < argc) limit = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
    }
    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    net.opt.use_int8_inference = int8;
    net.opt.lightmode = false;
    net.opt.num_threads = threads;
    if (net.load_param(argv[1]) != 0 || net.load_model(argv[2]) != 0) { fprintf(stderr, "load failed\n"); return 1; }
    cifar10::Batch b;
    if (!b.load(argv[3])) { fprintf(stderr, "data failed\n"); return 1; }
    const int n = limit > 0 ? (limit < (int)b.size() ? limit : (int)b.size()) : (int)b.size();
    int ok = 0; double ms = 0;
    for (int i = 0; i < n; i++) {
        cifar10::Sample s = b.at(i);
        ncnn::Mat in(cifar10::kWidth, cifar10::kHeight, 3);
        fill_input(in, s.pixels);
        auto t0 = std::chrono::steady_clock::now();
        ncnn::Extractor ex = net.create_extractor();
        ex.input("input", in);
        ncnn::Mat out;
        if (ex.extract("logits", out) != 0) { fprintf(stderr, "extract failed\n"); return 1; }
        ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        int best = 0;
        for (int k = 1; k < (int)out.total(); k++) if (out[k] > out[best]) best = k;
        if (best == (int)s.label) ok++;
    }
    printf("CPU %s | %d images | acc %.2f%% (%d/%d) | %.3f ms/img\n", int8 ? "int8" : "fp32", n, 100.0 * ok / n, ok, n, ms / n);
    return 0;
}
