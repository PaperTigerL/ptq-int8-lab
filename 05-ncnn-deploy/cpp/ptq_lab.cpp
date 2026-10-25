// PTQ 实验器（C++）—— 在 ncnn 的 fp32 前向上叠加运行时可配置的权重量化。
//
// 为什么不改 ncnn 内部
// --------------------
// ncnn 的量化是「离线预量化 + 打包」，为推理性能服务：权重在 load_model 里一次性
// 量化并 transform 成 pack8 布局。做 PTQ 实验要的是「秒级换配置」，两者目标冲突。
//
// 所以这里用外部驱动：
//   加载 fp32 模型 → 直接改 public 的 weight_data → 重建 pipeline → 前向
// 不改 ncnn 一行代码，量化算法完全由本文件控制。
//
// 两个必须的前提（踩过才知道）
// ---------------------------
//   1. net.opt.lightmode = false
//      默认 lightmode 会在 create_pipeline 之后 release 掉 weight_data，
//      外部拿到的是空 Mat（total() == 0）。代价是内存翻倍。
//   2. 改完权重必须 destroy_pipeline + create_pipeline
//      x86 的 forward 用的是 weight_data_tm（pipeline 里打包好的副本），
//      不改 pipeline 的话改动完全不生效 —— 静默地算旧权重。
//
// 当前阶段
// --------
// 只做「权重量化阶梯」（激活保持 fp32，即 e2 的 A32 档），用来验证本方案能否
// 复现 03-ptq-lab 的 Python 数字。激活量化（A8/A4）需要校准集 + 逐层 hook，是下一步。
//
// 用法
// ----
//   ./ptq_lab <param> <bin> <cifar_batch> [--w-bits N] [--limit N] [--threads N]
//   --w-bits 0 表示不量化（基线对照）
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "cifar10.h"
#include "layer.h"
#include "layer/convolution.h"
#include "net.h"

namespace {

const float kMean[3] = {0.4914f, 0.4822f, 0.4465f};
const float kStd[3] = {0.2470f, 0.2435f, 0.2616f};

// per-channel 对称 RTN。
// 口径必须与 03-ptq-lab/src/quant.py::quantize_per_channel_sym 完全一致：
//   scale = absmax / qmax
//   q     = round(w / scale) clamp 到 [-(qmax+1), qmax]
// 负侧是 -(qmax+1) 而不是 -qmax —— 0 必须落在网格上，所以负半轴多占一个码。
// 写成 [-qmax, qmax] 会和 Python 差一档，数字对不上。
void quantize_weight_per_channel(ncnn::Convolution* c, int bits)
{
    if (bits < 2 || bits > 8) return;

    const int qmax = (1 << (bits - 1)) - 1;
    const int qmin = -(qmax + 1);
    const int num_output = c->num_output;
    if (num_output <= 0) return;

    const int per_ch = c->weight_data_size / num_output;
    if (per_ch <= 0) return;

    float* w = c->weight_data;

    for (int p = 0; p < num_output; p++)
    {
        float* wp = w + (size_t)p * per_ch;

        float absmax = 0.f;
        for (int i = 0; i < per_ch; i++)
        {
            const float a = std::fabs(wp[i]);
            if (a > absmax) absmax = a;
        }
        if (absmax <= 0.f) continue;

        float scale = absmax / (float)qmax;
        if (scale < 1e-8f) scale = 1e-8f;

        for (int i = 0; i < per_ch; i++)
        {
            int q = (int)std::lround(wp[i] / scale);
            if (q > qmax) q = qmax;
            if (q < qmin) q = qmin;
            wp[i] = (float)q * scale;
        }
    }
}

void usage(const char* prog)
{
    fprintf(stderr,
            "用法：%s <param> <bin> <cifar_batch> [--w-bits N] [--limit N] [--threads N]\n"
            "  --w-bits  权重量化位宽，0 = 不量化（默认 0）\n"
            "  --limit   只跑前 N 张，0 = 全部（默认 0）\n"
            "  --threads ncnn 线程数（默认 1）\n",
            prog);
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        usage(argv[0]);
        return 2;
    }

    const char* param_path = argv[1];
    const char* bin_path = argv[2];
    const char* data_path = argv[3];

    int w_bits = 0;
    int limit = 0;
    int threads = 1;

    for (int i = 4; i < argc; i++)
    {
        if (strcmp(argv[i], "--w-bits") == 0 && i + 1 < argc) w_bits = atoi(argv[++i]);
        else if (strcmp(argv[i], "--limit") == 0 && i + 1 < argc) limit = atoi(argv[++i]);
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) threads = atoi(argv[++i]);
        else { fprintf(stderr, "未知参数 %s\n", argv[i]); return 2; }
    }

    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    net.opt.lightmode = false; // 必须：否则 weight_data 会被释放
    net.opt.num_threads = threads;

    if (net.load_param(param_path) != 0) { fprintf(stderr, "load_param 失败\n"); return 1; }
    if (net.load_model(bin_path) != 0) { fprintf(stderr, "load_model 失败\n"); return 1; }

    int quantized = 0;
    if (w_bits > 0)
    {
        for (size_t i = 0; i < net.layers().size(); i++)
        {
            ncnn::Layer* l = net.layers()[i];
            if (l->type != "Convolution") continue;

            ncnn::Convolution* c = (ncnn::Convolution*)l;
            if (c->weight_data.total() == 0u)
            {
                fprintf(stderr, "[警告] %s 的 weight_data 为空（lightmode 没关？）\n", c->name.c_str());
                continue;
            }
            quantize_weight_per_channel(c, w_bits);
            quantized++;
        }

        // 重建 pipeline：否则 x86 用的还是打包好的旧权重
        for (size_t i = 0; i < net.layers().size(); i++)
        {
            net.layers()[i]->destroy_pipeline(net.opt);
            net.layers()[i]->create_pipeline(net.opt);
        }
    }

    cifar10::Batch batch;
    if (!batch.load(data_path)) { fprintf(stderr, "读数据失败：%s\n", data_path); return 1; }

    const int n = (limit > 0) ? std::min(limit, batch.size()) : batch.size();

    printf("模型     : %s\n", param_path);
    printf("权重量化 : %s\n", w_bits > 0 ? std::to_string(w_bits).append(" bit（").append(std::to_string(quantized)).append(" 个 Conv）").c_str() : "不量化");
    printf("样本     : %d 张\n", n);

    int correct = 0;
    double total_ms = 0.0;

    for (int i = 0; i < n; i++)
    {
        cifar10::Sample s = batch.at(i);

        ncnn::Mat in(32, 32, 3);
        for (int c = 0; c < 3; c++)
        {
            const float mean = kMean[c] * 255.0f;
            const float norm = 1.0f / (255.0f * kStd[c]);
            float* p = in.channel(c);
            const uint8_t* src = s.pixels + (size_t)c * cifar10::kPlaneSize;
            for (int j = 0; j < cifar10::kPlaneSize; j++)
                p[j] = ((float)src[j] - mean) * norm;
        }

        const auto t0 = std::chrono::steady_clock::now();

        ncnn::Extractor ex = net.create_extractor();
        ex.input("input", in);
        ncnn::Mat out;
        if (ex.extract("logits", out) != 0) { fprintf(stderr, "extract 失败\n"); return 1; }

        const auto t1 = std::chrono::steady_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

        int best = 0;
        for (int k = 1; k < out.total(); k++)
            if (out[k] > out[best]) best = k;

        if (best == (int)s.label) correct++;
    }

    printf("----------------------------------------\n");
    printf("精度     : %.2f%%  (%d/%d)\n", 100.0 * correct / n, correct, n);
    printf("耗时     : %.3f ms/张\n", total_ms / n);

    return 0;
}
