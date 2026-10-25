// 验证 PTQ 观测钩子：能否拿到每层输入激活的真实范围。
//
// 这是"运行时可配激活量化"的地基：
//   拿不到 range -> 算不出激活 scale -> 只能从模型文件读 -> 回到离线预量化
#include <cstdio>
#include <cstring>

#include "cifar10.h"
#include "layer.h"
#include "layer/convolution.h"
#include "net.h"

static const float kMean[3] = {0.4914f, 0.4822f, 0.4465f};
static const float kStd[3] = {0.2470f, 0.2435f, 0.2616f};

int main(int argc, char** argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s <param> <bin> [n_samples]\n", argv[0]); return 2; }

    const int n_samples = (argc > 3) ? atoi(argv[3]) : 10;

    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    net.opt.lightmode = false;
    net.opt.num_threads = 1;
    // 必须关掉 fp16/bf16 存储：ncnn 默认从第二层起用 fp16 blob（elemsize=2），
    // 观测钩子只认 fp32（elemsize=4），不关的话除 conv1 外全部拿不到 range。
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_bf16_storage = false;

    if (net.load_param(argv[1]) != 0) return 1;
    if (net.load_model(argv[2]) != 0) return 1;

    // 打开所有 Convolution 的观测
    int n_conv = 0;
    for (size_t i = 0; i < net.layers().size(); i++)
    {
        ncnn::Layer* l = net.layers()[i];
        if (l->type != "Convolution") continue;
        ((ncnn::Convolution*)l)->ptq_collect = 1;
        n_conv++;
    }
    fprintf(stderr, "开启观测的 Convolution：%d 个\n", n_conv);

    cifar10::Batch batch;
    if (!batch.load("/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/data_raw/test_batch.raw"))
    {
        fprintf(stderr, "读数据失败\n");
        return 1;
    }

    // 跑几张，累积每层的 min/max
    for (int i = 0; i < n_samples && i < batch.size(); i++)
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
        ncnn::Extractor ex = net.create_extractor();
        ex.input("input", in);
        ncnn::Mat out;
        if (ex.extract("logits", out) != 0) return 1;
    }

    printf("\n%-34s %10s %10s %10s\n", "层", "min", "max", "absmax");
    printf("------------------------------------------------------------------\n");
    for (size_t i = 0; i < net.layers().size(); i++)
    {
        ncnn::Layer* l = net.layers()[i];
        if (l->type != "Convolution") continue;
        ncnn::Convolution* c = (ncnn::Convolution*)l;
        const float amax = std::max(std::fabs(c->ptq_in_min), std::fabs(c->ptq_in_max));
        printf("%-34s %10.4f %10.4f %10.4f\n", c->name.c_str(), c->ptq_in_min, c->ptq_in_max, amax);
    }
    return 0;
}
