// 样本级并行评测：把并行度从"单张内部"搬到"样本之间"。
//
// 为什么这可能快得多
// -----------------
// ncnn 的 conv 内部用 OMP 把一张图切给多核。但 CIFAR 32x32 的图太小，
// 一张图切 4 份之后每份的计算量已经接近 kernel 启动开销，加速比只有 1.65x。
//
// 换个切法：每张图只用 1 线程，但 12 张图同时跑。核数一样，但
// 固定开销（extractor 创建、内存分配、kernel 启动）在样本之间是并行的，
// 不占用同一条关键路径。
//
// 这一步不改 ncnn，只改评测循环的并行维度。
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "cifar10.h"
#include "net.h"

static const float kMean[3] = {0.4914f, 0.4822f, 0.4465f};
static const float kStd[3] = {0.2470f, 0.2435f, 0.2616f};
static const int kPlane = 32 * 32;

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        fprintf(stderr, "usage: %s <param> <bin> <cifar_batch> [--limit N] [--outer N] [--inner N]\n"
                        "  --outer  样本级并行线程数（默认 1）\n"
                        "  --inner  每张图内部 ncnn 线程数（默认 1）\n", argv[0]);
        return 2;
    }

    const char* param_path = argv[1];
    const char* bin_path = argv[2];
    const char* data_path = argv[3];

    int limit = 0, outer = 1, inner = 1;
    for (int i = 4; i < argc; i++)
    {
        if (!strcmp(argv[i], "--limit") && i + 1 < argc) limit = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--outer") && i + 1 < argc) outer = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--inner") && i + 1 < argc) inner = atoi(argv[++i]);
        else { fprintf(stderr, "未知参数 %s\n", argv[i]); return 2; }
    }

    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    net.opt.lightmode = false;
    net.opt.num_threads = inner;

    if (net.load_param(param_path) != 0) { fprintf(stderr, "load_param 失败\n"); return 1; }
    if (net.load_model(bin_path) != 0) { fprintf(stderr, "load_model 失败\n"); return 1; }

    cifar10::Batch batch;
    if (!batch.load(data_path)) { fprintf(stderr, "读数据失败\n"); return 1; }

    const int n = (limit > 0) ? std::min(limit, batch.size()) : batch.size();
    std::vector<int> hit(n, 0);

    const auto t0 = std::chrono::steady_clock::now();

    #pragma omp parallel for num_threads(outer) schedule(dynamic, 8)
    for (int i = 0; i < n; i++)
    {
        cifar10::Sample s = batch.at(i);

        ncnn::Mat in(32, 32, 3);
        for (int c = 0; c < 3; c++)
        {
            const float mean = kMean[c] * 255.0f;
            const float norm = 1.0f / (255.0f * kStd[c]);
            float* p = in.channel(c);
            const uint8_t* src = s.pixels + (size_t)c * kPlane;
            for (int j = 0; j < kPlane; j++) p[j] = ((float)src[j] - mean) * norm;
        }

        ncnn::Extractor ex = net.create_extractor();
        ex.input("input", in);
        ncnn::Mat out;
        if (ex.extract("logits", out) != 0) continue;

        int best = 0;
        for (int k = 1; k < out.total(); k++) if (out[k] > out[best]) best = k;
        hit[i] = (best == (int)s.label) ? 1 : 0;
    }

    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    int correct = 0;
    for (int i = 0; i < n; i++) correct += hit[i];

    printf("outer=%-2d inner=%-2d  精度 %.2f%%  %.3f ms/张  总 %.2f s\n",
           outer, inner, 100.0 * correct / n, ms / n, ms / 1000.0);
    return 0;
}
