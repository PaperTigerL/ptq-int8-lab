// 逐层对比工具：同一个模型、同一张图，分别走 CPU(Mat) 和 GPU(CudaMat)，
// 打印每一个 blob 的 L2 范数，定位第一个分叉的层。
//
// 用途：ncnn-with-cuda 这个 fork 的 CUDA int8 / fp32 路径精度不对时，
// 靠猜 kernel 是猜不出来的 —— 直接看数据在哪一层开始偏。
//
// 用法：
//   ./ncnn_cuda_cmp <param> <bin> <cifar_raw> [--int8|--fp32] [--img N]
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "allocator.h"
#include "gpu.h"
#include "net.h"

#include "cifar10.h"

class NetCUDA : public ncnn::Net
{
public:
    using ncnn::Net::blobs;
    using ncnn::Net::find_blob_index_by_name;
    using ncnn::Net::forward_layer;
};

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
        for (int i = 0; i < plane; i++)
            dst[i] = (float)src[i] * scale + bias;
    }
}

static double norm_of(const ncnn::Mat& m)
{
    if (m.empty()) return -1.0;
    size_t n = m.total();
    double s = 0.0;
    // int8 blob 按 signed char 读
    if (m.elemsize == 1)
    {
        const signed char* p = (const signed char*)m.data;
        for (size_t i = 0; i < n; i++) s += (double)p[i] * p[i];
    }
    else
    {
        const float* p = (const float*)m.data;
        for (size_t i = 0; i < n; i++) s += (double)p[i] * p[i];
    }
    return std::sqrt(s / (double)n);
}

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        fprintf(stderr, "usage: %s <param> <bin> <cifar_raw> [--int8|--fp32] [--img N]\n", argv[0]);
        return 2;
    }
    bool use_int8 = true;
    int img = 0;
    for (int i = 4; i < argc; i++)
    {
        if (!strcmp(argv[i], "--int8")) use_int8 = true;
        else if (!strcmp(argv[i], "--fp32")) use_int8 = false;
        else if (!strcmp(argv[i], "--img") && i + 1 < argc) img = atoi(argv[++i]);
    }

    ncnn::try_initialize_cuda_gpu_instances();
    if (ncnn::get_cuda_gpu_count() <= 0) { fprintf(stderr, "无 CUDA 设备\n"); return 1; }

    const ncnn::CudaDevice* dev = ncnn::get_current_gpu_device();
    // 对比工具追求的是「每次都拿到干净显存」，所以用裸分配器（不做任何复用）
    std::shared_ptr<ncnn::CudaAllocator> alloc = ncnn::get_current_gpu_allocator();

    NetCUDA net;
    net.opt.use_int8_inference = use_int8;
    net.opt.num_threads = 1;
    // lightmode 必须开着：关掉的话这个 fork 的 forward 会直接返回 -1
    // （实测：ret=-1，不是 -100 那种 create 失败）。中间 blob 会被回收，
    // 所以下面每个 blob 都单独从头跑一遍。
    net.opt.lightmode = true;
    net.opt.blob_cuda_allocator = alloc;
    net.opt.workspace_cuda_allocator = alloc;

    if (net.load_param(argv[1]) != 0) { fprintf(stderr, "load_param 失败\n"); return 1; }
    if (net.load_model(argv[2]) != 0) { fprintf(stderr, "load_model 失败\n"); return 1; }

    cifar10::Batch b;
    if (!b.load(argv[3])) { fprintf(stderr, "读数据失败\n"); return 1; }
    cifar10::Sample s = b.at(img);
    ncnn::Mat in(cifar10::kWidth, cifar10::kHeight, 3);
    fill_input(in, s.pixels);

    const int in_idx = net.find_blob_index_by_name("input");
    const int out_idx = net.find_blob_index_by_name("logits");
    if (in_idx < 0 || out_idx < 0) { fprintf(stderr, "找不到 input/logits\n"); return 1; }

    // 先确认整网能跑通（拿不到 logits 的话后面逐层对比没有意义）
    {
        std::vector<ncnn::CudaMat> g0(net.blobs.size());
        g0[in_idx] = ncnn::CudaMat(in, alloc);
        int r = net.forward_layer(net.blobs[out_idx].producer, g0, net.opt);
        cudaDeviceSynchronize();
        if (r != 0)
        {
            fprintf(stderr, "GPU 整网 forward 失败，ret=%d, cuda err=%d (%s)\n",
                    r, (int)cudaGetLastError(), cudaGetErrorString(cudaGetLastError()));
            return 1;
        }
    }

    // ---------- CPU 逐个抽 ----------
    printf("%-4s %-28s %-14s %-14s %-8s\n", "idx", "blob", "CPU(RMS)", "GPU(RMS)", "判定");
    printf("----------------------------------------------------------------------\n");
    bool diverged = false;
    for (size_t i = 0; i < net.blobs.size(); i++)
    {
        const std::string& name = net.blobs[i].name;
        if (name.empty()) continue;
        // 跳过没有生产者的 blob（权重）以及 input 本身：
        // input 的生产者是 Input 层，它没有 bottom，走 forward_layer 的
        // 多输入分支，而这个 fork 的 input_cuda 只实现了单输入路径 —— 直接段错误。
        if (net.blobs[i].producer < 0) continue;
        if ((int)i == in_idx) continue;

        fprintf(stderr, "[%zu] %s ... ", i, name.c_str());
        fflush(stderr);

        ncnn::Extractor ex = net.create_extractor();
        ex.set_light_mode(false);
        ex.input("input", in);
        fprintf(stderr, "cpu... ", i);
        fflush(stderr);
        ncnn::Mat cpu_out;
        if (ex.extract(name.c_str(), cpu_out) != 0) { fprintf(stderr, "cpu extract 失败\n"); continue; }
        fprintf(stderr, "gpu... ", i);
        fflush(stderr);

        // GPU 单独跑一次到这个 blob（lightmode 下中间 blob 会被回收，
        // 所以每个 blob 都得从头跑一遍；一张图 60 个 blob，代价可以忽略）
        std::vector<ncnn::CudaMat> g(net.blobs.size());
        g[in_idx] = ncnn::CudaMat(in, alloc);
        int r = net.forward_layer(net.blobs[i].producer, g, net.opt);
        cudaDeviceSynchronize();
        if (r != 0)
        {
            printf("%-4zu %-28s %-14s %-14s %-8s\n", i, name.c_str(), "-", "-", "GPU失败");
            continue;
        }
        if (g[i].empty())
        {
            fprintf(stderr, "gpu blob 空\n");
            printf("%-4zu %-28s %-14s %-14s %-8s\n", i, name.c_str(), "-", "-", "GPU空");
            continue;
        }
        ncnn::Mat gpu_out(g[i]);
        cudaDeviceSynchronize();
        fprintf(stderr, "ok\n");

        double nc = norm_of(cpu_out);
        double ng = norm_of(gpu_out);
        const char* verdict = "-";
        if (nc > 0 && ng >= 0)
        {
            double rel = std::fabs(nc - ng) / (nc > 1e-9 ? nc : 1.0);
            if (rel > 0.02) { verdict = !diverged ? "<<< 首个分叉" : "偏离"; diverged = true; }
            else verdict = "ok";
        }
        printf("%-4zu %-28s %-14.6f %-14.6f %-8s\n", i, name.c_str(), nc, ng, verdict);
    }
    return 0;
}
