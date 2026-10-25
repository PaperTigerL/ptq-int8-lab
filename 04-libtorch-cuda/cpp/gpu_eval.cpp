// GPU 评测器 v2：数据准备与 argmax 都搬到 GPU。
//
// v1 的瓶颈不在推理（纯 GPU 前向 0.0587 ms/img），而在
//   1) CPU 上逐张归一化 10000 张
//   2) 每 batch 把整个 logits 拷回 CPU 再做 argmax
// v2 把这两件都移到 GPU：raw 字节一次上传，归一化用一次 reshape + 广播，
// argmax 用 torch::argmax，只把 bs 个 int64 label 拷回来。
//
// 编译运行见 README.md。
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <torch/torch.h>
#include <torch/script.h>

#include "cifar10.h"

static const float kMean[3] = {0.4914f, 0.4822f, 0.4465f};
static const float kStd[3] = {0.2470f, 0.2435f, 0.2616f};
static const int kPlane = 32 * 32;

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        fprintf(stderr, "usage: %s <model.pt> <cifar_batch> [--limit N] [--batch N] [--cpu]\n", argv[0]);
        return 2;
    }

    const char* model_path = argv[1];
    const char* data_path = argv[2];

    int limit = 0, batch_size = 512;
    bool use_cpu = false;
    bool use_fp16 = false;
    for (int i = 3; i < argc; i++)
    {
        if (!strcmp(argv[i], "--limit") && i + 1 < argc) limit = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batch") && i + 1 < argc) batch_size = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cpu")) use_cpu = true;
        else if (!strcmp(argv[i], "--fp16")) use_fp16 = true;
        else { fprintf(stderr, "未知参数 %s\n", argv[i]); return 2; }
    }

    torch::Device dev = use_cpu ? torch::Device(torch::kCPU) : torch::Device(torch::kCUDA, 0);

    // cuDNN 自动为每个 conv 选最快算法（首次有探测开销，之后受益）
    if (!use_cpu)
    {
        at::globalContext().setBenchmark(true);
    }

    torch::jit::Module module;
    try
    {
        module = torch::jit::load(model_path);
        module.to(dev);
        if (use_fp16) module.to(torch::kHalf);
        module.eval();
    }
    catch (const std::exception& e)
    {
        fprintf(stderr, "加载失败: %s\n", e.what());
        return 1;
    }

    cifar10::Batch batch;
    if (!batch.load(data_path)) { fprintf(stderr, "读数据失败\n"); return 1; }

    const int n = (limit > 0) ? std::min(limit, batch.size()) : batch.size();

    // 归一化参数放 GPU，避免每 batch 重建
    const auto dt = use_fp16 ? torch::kHalf : torch::kFloat32;
    auto mean_t = torch::from_blob((void*)kMean, {3, 1, 1}, torch::kFloat32).clone().to(dev).mul(255.0).to(dt);
    auto std_t = torch::from_blob((void*)kStd, {3, 1, 1}, torch::kFloat32).clone().to(dev).mul(255.0).to(dt);
    auto inv_std = std_t.reciprocal();

    printf("模型 %s | 设备 %s | batch=%d | 样本=%d\n",
           model_path, use_cpu ? "cpu" : "cuda", batch_size, n);

    int correct = 0;
    double total_ms = 0.0;
    double prep_ms = 0.0;
    int processed = 0;

    // 复用上传缓冲：raw 字节（bs, 3073）一次 memcpy + 一次 cudaMemcpy
    std::vector<uint8_t> raw;
    raw.reserve((size_t)batch_size * cifar10::kRecordSize);

    while (processed < n)
    {
        const int bs = std::min(batch_size, n - processed);

        // 1) 拼 raw（只在 CPU 做 memcpy，不做任何浮点运算）
        raw.resize((size_t)bs * cifar10::kRecordSize);
        for (int b = 0; b < bs; b++)
        {
            cifar10::Sample s = batch.at(processed + b);
            memcpy(raw.data() + (size_t)b * cifar10::kRecordSize, s.pixels - 1, cifar10::kRecordSize);
        }

        const auto tp = std::chrono::steady_clock::now();
        // 2) 上传 -> GPU 上归一化
        auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8);
        auto raw_t = torch::from_blob(raw.data(), {(long)bs * cifar10::kRecordSize}, opts_u8)
                         .to(dev, /*non_blocking=*/true);
        // (bs*3073) -> 按样本切出 label 之后的部分 -> (bs, 3, 1024) -> (bs, 3, 32, 32)
        auto body = raw_t.reshape({bs, cifar10::kRecordSize}).slice(1, 1);
        auto px = body.reshape({bs, 3, kPlane}).to(torch::kFloat32);
        auto x = (px.reshape({bs, 3, 32, 32}) - mean_t) * inv_std;
        if (use_fp16) x = x.to(torch::kHalf);

        // 3) 前向 + GPU argmax
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(x);

        const auto t0 = std::chrono::steady_clock::now();
        auto out = module.forward(inputs).toTensor();
        if (use_fp16) out = out.to(torch::kFloat);
        auto pred = out.argmax(1);            // (bs,) int64，留在 GPU
        if (!use_cpu) torch::cuda::synchronize();
        const auto t1 = std::chrono::steady_clock::now();

        total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        prep_ms += std::chrono::duration<double, std::milli>(t0 - tp).count();

        // 4) 只拷回 label（bs 个 int64），比对在 CPU 做（便宜）
        auto pred_cpu = pred.to(torch::kCPU).contiguous();
        const long* pp = pred_cpu.data_ptr<long>();
        for (int b = 0; b < bs; b++)
            if (pp[b] == (long)batch.at(processed + b).label) correct++;

        processed += bs;
    }

    printf("----------------------------------------\n");
    printf("精度 %.2f%% (%d/%d)\n", 100.0 * correct / n, correct, n);
    printf("耗时 %.3f s  =  %.4f ms/img（其中数据准备+上传 %.3f s）\n", total_ms / 1000.0, total_ms / n, prep_ms / 1000.0);
    return 0;
}
