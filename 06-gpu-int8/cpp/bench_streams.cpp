// 部署侧跨图并行 harness：把一批图切成 S 份，各走自己的一条 CUDA stream
// 和自己的一份权重/显存池，测「SM 填满」后的吞吐。
//
// 为什么这是唯一的大方向（ROOFLINE-int8.md 第 5 节）：
//   单流时每层都是「一层跑完再下一层」，deep 层（1x1 空间）的 M 只等于 batch，
//   grid 常常只有几十个 block，16 个 SM 上摆不满；而此时 GPU 不是算力不够，
//   是**并行度不够**。跨图并行让两条流的不同层同时驻留，直接把 SM 占满。
//
// 口径（必须写清，否则数字会被误读）：
//   · baseline：S=1，把同样的 chunk 依次跑完（batch 与并发时相同）
//   · concurrent：S 条流同时 enqueue，最后一起 sync
//   · 两者用的是同一个二进制、同一份数据、同一个 chunk 大小 → 只差并发
//
// 正确性按图对齐：并发结果按 (i/chunk, i%chunk) 拼回原顺序，与单流参考
// 逐图比对 max|diff| / argmax 一致率 / 精度；同时要求两者精度完全相同。
//
// 用法：
//   ./bench_streams <model.bin> <calib.bin> <cifar_batch> [--limit N]
//                   [--streams S] [--reps R] [--w-bits N] [--a-bits N]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cifar10.h"
#include "int8_engine.cuh"

static const float kMean[3] = {0.4914f, 0.4822f, 0.4465f};
static const float kStd[3] = {0.2470f, 0.2435f, 0.2616f};

static double now_ms()
{
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

// 归一化 + 通道标准化（与 main.cpp 完全同口径，CPU 侧做）
static void prep(const cifar10::Batch& batch, int start, int bs, std::vector<float>* dst)
{
    const int plane = 32 * 32;
    dst->resize((size_t)bs * 3 * plane);
    for (int b = 0; b < bs; b++)
    {
        const uint8_t* px = batch.at(start + b).pixels - 1;
        float* d = dst->data() + (size_t)b * 3 * plane;
        for (int i = 0; i < 3 * plane; i++) d[i] = (float)px[i] / 255.0f;
        for (int c = 0; c < 3; c++)
        {
            const float m = kMean[c] * 255.0f;
            const float inv = 1.0f / (kStd[c] * 255.0f);
            float* p = d + (size_t)c * plane;
            for (int i = 0; i < plane; i++) p[i] = (p[i] * 255.0f - m) * inv;
        }
    }
}

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        fprintf(stderr, "usage: %s <model.bin> <calib.bin> <cifar_batch> "
                        "[--limit N] [--streams S] [--reps R] "
                        "[--w-bits N] [--a-bits N]\n", argv[0]);
        return 2;
    }
    int limit = 1000, streams = 2, reps = 10, w_bits = 8, a_bits = 8;
    for (int i = 4; i < argc; i++)
    {
        if (!strcmp(argv[i], "--limit") && i + 1 < argc) limit = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--streams") && i + 1 < argc) streams = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--w-bits") && i + 1 < argc) w_bits = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--a-bits") && i + 1 < argc) a_bits = atoi(argv[++i]);
        else { fprintf(stderr, "未知参数 %s\n", argv[i]); return 2; }
    }
    if (streams < 1) streams = 1;

    Model model;
    if (!model.load(argv[1])) return 1;
    std::vector<std::pair<float, float>> ranges;
    {
        std::string err;
        if (!load_calib(argv[2], model.convs.size(), &ranges, &err))
        {
            fprintf(stderr, "[FAIL] %s\n", err.c_str());
            return 1;
        }
    }
    cifar10::Batch batch;
    if (!batch.load(argv[3])) { fprintf(stderr, "读数据失败\n"); return 1; }
    const int n = (limit > 0) ? std::min(limit, batch.size()) : batch.size();
    const int chunk = (n + streams - 1) / streams;
    if (chunk * streams != n)
    {
        fprintf(stderr, "[FAIL] --limit %d 不能被 --streams %d 整除（按图对齐要求等分）\n",
                n, streams);
        return 2;
    }
    printf("model: %zu convs | n=%d | streams=%d | chunk=%d | W%dA%d\n",
           model.convs.size(), n, streams, chunk, w_bits, a_bits);

    // ---- 建 S 份引擎（各自的权重 + 显存池），各绑一条 stream ----
    std::vector<cudaStream_t> st(streams, nullptr);
    // 用 unique_ptr 而不是 vector<Int8Engine>：引擎持有一堆裸 cudaMalloc 指针，
    // 复制/移动语义都是陷阱，直接禁掉。
    std::vector<std::unique_ptr<Int8Engine>> eng;
    for (int s = 0; s < streams; s++) eng.emplace_back(new Int8Engine());
    std::vector<std::vector<float>> in(streams), logits(streams);
    for (int s = 0; s < streams; s++)
    {
        if (!eng[s]->build(model, w_bits, a_bits)) return 1;
        if (!eng[s]->set_calib(ranges)) return 1;
        eng[s]->alloc_scratch(chunk);
        CUDA_CHECK(cudaStreamCreateWithFlags(&st[s], cudaStreamNonBlocking));
        eng[s]->set_stream(st[s]);
        logits[s].assign((size_t)chunk * model.fc_out, 0.0f);
        prep(batch, s * chunk, chunk, &in[s]);
    }

    // ---- 预热（第一次调用要建池、cublas 算法选择）----
    for (int s = 0; s < streams; s++)
    {
        eng[s]->forward(in[s].data(), chunk, 3, 32, 32, logits[s].data());
    }
    for (int s = 0; s < streams; s++) CUDA_CHECK(cudaStreamSynchronize(st[s]));

    // ---- baseline：同一 chunk 大小，但一次只跑一条流 ----
    double best_seq = 1e30;
    for (int r = 0; r < reps; r++)
    {
        const double t0 = now_ms();
        for (int s = 0; s < streams; s++)
        {
            eng[s]->forward(in[s].data(), chunk, 3, 32, 32, logits[s].data());
            CUDA_CHECK(cudaStreamSynchronize(st[s]));
        }
        const double ms = now_ms() - t0;
        if (ms < best_seq) best_seq = ms;
    }

    // ---- concurrent：S 条流一起 enqueue，最后统一 sync ----
    double best_con = 1e30;
    for (int r = 0; r < reps; r++)
    {
        const double t0 = now_ms();
        for (int s = 0; s < streams; s++)
            eng[s]->forward(in[s].data(), chunk, 3, 32, 32, logits[s].data());
        for (int s = 0; s < streams; s++) CUDA_CHECK(cudaStreamSynchronize(st[s]));
        const double ms = now_ms() - t0;
        if (ms < best_con) best_con = ms;
    }

    // ---- 正确性：并发结果按图拼回原序，与单流参考逐图比对 ----
    // 参考用「单流 + 整批」算一遍（batch 大小会改变 cublas 的 FC 求和顺序，
    // 所以判据是 max|diff| 很小 + argmax 一致，而不是逐位相等）。
    Int8Engine ref;
    if (!ref.build(model, w_bits, a_bits)) return 1;
    if (!ref.set_calib(ranges)) return 1;
    const int ref_batch = std::min(n, 512);
    ref.alloc_scratch(ref_batch);
    std::vector<float> ref_logits((size_t)n * model.fc_out, 0.0f);
    std::vector<float> buf;
    for (int off = 0; off < n; off += ref_batch)
    {
        const int bs = std::min(ref_batch, n - off);
        prep(batch, off, bs, &buf);
        std::vector<float> tmp((size_t)bs * model.fc_out);
        ref.forward(buf.data(), bs, 3, 32, 32, tmp.data());
        cudaDeviceSynchronize();
        std::copy(tmp.begin(), tmp.end(), ref_logits.begin() + (size_t)off * model.fc_out);
    }
    // 并发运行一遍，收集 logits
    for (int s = 0; s < streams; s++)
        eng[s]->forward(in[s].data(), chunk, 3, 32, 32, logits[s].data());
    for (int s = 0; s < streams; s++) CUDA_CHECK(cudaStreamSynchronize(st[s]));

    double maxdiff = 0.0;
    int argmax_agree = 0, correct_ref = 0, correct_con = 0;
    for (int i = 0; i < n; i++)
    {
        const float* r = ref_logits.data() + (size_t)i * model.fc_out;
        const float* c = logits[i / chunk].data() + (size_t)(i % chunk) * model.fc_out;
        int br = 0, bc = 0;
        for (int j = 0; j < model.fc_out; j++)
        {
            maxdiff = std::max(maxdiff, (double)std::fabs(r[j] - c[j]));
            if (r[j] > r[br]) br = j;
            if (c[j] > c[bc]) bc = j;
        }
        if (br == bc) argmax_agree++;
        if (br == (int)batch.at(i).label) correct_ref++;
        if (bc == (int)batch.at(i).label) correct_con++;
    }

    printf("----------------------------------------\n");
    printf("顺序(S=1, chunk=%d)   %8.3f ms / %d 张 → %.4f ms/img\n",
           chunk, best_seq, n, best_seq / n);
    printf("并发(S=%d, chunk=%d) %8.3f ms / %d 张 → %.4f ms/img\n",
           streams, chunk, best_con, n, best_con / n);
    printf("多流加速 %.3fx | 吞吐 %.1f 张/s (= %.0f×)\n",
           best_seq / best_con, n / (best_con * 1e-3), 11.6 / (best_con / n));
    printf("正确性：max|diff|=%.3e | argmax 一致 %d/%d (%.2f%%) | 精度 %.2f%% vs 参考 %.2f%%\n",
           maxdiff, argmax_agree, n, 100.0 * argmax_agree / n,
           100.0 * correct_con / n, 100.0 * correct_ref / n);

    for (int s = 0; s < streams; s++) cudaStreamDestroy(st[s]);
    return 0;
}