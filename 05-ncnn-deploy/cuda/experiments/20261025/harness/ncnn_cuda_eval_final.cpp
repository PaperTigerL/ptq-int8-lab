// ncnn + CUDA 评测器（H-20261023-021 验收版，ResNet-18 CIFAR-10）。
//
// 与 h015 的差异（这是 2000 张 reset 精度崩掉的根因修复）：
//   1) 回退 arena 前必须先释放所有 CudaMat，再 cudaDeviceSynchronize()，最后 reset。
//      反序时上一张图的 CudaMat 句柄会在已经被本张图复用的地址上执行 release，
//      把新块的 liveness 抹掉，随后 reset 就会跨过仍活着的块 — 别名。
//   2) 每个 batch 开始前先释放 blob_mats / cu_in / cu_out，再 reset io arena。
//   3) 分配器记录 deferred_resets / stale_frees / live_blocks，运行结束打印；
//      deferred=0、stale=0 是"全程没有跨活块回退"的可核对证据。
//
// 计时口径与 h015 一致：CUDA event 掐 GPU 段 / 张数；H2D、D2H 单独计时。
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
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

    int forward_batch(std::vector<ncnn::CudaMat>& blobs, int in_idx, int out_idx, int out_producer,
                      const std::vector<ncnn::CudaMat>& inputs, int count,
                      std::vector<ncnn::CudaMat>& outputs,
                      std::shared_ptr<ncnn::CudaAllocator> out_alloc,
                      const std::function<void()>& before_each)
    {
        if (count < 0 || (size_t)count > inputs.size()) return -1;
        outputs.resize(count);
        for (int j = 0; j < count; j++)
        {
            before_each();
            blobs[in_idx] = inputs[j];
            if (forward_layer(out_producer, blobs, opt) != 0) return -1;
            outputs[j] = blobs[out_idx].clone(out_alloc);
        }
        return 0;
    }
};

static const float kMean[3] = {0.4914f, 0.4822f, 0.4465f};
static const float kStd[3] = {0.2470f, 0.2435f, 0.2616f};

// 单调 bump arena。free 只更新 liveness，不复用地址；reset 只回退到 mark。
class BumpCudaAllocator : public ncnn::CudaAllocator
{
public:
    BumpCudaAllocator(const ncnn::CudaDevice* dev, size_t cap_bytes)
        : ncnn::CudaAllocator(dev), cap(cap_bytes), used(0), mark(0), base(0)
    {
        if (cudaMalloc(&base, cap) != cudaSuccess)
        {
            fprintf(stderr, "arena cudaMalloc %zu 字节失败\n", cap);
            exit(1);
        }
    }

    virtual void* fastMalloc(size_t size) override
    {
        size = (size + 255) & ~((size_t)255);
        if (used + size > cap)
        {
            fprintf(stderr, "arena 溢出: 需要 %zu，已用 %zu / %zu\n", size, used, cap);
            return 0;
        }
        void* p = static_cast<char*>(base) + used;
        used += size;
        live[p] = size;
        return p;
    }

    virtual void fastFree(void* p) override
    {
        if (!p) return;
        std::map<void*, size_t>::iterator it = live.find(p);
        if (it == live.end())
        {
            // 只可能是重复释放，或上一轮 reset 后仍在释放的旧句柄。
            stale_count++;
            fprintf(stderr, "WARN stale fastFree %p（不在 live 表）\n", p);
            return;
        }
        live.erase(it);
    }

    void set_mark() { mark = used; }

    bool reset()
    {
        if (live_above_mark() != 0)
        {
            deferred_count++;
            return false;
        }
        used = mark;
        return true;
    }

    bool reset_all()
    {
        if (!live.empty())
        {
            deferred_count++;
            return false;
        }
        mark = 0;
        used = 0;
        return true;
    }

    size_t bytes_used() const { return used; }
    size_t live_blocks() const { return live.size(); }
    size_t deferred_resets() const { return deferred_count; }
    size_t stale_frees() const { return stale_count; }
    size_t peak_used() const { return used; }

    size_t live_above_mark() const
    {
        size_t n = 0;
        const char* m = static_cast<char*>(base) + mark;
        for (std::map<void*, size_t>::const_iterator it = live.begin(); it != live.end(); ++it)
            if (static_cast<const char*>(it->first) >= m) n++;
        return n;
    }

private:
    size_t cap;
    size_t used;
    size_t mark;
    void* base;
    std::map<void*, size_t> live;
    size_t deferred_count = 0;
    size_t stale_count = 0;
};

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

static inline double ms_of(const std::chrono::steady_clock::time_point& a,
                           const std::chrono::steady_clock::time_point& b)
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count() / 1e6;
}

static inline void fatal(const char* msg)
{
    fprintf(stderr, "致命错误: %s\n", msg);
    exit(1);
}

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        fprintf(stderr, "usage: %s <param> <bin> <cifar_raw> [--int8|--fp32] "
                        "[--limit N] [--repeat N] [--warmup N] [--batch N] "
                        "[--reset-every N|--no-reset] [--dump-wrong N]\n", argv[0]);
        return 2;
    }

    const char* param_path = argv[1];
    const char* bin_path = argv[2];
    const char* data_path = argv[3];

    bool use_int8 = true;
    bool no_reset = false;
    bool lightmode = true;
    bool reset_blob = true;
    bool reset_work = true;
    int reset_every = 1;
    int limit = 0;
    int repeat = 3;
    int warmup = 20;
    int batch = 256;
    int dump_wrong = 0;
    int dump_logits = 0;
    int dump_hashes = 0;
    for (int i = 4; i < argc; i++)
    {
        if (!strcmp(argv[i], "--int8")) use_int8 = true;
        else if (!strcmp(argv[i], "--fp32")) use_int8 = false;
        else if (!strcmp(argv[i], "--limit") && i + 1 < argc) limit = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--repeat") && i + 1 < argc) repeat = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batch") && i + 1 < argc) batch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-reset")) no_reset = true;
        else if (!strcmp(argv[i], "--no-lightmode")) lightmode = false;
        else if (!strcmp(argv[i], "--no-reset-blob")) reset_blob = false;
        else if (!strcmp(argv[i], "--no-reset-work")) reset_work = false;
        else if (!strcmp(argv[i], "--reset-every") && i + 1 < argc) reset_every = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dump-wrong") && i + 1 < argc) dump_wrong = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dump-logits") && i + 1 < argc) dump_logits = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dump-hashes") && i + 1 < argc) dump_hashes = atoi(argv[++i]);
        else { fprintf(stderr, "未知参数 %s\n", argv[i]); return 2; }
    }
    if (batch < 1) batch = 1;
    if (reset_every < 1) reset_every = 1;
    if (repeat < 1) repeat = 1;

    ncnn::try_initialize_cuda_gpu_instances();
    if (ncnn::get_cuda_gpu_count() <= 0)
        fatal("get_cuda_gpu_count() == 0，没有可用 CUDA 设备");

    const ncnn::CudaDevice* dev = ncnn::get_current_gpu_device();
    if (!dev) fatal("get_current_gpu_device() 返回空");

    static BumpCudaAllocator g_blob_arena(dev, 512u * 1024 * 1024);
    static BumpCudaAllocator g_work_arena(dev, 128u * 1024 * 1024);
    static BumpCudaAllocator g_io_arena(dev, 64u * 1024 * 1024);
    static BumpCudaAllocator g_weight_arena(dev, 256u * 1024 * 1024);
    static BumpCudaAllocator g_weight_work_arena(dev, 128u * 1024 * 1024);

    std::shared_ptr<ncnn::CudaAllocator> alloc(
        &g_blob_arena, [](ncnn::CudaAllocator*) {});
    std::shared_ptr<ncnn::CudaAllocator> walloc(
        &g_work_arena, [](ncnn::CudaAllocator*) {});
    std::shared_ptr<ncnn::CudaAllocator> ioalloc(
        &g_io_arena, [](ncnn::CudaAllocator*) {});
    std::shared_ptr<ncnn::CudaAllocator> weight_alloc(
        &g_weight_arena, [](ncnn::CudaAllocator*) {});
    std::shared_ptr<ncnn::CudaAllocator> weight_walloc(
        &g_weight_work_arena, [](ncnn::CudaAllocator*) {});

    NetCUDA net;
    net.opt.use_int8_inference = use_int8;
    net.opt.num_threads = 4;
    net.opt.lightmode = lightmode;
    net.opt.blob_cuda_allocator = weight_alloc;
    net.opt.workspace_cuda_allocator = weight_walloc;

    printf("phase: load_param\n"); fflush(stdout);
    if (net.load_param(param_path) != 0) fatal("load_param 失败");
    printf("phase: load_model\n"); fflush(stdout);
    if (net.load_model(bin_path) != 0) fatal("load_model 失败");
    printf("phase: load_done\n"); fflush(stdout);
    net.opt.blob_cuda_allocator = alloc;
    net.opt.workspace_cuda_allocator = walloc;

    printf("phase: data_load\n"); fflush(stdout);
    cifar10::Batch batch_data;
    if (!batch_data.load(data_path)) fatal("读数据失败");
    printf("phase: data_done %d\n", batch_data.size()); fflush(stdout);

    const int n = (limit > 0) ? std::min(limit, batch_data.size()) : batch_data.size();
    if (n < 1) fatal("样本数为 0");

    const int in_idx = net.find_blob_index_by_name("input");
    const int out_idx = net.find_blob_index_by_name("logits");
    if (in_idx < 0 || out_idx < 0) fatal("找不到 input / logits blob");
    const int out_producer = net.blobs[out_idx].producer;

    printf("模型 %s\n后端 CUDA | 精度 %s | 样本 %d | batch %d | reset_every %d\n",
           param_path, use_int8 ? "int8" : "fp32", n, batch,
           no_reset ? 0 : reset_every);

    std::vector<ncnn::CudaMat> blob_mats_gpu(net.blobs.size());
    std::vector<ncnn::CudaMat> cu_in(batch);
    std::vector<ncnn::CudaMat> cu_out(batch);

    int img_seq = 0;
    // 关键顺序：release -> sync -> reset。旧句柄必须先消失，否则它们的 release
    // 会落在本张图刚复用的地址上，把新块的 liveness 抹掉。
    auto clear_blobs = [&]() {
        if (dump_hashes > 0 && img_seq > 0 && img_seq - 1 < dump_hashes)
        {
            for (size_t i = 0; i < blob_mats_gpu.size(); i++)
            {
                if (blob_mats_gpu[i].empty()) continue;
                ncnn::Mat h(blob_mats_gpu[i]);
                const unsigned char* p = h;
                size_t bytes = h.total() * h.elemsize;
                unsigned long long hv = 1469598103934665603ULL;
                for (size_t k = 0; k < bytes; k++) hv = (hv ^ p[k]) * 1099511628211ULL;
                printf("HASH img=%d blob=%s total=%zu hash=%016llx\n", img_seq - 1, net.blobs[i].name.c_str(), h.total(), hv);
            }
            fflush(stdout);
        }
        for (size_t i = 0; i < blob_mats_gpu.size(); i++) blob_mats_gpu[i].release();
        if (!no_reset)
        {
            cudaDeviceSynchronize();
            if (img_seq % reset_every == 0)
            {
                if ((reset_blob && !g_blob_arena.reset()) || (reset_work && !g_work_arena.reset()))
                {
                    fprintf(stderr, "WARN reset deferred at image %d: live above mark "
                            "blob=%zu work=%zu\n", img_seq,
                            g_blob_arena.live_above_mark(), g_work_arena.live_above_mark());
                }
            }
        }
        img_seq++;
    };

    auto run_one = [&](int idx) {
        cifar10::Sample s = batch_data.at(idx);
        ncnn::Mat in(cifar10::kWidth, cifar10::kHeight, 3);
        fill_input(in, s.pixels);
        blob_mats_gpu[in_idx] = ncnn::CudaMat(in, alloc);
        return net.forward_layer(out_producer, blob_mats_gpu, net.opt);
    };

    const int w = std::min(warmup, n);
    for (int i = 0; i < w; i++)
    {
        if (run_one(i % n) != 0) fatal("forward 失败");
    }
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) fatal("预热阶段 CUDA 报错");

    // 水位线：预热后仍存活的常驻块都在 mark 以下；先把预热句柄释放干净再回退。
    g_blob_arena.set_mark();
    g_work_arena.set_mark();
    for (size_t i = 0; i < blob_mats_gpu.size(); i++) blob_mats_gpu[i].release();
    cudaDeviceSynchronize();
    if (!g_blob_arena.reset() || !g_work_arena.reset())
        fatal("预热后 arena 仍有 mark 以上的活块");
    printf("arena 水位线: blob %zu MB | workspace %zu MB\n",
           g_blob_arena.bytes_used() >> 20, g_work_arena.bytes_used() >> 20);

    double best_gpu = 1e30;
    double best_wall = 1e30;
    int best_correct = 0;
    int best_round = 0;
    std::vector<int> wrong_idx;

    for (int r = 0; r < repeat; r++)
    {
        int ok = 0;
        double gpu_ms = 0.0;
        double h2d_ms = 0.0;
        double d2h_ms = 0.0;
        std::vector<int> round_wrong;

        auto wall0 = std::chrono::steady_clock::now();

        for (int base = 0; base < n; base += batch)
        {
            const int cnt = std::min(batch, n - base);

            // io arena 整段作废前，先释放上一批的输入/输出和最后一张的 blob。
            for (size_t i = 0; i < blob_mats_gpu.size(); i++) blob_mats_gpu[i].release();
            for (int j = 0; j < batch; j++)
            {
                cu_in[j].release();
                cu_out[j].release();
            }
            cudaDeviceSynchronize();
            if (!g_io_arena.reset_all())
                fatal("io arena 在 batch 边界仍有活块");

            auto t0 = std::chrono::steady_clock::now();
            for (int j = 0; j < cnt; j++)
            {
                cifar10::Sample s = batch_data.at(base + j);
                ncnn::Mat in(cifar10::kWidth, cifar10::kHeight, 3);
                fill_input(in, s.pixels);
                cu_in[j] = ncnn::CudaMat(in, ioalloc);
            }
            cudaDeviceSynchronize();
            h2d_ms += ms_of(t0, std::chrono::steady_clock::now());

            cudaEvent_t ev0, ev1;
            cudaEventCreate(&ev0);
            cudaEventCreate(&ev1);
            cudaEventRecord(ev0);

            if (net.forward_batch(blob_mats_gpu, in_idx, out_idx, out_producer,
                                  cu_in, cnt, cu_out, ioalloc, clear_blobs) != 0)
                fatal("forward_batch 失败");

            cudaEventRecord(ev1);
            cudaEventSynchronize(ev1);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, ev0, ev1);
            gpu_ms += ms;
            cudaEventDestroy(ev0);
            cudaEventDestroy(ev1);

            if (cudaGetLastError() != cudaSuccess) fatal("推理阶段 CUDA 报错");

            auto t2 = std::chrono::steady_clock::now();
            for (int j = 0; j < cnt; j++)
            {
                ncnn::Mat out(cu_out[j]);
                const float* p = out;
                int best = 0;
                for (int k = 1; k < cifar10::kNumClasses; k++)
                    if (p[k] > p[best]) best = k;
                const int label = (int)batch_data.at(base + j).label;
                if (dump_logits > 0 && base + j < dump_logits)
                {
                    printf("logits[%d] pred=%d label=%d:", base + j, best, label);
                    for (int k = 0; k < cifar10::kNumClasses; k++) printf(" %.5f", p[k]);
                    printf("\n");
                }
                if (best == label) ok++;
                else if (dump_wrong > 0 && (int)round_wrong.size() < dump_wrong)
                    round_wrong.push_back(base + j);
            }
            cudaDeviceSynchronize();
            d2h_ms += ms_of(t2, std::chrono::steady_clock::now());
        }

        double wall_total = ms_of(wall0, std::chrono::steady_clock::now());

        size_t mem_free = 0, mem_total = 0;
        cudaMemGetInfo(&mem_free, &mem_total);
        const double gpu_per = gpu_ms / n;
        const double wall_per = wall_total / n;
        if (gpu_per < best_gpu) best_gpu = gpu_per;
        if (wall_per < best_wall) best_wall = wall_per;
        if (ok > best_correct) { best_correct = ok; best_round = r + 1; wrong_idx = round_wrong; }
        printf("  round %d: GPU %.4f ms/img | 墙钟 %.4f ms/img "
               "(H2D %.4f + D2H %.4f) | 精度 %.2f%% | 显存已用 %.1f MB\n",
               r + 1, gpu_per, wall_per, h2d_ms / n, d2h_ms / n, 100.0 * ok / n,
               (mem_total - mem_free) / 1048576.0);
    }

    printf("----------------------------------------\n");
    printf("精度 %.2f%% (%d/%d, round %d)\n",
           100.0 * best_correct / n, best_correct, n, best_round);
    printf("最优 GPU kernel  %.4f ms/img\n", best_gpu);
    printf("最优 含传输墙钟  %.4f ms/img\n", best_wall);
    printf("arena 统计: blob deferred=%zu stale=%zu live=%zu used=%zu | "
           "work deferred=%zu stale=%zu live=%zu | io deferred=%zu stale=%zu live=%zu\n",
           g_blob_arena.deferred_resets(), g_blob_arena.stale_frees(),
           g_blob_arena.live_blocks(), g_blob_arena.bytes_used(),
           g_work_arena.deferred_resets(), g_work_arena.stale_frees(),
           g_work_arena.live_blocks(),
           g_io_arena.deferred_resets(), g_io_arena.stale_frees(),
           g_io_arena.live_blocks());
    if (dump_wrong > 0 && !wrong_idx.empty())
    {
        printf("wrong:");
        for (size_t i = 0; i < wrong_idx.size(); i++) printf(" %d", wrong_idx[i]);
        printf("\n");
    }
    return 0;
}
