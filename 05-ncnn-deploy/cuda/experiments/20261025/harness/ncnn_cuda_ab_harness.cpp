// ncnn + CUDA 评测器（ResNet-18 CIFAR-10）。
//
// ============================================================================
// 为什么这份代码要单独写一遍（这是本目录最关键的一条结论）
// ============================================================================
// 这个 fork（atanmarko/ncnn-with-cuda）的 CUDA 通路**只在 Extractor 的 CudaMat
// 重载里触发**：
//
//     Extractor::input(const char* name, const CudaMat& in)
//     Extractor::extract(const char* name, CudaMat& feat)
//         -> Net::forward_layer(idx, blob_mats_gpu, opt)   // GPU
//
// 而传 ncnn::Mat 的重载走的是 CPU 通路：
//
//     Extractor::input(const char* name, const Mat& in)
//     Extractor::extract(const char* name, Mat& feat)
//         -> Net::forward_layer(idx, blob_mats, opt)       // CPU
//
// 也就是说：光设 net.opt.use_cuda_compute = true 是不够的。
// 事实上 use_cuda_compute 这个字段在 option.h 里只有声明，整个 src/ 没有任何
// 一行读它 —— 它是个死开关。真正的开关是「你往 Extractor 里塞的是 Mat 还是
// CudaMat」。之前那版评测器（走 Mat）的 LOG_LAYERS 日志里 0 个 cuda 层，
// 就是这个原因。
//
// ============================================================================
// 计时口径
// ============================================================================
// ncnn 的 Mat 只有 w/h/c，没有 batch 维，所以图里没法做 batch。
// 为了让 GPU 能连续干活，这里用「流水线」：一次上传 batch 张到显存，连续提交
// batch 次 forward，每张的输出顺手 clone 一份（10 个 float 的 D2D，可忽略），
// 最后只同步一次并统一回读。这样：
//
//   - 推理时间是 CUDA event 掐的一整段 GPU 时间 / 张数
//   - H2D、D2H 单独计时并单独打印，不混进推理时间
//   - 精度就是从这次计时跑出来的结果，不是另跑一遍 —— 数字和精度同源
//
// 归一化 (x/255 - mean)/std 把 255 吸收进常量，口径与 ptq-lab / libtorch / TRT 一致。
//
// 编译/运行见 build.sh / run.sh。
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
#include "layer/convolution.h"
#include "layer/relu.h"

#include <cmath>
#include <cstring>
#include <random>
#include <string>

// Net::forward_layer(CudaMat) 和 Net::blobs 都是 protected。
// 派生类可以把它们提到 public —— 这样才能自己驱动推理循环，
// 避开 Extractor::extract 每张图内部都 cudaDeviceSynchronize() 一次的开销。
class NetCUDA : public ncnn::Net
{
public:
    using ncnn::Net::blobs;
    using ncnn::Net::find_blob_index_by_name;
    using ncnn::Net::forward_layer;

    // Batched entry: accepts a vector of per-image CudaMats and returns one output
    // CudaMat per image. Internally it still calls forward_layer once per image, so
    // single-image semantics/kernels are unchanged; the batch dimension is the
    // harness pipeline, not a new tensor rank in ncnn's layers.
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

// ============================================================================
// 自研显存分配器：单调 arena（bump allocator）
// ============================================================================
// 为什么不用 fork 自带的两种分配器（都是实测结论，不是推测）：
//
//   1) 裸 CudaAllocator：fastMalloc 就是一次 cudaMalloc。ncnn 在 lightmode 下
//      每层每个 blob 都要 create/release，一张图几百次 cudaMalloc。
//      WSL 上 cudaMalloc 是毫秒级的 —— 实测 fp32 GPU 被拖到 126 ms/img，
//      比 CPU 的 11.6 ms/img 还慢 10 倍。精度正确（88%），速度不可用。
//
//   2) CudaUnlockedPoolAllocator / CudaPoolAllocator（benchncnn_gpu 的用法）：
//      速度正常（5.47 ms/img，比裸分配器快 23 倍），但**精度掉到 19%**。
//      batch=1 也一样错，所以不是跨图复用 —— 是块被回收后立刻又发给还活着的
//      blob，也就是别名（aliasing）。这个 fork 的 CudaMat 引用计数和 pool 的
//      budget/payout 配合不上。
//
// 两条路都走不通，就自己写一个：整块预分配 + 单调递增分配 + free 直接丢弃。
// 单调意味着**物理上不可能别名**，同时全程 0 次 cudaMalloc。
// 代价是显存只增不减，所以按「每张图重置到水位线」的方式复用：
//   - mark 之后的位置留给权重/流水线里的常驻对象
//   - 每张图开始 reset 到 mark，一张图的临时激活最多几 MB
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
        size = (size + 255) & ~((size_t)255); // 256 字节对齐
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
        // 单调分配不回收地址，但要知道这块是否还活着；reset 只回退到 mark，
        // 且不能跨过任何仍存活的块。
        live.erase(p);
    }

    void set_mark() { mark = used; }
    void reset()
    {
        for (std::map<void*, size_t>::const_iterator it = live.begin(); it != live.end(); ++it)
        {
            if (static_cast<char*>(it->first) >= static_cast<char*>(base) + mark)
            {
                deferred++;
                return; // 有活块在 mark 之上，这次不 reset，等它释放
            }
        }
        used = mark;
    }
    void reset_all()
    {
        // io arena 只在 batch 边界整段作废；此时上一批输出已 D2H 完成。
        live.clear();
        mark = 0;
        used = 0;
    }
    size_t bytes_used() const { return used; }
    size_t live_blocks() const { return live.size(); }
    size_t deferred_resets() const { return deferred; }

private:
    size_t cap;
    size_t used;
    size_t mark;
    void* base;
    std::map<void*, size_t> live;
    size_t deferred = 0;
};

// 基线：ncnn CPU fp32（04-libtorch-cuda README 实测，1x）
static const double kBaselineMsPerImg = 11.6;

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
                        "[--limit N] [--repeat N] [--warmup N] [--batch N]\n", argv[0]);
        return 2;
    }

    const char* param_path = argv[1];
    const char* bin_path = argv[2];
    const char* data_path = argv[3];

    bool use_int8 = true;
    bool no_reset = false;
    int reset_every = 1;
    int sync_every = 1;
    const char* ab_switch = "pool";
    int ab_images = 256;
    int ab_batch = 32;
    int ab_rounds = 8;
    int ab_warmup = 30;
    int limit = 0;
    int repeat = 3;
    int warmup = 20;
    int batch = 256;
    for (int i = 4; i < argc; i++)
    {
        if (!strcmp(argv[i], "--int8")) use_int8 = true;
        else if (!strcmp(argv[i], "--fp32")) use_int8 = false;
        else if (!strcmp(argv[i], "--limit") && i + 1 < argc) limit = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--repeat") && i + 1 < argc) repeat = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batch") && i + 1 < argc) batch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-reset")) no_reset = true;
        else if (!strcmp(argv[i], "--reset-every") && i + 1 < argc) reset_every = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sync-every") && i + 1 < argc) sync_every = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ab-switch") && i + 1 < argc) ab_switch = argv[++i];
        else if (!strcmp(argv[i], "--ab-images") && i + 1 < argc) ab_images = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ab-batch") && i + 1 < argc) ab_batch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ab-rounds") && i + 1 < argc) ab_rounds = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ab-warmup") && i + 1 < argc) ab_warmup = atoi(argv[++i]);
        else { fprintf(stderr, "未知参数 %s\n", argv[i]); return 2; }
    }
    if (ab_batch > 0) batch = ab_batch;
    if (batch < 1) batch = 1;
    if (reset_every < 1) reset_every = 1;
    if (repeat < 1) repeat = 1;

    // create_gpu_instance() 只存在于 NCNN_VULKAN 分支；CUDA 侧用
    // try_initialize_cuda_gpu_instances() 探测设备（get_cuda_gpu_count 内部也会调）。
    ncnn::try_initialize_cuda_gpu_instances();
    if (ncnn::get_cuda_gpu_count() <= 0)
        fatal("get_cuda_gpu_count() == 0，没有可用 CUDA 设备");

    // ⚠️ 坑 1：Option 的默认构造函数不会初始化 blob_cuda_allocator（option.cpp 里
    //    一行都没有），它是个空 shared_ptr。不显式赋值的话，第一个 create() 就会
    //    在 allocator->fastMalloc 上段错误。fork 自带的 benchncnn.cpp:337 也是这么赋的。
    //
    // ⚠️ 坑 2：不能用 get_current_gpu_allocator() 直接当 blob 分配器。它返回的是
    //    裸 CudaAllocator，fastMalloc 就是一次 cudaMalloc —— 每层每个 blob 都
    //    cudaMalloc 一次，实测把 fp32 GPU 拖到 126 ms/img（比 CPU 的 11.6 还慢
    //    10 倍）。必须用池化分配器，和 benchncnn_gpu.cpp:59-60 一致：
    //      blob      -> CudaUnlockedPoolAllocator
    //      workspace -> CudaPoolAllocator
    //
    // ⚠️ 坑 3：必须在 load_param/load_model **之前** 赋值。load 过程中
    //    layer->create_pipeline(opt) 会用到这些分配器，晚赋值会让权重走空指针
    //    （轻则权重没上传，重则分配器不认这块内存）。
    const ncnn::CudaDevice* dev = ncnn::get_current_gpu_device();
    if (!dev) fatal("get_current_gpu_device() 返回空");

    // blob/workspace 都走 arena；每张图 reset 一次。
    // 输入输出缓冲另开一块 arena，按 batch 重置（它们要跨图存活）。
    static BumpCudaAllocator g_blob_arena(dev, 1280u * 1024 * 1024);
    static BumpCudaAllocator g_work_arena(dev, 64u * 1024 * 1024);
    static BumpCudaAllocator g_io_arena(dev, 128u * 1024 * 1024);
    // 权重和激活分开：权重常驻、不参与每图 reset；激活/workspace 才可回退。
    // 同一块 arena 同时放权重和激活时，reset 会把仍被层持有的权重/惰性常驻块
    // 一起冲掉 —— 这是 reset-every 曲线在 mat.h 之外的另一半根因。
    static BumpCudaAllocator g_weight_arena(dev, 128u * 1024 * 1024);
    static BumpCudaAllocator g_weight_work_arena(dev, 64u * 1024 * 1024);

    std::shared_ptr<ncnn::CudaAllocator> alloc(
        &g_blob_arena, [](ncnn::CudaAllocator*) {}); // 静态对象，不析构
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
    net.opt.lightmode = true;
    // load_param/load_model 期间创建的权重、层 pipeline 常驻缓冲走 weight arena；
    // load 完成后再把 forward 用的 blob/workspace 切回可复位 arena。
    net.opt.blob_cuda_allocator = weight_alloc;
    net.opt.workspace_cuda_allocator = weight_walloc;

    if (net.load_param(param_path) != 0) fatal("load_param 失败");
    if (net.load_model(bin_path) != 0) fatal("load_model 失败");
    net.opt.blob_cuda_allocator = alloc;
    net.opt.workspace_cuda_allocator = walloc;

    cifar10::Batch batch_data;
    if (!batch_data.load(data_path)) fatal("读数据失败");

    const int n = (limit > 0) ? std::min(limit, batch_data.size()) : batch_data.size();
    if (n < 1) fatal("样本数为 0");

    const int in_idx = net.find_blob_index_by_name("input");
    const int out_idx = net.find_blob_index_by_name("logits");
    if (in_idx < 0 || out_idx < 0) fatal("找不到 input / logits blob");
    const int out_producer = net.blobs[out_idx].producer;

    printf("模型 %s\n后端 CUDA | 精度 %s | 样本 %d | batch %d\n",
           param_path, use_int8 ? "int8" : "fp32", n, batch);

    std::vector<ncnn::CudaMat> blob_mats_gpu(net.blobs.size());
    std::vector<ncnn::CudaMat> cu_in(batch);
    std::vector<ncnn::CudaMat> cu_out(batch);

    // 清空所有 blob 引用，并把 arena 回退到水位线。
    // 顺序很重要：先 release（各自减引用），再 reset（整段作废）。
    // 每 reset_every 张图才回退一次 arena。
    //   =1  每张都回退（最省显存，但这个 fork 下精度会崩）
    //   >1  连续 reset_every 张图各自拿新地址，第 reset_every 张之后才开始复用
    //   --no-reset 等价于无穷大
    int img_seq = 0;
    auto clear_blobs = [&]() {
        // 顺序：先等 GPU 完成 → 回退 arena → 再 release host 句柄。
        // 先 release 会让某些 CudaMat 变成空对象，下一轮 create 重新分配时
        // 可能拿到一块仍被上层缓存的旧地址，造成别名；先 reset 再 release
        // 可保证回退发生在所有消费者结束后。
        if (!no_reset)
        {
            if (sync_every <= 1)
            {
                cudaDeviceSynchronize();
                if (img_seq % reset_every == 0)
                {
                    // h015 语义：reset 在有活块时自动 defer。blob 每张图都有上一张的
                    // blob 句柄未释放，因此实际不回退（used 单调增长 = 无地址复用）；
                    // workspace 的临时块在 forward 内已释放，回退成功（used 保持 ~4MB）。
                    g_blob_arena.reset();
                    g_work_arena.reset();
                }
            }
            else if (img_seq % sync_every == 0)
            {
                // 延迟同步（opt-in）：每 N 张才等一次 GPU；blob arena 保持单调
                // （绝不地址复用），同步后只回收已无活块的 workspace。
                cudaDeviceSynchronize();
                g_work_arena.reset();
            }
        }
        for (size_t i = 0; i < blob_mats_gpu.size(); i++) blob_mats_gpu[i].release();
        img_seq++;
    };

    auto run_one = [&](int idx) {
        cifar10::Sample s = batch_data.at(idx);
        ncnn::Mat in(cifar10::kWidth, cifar10::kHeight, 3);
        fill_input(in, s.pixels);
        blob_mats_gpu[out_idx].release();
        blob_mats_gpu[in_idx] = ncnn::CudaMat(in, alloc);
        return net.forward_layer(out_producer, blob_mats_gpu, net.opt);
    };

    // ---------- 交错 A/B：同一进程、单 Net、运行时开关 ----------
    const std::string absw = ab_switch;
    if (absw != "pool" && absw != "binop" && absw != "relu")
        fatal("--ab-switch 必须是 pool|binop|relu");
    if (ab_images < 1 || ab_batch < 1 || ab_rounds < 1 || ab_warmup < 1)
        fatal("A/B 参数非法");

    // 测量口径：延迟同步到每 8 张，减少串行化噪声（T-12 已验证逐位一致）。
    sync_every = 8;

    // 非目标开关固定在基线：pool 默认开、binop 默认关、ReLU 图不融合。
    setenv("NCNN_CUDA_POOL_FUSE_PAD", "1", 1);
    setenv("NCNN_CUDA_BINARYOP_VEC", "0", 1);

    struct FusePair
    {
        int conv_idx;
        int relu_idx;
        int conv_top;
        int relu_top;
        std::vector<int> relu_consumers;
    };
    std::vector<FusePair> fuse_pairs;

    if (absw == "relu")
    {
        for (size_t i = 0; i < net.layers.size(); i++)
        {
            ncnn::Layer* layer = net.layers[i];
            if (layer->type != "Convolution") continue;
            ncnn::Convolution* conv = static_cast<ncnn::Convolution*>(layer);
            if (conv->activation_type != 0) continue;
            if (layer->tops.size() != 1) continue;
            const int top = layer->tops[0];
            if (net.blobs[top].consumers.size() != 1) continue;
            const int ri = net.blobs[top].consumers[0];
            ncnn::Layer* rl = net.layers[ri];
            if (rl->type != "ReLU") continue;
            if (static_cast<ncnn::ReLU*>(rl)->slope != 0.0f) continue;
            if (rl->bottoms.size() != 1 || rl->tops.size() != 1) continue;
            if (rl->bottoms[0] != top) continue;
            const int rt = rl->tops[0];
            FusePair p;
            p.conv_idx = (int)i;
            p.relu_idx = ri;
            p.conv_top = top;
            p.relu_top = rt;
            p.relu_consumers = net.blobs[rt].consumers;
            fuse_pairs.push_back(p);
        }
        if (fuse_pairs.empty()) fatal("没有可融合的 Conv+ReLU 对");
        printf("relu fuse pairs: %zu\n", fuse_pairs.size());
    }

    auto set_relu_fused = [&](bool on) {
        for (size_t k = 0; k < fuse_pairs.size(); k++)
        {
            const FusePair& p = fuse_pairs[k];
            ncnn::Convolution* conv = static_cast<ncnn::Convolution*>(net.layers[p.conv_idx]);
            if (on)
            {
                conv->activation_type = 1;
                for (size_t c = 0; c < p.relu_consumers.size(); c++)
                {
                    ncnn::Layer* next = net.layers[p.relu_consumers[c]];
                    for (size_t b = 0; b < next->bottoms.size(); b++)
                        if (next->bottoms[b] == p.relu_top) next->bottoms[b] = p.conv_top;
                }
                net.blobs[p.conv_top].consumers = p.relu_consumers;
                net.blobs[p.relu_top].consumers.clear();
            }
            else
            {
                conv->activation_type = 0;
                for (size_t c = 0; c < p.relu_consumers.size(); c++)
                {
                    ncnn::Layer* next = net.layers[p.relu_consumers[c]];
                    for (size_t b = 0; b < next->bottoms.size(); b++)
                        if (next->bottoms[b] == p.conv_top) next->bottoms[b] = p.relu_top;
                }
                net.blobs[p.conv_top].consumers.clear();
                net.blobs[p.conv_top].consumers.push_back(p.relu_idx);
                net.blobs[p.relu_top].consumers = p.relu_consumers;
            }
        }
    };

    auto set_config = [&](bool cand) {
        if (absw == "relu")
        {
            cudaDeviceSynchronize();
            set_relu_fused(cand);
        }
        else if (absw == "pool")
        {
            setenv("NCNN_CUDA_POOL_FUSE_PAD", cand ? "1" : "0", 1);
        }
        else
        {
            setenv("NCNN_CUDA_BINARYOP_VEC", cand ? "1" : "0", 1);
        }
    };

    auto median = [](std::vector<double> v) {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };

    set_config(false);
    cudaDeviceSynchronize();

    // 预热 30 张，两种配置交替。
    const int w = std::min(ab_warmup, batch_data.size());
    for (int i = 0; i < w; i++)
    {
        set_config((i & 1) != 0);
        if (run_one(i % batch_data.size()) != 0) fatal("warmup forward 失败");
        cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();
    g_blob_arena.set_mark();
    g_work_arena.set_mark();
    clear_blobs();

    printf("A/B switch=%s images=%d batch=%d rounds=%d warmup=%d sync_every=%d\n",
           ab_switch, ab_images, ab_batch, ab_rounds, ab_warmup, sync_every);

    const int start = ab_warmup;
    if (start + ab_images > batch_data.size()) fatal("--ab-images 超出数据范围");
    const int ab_check = std::min(128, ab_images);

    std::vector<std::vector<float> > logits_base(ab_check, std::vector<float>(10, 0.f));
    std::vector<std::vector<float> > logits_cand(ab_check, std::vector<float>(10, 0.f));
    std::vector<char> has_base(ab_check, 0), has_cand(ab_check, 0);
    std::vector<double> round_base(ab_rounds), round_cand(ab_rounds), round_impr(ab_rounds);
    std::vector<double> all_base, all_cand;
    long top1_base = 0, top1_cand = 0, n_base = 0, n_cand = 0;

    for (int r = 0; r < ab_rounds; r++)
    {
        std::vector<double> bms, cms;
        for (int b = 0; b * ab_batch < ab_images; b++)
        {
            const int base = start + b * ab_batch;
            const int cnt = std::min(ab_batch, ab_images - b * ab_batch);
            const bool cand = (((b + r) & 1) != 0);
            set_config(cand);
            cudaDeviceSynchronize();

            // A/B: io arena 保持单调（128MB 足够 2048 次前向），避免跨轮复用地址
            // 触发 CudaMat 句柄/refcount 竞态。
            for (int j = 0; j < cnt; j++)
            {
                cifar10::Sample s = batch_data.at(base + j);
                ncnn::Mat in(cifar10::kWidth, cifar10::kHeight, 3);
                fill_input(in, s.pixels);
                cu_in[j] = ncnn::CudaMat(in, ioalloc);
            }
            cudaDeviceSynchronize();

            cudaEvent_t ev0, ev1;
            cudaEventCreate(&ev0);
            cudaEventCreate(&ev1);
            cudaEventRecord(ev0);
            if (net.forward_batch(blob_mats_gpu, in_idx, out_idx, out_producer,
                                  cu_in, cnt, cu_out, ioalloc, clear_blobs) != 0)
                fatal("forward_batch 失败");
            cudaEventRecord(ev1);
            cudaEventSynchronize(ev1);
            float ms = 0.f;
            cudaEventElapsedTime(&ms, ev0, ev1);
            const double per = ms / cnt;
            (cand ? cms : bms).push_back(per);
            (cand ? all_cand : all_base).push_back(per);
            cudaEventDestroy(ev0);
            cudaEventDestroy(ev1);

            for (int j = 0; j < cnt; j++)
            {
                ncnn::Mat out(cu_out[j]);
                const float* p = out;
                int best = 0;
                for (int k = 1; k < cifar10::kNumClasses; k++)
                    if (p[k] > p[best]) best = k;
                const int label = (int)batch_data.at(base + j).label;
                if (cand) { n_cand++; if (best == label) top1_cand++; }
                else { n_base++; if (best == label) top1_base++; }

                const int ci = base + j - start;
                if (ci >= 0 && ci < ab_check)
                {
                    if (cand && !has_cand[ci])
                    {
                        for (int k = 0; k < 10; k++) logits_cand[ci][k] = p[k];
                        has_cand[ci] = 1;
                    }
                    if (!cand && !has_base[ci])
                    {
                        for (int k = 0; k < 10; k++) logits_base[ci][k] = p[k];
                        has_base[ci] = 1;
                    }
                }
            }
            cudaDeviceSynchronize();
        }

        const double bmed = median(bms);
        const double cmed = median(cms);
        round_base[r] = bmed;
        round_cand[r] = cmed;
        round_impr[r] = bmed > 0.0 ? (bmed - cmed) / bmed * 100.0 : 0.0;
        printf("round %d: base=%.4f cand=%.4f impr=%+.2f%%\n", r + 1, bmed, cmed, round_impr[r]);
    }

    const double base_med = median(all_base);
    const double cand_med = median(all_cand);
    const double base_min = *std::min_element(all_base.begin(), all_base.end());
    const double cand_min = *std::min_element(all_cand.begin(), all_cand.end());
    const double med_impr = median(round_impr);

    std::mt19937 rng(12345);
    std::uniform_int_distribution<size_t> dist(0, round_impr.size() - 1);
    std::vector<double> boot;
    boot.reserve(10000);
    for (int it = 0; it < 10000; it++)
    {
        std::vector<double> s;
        s.reserve(round_impr.size());
        for (size_t k = 0; k < round_impr.size(); k++) s.push_back(round_impr[dist(rng)]);
        boot.push_back(median(s));
    }
    std::sort(boot.begin(), boot.end());
    const double ci_lo = boot[(size_t)(0.025 * boot.size())];
    const double ci_hi = boot[(size_t)(0.975 * boot.size())];

    int cmp_n = 0, diff_bits = 0;
    float max_abs = 0.f;
    for (int i = 0; i < ab_check; i++)
    {
        if (!has_base[i] || !has_cand[i]) continue;
        cmp_n++;
        for (int k = 0; k < 10; k++)
        {
            if (std::memcmp(&logits_base[i][k], &logits_cand[i][k], sizeof(float)) != 0) diff_bits++;
            const float d = fabsf(logits_base[i][k] - logits_cand[i][k]);
            if (d > max_abs) max_abs = d;
        }
    }

    printf("SUMMARY switch=%s base_med=%.4f base_min=%.4f cand_med=%.4f cand_min=%.4f "
           "med_impr=%.2f%% ci=[%.2f,%.2f] top1_base=%ld/%ld top1_cand=%ld/%ld "
           "logits_n=%d max_abs=%.9g differing=%d\n",
           ab_switch, base_med, base_min, cand_med, cand_min, med_impr, ci_lo, ci_hi,
           top1_base, n_base, top1_cand, n_cand, cmp_n, (double)max_abs, diff_bits);

    const bool recommend = (ci_lo > 0.0 && med_impr >= 2.0);
    printf("DECISION switch=%s %s (ci_lo=%.2f med_impr=%.2f%% rule=ci_lo>0 && med_impr>=2%%)\n",
           ab_switch, recommend ? "default_on" : "opt_in_off", ci_lo, med_impr);
    return 0;
}
