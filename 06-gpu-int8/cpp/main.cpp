// GPU int8 引擎的评测入口。
static int run(int argc, char** argv);
#include <stdexcept>
//
// 用法：
//   ./gpu_int8 <model.bin> <calib.bin> <cifar_batch> [--limit N] [--batch N]
//             [--w-bits N] [--a-bits N] [--act-minmax]
//
// 计时口径（必须写清楚，否则倍数会被误读）：
//   报两个数：
//     前向耗时  —— 纯 GPU 前向（不含数据准备）
//     总耗时    —— 含数据归一化 + 上传 + 整批 argmax
//   倍数基准是 lab/01 记的 ncnn CPU fp32 11.6 ms/img（同精度口径）。
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "cifar10.h"
#include "int8_engine.cuh"

static const float kMean[3] = {0.4914f, 0.4822f, 0.4465f};
static const float kStd[3] = {0.2470f, 0.2435f, 0.2616f};

int main(int argc, char** argv)
{
    // libtorch 会抛 c10::Error；不接住的话它打印裸栈（看不出原因），
    // 看起来像段错误，其实是维度/设备不合法。
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        fprintf(stderr, "运行失败: %s\n", e.what());
        return 1;
    }
}

static int run(int argc, char** argv)
{
    if (argc < 4)
    {
        fprintf(stderr, "usage: %s <model.bin> <calib.bin> <cifar_batch> "
                        "[--limit N] [--batch N] [--w-bits N] [--a-bits N] "
                        "[--act-minmax]\n", argv[0]);
        return 2;
    }

    Model model;
    if (!model.load(argv[1])) return 1;

    // PTQC 严格校验：8+8n 且 n == 模型 conv 数；坏文件在这里返回错误，
    // 不会进 set_calib 更不会越界/exit。
    std::vector<std::pair<float, float>> ranges;
    {
        std::string cerr;
        if (!load_calib(argv[2], model.convs.size(), &ranges, &cerr))
        {
            fprintf(stderr, "[FAIL] 校准文件不可用：%s\n", cerr.c_str());
            return 1;
        }
    }

    // 默认 512：这是这台 4GB 卡（GTX 1650 Ti）的实测最优点。
    // 250→0.119 ms/img，500→0.057，1000 开始显存吃紧，2000 直接 OOM。
    int limit = 0, batch_size = 512, w_bits = 8, a_bits = 8;
    bool act_minmax = false;           // 默认 AbsMax（与 09 校准一致）
    const char* dump_path = nullptr;   // --dump <path>：导出 logits 用于端到端比对
    for (int i = 4; i < argc; i++)
    {
        if (!strcmp(argv[i], "--limit") && i + 1 < argc) limit = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batch") && i + 1 < argc) batch_size = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--w-bits") && i + 1 < argc) w_bits = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dump")) dump_path = argv[++i];
        else if (!strcmp(argv[i], "--a-bits") && i + 1 < argc) a_bits = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--act-minmax")) act_minmax = true;
        else { fprintf(stderr, "未知参数 %s\n", argv[i]); return 2; }
    }

    // ⭐ batch 硬上限 —— 在加载数据/建引擎**之前**就拦下。
    // 为什么必须前置：等到 forward 里cudaMalloc 失败，
    // 前面已经把模型权重、校准、测试数据都读进内存了，
    // 白跑一轮。实测 batch=2000 时 im2col 单层就1.2 GB，
    // 加上 cuBLAS workspace 直接撞 4 GB 物理上限。
    //
    // 上限 1024：实测 1024 还能跑但显存接近 80%，且速度反而不如 512。
    // 超过就直接拒绝并给出建议值，不做「自动降级」——
    // 自动降级会让用户以为跑的是他指定的 batch，测出来的口径就不对了。
    {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const size_t used = total_b - free_b;
        const size_t headroom = total_b / 10;
        fprintf(stderr, "GPU 显存：总 %zu MB，已用 %zu MB，可用 %zu MB\n",
                total_b / 1048576, used / 1048576, free_b / 1048576);

        // ⭐ 上限随路径变化：
        //   全线 implicit（USE_IMPLICIT_GEMM=1 或 LAYER_IMPL=all:implicit）不落
        //   im2col、不落 int32 acc，每张图中间量约 190KB，batch=2048 约 400MB。
        //   默认的**按层混合**仍有 16 层要落 im2col + int32 acc，所以沿用旧口径
        //   512（这也是实测最优点）。上限由 layer_impl_batch_cap() 统一给出，
        //   与引擎读同一套 env/层选择表，避免「入口说能跑、引擎里 OOM」。
        const int batch_cap = ptqint8::layer_impl_batch_cap();
        const bool implicit_path = (batch_cap >= 2048);
        if (batch_size > batch_cap)
        {
            fprintf(stderr,
                    "[FAIL] --batch %d 超过安全上限 %d\n"
                    "       本机总显存 %zu MB，已被其他进程占用 %zu MB。\n"
                    "       全线 implicit 上限 2048（每图中间量约 190KB），\n"
                    "       按层混合 / 旧 im2col 路径上限 512（仍有层要落 im2col + int32 acc）。\n",
                    batch_size, batch_cap, total_b / 1048576, used / 1048576);
            return 2;
        }
        // 按「最坏的一层」估算：layer1 的 im2col 是 batch × 1024 × 576 int8
        //                            acc 是 batch × 1024 × 64 × 4 int32
        // 旧路径：im2col (batch×1024×576) + acc (batch×1024×64×4) 都要落显存。
        // implicit：只有激活和输出，每图约 190KB（实测池峰值），按 256KB 估。
        const size_t est = implicit_path
            ? (size_t)batch_size * (256u << 10)
            : (size_t)batch_size * (1024 * 576 + 1024 * 64 * 4 + 64 * 1024);
        if (free_b < est + headroom)
        {
            // ⚠️ 这里要区分两种「显存不够」：
            //   (a) batch 太大 → 提示降batch
            //   (b) batch 合理但**别的进程占了显存** → 提示先释放显存，不要误导成降 batch
            //     实测踩过：WSL 侧的 CUDA context 没回收（vmmemWSL 占 2.4 GB），
            //     nvidia-smi 里却「No running processes found」——
            //     误报成「batch 太大」会让人一路降 batch，最后降到 1 还是跑不了。
            const size_t est512 = (size_t)512 * (1024 * 576 + 1024 * 64 * 4 + 64 * 1024);
            if (batch_size <= 512 && free_b < est512 + headroom)
            {
                fprintf(stderr,
                        "[FAIL] batch=%d 本身合理（需约 %zu MB），但**显存被其他进程占用**。\n"
                        "       当前只剩 %zu MB可用（已用 %zu MB / 总 %zu MB）。\n"
                        "       处理方式（不要降 batch，降了也没用）：\n"
                        "         · Windows 侧：关闭占用 GPU 的程序后重试\n"
                        "         · WSL 侧：wsl --shutdown 后重开（CUDA context 不会自动回收）\n"
                        "         · 若只想确认引擎本身没问题，可先用 --limit 250 --batch 250\n",
                        batch_size, est / 1048576, free_b / 1048576,
                        used / 1048576, total_b / 1048576);
            }
            else
            {
                fprintf(stderr,
                        "[FAIL] batch=%d 预计需要约 %zu MB，但只剩 %zu MB 可用\n"
                        "       请降到 --batch 512 或更小。\n",
                        batch_size, est / 1048576, free_b / 1048576);
            }
            return 2;
        }
    }



    Int8Engine engine;
    if (!engine.build(model, w_bits, a_bits)) return 1;
    if (act_minmax) engine.set_act_mode(ptq::ActRangeMode::MinMax);
    {
        std::string cerr;
        if (!engine.set_calib(ranges, &cerr))
        {
            fprintf(stderr, "[FAIL] set_calib: %s\n", cerr.c_str());
            return 1;
        }
    }
    printf("设备：CUDA | W%dA%d | conv %d 层 | 激活 scale=%s\n",
           w_bits, a_bits, (int)model.convs.size(),
           act_minmax ? "(hi-lo)/127（对照）" : "absmax/127");
    engine.print_calib();

    cifar10::Batch batch;
    if (!batch.load(argv[3])) { fprintf(stderr, "读数据失败\n"); return 1; }
    const int n = (limit > 0) ? std::min(limit, batch.size()) : batch.size();
    engine.alloc_scratch(batch_size);

    const int plane = 32 * 32;
    // ⭐ 输入上传缓冲必须是 pinned（cudaMallocHost）：
    //   同一份 512×3×32×32（6.29 MB）的 cudaMemcpy 实测（probe_overhead）：
    //     pageable 2.44 ms（2.6 GB/s） / pinned 1.05 ms（6.0 GB/s）。
    //   每次 forward 都要上传一次，pin 下来省 1.39 ms/batch（约 0.0027 ms/img）。
    //   host 侧照常写（归一化），语义与数值完全不变。
    float* h_in = nullptr;
    size_t h_in_cap = 0;                       // 已分配的 float 数
    auto ensure_in = [&](size_t floats) {
        if (h_in_cap >= floats) return;
        if (h_in) cudaFreeHost(h_in);
        h_in = nullptr;
        if (cudaMallocHost(reinterpret_cast<void**>(&h_in), floats * sizeof(float)) != cudaSuccess)
        {
            fprintf(stderr, "[FAIL] cudaMallocHost %zu floats 失败\n", floats);
            std::exit(1);
        }
        h_in_cap = floats;
    };
    std::vector<uint8_t> raw;
    raw.reserve((size_t)batch_size * cifar10::kRecordSize);

    std::vector<float> logits((size_t)batch_size * model.fc_out);
    // --dump：累积全部 logits，评测结束后一次性写出。
    // 为什么不在循环里直接写：每 batch 一次 fwrite 会把「计算」和「I/O」混在
    // 一起计时里，破坏速度数字。这里累积到内存，计时之后再落盘。
    std::vector<float> all_logits;
    if (dump_path) all_logits.reserve((size_t)n * model.fc_out);

    // 预热：CUDA context 建立 + cublas/libtorch 首次初始化，代价是几十毫秒，
// 不预热的话单 batch 的测量会被它完全淹没（实测 512 张时 0.61 ms/img
// vs 1000 张时 0.061 ms/img，差 10 倍，纯粹是首次初始化）。
    // 要跑**多个 batch** 才能测到稳态 —— 单 batch 的一次性测量不可信。
    const int warm_n = 32;      // ⚠️ at::_int_mm 要求 M > 16
    {
        std::vector<float> warm((size_t)warm_n * 3 * 32 * 32, 0.5f);
        std::vector<float> wl(warm_n * model.fc_out);
        engine.forward(warm.data(), warm_n, 3, 32, 32, wl.data());
        cudaDeviceSynchronize();
    }
    {
        // 再用真实 batch 尺寸预热一遍：pool 的大小与 batch 有关，
        // 不同尺寸的第一次调用仍会触发新的 cudaMalloc。
        const int wb = std::min(batch_size, n);
        std::vector<float> warm2((size_t)wb * 3 * 32 * 32, 0.25f);
        std::vector<float> wl2((size_t)wb * model.fc_out);
        engine.forward(warm2.data(), wb, 3, 32, 32, wl2.data());
        cudaDeviceSynchronize();
    }

    int correct = 0;
    double fwd_ms = 0.0, prep_ms = 0.0;
    int processed = 0;

    while (processed < n)
    {
        const int bs = std::min(batch_size, n - processed);

        // ⚠️ 第一个 batch 要跳过计时：显存池刚建立（pool_reset + 一堆
        //   cudaMalloc），比稳态慢一个数量级。
        //   实测：W8A8 首 batch 0.164 ms/img vs 后续 0.069 ms/img。
        //   不跳过就会把一次性开销算进「平均速度」，倍数虚低 2.4 倍。
        const bool warmup_batch = (processed == 0);

        const auto tp = std::chrono::steady_clock::now();
        // 归一化在 CPU（实测搬到 GPU 反而更慢：小张量上 kernel 启动开销 > 计算）
        ensure_in((size_t)bs * 3 * plane);
        for (int b = 0; b < bs; b++)
        {
            const uint8_t* px = batch.at(processed + b).pixels - 1; // 跳过 label
            float* dst = h_in + (size_t)b * 3 * plane;
            for (int i = 0; i < 3 * plane; i++)
                dst[i] = (float)px[i] / 255.0f;
        }
        for (int b = 0; b < bs; b++)
        {
            float* dst = h_in + (size_t)b * 3 * plane;
            for (int c = 0; c < 3; c++)
            {
                const float m = kMean[c] * 255.0f;
                const float inv = 1.0f / (kStd[c] * 255.0f);
                float* p = dst + (size_t)c * plane;
                for (int i = 0; i < plane; i++) p[i] = (p[i] * 255.0f - m) * inv;
            }
        }

        const auto t0 = std::chrono::steady_clock::now();
        engine.forward(h_in, bs, 3, 32, 32, logits.data());
        cudaDeviceSynchronize();
        const auto t1 = std::chrono::steady_clock::now();

        if (!warmup_batch)
        {
            fwd_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
            prep_ms += std::chrono::duration<double, std::milli>(t0 - tp).count();
        }

        if (dump_path)
            all_logits.insert(all_logits.end(), logits.begin(),
                              logits.begin() + (long)bs * model.fc_out);

        for (int b = 0; b < bs; b++)
        {
            const float* lg = logits.data() + (size_t)b * model.fc_out;
            int best = 0;
            for (int j = 1; j < model.fc_out; j++) if (lg[j] > lg[best]) best = j;
            if (best == (int)batch.at(processed + b).label) correct++;
        }
        processed += bs;
    }

    // ---------------- 稳态基准 ----------------
    //
    // ⚠️ 为什么「边评测边计时」不可信：
    //   每个 batch 的耗时差异很大（显存池重建、数据页首次触碰、
    //   GPU 时钟爬坡）。实测 10000 张跑 5 个 batch，各 batch 之间差 6 倍。
    //   平均值把「一次性开销」和「稳态速度」混在一起，报出的倍数没有意义。
    //
    //   正确口径：**固定一批输入，重复跑 K 次，取最快那一次**。
    //   理由：稳态速度的上限就是重复执行能达到的最快值 ——
    //   任何更慢的测量都包含了与吞吐无关的一次性因素。
    //   这也是 GPU benchmark 的通行做法（如 PyTorch 的 benchmark=True）。
    double steady_ms_img = 0.0;
    // 条件放宽：只要有 ≥32 张就能测稳态。
    // 原来要求 n > batch_size，导致 n == batch_size（如 --limit 512 --batch 512）
    // 时稳态段被跳过，输出「稳态 0.0000ms/img 加速 inf×」——看起来像跑通了，其实没测。
    if (n >= 32)
    {
        const int bs = std::min(batch_size, n);
        ensure_in((size_t)bs * 3 * plane);
        for (int b = 0; b < bs; b++)
        {
            const uint8_t* px = batch.at(b).pixels - 1;
            float* dst = h_in + (size_t)b * 3 * plane;
            for (int i = 0; i < 3 * plane; i++) dst[i] = (float)px[i] / 255.0f;
            for (int c = 0; c < 3; c++)
            {
                const float m = kMean[c] * 255.0f;
                const float inv = 1.0f / (kStd[c] * 255.0f);
                float* p = dst + (size_t)c * plane;
                for (int i = 0; i < plane; i++) p[i] = (p[i] * 255.0f - m) * inv;
            }
        }

        // ⭐ reps 从 20 提到 60：这台卡空闲时是 1350 MHz，持续负载才爬到
        // 1873–1920 MHz（ROOFLINE-int8.md 实测）。20 次 ≈ 260 ms 的负载
        // 不足以完成爬升，同一个二进制会给出 0.0374（冷）和 0.0261（热）
        // 两个数 —— 差 1.4 倍，纯测量口径。
        // 口径固定为「重复 60 次取最快」（约 0.8 s 负载，已过爬升段），
        // 这样任何前后对比都在同一频率上。
        // ⭐ 先「预热到频率」再测：空闲时 1350 MHz，持续负载才爬到 1873–1920 MHz。
        // 不加这一步，同一个二进制会给出 0.0387（冷）和 0.0261（热）两个数，
        // 比值 1.43 正好是频率比 —— 纯测量口径，不是代码差异。
        // ⚠️ 只预热 25 次（≈300 ms）：实测 100 次会把整卡顶到 50 W 功耗墙
        //    （nvidia-smi 采样：max power 50.22 W / 75.5 °C），之后所有版本
        //    （im2col tiled / fast / implicit）都被压到 0.041-0.042 ms/img，
        //    连 A/B 的差异都看不出来。预热是为了爬频，不是为了把卡烤热。
        {
            std::vector<float> wl((size_t)bs * model.fc_out);
            for (int r = 0; r < 25; r++)
                engine.forward(h_in, bs, 3, 32, 32, wl.data());
            cudaDeviceSynchronize();
        }
        double best = 1e30;
        const int reps = 60;
        for (int r = 0; r < reps; r++)
        {
            const auto t0 = std::chrono::steady_clock::now();
            engine.forward(h_in, bs, 3, 32, 32, logits.data());
            cudaDeviceSynchronize();
            const auto t1 = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (ms < best) best = ms;
        }
        steady_ms_img = best / bs;
    }

    const double ms_img = steady_ms_img > 0 ? steady_ms_img : fwd_ms / std::max(1, n);
    printf("----------------------------------------\n");
    fflush(stdout);   // 被 timeout 杀时缓冲会丢，评测结果必须先落盘
    printf("精度 %.2f%%（%d/%d）\n", 100.0 * correct / n, correct, n);
    printf("全量前向 %.3f s（含数据准备 %.3f s，含一次性开销）\n",
           (fwd_ms + prep_ms) / 1000.0, prep_ms / 1000.0);
    printf("稳态 %.4f ms/img（batch=%d，预热后重复 60 次取最快）\n", ms_img, batch_size);
    printf("加速 %.0f×（vs ncnn CPU fp32 11.6 ms/img）\n", 11.6 / ms_img);
    engine.print_prof();

    if (dump_path)
    {
        // 二进制格式：magic "PTQL" + n_samples + fc_out，然后 n×fc_out 个 float32。
        // 用二进制而不是文本：10000×10 个 float 文本约 1 MB，二进制 400 KB，
        // 且比对脚本读起来快一个量级。
        FILE* f = fopen(dump_path, "wb");
        if (!f) { fprintf(stderr, "写 %s 失败\n", dump_path); return 1; }
        int32_t hdr[2] = {(int32_t)n, (int32_t)model.fc_out};
        fwrite("PTQL", 1, 4, f);
        fwrite(hdr, sizeof(int32_t), 2, f);
        fwrite(all_logits.data(), sizeof(float), all_logits.size(), f);
        fclose(f);
        printf("已导出 logits → %s（%d × %d）\n", dump_path, n, model.fc_out);
    }
    if (h_in) cudaFreeHost(h_in);
    return 0;
}
