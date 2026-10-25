// ncnn CIFAR-10 精度/延迟评测入口。
//
// 这个程序要回答的问题只有一个：
//   「这个 ncnn 模型，在 CIFAR-10 测试集上，精度是多少、每张多快？」
//
// 它刻意不做的事：
//   - 不读 npy / 不做 Python 侧预处理（那是 dump_calib_npy.py 的事）
//   - 不打印 logits（要调试用 --verbose 看前几张就够了，刷屏没意义）
//   - 不猜模型结构（层数、blob 名全部从 param 里来）
//
// 用法：
//   ./cifar10_eval --param m.param --bin m.bin --data test_batch [--threads 12]
//                  [--limit 1000] [--warmup 5] [--input input] [--output logits]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#include "cifar10.h"
#include "evaluator.h"

namespace {

void usage(const char* argv0) {
    std::fprintf(stderr,
                 "用法：%s --param <p> --bin <b> --data <CIFAR-10 批次> [选项]\n"
                 "  --param      ncnn .param 文件\n"
                 "  --bin        ncnn .bin 文件\n"
                 "  --data       CIFAR-10 二进制批次（test_batch / data_batch_*）\n"
                 "  --threads    ncnn 线程数，默认 0 = 用硬件并发数\n"
                 "  --limit      只跑前 N 张，默认 0 = 全部\n"
                 "  --warmup     计时前空跑几张，默认 5\n"
                 "  --input      输入 blob 名，默认自动探测\n"
                 "  --output     输出 blob 名，默认自动探测\n"
                 "  --tag        报告里显示的名字，默认取 param 文件名\n",
                 argv0);
}

}  // namespace

int main(int argc, char** argv) {
    std::string param, bin, data, tag, in_name, out_name;
    int threads = 0;
    int limit = 0;
    int warmup = 5;

    for (int i = 1; i < argc; i++) {
        const char* a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "参数 %s 缺值\n", a);
                std::exit(2);
            }
            return argv[++i];
        };
        if (std::strcmp(a, "--param") == 0) param = next();
        else if (std::strcmp(a, "--bin") == 0) bin = next();
        else if (std::strcmp(a, "--data") == 0) data = next();
        else if (std::strcmp(a, "--threads") == 0) threads = std::atoi(next());
        else if (std::strcmp(a, "--limit") == 0) limit = std::atoi(next());
        else if (std::strcmp(a, "--warmup") == 0) warmup = std::atoi(next());
        else if (std::strcmp(a, "--input") == 0) in_name = next();
        else if (std::strcmp(a, "--output") == 0) out_name = next();
        else if (std::strcmp(a, "--tag") == 0) tag = next();
        else if (std::strcmp(a, "--help") == 0 || std::strcmp(a, "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "不认识的参数：%s\n", a);
            usage(argv[0]);
            return 2;
        }
    }

    if (param.empty() || bin.empty() || data.empty()) {
        std::fprintf(stderr, "--param / --bin / --data 三个都必须给\n");
        usage(argv[0]);
        return 2;
    }
    if (tag.empty()) {
        // 取 param 文件名去扩展名，报告里好认
        size_t slash = param.find_last_of("/\\");
        size_t dot = param.find_last_of('.');
        size_t start = (slash == std::string::npos) ? 0 : slash + 1;
        tag = param.substr(start, dot == std::string::npos ? std::string::npos
                                                           : dot - start);
    }

    if (threads <= 0) {
        threads = static_cast<int>(std::thread::hardware_concurrency());
        if (threads <= 0) threads = 1;
    }

    cifar10::Batch batch;
    if (!batch.load(data)) return 1;

    eval::Evaluator ev;
    if (!ev.load(param, bin, threads)) return 1;
    if (!in_name.empty() || !out_name.empty()) ev.set_blob_names(
        in_name.empty() ? ev.input_name() : in_name,
        out_name.empty() ? ev.output_name() : out_name);

    std::printf("模型     : %s（%d 层，输入 blob=%s，输出 blob=%s）\n",
                tag.c_str(), ev.layer_count(), ev.input_name().c_str(),
                ev.output_name().c_str());
    std::printf("数据集   : %s（%d 张）\n", batch.path().c_str(), batch.size());
    std::printf("线程     : %d，warmup %d，limit %d\n", threads, warmup, limit);
    std::fflush(stdout);

    const eval::Result r = ev.run(batch, limit, warmup);
    if (r.total == 0) return 1;

    std::printf("----------------------------------------\n");
    std::printf("正确     : %d / %d\n", r.correct, r.total);
    std::printf("精度     : %.2f%%  (%.4f)\n", r.accuracy * 100.0, r.accuracy);
    std::printf("耗时     : %.1f ms 总，%.3f ms/张，%.1f 张/秒\n",
                r.total_ms, r.ms_per_image, r.img_per_sec);
    return 0;
}
