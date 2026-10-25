#include "evaluator.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace eval {

// 检测 param 里是否有 sub-8bit 的 Convolution（param id 20 < 8）。
//
// 为什么必须据此关闭 packing：sub-8bit 在 ncnn 里走的是标量路径
// （Quantize_x86/Requantize_x86 在 bits!=8 时回落到基类通用实现），
// 而基类 Quantize::forward 只处理 elempack==1 的连续 blob ——
// 它不认 pack8 的通道布局，硬跑会把数据搅乱（实测精度直接掉到 10%）。
// pack8 下的 int4 需要专门写 SIMD 内核，那是 P2 的事，P0 先要数值正确。
static bool param_has_sub8bit(const std::string& path) {
    std::ifstream f(path.c_str());
    if (!f) return false;

    std::string line;
    while (std::getline(f, line)) {
        if (line.compare(0, 11, "Convolution") != 0) continue;

        const std::string key = " 20=";
        size_t p = line.find(key);
        if (p == std::string::npos) continue;

        int bits = std::atoi(line.c_str() + p + key.size());
        if (bits > 0 && bits < 8) return true;
    }
    return false;
}

Evaluator::Evaluator() {
    // (p - m*255) * 1/(255*s) == (p/255 - m)/s
    const float m[3] = {0.4914f, 0.4822f, 0.4465f};
    const float s[3] = {0.2470f, 0.2435f, 0.2616f};
    for (int c = 0; c < 3; c++) {
        mean_[c] = m[c] * 255.0f;
        norm_[c] = 1.0f / (255.0f * s[c]);
    }
}

bool Evaluator::load(const std::string& param_path, const std::string& bin_path,
                     int threads) {
    net_.opt.num_threads = threads;
    net_.opt.lightmode = false;

    // sub-8bit 现在不再需要关闭 packing：Quantize/Requantize 的通用实现已支持
    // elempack，pack8 布局可以一路保持，conv 仍走 ncnn 的 int8 SIMD。
    // 这里只留一条提示，方便对比日志。
    if (param_has_sub8bit(param_path)) {
        std::fprintf(stderr, "[eval] 检测到 sub-8bit（Convolution param 20<8），packing 保持开启\n");
    }

    if (net_.load_param(param_path.c_str()) != 0) {
        std::fprintf(stderr, "[eval] load_param 失败：%s\n", param_path.c_str());
        return false;
    }
    if (net_.load_model(bin_path.c_str()) != 0) {
        std::fprintf(stderr, "[eval] load_model 失败：%s\n", bin_path.c_str());
        return false;
    }

    layer_count_ = static_cast<int>(net_.layers().size());

    // 输入/输出 blob 名：优先用 ncnn 自己推导的（Net::input_names/output_names），
    // 推导不出来再退回 onnx2ncnn 的默认命名。不写死是为了换模型时不用改代码。
    const std::vector<const char*>& ins = net_.input_names();
    const std::vector<const char*>& outs = net_.output_names();
    in_name_ = (!ins.empty() && ins[0]) ? ins[0] : "input";
    out_name_ = (!outs.empty() && outs[0]) ? outs[0] : "logits";
    return true;
}

void Evaluator::set_preprocess(const float mean[3], const float norm[3]) {
    for (int c = 0; c < 3; c++) {
        mean_[c] = mean[c];
        norm_[c] = norm[c];
    }
}

void Evaluator::set_blob_names(const std::string& in, const std::string& out) {
    in_name_ = in;
    out_name_ = out;
}

ncnn::Mat Evaluator::make_input(const uint8_t* p) const {
    // ncnn::Mat 是平面式 CHW：channel(c) 是连续的 w*h 块，
    // 和 CIFAR-10 二进制的三个平面对应，直接按平面填。
    ncnn::Mat in(cifar10::kWidth, cifar10::kHeight, 3, sizeof(float));
    float* r = static_cast<float*>(in.channel(0));
    float* g = static_cast<float*>(in.channel(1));
    float* b = static_cast<float*>(in.channel(2));
    const int plane = cifar10::kPlaneSize;
    for (int i = 0; i < plane; i++) {
        r[i] = (static_cast<float>(p[i]) - mean_[0]) * norm_[0];
        g[i] = (static_cast<float>(p[i + plane]) - mean_[1]) * norm_[1];
        b[i] = (static_cast<float>(p[i + 2 * plane]) - mean_[2]) * norm_[2];
    }
    return in;
}

Result Evaluator::run(const cifar10::Batch& batch, int limit, int warmup) {
    Result r;
    const int n = (limit > 0 && limit < batch.size()) ? limit : batch.size();

    // ---- warmup：前几次调用要付内存分配、页错误、指令缓存这些一次性成本，
    //      不计入耗时，否则第一张图的延迟会难看得误导人。
    //
    // ⚠️ Extractor 必须**每张图新建一个**，不能提到循环外复用。
    //    这不是风格问题，是这个版本 ncnn 的硬语义：
    //      - Extractor::input()  只写 d->blob_mats[blob_index] = in，不清其他 blob
    //      - Extractor::extract() 只在 d->blob_mats[blob_index].dims == 0 时
    //        才真正跑前向（src/net.cpp:3254），否则直接返回上一次的结果
    //    提出来复用 → 第二张起全部返回第一张的预测。实测 2000 张准确率
    //    从 87% 掉到 9.95%（十个类随机猜），耗时掉到 0.002ms/张（根本没算）。
    //    这个坑没有任何报错，只会静默给出错的数字，所以钉在这里。
    for (int i = 0; i < warmup && i < n; i++) {
        ncnn::Mat in = make_input(batch.at(i).pixels);
        ncnn::Extractor ex = net_.create_extractor();
        ncnn::Mat out;
        ex.input(in_name_.c_str(), in);
        ex.extract(out_name_.c_str(), out);
    }

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; i++) {
        const cifar10::Sample s = batch.at(i);

        ncnn::Mat in = make_input(s.pixels);

        ncnn::Extractor ex = net_.create_extractor();
        if (ex.input(in_name_.c_str(), in) != 0) {
            std::fprintf(stderr, "[eval] input 失败（blob 名 %s 不对？）\n",
                         in_name_.c_str());
            return r;
        }

        ncnn::Mat out;
        if (ex.extract(out_name_.c_str(), out) != 0) {
            std::fprintf(stderr, "[eval] extract 失败（blob 名 %s 不对？）\n",
                         out_name_.c_str());
            return r;
        }

        // argmax。ncnn 的 Mat 此时是 1x1x10 的 fp32。
        int best = 0;
        float best_val = 0.0f;
        const int classes = out.w * out.h * out.c;
        const float* logits = static_cast<const float*>(out.data);
        for (int c = 0; c < classes; c++) {
            if (c == 0 || logits[c] > best_val) {
                best_val = logits[c];
                best = c;
            }
        }

        if (best == static_cast<int>(s.label)) r.correct++;
    }
    const auto t1 = std::chrono::steady_clock::now();

    r.total = n;
    r.total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    r.accuracy = n > 0 ? static_cast<double>(r.correct) / n : 0.0;
    r.ms_per_image = n > 0 ? r.total_ms / n : 0.0;
    r.img_per_sec = r.total_ms > 0.0 ? n * 1000.0 / r.total_ms : 0.0;
    return r;
}

}  // namespace eval
