// 模型加载 + 单张前向 + 精度/延迟统计。
//
// 这个类的职责只有一件事：给定一个 ncnn 模型和一个 CIFAR-10 批次，
// 算出「准确率」和「每张图耗时」。数据怎么来（cifar10 模块）、
// 结果怎么打印（main）都不归它管。
//
// 两个实现上的选择，理由是明确的：
//
// 1) 预处理不调 substract_mean_normalize，而是手工填三个 channel。
//    CIFAR-10 二进制是**平面式**（R 平面、G 平面、B 平面各自连续），
//    ncnn::Mat 也是平面式，直接按平面拷就行。若先转成交错的 HWC 再让
//    ncnn 去解，多一次全图拷贝且容易搞错通道顺序 —— 而通道顺序错了
//    精度不会报错，只会静默掉几个点。
//
// 2) mean/norm 用 (x - mean) * norm 的形式，等价于 PyTorch 的
//    Normalize((x/255 - m) / s)。取 mean_c = m_c * 255、
//    norm_c = 1 / (255 * s_c) 即可严格相等，不引入额外舍入。
#ifndef EVALUATOR_H
#define EVALUATOR_H

#include <string>

#include "cifar10.h"
#include "net.h"

namespace eval {

struct Result {
    int total = 0;
    int correct = 0;
    double accuracy = 0.0;      // 0..1
    double total_ms = 0.0;      // 不含 warmup
    double ms_per_image = 0.0;
    double img_per_sec = 0.0;
};

class Evaluator {
public:
    // mean/norm 三通道，形式为 (pixel - mean) * norm。
    // CIFAR-10 默认值对应 Normalize(mean=(0.4914,0.4822,0.4465),
    //                             std=(0.2470,0.2435,0.2616))
    Evaluator();

    bool load(const std::string& param_path, const std::string& bin_path,
              int threads);

    // 不传则用 CIFAR-10 口径
    void set_preprocess(const float mean[3], const float norm[3]);

    // 输入/输出 blob 名。load() 后若自动探测失败，可用这个覆盖。
    void set_blob_names(const std::string& in, const std::string& out);

    Result run(const cifar10::Batch& batch, int limit = 0, int warmup = 5);

    const std::string& input_name() const { return in_name_; }
    const std::string& output_name() const { return out_name_; }
    int layer_count() const { return layer_count_; }

private:
    // 把 CIFAR-10 的一帧填成 ncnn 能吃的前向输入 Mat
    ncnn::Mat make_input(const uint8_t* pixels) const;

    ncnn::Net net_;
    std::string in_name_;
    std::string out_name_;
    int layer_count_ = 0;
    float mean_[3];
    float norm_[3];
};

}  // namespace eval

#endif  // EVALUATOR_H
