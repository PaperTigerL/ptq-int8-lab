// CIFAR-10 二进制批次读取。
//
// 为什么要自己读，而不是让 Python 转成 npy 再喂进来：
//   部署侧拿到的就是原始的 CIFAR-10 二进制文件（data_batch_* / test_batch），
//   不是谁预处理好的张量。把「二进制 → 可推理的 Mat」这条路走通，
//   才算把数据管线接到了模型上；跳过它就少验一环。
//
// 文件格式（CIFAR-10 官方二进制，每样本 1 + 3072 字节）：
//   [0]        label，1 字节，0..9
//   [1..3072]  图像，3 个 1024 字节的平面，顺序 R、G、B
//              每个平面内是 32x32 行优先（row-major）
//   注意：是**平面式 CHW**，不是交错的 HWC。这一点直接决定后面怎么填 ncnn::Mat。
#ifndef CIFAR10_H
#define CIFAR10_H

#include <cstdint>
#include <string>
#include <vector>

namespace cifar10 {

constexpr int kWidth = 32;
constexpr int kHeight = 32;
constexpr int kChannels = 3;
constexpr int kPlaneSize = kWidth * kHeight;          // 1024
constexpr int kRecordSize = 1 + kChannels * kPlaneSize;  // 3073
constexpr int kNumClasses = 10;

// 一张图：标签 + 指向原始字节的指针（不拷贝，调用方持有缓冲区）。
struct Sample {
    uint8_t label;
    const uint8_t* pixels;   // 指向 3072 字节，R/G/B 三个平面
};

// 一整个批次文件（data_batch_1..5 或 test_batch）。
// 10000 张，30730000 字节。一次性读进内存，评测时反复用。
class Batch {
public:
    bool load(const std::string& path);

    int size() const { return static_cast<int>(labels_.size()); }
    const std::string& path() const { return path_; }

    Sample at(int i) const {
        return Sample{labels_[i], data_.data() + static_cast<size_t>(i) * kRecordSize + 1};
    }

private:
    std::string path_;
    std::vector<uint8_t> data_;
    std::vector<uint8_t> labels_;
};

}  // namespace cifar10

#endif  // CIFAR10_H
