#include "cifar10.h"

#include <cstdio>
#include <fstream>

namespace cifar10 {

bool Batch::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "[cifar10] 打不开批次文件：%s\n", path.c_str());
        return false;
    }

    f.seekg(0, std::ios::end);
    const std::streamoff size = f.tellg();
    f.seekg(0, std::ios::beg);

    if (size <= 0 || size % kRecordSize != 0) {
        std::fprintf(stderr,
                     "[cifar10] 文件大小 %lld 不是 %d 的整数倍，不是合法的 "
                     "CIFAR-10 二进制批次：%s\n",
                     static_cast<long long>(size), kRecordSize, path.c_str());
        return false;
    }

    const int n = static_cast<int>(size / kRecordSize);
    data_.resize(static_cast<size_t>(size));
    f.read(reinterpret_cast<char*>(data_.data()), size);
    if (!f) {
        std::fprintf(stderr, "[cifar10] 读取失败：%s\n", path.c_str());
        return false;
    }

    labels_.resize(n);
    for (int i = 0; i < n; i++) {
        labels_[i] = data_[static_cast<size_t>(i) * kRecordSize];
    }

    path_ = path;
    return true;
}

}  // namespace cifar10
