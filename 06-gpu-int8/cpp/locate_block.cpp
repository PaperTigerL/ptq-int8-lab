// 二分定位：GPU 与 numpy 参考在**第几个 block 之后**开始分歧。
//
// ⭐ 为什么用这个而不是逐层 probe：
//   端到端比对只告诉你「不一致」，不告诉你「哪一层开始错」。
//   逐层编 probe 找错层，每层要重编译一次（这一轮实测每次 1-2 分钟）——
//   而这里只需对每个 block 各跑一次 GPU 与参考，比对激活的 max|diff|，
//   一次跑完全部 8 个 block，立刻看出分歧从哪开始。
//
// 用法：./locate_block <model> <calib> <data> <N>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "int8_engine.cuh"
#include "cifar10.h"

static const float kMean[3] = {0.4914f, 0.4822f, 0.4465f};
static const float kStd[3] = {0.2470f, 0.2435f, 0.2616f};

int main(int argc, char** argv)
{
    if (argc < 5)
    {
        fprintf(stderr, "usage: %s <model.bin> <calib.bin> <cifar_batch> <N>\n", argv[0]);
        return 2;
    }
    const int n = atoi(argv[4]);
    const int n_blocks = 8;

    Model model;
    if (!model.load(argv[1])) return 1;
    std::vector<std::pair<float, float>> ranges;
    {
        std::string cerr;
        if (!load_calib(argv[2], model.convs.size(), &ranges, &cerr))
        {
            fprintf(stderr, "[FAIL] 校准文件不可用：%s\n", cerr.c_str());
            return 1;
        }
    }

    Int8Engine engine;
    if (!engine.build(model, 8, 8)) return 1;
    {
        std::string cerr;
        if (!engine.set_calib(ranges, &cerr))
        {
            fprintf(stderr, "[FAIL] set_calib: %s\n", cerr.c_str());
            return 1;
        }
    }
    engine.alloc_scratch(n);

    cifar10::Batch batch;
    if (!batch.load(argv[3])) return 1;

    const int plane = 32 * 32;
    std::vector<float> in((size_t)n * 3 * plane);
    for (int b = 0; b < n; b++)
    {
        const uint8_t* px = batch.at(b).pixels - 1;
        float* dst = in.data() + (size_t)b * 3 * plane;
        for (int i = 0; i < 3 * plane; i++) dst[i] = (float)px[i] / 255.0f;
        for (int c = 0; c < 3; c++)
        {
            const float m = kMean[c] * 255.0f;
            const float inv = 1.0f / (kStd[c] * 255.0f);
            float* p = dst + (size_t)c * plane;
            for (int i = 0; i < plane; i++) p[i] = (p[i] * 255.0f - m) * inv;
        }
    }

    // 每个 block 之后的激活尺寸。
    // ⚠️ 这张表是 CIFAR-10 的 32x32 输入推出来的：
    //   32 → conv1(s2) 16 → maxpool(s2) 8 → layer1 8x8
    //   → layer2(s2) 4x4 → layer3(s2) 2x2 → layer4(s2) 1x1
    // 第一版我按 ImageNet 的 224 记成了 16/8/4/2/1，被参考侧的实际输出打回。
    // **凡是「硬编码的形状表」，都必须由参考实现打印一次来核对，不能凭印象写。**
    const int dim[9][3] = {{64,8,8},{64,8,8},{64,8,8},{128,4,4},{128,4,4},
                           {256,2,2},{256,2,2},{512,1,1},{512,1,1}};

    // 先导出 stem（CIFAR 32x32 → conv1(s2)16 → maxpool(s2)8）
    {
        const int C = 64, H = 8, W = 8;
        std::vector<float> mid((size_t)n * C * H * W);
        engine.forward(in.data(), n, 3, 32, 32, nullptr, -2, mid.data());
        CUDA_CHECK(cudaDeviceSynchronize());
        FILE* f = fopen("/tmp/gpu_mid_stem.bin", "wb");
        fwrite("PTQM", 1, 4, f);
        int32_t hdr[4] = {(int32_t)n, C, H, W};
        fwrite(hdr, sizeof(int32_t), 4, f);
        fwrite(mid.data(), sizeof(float), mid.size(), f);
        fclose(f);
        printf("导出 stem 后：(%d,%d,%d,%d) → /tmp/gpu_mid_stem.bin\n", n, C, H, W);
    }

    for (int bi = 0; bi < n_blocks; bi++)
    {
        const int C = dim[bi + 1][0];
        const size_t sz = (size_t)n * C * dim[bi + 1][1] * dim[bi + 1][2];
        std::vector<float> mid(sz);
        engine.forward(in.data(), n, 3, 32, 32, nullptr, bi, mid.data());
        CUDA_CHECK(cudaDeviceSynchronize());

        char path[256];
        snprintf(path, sizeof(path), "/tmp/gpu_mid_b%d.bin", bi);
        FILE* f = fopen(path, "wb");
        fwrite("PTQM", 1, 4, f);
        int32_t hdr[4] = {(int32_t)n, (int32_t)C, dim[bi + 1][1], dim[bi + 1][2]};
        fwrite(hdr, sizeof(int32_t), 4, f);
        fwrite(mid.data(), sizeof(float), mid.size(), f);
        fclose(f);
        printf("导出 block %d 之后：(%d,%d,%d,%d) → %s\n",
               bi, n, C, dim[bi + 1][1], dim[bi + 1][2], path);
    }
    return 0;
}