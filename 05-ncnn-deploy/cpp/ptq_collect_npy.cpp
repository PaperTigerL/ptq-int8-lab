// 验证 PTQ 观测钩子：用 ncnn2table 的同一批校准 npy 跑，才能和它记录的 max 对比。
//
// 之前的 CIFAR raw 版本跑的是测试集，ncnn2table 跑的是训练集前 1024 张，
// 数据不同就没法判断 range 差 2.5 倍到底是谁的问题。
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "layer.h"
#include "layer/convolution.h"
#include "net.h"

static const int kPlane = 32 * 32;

// npy: magic(6) + ver(2) + hlen(2) + header(hlen) + float32 数据
static bool load_npy(const std::string& path, float* out, int n)
{
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[6];
    if (fread(magic, 1, 6, f) != 6 || memcmp(magic, "\x93NUMPY", 6) != 0) { fclose(f); return false; }
    char ver[2];
    if (fread(ver, 1, 2, f) != 2) { fclose(f); return false; }
    unsigned short hlen = 0;
    // size=sizeof(hlen), nmemb=1 —— 写成 (2, 2) 会读 4 字节，把 hlen 和位置都搞错
    if (fread(&hlen, sizeof(hlen), 1, f) != 1) { fclose(f); return false; }
    if (fseek(f, hlen, SEEK_CUR) != 0) { fclose(f); return false; }
    const size_t want = (size_t)n * sizeof(float);
    const size_t got = fread(out, 1, want, f);
    fclose(f);
    return got == want;
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        fprintf(stderr, "usage: %s <param> <bin> [filelist] [n_samples]\n", argv[0]);
        return 2;
    }

    const std::string filelist = (argc > 3) ? argv[3]
        : "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/calib/filelist.txt";
    const int n_samples = (argc > 4) ? atoi(argv[4]) : 1024;

    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    net.opt.lightmode = false;
    net.opt.num_threads = 1;
    if (net.load_param(argv[1]) != 0) return 1;
    if (net.load_model(argv[2]) != 0) return 1;

    int n_conv = 0;
    for (size_t i = 0; i < net.layers().size(); i++)
    {
        ncnn::Layer* l = net.layers()[i];
        if (l->type != "Convolution") continue;
        ((ncnn::Convolution*)l)->ptq_collect = 1;
        n_conv++;
    }

    std::vector<std::string> npys;
    {
        FILE* f = fopen(filelist.c_str(), "r");
        if (!f) { fprintf(stderr, "读不到 filelist：%s\n", filelist.c_str()); return 1; }
        char line[1024];
        while (fgets(line, sizeof(line), f) && (int)npys.size() < n_samples)
        {
            std::string s(line);
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
            if (!s.empty()) npys.push_back(s);
        }
        fclose(f);
    }
    fprintf(stderr, "观测 %d 个 Conv，校准样本 %d 个（来自 %s）\n", n_conv, (int)npys.size(), filelist.c_str());

    // 用 vector 而不是栈数组：栈数组会让 _FORTIFY_SOURCE 在编译期推断出
    // 一个刚好相等的上界，运行时直接 abort（即使逻辑上没越界）。
    std::vector<float> buf(3 * kPlane);
    for (size_t i = 0; i < npys.size(); i++)
    {
        if (!load_npy(npys[i], buf.data(), (int)buf.size())) { fprintf(stderr, "npy 失败：%s\n", npys[i].c_str()); return 1; }

        ncnn::Mat in(32, 32, 3);
        for (int c = 0; c < 3; c++)
            memcpy(in.channel(c), buf.data() + c * kPlane, kPlane * sizeof(float));

        ncnn::Extractor ex = net.create_extractor();
        ex.input("input", in);
        ncnn::Mat out;
        if (ex.extract("logits", out) != 0) return 1;
    }

    printf("\n%-42s %10s %10s %10s\n", "层", "min", "max", "absmax");
    printf("--------------------------------------------------------------------------------\n");
    for (size_t i = 0; i < net.layers().size(); i++)
    {
        ncnn::Layer* l = net.layers()[i];
        if (l->type != "Convolution") continue;
        ncnn::Convolution* c = (ncnn::Convolution*)l;
        const float amax = std::max(std::fabs(c->ptq_in_min), std::fabs(c->ptq_in_max));
        printf("%-42s %10.4f %10.4f %10.4f\n", c->name.c_str(), c->ptq_in_min, c->ptq_in_max, amax);
    }
    return 0;
}
