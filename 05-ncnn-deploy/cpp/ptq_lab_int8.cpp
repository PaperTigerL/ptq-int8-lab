// PTQ 实验器 v2：在 ncnn 内部完成「校准 → 量化 → int8 前向」全链路。
//
// 与 v1 的区别
// ------------
// v1 把权重量化后写回 fp32，走 fp32 前向（11.6ms/张）—— 丢了 ncnn 的效率。
// v2 直接把权重量化成 int8 Mat，设好激活 scale，重建 pipeline 让 ncnn 走
// 它自己的 int8 SIMD 路径（目标 6.6ms/张，4 线程 3.4ms）。
//
// 三段式
// ------
//   ① 校准：打开 ptq_collect 跑校准集，拿到每层输入的真实范围
//   ② 量化：权重 per-channel → int8；激活按 range 算 scale；挂到层上
//   ③ 评测：重建 pipeline 后跑测试集，走 int8 kernel
//
// 为什么激活 scale 要自己测
// -------------------------
// ncnn 的激活 scale 只能从模型文件读（ncnn2int8 离线生成）。要"运行时换配置"，
// 就必须自己拿到每层输入范围 —— 这正是 forward 里那个钩子的用途。
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "cifar10.h"
#include "layer.h"
#include "layer/convolution.h"
#include "net.h"

namespace {

const float kMean[3] = {0.4914f, 0.4822f, 0.4465f};
const float kStd[3] = {0.2470f, 0.2435f, 0.2616f};
const int kPlane = 32 * 32;

// npy: magic(6) + ver(2) + hlen(2) + header + float32
bool load_npy(const std::string& path, float* out, int n)
{
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[6];
    if (fread(magic, 1, 6, f) != 6 || memcmp(magic, "\x93NUMPY", 6) != 0) { fclose(f); return false; }
    char ver[2];
    if (fread(ver, 1, 2, f) != 2) { fclose(f); return false; }
    unsigned short hlen = 0;
    if (fread(&hlen, sizeof(hlen), 1, f) != 1) { fclose(f); return false; }
    if (fseek(f, hlen, SEEK_CUR) != 0) { fclose(f); return false; }
    const size_t want = (size_t)n * sizeof(float);
    const size_t got = fread(out, 1, want, f);
    fclose(f);
    return got == want;
}

// 权重 per-channel 对称量化到 int8，口径与 03-ptq-lab 一致：
//   scale = absmax / qmax, q = round(w/scale) clamp 到 [-(qmax+1), qmax]
void quantize_weight_to_int8(ncnn::Convolution* c, int bits, ncnn::Mat& out_w, ncnn::Mat& out_scales)
{
    const int qmax = (1 << (bits - 1)) - 1;
    const int qmin = -(qmax + 1);
    const int num_output = c->num_output;
    const int per_ch = c->weight_data_size / num_output;

    out_w.create(c->weight_data_size, (size_t)1u);
    out_scales.create(num_output, (size_t)4u);

    signed char* q = out_w;
    float* sc = out_scales;
    const float* w = c->weight_data;

    for (int p = 0; p < num_output; p++)
    {
        const float* wp = w + (size_t)p * per_ch;
        float absmax = 0.f;
        for (int i = 0; i < per_ch; i++)
        {
            const float a = std::fabs(wp[i]);
            if (a > absmax) absmax = a;
        }
        // 注意：ncnn 的 *_int8_scales 存的是**乘数**（qmax/absmax），不是步长。
        // 步长写法 absmax/qmax 会让 requantize 的 scale_in 差 qmax² 倍，精度直接崩到随机。
        float scale = (absmax > 0.f) ? (float)qmax / absmax : 0.f;
        sc[p] = scale;

        for (int i = 0; i < per_ch; i++)
        {
            if (scale <= 0.f) { q[(size_t)p * per_ch + i] = 0; continue; }
            int v = (int)std::lround(wp[i] * scale);
            if (v > qmax) v = qmax;
            if (v < qmin) v = qmin;
            q[(size_t)p * per_ch + i] = (signed char)v;
        }
    }
}

void usage(const char* p)
{
    fprintf(stderr,
            "用法：%s <param> <bin> <cifar_batch> [选项]\n"
            "  --bits N       权重量化位宽，0=不量化（默认 0）\n"
            "  --calib N      校准样本数，0=跳过校准（默认 1024）\n"
            "  --limit N      评测样本数，0=全部（默认 0）\n"
            "  --threads N    线程数（默认 4）\n"
            "  --calib-file F npy filelist（默认用 calib/filelist.txt）\n", p);
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 4) { usage(argv[0]); return 2; }

    const char* param_path = argv[1];
    const char* bin_path = argv[2];
    const char* data_path = argv[3];

    int bits = 0, calib_n = 1024, limit = 0, threads = 4;
    std::string calib_file = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/calib/filelist.txt";

    for (int i = 4; i < argc; i++)
    {
        if (!strcmp(argv[i], "--bits") && i + 1 < argc) bits = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--calib") && i + 1 < argc) calib_n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--limit") && i + 1 < argc) limit = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--calib-file") && i + 1 < argc) calib_file = argv[++i];
        else { fprintf(stderr, "未知参数 %s\n", argv[i]); return 2; }
    }

    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    net.opt.lightmode = false;
    net.opt.num_threads = threads;

    if (net.load_param(param_path) != 0) { fprintf(stderr, "load_param 失败\n"); return 1; }
    if (net.load_model(bin_path) != 0) { fprintf(stderr, "load_model 失败\n"); return 1; }

    // ---------- ① 校准：拿每层输入范围 ----------
    if (bits > 0 && calib_n > 0)
    {
        for (size_t i = 0; i < net.layers().size(); i++)
        {
            ncnn::Layer* l = net.layers()[i];
            if (l->type == "Convolution") ((ncnn::Convolution*)l)->ptq_collect = 1;
        }

        std::vector<std::string> npys;
        FILE* f = fopen(calib_file.c_str(), "r");
        if (!f) { fprintf(stderr, "读不到校准列表 %s\n", calib_file.c_str()); return 1; }
        char line[1024];
        while (fgets(line, sizeof(line), f) && (int)npys.size() < calib_n)
        {
            std::string s(line);
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
            if (!s.empty()) npys.push_back(s);
        }
        fclose(f);

        std::vector<float> buf(3 * kPlane);
        for (size_t i = 0; i < npys.size(); i++)
        {
            if (!load_npy(npys[i], buf.data(), (int)buf.size())) { fprintf(stderr, "npy 失败\n"); return 1; }
            ncnn::Mat in(32, 32, 3);
            for (int c = 0; c < 3; c++) memcpy(in.channel(c), buf.data() + c * kPlane, kPlane * sizeof(float));
            ncnn::Extractor ex = net.create_extractor();
            ex.input("input", in);
            ncnn::Mat out;
            if (ex.extract("logits", out) != 0) return 1;
        }
        fprintf(stderr, "校准完成：%d 张\n", (int)npys.size());
    }

    // ---------- ② 量化 + 挂到层上 ----------
    if (bits > 0)
    {
        int nq = 0;
        for (size_t i = 0; i < net.layers().size(); i++)
        {
            ncnn::Layer* l = net.layers()[i];
            if (l->type != "Convolution") continue;
            ncnn::Convolution* c = (ncnn::Convolution*)l;
            c->ptq_collect = 0;

            if (c->weight_data.total() == 0u || c->weight_data.elembits() != 32)
            {
                fprintf(stderr, "[跳过] %s 权重不是 fp32\n", c->name.c_str());
                continue;
            }

            ncnn::Mat wq, ws;
            quantize_weight_to_int8(c, bits, wq, ws);

            // 激活 scale：对称，qmax = 2^(bits-1)-1
            const int qmax = (1 << (bits - 1)) - 1;
            const float amax = std::max(std::fabs(c->ptq_in_min), std::fabs(c->ptq_in_max));
            float a_scale = (amax > 0.f) ? amax / (float)qmax : 0.f;

            ncnn::Mat as(1, (size_t)4u);
            as[0] = (a_scale > 0.f) ? 1.0f / a_scale : 0.f; // ncnn 存的是乘数

            c->weight_data = wq;
            c->weight_data_int8_scales = ws;
            c->bottom_blob_int8_scales = as;
            c->int8_scale_term = 2;
            c->int8_bits = bits;
            nq++;
        }
        fprintf(stderr, "量化完成：%d 个 Conv 到 %d bit\n", nq, bits);

        // ---------- 重建 pipeline（关键）----------
        for (size_t i = 0; i < net.layers().size(); i++)
        {
            net.layers()[i]->destroy_pipeline(net.opt);
            net.layers()[i]->create_pipeline(net.opt);
        }
        fprintf(stderr, "pipeline 已重建\n");
    }

    // ---------- ③ 评测 ----------
    cifar10::Batch batch;
    if (!batch.load(data_path)) { fprintf(stderr, "读数据失败\n"); return 1; }

    const int n = (limit > 0) ? std::min(limit, batch.size()) : batch.size();
    printf("模型 %s | bits=%d | 样本=%d | 线程=%d\n", param_path, bits, n, threads);

    int correct = 0;
    double total_ms = 0.0;

    for (int i = 0; i < n; i++)
    {
        cifar10::Sample s = batch.at(i);
        ncnn::Mat in(32, 32, 3);
        for (int c = 0; c < 3; c++)
        {
            const float mean = kMean[c] * 255.0f;
            const float norm = 1.0f / (255.0f * kStd[c]);
            float* p = in.channel(c);
            const uint8_t* src = s.pixels + (size_t)c * kPlane;
            for (int j = 0; j < kPlane; j++) p[j] = ((float)src[j] - mean) * norm;
        }

        const auto t0 = std::chrono::steady_clock::now();
        ncnn::Extractor ex = net.create_extractor();
        ex.input("input", in);
        ncnn::Mat out;
        if (ex.extract("logits", out) != 0) { fprintf(stderr, "extract 失败\n"); return 1; }
        total_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        int best = 0;
        for (int k = 1; k < out.total(); k++) if (out[k] > out[best]) best = k;
        if (best == (int)s.label) correct++;
    }

    printf("精度 %.2f%% (%d/%d)  %.2f ms/张\n", 100.0 * correct / n, correct, n, total_ms / n);
    return 0;
}
