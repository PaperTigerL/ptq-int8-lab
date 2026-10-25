// PTQ 实验器 v3：一次校准，扫多个位宽配置。
//
// 为什么要 sweep
// -------------
// 做实验的真实场景是"同一模型、一组位宽"，而不是"一个位宽跑一次进程"。
// v2 每跑一个配置都要重新读模型 + 重跑 1024 张校准（≈3.5s 固定开销）；
// 扫 5 个配置就白烧 17 秒。v3 把校准结果缓存在内存里，只做一次。
//
// 与 v2 的另一个差别：Extractor 在评测循环外复用一个，不再每张 create_extractor。
//
// 局限（必须说清）
// ---------------
// ncnn 的 Convolution 是单张推理，没有 batch 维度，所以拿不到 batch 摊销。
// 想再快只能减评测样本或换代理指标。
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

struct Range { float lo, hi; };

// 校准结果缓存：每次进程启动重跑 1024 张校准要约 1.2s，
// 批量扫描时这是纯浪费。格式：每行 "lo hi"，顺序与 param 里 Convolution 一致。
bool save_ranges(const std::string& path, const std::vector<Range>& r)
{
    FILE* f = fopen(path.c_str(), "w");
    if (!f) return false;
    for (size_t i = 0; i < r.size(); i++) fprintf(f, "%.9g %.9g\n", r[i].lo, r[i].hi);
    fclose(f);
    return true;
}

bool load_ranges(const std::string& path, std::vector<Range>& r)
{
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return false;
    r.clear();
    float lo = 0.f, hi = 0.f;
    while (fscanf(f, "%f %f", &lo, &hi) == 2) r.push_back(Range{lo, hi});
    fclose(f);
    return !r.empty();
}

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

// 校准：跑一遍校准集，记下每层 Convolution 输入的范围。只做一次。
void calibrate(ncnn::Net& net, const std::string& filelist, int n_samples, std::vector<Range>& out)
{
    std::vector<ncnn::Convolution*> convs;
    for (size_t i = 0; i < net.layers().size(); i++)
    {
        ncnn::Layer* l = net.layers()[i];
        if (l->type != "Convolution") continue;
        ncnn::Convolution* c = (ncnn::Convolution*)l;
        c->ptq_collect = 1;
        c->ptq_in_min = 1e30f;
        c->ptq_in_max = -1e30f;
        convs.push_back(c);
    }

    std::vector<std::string> npys;
    FILE* f = fopen(filelist.c_str(), "r");
    if (!f) { fprintf(stderr, "读不到校准列表\n"); return; }
    char line[1024];
    while (fgets(line, sizeof(line), f) && (int)npys.size() < n_samples)
    {
        std::string s(line);
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
        if (!s.empty()) npys.push_back(s);
    }
    fclose(f);

    std::vector<float> buf(3 * kPlane);
    for (size_t i = 0; i < npys.size(); i++)
    {
        if (!load_npy(npys[i], buf.data(), (int)buf.size())) return;
        ncnn::Mat in(32, 32, 3);
        for (int c = 0; c < 3; c++) memcpy(in.channel(c), buf.data() + c * kPlane, kPlane * sizeof(float));
        ncnn::Extractor ex = net.create_extractor();
        ex.input("input", in);
        ncnn::Mat out;
        if (ex.extract("logits", out) != 0) return;
    }

    out.clear();
    for (size_t i = 0; i < convs.size(); i++)
    {
        convs[i]->ptq_collect = 0;
        out.push_back(Range{convs[i]->ptq_in_min, convs[i]->ptq_in_max});
    }
    fprintf(stderr, "校准完成：%zu 张，%zu 层\n", npys.size(), convs.size());
}

// 权重 per-channel 对称量化到 int8。
// ncnn 的 *_int8_scales 存的是**乘数** qmax/absmax，不是步长 —— 写错精度直接崩。
void quantize_weight(ncnn::Convolution* c, int bits, ncnn::Mat& wq, ncnn::Mat& ws)
{
    const int qmax = (1 << (bits - 1)) - 1;
    const int qmin = -(qmax + 1);
    const int num_output = c->num_output;
    const int per_ch = c->weight_data_size / num_output;

    wq.create(c->weight_data_size, (size_t)1u);
    ws.create(num_output, (size_t)4u);

    signed char* q = wq;
    float* sc = ws;
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
        const float mult = (absmax > 0.f) ? (float)qmax / absmax : 0.f;
        sc[p] = mult;
        for (int i = 0; i < per_ch; i++)
        {
            if (mult <= 0.f) { q[(size_t)p * per_ch + i] = 0; continue; }
            int v = (int)std::lround(wp[i] * mult);
            if (v > qmax) v = qmax;
            if (v < qmin) v = qmin;
            q[(size_t)p * per_ch + i] = (signed char)v;
        }
    }
}

// 把量化配置挂到层上（权重 int8 + 激活乘数），然后重建 pipeline 让 int8 kernel 就位。
int apply_quant(ncnn::Net& net, int bits, const std::vector<Range>& ranges)
{
    const int qmax = (1 << (bits - 1)) - 1;
    int nq = 0, idx = 0;

    for (size_t i = 0; i < net.layers().size(); i++)
    {
        ncnn::Layer* l = net.layers()[i];
        if (l->type != "Convolution") continue;
        ncnn::Convolution* c = (ncnn::Convolution*)l;

        if (idx >= (int)ranges.size()) break;
        if (c->weight_data.total() == 0u || c->weight_data.elembits() != 32) { idx++; continue; }

        ncnn::Mat wq, ws;
        quantize_weight(c, bits, wq, ws);

        const float amax = std::max(std::fabs(ranges[idx].lo), std::fabs(ranges[idx].hi));
        ncnn::Mat as(1, (size_t)4u);
        as[0] = (amax > 0.f) ? (float)qmax / amax : 0.f; // 乘数

        c->weight_data = wq;
        c->weight_data_int8_scales = ws;
        c->bottom_blob_int8_scales = as;
        c->int8_scale_term = 2;
        c->int8_bits = bits;
        idx++;
        nq++;
    }

    for (size_t i = 0; i < net.layers().size(); i++)
    {
        net.layers()[i]->destroy_pipeline(net.opt);
        net.layers()[i]->create_pipeline(net.opt);
    }
    return nq;
}

double evaluate(ncnn::Net& net, const cifar10::Batch& batch, int n)
{
    int correct = 0;
    double total_ms = 0.0;
    // 注意：Extractor 不能跨样本复用。复用后 extract 直接返回缓存/空结果，
    // 实测 0.03ms/张、精度掉到随机 —— 必须每张新建（开销小，别 optimization 过头）。
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
        if (ex.extract("logits", out) != 0) break;
        total_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        int best = 0;
        for (int k = 1; k < out.total(); k++) if (out[k] > out[best]) best = k;
        if (best == (int)s.label) correct++;
    }
    printf("  -> 精度 %.2f%% (%d/%d)  %.2f ms/张\n", 100.0 * correct / n, correct, n, total_ms / n);
    return total_ms;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        fprintf(stderr, "用法：%s <param> <bin> <cifar_batch> [--sweep 8,4,3] [--calib N] [--limit N] [--threads N]\n", argv[0]);
        return 2;
    }

    const char* param_path = argv[1];
    const char* bin_path = argv[2];
    const char* data_path = argv[3];

    std::string sweep = "8";
    int calib_n = 1024, limit = 0, threads = 4;
    std::string calib_file = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/calib/filelist.txt";
    std::string cache;

    for (int i = 4; i < argc; i++)
    {
        if (!strcmp(argv[i], "--sweep") && i + 1 < argc) sweep = argv[++i];
        else if (!strcmp(argv[i], "--calib") && i + 1 < argc) calib_n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--limit") && i + 1 < argc) limit = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--calib-file") && i + 1 < argc) calib_file = argv[++i];
        else if (!strcmp(argv[i], "--calib-cache") && i + 1 < argc) cache = argv[++i];
        else { fprintf(stderr, "未知参数 %s\n", argv[i]); return 2; }
    }

    std::vector<int> bits_list;
    {
        std::string cur;
        for (char ch : sweep)
        {
            if (ch == ',') { if (!cur.empty()) bits_list.push_back(atoi(cur.c_str())); cur.clear(); }
            else cur.push_back(ch);
        }
        if (!cur.empty()) bits_list.push_back(atoi(cur.c_str()));
    }

    cifar10::Batch batch;
    if (!batch.load(data_path)) { fprintf(stderr, "读数据失败\n"); return 1; }
    const int n = (limit > 0) ? std::min(limit, batch.size()) : batch.size();

    // 校准：优先用缓存（批量扫描时省掉每次 1.2s 的固定开销）
    std::vector<Range> ranges;
    if (!cache.empty() && load_ranges(cache, ranges))
    {
        fprintf(stderr, "复用校准缓存 %s（%zu 层）\n", cache.c_str(), ranges.size());
    }
    else
    {
        ncnn::Net net;
        net.opt.use_vulkan_compute = false;
        net.opt.lightmode = false;
        net.opt.num_threads = threads;
        if (net.load_param(param_path) != 0 || net.load_model(bin_path) != 0) return 1;
        calibrate(net, calib_file, calib_n, ranges);
        if (!cache.empty()) save_ranges(cache, ranges);
    }

    printf("sweep=%s | 样本=%d | 线程=%d | 校准 %zu 层\n", sweep.c_str(), n, threads, ranges.size());

    const auto t_all0 = std::chrono::steady_clock::now();
    double sum_eval_ms = 0.0;

    for (size_t bi = 0; bi < bits_list.size(); bi++)
    {
        const int bits = bits_list[bi];
        ncnn::Net net;                      // 每个配置重载 fp32（上一个配置已把权重改成 int8）
        net.opt.use_vulkan_compute = false;
        net.opt.lightmode = false;
        net.opt.num_threads = threads;
        if (net.load_param(param_path) != 0 || net.load_model(bin_path) != 0) return 1;

        const auto t0 = std::chrono::steady_clock::now();
        const int nq = apply_quant(net, bits, ranges);
        const double qms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        printf("[W%d] 量化 %d 层 + 重建 %.1f ms\n", bits, nq, qms);
        sum_eval_ms += evaluate(net, batch, n);
    }

    const double all_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_all0).count();
    printf("\n%d 个配置总耗时 %.1f s（评测 %.1f s）\n", (int)bits_list.size(), all_ms / 1000.0, sum_eval_ms / 1000.0);
    return 0;
}
