#!/usr/bin/env python3
"""Turn the eval harness copy into a same-process interleaved A/B benchmark.

- one Net; per-batch config toggle:
    pool  -> NCNN_CUDA_POOL_FUSE_PAD=1/0
    binop -> NCNN_CUDA_BINARYOP_VEC=1/0
    relu  -> rewire Conv->ReLU pairs at runtime (conv activation on/off)
- 8 rounds, 30 warmup images, config alternates every batch;
- per-round medians, overall median/min, bootstrap 95% CI of the paired
  per-round improvement; 128-image bitwise logits check;
- decision rule: CI lower bound > 0 and median improvement >= 2%.
"""
import io

PATH = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda/ncnn_cuda_ab_harness.cpp"

INC_OLD = '#include "cifar10.h"'
INC_NEW = '''#include "cifar10.h"
#include "layer/convolution.h"
#include "layer/relu.h"

#include <cmath>
#include <cstring>
#include <random>
#include <string>'''

ARG_OLD = "    int sync_every = 1;"
ARG_NEW = """    int sync_every = 1;
    const char* ab_switch = "pool";
    int ab_images = 256;
    int ab_batch = 32;
    int ab_rounds = 8;
    int ab_warmup = 30;"""

PARSE_OLD = '        else if (!strcmp(argv[i], "--sync-every") && i + 1 < argc) sync_every = atoi(argv[++i]);'
PARSE_NEW = '''        else if (!strcmp(argv[i], "--sync-every") && i + 1 < argc) sync_every = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ab-switch") && i + 1 < argc) ab_switch = argv[++i];
        else if (!strcmp(argv[i], "--ab-images") && i + 1 < argc) ab_images = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ab-batch") && i + 1 < argc) ab_batch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ab-rounds") && i + 1 < argc) ab_rounds = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ab-warmup") && i + 1 < argc) ab_warmup = atoi(argv[++i]);'''

TAIL = r'''    // ---------- 交错 A/B：同一进程、单 Net、运行时开关 ----------
    const std::string absw = ab_switch;
    if (absw != "pool" && absw != "binop" && absw != "relu")
        fatal("--ab-switch 必须是 pool|binop|relu");
    if (ab_images < 1 || ab_batch < 1 || ab_rounds < 1 || ab_warmup < 1)
        fatal("A/B 参数非法");

    // 测量口径：延迟同步到每 8 张，减少串行化噪声（T-12 已验证逐位一致）。
    sync_every = 8;

    // 非目标开关固定在基线：pool 默认开、binop 默认关、ReLU 图不融合。
    setenv("NCNN_CUDA_POOL_FUSE_PAD", "1", 1);
    setenv("NCNN_CUDA_BINARYOP_VEC", "0", 1);

    struct FusePair
    {
        int conv_idx;
        int relu_idx;
        int conv_top;
        int relu_top;
        std::vector<int> relu_consumers;
    };
    std::vector<FusePair> fuse_pairs;

    if (absw == "relu")
    {
        for (size_t i = 0; i < net.layers.size(); i++)
        {
            ncnn::Layer* layer = net.layers[i];
            if (layer->type != "Convolution") continue;
            ncnn::Convolution* conv = static_cast<ncnn::Convolution*>(layer);
            if (conv->activation_type != 0) continue;
            if (layer->tops.size() != 1) continue;
            const int top = layer->tops[0];
            if (net.blobs[top].consumers.size() != 1) continue;
            const int ri = net.blobs[top].consumers[0];
            ncnn::Layer* rl = net.layers[ri];
            if (rl->type != "ReLU") continue;
            if (static_cast<ncnn::ReLU*>(rl)->slope != 0.0f) continue;
            if (rl->bottoms.size() != 1 || rl->tops.size() != 1) continue;
            if (rl->bottoms[0] != top) continue;
            const int rt = rl->tops[0];
            FusePair p;
            p.conv_idx = (int)i;
            p.relu_idx = ri;
            p.conv_top = top;
            p.relu_top = rt;
            p.relu_consumers = net.blobs[rt].consumers;
            fuse_pairs.push_back(p);
        }
        if (fuse_pairs.empty()) fatal("没有可融合的 Conv+ReLU 对");
        printf("relu fuse pairs: %zu\n", fuse_pairs.size());
    }

    auto set_relu_fused = [&](bool on) {
        for (size_t k = 0; k < fuse_pairs.size(); k++)
        {
            const FusePair& p = fuse_pairs[k];
            ncnn::Convolution* conv = static_cast<ncnn::Convolution*>(net.layers[p.conv_idx]);
            if (on)
            {
                conv->activation_type = 1;
                for (size_t c = 0; c < p.relu_consumers.size(); c++)
                {
                    ncnn::Layer* next = net.layers[p.relu_consumers[c]];
                    for (size_t b = 0; b < next->bottoms.size(); b++)
                        if (next->bottoms[b] == p.relu_top) next->bottoms[b] = p.conv_top;
                }
                net.blobs[p.conv_top].consumers = p.relu_consumers;
                net.blobs[p.relu_top].consumers.clear();
            }
            else
            {
                conv->activation_type = 0;
                for (size_t c = 0; c < p.relu_consumers.size(); c++)
                {
                    ncnn::Layer* next = net.layers[p.relu_consumers[c]];
                    for (size_t b = 0; b < next->bottoms.size(); b++)
                        if (next->bottoms[b] == p.conv_top) next->bottoms[b] = p.relu_top;
                }
                net.blobs[p.conv_top].consumers.clear();
                net.blobs[p.conv_top].consumers.push_back(p.relu_idx);
                net.blobs[p.relu_top].consumers = p.relu_consumers;
            }
        }
    };

    auto set_config = [&](bool cand) {
        if (absw == "relu")
        {
            cudaDeviceSynchronize();
            set_relu_fused(cand);
        }
        else if (absw == "pool")
        {
            setenv("NCNN_CUDA_POOL_FUSE_PAD", cand ? "1" : "0", 1);
        }
        else
        {
            setenv("NCNN_CUDA_BINARYOP_VEC", cand ? "1" : "0", 1);
        }
    };

    auto median = [](std::vector<double> v) {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };

    set_config(false);
    cudaDeviceSynchronize();

    // 预热 30 张，两种配置交替。
    const int w = std::min(ab_warmup, batch_data.size());
    for (int i = 0; i < w; i++)
    {
        set_config((i & 1) != 0);
        if (run_one(i % batch_data.size()) != 0) fatal("warmup forward 失败");
        cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();
    g_blob_arena.set_mark();
    g_work_arena.set_mark();
    clear_blobs();

    printf("A/B switch=%s images=%d batch=%d rounds=%d warmup=%d sync_every=%d\n",
           ab_switch, ab_images, ab_batch, ab_rounds, ab_warmup, sync_every);

    const int start = ab_warmup;
    if (start + ab_images > batch_data.size()) fatal("--ab-images 超出数据范围");
    const int ab_check = std::min(128, ab_images);

    std::vector<std::vector<float> > logits_base(ab_check, std::vector<float>(10, 0.f));
    std::vector<std::vector<float> > logits_cand(ab_check, std::vector<float>(10, 0.f));
    std::vector<char> has_base(ab_check, 0), has_cand(ab_check, 0);
    std::vector<double> round_base(ab_rounds), round_cand(ab_rounds), round_impr(ab_rounds);
    std::vector<double> all_base, all_cand;
    long top1_base = 0, top1_cand = 0, n_base = 0, n_cand = 0;

    for (int r = 0; r < ab_rounds; r++)
    {
        std::vector<double> bms, cms;
        for (int b = 0; b * ab_batch < ab_images; b++)
        {
            const int base = start + b * ab_batch;
            const int cnt = std::min(ab_batch, ab_images - b * ab_batch);
            const bool cand = (((b + r) & 1) != 0);
            set_config(cand);
            cudaDeviceSynchronize();

            g_io_arena.reset_all();
            for (int j = 0; j < cnt; j++)
            {
                cifar10::Sample s = batch_data.at(base + j);
                ncnn::Mat in(cifar10::kWidth, cifar10::kHeight, 3);
                fill_input(in, s.pixels);
                cu_in[j] = ncnn::CudaMat(in, ioalloc);
            }
            cudaDeviceSynchronize();

            cudaEvent_t ev0, ev1;
            cudaEventCreate(&ev0);
            cudaEventCreate(&ev1);
            cudaEventRecord(ev0);
            if (net.forward_batch(blob_mats_gpu, in_idx, out_idx, out_producer,
                                  cu_in, cnt, cu_out, ioalloc, clear_blobs) != 0)
                fatal("forward_batch 失败");
            cudaEventRecord(ev1);
            cudaEventSynchronize(ev1);
            float ms = 0.f;
            cudaEventElapsedTime(&ms, ev0, ev1);
            const double per = ms / cnt;
            (cand ? cms : bms).push_back(per);
            (cand ? all_cand : all_base).push_back(per);
            cudaEventDestroy(ev0);
            cudaEventDestroy(ev1);

            for (int j = 0; j < cnt; j++)
            {
                ncnn::Mat out(cu_out[j]);
                const float* p = out;
                int best = 0;
                for (int k = 1; k < cifar10::kNumClasses; k++)
                    if (p[k] > p[best]) best = k;
                const int label = (int)batch_data.at(base + j).label;
                if (cand) { n_cand++; if (best == label) top1_cand++; }
                else { n_base++; if (best == label) top1_base++; }

                const int ci = base + j - start;
                if (ci >= 0 && ci < ab_check)
                {
                    if (cand && !has_cand[ci])
                    {
                        for (int k = 0; k < 10; k++) logits_cand[ci][k] = p[k];
                        has_cand[ci] = 1;
                    }
                    if (!cand && !has_base[ci])
                    {
                        for (int k = 0; k < 10; k++) logits_base[ci][k] = p[k];
                        has_base[ci] = 1;
                    }
                }
            }
            cudaDeviceSynchronize();
        }

        const double bmed = median(bms);
        const double cmed = median(cms);
        round_base[r] = bmed;
        round_cand[r] = cmed;
        round_impr[r] = bmed > 0.0 ? (bmed - cmed) / bmed * 100.0 : 0.0;
        printf("round %d: base=%.4f cand=%.4f impr=%+.2f%%\n", r + 1, bmed, cmed, round_impr[r]);
    }

    const double base_med = median(all_base);
    const double cand_med = median(all_cand);
    const double base_min = *std::min_element(all_base.begin(), all_base.end());
    const double cand_min = *std::min_element(all_cand.begin(), all_cand.end());
    const double med_impr = median(round_impr);

    std::mt19937 rng(12345);
    std::uniform_int_distribution<size_t> dist(0, round_impr.size() - 1);
    std::vector<double> boot;
    boot.reserve(10000);
    for (int it = 0; it < 10000; it++)
    {
        std::vector<double> s;
        s.reserve(round_impr.size());
        for (size_t k = 0; k < round_impr.size(); k++) s.push_back(round_impr[dist(rng)]);
        boot.push_back(median(s));
    }
    std::sort(boot.begin(), boot.end());
    const double ci_lo = boot[(size_t)(0.025 * boot.size())];
    const double ci_hi = boot[(size_t)(0.975 * boot.size())];

    int cmp_n = 0, diff_bits = 0;
    float max_abs = 0.f;
    for (int i = 0; i < ab_check; i++)
    {
        if (!has_base[i] || !has_cand[i]) continue;
        cmp_n++;
        for (int k = 0; k < 10; k++)
        {
            if (std::memcmp(&logits_base[i][k], &logits_cand[i][k], sizeof(float)) != 0) diff_bits++;
            const float d = fabsf(logits_base[i][k] - logits_cand[i][k]);
            if (d > max_abs) max_abs = d;
        }
    }

    printf("SUMMARY switch=%s base_med=%.4f base_min=%.4f cand_med=%.4f cand_min=%.4f "
           "med_impr=%.2f%% ci=[%.2f,%.2f] top1_base=%ld/%ld top1_cand=%ld/%ld "
           "logits_n=%d max_abs=%.9g differing=%d\n",
           ab_switch, base_med, base_min, cand_med, cand_min, med_impr, ci_lo, ci_hi,
           top1_base, n_base, top1_cand, n_cand, cmp_n, (double)max_abs, diff_bits);

    const bool recommend = (ci_lo > 0.0 && med_impr >= 2.0);
    printf("DECISION switch=%s %s (ci_lo=%.2f med_impr=%.2f%% rule=ci_lo>0 && med_impr>=2%%)\n",
           ab_switch, recommend ? "default_on" : "opt_in_off", ci_lo, med_impr);
    return 0;
}
'''


def repl(old, new):
    with io.open(PATH, "r", encoding="utf-8") as f:
        text = f.read()
    if old not in text:
        raise SystemExit("marker missing: %r" % old[:60])
    text = text.replace(old, new, 1)
    with io.open(PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


repl(INC_OLD, INC_NEW)
repl(ARG_OLD, ARG_NEW)
repl(PARSE_OLD, PARSE_NEW)

with io.open(PATH, "r", encoding="utf-8") as f:
    text = f.read()
pos = text.index("    // ---------- 预热 ----------")
text = text[:pos] + TAIL
with io.open(PATH, "w", encoding="utf-8", newline="\n") as f:
    f.write(text)
print("A/B harness built")
