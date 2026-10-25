// 验证单层 batch int8 前向：输出是否一致、速度是否更快。
//
// 只测一层，因为整个网络的 batch 还需要 ReLU/Pooling/Add 也支持 batch，
// 那是下一步。先把"conv 层 batch 本身对不对、值不值"这件事钉死。
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "layer.h"
#include "layer/x86/convolution_x86.h"
#include "net.h"


static int outw_dummy(ncnn::Convolution_x86* c)
{
    const int ke = c->dilation_w * (c->kernel_w - 1) + 1;
    return (32 + 2 * c->pad_left - ke) / c->stride_w + 1;
}
static int outh_dummy(ncnn::Convolution_x86* c)
{
    const int ke = c->dilation_h * (c->kernel_h - 1) + 1;
    return (32 + 2 * c->pad_top - ke) / c->stride_h + 1;
}

int main(int argc, char** argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s <param> <bin> [batch] [target_layer]\n", argv[0]); return 2; }

    const int n_batch = (argc > 3) ? atoi(argv[3]) : 8;
    const char* target = (argc > 4) ? argv[4] : "layer1.0.conv1";

    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    net.opt.lightmode = false;
    net.opt.num_threads = 1;
    // 必须走 im2col sgemm，winograd/packed 路径没有 batch 版
    net.opt.use_int8_inference = true;
    net.opt.use_sgemm_convolution = true;
    net.opt.use_winograd_convolution = false;
    net.opt.use_packing_layout = false;

    if (net.load_param(argv[1]) != 0) return 1;
    if (net.load_model(argv[2]) != 0) return 1;

    // 找目标层
    ncnn::Convolution_x86* c = 0;
    for (size_t i = 0; i < net.layers().size(); i++)
    {
        ncnn::Layer* l = net.layers()[i];
        if (l->type != "Convolution") continue;
        if (strstr(l->name.c_str(), target) != 0) { c = (ncnn::Convolution_x86*)l; break; }
    }
    if (!c) { fprintf(stderr, "找不到层 %s\n", target); return 1; }

    fprintf(stderr, "目标层：%s  输入 %d 通道  kernel %dx%d  stride %d  int8_scale_term=%d\n",
            c->name.c_str(), c->weight_data_size / c->num_output / (c->kernel_w * c->kernel_h),
            c->kernel_w, c->kernel_h, c->stride_w, c->int8_scale_term);
    fprintf(stderr, "opt: sgemm=%d winograd=%d packing=%d | weight_sgemm_data=%d  weight_data_tm=%d\n",
            (int)net.opt.use_sgemm_convolution, (int)net.opt.use_winograd_convolution,
            (int)net.opt.use_packing_layout, (int)c->weight_sgemm_data.total(), (int)c->weight_data_tm.total());

    if (c->weight_sgemm_data.empty())
    {
        fprintf(stderr, "[FAIL] 这层没走 sgemm 路径（weight_sgemm_data 为空），batch 版不可用\n");
        return 3;
    }
    if (c->int8_scale_term <= 100)
    {
        fprintf(stderr, "[FAIL] 这层不是 fused requantize（int8_scale_term=%d）\n", c->int8_scale_term);
        return 3;
    }

    // 造 N 张输入（in_ch, 32, 32），用固定值便于对比
    const int in_ch = c->weight_data_size / c->num_output / (c->kernel_w * c->kernel_h);
    const int in_w = 32, in_h = 32;

    std::vector<ncnn::Mat> ins(n_batch);
    for (int b = 0; b < n_batch; b++)
    {
        ins[b].create(in_w, in_h, in_ch, (size_t)4u, 1);
        for (int q = 0; q < in_ch; q++)
        {
            float* p = ins[b].channel(q);
            for (int i = 0; i < in_w * in_h; i++)
                p[i] = std::sin((float)(i + q * 7 + b * 13)) * 0.5f + 0.5f;
        }
    }

    ncnn::Option opt = net.opt;

    // --- 逐张（同时导出 int32）---
    const int n32 = outw_dummy(c) * outh_dummy(c) * c->num_output;
    std::vector<int> i32_single(n32, -12345), i32_batch(n32, -12345);
    std::vector<ncnn::Mat> outs1(n_batch);

    ncnn::Convolution::g_ptq_dump_int32 = i32_single.data();
    ncnn::Convolution::g_ptq_dump_cap = n32;
    std::vector<signed char> pad_single(4096, -99), pad_batch(4096, -99);

    ncnn::Convolution::g_ptq_dump_padded = pad_single.data();

    auto t0 = std::chrono::steady_clock::now();
    for (int b = 0; b < n_batch; b++)
        c->forward(ins[b], outs1[b], opt);
    auto t1 = std::chrono::steady_clock::now();

    // --- batch（同样导出，只取第一张）---
    std::vector<ncnn::Mat> outs2;
    ncnn::Convolution::g_ptq_dump_int32 = i32_batch.data();
    ncnn::Convolution::g_ptq_dump_padded = pad_batch.data();

    auto t2 = std::chrono::steady_clock::now();
    int ret = c->forward_batch_int8(ins, outs2, opt);
    auto t3 = std::chrono::steady_clock::now();
    ncnn::Convolution::g_ptq_dump_int32 = 0;
    ncnn::Convolution::g_ptq_dump_padded = 0;

    // padded 对比（GEMM 的输入）
    {
        long long pbad = 0, pfirst = -1;
        for (int i = 0; i < 4096; i++)
        {
            if (pad_single[i] != pad_batch[i])
            {
                pbad++;
                if (pfirst < 0) pfirst = i;
            }
        }
        printf("padded 对比: 4096 字节，不一致 %lld，首个 %lld\n", pbad, pfirst);
        printf("  single[0..5] = %d,%d,%d,%d,%d,%d\n",
               pad_single[0], pad_single[1], pad_single[2], pad_single[3], pad_single[4], pad_single[5]);
        printf("  batch [0..5] = %d,%d,%d,%d,%d,%d\n",
               pad_batch[0], pad_batch[1], pad_batch[2], pad_batch[3], pad_batch[4], pad_batch[5]);
    }

    // int32 对比（第一张）
    {
        long long bad = 0, first = -1;
        // 单张 dump 来自 top_blob_int32（pack8: c=8, ep=8, cstep=1024）
        // batch dump 来自 top32[0]（pack1: c=64, ep=1）
        // 必须按 (channel, spatial) 逻辑位置比，线性比毫无意义。
        for (int ch = 0; ch < c->num_output; ch++)
        {
            for (int sp = 0; sp < 1024; sp++)
            {
                const int v_single = i32_single[(ch / 8) * 8192 + sp * 8 + (ch % 8)];
                const int v_batch = i32_batch[(ch / 8) * 8192 + sp * 8 + (ch % 8)];
                if (v_single != v_batch)
                {
                    bad++;
                    if (first < 0) { first = ch * 1024 + sp; printf("  首个不一致: ch=%d sp=%d single=%d batch=%d\n", ch, sp, v_single, v_batch); }
                }
            }
        }
        printf("int32 对比: 共 %d 个，不一致 %lld (%.2f%%), 首个 %lld\n",
               n32, bad, 100.0 * bad / n32, first);
        printf("  样本 [0..3] single=%d,%d,%d,%d  batch=%d,%d,%d,%d\n",
               i32_single[0], i32_single[1], i32_single[2], i32_single[3],
               i32_batch[0], i32_batch[1], i32_batch[2], i32_batch[3]);
    }

    if (ret != 0)
    {
        fprintf(stderr, "[FAIL] forward_batch_int8 返回 %d\n", ret);
        return 3;
    }

    const double ms_single = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double ms_batch = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // 对比输出。
    // 注意：单张 forward 输出 pack8（num_output%8==0 时），batch 版输出 pack1。
    // total() 相同但内存布局不同（pack8 是跨通道交错），必须先统一到 pack1，
    // 否则逐位比较必然不等 —— 这不是算法错，是布局差异。
    double max_diff = 0.0;
    long long n_elem = 0;
    long long n_bad = 0;
    long long first_bad = -1;
    for (int b = 0; b < n_batch; b++)
    {
        ncnn::Mat a1, d1;
        ncnn::convert_packing(outs1[b], a1, 1);
        ncnn::convert_packing(outs2[b], d1, 1);

        if (a1.total() != d1.total()) { fprintf(stderr, "[FAIL] 形状不一致 %d vs %d\n", (int)a1.total(), (int)d1.total()); return 3; }

        if (b == 0)
            fprintf(stderr, "布局: forward c=%d ep=%d total=%d | batch c=%d ep=%d total=%d | 转换后 c=%d ep=%d\n",
                    outs1[0].c, outs1[0].elempack, (int)outs1[0].total(),
                    outs2[0].c, outs2[0].elempack, (int)outs2[0].total(),
                    a1.c, a1.elempack);

        const signed char* a = a1;
        const signed char* d = d1;
        for (size_t i = 0; i < a1.total(); i++)
        {
            const double diff = std::fabs((double)a[i] - (double)d[i]);
            if (diff > max_diff) max_diff = diff;
            if (a[i] != d[i])
            {
                n_bad++;
                if (first_bad < 0) first_bad = (long long)i;
            }
            n_elem++;
        }
    }

    printf("\nbatch=%d  逐张 %.3f ms  batch %.3f ms  加速 %.2fx\n",
           n_batch, ms_single, ms_batch, ms_single / ms_batch);
    printf("输出元素 %lld 个，最大偏差 %.0f，不一致 %lld (%.2f%%)\n",
           n_elem, max_diff, n_bad, n_elem ? 100.0 * n_bad / n_elem : 0.0);
    if (first_bad >= 0)
        printf("首个不一致下标 %lld（每通道 %d 个元素，%d 通道）\n",
               first_bad, (int)(outs1[0].total() / outs1[0].c * outs1[0].elempack), outs1[0].c);
    printf("%s\n", max_diff == 0.0 ? "结论：输出完全一致" : "结论：有偏差，batch 实现有 bug");

    return max_diff == 0.0 ? 0 : 4;
}
