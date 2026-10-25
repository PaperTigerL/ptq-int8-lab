#!/usr/bin/env python3
"""Replace the measurement tail of ncnn_cuda_profile_layers.cpp with a
per-layer CUDA-event profiling loop. Run once after copying the eval harness."""
import io

PATH = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda/ncnn_cuda_profile_layers.cpp"
MARK = "    // ---------- 预热 ----------"

NEW_TAIL = r'''    // ---------- 预热（与验收 harness 相同的单次递归路径）----------
    const int w = std::min(warmup, n);
    for (int i = 0; i < w; i++)
    {
        if (run_one(i % n) != 0) fatal("forward 失败");
    }
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) fatal("预热阶段 CUDA 报错");

    g_blob_arena.set_mark();
    g_work_arena.set_mark();
    clear_blobs();
    printf("arena 水位线: blob %zu MB | workspace %zu MB\n",
           g_blob_arena.bytes_used() >> 20, g_work_arena.bytes_used() >> 20);

    // ---------- 逐层 profile ----------
    // 按 param 的拓扑顺序逐层驱动 forward_layer；每个 layer 前后各记一个
    // CUDA event，整图结束后一次性同步再求差。事件只记录不阻塞，
    // 因此不会把各层串行化；每图一次同步本来就在 clear_blobs 里。
    const int L = (int)net.layers.size();
    printf("== layer profile: %d images, %d layers ==\n", n, L);

    std::vector<cudaEvent_t> ev_start(L), ev_end(L);
    for (int i = 0; i < L; i++)
    {
        cudaEventCreate(&ev_start[i]);
        cudaEventCreate(&ev_end[i]);
    }
    std::vector<double> layer_ms(L, 0.0);
    int ok = 0;
    double h2d_ms = 0.0;
    double d2h_ms = 0.0;

    auto wall0 = std::chrono::steady_clock::now();
    for (int idx = 0; idx < n; idx++)
    {
        cifar10::Sample s = batch_data.at(idx);

        auto t0 = std::chrono::steady_clock::now();
        ncnn::Mat in(cifar10::kWidth, cifar10::kHeight, 3);
        fill_input(in, s.pixels);
        blob_mats_gpu[out_idx].release();
        blob_mats_gpu[in_idx] = ncnn::CudaMat(in, alloc);
        cudaDeviceSynchronize();
        h2d_ms += ms_of(t0, std::chrono::steady_clock::now());

        // layer 0 是 Input：harness 已直接写入 blob，跳过。
        for (int i = 1; i < L; i++)
        {
            cudaEventRecord(ev_start[i]);
            if (net.forward_layer(i, blob_mats_gpu, net.opt) != 0)
                fatal("forward_layer 失败");
            cudaEventRecord(ev_end[i]);
        }
        cudaEventSynchronize(ev_end[L - 1]);
        for (int i = 1; i < L; i++)
        {
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, ev_start[i], ev_end[i]);
            layer_ms[i] += ms;
        }
        if (cudaGetLastError() != cudaSuccess) fatal("逐层推理阶段 CUDA 报错");

        auto t2 = std::chrono::steady_clock::now();
        ncnn::Mat out(blob_mats_gpu[out_idx]);
        const float* p = out;
        int best = 0;
        for (int k = 1; k < cifar10::kNumClasses; k++)
            if (p[k] > p[best]) best = k;
        if (best == (int)batch_data.at(idx).label) ok++;
        cudaDeviceSynchronize();
        d2h_ms += ms_of(t2, std::chrono::steady_clock::now());

        clear_blobs();
    }
    double wall_total = ms_of(wall0, std::chrono::steady_clock::now());

    double total_layer_ms = 0.0;
    for (int i = 1; i < L; i++) total_layer_ms += layer_ms[i];

    printf("精度 %.2f%% (%d/%d)\n", 100.0 * ok / n, ok, n);
    printf("total_layer_ms_per_img %.4f\n", total_layer_ms / n);
    printf("h2d_ms_per_img %.4f\n", h2d_ms / n);
    printf("d2h_ms_per_img %.4f\n", d2h_ms / n);
    printf("wall_ms_per_img %.4f\n", wall_total / n);

    // 按类型汇总
    std::map<std::string, double> by_type;
    for (int i = 1; i < L; i++)
        by_type[net.layers[i]->type] += layer_ms[i];
    printf("== by layer type (ms/img, pct of layer total) ==\n");
    for (std::map<std::string, double>::const_iterator it = by_type.begin(); it != by_type.end(); ++it)
    {
        printf("TYPE %-20s %8.4f %6.2f%%\n", it->first.c_str(), it->second / n,
               100.0 * it->second / total_layer_ms);
    }

    // 单层明细，按耗时降序
    std::vector<int> order;
    for (int i = 1; i < L; i++)
        if (layer_ms[i] > 0.0) order.push_back(i);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return layer_ms[a] > layer_ms[b]; });
    printf("== layers (ms/img, pct of layer total) ==\n");
    for (size_t k = 0; k < order.size(); k++)
    {
        const int i = order[k];
        printf("LAYER %3d %-20s %-34s %8.4f %6.2f%%\n", i, net.layers[i]->type,
               net.layers[i]->name.c_str(), layer_ms[i] / n,
               100.0 * layer_ms[i] / total_layer_ms);
    }

    printf("arena_used_mb blob=%.1f work=%.1f io=%.1f\n",
           g_blob_arena.bytes_used() / 1048576.0,
           g_work_arena.bytes_used() / 1048576.0,
           g_io_arena.bytes_used() / 1048576.0);
    return 0;
}
'''


def main():
    with io.open(PATH, "r", encoding="utf-8") as f:
        text = f.read()
    pos = text.index(MARK)
    out = text[:pos] + NEW_TAIL
    with io.open(PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(out)
    print("patched", PATH, "at", pos)


if __name__ == "__main__":
    main()
