#!/usr/bin/env python3
"""H-048: replace the 50-iter replay probe with the formal graph measurement loop."""
import io
import sys

CPP = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda/ncnn_cuda_graph_eval.cpp"

OLD = """                        cudaEvent_t e0, e1;
                        cudaEventCreate(&e0);
                        cudaEventCreate(&e1);
                        cudaEventRecord(e0, gs);
                        for (int r = 0; r < 50; r++) cudaGraphLaunch(exec, gs);
                        cudaEventRecord(e1, gs);
                        cudaEventSynchronize(e1);
                        float gm = 0.0f;
                        cudaEventElapsedTime(&gm, e0, e1);
                        printf("[graph] replay %.4f ms/img (50 iters)\\n", gm / 50.0);
                        cudaEventDestroy(e0);
                        cudaEventDestroy(e1);
                        cudaGraphExecDestroy(exec);
"""

NEW = """                        // 正式测量：每张 H2D 到固定输入缓冲 -> graph 重放 -> D2H，
                        // 全部排在 gs 上，批次末同步一次；精度来自这条链的真实输出。
                        std::vector<float> outs((size_t)n * 10);
                        int gok = 0;
                        auto gw0 = std::chrono::steady_clock::now();
                        for (int j = 0; j < n; j++)
                        {
                            cifar10::Sample s = batch_data.at(j);
                            ncnn::Mat in(cifar10::kWidth, cifar10::kHeight, 3);
                            fill_input(in, s.pixels);
                            cudaMemcpyAsync(cu_in0.get_raw_data(), in.data, 12288,
                                            cudaMemcpyHostToDevice, gs);
                            cudaGraphLaunch(exec, gs);
                            cudaMemcpyAsync(&outs[(size_t)j * 10], cap_buf, 40,
                                            cudaMemcpyDeviceToHost, gs);
                        }
                        cudaStreamSynchronize(gs);
                        const double gwall = ms_of(gw0, std::chrono::steady_clock::now());
                        for (int j = 0; j < n; j++)
                        {
                            const float* p = &outs[(size_t)j * 10];
                            int best = 0;
                            for (int k = 1; k < cifar10::kNumClasses; k++)
                                if (p[k] > p[best]) best = k;
                            if (best == (int)batch_data.at(j).label) gok++;
                        }

                        cudaEvent_t e0, e1;
                        cudaEventCreate(&e0);
                        cudaEventCreate(&e1);
                        cudaEventRecord(e0, gs);
                        for (int j = 0; j < n; j++) cudaGraphLaunch(exec, gs);
                        cudaEventRecord(e1, gs);
                        cudaEventSynchronize(e1);
                        float gpu_total = 0.0f;
                        cudaEventElapsedTime(&gpu_total, e0, e1);
                        const double gpu_per = gpu_total / n;
                        const double wall_per = gwall / n;
                        cudaEventDestroy(e0);
                        cudaEventDestroy(e1);

                        printf("  round 1: GPU %.4f ms/img | 墙钟 %.4f ms/img "
                               "(H2D 0.0000 + D2H 0.0000) | 精度 %.2f%% | 显存已用 %.1f MB\\n",
                               gpu_per, wall_per, 100.0 * gok / n,
                               (double)(n * 12288) / 1048576.0);
                        printf("----------------------------------------\\n");
                        printf("精度 %.2f%% (%d/%d)\\n", 100.0 * gok / n, gok, n);
                        printf("最优 GPU kernel  %.4f ms/img  ->  %.1fx\\n", gpu_per, kBaselineMsPerImg / gpu_per);
                        printf("最优 含传输墙钟  %.4f ms/img  ->  %.1fx\\n", wall_per, kBaselineMsPerImg / wall_per);
                        cudaGraphExecDestroy(exec);
                        cudaGraphDestroy(graph);
                        cudaStreamDestroy(gs);
                        return 0;
"""

text = io.open(CPP, encoding="utf-8").read()
if OLD not in text:
    sys.exit("timing block marker missing")
text = text.replace(OLD, NEW, 1)
io.open(CPP, "w", encoding="utf-8", newline="\n").write(text)
print("formal graph loop inserted")
