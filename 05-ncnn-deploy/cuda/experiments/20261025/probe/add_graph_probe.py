#!/usr/bin/env python3
"""Insert the H-048 CUDA Graph capture probe into the harness copy."""
import io
import sys

CPP = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda/ncnn_cuda_graph_eval.cpp"

OLD_INC = "#include <cuda_runtime.h>\n"
NEW_INC = "#include <cuda_runtime.h>\n\n#include <dlfcn.h>\n"

ANCHOR = """    printf("arena 水位线: blob %zu MB | workspace %zu MB\\n",
           g_blob_arena.bytes_used() >> 20, g_work_arena.bytes_used() >> 20);
"""

PROBE = ANCHOR + """
    // ---------- H-048: CUDA Graph 捕获探针（NCNN_CUDA_GRAPH=1） ----------
    // ncnn 的 kernel 全在默认流上发射，CUDA 不允许捕获默认流。这里用
    // graph_shim.so（LD_PRELOAD 拦截 cudaLaunchKernel）把发射临时改到捕获流。
    const char* graph_env = std::getenv("NCNN_CUDA_GRAPH");
    if (graph_env && graph_env[0] == '1')
    {
        typedef void (*shim_set_fn)(void*, int);
        shim_set_fn shim_set = (shim_set_fn)dlsym(RTLD_DEFAULT, "ncnn_graph_shim_set");
        if (!shim_set)
        {
            printf("[graph] 未加载 graph_shim.so，跳过\\n");
        }
        else
        {
            cifar10::Sample s0 = batch_data.at(0);
            float out_ref[10];
            {
                ncnn::Mat in(cifar10::kWidth, cifar10::kHeight, 3);
                fill_input(in, s0.pixels);
                clear_blobs();
                blob_mats_gpu[in_idx] = ncnn::CudaMat(in, ioalloc);
                if (net.forward_layer(out_producer, blob_mats_gpu, net.opt) != 0)
                    fatal("graph ref forward 失败");
                cudaDeviceSynchronize();
                cudaMemcpy(out_ref, blob_mats_gpu[out_idx].get_raw_data(),
                           sizeof(out_ref), cudaMemcpyDeviceToHost);
            }

            clear_blobs();
            ncnn::Mat in0(cifar10::kWidth, cifar10::kHeight, 3);
            fill_input(in0, s0.pixels);
            ncnn::CudaMat cu_in0(in0, ioalloc);
            blob_mats_gpu[in_idx] = cu_in0;
            cudaDeviceSynchronize();

            cudaStream_t gs = 0;
            cudaStreamCreateWithFlags(&gs, cudaStreamNonBlocking);
            shim_set((void*)gs, 1);
            cudaError_t brc = cudaStreamBeginCapture(gs, cudaStreamCaptureModeRelaxed);
            printf("[graph] beginCapture=%d (%s)\\n", (int)brc, cudaGetErrorString(brc));
            if (brc != cudaSuccess)
            {
                shim_set(0, 0);
                printf("[graph] 捕获不可用，回退普通路径\\n");
            }
            else
            {
                int frc = net.forward_layer(out_producer, blob_mats_gpu, net.opt);
                cudaGraph_t graph = 0;
                cudaError_t erc = cudaStreamEndCapture(gs, &graph);
                shim_set(0, 0);
                printf("[graph] forward=%d endCapture=%d (%s)\\n",
                       frc, (int)erc, cudaGetErrorString(erc));
                if (erc == cudaSuccess && graph)
                {
                    size_t nnodes = 0;
                    cudaGraphGetNodes(graph, 0, &nnodes);
                    printf("[graph] nodes=%zu\\n", nnodes);
                    void* cap_buf = blob_mats_gpu[out_idx].get_raw_data();
                    cudaGraphExec_t exec = 0;
                    cudaError_t ir = cudaGraphInstantiate(&exec, graph, 0, 0, 0);
                    printf("[graph] instantiate=%d (%s)\\n", (int)ir, cudaGetErrorString(ir));
                    if (ir == cudaSuccess)
                    {
                        cudaGraphLaunch(exec, gs);
                        cudaStreamSynchronize(gs);
                        float out_cap[10];
                        cudaMemcpy(out_cap, cap_buf, sizeof(out_cap), cudaMemcpyDeviceToHost);
                        double maxd = 0.0;
                        int neq = 0;
                        for (int k = 0; k < 10; k++)
                        {
                            double d = out_cap[k] - out_ref[k];
                            if (d < 0) d = -d;
                            if (d > maxd) maxd = d;
                            if (out_cap[k] != out_ref[k]) neq++;
                        }
                        printf("[graph] replay vs normal: max_abs=%.9g differing=%d\\n", maxd, neq);

                        cudaEvent_t e0, e1;
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
                    }
                    cudaGraphDestroy(graph);
                }
            }
            cudaStreamDestroy(gs);
        }
    }
"""

text = io.open(CPP, encoding="utf-8").read()
if NEW_INC not in text:
    if OLD_INC not in text:
        sys.exit("include marker missing")
    text = text.replace(OLD_INC, NEW_INC, 1)
if ANCHOR not in text:
    sys.exit("anchor missing")
text = text.replace(ANCHOR, PROBE, 1)
io.open(CPP, "w", encoding="utf-8", newline="\n").write(text)
print("probe inserted")
