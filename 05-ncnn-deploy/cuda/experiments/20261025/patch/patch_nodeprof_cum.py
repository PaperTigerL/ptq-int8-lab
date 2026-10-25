#!/usr/bin/env python3
"""H-071: switch node profiling to cumulative-prefix timing.

Isolated single-node timing is dominated by the ~35us per-launch floor, so we
time prefix graphs (first k nodes enabled) and take differences. One graph
launch per replay is common to all k, so the difference is the kernel's own cost
in its real predecessor context.
"""
import io
import sys

CPP = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda/ncnn_cuda_graph_eval.cpp"

START = '                        if (std::getenv("NCNN_CUDA_GRAPH_NODEPROF"))\n'
END = '                        const char* dump_path = std::getenv("NCNN_CUDA_GRAPH_DUMP");\n'

NEW = r'''                        if (std::getenv("NCNN_CUDA_GRAPH_NODEPROF"))
                        {
                            size_t nn = 0;
                            cudaGraphGetNodes(graph, 0, &nn);
                            std::vector<cudaGraphNode_t> nodes(nn);
                            cudaGraphGetNodes(graph, nodes.data(), &nn);
                            printf("[nodeprof] nodes=%zu\n", nn);
                            std::vector<long> offs(nn, -1);
                            for (size_t i = 0; i < nn; i++)
                            {
                                cudaGraphNodeType nt = cudaGraphNodeTypeEmpty;
                                cudaGraphNodeGetType(nodes[i], &nt);
                                if (nt == cudaGraphNodeTypeKernel)
                                {
                                    cudaKernelNodeParams kp;
                                    memset(&kp, 0, sizeof(kp));
                                    if (cudaGraphKernelNodeGetParams(nodes[i], &kp) == cudaSuccess)
                                    {
                                        Dl_info info;
                                        if (dladdr((void*)kp.func, &info) && info.dli_fbase)
                                            offs[i] = (long)((const char*)kp.func - (const char*)info.dli_fbase);
                                    }
                                }
                                printf("NODE %zu %ld\n", i, offs[i]);
                            }
                            cudaEvent_t a, b;
                            cudaEventCreate(&a);
                            cudaEventCreate(&b);
                            for (size_t k = 0; k <= nn; k++)
                            {
                                for (size_t j = 0; j < nn; j++)
                                    cudaGraphNodeSetEnabled(exec, nodes[j], (j < k) ? 1u : 0u);
                                cudaGraphLaunch(exec, gs);
                                cudaStreamSynchronize(gs);
                                cudaEventRecord(a, gs);
                                for (int r = 0; r < 20; r++) cudaGraphLaunch(exec, gs);
                                cudaEventRecord(b, gs);
                                cudaEventSynchronize(b);
                                float ms = 0.0f;
                                cudaEventElapsedTime(&ms, a, b);
                                printf("CUM %zu %.6f\n", k, ms / 20.0);
                            }
                            cudaEventDestroy(a);
                            cudaEventDestroy(b);
                            for (size_t j = 0; j < nn; j++) cudaGraphNodeSetEnabled(exec, nodes[j], 1u);
                            cudaStreamDestroy(gs);
                            cudaGraphExecDestroy(exec);
                            cudaGraphDestroy(graph);
                            return 0;
                        }

'''

t = io.open(CPP, encoding="utf-8").read()
s = t.find(START)
e = t.find(END, s)
if s < 0 or e < 0:
    sys.exit("block not found")
t = t[:s] + NEW + t[e:]
io.open(CPP, "w", encoding="utf-8", newline="\n").write(t)
print("cumulative nodeprof installed")
