#!/usr/bin/env python3
"""H-071: per-node timing of the captured graph (graph-form kernel profile).

Enable with NCNN_CUDA_GRAPH_NODEPROF=1. Each kernel node is isolated with
cudaGraphNodeSetEnabled on the instantiated graph and timed with events, so the
numbers are kernel-only GPU time with launch gaps removed.
"""
import io
import sys

CPP = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda/ncnn_cuda_graph_eval.cpp"

ANCHOR = """                        const char* dump_path = std::getenv("NCNN_CUDA_GRAPH_DUMP");
"""

BLOCK = """                        if (std::getenv("NCNN_CUDA_GRAPH_NODEPROF"))
                        {
                            size_t nn = 0;
                            cudaGraphGetNodes(graph, 0, &nn);
                            std::vector<cudaGraphNode_t> nodes(nn);
                            cudaGraphGetNodes(graph, nodes.data(), &nn);
                            printf("[nodeprof] nodes=%zu\\n", nn);
                            for (size_t i = 0; i < nn; i++)
                            {
                                for (size_t j = 0; j < nn; j++)
                                    cudaGraphNodeSetEnabled(exec, nodes[j], (j == i) ? 1u : 0u);
                                cudaGraphLaunch(exec, gs);
                                cudaStreamSynchronize(gs);

                                cudaGraphNodeType nt = cudaGraphNodeTypeEmpty;
                                cudaGraphNodeGetType(nodes[i], &nt);
                                long off = -1;
                                if (nt == cudaGraphNodeTypeKernel)
                                {
                                    cudaKernelNodeParams kp;
                                    memset(&kp, 0, sizeof(kp));
                                    if (cudaGraphKernelNodeGetParams(nodes[i], &kp) == cudaSuccess)
                                    {
                                        Dl_info info;
                                        if (dladdr((void*)kp.func, &info) && info.dli_fbase)
                                            off = (long)((const char*)kp.func - (const char*)info.dli_fbase);
                                    }
                                }

                                cudaEvent_t a, b;
                                cudaEventCreate(&a);
                                cudaEventCreate(&b);
                                cudaEventRecord(a, gs);
                                for (int r = 0; r < 20; r++) cudaGraphLaunch(exec, gs);
                                cudaEventRecord(b, gs);
                                cudaEventSynchronize(b);
                                float ms = 0.0f;
                                cudaEventElapsedTime(&ms, a, b);
                                printf("NODEPROF %zu %ld %.6f\\n", i, off, ms / 20.0);
                                cudaEventDestroy(a);
                                cudaEventDestroy(b);
                            }
                            for (size_t j = 0; j < nn; j++) cudaGraphNodeSetEnabled(exec, nodes[j], 1u);
                            cudaStreamDestroy(gs);
                            cudaGraphExecDestroy(exec);
                            cudaGraphDestroy(graph);
                            return 0;
                        }

"""

t = io.open(CPP, encoding="utf-8").read()
if ANCHOR not in t:
    sys.exit("anchor missing")
io.open(CPP, "w", encoding="utf-8", newline="\n").write(t.replace(ANCHOR, BLOCK + ANCHOR, 1))
print("nodeprof inserted")
