#!/usr/bin/env python3
"""H-055: make CUDA Graph default-on in the eval harness, explicit off with
NCNN_CUDA_GRAPH=0, and log fallback reasons."""
import io
import sys

CPP = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda/ncnn_cuda_graph_eval.cpp"

OLD_ENV = """    const char* graph_env = std::getenv("NCNN_CUDA_GRAPH");
    if (graph_env && graph_env[0] == '1')
"""
NEW_ENV = """    const char* graph_env = std::getenv("NCNN_CUDA_GRAPH");
    // 默认开：只有显式 NCNN_CUDA_GRAPH=0 才关闭。未加载 shim、捕获失败或
    // instantiate 失败都会打印原因并回退普通路径（数值路径不变）。
    const bool graph_enabled = !(graph_env && graph_env[0] == '0' && graph_env[1] == '\\0');
    if (graph_enabled)
"""

OLD_END = """                if (erc == cudaSuccess && graph)
                {
"""
NEW_END = """                if (erc != cudaSuccess || !graph)
                    printf("[graph] 回退普通路径：endCapture 失败\\n");
                if (erc == cudaSuccess && graph)
                {
"""

OLD_INST = """                    printf("[graph] instantiate=%d (%s)\\n", (int)ir, cudaGetErrorString(ir));
                    if (ir == cudaSuccess)
"""
NEW_INST = """                    printf("[graph] instantiate=%d (%s)\\n", (int)ir, cudaGetErrorString(ir));
                    if (ir != cudaSuccess)
                        printf("[graph] 回退普通路径：instantiate 失败\\n");
                    if (ir == cudaSuccess)
"""

text = io.open(CPP, encoding="utf-8").read()
for old, new in ((OLD_ENV, NEW_ENV), (OLD_END, NEW_END), (OLD_INST, NEW_INST)):
    if old not in text:
        sys.exit("marker missing: %r" % old[:60])
    text = text.replace(old, new, 1)
io.open(CPP, "w", encoding="utf-8", newline="\n").write(text)
print("default-on patched")
