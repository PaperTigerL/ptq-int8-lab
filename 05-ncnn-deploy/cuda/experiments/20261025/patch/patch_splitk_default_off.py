#!/usr/bin/env python3
"""Default split-K off (rollback after no reliable gain); keep it opt-in.

NCNN_CUDA_SPLIT_HEURISTIC=1 -> per-layer occupancy heuristic
NCNN_CUDA_SPLIT=1|2|4       -> force a factor (1 = off)
NCNN_CUDA_SPLIT_MIN_K/MAX   -> cap the heuristic for experiments
NCNN_CUDA_SPLIT_DEBUG=1     -> print the per-layer decision
"""
import io

PATH = "/home/user/ncnn-cuda/src/layer/cuda/convolution_cuda.cpp"
OLD = """        const int Ktot = C4 * kernel_h * kernel_w;
        int split = implicit_conv_int8_recommend_split(M, num_output, C4, kernel_h, kernel_w);
        const char* min_k_env = std::getenv("NCNN_CUDA_SPLIT_MIN_K");
        if (min_k_env && Ktot < atoi(min_k_env)) split = 1;
        const char* max_env = std::getenv("NCNN_CUDA_SPLIT_MAX");
        if (max_env)
        {
            const int cap = atoi(max_env);
            if (cap >= 1 && split > cap) split = cap;
        }
        const char* split_env = std::getenv("NCNN_CUDA_SPLIT");"""
NEW = """        // 默认关闭：本机 A/B 未测到超过噪声的收益（见 T-20261024-07 报告）。
        // 需要时用 NCNN_CUDA_SPLIT_HEURISTIC=1 开占用率启发式，或直接 force。
        static const bool split_heuristic = std::getenv("NCNN_CUDA_SPLIT_HEURISTIC") != nullptr;
        static const char* min_k_env = std::getenv("NCNN_CUDA_SPLIT_MIN_K");
        static const char* max_env = std::getenv("NCNN_CUDA_SPLIT_MAX");
        static const char* split_env = std::getenv("NCNN_CUDA_SPLIT");
        static const bool split_debug = std::getenv("NCNN_CUDA_SPLIT_DEBUG") != nullptr;

        const int Ktot = C4 * kernel_h * kernel_w;
        int split = 1;
        if (split_heuristic)
            split = implicit_conv_int8_recommend_split(M, num_output, C4, kernel_h, kernel_w);
        if (min_k_env && Ktot < atoi(min_k_env)) split = 1;
        if (max_env)
        {
            const int cap = atoi(max_env);
            if (cap >= 1 && split > cap) split = cap;
        }"""
NEW2 = """        if (split_debug)"""

with io.open(PATH, "r", encoding="utf-8") as f:
    text = f.read()
if OLD not in text:
    raise SystemExit("marker missing")
text = text.replace(OLD, NEW, 1)
# The debug print used a direct getenv; switch it to the cached flag.
text = text.replace('        if (std::getenv("NCNN_CUDA_SPLIT_DEBUG"))', NEW2, 1)
with io.open(PATH, "w", encoding="utf-8", newline="\n") as f:
    f.write(text)
print("split-K default off (opt-in)")
