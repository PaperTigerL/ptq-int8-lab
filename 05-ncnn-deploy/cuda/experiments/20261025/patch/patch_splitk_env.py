#!/usr/bin/env python3
"""Add split-K tuning knobs: NCNN_CUDA_SPLIT_MAX and NCNN_CUDA_SPLIT_MIN_K.

NCNN_CUDA_SPLIT=1|2|4 still force-overrides everything (for experiments).
"""
import io

PATH = "/home/user/ncnn-cuda/src/layer/cuda/convolution_cuda.cpp"
OLD = """        int split = implicit_conv_int8_recommend_split(M, num_output, C4, kernel_h, kernel_w);
        const char* split_env = std::getenv("NCNN_CUDA_SPLIT");"""
NEW = """        const int Ktot = C4 * kernel_h * kernel_w;
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

with io.open(PATH, "r", encoding="utf-8") as f:
    text = f.read()
if OLD not in text:
    raise SystemExit("marker missing")
text = text.replace(OLD, NEW, 1)
with io.open(PATH, "w", encoding="utf-8", newline="\n") as f:
    f.write(text)
print("added split tuning knobs")
