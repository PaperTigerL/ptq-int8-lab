#!/usr/bin/env python3
"""Make BinaryOp float4 opt-in: NCNN_CUDA_BINARYOP_VEC=1 enables it."""
import io

PATH = "/home/user/ncnn-cuda/src/layer/cuda/binaryop_cuda.cu"
OLD = """            const char* vec_env = std::getenv("NCNN_CUDA_BINARYOP_VEC");
            const bool vec_off = vec_env && vec_env[0] == '0' && vec_env[1] == '\\0';
            const bool aligned = (((size_t)a_input | (size_t)b_input | (size_t)c_output) & 15u) == 0;
            if (contiguous && !vec_off && aligned)"""
NEW = """            const char* vec_env = std::getenv("NCNN_CUDA_BINARYOP_VEC");
            // 默认关闭；NCNN_CUDA_BINARYOP_VEC=1 开启（本单按 opt-in 交付）。
            const bool vec_on = vec_env && vec_env[0] == '1' && vec_env[1] == '\\0';
            const bool aligned = (((size_t)a_input | (size_t)b_input | (size_t)c_output) & 15u) == 0;
            if (contiguous && vec_on && aligned)"""

with io.open(PATH, "r", encoding="utf-8") as f:
    text = f.read()
if OLD not in text:
    raise SystemExit("marker missing")
text = text.replace(OLD, NEW, 1)
with io.open(PATH, "w", encoding="utf-8", newline="\n") as f:
    f.write(text)
print("BinaryOp vec default off")
