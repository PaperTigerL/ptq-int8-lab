#!/usr/bin/env python3
"""Add an env-guarded debug line for the BinaryOp vector path."""
import io

PATH = "/home/user/ncnn-cuda/src/layer/cuda/binaryop_cuda.cu"
OLD = """            if (contiguous && !vec_off && aligned)
            {
                const int total = a_info.cstep * a_info.c;
                const int n4 = total / 4;"""
NEW = """            if (contiguous && !vec_off && aligned)
            {
                const int total = a_info.cstep * a_info.c;
                if (std::getenv("NCNN_CUDA_BINARYOP_VEC_DEBUG"))
                    fprintf(stderr, "binop vec4 %dx%dx%d cstep=%d/%d/%d total=%d\\n",
                            a_info.w, a_info.h, a_info.c, a_info.cstep, b_info.cstep, c_info.cstep, total);
                const int n4 = total / 4;"""

with io.open(PATH, "r", encoding="utf-8") as f:
    text = f.read()
if OLD not in text:
    raise SystemExit("marker missing")
text = text.replace(OLD, NEW, 1)
with io.open(PATH, "w", encoding="utf-8", newline="\n") as f:
    f.write(text)
print("debug line added")
