#!/usr/bin/env python3
"""Add an env-guarded split decision log (NCNN_CUDA_SPLIT_DEBUG)."""
import io

PATH = "/home/user/ncnn-cuda/src/layer/cuda/convolution_cuda.cpp"
OLD = """        if (split > 1)
        {
            const size_t partial_elems"""
NEW = """        if (std::getenv("NCNN_CUDA_SPLIT_DEBUG"))
            fprintf(stderr, "implicit split M=%d Cout=%d K=%d -> %d\\n",
                    M, num_output, C4 * kernel_h * kernel_w, split);

        if (split > 1)
        {
            const size_t partial_elems"""

with io.open(PATH, "r", encoding="utf-8") as f:
    text = f.read()
if OLD not in text:
    raise SystemExit("marker missing")
text = text.replace(OLD, NEW, 1)
with io.open(PATH, "w", encoding="utf-8", newline="\n") as f:
    f.write(text)
print("added split debug log")
