#!/usr/bin/env python3
"""H-065: temporary env-gated debug print to prove the 2px dispatch fires."""
import io
import sys

CPP = "/home/user/ncnn-cuda/src/layer/cuda/convolution_cuda.cpp"
OLD = """            && dilation_w == 1 && dilation_h == 1 && OW_ >= 2)
        {
            implicit_conv_int8_launch_2px(
"""
NEW = """            && dilation_w == 1 && dilation_h == 1 && OW_ >= 2)
        {
            if (std::getenv("NCNN_CUDA_CONV2PX_DEBUG"))
                fprintf(stderr, "conv2px C=%d Cout=%d OW=%d OH=%d\\n",
                        C, num_output, OW_, OH_);
            implicit_conv_int8_launch_2px(
"""

t = io.open(CPP, encoding="utf-8").read()
if OLD not in t:
    sys.exit("marker missing")
io.open(CPP, "w", encoding="utf-8", newline="\n").write(t.replace(OLD, NEW, 1))
print("debug print added")
