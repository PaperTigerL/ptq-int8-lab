#!/usr/bin/env python3
import io
import sys
CPP = "/home/user/ncnn-cuda/src/layer/cuda/convolution_cuda.cpp"
OLD = """            if (conv_border)
            {
                const size_t unpadded_elems = (size_t)C4 * H * W * 4;
"""
NEW = """            if (conv_border)
            {
                const float* act_params = (activation_type == 3 && !gpu_activation_params.empty())
                    ? static_cast<const float*>(gpu_activation_params.get_craw_data()) : nullptr;
                const size_t unpadded_elems = (size_t)C4 * H * W * 4;
"""
t = io.open(CPP, encoding="utf-8").read()
if OLD not in t:
    sys.exit("marker missing")
io.open(CPP, "w", encoding="utf-8", newline="\n").write(t.replace(OLD, NEW, 1))
print("act_params fixed")
