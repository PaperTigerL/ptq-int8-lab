#!/usr/bin/env python3
"""H-074: flip fused quant+pack to default-on (NCNN_CUDA_FUSE_QUANT_PACK=0 disables)."""
import io
import sys

CPP = "/home/user/ncnn-cuda/src/layer/cuda/convolution_cuda.cpp"
OLD = """    // opt-in：float 输入先量化再 pack 的两颗 kernel 合成一颗（逐位一致）。
    static const bool fuse_quant_pack = []() {
        const char* e = std::getenv("NCNN_CUDA_FUSE_QUANT_PACK");
        return e && e[0] == '1' && e[1] == '\\0';
    }();
"""
NEW = """    // 默认开：float 输入先量化再 pack 的两颗 kernel 合成一颗（逐位一致）。
    // NCNN_CUDA_FUSE_QUANT_PACK=0 可显式关闭。
    static const bool fuse_quant_pack = []() {
        const char* e = std::getenv("NCNN_CUDA_FUSE_QUANT_PACK");
        return !(e && e[0] == '0' && e[1] == '\\0');
    }();
"""
t = io.open(CPP, encoding="utf-8").read()
if OLD not in t:
    sys.exit("marker missing")
io.open(CPP, "w", encoding="utf-8", newline="\n").write(t.replace(OLD, NEW, 1))
print("flipped to default-on")
