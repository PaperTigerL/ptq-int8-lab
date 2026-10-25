#!/usr/bin/env python3
"""Flip ReLU/Clip epilogue fusion to default-on; NCNN_CUDA_FUSE_ACT=0 disables."""
import io

PATH = "/home/user/ncnn-cuda/src/net.cpp"
OLD = """    const char* fuse_act_env = getenv("NCNN_CUDA_FUSE_ACT");
    // 默认关闭；NCNN_CUDA_FUSE_ACT=1 开启（本单按 opt-in 交付）。
    if (!fuse_act_env || !(fuse_act_env[0] == '1' && fuse_act_env[1] == '\\0'))
        return 0;"""
NEW = """    const char* fuse_act_env = getenv("NCNN_CUDA_FUSE_ACT");
    // 默认开启（T-16 8 轮交错 A/B：+12.30%，CI [2.94,14.30]）；
    // NCNN_CUDA_FUSE_ACT=0 显式关闭，回退到独立 ReLU/Clip kernel。
    if (fuse_act_env && fuse_act_env[0] == '0' && fuse_act_env[1] == '\\0')
        return 0;"""

with io.open(PATH, "r", encoding="utf-8") as f:
    text = f.read()
if OLD not in text:
    raise SystemExit("marker missing")
text = text.replace(OLD, NEW, 1)
with io.open(PATH, "w", encoding="utf-8", newline="\n") as f:
    f.write(text)
print("fusion default on")
