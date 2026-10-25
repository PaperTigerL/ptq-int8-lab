#!/usr/bin/env python3
"""Make the activation fusion opt-in: NCNN_CUDA_FUSE_ACT=1 enables it."""
import io

PATH = "/home/user/ncnn-cuda/src/net.cpp"
OLD = """    const char* fuse_act_env = getenv("NCNN_CUDA_FUSE_ACT");
    if (fuse_act_env && fuse_act_env[0] == '0' && fuse_act_env[1] == '\\0')
        return 0;"""
NEW = """    const char* fuse_act_env = getenv("NCNN_CUDA_FUSE_ACT");
    // 默认关闭；NCNN_CUDA_FUSE_ACT=1 开启（本单按 opt-in 交付）。
    if (!fuse_act_env || !(fuse_act_env[0] == '1' && fuse_act_env[1] == '\\0'))
        return 0;"""

with io.open(PATH, "r", encoding="utf-8") as f:
    text = f.read()
if OLD not in text:
    raise SystemExit("marker missing")
text = text.replace(OLD, NEW, 1)
with io.open(PATH, "w", encoding="utf-8", newline="\n") as f:
    f.write(text)
print("fusion default off")
