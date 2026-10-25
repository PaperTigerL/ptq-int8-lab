#!/usr/bin/env bash
# ncnn-with-cuda 用的是 -fno-exceptions（移动端体积考虑），但这个分支在
# innerproduct_cuda.cpp / padding_cuda.cpp 里写了 throw，所以必须加 -fexceptions 才能编过。
# 那两个 throw 分支在我们的模型上都到不了（fc 是 512*10 < 65000，ResNet 也不用 Padding 的 int8 路径）。
set -u
export PATH=/home/user/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64:${LD_LIBRARY_PATH:-}

cd /home/user/ncnn-cuda/build
cmake -DNCNN_BUILD_TOOLS=OFF -DCMAKE_CXX_FLAGS="-fexceptions" .. 2>&1 | tail -3
echo "=== make ==="
make -j"$(nproc)" 2>&1 | grep -E "error|Error|\] Linking|\] Built target ncnn$" | tail -20
echo "=== 产物 ==="
find /home/user/ncnn-cuda/build -name "libncnn*" | head
