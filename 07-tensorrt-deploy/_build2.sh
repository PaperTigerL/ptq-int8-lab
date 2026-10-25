#!/usr/bin/env bash
# 只编 library + custom layer registry，不编 tools（老 tools 在新 GCC 下缺 <cstdint>，用不上）。
# 注意：WSL 命令经 Git Bash 转发会吃掉 $VAR，所以必须写成脚本文件。
set -u
export PATH=/home/user/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64:${LD_LIBRARY_PATH:-}

cd /home/user/ncnn-cuda/build
cmake -DNCNN_BUILD_TOOLS=OFF .. 2>&1 | tail -5
echo "=== make ==="
make -j"$(nproc)" 2>&1 | tail -25
echo "=== 产物 ==="
ls -la src/libncnn.a 2>/dev/null
find . -name "libncnn*" | head
