#!/usr/bin/env bash
# 打补丁 + 编译 ncnn-with-cuda。
# 注意：WSL 命令经 Git Bash 转发会吃掉 $VAR 并且引号嵌套容易炸，所以一律写成脚本文件。
set -u
export PATH=/home/user/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64:${LD_LIBRARY_PATH:-}

python3 /mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda/patch_ncnn_cuda.py

cd /home/user/ncnn-cuda/build
make -j"$(nproc)" 2>&1 | grep -E "error|Error|Built target ncnn$" | tail -15
echo "=== 产物 ==="
find /home/user/ncnn-cuda/build -name "libncnn.a" | head
