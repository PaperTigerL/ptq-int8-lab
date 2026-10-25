#!/usr/bin/env bash
# 收口验收：终版 A/B（pinned 输入后的 cublas vs 按层混合）+ verify.sh
# （探针+单层+W8/W6/W4 全量）+ verify_e2e.sh（numpy 独立参考）。
# 由外部 flock 包裹。
set -u
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$HOME/libtorch/lib:$HOME/cuda/lib64
echo "==== nvidia-smi ===="
nvidia-smi --query-gpu=clocks.current.graphics,memory.used --format=csv,noheader
echo ""
echo "########## A/B (pinned input, batch=512) ##########"
bash _run_ab3.sh
echo ""
echo "########## VERIFY ##########"
bash verify.sh
echo ""
bash verify_e2e.sh
echo ""
echo "==== nvidia-smi ===="
nvidia-smi --query-gpu=clocks.current.graphics --format=csv,noheader
