#!/usr/bin/env bash
# work/ 里的中间产物（ONNX / engine plan）之前没入库、已经不在了，
# 这里重新导出 ONNX 再跑 CUDA Graph 评测。
#
# 注意：WSL 命令经 Git Bash 转发会吃掉 $VAR，必须写成脚本文件。
set -u
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64:${LD_LIBRARY_PATH:-}
PY=/home/user/pyenv-torch/bin/python
HERE=/mnt/d/WorkBuddy/ShowBro/lab/07-tensorrt-deploy

MODEL=${1:-fp16}
BATCH=${2:-512}
LIMIT=${3:-2048}
REPEAT=${4:-3}

echo "===== 1. 导出 ONNX ====="
"$PY" "$HERE/scripts/export_onnx.py" "$MODEL" 2>&1 | tail -5

echo
echo "===== 2. CUDA Graph 评测（含非 graph 对照）====="
LIM=""
if [ "$LIMIT" != "0" ]; then LIM="--limit $LIMIT"; fi
"$PY" "$HERE/scripts/trt_eval_graph.py" \
    --model "$MODEL" --batch "$BATCH" --repeat "$REPEAT" $LIM --compare
