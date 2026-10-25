#!/usr/bin/env bash
# 跑 CUDA Graph 版 TensorRT 评测（带非 graph 对照）。
#
# 注意：WSL 命令经 Git Bash 转发会吃掉 $VAR，所以必须写成脚本文件；
#      并且要用 `wsl.exe bash -lc 'bash <脚本>'` 调用。
set -u
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64:${LD_LIBRARY_PATH:-}
PY=/home/user/pyenv-torch/bin/python
HERE=/mnt/d/WorkBuddy/ShowBro/lab/07-tensorrt-deploy

MODEL=${1:-fp16}
BATCH=${2:-512}
LIMIT=${3:-0}
REPEAT=${4:-5}

LIM=""
if [ "$LIMIT" != "0" ]; then LIM="--limit $LIMIT"; fi

# 不等 GPU 空闲：这台机器上 GPU 常驻着 Windows 桌面进程（B站 / Edge WebView /
# StartMenu 等），wait_idle.py 会一直等到超时。跟 trt_eval.py 保持一致 ——
# 每轮打印占用，多轮取最优。
echo "===== CUDA Graph 评测（含非 graph 对照）====="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>&1 | head -1
"$PY" "$HERE/scripts/trt_eval_graph.py" \
    --model "$MODEL" --batch "$BATCH" --repeat "$REPEAT" $LIM --compare
