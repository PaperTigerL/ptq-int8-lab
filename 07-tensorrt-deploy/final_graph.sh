#!/usr/bin/env bash
# CUDA Graph 正式实测（全量 10000 张，含非 graph 对照），结果落到 results/。
#
# 注意：WSL 命令经 Git Bash 转发会吃掉 $VAR，必须写成脚本文件；
#      并且要用 `wsl.exe bash -lc 'bash <脚本>'` 调用。
set -u
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64:${LD_LIBRARY_PATH:-}
PY=/home/user/pyenv-torch/bin/python
HERE=/mnt/d/WorkBuddy/ShowBro/lab/07-tensorrt-deploy
OUT=$HERE/results

mkdir -p "$OUT"
{
  echo "# cudagraph $(date '+%Y-%m-%d %H:%M:%S')  fp16 batch=512 全量 10000 张 repeat=5"
  echo "# 口径：只计推理（graph replay / execute_async_v3）+ 同步，不含 H2D/D2H"
  echo "# 基线 11.6 ms/img = 1x"
  echo
} > "$OUT/cudagraph.txt"

"$PY" "$HERE/scripts/trt_eval_graph.py" \
    --model fp16 --batch 512 --repeat 5 --compare 2>&1 | tee -a "$OUT/cudagraph.txt" | tail -30
