#!/usr/bin/env bash
# push-20261025b 的单档 bench：等 GPU 空闲 → flock -x → 跑 → 释放。
# 与 scripts/push_one.sh 同纪律，只是把产物目录换到 results/push-20261025b
# （不覆盖 push-20261025 的既有 results）。
# usage: push_b_one.sh <onnx> <tag> [push_bench 额外参数...]
set -e
REPO=/mnt/d/WorkBuddy/ShowBro/lab/07-tensorrt-deploy
P=/home/user/pyenv-torch/bin/python
cd "$REPO"
export LD_LIBRARY_PATH=/usr/lib/wsl/lib
OUT=results/push-20261025b
ONNX="$1"; TAG="$2"; shift 2
$P scripts/wait_idle.py 2>&1 | tail -1
echo "[gpu] $($P -c 'import subprocess;print(subprocess.run(["nvidia-smi","--query-gpu=utilization.gpu,memory.used,clocks.sm","--format=csv,noheader"],capture_output=True,text=True).stdout.strip())')"
flock -x /tmp/ptq-gpu.lock -c "cd $REPO && LD_LIBRARY_PATH=/usr/lib/wsl/lib $P scripts/push_bench.py --onnx '$ONNX' --tag '$TAG' --jsonl $OUT/bench.jsonl --engine-dir work/push-20261025b/engines --timing-cache work/push-20261025b/timing.cache $*"
