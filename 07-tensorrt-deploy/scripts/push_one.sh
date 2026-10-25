#!/usr/bin/env bash
# 跑一档 bench：先等 GPU 空闲，再拿共享锁（一次只跑一个重活），跑完释放。
# usage: push_one.sh <onnx> <batch> <tag> [bench 额外参数...]
set -e
REPO=/mnt/d/WorkBuddy/ShowBro/lab/07-tensorrt-deploy
P=/home/user/pyenv-torch/bin/python
cd "$REPO"
export LD_LIBRARY_PATH=/usr/lib/wsl/lib
ONNX="$1"; BATCH="$2"; TAG="$3"; shift 3
$P scripts/wait_idle.py 2>&1 | tail -1
echo "[gpu] $($P -c 'import subprocess;print(subprocess.run(["nvidia-smi","--query-gpu=utilization.gpu,memory.used","--format=csv,noheader"],capture_output=True,text=True).stdout.strip())')"
flock -x /tmp/ptq-gpu.lock -c "cd $REPO && LD_LIBRARY_PATH=/usr/lib/wsl/lib $P scripts/push_bench.py --onnx '$ONNX' --batch $BATCH --tag '$TAG' $*"
