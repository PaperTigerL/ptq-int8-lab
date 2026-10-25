#!/usr/bin/env bash
# 环境漂移对照：用**07 原轮的两个 engine**（不是重建的）重测 + 一个与 TRT 无关的算力基准。
#
# 背景：同一份 int8 engine，15:47 测 0.00733，15:49 复测 0.00920（+25%），
#   之后三次复测都在 0.0089–0.0092。需要判：是这份 engine 的 tactic 差，
#   还是整台机器的状态变了。判据：原轮 engine 若同样变慢 → 环境问题。
set -uo pipefail
REPO=/mnt/d/WorkBuddy/ShowBro/lab/07-tensorrt-deploy
cd "$REPO"
export LD_LIBRARY_PATH=/usr/lib/wsl/lib
P=/home/user/pyenv-torch/bin/python
R=results/push-20261025/repro
J=$R/bench_drift.jsonl
E=work/push-20261025/engines
exec > >(tee "$R/env_drift.txt") 2>&1

step() { echo; echo "### $*"; }

step "1/3 07 原轮的 int8 engine（f78b02ec），不重建"
flock -x /tmp/ptq-gpu.lock -c "cd $REPO && LD_LIBRARY_PATH=/usr/lib/wsl/lib $P scripts/push_bench.py \
  --onnx results/push-20261025/resnet18_cifar10_int8_c1x1.onnx --batch 512 --tag orig_int8_engine \
  --skip 1000 --preboost 100 --engine-path $E/resnet18_cifar10_int8_c1x1__b512_o3_default_w1.0_aNone_xNone_d0_f78b02ece7.trt \
  --jsonl $J" | grep -E "dataset round|steady:|^=="

step "2/3 07 原轮的 fp16 engine（d0ebc6f4），不重建"
flock -x /tmp/ptq-gpu.lock -c "cd $REPO && LD_LIBRARY_PATH=/usr/lib/wsl/lib $P scripts/push_bench.py \
  --onnx results/push-20261025/resnet18_cifar10_fp16_c1x1.onnx --batch 512 --tag orig_fp16_engine \
  --preboost 100 --engine-path $E/resnet18_cifar10_fp16_c1x1__b512_o5_default_w1.0_aNone_xNone_d1_d0ebc6f4f8.trt \
  --jsonl $J" | grep -E "dataset round|steady:|^=="

step "3/3 与 TRT 无关的绝对算力基准（torch mm / _int_mm / memcpy）"
flock -x /tmp/ptq-gpu.lock -c "cd $REPO && LD_LIBRARY_PATH=/usr/lib/wsl/lib $P results/push-20261025/repro/env_check.py"
