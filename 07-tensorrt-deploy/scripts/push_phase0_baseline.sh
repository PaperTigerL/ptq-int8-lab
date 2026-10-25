#!/usr/bin/env bash
cd /mnt/d/WorkBuddy/ShowBro/lab/07-tensorrt-deploy
export LD_LIBRARY_PATH=/usr/lib/wsl/lib
P=/home/user/pyenv-torch/bin/python
mkdir -p results/push-20261025
flock -x /tmp/ptq-gpu.lock -c '
set -x
cd /mnt/d/WorkBuddy/ShowBro/lab/07-tensorrt-deploy
export LD_LIBRARY_PATH=/usr/lib/wsl/lib
P=/home/user/pyenv-torch/bin/python
nvidia-smi --query-gpu=utilization.gpu,memory.used,clocks.sm --format=csv,noheader
$P scripts/wait_idle.py 2>&1 | tail -2
$P scripts/push_bench.py --onnx work/resnet18_cifar10_fp16.onnx --batch 512 --tag fp16_base_b512 --profile
$P scripts/push_bench.py --onnx results/push-20261025/resnet18_cifar10_fp16_c1x1.onnx --batch 512 --tag fp16_c1x1_b512 --profile
$P scripts/push_bench.py --onnx work/resnet18_cifar10_int8.onnx --batch 512 --skip 1000 --tag int8_base_b512 --profile
$P scripts/push_bench.py --onnx results/push-20261025/resnet18_cifar10_int8_c1x1.onnx --batch 512 --skip 1000 --tag int8_c1x1_b512 --profile
' 2>&1 | tee results/push-20261025/phase0.log
