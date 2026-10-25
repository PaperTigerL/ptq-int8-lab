#!/usr/bin/env bash
# Phase C：int8 同口径（preboost）复测（两个 engine 都已在缓存里，无 rebuild）
cd /mnt/d/WorkBuddy/ShowBro/lab/07-tensorrt-deploy
{
  bash scripts/push_one.sh work/resnet18_cifar10_int8.onnx 512 int8_base_pb --skip 1000 --preboost 100
  bash scripts/push_one.sh results/push-20261025/resnet18_cifar10_int8_c1x1.onnx 512 int8_c1x1_pb --skip 1000 --preboost 100
} 2>&1 | tee results/push-20261025/phaseC.log
