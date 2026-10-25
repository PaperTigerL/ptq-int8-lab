#!/usr/bin/env bash
# Phase A：builder 旋钮扫描（fp16 + c1x1 图），dataset 每轮先 preboost 顶 clock
cd /mnt/d/WorkBuddy/ShowBro/lab/07-tensorrt-deploy
LOG=results/push-20261025/phaseA.log
G=results/push-20261025/resnet18_cifar10_fp16_c1x1.onnx
B=work/resnet18_cifar10_fp16.onnx
{
  bash scripts/push_one.sh "$B" 512 fp16_base_pb     --preboost 100
  bash scripts/push_one.sh "$G" 512 fp16c_o3_pb      --preboost 100
  bash scripts/push_one.sh "$G" 512 fp16c_o0_pb      --opt 0 --preboost 100
  bash scripts/push_one.sh "$G" 512 fp16c_o2_pb      --opt 2 --preboost 100
  bash scripts/push_one.sh "$G" 512 fp16c_o5_pb      --opt 5 --preboost 100
  bash scripts/push_one.sh "$G" 512 fp16c_o5_jit_pb  --opt 5 --tactic jit --preboost 100
  bash scripts/push_one.sh "$G" 512 fp16c_o5_ws05_pb --opt 5 --workspace-gb 0.5 --preboost 100
  bash scripts/push_one.sh "$G" 1024 fp16c_o5_b1024_pb --opt 5 --preboost 100
  bash scripts/push_one.sh "$G" 2048 fp16c_o5_b2048_pb --opt 5 --preboost 100
  bash scripts/push_one.sh "$G" 512 fp16c_o5_dio_pb   --opt 5 --direct-io --preboost 100
} 2>&1 | tee "$LOG"
