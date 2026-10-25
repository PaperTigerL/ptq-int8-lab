#!/usr/bin/env bash
# Phase B：block-GEMM、CUDA Graph、流数、int8/fp16 混跑
cd /mnt/d/WorkBuddy/ShowBro/lab/07-tensorrt-deploy
LOG=results/push-20261025/phaseB.log
C=results/push-20261025/resnet18_cifar10_fp16_c1x1.onnx
BG=results/push-20261025/resnet18_cifar10_fp16_c1x1_bg.onnx
MIX=results/push-20261025/resnet18_cifar10_mixed_bg.onnx
{
  echo '### CUDA Graph on collapsed fp16 (cache hit)'
  bash scripts/push_one.sh "$C" 512 fp16c_o3_graph --preboost 100 --graph
  echo '### block-GEMM fp16 (layer3 2x2 -> 4x4 block GEMM)'
  bash scripts/push_one.sh "$BG" 512 fp16c_bg_o3_pb --preboost 100 --profile --graph
  echo '### block-GEMM fp16, opt5'
  bash scripts/push_one.sh "$BG" 512 fp16c_bg_o5_pb --opt 5 --preboost 100
  echo '### aux streams = 2 (builder knob, fp16 c1x1 opt3)'
  bash scripts/push_one.sh "$C" 512 fp16c_o3_aux2 --max-aux-streams 2 --preboost 100
  echo '### mixed int8/fp16 (int8 Conv+MatMul, 其余 fp16)'
  bash scripts/push_one.sh "$MIX" 512 mixed_bg_o3 --skip 1000 --preboost 100 --profile
} 2>&1 | tee "$LOG"
