#!/usr/bin/env bash
# M2 验收：R50 ep18 / MV2 ep19 × W8/W4，全量 10k 评测（WSL 侧运行）
set -x
cd /mnt/d/WorkBuddy/ShowBro/lab/09-ptq-toolkit
export PYTHONPATH=.
PY=/home/user/pyenv-torch/bin/python
R50=/mnt/d/WorkBuddy/ShowBro/paper-code/ckpts/ckpt_resnet50_r50aug20_ep18.pth
MV2=/mnt/d/WorkBuddy/ShowBro/paper-code/ckpts/ckpt_mobilenet_v2_mv2aug20_ep19.pth
"$PY" -m ptq_toolkit build --model resnet50     --weights "$R50" --w-bits 8 --a-bits 8 --calib-n 512 --eval-n 10000 --device cpu --out-dir results/m2_accept || echo "BUILD_FAIL r50_w8 rc=$?"
"$PY" -m ptq_toolkit build --model resnet50     --weights "$R50" --w-bits 4 --a-bits 8 --calib-n 512 --eval-n 10000 --device cpu --out-dir results/m2_accept || echo "BUILD_FAIL r50_w4 rc=$?"
"$PY" -m ptq_toolkit build --model mobilenet_v2 --weights "$MV2" --w-bits 8 --a-bits 8 --calib-n 512 --eval-n 10000 --device cpu --out-dir results/m2_accept || echo "BUILD_FAIL mv2_w8 rc=$?"
"$PY" -m ptq_toolkit build --model mobilenet_v2 --weights "$MV2" --w-bits 4 --a-bits 8 --calib-n 512 --eval-n 10000 --device cpu --out-dir results/m2_accept || echo "BUILD_FAIL mv2_w4 rc=$?"
echo "ALL_DONE"
