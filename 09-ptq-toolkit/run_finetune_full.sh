#!/usr/bin/env bash
# 一键：CIFAR-10 全量微调（后台 nohup，日志落 results/）
#
# 用法
#   bash run_finetune_full.sh                  # 默认 ResNet-50 / 20 epoch
#   bash run_finetune_full.sh --resume         # 断点续跑（没有同协议历史就拒绝）
#   bash run_finetune_full.sh --wait           # 会话保持到训练结束（WSL 用）
#   ARCH=mobilenet_v2 TAG=mv2tf20 EPOCHS=20 LOG=results/finetune_mv2_full.log \
#     bash run_finetune_full.sh --wait
#
# 协议（写进 history/ckpt 的 protocol 字段）：
#   全量 50000 · batch 64 · lr 0.01 cosine · wd 5e-4 ·
#   RandomCrop(32,padding=4)+HFlip · label_smoothing 0.1
#
# 产物：paper-code/ckpts/ckpt_<arch>_<tag>_ep<N>.pth + history_<...>.json
# 日志：$LOG（默认 results/finetune_full.log）
# PID ：results/finetune_<tag>.pid
set -euo pipefail
cd /mnt/d/WorkBuddy/ShowBro/lab/09-ptq-toolkit
export PYTHONPATH=.

PY=${PY:-/home/user/pyenv-torch/bin/python}
ARCH=${ARCH:-resnet50}
EPOCHS=${EPOCHS:-20}
TAG=${TAG:-r50tf20}
WEIGHTS=${WEIGHTS:-imagenet}   # imagenet=正式协议；scratch 仅消融
BATCH=${BATCH:-64}
LR=${LR:-0.01}
WD=${WD:-5e-4}
LOG=${LOG:-results/finetune_full.log}
RESUME=0
WAIT=0

for arg in "$@"; do
  case "$arg" in
    --resume) RESUME=1 ;;
    --wait) WAIT=1 ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    *)
      echo "未知参数: $arg（本脚本只接受 --resume / --wait；其它配置用环境变量）" >&2
      exit 2
      ;;
  esac
done

mkdir -p results
cmd=(nice -n 19 "$PY" -u tools/finetune.py
     --arch "$ARCH" --run-tag "$TAG" --epochs "$EPOCHS"
     --batch-size "$BATCH" --lr "$LR" --weight-decay "$WD"
     --weights "$WEIGHTS" --device cuda)
if [ "$RESUME" -eq 1 ]; then
  cmd+=(--resume)
fi

echo "[launch] ${cmd[*]}"
echo "[launch] log -> $LOG"
nohup "${cmd[@]}" > "$LOG" 2>&1 &
pid=$!
echo "$pid" > "results/finetune_${TAG}.pid"
echo "[launch] pid=$pid（已脱离终端；tail -f $LOG 看进度）"

# WSL 的 systemd 会在最后一个 wsl.exe 会话退出后关闭发行版，
# 连带杀掉 nohup 进程。/ 需要无人值守时加 --wait 保持会话，
# 直到训练进程结束（日志/产物/pid 语义完全不变）。
if [ "$WAIT" -eq 1 ]; then
  echo "[launch] --wait：保持会话，等待 pid $pid 结束"
  wait "$pid"
  rc=$?
  echo "[launch] 训练结束 rc=$rc at $(date '+%F %T')"
  exit "$rc"
fi
