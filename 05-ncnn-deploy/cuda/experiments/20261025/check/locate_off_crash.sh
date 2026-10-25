#!/usr/bin/env bash
# H-055: where does graph-off 512/batch64 crash; does another batch work?
set -u
LAB=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy
HERE=$LAB/cuda
OUT=$HERE/results-h055
P=$LAB/models/resnet18_cifar10_int8_old.param
M=$LAB/models/resnet18_cifar10_int8_old.bin
D=$LAB/data_raw/test_batch.raw
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64

echo "== graph off 512 b64, unbuffered =="
NCNN_CUDA_GRAPH=0 stdbuf -o0 -e0 "$HERE/ncnn_cuda_graph_eval" "$P" "$M" "$D" --int8 \
  --limit 512 --batch 64 --repeat 1 --warmup 30 > "$OUT/off512_b64_unbuf.log" 2>&1
echo "rc=$?"; tail -4 "$OUT/off512_b64_unbuf.log"

for b in 32 128 256; do
  NCNN_CUDA_GRAPH=0 "$HERE/ncnn_cuda_graph_eval" "$P" "$M" "$D" --int8 --limit 512 --batch $b \
    --repeat 1 --warmup 30 > "$OUT/off512_b$b.log" 2>&1
  echo "batch=$b rc=$?"; tail -1 "$OUT/off512_b$b.log" | head -c 80; echo
done
