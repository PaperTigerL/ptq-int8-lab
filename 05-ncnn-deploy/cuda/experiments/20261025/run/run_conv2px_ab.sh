#!/usr/bin/env bash
# H-065: rebuild eval, prove dispatch, then run 8-round A/B.
set -eu
LAB=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy
HERE=$LAB/cuda
OUT=$HERE/results-h065
export PATH=/home/user/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64
P=$LAB/models/resnet18_cifar10_int8_old.param
M=$LAB/models/resnet18_cifar10_int8_old.bin
D=$LAB/data_raw/test_batch.raw

NCNN_BUILD_DIR=/home/user/ncnn-cuda/build-ptq EVAL_SRC="$HERE/ncnn_cuda_eval_2000c.cpp" \
  EVAL_BIN="$HERE/ncnn_cuda_eval_2000c_conv2px" bash "$HERE/build.sh" > "$OUT/build_eval2.log" 2>&1
tail -1 "$OUT/build_eval2.log"

echo "=== dispatch proof (limit 2, warmup 1) ==="
NCNN_CUDA_CONV2PX=1 NCNN_CUDA_CONV2PX_DEBUG=1 "$HERE/ncnn_cuda_eval_2000c_conv2px" \
  "$P" "$M" "$D" --int8 --limit 2 --batch 2 --repeat 1 --warmup 1 > "$OUT/dispatch_proof.log" 2>&1
grep -c "^conv2px" "$OUT/dispatch_proof.log" || true
grep "^conv2px" "$OUT/dispatch_proof.log" | sort | uniq -c | head -20
grep -E "^精度" "$OUT/dispatch_proof.log"

python3 "$HERE/ab_conv2px.py" 2>&1 | tail -22
