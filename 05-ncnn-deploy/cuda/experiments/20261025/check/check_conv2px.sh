#!/usr/bin/env bash
# H-065: build candidate binaries and verify 2px kernel logits/top1/top5.
set -eu
LAB=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy
HERE=$LAB/cuda
OUT=$HERE/results-h065
mkdir -p "$OUT"
export PATH=/home/user/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64
P=$LAB/models/resnet18_cifar10_int8_old.param
M=$LAB/models/resnet18_cifar10_int8_old.bin
D=$LAB/data_raw/test_batch.raw
BLD=/home/user/ncnn-cuda/build-ptq

NCNN_BUILD_DIR=$BLD EVAL_SRC="$HERE/ncnn_cuda_eval_2000c.cpp" \
  EVAL_BIN="$HERE/ncnn_cuda_eval_2000c_conv2px" bash "$HERE/build.sh" > "$OUT/build_eval.log" 2>&1
NCNN_BUILD_DIR=$BLD EVAL_SRC="$HERE/ncnn_cuda_dump_logits.cpp" \
  EVAL_BIN="$HERE/ncnn_cuda_dump_logits_conv2px" bash "$HERE/build.sh" > "$OUT/build_dump.log" 2>&1
NCNN_BUILD_DIR=$BLD EVAL_SRC="$HERE/ncnn_cuda_profile_layers.cpp" \
  EVAL_BIN="$HERE/ncnn_cuda_profile_layers_conv2px" bash "$HERE/build.sh" > "$OUT/build_prof.log" 2>&1
tail -1 "$OUT/build_eval.log"; tail -1 "$OUT/build_dump.log"; tail -1 "$OUT/build_prof.log"

echo "=== 128 logits, conv2px on ==="
NCNN_CUDA_CONV2PX=1 "$HERE/ncnn_cuda_dump_logits_conv2px" "$P" "$M" "$D" --int8 \
  --limit 128 --batch 128 --repeat 1 --warmup 20 --reset-every 1 \
  --dump-logits "$OUT/logits_conv2px_128.txt" > "$OUT/run_conv2px_128.log" 2>&1
grep -E "^精度|^最优" "$OUT/run_conv2px_128.log"

echo "=== 512 logits, conv2px on ==="
NCNN_CUDA_CONV2PX=1 "$HERE/ncnn_cuda_dump_logits_conv2px" "$P" "$M" "$D" --int8 \
  --limit 512 --batch 512 --repeat 1 --warmup 20 --reset-every 1 \
  --dump-logits "$OUT/logits_conv2px_512.txt" > "$OUT/run_conv2px_512.log" 2>&1
grep -E "^精度|^最优" "$OUT/run_conv2px_512.log"

echo "=== diffs ==="
python3 "$HERE/analyze_logits.py" "$HERE/results-h039/logits_default_128.txt" "$OUT/logits_conv2px_128.txt"
python3 "$HERE/analyze_logits.py" "$HERE/results-h035/logits_ipgemv_512.txt" "$OUT/logits_conv2px_512.txt"
