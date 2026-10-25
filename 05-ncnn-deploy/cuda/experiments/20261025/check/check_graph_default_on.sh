#!/usr/bin/env bash
# H-055: default-on verification: with shim (graph), explicit off, shim missing
# (fallback reason), plus 128/512 logits.
set -eu
LAB=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy
HERE=$LAB/cuda
OUT=$HERE/results-h055
mkdir -p "$OUT"
export PATH=/home/user/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64
P=$LAB/models/resnet18_cifar10_int8_old.param
M=$LAB/models/resnet18_cifar10_int8_old.bin
D=$LAB/data_raw/test_batch.raw

gcc -O2 -shared -fPIC -o "$HERE/graph_shim.so" "$HERE/graph_shim.c" -ldl
NCNN_BUILD_DIR=/home/user/ncnn-cuda/build-ptq \
  EVAL_SRC="$HERE/ncnn_cuda_graph_eval.cpp" EVAL_BIN="$HERE/ncnn_cuda_graph_eval" \
  bash "$HERE/build.sh" > "$OUT/build.log" 2>&1
tail -1 "$OUT/build.log"

echo "=== A: default-on + shim, 128 ==="
LD_PRELOAD="$HERE/graph_shim.so" NCNN_CUDA_GRAPH_DUMP="$OUT/logits_default_on_128.txt" \
  "$HERE/ncnn_cuda_graph_eval" "$P" "$M" "$D" --int8 --limit 128 --batch 64 --repeat 1 --warmup 20 \
  > "$OUT/run_default_on_128.log" 2>&1
grep -E "^\[graph\]|^精度|^最优" "$OUT/run_default_on_128.log"

echo "=== B: default-on + shim, 512 ==="
LD_PRELOAD="$HERE/graph_shim.so" NCNN_CUDA_GRAPH_DUMP="$OUT/logits_default_on_512.txt" \
  "$HERE/ncnn_cuda_graph_eval" "$P" "$M" "$D" --int8 --limit 512 --batch 64 --repeat 1 --warmup 20 \
  > "$OUT/run_default_on_512.log" 2>&1
grep -E "^\[graph\]|^精度|^最优" "$OUT/run_default_on_512.log"

echo "=== C: explicit off (NCNN_CUDA_GRAPH=0) ==="
NCNN_CUDA_GRAPH=0 "$HERE/ncnn_cuda_graph_eval" "$P" "$M" "$D" --int8 --limit 128 \
  --batch 64 --repeat 1 --warmup 20 > "$OUT/run_off_128.log" 2>&1
grep -E "^\[graph\]|^精度|^最优" "$OUT/run_off_128.log"

echo "=== D: default-on without shim (must fall back with reason) ==="
"$HERE/ncnn_cuda_graph_eval" "$P" "$M" "$D" --int8 --limit 128 --batch 64 --repeat 1 --warmup 20 \
  > "$OUT/run_noshim_128.log" 2>&1
grep -E "^\[graph\]|^精度|^最优" "$OUT/run_noshim_128.log"

echo "=== logits checks ==="
python3 "$HERE/analyze_logits.py" "$HERE/results-h039/logits_default_128.txt" "$OUT/logits_default_on_128.txt"
python3 "$HERE/analyze_logits.py" "$HERE/results-h035/logits_ipgemv_512.txt" "$OUT/logits_default_on_512.txt"
