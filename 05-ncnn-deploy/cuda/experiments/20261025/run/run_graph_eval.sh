#!/usr/bin/env bash
# H-048: build graph eval, run 128 (with logits dump) and 512.
set -eu
LAB=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy
HERE=$LAB/cuda
OUT=$HERE/results-h048
export PATH=/home/user/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64

gcc -O2 -shared -fPIC -o "$HERE/graph_shim.so" "$HERE/graph_shim.c" -ldl
NCNN_BUILD_DIR=/home/user/ncnn-cuda/build-ptq \
  EVAL_SRC="$HERE/ncnn_cuda_graph_eval.cpp" EVAL_BIN="$HERE/ncnn_cuda_graph_eval" \
  bash "$HERE/build.sh" > "$OUT/build_graph_eval.log" 2>&1
tail -1 "$OUT/build_graph_eval.log"

run() {
  local tag=$1 limit=$2
  NCNN_CUDA_GRAPH=1 NCNN_CUDA_GRAPH_DUMP="$OUT/logits_graph_${tag}.txt" \
  LD_PRELOAD="$HERE/graph_shim.so" \
    "$HERE/ncnn_cuda_graph_eval" "$LAB/models/resnet18_cifar10_int8_old.param" \
    "$LAB/models/resnet18_cifar10_int8_old.bin" "$LAB/data_raw/test_batch.raw" \
    --int8 --limit "$limit" --batch 64 --repeat 1 --warmup 20 \
    > "$OUT/run_graph_${tag}.log" 2>&1 || true
  echo "=== $tag ==="
  grep -E "^\[graph\]|^精度|^  round|^最优" "$OUT/run_graph_${tag}.log"
}

run 128 128
run 512 512

echo "== 128 graph vs 8959d4c(默认路径) =="
python3 "$HERE/analyze_logits.py" "$HERE/results-h039/logits_default_128.txt" "$OUT/logits_graph_128.txt"
echo "== 512 graph vs cf8fc38 =="
python3 "$HERE/analyze_logits.py" "$HERE/results-h035/logits_ipgemv_512.txt" "$OUT/logits_graph_512.txt"
