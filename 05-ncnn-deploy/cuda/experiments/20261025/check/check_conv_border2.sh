#!/usr/bin/env bash
# H-079: build + verify (patches already applied).
set -eu
LAB=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy
HERE=$LAB/cuda
OUT=$HERE/results-h079
export PATH=/home/user/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64
P=$LAB/models/resnet18_cifar10_int8_old.param
M=$LAB/models/resnet18_cifar10_int8_old.bin
D=$LAB/data_raw/test_batch.raw
BLD=/home/user/ncnn-cuda/build-ptq

cd /home/user/ncnn-cuda && cmake --build build-ptq -j8 > "$OUT/build_lib2.log" 2>&1
tail -2 "$OUT/build_lib2.log"
NCNN_BUILD_DIR=$BLD EVAL_SRC="$HERE/ncnn_cuda_eval_2000c.cpp" \
  EVAL_BIN="$HERE/ncnn_cuda_eval_2000c_border" bash "$HERE/build.sh" > "$OUT/build_eval2.log" 2>&1
NCNN_BUILD_DIR=$BLD EVAL_SRC="$HERE/ncnn_cuda_dump_logits.cpp" \
  EVAL_BIN="$HERE/ncnn_cuda_dump_logits_border" bash "$HERE/build.sh" > "$OUT/build_dump2.log" 2>&1
tail -1 "$OUT/build_eval2.log"

for n in 128 512; do
  NCNN_CUDA_CONV_BORDER=1 "$HERE/ncnn_cuda_dump_logits_border" "$P" "$M" "$D" --int8 \
    --limit $n --batch $n --repeat 1 --warmup 20 --reset-every 1 \
    --dump-logits "$OUT/logits_border_$n.txt" > "$OUT/run_border_$n.log" 2>&1
  echo "=== $n ==="; grep -E "^精度|^最优" "$OUT/run_border_$n.log"
done
python3 "$HERE/analyze_logits.py" "$HERE/results-h039/logits_default_128.txt" "$OUT/logits_border_128.txt"
python3 "$HERE/analyze_logits.py" "$HERE/results-h035/logits_ipgemv_512.txt" "$OUT/logits_border_512.txt"

echo "=== env off sanity 128 ==="
"$HERE/ncnn_cuda_eval_2000c_border" "$P" "$M" "$D" --int8 --limit 128 --batch 64 \
  --repeat 1 --warmup 20 > "$OUT/run_off_128.log" 2>&1
grep -E "^精度" "$OUT/run_off_128.log"
