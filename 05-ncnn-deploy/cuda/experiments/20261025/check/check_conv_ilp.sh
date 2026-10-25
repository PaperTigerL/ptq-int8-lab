#!/usr/bin/env bash
# H-086: build ILP kernel, verify numerics, run A/B.
set -eu
LAB=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy
HERE=$LAB/cuda
OUT=$HERE/results-h086
export PATH=/home/user/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64
P=$LAB/models/resnet18_cifar10_int8_old.param
M=$LAB/models/resnet18_cifar10_int8_old.bin
D=$LAB/data_raw/test_batch.raw
BLD=/home/user/ncnn-cuda/build-ptq

cd /home/user/ncnn-cuda
python3 "$HERE/patch_conv_ilp.py"
cmake --build build-ptq -j8 > "$OUT/build_lib_ilp.log" 2>&1
tail -2 "$OUT/build_lib_ilp.log"

NCNN_BUILD_DIR=$BLD EVAL_SRC="$HERE/ncnn_cuda_eval_2000c.cpp" \
  EVAL_BIN="$HERE/ncnn_cuda_eval_2000c_flat" bash "$HERE/build.sh" > "$OUT/build_eval_ilp.log" 2>&1
NCNN_BUILD_DIR=$BLD EVAL_SRC="$HERE/ncnn_cuda_dump_logits.cpp" \
  EVAL_BIN="$HERE/ncnn_cuda_dump_logits_flat" bash "$HERE/build.sh" > "$OUT/build_dump_ilp.log" 2>&1
tail -1 "$OUT/build_eval_ilp.log"

for n in 128 512; do
  NCNN_CUDA_CONV_ILP=1 "$HERE/ncnn_cuda_dump_logits_flat" "$P" "$M" "$D" --int8 \
    --limit $n --batch $n --repeat 1 --warmup 20 --reset-every 1 \
    --dump-logits "$OUT/logits_ilp_$n.txt" > "$OUT/run_ilp_$n.log" 2>&1
  echo "=== ILP $n ==="; grep -E "^精度|^最优" "$OUT/run_ilp_$n.log"
done
python3 "$HERE/analyze_logits.py" "$HERE/results-h039/logits_default_128.txt" "$OUT/logits_ilp_128.txt"
python3 "$HERE/analyze_logits.py" "$HERE/results-h035/logits_ipgemv_512.txt" "$OUT/logits_ilp_512.txt"
