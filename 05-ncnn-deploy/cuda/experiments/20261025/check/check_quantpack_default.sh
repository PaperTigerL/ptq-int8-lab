#!/usr/bin/env bash
# H-074: flip default, rebuild, re-verify numerics (env unset = fused).
set -eu
LAB=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy
HERE=$LAB/cuda
OUT=$HERE/results-h074
export PATH=/home/user/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64
P=$LAB/models/resnet18_cifar10_int8_old.param
M=$LAB/models/resnet18_cifar10_int8_old.bin
D=$LAB/data_raw/test_batch.raw
BLD=/home/user/ncnn-cuda/build-ptq

python3 "$HERE/flip_quantpack_default.py"
cd /home/user/ncnn-cuda && cmake --build build-ptq -j8 > "$OUT/build_lib2.log" 2>&1
tail -1 "$OUT/build_lib2.log"

NCNN_BUILD_DIR=$BLD EVAL_SRC="$HERE/ncnn_cuda_eval_2000c.cpp" \
  EVAL_BIN="$HERE/ncnn_cuda_eval_2000c_quantpack" bash "$HERE/build.sh" > "$OUT/build_eval2.log" 2>&1
NCNN_BUILD_DIR=$BLD EVAL_SRC="$HERE/ncnn_cuda_dump_logits.cpp" \
  EVAL_BIN="$HERE/ncnn_cuda_dump_logits_quantpack" bash "$HERE/build.sh" > "$OUT/build_dump2.log" 2>&1
NCNN_BUILD_DIR=$BLD EVAL_SRC="$HERE/ncnn_cuda_profile_layers.cpp" \
  EVAL_BIN="$HERE/ncnn_cuda_profile_layers_quantpack" bash "$HERE/build.sh" > "$OUT/build_prof2.log" 2>&1
tail -1 "$OUT/build_eval2.log"

echo "=== default (fused) 128 logits ==="
"$HERE/ncnn_cuda_dump_logits_quantpack" "$P" "$M" "$D" --int8 --limit 128 --batch 128 \
  --repeat 1 --warmup 20 --reset-every 1 --dump-logits "$OUT/logits_default_128.txt" \
  > "$OUT/run_default_128.log" 2>&1
grep -E "^精度|^最优" "$OUT/run_default_128.log"
echo "=== default (fused) 512 logits ==="
"$HERE/ncnn_cuda_dump_logits_quantpack" "$P" "$M" "$D" --int8 --limit 512 --batch 512 \
  --repeat 1 --warmup 20 --reset-every 1 --dump-logits "$OUT/logits_default_512.txt" \
  > "$OUT/run_default_512.log" 2>&1
grep -E "^精度|^最优" "$OUT/run_default_512.log"
echo "=== explicit off (NCNN_CUDA_FUSE_QUANT_PACK=0) 128 ==="
NCNN_CUDA_FUSE_QUANT_PACK=0 "$HERE/ncnn_cuda_eval_2000c_quantpack" "$P" "$M" "$D" --int8 \
  --limit 128 --batch 64 --repeat 1 --warmup 20 > "$OUT/run_off_128.log" 2>&1
grep -E "^精度" "$OUT/run_off_128.log"
echo "=== diffs ==="
python3 "$HERE/analyze_logits.py" "$HERE/results-h039/logits_default_128.txt" "$OUT/logits_default_128.txt"
python3 "$HERE/analyze_logits.py" "$HERE/results-h035/logits_ipgemv_512.txt" "$OUT/logits_default_512.txt"
