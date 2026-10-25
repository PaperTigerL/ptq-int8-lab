#!/usr/bin/env bash
# H-090: rebuild eval/dump binaries from the final (guarded) library and re-verify bit-exact logits.
set -eu
LAB=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy
HERE=$LAB/cuda
OUT=$HERE/results-h090
BLD=/home/user/ncnn-cuda/build-ptq
P=$LAB/models/resnet18_cifar10_int8_old.param
M=$LAB/models/resnet18_cifar10_int8_old.bin
D=$LAB/data_raw/test_batch.raw
export PATH=/home/user/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64

NCNN_BUILD_DIR=$BLD EVAL_SRC="$HERE/ncnn_cuda_dump_logits.cpp" \
  EVAL_BIN="$HERE/ncnn_cuda_dump_logits_smw" bash "$HERE/build.sh" > "$OUT/build_dump_final.log" 2>&1
NCNN_BUILD_DIR=$BLD EVAL_SRC="$HERE/ncnn_cuda_eval_2000c.cpp" \
  EVAL_BIN="$HERE/ncnn_cuda_eval_2000c_smw" bash "$HERE/build.sh" > "$OUT/build_eval_final.log" 2>&1
tail -1 "$OUT/build_eval_final.log"

for n in 128 512; do
  NCNN_CUDA_CONV_SHMW=1 "$HERE/ncnn_cuda_dump_logits_smw" "$P" "$M" "$D" --int8 \
    --limit $n --batch $n --repeat 1 --warmup 20 --reset-every 1 \
    --dump-logits "$OUT/logits_final_on_$n.txt" > "$OUT/run_final_on_$n.log" 2>&1
  "$HERE/ncnn_cuda_dump_logits_smw" "$P" "$M" "$D" --int8 \
    --limit $n --batch $n --repeat 1 --warmup 20 --reset-every 1 \
    --dump-logits "$OUT/logits_final_off_$n.txt" > "$OUT/run_final_off_$n.log" 2>&1
  echo "=== final $n : on vs off (guarded build) ==="
  python3 "$HERE/analyze_logits.py" "$OUT/logits_final_off_$n.txt" "$OUT/logits_final_on_$n.txt" | tail -1
  echo "=== final $n : on vs pre-guard build ==="
  python3 "$HERE/analyze_logits.py" "$OUT/logits_off_$n.txt" "$OUT/logits_final_on_$n.txt" | tail -1
  echo "=== final $n : on vs historical default ==="
  if [ "$n" = "128" ]; then REF=$HERE/results-h039/logits_default_128.txt; else REF=$HERE/results-h035/logits_ipgemv_512.txt; fi
  python3 "$HERE/analyze_logits.py" "$REF" "$OUT/logits_final_on_$n.txt" | tail -1
  grep -E '^精度' "$OUT/run_final_on_$n.log"
done
echo DONE
