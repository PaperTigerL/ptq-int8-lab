#!/usr/bin/env bash
# H-090: build shared-memory weight staging kernel, resource report, verify numerics.
set -eu
LAB=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy
HERE=$LAB/cuda
OUT=$HERE/results-h090
mkdir -p "$OUT"
export PATH=/home/user/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64
P=$LAB/models/resnet18_cifar10_int8_old.param
M=$LAB/models/resnet18_cifar10_int8_old.bin
D=$LAB/data_raw/test_batch.raw
BLD=/home/user/ncnn-cuda/build-ptq

# --- resource usage (ptxas -v) for base vs smw ---
cd /tmp
cp /home/user/ncnn-cuda/src/layer/cuda/implicit_conv_cuda.cu smw_h090.cu
nvcc -c smw_h090.cu -o /tmp/smw_h090.o -DNCNN_CUDA=1 -Xcompiler -O3 -DNDEBUG -arch=sm_75 \
  -I/home/user/cuda/include -I/home/user/ncnn-cuda/src -I$BLD/src \
  -I/home/user/ncnn-cuda/src/layer -I/home/user/ncnn-cuda/src/layer/cuda \
  -Xptxas -v > "$OUT/ptxas_smw.log" 2>&1
grep -A2 -E "implicit_conv_int8_kernel(_smw|_flat|EPKa)?" "$OUT/ptxas_smw.log" | grep -E "entry|registers|spill" | head -20

cd /home/user/ncnn-cuda && cmake --build build-ptq -j8 > "$OUT/build_lib.log" 2>&1
tail -2 "$OUT/build_lib.log"

NCNN_BUILD_DIR=$BLD EVAL_SRC="$HERE/ncnn_cuda_eval_2000c.cpp" \
  EVAL_BIN="$HERE/ncnn_cuda_eval_2000c_smw" bash "$HERE/build.sh" > "$OUT/build_eval.log" 2>&1
NCNN_BUILD_DIR=$BLD EVAL_SRC="$HERE/ncnn_cuda_dump_logits.cpp" \
  EVAL_BIN="$HERE/ncnn_cuda_dump_logits_smw" bash "$HERE/build.sh" > "$OUT/build_dump.log" 2>&1
tail -1 "$OUT/build_eval.log"

# --- numerics: switch OFF vs switch ON from the same binary, plus historical baseline ---
for n in 128 512; do
  "$HERE/ncnn_cuda_dump_logits_smw" "$P" "$M" "$D" --int8 \
    --limit $n --batch $n --repeat 1 --warmup 20 --reset-every 1 \
    --dump-logits "$OUT/logits_off_$n.txt" > "$OUT/run_off_$n.log" 2>&1
  NCNN_CUDA_CONV_SHMW=1 "$HERE/ncnn_cuda_dump_logits_smw" "$P" "$M" "$D" --int8 \
    --limit $n --batch $n --repeat 1 --warmup 20 --reset-every 1 \
    --dump-logits "$OUT/logits_smw_$n.txt" > "$OUT/run_smw_$n.log" 2>&1
  echo "=== $n off-vs-on ==="
  python3 "$HERE/analyze_logits.py" "$OUT/logits_off_$n.txt" "$OUT/logits_smw_$n.txt"
  echo "=== $n on-vs-historical-default ==="
  if [ "$n" = "128" ]; then REF=$HERE/results-h039/logits_default_128.txt; else REF=$HERE/results-h035/logits_ipgemv_512.txt; fi
  python3 "$HERE/analyze_logits.py" "$REF" "$OUT/logits_smw_$n.txt"
  grep -E "^精度" "$OUT/run_off_$n.log" "$OUT/run_smw_$n.log"
done
echo "DONE"
