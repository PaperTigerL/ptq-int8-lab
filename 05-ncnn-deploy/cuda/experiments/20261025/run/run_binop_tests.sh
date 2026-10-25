#!/usr/bin/env bash
set -uo pipefail
cd /mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64
P=../models/resnet18_cifar10_int8_old.param
M=../models/resnet18_cifar10_int8_old.bin
D=../data_raw/test_batch.raw
OUT=results-h030
mkdir -p "$OUT"

echo "== classification (2 images, vec debug) =="
NCNN_CUDA_BINARYOP_VEC_DEBUG=1 ./ncnn_cuda_eval_2000c_binop "$P" "$M" "$D" --int8 --limit 2 --batch 2 \
  --repeat 1 --warmup 1 --reset-every 1 2>&1 | grep "binop vec4" | sort | uniq -c

echo "== 128 dumps =="
./ncnn_cuda_dump_logits_binop "$P" "$M" "$D" --int8 --limit 128 --batch 128 --repeat 1 \
  --warmup 20 --reset-every 1 --dump-logits "$OUT/logits_vec_128.txt" > "$OUT/run_vec_128.log" 2>&1
echo "vec rc=$?"
NCNN_CUDA_BINARYOP_VEC=0 ./ncnn_cuda_dump_logits_binop "$P" "$M" "$D" --int8 --limit 128 --batch 128 --repeat 1 \
  --warmup 20 --reset-every 1 --dump-logits "$OUT/logits_vecoff_128.txt" > "$OUT/run_vecoff_128.log" 2>&1
echo "off rc=$?"
md5sum "$OUT/logits_vec_128.txt" "$OUT/logits_vecoff_128.txt" results-h029/logits_pool_128.txt
python3 analyze_logits.py results-h029/logits_pool_128.txt "$OUT/logits_vec_128.txt"

echo "== 512 dumps =="
./ncnn_cuda_dump_logits_binop "$P" "$M" "$D" --int8 --limit 512 --batch 512 --repeat 1 \
  --warmup 20 --reset-every 1 --dump-logits "$OUT/logits_vec_512.txt" > "$OUT/run_vec_512.log" 2>&1
echo "vec512 rc=$?"
python3 analyze_logits.py results-h029/logits_pool_512.txt "$OUT/logits_vec_512.txt"

echo "== profile 512 (BinaryOp) =="
./ncnn_cuda_profile_layers_binop "$P" "$M" "$D" --int8 --limit 512 --batch 64 --repeat 1 --warmup 20 \
  --reset-every 1 > "$OUT/profile_vec_512.log" 2>&1
echo "prof rc=$?"
grep -E "TYPE BinaryOp|total_layer|h2d|d2h" "$OUT/profile_vec_512.log"
echo "-- baseline (4dfc722) --"
grep -E "TYPE BinaryOp|total_layer|h2d|d2h" results-h029/profile_pool_512.log

echo "== nsys 128 =="
/home/user/cuda/bin/nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true \
  -o "$OUT/nsys_vec" ./ncnn_cuda_eval_2000c_binop "$P" "$M" "$D" --int8 --limit 128 --batch 128 \
  --repeat 1 --warmup 20 --reset-every 1 > "$OUT/nsys_vec_run.log" 2>&1
echo "nsys rc=$?"
/home/user/cuda/bin/nsys stats --report cuda_api_sum "$OUT/nsys_vec.nsys-rep" 2>/dev/null | grep -E "cudaLaunchKernel|cudaMemcpy |cudaMemset"

echo "== A/B 512 =="
python3 ab_binop_512.py 5 > "$OUT/ab_binop_512.txt" 2>&1
tail -12 "$OUT/ab_binop_512.txt"
