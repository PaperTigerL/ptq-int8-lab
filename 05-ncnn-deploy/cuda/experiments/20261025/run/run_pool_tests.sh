#!/usr/bin/env bash
set -uo pipefail
cd /mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64
P=../models/resnet18_cifar10_int8_old.param
M=../models/resnet18_cifar10_int8_old.bin
D=../data_raw/test_batch.raw
OUT=results-h029
mkdir -p "$OUT"

echo "== 128 fused dump =="
./ncnn_cuda_dump_logits_pool "$P" "$M" "$D" --int8 --limit 128 --batch 128 --repeat 1 \
  --warmup 20 --reset-every 1 --dump-logits "$OUT/logits_pool_128.txt" > "$OUT/run_pool_128.log" 2>&1
echo "fused rc=$?"
NCNN_CUDA_POOL_FUSE_PAD=0 ./ncnn_cuda_dump_logits_pool "$P" "$M" "$D" --int8 --limit 128 --batch 128 --repeat 1 \
  --warmup 20 --reset-every 1 --dump-logits "$OUT/logits_pooloff_128.txt" > "$OUT/run_pooloff_128.log" 2>&1
echo "off rc=$?"

echo "== analysis 128 =="
python3 analyze_logits.py "$OUT/logits_base_128.txt" "$OUT/logits_pool_128.txt"
python3 analyze_logits.py "$OUT/logits_pooloff_128.txt" "$OUT/logits_pool_128.txt"

echo "== 512 dumps =="
./ncnn_cuda_dump_logits_pool "$P" "$M" "$D" --int8 --limit 512 --batch 512 --repeat 1 \
  --warmup 20 --reset-every 1 --dump-logits "$OUT/logits_pool_512.txt" > "$OUT/run_pool_512.log" 2>&1
echo "fused512 rc=$?"
./ncnn_cuda_dump_logits_fuse "$P" "$M" "$D" --int8 --limit 512 --batch 512 --repeat 1 \
  --warmup 20 --reset-every 1 --dump-logits "$OUT/logits_base_pool_512.txt" > "$OUT/run_base_pool_512.log" 2>&1
echo "base512 rc=$?"
python3 analyze_logits.py "$OUT/logits_base_pool_512.txt" "$OUT/logits_pool_512.txt"

echo "== nsys 128 (api counts) =="
/home/user/cuda/bin/nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true \
  -o "$OUT/nsys_pool_base" ./ncnn_cuda_eval_2000c_9561414 "$P" "$M" "$D" --int8 --limit 128 --batch 128 \
  --repeat 1 --warmup 20 --reset-every 1 > "$OUT/nsys_pool_base_run.log" 2>&1
echo "nsys_base rc=$?"
NCNN_CUDA_POOL_FUSE_PAD=1 /home/user/cuda/bin/nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true \
  -o "$OUT/nsys_pool_fused" ./ncnn_cuda_eval_2000c_pool "$P" "$M" "$D" --int8 --limit 128 --batch 128 \
  --repeat 1 --warmup 20 --reset-every 1 > "$OUT/nsys_pool_fused_run.log" 2>&1
echo "nsys_fused rc=$?"
echo "-- base --"
/home/user/cuda/bin/nsys stats --report cuda_api_sum "$OUT/nsys_pool_base.nsys-rep" 2>/dev/null | grep -E "cudaLaunchKernel|cudaMemset|cudaMemcpy "
echo "-- fused --"
/home/user/cuda/bin/nsys stats --report cuda_api_sum "$OUT/nsys_pool_fused.nsys-rep" 2>/dev/null | grep -E "cudaLaunchKernel|cudaMemset|cudaMemcpy "

echo "== profile 512 (Pooling layer) =="
./ncnn_cuda_profile_layers "$P" "$M" "$D" --int8 --limit 512 --batch 64 --repeat 1 --warmup 20 \
  --reset-every 1 > "$OUT/profile_base_512.log" 2>&1
echo "prof_base rc=$?"
./ncnn_cuda_profile_layers_pool "$P" "$M" "$D" --int8 --limit 512 --batch 64 --repeat 1 --warmup 20 \
  --reset-every 1 > "$OUT/profile_pool_512.log" 2>&1
echo "prof_pool rc=$?"
grep -E "TYPE Pooling|maxpool|total_layer|h2d|d2h" "$OUT/profile_base_512.log"
echo "--"
grep -E "TYPE Pooling|maxpool|total_layer|h2d|d2h" "$OUT/profile_pool_512.log"

echo "== A/B 512 =="
python3 ab_pool_512.py 5 > "$OUT/ab_pool_512.txt" 2>&1
tail -12 "$OUT/ab_pool_512.txt"
