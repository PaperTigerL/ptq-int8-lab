#!/usr/bin/env bash
set -uo pipefail
cd /mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64
P=../models/resnet18_cifar10_int8_old.param
M=../models/resnet18_cifar10_int8_old.bin
D=../data_raw/test_batch.raw
OUT=results-h028
mkdir -p "$OUT"

echo "== 128-image top1/top5 + diff =="
python3 analyze_logits.py "$OUT/logits_base_128.txt" "$OUT/logits_relu_fused_128.txt"
python3 analyze_logits.py "$OUT/logits_clip_unfused_128.txt" "$OUT/logits_clip_fused_128.txt"
python3 analyze_logits.py "$OUT/logits_leaky_unfused_128.txt" "$OUT/logits_leaky_fused_128.txt"

echo "== 512 dumps =="
./ncnn_cuda_dump_logits_fuse "$P" "$M" "$D" --int8 --limit 512 --batch 512 --repeat 1 \
  --warmup 20 --reset-every 1 --dump-logits "$OUT/logits_base_512.txt" > "$OUT/run_base_512.log" 2>&1
echo "base512 rc=$?"
NCNN_CUDA_FUSE_ACT=1 ./ncnn_cuda_dump_logits_fuse "$P" "$M" "$D" --int8 --limit 512 --batch 512 --repeat 1 \
  --warmup 20 --reset-every 1 --dump-logits "$OUT/logits_fused_512.txt" > "$OUT/run_fused_512.log" 2>&1
echo "fused512 rc=$?"
python3 analyze_logits.py "$OUT/logits_base_512.txt" "$OUT/logits_fused_512.txt"

echo "== nsys 128 =="
/home/user/cuda/bin/nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true \
  -o "$OUT/nsys_base" ./ncnn_cuda_eval_2000c_644efc3 "$P" "$M" "$D" --int8 --limit 128 --batch 128 \
  --repeat 1 --warmup 20 --reset-every 1 > "$OUT/nsys_base_run.log" 2>&1
echo "nsys_base rc=$?"
NCNN_CUDA_FUSE_ACT=1 /home/user/cuda/bin/nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true \
  -o "$OUT/nsys_fused" ./ncnn_cuda_eval_2000c_fuse "$P" "$M" "$D" --int8 --limit 128 --batch 128 \
  --repeat 1 --warmup 20 --reset-every 1 > "$OUT/nsys_fused_run.log" 2>&1
echo "nsys_fused rc=$?"
for tag in base fused; do
  echo "-- $tag --"
  nsys stats --report cuda_api_sum "$OUT/nsys_${tag}.nsys-rep" 2>/dev/null | grep -E "cudaLaunchKernel|cudaMemcpy "
  grep -E "最优 GPU kernel|最优 含传输墙钟" "$OUT/nsys_${tag}_run.log"
done

echo "== A/B 512 (5 rounds) =="
python3 ab_h028.py 5 > "$OUT/ab_fuse_512.txt" 2>&1
tail -12 "$OUT/ab_fuse_512.txt"
