#!/usr/bin/env bash
# H-065: confirm the 2px kernel is used, and compare per-layer conv time on/off.
set -eu
LAB=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy
HERE=$LAB/cuda
OUT=$HERE/results-h065
export PATH=/home/user/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64
P=$LAB/models/resnet18_cifar10_int8_old.param
M=$LAB/models/resnet18_cifar10_int8_old.bin
D=$LAB/data_raw/test_batch.raw

nsys profile -t cuda --force-overwrite true -o "$OUT/nsys_conv2px" \
  env NCNN_CUDA_CONV2PX=1 "$HERE/ncnn_cuda_eval_2000c_conv2px" "$P" "$M" "$D" --int8 \
  --limit 128 --batch 64 --repeat 1 --warmup 20 > "$OUT/nsys_conv2px_run.log" 2>&1 || true
nsys stats --report cuda_gpu_kern_sum "$OUT/nsys_conv2px.nsys-rep" > "$OUT/nsys_conv2px_stats.txt" 2>&1 || true
echo "=== kernel names with 2px ==="
grep -i "2px\|implicit_conv_int8" "$OUT/nsys_conv2px_stats.txt" | head -5 || echo "(nsys 无 kernel activity，改用源码确认)"

for mode in 0 1; do
  NCNN_CUDA_CONV2PX=$mode "$HERE/ncnn_cuda_profile_layers_conv2px" "$P" "$M" "$D" --int8 \
    --limit 512 --batch 64 --repeat 1 --warmup 20 > "$OUT/profile_conv2px_$mode.log" 2>&1 || true
  echo "=== profile conv2px=$mode ==="
  grep -E "^total_layer_ms|^TYPE Convolution|^精度" "$OUT/profile_conv2px_$mode.log"
done
