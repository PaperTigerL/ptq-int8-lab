#!/usr/bin/env bash
set -uo pipefail
cd /mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64
P=../models/resnet18_cifar10_int8_old.param
M=../models/resnet18_cifar10_int8_old.bin
D=../data_raw/test_batch.raw
OUT=results-h035
mkdir -p "$OUT"

echo "== profile 512 (InnerProduct) =="
./ncnn_cuda_profile_layers_ipgemv "$P" "$M" "$D" --int8 --limit 512 --batch 64 --repeat 1 --warmup 20 \
  --reset-every 1 > "$OUT/profile_ipgemv_512.log" 2>&1
echo "rc=$?"
grep -E "TYPE InnerProduct|fc/Gemm|total_layer|h2d|d2h" "$OUT/profile_ipgemv_512.log"
echo "-- baseline a82d8d7 --"
grep -E "TYPE InnerProduct|fc/Gemm|total_layer|h2d|d2h" results-h029/profile_pool_512.log

echo "== nsys 128 =="
/home/user/cuda/bin/nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true \
  -o "$OUT/nsys_ipgemv" ./ncnn_cuda_eval_2000c_ipgemv "$P" "$M" "$D" --int8 --limit 128 --batch 128 \
  --repeat 1 --warmup 20 --reset-every 1 > "$OUT/nsys_ipgemv_run.log" 2>&1
echo "rc=$?"
/home/user/cuda/bin/nsys stats --report cuda_api_sum "$OUT/nsys_ipgemv.nsys-rep" 2>/dev/null | grep -E "cudaLaunchKernel|cudaMemcpy |cudaMemset"

echo "== A/B 8 rounds =="
python3 ab_ipgemv.py 8 > "$OUT/ab_ipgemv.log" 2>&1
tail -6 "$OUT/ab_ipgemv.log"
