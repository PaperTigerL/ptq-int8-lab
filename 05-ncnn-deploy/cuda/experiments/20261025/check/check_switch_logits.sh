#!/usr/bin/env bash
set -uo pipefail
cd /mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64
BIN=./ncnn_cuda_dump_logits_sync
P=../models/resnet18_cifar10_int8_old.param
M=../models/resnet18_cifar10_int8_old.bin
D=../data_raw/test_batch.raw
OUT=results-h033
mkdir -p "$OUT"

run() {
  local tag="$1"; shift
  env "$@" "$BIN" "$P" "$M" "$D" --int8 --limit 128 --batch 128 --repeat 1 \
    --warmup 20 --reset-every 1 --dump-logits "$OUT/logits_${tag}_128.txt" \
    > "$OUT/run_${tag}_128.log" 2>&1
  echo "$tag rc=$? $(grep '^精度' "$OUT/run_${tag}_128.log" | tail -1)"
}

# pool: 其它开关固定 baseline（binop=0, relu off）
run pool_base   NCNN_CUDA_POOL_FUSE_PAD=0 NCNN_CUDA_BINARYOP_VEC=0
run pool_cand   NCNN_CUDA_POOL_FUSE_PAD=1 NCNN_CUDA_BINARYOP_VEC=0
# binop: pool=1, relu off
run binop_base  NCNN_CUDA_POOL_FUSE_PAD=1 NCNN_CUDA_BINARYOP_VEC=0
run binop_cand  NCNN_CUDA_POOL_FUSE_PAD=1 NCNN_CUDA_BINARYOP_VEC=1
# relu: pool=1, binop=0, fuse off/on
run relu_base   NCNN_CUDA_POOL_FUSE_PAD=1 NCNN_CUDA_BINARYOP_VEC=0
run relu_cand   NCNN_CUDA_POOL_FUSE_PAD=1 NCNN_CUDA_BINARYOP_VEC=0 NCNN_CUDA_FUSE_ACT=1

echo "== md5 =="
md5sum "$OUT"/logits_pool_base_128.txt "$OUT"/logits_pool_cand_128.txt \
       "$OUT"/logits_binop_base_128.txt "$OUT"/logits_binop_cand_128.txt \
       "$OUT"/logits_relu_base_128.txt "$OUT"/logits_relu_cand_128.txt
echo "== diffs =="
python3 analyze_logits.py "$OUT/logits_pool_base_128.txt" "$OUT/logits_pool_cand_128.txt"
python3 analyze_logits.py "$OUT/logits_binop_base_128.txt" "$OUT/logits_binop_cand_128.txt"
python3 analyze_logits.py "$OUT/logits_relu_base_128.txt" "$OUT/logits_relu_cand_128.txt"
