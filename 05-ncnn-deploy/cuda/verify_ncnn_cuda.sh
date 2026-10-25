#!/usr/bin/env bash
# ncnn CUDA 一键验收：重建 fork → 编译评测器 → fp32 复用精度 → int8 精度 → batch 64/256 性能。
# 只读模型与数据；失败即非零退出。输出 results/verify_ncnn_cuda.json + 短表。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB="/mnt/d/WorkBuddy/ShowBro/lab"
NCNN="${NCNN_DIR:-/home/user/ncnn-cuda}"
NCNN_BUILD="${NCNN_BUILD_DIR:-$NCNN/build-steward}"
CUDA_HOME="/home/user/cuda"
MODELS="$LAB/05-ncnn-deploy/models"
DATA="$LAB/05-ncnn-deploy/data_raw/test_batch.raw"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
OUT="$HERE/results/verify_ncnn_cuda_$RUN_ID"
BIN="$OUT/ncnn_cuda_eval"
LATEST_JSON="$HERE/results/verify_ncnn_cuda.json"
LATEST_TSV="$HERE/results/verify_ncnn_cuda.tsv"
mkdir -p "$OUT" "$HERE/results"

export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

log() { printf '%s\n' "$*" | tee -a "$OUT/verify.log"; }
die() { log "FAIL: $*"; exit 1; }

FP32_LIMIT="${FP32_LIMIT:-200}"
INT8_LIMIT="${INT8_LIMIT:-256}"
FP32_MIN="${FP32_MIN:-87.7}"
INT8_MIN="${INT8_MIN:-86.7}"

# 共享工作树：构建必须串行，且只使用 steward 独立目录。
# 其他工人可能不持锁直接跑 _rebuild.sh，所以拿锁后再等构建进程清空。
LOCK_FILE="${NCNN}/.build.lock"
exec 9>"$LOCK_FILE"
if ! flock -w 1800 9; then
  die "等待构建锁超时（$LOCK_FILE）"
fi
waited=0
while pgrep -f 'cmake|ninja|make|nvcc|cc1plus' >/dev/null 2>&1; do
  if [ "$waited" -ge 300 ]; then
    die "其他 cmake/make/nvcc 进程仍在运行，等待 300s 超时"
  fi
  sleep 2
  waited=$((waited + 2))
done

# ---------- 1. 重建 fork ----------
log "[1/4] 重建 ncnn CUDA fork（独立目录 build-steward）"
python3 "$HERE/patch_ncnn_cuda.py" > "$OUT/patch.log" 2>&1 || die "patch_ncnn_cuda.py 失败"
rm -rf "$NCNN_BUILD"
mkdir -p "$NCNN_BUILD"
cd "$NCNN_BUILD"
cmake -DCMAKE_BUILD_TYPE=Release \
      -DNCNN_CUDA=ON \
      -DNCNN_VULKAN=OFF \
      -DNCNN_BUILD_TOOLS=OFF \
      -DNCNN_BUILD_TESTS=OFF \
      -DNCNN_BUILD_BENCHMARK=OFF \
      -DNCNN_BUILD_EXAMPLES=OFF \
      -DNCNN_OPENMP=ON \
      -DLOG_LAYERS=OFF \
      -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" \
      .. > "$OUT/cmake.log" 2>&1 || { tail -40 "$OUT/cmake.log"; die "cmake 失败"; }
make -j"$(nproc)" > "$OUT/make.log" 2>&1 || { tail -60 "$OUT/make.log"; die "make 失败"; }
test -f "$NCNN_BUILD/src/libncnn.a" || die "缺少 libncnn.a"
test -f "$NCNN_BUILD/src/libncnn_cuda_lib.a" || die "缺少 libncnn_cuda_lib.a"
log "  libncnn.a / libncnn_cuda_lib.a OK"

# ---------- 2. 编译评测器 ----------
log "[2/4] 编译评测器"
cd "$HERE"
NCNN_DIR="$NCNN" NCNN_BUILD_DIR="$NCNN_BUILD" EVAL_BIN="$BIN" EVAL_SRC="$HERE/ncnn_cuda_eval_h015.cpp" bash "$HERE/build.sh" > "$OUT/compile.log" 2>&1 || { tail -60 "$OUT/compile.log"; die "评测器编译失败"; }
test -x "$BIN" || die "评测器不存在：$BIN"
log "  $BIN OK"
flock -u 9

# ---------- 3/4. 评测 ----------
parse_precision() {
  awk '/精度 [0-9]+\.[0-9]+%/{v=$2; gsub(/%/,"",v); last=v} END{print last}' "$1"
}
parse_ms() {
  awk '/最优 含传输墙钟/{v=$3; last=v} END{print last}' "$1"
}
run_eval() {
  local name="$1" mode="$2" param="$3" bin="$4" limit="$5" batch="$6"
  local logf="$OUT/$name.log"
  local env_prefix=()
  if [ "$mode" = "int8" ]; then env_prefix=(env NCNN_CUDA_IMPLICIT=1); fi
  log "  run $name: $mode limit=$limit batch=$batch" >&2
  if ! "${env_prefix[@]}" "$BIN" "$param" "$bin" "$DATA" \
        "--$mode" --limit "$limit" --batch "$batch" --repeat 1 --warmup 5 --reset-every 1 \
        > "$logf" 2>&1; then
    tail -30 "$logf" || true
    die "$name 运行失败"
  fi
  local p m
  p="$(parse_precision "$logf")"
  m="$(parse_ms "$logf")"
  [ -n "$p" ] || die "$name 未解析到精度"
  [ -n "$m" ] || die "$name 未解析到 ms/img"
  printf '%s\t%s\t%s\n' "$name" "$p" "$m"
}

log "[3/4] fp32 复用精度（$FP32_LIMIT 张，batch 64，默认 reset）"
FP32_LINE="$(run_eval fp32_reuse fp32 "$MODELS/resnet18_cifar10.param" "$MODELS/resnet18_cifar10.bin" "$FP32_LIMIT" 64)"
FP32_P="$(printf '%s' "$FP32_LINE" | cut -f2)"
FP32_MS="$(printf '%s' "$FP32_LINE" | cut -f3)"
log "  fp32 精度 ${FP32_P}% | ${FP32_MS} ms/img"

log "[4/4] int8 精度 + batch 64/256 性能（$INT8_LIMIT 张）"
INT8_B64_LINE="$(run_eval int8_b64 int8 "$MODELS/resnet18_cifar10_int8_old.param" "$MODELS/resnet18_cifar10_int8_old.bin" "$INT8_LIMIT" 64)"
INT8_P="$(printf '%s' "$INT8_B64_LINE" | cut -f2)"
INT8_B64_MS="$(printf '%s' "$INT8_B64_LINE" | cut -f3)"
log "  int8 精度 ${INT8_P}% | batch64 ${INT8_B64_MS} ms/img"

INT8_B256_LINE="$(run_eval int8_b256 int8 "$MODELS/resnet18_cifar10_int8_old.param" "$MODELS/resnet18_cifar10_int8_old.bin" "$INT8_LIMIT" 256)"
INT8_B256_MS="$(printf '%s' "$INT8_B256_LINE" | cut -f3)"
log "  batch256 ${INT8_B256_MS} ms/img"

# ---------- 结果 ----------
NOW="$(date -Iseconds)"
cat > "$OUT/verify.json" <<EOF
{
  "schema": "verify-ncnn-cuda@1",
  "generatedAt": "$NOW",
  "runId": "$RUN_ID",
  "limits": {"fp32": $FP32_LIMIT, "int8": $INT8_LIMIT},
  "fp32_reuse": {"precision_pct": $FP32_P, "ms_img": $FP32_MS},
  "int8": {"precision_pct": $INT8_P, "batch64_ms_img": $INT8_B64_MS, "batch256_ms_img": $INT8_B256_MS},
  "thresholds": {"fp32_min": $FP32_MIN, "int8_min": $INT8_MIN},
  "pass": false
}
EOF
cp "$OUT/verify.json" "$LATEST_JSON"
printf 'metric\tvalue\n' > "$LATEST_TSV"
printf 'fp32_reuse_precision_pct\t%s\nint8_precision_pct\t%s\nbatch64_ms_img\t%s\nbatch256_ms_img\t%s\n' \
  "$FP32_P" "$INT8_P" "$INT8_B64_MS" "$INT8_B256_MS" >> "$LATEST_TSV"
cp "$LATEST_TSV" "$OUT/verify.tsv"

PASS=1
awk -v v="$FP32_P" -v m="$FP32_MIN" 'BEGIN{exit !(v>=m)}' || PASS=0
awk -v v="$INT8_P" -v m="$INT8_MIN" 'BEGIN{exit !(v>=m)}' || PASS=0
awk -v v="$INT8_B64_MS" 'BEGIN{exit !(v>0)}' || PASS=0
awk -v v="$INT8_B256_MS" 'BEGIN{exit !(v>0)}' || PASS=0

if [ "$PASS" -eq 1 ]; then
  sed -i 's/"pass": false/"pass": true/' "$OUT/verify.json"
  cp "$OUT/verify.json" "$LATEST_JSON"
fi

cat <<EOF
metric                         value
fp32_reuse_precision_pct       $FP32_P
int8_precision_pct             $INT8_P
batch64_ms_img                 $INT8_B64_MS
batch256_ms_img                $INT8_B256_MS
json                           $LATEST_JSON
logdir                         $OUT
EOF

[ "$PASS" -eq 1 ] || die "验收阈值未通过（fp32>=${FP32_MIN}%, int8>=${INT8_MIN}%）"
log "PASS: 四项数字达标"
exit 0
