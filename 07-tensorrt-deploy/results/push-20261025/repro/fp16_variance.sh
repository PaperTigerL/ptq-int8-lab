#!/usr/bin/env bash
# fp16 复现差异排查：同一张图、同一配置，engine 重建 2 次 + 原 engine 复测 1 次。
#
# 背景：2026-10-25 复现时，int8 c1x1 对上原轮（−2.55%，PASS），
#   但 fp16（opt5+DIRECT_IO，图逐字节相同）测到 0.01382 / steady 0.01375，
#   比原轮 0.01068 / 0.01222 慢 29% / 12.5%。
#   图相同、口径相同 → 只能是 TRT 选到的 tactic 不同，或者当轮机器状态不同。
#   这个脚本把两种可能分开：先复测同一 engine（机器状态），再重建（tactic）。
set -euo pipefail
REPO=/mnt/d/WorkBuddy/ShowBro/lab/07-tensorrt-deploy
cd "$REPO"
export LD_LIBRARY_PATH=/usr/lib/wsl/lib
R=results/push-20261025/repro
G="$R/resnet18_cifar10_fp16_c1x1.onnx"
J="$R/bench_fp16v.jsonl"
rm -f "$J"
exec > >(tee "$R/fp16_variance.txt") 2>&1

step() { echo; echo "### $*"; }

step "A：复测**已有** engine（不重建）—— 如果还是 0.0138，说明不是机器状态"
bash scripts/push_one.sh "$G" 512 fp16_cached_rerun \
    --opt 5 --direct-io --preboost 100 \
    --jsonl "$J" --engine-dir "$R/engines" --timing-cache "$R/timing.cache"

step "B：强制重建 engine 1（--rebuild）"
cp "$R/engines/resnet18_cifar10_fp16_c1x1__b512_o5_default_w1.0_aNone_xNone_d1_1c2d4767fb.trt" \
   "$R/engines/fp16_first_1c2d4767fb.trt" 2>/dev/null || true
bash scripts/push_one.sh "$G" 512 fp16_rebuild_1 \
    --opt 5 --direct-io --preboost 100 --rebuild \
    --jsonl "$J" --engine-dir "$R/engines" --timing-cache "$R/timing.cache"
cp "$R/engines/resnet18_cifar10_fp16_c1x1__b512_o5_default_w1.0_aNone_xNone_d1_1c2d4767fb.trt" \
   "$R/engines/fp16_rebuild_1.trt" 2>/dev/null || true

step "C：强制重建 engine 2（--rebuild）"
bash scripts/push_one.sh "$G" 512 fp16_rebuild_2 \
    --opt 5 --direct-io --preboost 100 --rebuild \
    --jsonl "$J" --engine-dir "$R/engines" --timing-cache "$R/timing.cache"
cp "$R/engines/resnet18_cifar10_fp16_c1x1__b512_o5_default_w1.0_aNone_xNone_d1_1c2d4767fb.trt" \
   "$R/engines/fp16_rebuild_2.trt" 2>/dev/null || true

step "汇总：同一张图、同一配置，三次 engine 的结果"
python3 - "$J" <<'PY'
import json, sys
for ln in open(sys.argv[1], encoding="utf-8"):
    ln = ln.strip()
    if not ln.startswith("{"):
        continue
    r = json.loads(ln)
    print("%-18s dataset %.5f (min %.5f, spread %.1f%%)  steady %.5f  acc %.2f%%  build %.0fs"
          % (r["tag"], r["dataset_median"], r["dataset_min"],
             100 * (r["dataset_max"] - r["dataset_min"]) / r["dataset_median"],
             r["steady_median"], r["acc"], r["build_s"]))
PY
sha256sum "$R"/engines/fp16_*.trt | tee "$R/sha256_fp16v.txt"
