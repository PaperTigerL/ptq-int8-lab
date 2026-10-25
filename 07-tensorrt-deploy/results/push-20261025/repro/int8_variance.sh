#!/usr/bin/env bash
# int8 构建抖动探针：同一张图、同一配置，重建第二次 + 复测一次。
#
# 背景：fp16 同一张图重建 3 次，dataset 落在 0.0107–0.0138（1/3 的构建选到差 tactic，
#   掉 12–29%）。int8 的 1559× 是 08-report 的头条数字，必须回答「它稳不稳」。
#   第一轮复现拿到 0.00725（−2.55%），这里是第二个独立构建。
set -euo pipefail
REPO=/mnt/d/WorkBuddy/ShowBro/lab/07-tensorrt-deploy
cd "$REPO"
export LD_LIBRARY_PATH=/usr/lib/wsl/lib
R=results/push-20261025/repro
G="$R/resnet18_cifar10_int8_c1x1.onnx"
E="$R/engines/resnet18_cifar10_int8_c1x1__b512_o3_default_w1.0_aNone_xNone_d0_78b88e1614.trt"
J="$R/bench_int8v.jsonl"
rm -f "$J"
exec > >(tee "$R/int8_variance.txt") 2>&1

step() { echo; echo "### $*"; }

step "保留第一次复现（冷构建 247 s）的 engine"
cp "$E" "$R/engines/int8_repro_build1.trt"

step "第二条：强制重建 int8 engine（--rebuild）"
bash scripts/push_one.sh "$G" 512 int8_rebuild_2 \
    --skip 1000 --preboost 100 --rebuild \
    --jsonl "$J" --engine-dir "$R/engines" --timing-cache "$R/timing.cache"
cp "$E" "$R/engines/int8_repro_build2.trt"

step "同 engine 复测（不重建），看测量本身抖多少"
bash scripts/push_one.sh "$G" 512 int8_rebuild_2_rerun \
    --skip 1000 --preboost 100 \
    --jsonl "$J" --engine-dir "$R/engines" --timing-cache "$R/timing.cache"

step "汇总"
python3 - "$J" <<'PY'
import json, sys
for ln in open(sys.argv[1], encoding="utf-8"):
    ln = ln.strip()
    if not ln.startswith("{"):
        continue
    r = json.loads(ln)
    print("%-24s dataset %.5f  steady %.5f  acc %.2f%%  build %.0fs"
          % (r["tag"], r["dataset_median"], r["steady_median"], r["acc"], r["build_s"]))
PY
sha256sum "$R"/engines/int8_repro_build*.trt | tee "$R/sha256_int8v.txt"
