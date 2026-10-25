#!/usr/bin/env bash
# 慢/快两态到底差在哪：在 bench 跑的同时每 0.25 s 采一次 nvidia-smi。
#
# 背景：同一份 engine（07 原轮 f78b02ec）在两个相隔 2 分钟的窗口里
#   测出 0.00734 与 0.0093（相差 26%），steady 也从 0.00787 变 0.0088–0.0091。
#   所以「绝对延迟」不是常数，必须知道是哪一项在变：SM clock？内存 clock？功率？
set -uo pipefail
REPO=/mnt/d/WorkBuddy/ShowBro/lab/07-tensorrt-deploy
cd "$REPO"
export LD_LIBRARY_PATH=/usr/lib/wsl/lib
P=/home/user/pyenv-torch/bin/python
R=results/push-20261025/repro
E=work/push-20261025/engines/resnet18_cifar10_int8_c1x1__b512_o3_default_w1.0_aNone_xNone_d0_f78b02ece7.trt
CSV=$R/clock_trace.csv
exec > >(tee "$R/clock_trace.txt") 2>&1

: > "$CSV"
( while :; do
    nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu \
      --format=csv,noheader,nounits || true
    sleep 0.25
  done ) >> "$CSV" &
SAMPLER=$!

flock -x /tmp/ptq-gpu.lock -c "cd $REPO && LD_LIBRARY_PATH=/usr/lib/wsl/lib $P scripts/push_bench.py \
  --onnx results/push-20261025/resnet18_cifar10_int8_c1x1.onnx --batch 512 --tag clock_trace \
  --skip 1000 --preboost 100 --engine-path $E --jsonl $R/bench_drift.jsonl" \
  | grep -E "dataset round|steady:|^==|steady clocks"

kill $SAMPLER 2>/dev/null || true
wait $SAMPLER 2>/dev/null || true

echo
echo "=== 采样统计（$CSV，共 $(wc -l < "$CSV") 个样本）==="
python3 - "$CSV" <<'PY'
import statistics, sys
rows = []
for ln in open(sys.argv[1]):
    p = [x.strip() for x in ln.split(",")]
    if len(p) < 5 or not p[0].replace(".", "").isdigit():
        continue
    rows.append(p)
if not rows:
    print("没有采到有效样本"); raise SystemExit
def col(i):
    return [float(r[i]) for r in rows]
for i, name in enumerate(["clocks.sm (MHz)", "clocks.mem (MHz)", "power (W)",
                          "temp (C)", "util (%)"]):
    v = col(i)
    print("%-18s min %6.1f  median %6.1f  max %6.1f" %
          (name, min(v), statistics.median(v), max(v)))
busy = [r for r in rows if float(r[4]) > 50]
if busy:
    v = [float(r[0]) for r in busy]
    print("只取 util>50%% 的 %d 个样本: SM clock min %.0f / median %.0f / max %.0f MHz"
          % (len(v), min(v), statistics.median(v), max(v)))
PY
