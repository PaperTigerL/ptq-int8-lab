#!/usr/bin/env bash
# 全量构建（引擎 + 逐层 bench + 多流 harness），迭代用。
set -e
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$HOME/libtorch/lib:$HOME/cuda/lib64
nice -n 19 bash build.sh
nice -n 19 bash build_bench_layers.sh
nice -n 19 bash build_streams.sh
nice -n 19 bash build_select.sh
echo "ALL BUILT"
