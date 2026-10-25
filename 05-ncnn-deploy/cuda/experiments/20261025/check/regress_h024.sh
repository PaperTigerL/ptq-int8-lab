#!/usr/bin/env bash
# Single command: 2000-image x 4-config regression vs H-20261023-021 baseline.
# usage: bash regress_h024.sh [--limit N]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64:${LD_LIBRARY_PATH:-}
exec python3 "$HERE/regress_h024.py" "$@"
