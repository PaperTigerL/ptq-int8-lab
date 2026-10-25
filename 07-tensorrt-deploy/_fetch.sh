#!/usr/bin/env bash
# 拉 ncnn 的 CUDA 分支。GitHub 直连在 WSL 里 TLS 会被掐，所以多源重试。
set -u
cd /home/user

rm -rf ncnn-cuda ncnn-cuda.tar.gz

try() {
    echo "--- 尝试: $1"
    if rm -rf ncnn-cuda && timeout 180 git clone --depth 1 "$1" ncnn-cuda 2>&1 | tail -2; then
        if [ -d ncnn-cuda/src ]; then echo "OK via $1"; return 0; fi
    fi
    return 1
}

try_tar() {
    echo "--- 尝试 tarball: $1"
    if timeout 240 curl -L --retry 2 -o ncnn-cuda.tar.gz "$1" 2>&1 | tail -1; then
        if tar xzf ncnn-cuda.tar.gz 2>/dev/null; then
            local d
            d=$(tar tzf ncnn-cuda.tar.gz 2>/dev/null | head -1 | cut -d/ -f1)
            if [ -n "$d" ] && [ -d "$d/src" ]; then
                mv "$d" ncnn-cuda
                echo "OK via tarball $1"; return 0
            fi
        fi
    fi
    return 1
}

try "https://github.com/atanmarko/ncnn-with-cuda.git" \
    || try "https://kkgithub.com/atanmarko/ncnn-with-cuda.git" \
    || try "https://ghfast.top/https://github.com/atanmarko/ncnn-with-cuda.git" \
    || try "https://gh-proxy.com/https://github.com/atanmarko/ncnn-with-cuda.git" \
    || try_tar "https://codeload.github.com/atanmarko/ncnn-with-cuda/tar.gz/refs/heads/master" \
    || { echo "全部源都失败"; exit 1; }

echo "=== 目录 ==="
ls ncnn-cuda | head -12
echo "=== CUDA 层 ==="
ls ncnn-cuda/src/layer/cuda 2>/dev/null | head -30
