#!/usr/bin/env bash
# 探测 WSL 侧 Vulkan 到底能不能用。
# 我在 README 里写了「真要 ncnn 上 GPU，正确做法是换 Vulkan 后端」——
# 这句话必须验证过才能留在文档里。这里拿事实。
# 注意：WSL 命令经 Git Bash 转发会吃掉 $VAR，一律写脚本文件。
set -u

echo "=== 1. ICD 清单 ==="
ls -la /usr/share/vulkan/icd.d/ 2>&1 || echo "（/usr/share/vulkan/icd.d 不存在）"
echo
for f in /usr/share/vulkan/icd.d/*.json; do
    [ -e "$f" ] || continue
    echo "--- $f ---"
    cat "$f"
    echo
done

echo "=== 2. vulkaninfo / vulkan-tools 有没有 ==="
command -v vulkaninfo 2>&1 || echo "vulkaninfo 未安装"

echo
echo "=== 3. 驱动库 ==="
ls -la /usr/lib/x86_64-linux-gnu/libvulkan* 2>&1 | head
echo "--- WSL 直通的 NVIDIA 库 ---"
ls /usr/lib/wsl/lib/ 2>&1 | head -30

echo
echo "=== 4. 有没有 NVIDIA 的 ICD json（nvidia_icd.json）==="
find / -name "*icd*.json" -not -path "*/proc/*" 2>/dev/null | head -20
