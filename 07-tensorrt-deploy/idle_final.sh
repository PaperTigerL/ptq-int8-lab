#!/usr/bin/env bash
# ⭐ 相对基线门控的最终测量 —— results/idle_final.txt
#
# ⚠️ 为什么门控要用**相对基线**而不是绝对阈值：
#   这台机器的 GPU 基线就是 10~18% —— 桌面合成器 + 常驻 Electron 窗口
#   （WorkBuddy / 豆包 / WXWork / DingTalk / B站，约 790 MiB）撑着。
#   它们是**图形进程**，nvidia-smi 的 compute-apps 里查不到，
#   但 utilization.gpu 常年有底噪。
#   所以 util<5% 永远到不了（第一版门控就是这么写的，结果三个配置
#   各等 900s 然后全部 SKIP，白跑 45 分钟）。
#   2026-10-19 21:00 那次 final.txt 开跑前也是 13%。
#
#   判据：**util <= 20% 连续 3 次**（默认 20，落在图形底噪 10~18% 与
#   真实负载 95~100% 之间）。曾试过「相对基线」（baseline+8），
#   实测失败 —— 采样里出现一次 0% 就把 min 基线拉到 0，相对判据塌成
#   util<=8%，比不做还糟。详见 scripts/wait_idle.py 的文档。
#
# ⚠️ 等不到也**不跳过**：
#   等不到安静窗口时，把当时的 util 记进该轮注释，照常测量。
#   事后由 summarize_idle.py 按 util 分档：
#     · 低占用轮（util <= baseline+8）的 median 才是可信值
#     · 高占用轮单独列出，不混进主结论
#   跳过等于白跑 —— 被占用时测的数字是**下界**，仍然有参考价值，
#   前提是**如实标注**它是在什么负载下测的。
#
# 方向性纠正：README 原来写「这些数字是上界」，**写反了**。
#   被别的进程占着只会让数字**偏慢**（抢 SM、抢显存带宽、上下文切换），
#   所以「被占时的数字」是**下界**，空闲测出的 min 才是性能**上限**。
#
# 用法：
#    ROUNDS=5 bash idle_final.sh            # 默认 5 轮，batch=512
#    ROUNDS=20 bash idle_final.sh           # 更稳
#    THRESH=30 bash idle_final.sh           # 机器比较空时放宽阈值
set -u
cd /mnt/d/WorkBuddy/ShowBro/lab/07-tensorrt-deploy/scripts
export TRT_WORKDIR=/mnt/d/WorkBuddy/_scratch/libtorch_cuda
export LD_LIBRARY_PATH=/usr/lib/wsl/lib
P=$HOME/pyenv-torch/bin/python
ROUNDS=${ROUNDS:-5}
BATCH=${BATCH:-512}
THRESH=${THRESH:-20}
GATE_WAIT=${GATE_WAIT:-60}
RESULTS=/mnt/d/WorkBuddy/ShowBro/lab/07-tensorrt-deploy/results
mkdir -p "$RESULTS"
# ⭐ 每次测量写一个带时间戳的文件，**不覆盖**。
#   原因：低档轮（util<=20%）很难一次凑够 —— 实测一轮6次里常只有 1~2 轮低档，
#   1~2 轮的 median 就是那个值本身，统计上不可靠。
#   累积多轮再汇总，才能拿到有样本量的可信值。
STAMP=$(date '+%Y%m%d_%H%M%S')
OUT=$RESULTS/idle_${STAMP}.txt

# 当前 util（门控后、测量前采样一次）
cur_util() {
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits \
        2>/dev/null | head -1 | tr -d ' '
}

# 门控：相对基线。等不到返回非 0，但**不阻止测量**。
gate() {
    $P wait_idle.py --thresh "$THRESH" --samples 3 \
        --max-wait "$GATE_WAIT" > /tmp/gate.log 2>&1
    local rc=$?
    local util
    util=$(grep -o 'GATE_UTIL=[0-9]*' /tmp/gate.log | tail -1 | cut -d= -f2)
    util=${util:-$(cur_util)}
    {
        echo "门控：判据 util<= ${THRESH}% 连续 3 次，最多等 ${GATE_WAIT}s"
        echo "门控时刻 util=${util}%  结果：$( [ "$rc" -eq 0 ] \
            && echo '通过（安静窗口出现）' || echo '未通过（等满超时，**照常测量**）')"
    } >> "$OUT"
    return 0     # ⛔ 永远返回 0：门控只提供信息，不阻止测量
}

run() {
    local model=$1 skip=$2
    echo "" >> "$OUT"
    echo "--- $model batch=$BATCH skip=$skip ---" >> "$OUT"
    gate
    $P trt_eval.py --model "$model" --batch "$BATCH" --skip "$skip" \
        --repeat "$ROUNDS" 2>&1 | tee -a "$OUT" \
        | grep -E "round|最优|精度 |波动"
}

{
    echo "# idle_final $(date '+%F %T')"
    echo "#"
echo "# 门控判据：util <= ${THRESH}% 连续 3 次，最多等 ${GATE_WAIT}s。"
    echo "#   **等不到也照常测量，不跳过** —— 跳过 = 白跑；"
    echo "#   测了并标注 = 至少拿到「被占用时的下界」。"
    echo "#"
    echo "# 为什么阈值是 ${THRESH}% 而不是 5%：本机**空闲时** util 就有 10~18%"
    echo "#   —— 桌面合成器 + 常驻 Electron 窗口（WorkBuddy/豆包/WXWork/"
    echo "#   DingTalk/B站，约 790 MiB）撑着。它们是图形进程，"
    echo "#   nvidia-smi 的 compute-apps 里查不到，但让 utilization.gpu 常年有底噪。"
    echo "#   而**真实计算负载时** util 稳定 95~100%。"
    echo "#   所以 ${THRESH}% 恰好落在两者之间，是有效的分界。"
    echo "#   （曾试「相对基线 baseline+8」，实测失败：采样里一次 0% 就把 min 基线"
    echo "#     拉到 0，判据塌成 util<=8%，比不做还糟。）"
    echo "#   第一版门控就是这么写的，三个配置各等 900s 然后全部 SKIP。"
    echo "#"
    echo "# 每配置 $ROUNDS 轮，batch=$BATCH，每轮记录 util 与显存。"
    echo "# 事后按 util 分档：util<=${THRESH}% 的轮次为「低档」，其 median 是可信值；"
    echo "#   util>${THRESH}% 的轮次为「高档」（被占用时的下界），单独列出。"
    echo "#   （见文件末尾的分档统计）。"
    echo "# 口径：只计 forward + synchronize，不含数据准备。"
    echo "# 注意：被别的进程占着只会让数字偏慢，所以被占时的数据是下界不是上界。"
} > "$OUT"

echo "开始门控测量（每配置 $ROUNDS 轮，batch=$BATCH，判据 util<=$THRESH%）..."
run fp32 0
run fp16 0
run int8 1000

$P summarize_idle.py
echo ""
echo "写入 $OUT"
echo "提示：多跑几轮累积低档样本（summarize_idle.py 会合并所有 idle_*.txt）"
