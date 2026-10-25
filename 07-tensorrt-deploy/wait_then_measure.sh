#!/usr/bin/env bash
# 等 GPU 真的空闲下来（util<=20% 连续 N 次）再跑测量。
#
# 与 wait_idle.py 的区别：这个等**更久**（默认 30 分钟），因为
# 「等不到也照常测」虽然不白跑，但拿到的是下界而不是可信值。
# 所以值得多等一会儿去拿真正的空闲数据。
#
# 用法：
#    bash wait_then_measure.sh                # 最多等 30 分钟
#    WAIT_SECONDS=3600 ROUNDS=6 bash wait_then_measure.sh
set -u
WAIT_SECONDS=${WAIT_SECONDS:-1800}
ROUNDS=${ROUNDS:-6}
DELIVER=/mnt/d/WorkBuddy/ShowBro/lab/07-tensorrt-deploy
STABLE=3          # 连续多少次安静才算空闲
THRESH=20         # 安静阈值

echo "等待 GPU 空闲（util <= ${THRESH}% 连续 ${STABLE} 次，最多 ${WAIT_SECONDS}s）..."
T0=$(date +%s)
STREAK=0
while [ $(( $(date +%s) - T0 )) -lt "$WAIT_SECONDS" ]; do
    U=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    U=${U:-100}
    if [ "$U" -le "$THRESH" ]; then
        STREAK=$((STREAK + 1))
        echo "  安静 $STREAK/$STABLE (util ${U}%)"
        if [ "$STREAK" -ge "$STABLE" ]; then
            echo "GPU 空闲了，开始测量。"
            cd "$DELIVER"
            ROUNDS="$ROUNDS" bash idle_final.sh
            exit $?
        fi
    else
        [ "$STREAK" -gt 0 ] && echo "  又忙了，计数清零 (util ${U}%)"
        STREAK=0
    fi
    sleep 5
done

echo "等了 ${WAIT_SECONDS}s 机器一直没空闲 —— 直接跑（会拿到下界数据并标注）。"
cd "$DELIVER"
ROUNDS="$ROUNDS" bash idle_final.sh
