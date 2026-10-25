#!/usr/bin/env bash
# 反复跑「等空闲 → 测量」，直到低档样本攒够或到达轮数上限。
#
# ⭐ 为什么需要这个：
#   低档轮（util<=20%）在这台机器上很难一次凑够。
#   实测两次测量共 12 轮，低档只有 3 轮（fp32 一轮、fp16 两轮、int8 零轮）。
#   而 1~2 轮的 median 就是那个值本身，统计上不可靠 ——
#   int8 甚至一轮低档都没有，只能拿到下界。
#
#   所以这里循环多次：每次等 GPU 空闲 → 测 → 写一个带时间戳的文件，
#   最后由 summarize_idle.py 合并所有文件重算。
#
# 用法：
#   ROUNDS=6 CYCLES=5 bash accumulate_measure.sh
set -u
ROUNDS=${ROUNDS:-6}
CYCLES=${CYCLES:-5}
WAIT=${WAIT_SECONDS:-600}
DELIVER=/mnt/d/WorkBuddy/ShowBro/lab/07-tensorrt-deploy

for c in $(seq 1 "$CYCLES"); do
    echo ""
    echo "═══ 第 $c/$CYCLES 轮 ═══"
    WAIT_SECONDS="$WAIT" ROUNDS="$ROUNDS" bash "$DELIVER/wait_then_measure.sh" || {
        echo "第 $c 轮出错，继续下一轮"
        continue
    }
    # 每轮之后立刻汇总一次，看低档样本有没有攒够
    P=$HOME/pyenv-torch/bin/python
    export LD_LIBRARY_PATH=/usr/lib/wsl/lib
    (cd "$DELIVER/scripts" && $P summarize_idle.py --dry 2>/dev/null \
        | grep -E "低档|⛔ 以下" | head -8)
done

echo ""
echo "═══ 全部轮次结束，最终汇总 ═══"
(cd "$DELIVER/scripts" && $P summarize_idle.py)
