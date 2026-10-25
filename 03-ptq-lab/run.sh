#!/usr/bin/env bash
# PTQ 实验平台 · 启动脚本（WSL 专用）
#
#   ./run.sh            看用法
#   ./run.sh e1         跑实验 1
#   ./run.sh all        按顺序跑完全部七个实验
#   ./run.sh status     汇总 results/ 里已有的结果
#   ./run.sh check      只跑量化算子测试（填完 TODO 后用它验收）
#   ./run.sh plot       把结果画成论文用图（输出到 results/figs/）
#   ./run.sh archive    归档旧结果，只留每个实验最新的一份
#   ./run.sh mine       关掉参考实现，用你自己的 quant.py 跑全套
#
# Windows 下别直接双击这个脚本，进 WSL 再跑：
#   wsl
#   cd /mnt/d/WorkBuddy/ShowBro/lab/03-ptq-lab
#   ./run.sh
set -u

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$LAB_DIR" || exit 1
PY="$HOME/pyenv-torch/bin/python"

if [ ! -x "$PY" ]; then
    echo "找不到 WSL 里的 torch 环境：$PY"
    echo "请先执行 ./setup.sh"
    exit 1
fi

export PTQLAB_REFERENCE="${PTQLAB_REFERENCE:-1}"

hr() { echo "------------------------------------------------------------"; }

# 语法体检：填到一半写出缩进错误时，直接 import quant 会抛 SyntaxError，
# traceback 里全是 importlib 的内部帧，看不出到底哪一行错了。
# 而 guard() 只拦 TodoNotImplemented，拦不住 SyntaxError ——
# 结果就是「填到一半反而一点进度都看不到」，跟留白设计的目的相反。
# 所以跑测试前先单独扫一遍 src/，把行号和那一行原文直接打出来。
syntax_gate() {
    "$PY" - <<'PYEOF'
import ast, sys, pathlib
bad = []
for p in sorted(pathlib.Path("src").rglob("*.py")):
    try:
        ast.parse(p.read_text(encoding="utf-8"))
    except SyntaxError as e:
        bad.append((str(p), e.lineno, e.msg, (e.text or "").rstrip()))
    except UnicodeDecodeError as e:
        bad.append((str(p), 0, "文件编码不是 UTF-8: %s" % e, ""))
if bad:
    print("src/ 下有语法错误 —— 先修掉，否则后面的测试连 import 都过不去：")
    for f, ln, msg, txt in bad:
        print("")
        print("  %s:%d" % (f, ln))
        print("    %s" % msg)
        if txt:
            print("    > %s" % txt)
    print("")
    print("修完再跑一次就行，不用清理别的东西。")
    sys.exit(1)
print("语法体检：通过")
PYEOF
}

run_one() {
    local f="$1"
    hr
    echo "### $f      PTQLAB_REFERENCE=$PTQLAB_REFERENCE"
    hr
    "$PY" -u "experiments/$f.py"
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "### $f 退出码 $rc"
        return $rc
    fi
    return 0
}

usage() {
    cat <<'EOF'
PTQ 实验平台 · 启动脚本

  ./run.sh e1      实验 1  基线确认 + BN 折叠自检   (~1 分钟)
  ./run.sh e2      实验 2  RTN 量化阶梯 + 掉点归因  (~1 分钟)
  ./run.sh e3      实验 3  激活校准策略对比         (~2 分钟)
  ./run.sh e4      实验 4  逐层量化敏感度           (~2 分钟)
  ./run.sh e5      实验 5  混合精度（主战场）        (~3 分钟)
  ./run.sh e6      实验 6  AdaRound 取整优化        (~3 分钟)
  ./run.sh e7      实验 7  组合实验 + 稳健性（论文主表）(~6 分钟)

  ./run.sh all     按顺序跑 e1..e7                  (~20 分钟)
  ./run.sh status  汇总 results/ 已有结果
  ./run.sh csv     重建 all.csv（列错位 / 老数据缺失时用）
  ./run.sh check   跑量化算子性质测试（精度够不够）
  ./run.sh edge    跑量化算子边界测试（奇形怪状的输入会不会炸）
  ./run.sh selftest 一键自检：依赖 → 语法 → 性质 → 边界，任一步失败即停
  ./run.sh plot    把结果画成图（论文用，输出到 results/figs/）
  ./run.sh archive 归档旧结果（顶层只留每个实验最新一份，旧的不删，挪进 archive/）
  ./run.sh mine    关掉参考实现，用你自己的实现跑全套

环境变量
  PTQLAB_REFERENCE=1  用参考实现顶替 TODO（默认，保证一跑就有结果）
  PTQLAB_REFERENCE=0  用你自己的 quant.py（TODO 填完再开这个）

建议顺序
  第一次：./run.sh e1       先确认流水线通、BN 折叠 PASS
  然后  ：./run.sh all      拿一整套基线数据
          ./run.sh plot     把结果画成图（论文素材在 results/figs/）
  填完 TODO 后：PTQLAB_REFERENCE=0 ./run.sh mine
EOF
}

case "${1:-}" in
    e1) run_one e1_baseline ;;
    e2) run_one e2_rtn_ladder ;;
    e3) run_one e3_act_calib ;;
    e4) run_one e4_sensitivity ;;
    e5) run_one e5_mixed ;;
    e6) run_one e6_adaround ;;
    e7) run_one e7_combo ;;
    all)
        for s in e1_baseline e2_rtn_ladder e3_act_calib \
                 e4_sensitivity e5_mixed e6_adaround e7_combo; do
            run_one "$s" || exit $?
        done
        echo ""
        hr
        echo "全套跑完，汇总如下："
        "$PY" -u tools/report.py
        ;;
    mine)
        PTQLAB_REFERENCE=0
        export PTQLAB_REFERENCE
        exec "$0" all
        ;;
    status)
        "$PY" -u tools/report.py
        ;;
    csv)
        "$PY" -u tools/rebuild_csv.py
        ;;
    check)
        syntax_gate || exit 1
        "$PY" -u tests/test_quant.py
        ;;
    edge)
        syntax_gate || exit 1
        "$PY" -u tests/test_edge_cases.py
        ;;
    selftest)
        # 一键自检：依赖 → 语法 → 性质 → 边界
        # 退出码 3 = 还有 TODO 没填，这是提醒不是失败。
        # 填的过程中要能继续往下看进度，所以只有 1（有 FAIL）才真的停下来。
        hr
        echo "### 1/4 依赖自检"
        "$PY" -u -c "import config as C; import sys; sys.exit(0 if C.print_asset_report() else 1)" \
            || exit $?
        hr
        echo "### 2/4 语法体检"
        syntax_gate || exit 1
        hr
        echo "### 3/4 算子性质测试"
        "$PY" -u tests/test_quant.py
        rc1=$?
        [ "$rc1" = "1" ] && exit 1
        echo ""
        hr
        echo "### 4/4 算子边界测试"
        "$PY" -u tests/test_edge_cases.py
        rc2=$?
        [ "$rc2" = "1" ] && exit 1
        hr
        if [ "$rc1" = "0" ] && [ "$rc2" = "0" ]; then
            echo "自检全部通过"
        else
            echo "自检结束：还有 TODO 没填（退出码 3 是提醒，不是错误）"
        fi
        exit $(( rc1 | rc2 ))
        ;;
    plot)
        "$PY" -u tools/plot.py
        ;;
    archive)
        "$PY" -u tools/archive.py
        ;;
    clean)
        echo "将清空 results/ 下所有 json 与 all.csv"
        rm -f results/*.json results/all.csv
        echo "已清空"
        ;;
    *)
        usage
        ;;
esac
