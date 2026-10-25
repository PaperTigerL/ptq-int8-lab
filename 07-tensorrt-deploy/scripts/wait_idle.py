"""等 GPU 安静再测 —— 短超时 + 绝不跳过。

⭐ 本机的真实情况（2026-10-20 实测）：
   · **空闲时** util 在 10~18%（桌面合成器 + 常驻 Electron 窗口，
     WorkBuddy / 豆包 / WXWork / DingTalk / B站 等，约 790 MiB。
     它们是图形进程，nvidia-smi 的 compute-apps 里查不到）
   · **有真实计算负载时** util 稳定在 95~100%

   所以绝对阈值 `< 20%` 是**有效的**：能区分「图形底噪」和「真实负载」。

⚠️ 为什么**不做**「相对基线」（踩过一次，很 instructive）：
   曾经想用 `util <= baseline + 8`、baseline = 开跑前 10 次采样的 min。
   实测失败：
     采样 util = [95, 97, 96, 96, 87, 100, 36, 0, 100, 100]
     → min = 0  → 判据退化成 util <= 8%
   **min 恰好采到一次 0%**，把基线拉到 0，相对判据就塌成了绝对阈值 ——
   比不做还糟。而 median 在这种样本下是 100，判据变成 util <= 108，
   等于没有门控。两头都会坏。

   结论：**「估基线」这个思路本身不对**。底噪不是稳定常数，
   猜它不如直接用物理上有意义的绝对阈值（20%恰好落在 10~18 与 95~100 之间）。

⚠️ **绝不 SKIP**：等不到就照常测，事后按 util 分档。
   跳过 = 白跑；测了并标注 = 至少拿到「被占用时的下界」，仍有参考价值。

用法
    python wait_idle.py                    # 默认 util<=20% 连续 3 次，最多 60s
    python wait_idle.py --thresh 30# 机器比较空时放宽
    python wait_idle.py --max-wait 30        # 没耐心
"""
import argparse
import subprocess
import sys
import time


def sample():
    """返回 (显存 MiB, 利用率 %)。

    ⚠️ 字段顺序与 query 的书写顺序一致：memory.used 在前、utilization.gpu 在后。
       别写 `u, m = sample()` —— 那是把显存当 util，
       实测会算出「baseline = 2806%」这种数（显存 2804 MiB 被当成了 util）。
    """
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=15).stdout.strip().splitlines()[0]
    used, util = (int(float(x)) for x in out.split(","))
    # 防御：util 不可能超过 100，若超过说明字段取反了
    if util > 100:
        used, util = util, used
    return used, util


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresh", type=int, default=20,
                    help="安静判据 util<=thresh（默认 20：区分图形底噪 10~18%% "
                         "与真实负载 95~100%%）")
    ap.add_argument("--samples", type=int, default=3, help="连续多少次都要安静")
    ap.add_argument("--max-wait", type=int, default=60,
                    help="最多等多少秒（默认 60；等不到也不该放弃测量）")
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()

    try:
        used, util = sample()
    except Exception as e:
        print("nvidia-smi 读不到:", e)
        return 2

    print(f"判据：util <= {args.thresh}% 连续 {args.samples} 次"
          f"（本机空闲时 10~18% 是图形进程底噪，真实负载 95~100%）")
    print(f"起始采样：util {util}%, mem {used} MiB")
    print(f"最多等 {args.max_wait}s；**等不到也照常测量**（跳过 = 白跑）")

    t0 = time.time()
    streak = 0
    while time.time() - t0 < args.max_wait:
        try:
            used, util = sample()
        except Exception as e:
            print("nvidia-smi 读不到:", e)
            return 2
        if util <= args.thresh:
            streak += 1
            print(f"  安静 {streak}/{args.samples}  "
                  f"(util {util}% <= {args.thresh}%, mem {used} MiB)")
            if streak >= args.samples:
                print(f"GPU 安静，等了 {time.time() - t0:.0f}s")
                print(f"GATE_UTIL={util}")
                return 0
        else:
            if streak:
                print("  又有负载，计数清零")
            streak = 0
            print(f"  忙(util {util}% > {args.thresh}%)", end="")
        time.sleep(args.interval)

    used, util = sample()
    print()
    print(f"等了 {args.max_wait}s 没等到安静窗口"
          f"（当前 util {util}%, mem {used} MiB, 判据 <= {args.thresh}%）")
    print(f"GATE_UTIL={util}")
    print("→ 退出码 1。**调用方应该照常测量**，把 util 记进注释，")
    print("   事后按 util 分档统计（见 summarize_idle.py）。")
    print("   被占用时测到的数字是**下界**，仍有参考价值 —— 前提是如实标注。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
