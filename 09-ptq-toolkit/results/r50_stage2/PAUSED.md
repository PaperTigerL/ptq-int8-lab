# R50 stage2 · 已暂停（用户叫停）

- 时间：2026-10-23 10:13 (+08:00)
- 交接单：`H-20261023-007`（保持 working，未 complete）
- 状态：**paused_by_user**，不自动恢复

## 为什么停

用户明确要求暂停整个任务：本轮把 **旧 stage2 协议**（从 ep26 84.52% 起、10 epoch、
lr 0.003、aug+ls）当成「项目优化」执行，属于**复现旧实验**，不是当前项目需要的优化方向。
启动前没有先确认「stage2 的优化目标和指标」，这是我的判断错误。

## 已停止的证据

- 训练进程 PID 419 已 `SIGTERM`，`pgrep -af finetune.py` 为空。
- `nvidia-smi` 计算进程列表：`No running processes found`。
- GPU 前后快照：
  - `D:/WorkBuddy/ShowBro/lab/09-ptq-toolkit/results/r50_stage2/nvidia_smi_before.txt`
  - `D:/WorkBuddy/ShowBro/lab/09-ptq-toolkit/results/r50_stage2/nvidia_smi_after_abort.txt`
- 未删除任何 ckpt、日志或 history。

## 断点（7/10 epoch）

起点：`ckpt_resnet50_ep26.pth`（84.52%，全量 10k 独立复评 delta +0.00pp，exit 0）

| epoch | train | test | seconds | peak MiB |
|---:|---:|---:|---:|---:|
| 1 | 83.11% | 85.16% | 237.1 | 2078.2 |
| 2 | 86.40% | 85.68% | 270.2 | 2078.2 |
| 3 | 87.64% | 85.62% | 378.5 | 2078.2 |
| 4 | 88.94% | 85.88% | 571.9 | 2078.2 |
| 5 | 90.05% | 86.61% | 494.4 | 2078.2 |
| 6 | 91.20% | 87.71% | 415.4 | 2078.2 |
| 7 | 92.18% | 89.00% | 381.6 | 2078.2 |

- 最后落盘：`ckpt_resnet50_r50aug10_ep7.pth`
- ep8–ep10 未训练；ep10 全量复评、链审计、报告、relay、handoff complete 均未做。
- 机器可读断点：`results/r50_stage2/partial_summary.json`

## 未做的事

- 没有生成 `posts/ptq/20261023-r50-stage2.md` 最终报告。
- 没有生成 `out/agent-hub/relay/H-20261023-007.json`，没有 relay-check，没有 complete。
- 没有改 06 C++、没有 git push、没有写 bin/team/status、没有重启 7810/8732。
- stage2 的 ep1–ep7 只是**中间复现数据**，不声明为项目提升，也不引用为最终精度。

## 恢复条件

只有在明确「当前项目要优化的指标/瓶颈」（例如基于已交付 R50 91.85% 的
PTQ/W4A8 端到端、引擎加速、或指定模型指标）并重新授权后，才启动新的 GPU 任务。
旧的 `r50aug10` 断点保留取证，不默认续跑。
