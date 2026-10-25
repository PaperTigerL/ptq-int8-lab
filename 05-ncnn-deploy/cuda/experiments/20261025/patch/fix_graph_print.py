#!/usr/bin/env python3
import io
p = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda/ncnn_cuda_graph_eval.cpp"
t = io.open(p, encoding="utf-8").read()
old = '''                        printf("  round 1: GPU %.4f ms/img | 墙钟 %.4f ms/img "
                               "(H2D 0.0000 + D2H 0.0000) | 精度 %.2f%% | 显存已用 %.1f MB\\n",
                               gpu_per, wall_per, 100.0 * gok / n,
                               (double)(n * 12288) / 1048576.0);'''
new = '''                        printf("  round 1: GPU %.4f ms/img | 墙钟 %.4f ms/img "
                               "(H2D+D2H 含在墙钟) | 精度 %.2f%% | 显存已用 %.1f MB\\n",
                               gpu_per, wall_per, 100.0 * gok / n,
                               g_blob_arena.bytes_used() / 1048576.0);'''
assert old in t, "marker missing"
io.open(p, "w", encoding="utf-8", newline="\n").write(t.replace(old, new, 1))
print("fixed print")
