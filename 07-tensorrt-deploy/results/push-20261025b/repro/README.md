# push-20261025b · 复现

一条命令一个阶段的原始命令清单。全部在 **WSL2 Ubuntu** 里跑，工作目录
/mnt/d/WorkBuddy/ShowBro/lab/07-tensorrt-deploy。

    P=/home/user/pyenv-torch/bin/python
    export LD_LIBRARY_PATH=/usr/lib/wsl/lib
    R=results/push-20261025b          # 本轮产物（不覆盖 push-20261025）
    B=results/push-20261025            # 上一轮的底稿图（c1x1 折叠后的 fp32/fp16/int8）

环境（实测）：python 3.12.3 / onnx 1.22.0 / onnxruntime 1.31.0 / tensorrt 11.4.0.106 /
torch 2.6.0+cu124 / GTX 1650 Ti 4 GB / driver 621.01。

## 0. 前置：上一轮的 c1x1 底稿

本轮**不重做** 1×1 折叠，直接吃 results/push-20261025 里的三张图
（resnet18_cifar10_{fp32,fp16,int8}_c1x1.onnx）。要重生成它们见 push-20261025/README §10。

## 1. 位置分解（本轮新增）+ ORT 语义校验

    $P scripts/push_decompose.py --in $B/resnet18_cifar10_fp32_c1x1.onnx \
        --out $R/resnet18_cifar10_fp32_dec_all.onnx --verify-ort

    # 归因用的两个单族变体
    $P scripts/push_decompose.py --in $B/resnet18_cifar10_fp32_c1x1.onnx \
        --out $R/resnet18_cifar10_fp32_dec_layer3.onnx --only layer3 --verify-ort
    $P scripts/push_decompose.py --in $B/resnet18_cifar10_fp32_c1x1.onnx \
        --out $R/resnet18_cifar10_fp32_dec_strand.onnx --only strand --verify-ort

    # fp16 侧（权重是 fp16，脚本按原 dtype 切片）
    $P scripts/push_decompose.py --in $B/resnet18_cifar10_fp16_c1x1.onnx \
        --out $R/resnet18_cifar10_fp16_dec_all.onnx --verify-ort --verify-n 100

判据：logits max|Δ| < 1e-6（fp32 图）且预测 200/200 一致。实测 4.77e-7 / 200-200。

## 2. int8 量化（口径与 push-20261025 完全一致）

    for v in all layer3 strand; do
      $P scripts/push_quantize_variant.py \
         --src $R/resnet18_cifar10_fp32_dec_$v.onnx \
         --dst $R/resnet18_cifar10_int8_dec_$v.onnx
    done

校准 = test_batch.raw 的前 1000 张（CifarReader），所以**评测必须 --skip 1000**，否则是在标定样本上测精度。

## 3. 测：单档入口

    bash scripts/push_b_one.sh <onnx> <tag> [push_bench.py 的参数...]

它做三件事：wait_idle.py 等 GPU 安静 → flock -x /tmp/ptq-gpu.lock 拿独占锁 →
push_bench.py 跑（--jsonl/--engine-dir/--timing-cache 全部指向 push-20261025b，不动上一轮产物）。

本轮各档的完整命令（stage 脚本在 work/scratch/stage*_*.sh，同一串命令抄在这里）：

    # 基线（engine 直接复用 push-20261025 的缓存，0 构建）
    bash scripts/push_b_one.sh $B/../resnet18_cifar10_int8_c1x1.onnx int8_base_ab1 \
        --batch 512 --opt 3 --skip 1000 --preboost 100 \
        --engine-path work/push-20261025/engines/resnet18_cifar10_int8_c1x1__b512_o3_default_w1.0_aNone_xNone_d0_f78b02ece7.trt

    # 分解图，opt3（与基线同配置 —— 主 A/B 用这一档）
    bash scripts/push_b_one.sh $R/resnet18_cifar10_int8_dec_all.onnx int8_decall_ab1 \
        --batch 512 --opt 3 --skip 1000 --preboost 100 --profile

    # 旋钮：opt5 / opt5+DIRECT_IO / opt3+DIRECT_IO
    bash scripts/push_b_one.sh $R/resnet18_cifar10_int8_dec_all.onnx int8_decall_o5 \
        --batch 512 --opt 5 --skip 1000 --preboost 100 --profile
    bash scripts/push_b_one.sh $R/resnet18_cifar10_int8_dec_all.onnx int8_decall_o5dio \
        --batch 512 --opt 5 --direct-io --skip 1000 --preboost 100 --profile

    # fp16 侧（opt5 + DIRECT_IO，与 push-20261025 的 fp16 主配置一致）
    bash scripts/push_b_one.sh $R/resnet18_cifar10_fp16_dec_all.onnx fp16_decall_o5dio \
        --batch 512 --opt 5 --direct-io --preboost 100 --profile

## 4. 汇总

    python3 scripts/push_b_report.py     # 主表 + 逐层 A/B（默认读 push-20261025b/bench.jsonl）
    python3 scripts/push_summary.py $R/bench.jsonl

## 5. 复现判据（照 push-20261025/repro 的规矩先写死，再跑）

| 项 | 判据 | 本轮实测 |
|---|---|---|
| 语义等价（fp32） | logits max\|Δ\| < 1e-6、预测 200/200 | 4.77e-7 / 200-200 ✅ |
| int8 精度不掉 | ≥ 87.0%（10 000 张） | 87.21%（原图 87.22%）✅ |
| int8 延迟 | 同场 A/B dataset 中位 ≤ 0.90× 基线 | **0.805 / 0.796×** ✅ |
| 目标 <0.006 | 发布口径折算值 | 0.00595（opt5 0.0058）**擦线**，§6 见 README |

> 关于「同场」：本 session 基线自身 0.00799~0.00823，而发布时是 0.00744（clock/温度都一致，原因未定位）。
> 所以延迟判据用**同 session 的比值**，不用绝对值；跨 session 的绝对比较一律标注为折算。
