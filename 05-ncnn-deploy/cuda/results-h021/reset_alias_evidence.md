# Blob-arena reset alias evidence (H-20261023-021)
# Model: resnet18_cifar10_int8_old, first 20 CIFAR-10 test images, batch 20, warmup 5,
# harness: ncnn_cuda_eval_final (release-before-reset + liveness counters), build-ptq (fork 8918acf).
# Command shape:
#   ./ncnn_cuda_eval_final <param> <bin> test_batch.raw --int8 --limit 20 --batch 20 \
#       --repeat 1 --warmup 5 --reset-every N --dump-wrong 20
#
# reset_every  accuracy        wrong image indices
# 1            55.00% (11/20)  4 6 9 11 12 13 16 17 19
# 2            80.00% (16/20)  5 9 13 18
# 4            90.00% (18/20)  7 12
# 8           100.00% (20/20)  -
# 16          100.00% (20/20)  -
# no-reset    100.00% (20/20)  -
#
# Isolation:
#   reset blob only, keep work arena growing : 55.00% (11/20)
#   reset work only, keep blob arena growing : 100.00% (20/20)
# => the corruption is tied to reusing blob-arena addresses, not workspace.
#
# Liveness counters at end of failing run: blob deferred=0 stale=0 live=1
# => no CudaMat was released after reset and no live block was skipped; the
#    reused address is read by something that is not tracked by the arena.
#
# Logits hash (FNV-1a over the 10 output floats), reset_every=1 vs no-reset:
#   img 0: identical
#   img 1: 5634c6671404d1e9 vs 4486b6edbf210b31  (first divergence)
#
# Independent confirmation with the h019 harness (binary from 76b87e0 lineage):
#   h019 --reset-every 1, 100 images, batch 64: 41.00%
#   h019 --no-reset,      100 images, batch 64: 89.00%
#   h015 source + build-ptq, both reset-every 1 and no-reset: 89.00%
#   (h015 resets before releasing, so its reset path defers and never reuses)
#
# Decision for the 2000-image acceptance: monotonic arena, no reset, each 500-image
# chunk in a fresh process. Addresses are never reused, so this class of alias is
# physically impossible; the accuracy then equals CPU int8 exactly (1739/2000).
