#!/usr/bin/env python3
"""Create Clip and LeakyReLU param variants for activation-fusion tests."""
import io

SRC = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/models/resnet18_cifar10_int8_old.param"
OUT = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda/results-h028"
LINE = "ReLU                     /relu/Relu               1 1 /conv1/Conv_output_0 /relu/Relu_output_0"

text = io.open(SRC, encoding="utf-8").read()
if LINE not in text:
    raise SystemExit("ReLU line not found")

clip = text.replace(LINE, "Clip                     /relu/Relu               1 1 /conv1/Conv_output_0 /relu/Relu_output_0 0=0 1=100")
leaky = text.replace(LINE, LINE + " 0=0.1")

io.open(OUT + "/resnet18_cifar10_int8_old_clip.param", "w", encoding="utf-8", newline="\n").write(clip)
io.open(OUT + "/resnet18_cifar10_int8_old_leaky.param", "w", encoding="utf-8", newline="\n").write(leaky)
print("wrote clip/leaky params")
