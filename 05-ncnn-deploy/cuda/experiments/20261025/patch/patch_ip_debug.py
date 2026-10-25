#!/usr/bin/env python3
"""Temporary debug print of the int8 InnerProduct shapes."""
import io

PATH = "/home/user/ncnn-cuda/src/layer/cuda/innerproduct_cuda.cu"

INC_OLD = '#include "innerproduct_cuda.h"'
INC_NEW = '#include <cstdio>\n#include <cstdlib>\n\n#include "innerproduct_cuda.h"'

MARK = """int innerproduct_cuda_forward_int8(const CudaMat& bottom_blob, CudaMat& top_blob, const InnerProduct_cuda::InnerProduct_info& info,
                                   float* gpu_scratchpad_memory, int gpu_scratchpad_memory_size)
{
    checkCudaErrors(cudaMemset(gpu_scratchpad_memory, 0, gpu_scratchpad_memory_size));"""

DEBUG = r'''int innerproduct_cuda_forward_int8(const CudaMat& bottom_blob, CudaMat& top_blob, const InnerProduct_cuda::InnerProduct_info& info,
                                   float* gpu_scratchpad_memory, int gpu_scratchpad_memory_size)
{
    if (std::getenv("NCNN_CUDA_IP_DEBUG"))
        printf("IPINT8 a=%dx%dx%d cstep=%d | weight=%dx%dx%d cstep=%d | num_output=%d scratch_mb=%d\n",
               bottom_blob.w, bottom_blob.h, bottom_blob.c, (int)bottom_blob.cstep,
               info.gpu_weight_data->w, info.gpu_weight_data->h, info.gpu_weight_data->c,
               (int)info.gpu_weight_data->cstep, info.num_output, gpu_scratchpad_memory_size >> 20);

    checkCudaErrors(cudaMemset(gpu_scratchpad_memory, 0, gpu_scratchpad_memory_size));'''

with io.open(PATH, "r", encoding="utf-8") as f:
    text = f.read()
if MARK not in text:
    raise SystemExit("marker missing")
text = text.replace(MARK, DEBUG, 1)
if INC_OLD in text:
    text = text.replace(INC_OLD, INC_NEW, 1)
with io.open(PATH, "w", encoding="utf-8", newline="\n") as f:
    f.write(text)
print("IP debug added to int8 launcher")
