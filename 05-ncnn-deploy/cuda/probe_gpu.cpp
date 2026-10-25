// 探针：确认 ncnn 内部看到的 CUDA 设备数到底是多少。
//
// 为什么需要它：create_layer() 选 CUDA 层的唯一条件是 ncnn::get_cuda_gpu_count() > 0，
// 跟 opt.use_cuda_compute 无关。日志显示所有层都建成了 CPU 版，
// 所以必须直接问一句：这个数到底是多少。
#include <cstdio>
#include <cuda_runtime.h>

#include "gpu.h"
#include "net.h"

int main()
{
    int rt_count = 0;
    cudaError_t err = cudaGetDeviceCount(&rt_count);
    printf("cudaGetDeviceCount -> %d (err=%d: %s)\n",
           rt_count, (int)err, cudaGetErrorString(err));

    printf("ncnn::get_cuda_gpu_count() -> %d\n", ncnn::get_cuda_gpu_count());

    // 顺便看 create_layer 到底建的是什么
    ncnn::Layer* l = ncnn::create_layer("Convolution");
    printf("create_layer(Convolution) -> %p, support_cuda=%d\n",
           (void*)l, l ? (int)l->support_cuda : -1);
    delete l;
    return 0;
}
