#!/usr/bin/env python3
"""Replace global average pooling with a lock-free one-thread-per-channel kernel.

Old path per image: cudaMemset(top, 0) + one block per channel with a global
spin lock around `output[channel] += rowsum` + a second kernel for division.
For 512 channels the lock serialises every channel.

New path: one kernel, one thread per channel, writes the result directly.  The
float summation order is unchanged (row-major sums, then sum of row sums, then
divide by H*W), so results are bit-identical.  Padding columns/rows in the old
kernel contributed exact +0.0f terms and are simply skipped.
"""
import io

PATH = "/home/user/ncnn-cuda/src/layer/cuda/pooling_cuda.cu"
ANCHOR = "int pooling_cuda_forward_global(const CudaMat& bottom_blob, CudaMat& top_blob, const Pooling_cuda::Pooling_info& info)"

NEW_KERNEL = r'''// 全局平均池化：一个线程负责一个通道。
// 求和顺序保持与旧实现一致：先算每行内和，再逐行累加，最后除以 H*W；
// 旧实现里补到 32 列的 0 和补到 8 行的 0 都是精确的 +0.0f，这里直接跳过。
// 省掉 memset、每通道一个 block 的全局锁累加，以及第二个除法 kernel。
__global__ void gpu_pooling_cuda_forward_global_avg_fast(const float* __restrict__ a_input,
                                                         const ncnn::CudaMatInfo a_info,
                                                         float* __restrict__ output)
{
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= a_info.c) return;

    const float* p = a_input + (size_t)channel * a_info.cstep;
    float total = 0.f;
    for (int row = 0; row < a_info.h; row++)
    {
        float row_sum = 0.f;
        const float* q = p + (size_t)row * a_info.w;
        for (int col = 0; col < a_info.w; col++)
            row_sum += q[col];
        total += row_sum;
    }
    output[channel] = total / (float)(a_info.w * a_info.h);
}

'''

OLD_AVE = """    else if (info.pooling_type == Pooling::PoolMethod_AVE)
    {
        top_blob.fill(0);
        gpu_pooling_cuda_forward_global_sum<<<grid_size, block_size, block_size.x * block_size.y * sizeof(float)>>>(static_cast<const float*>(bottom_blob.get_craw_data()),
                                                                                                                      bottom_blob_info,
                                                                                                                      static_cast<float*>(top_blob.get_raw_data()),
                                                                                                                      top_blob_info,
                                                                                                                      info, info.pooling_lock);

        const int thread_per_block_z_sum = 16;

        const dim3 block_size_sum(1, 1, thread_per_block_z_sum);
        const dim3 grid_size_sum(1, 1, (total_number_of_channels - 1) / thread_per_block_z_sum + 1);

        gpu_pooling_cuda_forward_global_sum_total<<<grid_size_sum, block_size_sum>>>(static_cast<float*>(top_blob.get_raw_data()),
                                                                                       bottom_blob_info,
                                                                                       (bottom_blob_info.w * bottom_blob_info.h));
    }"""

NEW_AVE = """    else if (info.pooling_type == Pooling::PoolMethod_AVE)
    {
        const int threads = 128;
        const dim3 grid_avg((total_number_of_channels + threads - 1) / threads);
        gpu_pooling_cuda_forward_global_avg_fast<<<grid_avg, threads>>>(
            static_cast<const float*>(bottom_blob.get_craw_data()), bottom_blob_info,
            static_cast<float*>(top_blob.get_raw_data()));
    }"""


def main():
    with io.open(PATH, "r", encoding="utf-8") as f:
        text = f.read()
    if OLD_AVE not in text:
        raise SystemExit("AVE branch marker not found")
    text = text.replace(OLD_AVE, NEW_AVE)
    pos = text.index(ANCHOR)
    text = text[:pos] + NEW_KERNEL + text[pos:]
    with io.open(PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print("patched", PATH)


if __name__ == "__main__":
    main()
