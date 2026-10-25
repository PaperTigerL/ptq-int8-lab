#!/usr/bin/env python3
"""H-039: make raw int8 weight H2D lazy (opt-in NCNN_CUDA_LAZY_W8=1).

Convolution_cuda::load_model always uploads weight_data. For int8 weights the
default implicit kernel reads gpu_weight_data_implicit (packed), so that raw
upload is dead weight on the default path (~11.8MB + 83 calls at load). Upload
it on first use instead (fp32 forward or int8 fallback kernel).
"""
import io
import sys

CPP = "/home/user/ncnn-cuda/src/layer/cuda/convolution_cuda.cpp"
HDR = "/home/user/ncnn-cuda/src/layer/cuda/convolution_cuda.h"

OLD_LOAD = """    gpu_weight_data = CudaMat{weight_data, _cuda_allocator};

    if (bias_term)"""

NEW_LOAD = """    // 默认 implicit kernel 只读打包权重；原始 int8 权重只有 fp32/回退路径用。
    // NCNN_CUDA_LAZY_W8=1 时跳过预上传（约 11.8MB + 每层 1 次 H2D），首次用到再补。
    static const bool lazy_w8 = []() {
        const char* e = std::getenv("NCNN_CUDA_LAZY_W8");
        return e && e[0] == '1' && e[1] == '\\0';
    }();
    if (!lazy_w8 || weight_data.elemsize != (size_t)1u)
        gpu_weight_data = CudaMat{weight_data, _cuda_allocator};

    if (bias_term)"""

OLD_HELPER_ANCHOR = "int Convolution_cuda::create_pipeline(const Option& opt)\n"
NEW_HELPER = """void Convolution_cuda::ensure_gpu_weight_data() const
{
    if (!gpu_weight_data.empty() || weight_data.empty())
        return;
    gpu_weight_data = CudaMat{weight_data, _cuda_allocator};
}

int Convolution_cuda::create_pipeline(const Option& opt)
"""

OLD_DECL = "    int forward_int8(const CudaMat& bottom_blob, CudaMat& top_blob, const Option& opt) const;\n"
NEW_DECL = OLD_DECL + "\n    // 懒上传原始权重（NCNN_CUDA_LAZY_W8=1）：implicit 路径不读它。\n    void ensure_gpu_weight_data() const;\n"

OLD_MEMBER = "    CudaMat gpu_weight_data;\n"
NEW_MEMBER = "    mutable CudaMat gpu_weight_data;\n"

OLD_FWD = """//    int result = convolution_cuda_forward_03(bottom_blob_bordered, top_blob, Convolution_info(*this, gpu_space_ofs));
"""
NEW_FWD = """    ensure_gpu_weight_data();

//    int result = convolution_cuda_forward_03(bottom_blob_bordered, top_blob, Convolution_info(*this, gpu_space_ofs));
"""

OLD_FALLBACK = """    int *gpu_space_ofs = static_cast<int*>(opt.blob_cuda_allocator->fastMalloc(_space_ofs.size() * sizeof(int)));
    checkCudaErrors(cudaMemcpy(gpu_space_ofs, _space_ofs.data(), _space_ofs.size() * sizeof(int), cudaMemcpyHostToDevice));
    int result = convolution_cuda_forward_int8(bottom_blob_bordered, top_blob, Convolution_info(*this, gpu_space_ofs));
"""
NEW_FALLBACK = """    ensure_gpu_weight_data();

    int *gpu_space_ofs = static_cast<int*>(opt.blob_cuda_allocator->fastMalloc(_space_ofs.size() * sizeof(int)));
    checkCudaErrors(cudaMemcpy(gpu_space_ofs, _space_ofs.data(), _space_ofs.size() * sizeof(int), cudaMemcpyHostToDevice));
    int result = convolution_cuda_forward_int8(bottom_blob_bordered, top_blob, Convolution_info(*this, gpu_space_ofs));
"""


def patch(path, pairs, count=1):
    text = io.open(path, encoding="utf-8").read()
    for old, new, n in pairs:
        got = text.count(old)
        if got != n:
            sys.exit("marker count %d != %d in %s: %r" % (got, n, path, old[:70]))
        text = text.replace(old, new)
    io.open(path, "w", encoding="utf-8", newline="\n").write(text)


patch(CPP, [
    (OLD_LOAD, NEW_LOAD, 2),
    (OLD_HELPER_ANCHOR, NEW_HELPER, 1),
    (OLD_FWD, NEW_FWD, 1),
    (OLD_FALLBACK, NEW_FALLBACK, 1),
])
patch(HDR, [
    (OLD_DECL, NEW_DECL, 1),
    (OLD_MEMBER, NEW_MEMBER, 1),
])
print("patched lazy_w8")
