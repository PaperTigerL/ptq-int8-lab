#!/usr/bin/env python3
"""Cache the last uploaded int8 scales per convolution layer.

forward_int8 used to upload two 4-byte floats every call (2 cudaMemcpy per
conv per image).  The device values only change when the host scales change, so
keep the last uploaded pair and skip the copies when the bits are identical.

Invalidation:
  - constructor / create_pipeline / both load_model overloads clear the cache;
  - forward_int8 additionally compares the current bits, so any external change
    (scale reload, weight reload, NaN, -0.0) uploads again.

Correctness: when the bits match, the device buffer already holds exactly those
bits; the kernel therefore sees the same values as before -> bit-identical.
"""
import io

H_PATH = "/home/user/ncnn-cuda/src/layer/cuda/convolution_cuda.h"
CPP_PATH = "/home/user/ncnn-cuda/src/layer/cuda/convolution_cuda.cpp"

H_OLD = """    float* gpu_bottom_blob_int8_scale;
    float* gpu_top_blob_int8_scale;"""
H_NEW = """    float* gpu_bottom_blob_int8_scale;
    float* gpu_top_blob_int8_scale;
    // 上一次成功上传到 device 的两个 int8 scale（按位比较）；未上传时 valid=false。
    mutable float cached_bottom_blob_int8_scale;
    mutable float cached_top_blob_int8_scale;
    mutable bool int8_scale_cache_valid;"""

CTOR_OLD = """    gpu_bottom_blob_int8_scale = static_cast<float *>(_cuda_allocator->fastMalloc(sizeof(float)));
    gpu_top_blob_int8_scale = static_cast<float *>(_cuda_allocator->fastMalloc(sizeof(float)));"""
CTOR_NEW = """    gpu_bottom_blob_int8_scale = static_cast<float *>(_cuda_allocator->fastMalloc(sizeof(float)));
    gpu_top_blob_int8_scale = static_cast<float *>(_cuda_allocator->fastMalloc(sizeof(float)));
    cached_bottom_blob_int8_scale = 0.0f;
    cached_top_blob_int8_scale = 0.0f;
    int8_scale_cache_valid = false;"""

# Invalidate at the end of create_pipeline, right before returning 0.
CREATE_OLD = """        }
    }

    return 0;
}

int Convolution_cuda::forward(const CudaMat& bottom_blob, CudaMat& top_blob, const Option& opt) const"""
CREATE_NEW = """        }
    }

    // pipeline 重建后 device 侧 buffer 仍存在，但宿主 scale 可能已被重新加载；
    // 先失效，首次 forward 再按值上传一次。
    int8_scale_cache_valid = false;

    return 0;
}

int Convolution_cuda::forward(const CudaMat& bottom_blob, CudaMat& top_blob, const Option& opt) const"""

# Both load_model overloads: invalidate after the two uploads.
LOAD_OLD = """        gpu_weight_data_int8_scales = CudaMat{weight_data_int8_scales, _cuda_allocator};
        checkCudaErrors(cudaMemcpy(gpu_bottom_blob_int8_scale, &bottom_blob_int8_scale, sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(gpu_top_blob_int8_scale, &top_blob_int8_scale, sizeof(float), cudaMemcpyHostToDevice));"""
LOAD_NEW = """        gpu_weight_data_int8_scales = CudaMat{weight_data_int8_scales, _cuda_allocator};
        checkCudaErrors(cudaMemcpy(gpu_bottom_blob_int8_scale, &bottom_blob_int8_scale, sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(gpu_top_blob_int8_scale, &top_blob_int8_scale, sizeof(float), cudaMemcpyHostToDevice));
        int8_scale_cache_valid = false;"""

FWD_OLD = """    // net.cpp 在 load_model 之后才按拓扑设置 top_blob_int8_scale /
    // use_int8_requantize；load_model 里那次上传拿到的是 stale 0。
    // 每个 forward 重传两个 4B 标量，CPU 侧字段此时才是最终值。
    checkCudaErrors(cudaMemcpy(gpu_bottom_blob_int8_scale, &bottom_blob_int8_scale,
                               sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(gpu_top_blob_int8_scale, &top_blob_int8_scale,
                               sizeof(float), cudaMemcpyHostToDevice));"""
FWD_NEW = """    // 只在 scale 的位模式变化时上传；load_model/create_pipeline 会清缓存，
    // 因此权重或 scale 重载后第一次 forward 必然重传。位相同则 device 上
    // 已经是同一组值，kernel 看到的输入不变（整数累加与 epilogue 逐位不变）。
    if (!int8_scale_cache_valid
        || std::memcmp(&cached_bottom_blob_int8_scale, &bottom_blob_int8_scale, sizeof(float)) != 0
        || std::memcmp(&cached_top_blob_int8_scale, &top_blob_int8_scale, sizeof(float)) != 0)
    {
        checkCudaErrors(cudaMemcpy(gpu_bottom_blob_int8_scale, &bottom_blob_int8_scale,
                                   sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(gpu_top_blob_int8_scale, &top_blob_int8_scale,
                                   sizeof(float), cudaMemcpyHostToDevice));
        cached_bottom_blob_int8_scale = bottom_blob_int8_scale;
        cached_top_blob_int8_scale = top_blob_int8_scale;
        int8_scale_cache_valid = true;
    }"""

INC_OLD = '#include "layer_type.h"\n#include "convolution_cuda.h"'
INC_NEW = '#include <cstring>\n\n#include "layer_type.h"\n#include "convolution_cuda.h"'


def main():
    for path, old, new in ((H_PATH, H_OLD, H_NEW), (CPP_PATH, INC_OLD, INC_NEW),
                           (CPP_PATH, CTOR_OLD, CTOR_NEW), (CPP_PATH, CREATE_OLD, CREATE_NEW),
                           (CPP_PATH, FWD_OLD, FWD_NEW)):
        with io.open(path, "r", encoding="utf-8") as f:
            text = f.read()
        if old not in text:
            raise SystemExit("marker missing in %s: %r" % (path, old[:60]))
        text = text.replace(old, new, 1)
        with io.open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)

    # 两个 load_model 重载都要失效（上面循环只替换第一处）。
    with io.open(CPP_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    n = text.count(LOAD_OLD)
    if n != 2:
        raise SystemExit("expected 2 load_model blocks, found %d" % n)
    text = text.replace(LOAD_OLD, LOAD_NEW)
    with io.open(CPP_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print("patched scale cache")


if __name__ == "__main__":
    main()
