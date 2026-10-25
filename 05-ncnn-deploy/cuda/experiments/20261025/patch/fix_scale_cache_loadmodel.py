#!/usr/bin/env python3
"""One-off: add the missing cache invalidation to the second load_model overload."""
import io

PATH = "/home/user/ncnn-cuda/src/layer/cuda/convolution_cuda.cpp"
OLD = """        checkCudaErrors(cudaMemcpy(gpu_top_blob_int8_scale, &top_blob_int8_scale, sizeof(float), cudaMemcpyHostToDevice));
    }

    return 0;
}

void Convolution_cuda::make_padding"""
NEW = """        checkCudaErrors(cudaMemcpy(gpu_top_blob_int8_scale, &top_blob_int8_scale, sizeof(float), cudaMemcpyHostToDevice));
        int8_scale_cache_valid = false;
    }

    return 0;
}

void Convolution_cuda::make_padding"""

with io.open(PATH, "r", encoding="utf-8") as f:
    text = f.read()
if OLD not in text:
    raise SystemExit("marker missing")
text = text.replace(OLD, NEW, 1)
with io.open(PATH, "w", encoding="utf-8", newline="\n") as f:
    f.write(text)
print("fixed second load_model")
