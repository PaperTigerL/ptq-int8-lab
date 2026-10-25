#!/usr/bin/env python3
"""Fuse ReLU(slope=0)/Clip into the producing int8 Convolution epilogue.

- runs at the end of Net::fuse_network, after the int8 requantize pass;
- only int8, groups==1 Convolutions with activation_type==0 and a single
  ReLU/Clip consumer are fused;
- consumers of the activation's top blob are rewired to the conv's top blob,
  so the activation layer is no longer on the execution path (if it is
  requested explicitly it still runs and is idempotent);
- NCNN_CUDA_FUSE_ACT=0 disables the pass for A/B controls;
- the conv epilogue gains Clip (activation_type=3) via activation_params.
"""
import io

NET = "/home/user/ncnn-cuda/src/net.cpp"
CU_H = "/home/user/ncnn-cuda/src/layer/cuda/implicit_conv_cuda.h"
CU_CU = "/home/user/ncnn-cuda/src/layer/cuda/implicit_conv_cuda.cu"
CU_CPP = "/home/user/ncnn-cuda/src/layer/cuda/convolution_cuda.cpp"

# ---------- net.cpp ----------
NET_INC_OLD = '#include "relu.h"'
NET_INC_NEW = '#include <stdio.h>\n#include <stdlib.h>\n\n#include "relu.h"\n#include "clip.h"'

NET_PASS = r'''
    // 把 int8 Conv 后接的 ReLU(slope=0)/Clip 融合进卷积 epilogue。
    // 放在 int8 scale 传播之后，requantize 决策不受影响。
    const char* fuse_act_env = getenv("NCNN_CUDA_FUSE_ACT");
    if (fuse_act_env && fuse_act_env[0] == '0' && fuse_act_env[1] == '\0')
        return 0;

    for (size_t i = 0; i < layers.size(); i++)
    {
        Layer* layer = layers[i];
        if (layer->type != "Convolution") continue;

        Convolution* conv = (Convolution*)layer;
        if (conv->weight_data.elemsize != 1u || conv->int8_scale_term == 0 || conv->groups != 1) continue;
        if (conv->activation_type != 0) continue;
        if (layer->tops.size() != 1) continue;

        const int top = layer->tops[0];
        if (blobs[top].consumers.size() != 1) continue;

        const int act_index = blobs[top].consumers[0];
        Layer* act = layers[act_index];
        if (act->bottoms.size() != 1 || act->tops.size() != 1) continue;
        if (act->bottoms[0] != top) continue;

        int act_type = 0;
        if (act->type == "ReLU")
        {
            if (((ReLU*)act)->slope != 0.0f) continue;
            act_type = 1;
        }
        else if (act->type == "Clip")
        {
            act_type = 3;
        }
        else
        {
            continue;
        }

        const int act_top = act->tops[0];
        if (act_top == top) continue;

        conv->activation_type = act_type;
        if (act_type == 3)
        {
            conv->activation_params = Mat(2, (size_t)4u);
            conv->activation_params[0] = ((Clip*)act)->min;
            conv->activation_params[1] = ((Clip*)act)->max;
        }

        // 消费者改读 conv 的 top blob；激活层不再位于执行路径上。
        const std::vector<int> act_consumers = blobs[act_top].consumers;
        for (size_t c = 0; c < act_consumers.size(); c++)
        {
            Layer* next = layers[act_consumers[c]];
            for (size_t b = 0; b < next->bottoms.size(); b++)
                if (next->bottoms[b] == act_top) next->bottoms[b] = top;
            blobs[top].consumers.push_back(act_consumers[c]);
        }
        blobs[act_top].consumers.clear();

        if (getenv("NCNN_CUDA_FUSE_ACT_DEBUG"))
            fprintf(stderr, "fuse act: %s + %s %s -> activation_type=%d\n",
                    layer->name.c_str(), act->type.c_str(), act->name.c_str(), act_type);
    }

    return 0;
}
'''

NET_ANCHOR_OLD = """#endif
    return 0;
}

void Net::clear()"""
NET_ANCHOR_NEW = """#endif
""" + NET_PASS + """
void Net::clear()"""

# ---------- convolution_cuda.cpp: re-upload activation params after fusion ----------
CPP_OLD = """    int8_scale_cache_valid = false;

    return 0;
}

int Convolution_cuda::forward"""
CPP_NEW = """    int8_scale_cache_valid = false;

    // 图融合可能在 load_param 之后才写入 activation_params，这里补一次上传；
    // implicit / fallback kernel 的 Clip epilogue 都读这块显存。
    if (activation_type == 3 && !activation_params.empty())
        gpu_activation_params = CudaMat{activation_params, _cuda_allocator};

    return 0;
}

int Convolution_cuda::forward"""

# ---------- implicit header ----------
H_LAUNCH_OLD = """                               bool requantize, int activation_type,
                               void* output, int out_cstep);"""
H_LAUNCH_NEW = """                               bool requantize, int activation_type,
                               const float* activation_params,
                               void* output, int out_cstep);"""
H_REDUCE_OLD = """                                      bool requantize, int activation_type,
                                      void* output, int out_cstep);"""
H_REDUCE_NEW = """                                      bool requantize, int activation_type,
                                      const float* activation_params,
                                      void* output, int out_cstep);"""

# ---------- implicit .cu kernel + launchers ----------
KERNEL_SIG_OLD = """                                          bool requantize, int activation_type,
                                          void* __restrict__ output, int out_cstep)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int M = OH * OW;
    const int m = blockIdx.x * 8 + warp;"""
KERNEL_SIG_NEW = """                                          bool requantize, int activation_type,
                                          const float* __restrict__ activation_params,
                                          void* __restrict__ output, int out_cstep)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int M = OH * OW;
    const int m = blockIdx.x * 8 + warp;"""

EPILOGUE_OLD = """    if (requantize)
    {
        signed char q = float2int8_ncnn(y * (*top_scale));
        if (activation_type == 1 && q < 0) q = 0;
        reinterpret_cast<signed char*>(output)[oidx] = q;
    }
    else
    {
        if (activation_type == 1 && y < 0.0f) y = 0.0f;
        reinterpret_cast<float*>(output)[oidx] = y;
    }"""
EPILOGUE_NEW = """    if (requantize)
    {
        signed char q = float2int8_ncnn(y * (*top_scale));
        if (activation_type == 1 && q < 0) q = 0;
        else if (activation_type == 3 && activation_params)
        {
            const signed char qmin = float2int8_ncnn(activation_params[0]);
            const signed char qmax = float2int8_ncnn(activation_params[1]);
            if (q < qmin) q = qmin;
            if (q > qmax) q = qmax;
        }
        reinterpret_cast<signed char*>(output)[oidx] = q;
    }
    else
    {
        if (activation_type == 1 && y < 0.0f) y = 0.0f;
        else if (activation_type == 3 && activation_params)
        {
            y = fminf(fmaxf(y, activation_params[0]), activation_params[1]);
        }
        reinterpret_cast<float*>(output)[oidx] = y;
    }"""

LAUNCH_OLD = """    implicit_conv_int8_kernel<<<grid, 256>>>(
        packed, wimp, wscales, bias, bottom_scale, top_scale,
        C4, H, W, Cout, kh, kw, dh, dw, sh, sw, OH, OW,
        requantize, activation_type, output, out_cstep);"""
LAUNCH_NEW = """    implicit_conv_int8_kernel<<<grid, 256>>>(
        packed, wimp, wscales, bias, bottom_scale, top_scale,
        C4, H, W, Cout, kh, kw, dh, dw, sh, sw, OH, OW,
        requantize, activation_type, activation_params, output, out_cstep);"""

REDUCE_SIG_OLD = """                                                 bool requantize, int activation_type,
                                                 void* __restrict__ output, int out_cstep,
                                                 int total, int Cout)"""
REDUCE_SIG_NEW = """                                                 bool requantize, int activation_type,
                                                 const float* __restrict__ activation_params,
                                                 void* __restrict__ output, int out_cstep,
                                                 int total, int Cout)"""

REDUCE_EPI_OLD = """    if (requantize)
    {
        signed char q = float2int8_ncnn(y * (*top_scale));
        if (activation_type == 1 && q < 0) q = 0;
        reinterpret_cast<signed char*>(output)[oidx] = q;
    }
    else
    {
        if (activation_type == 1 && y < 0.0f) y = 0.0f;
        reinterpret_cast<float*>(output)[oidx] = y;
    }"""
REDUCE_EPI_NEW = EPILOGUE_NEW

REDUCE_LAUNCH_OLD = """    implicit_conv_int8_reduce_kernel<<<(total + 255) / 256, 256>>>(
        partial, split, wscales, bias, bottom_scale, top_scale,
        requantize, activation_type, output, out_cstep, total, Cout);"""
REDUCE_LAUNCH_NEW = """    implicit_conv_int8_reduce_kernel<<<(total + 255) / 256, 256>>>(
        partial, split, wscales, bias, bottom_scale, top_scale,
        requantize, activation_type, activation_params, output, out_cstep, total, Cout);"""

# ---------- convolution_cuda.cpp call sites ----------
ACT_PARAM_DEF_OLD = """        const int Ktot = C4 * kernel_h * kernel_w;"""
ACT_PARAM_DEF_NEW = """        const float* act_params = (activation_type == 3 && !gpu_activation_params.empty())
            ? static_cast<const float*>(gpu_activation_params.get_craw_data()) : nullptr;

        const int Ktot = C4 * kernel_h * kernel_w;"""

CALL_REDUCE_OLD = """                    gpu_bottom_blob_int8_scale, gpu_top_blob_int8_scale,
                    use_int8_requantize, activation_type,
                    top_blob.get_raw_data(), (int)top_blob.cstep);
                return;"""
CALL_REDUCE_NEW = """                    gpu_bottom_blob_int8_scale, gpu_top_blob_int8_scale,
                    use_int8_requantize, activation_type, act_params,
                    top_blob.get_raw_data(), (int)top_blob.cstep);
                return;"""

CALL_DEFAULT_OLD = """            gpu_bottom_blob_int8_scale, gpu_top_blob_int8_scale,
            num_output, kernel_w, kernel_h, dilation_w, dilation_h,
            stride_w, stride_h, OW_, OH_,
            use_int8_requantize, activation_type,
            top_blob.get_raw_data(), (int)top_blob.cstep);"""
CALL_DEFAULT_NEW = """            gpu_bottom_blob_int8_scale, gpu_top_blob_int8_scale,
            num_output, kernel_w, kernel_h, dilation_w, dilation_h,
            stride_w, stride_h, OW_, OH_,
            use_int8_requantize, activation_type, act_params,
            top_blob.get_raw_data(), (int)top_blob.cstep);"""


def repl(path, old, new, count=1):
    with io.open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if old not in text:
        raise SystemExit("marker missing in %s: %r" % (path, old[:80]))
    text = text.replace(old, new, count)
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def main():
    repl(NET, NET_INC_OLD, NET_INC_NEW)
    repl(NET, NET_ANCHOR_OLD, NET_ANCHOR_NEW)
    repl(CU_CPP, CPP_OLD, CPP_NEW)
    repl(CU_H, H_LAUNCH_OLD, H_LAUNCH_NEW)
    repl(CU_H, H_REDUCE_OLD, H_REDUCE_NEW)
    repl(CU_CU, KERNEL_SIG_OLD, KERNEL_SIG_NEW)
    repl(CU_CU, EPILOGUE_OLD, EPILOGUE_NEW)
    repl(CU_CU, LAUNCH_OLD, LAUNCH_NEW)
    repl(CU_CU, REDUCE_SIG_OLD, REDUCE_SIG_NEW)
    repl(CU_CU, REDUCE_EPI_OLD, REDUCE_EPI_NEW)
    repl(CU_CU, REDUCE_LAUNCH_OLD, REDUCE_LAUNCH_NEW)
    repl(CU_CPP, ACT_PARAM_DEF_OLD, ACT_PARAM_DEF_NEW)
    repl(CU_CPP, CALL_REDUCE_OLD, CALL_REDUCE_NEW)
    repl(CU_CPP, CALL_DEFAULT_OLD, CALL_DEFAULT_NEW)
    # the implicit path must accept the fused Clip activation as well
    repl(CU_CPP, "(activation_type == 0 || activation_type == 1)",
         "(activation_type == 0 || activation_type == 1 || activation_type == 3)", 2)
    print("patched ReLU/Clip fusion")


if __name__ == "__main__":
    main()
