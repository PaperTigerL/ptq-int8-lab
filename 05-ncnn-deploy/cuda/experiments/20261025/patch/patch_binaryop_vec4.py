#!/usr/bin/env python3
"""Vectorize the same-shape contiguous BinaryOp path with float4.

The existing gpu_binaryop_forward_33 computes channel/row/column per element
(three integer divisions) even when both inputs and the output are the same
3-D shape and fully contiguous.  For that case we treat the tensor as a flat
array and process four floats per thread with float4 loads/stores; the 0-3
element tail and every broadcast / non-contiguous / non-16B-aligned case keep
the original scalar kernel.

NCNN_CUDA_BINARYOP_VEC=0 disables the vector path (default on for A/B).
Add is element-wise with no reordering, so results are bit-identical.
"""
import io

CU = "/home/user/ncnn-cuda/src/layer/cuda/binaryop_cuda.cu"

INC_OLD = "#include <iostream>"
INC_NEW = "#include <cstdlib>\n#include <iostream>"

KERNELS = r'''
// 同形状连续 BinaryOp：flat float4 路径，尾部 0-3 个元素走标量。
template<typename Op>
__global__ void gpu_binaryop_forward_33_vec4(const float4* __restrict__ a_input,
                                             const float4* __restrict__ b_input,
                                             float4* __restrict__ out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float4 av = a_input[i];
    const float4 bv = b_input[i];
    Op op;
    float4 cv;
    cv.x = op(av.x, bv.x);
    cv.y = op(av.y, bv.y);
    cv.z = op(av.z, bv.z);
    cv.w = op(av.w, bv.w);
    out[i] = cv;
}

template<typename Op>
__global__ void gpu_binaryop_forward_33_tail(const float* __restrict__ a_input,
                                             const float* __restrict__ b_input,
                                             float* __restrict__ out,
                                             int start, int total)
{
    const int i = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    Op op;
    out[i] = op(a_input[i], b_input[i]);
}

'''

DISPATCH_OLD = """        if (b_info.dims == 3) {
            if (a_info.w == 1 && a_info.h == 1 && channels_a == channels_b)"""
DISPATCH_NEW = """        if (b_info.dims == 3) {
            const bool same_shape = a_info.w == b_info.w && a_info.h == b_info.h && a_info.c == b_info.c;
            const bool contiguous = same_shape
                && a_info.cstep == a_info.w * a_info.h
                && b_info.cstep == b_info.w * b_info.h
                && c_info.cstep == c_info.w * c_info.h;
            const char* vec_env = std::getenv("NCNN_CUDA_BINARYOP_VEC");
            const bool vec_off = vec_env && vec_env[0] == '0' && vec_env[1] == '\\0';
            const bool aligned = (((size_t)a_input | (size_t)b_input | (size_t)c_output) & 15u) == 0;
            if (contiguous && !vec_off && aligned)
            {
                const int total = a_info.cstep * a_info.c;
                const int n4 = total / 4;
                if (n4 > 0)
                {
                    const int threads = 256;
                    gpu_binaryop_forward_33_vec4<Op><<<(n4 + threads - 1) / threads, threads>>>(
                        reinterpret_cast<const float4*>(a_input),
                        reinterpret_cast<const float4*>(b_input),
                        reinterpret_cast<float4*>(c_output), n4);
                }
                const int tail = total - n4 * 4;
                if (tail > 0)
                    gpu_binaryop_forward_33_tail<Op><<<1, 32>>>(
                        a_input, b_input, c_output, n4 * 4, total);
                return 0;
            }

            if (a_info.w == 1 && a_info.h == 1 && channels_a == channels_b)"""


def repl(path, old, new):
    with io.open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if old not in text:
        raise SystemExit("marker missing: %r" % old[:70])
    text = text.replace(old, new, 1)
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def main():
    repl(CU, INC_OLD, INC_NEW)
    with io.open(CU, "r", encoding="utf-8") as f:
        cu = f.read()
    # insert kernels before the binary_op_cuda dispatcher
    pos = cu.index("template<typename Op>\nstatic int binary_op_cuda(")
    cu = cu[:pos] + KERNELS + cu[pos:]
    with io.open(CU, "w", encoding="utf-8", newline="\n") as f:
        f.write(cu)
    repl(CU, DISPATCH_OLD, DISPATCH_NEW)
    print("patched BinaryOp float4")


if __name__ == "__main__":
    main()
