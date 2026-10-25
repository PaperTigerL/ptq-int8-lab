#!/usr/bin/env python3
"""Unroll the implicit conv C4 loop 4-wide to expose memory-level parallelism.

The old inner loop is `for (c4...) { load input; load weight; dp4a; }` with a
runtime trip count, so ptxas cannot hoist loads across iterations.  Each dp4a
therefore waits on a weight load (L2 latency) and the warp stalls.

Unrolling by 4 issues eight independent loads up front, then four dp4a.  The
accumulation order stays c4-ascending, so int32 results are bit-identical.
"""
import io

PATH = "/home/user/ncnn-cuda/src/layer/cuda/implicit_conv_cuda.cu"
OLD = """            for (int c4 = 0; c4 < C4; c4++)
            {
                const int a = qin[(size_t)c4 * S + pix];
                const int w = win[((size_t)tap * C4 + c4) * Cout + co];
                acc = dp4a_i8(a, w, acc);
            }"""
NEW = """            const int* qp = qin + pix;
            const int* wp = win + (size_t)tap * C4 * Cout + co;
            int c4 = 0;
            for (; c4 + 3 < C4; c4 += 4)
            {
                // 先发 8 个独立加载，再做 4 个 dp4a：把权重加载延迟藏到 4 路并行里。
                const int a0 = qp[(size_t)(c4 + 0) * S];
                const int a1 = qp[(size_t)(c4 + 1) * S];
                const int a2 = qp[(size_t)(c4 + 2) * S];
                const int a3 = qp[(size_t)(c4 + 3) * S];
                const int w0 = wp[(size_t)(c4 + 0) * Cout];
                const int w1 = wp[(size_t)(c4 + 1) * Cout];
                const int w2 = wp[(size_t)(c4 + 2) * Cout];
                const int w3 = wp[(size_t)(c4 + 3) * Cout];
                // 累加顺序保持 c4 升序，int32 结果逐位不变。
                acc = dp4a_i8(a0, w0, acc);
                acc = dp4a_i8(a1, w1, acc);
                acc = dp4a_i8(a2, w2, acc);
                acc = dp4a_i8(a3, w3, acc);
            }
            for (; c4 < C4; c4++)
            {
                const int a = qp[(size_t)c4 * S];
                const int w = wp[(size_t)c4 * Cout];
                acc = dp4a_i8(a, w, acc);
            }"""


def main():
    with io.open(PATH, "r", encoding="utf-8") as f:
        text = f.read()
    if OLD not in text:
        raise SystemExit("inner loop marker not found")
    text = text.replace(OLD, NEW)
    with io.open(PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print("patched", PATH)


if __name__ == "__main__":
    main()
