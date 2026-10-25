// GPU int8 推理引擎 —— ResNet-18 CIFAR-10。
//
// 定位：把 03-ptq-lab 的 PTQ 语义（per-channel 权重 + per-tensor 激活）
// 落到 C++/CUDA 上，直接吃 Turing 的 int8 tensor core。
//
// 为什么不用 PyTorch 的 quantized ops：
//   1. torch.ao.nn.quantized 的 CUDA 后端覆盖不全；
//   2. fbgemm/qnnpack 的粒度是我们实验要改的 per-channel，不是 per-tensor；
//   3. 目标是**可改语义的实验平台**，库越厚越难改。
//   底层只借一样东西：cuBLAS 的 CUBLAS_COMPUTE_32I —— 那就是 tensor core。
//
// 职责边界（为什么校准不在 C++ 侧做）：
//   校准要跑 **fp32** 前向统计每层输入范围。C++ 侧要复刻 fp32 卷积正确性，
//   等于重写一遍 cuDNN —— 收益为零、风险极高。所以：
//     Python 侧（已验证的模型）跑 fp32 前向 → 导出 (lo,hi)
//     C++ 侧只做：读 scale → 量化 → im2col → GEMM → requant
//   这是分工，不是绕过：真正要改的**量化语义**在 C++ 里（ptq_quant.h），
//   校准只是给它喂 range。
//
// 融合设计（依据是分段实测，不是猜）：
//   quantize 10.7ms | im2col 7.9ms | gemm 6.2ms = 24.9ms，GEMM 只占 25%。
//   → 瓶颈是内存搬运。所以把 quantize + im2col 合成**一个** kernel，
//     中间永不落 fp32，im2col 输出只写一次。
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include "implicit_conv.cuh"
#include "layer_impl.h"      // 按层选择：implicit vs im2col+cublas
#include "stream_ctx.cuh"
#include "int8_gemm.cuh"
#include "ptq_model.h"
#include "ptq_quant.h"

namespace ptqint8 {
// kernel 的 host 侧包装（launcher 必须在 .cu 里，见 int8_kernels.cu 注释）
// c_base：分组卷积时该组在输入通道维的起点；C 仍是**总**输入通道
//         （x 的 sample stride 必须是 C*H*W，不能传每组通道数）
void quant_im2col_launch(const float*, int8_t*, float, int, int,
                         int, int, int, int, int, int, int, int, int, int, int, int, int, int,
                         int c_base, int);
// 快速版：grid=(OW,OH,N) 消除逐元素 64 位 div/mod（实测 im2col 从 19.1 ms
// 降到 ~1 ms，见 ROOFLINE-int8.md）。语义与上面旧版逐位一致。
void quant_im2col_fast_launch(const float*, int8_t*, float, int, int,
                              int, int, int, int, int, int, int, int, int, int,
                              int, int, int, int, int c_base);
void quant_im2col_vec_launch(const float* x, int8_t* out, float scale, int qmax,
                             int N, int C, int H, int W,
                             int kh, int kw, int sh, int sw, int ph, int pw,
                             int OH, int OW, int K, int Kpad, int c_base);
void quant_im2col_tap_launch(const float* x, int8_t* out, float scale, int qmax,
                             int N, int C, int H, int W,
                             int kh, int kw, int sh, int sw, int ph, int pw,
                             int OH, int OW, int Kpad, int c_base);
void quant_im2col_tiled_launch(const float* x, int8_t* out, float scale, int qmax,
                               int N, int C, int H, int W,
                               int kh, int kw, int sh, int sw, int ph, int pw,
                               int OH, int OW, int K, int Kpad, int c_base);
void quant_im2col_kmajor_launch(const float* x, int8_t* out, float scale, int qmax,
                                int N, int C, int H, int W,
                                int kh, int kw, int sh, int sw, int ph, int pw,
                                int OH, int OW, int K, int Kpad, int c_base);
void requant_t_launch(const int32_t* acc, float* out,
                      const float* wscale, const float* bias, const float* zp_corr,
                      float ascale, int Cout, int Lpos, int N, bool relu,
                      int cout_stride, int co_base);
void reduce_minmax_launch(const float*, size_t, float*, int);
// cout_stride：输出张量的总通道数；co_base：该组输出通道起点。
// 分组卷积逐组写回同一块 (N, cout_stride, HW) 输出，不能假设组内 cout 就是 stride。
void requant_launch(const int32_t*, float*, const float*, const float*,
                    const float*, float, int, int, int, bool,
                    int cout_stride, int co_base, int);
// 转置瓦片版：acc 读合并、out 写合并，算式与旧版逐位一致（见 ROOFLINE-int8.md）。
void requant_tiled_launch(const int32_t*, float*, const float*, const float*,
                          const float*, float, int, int, int, bool,
                          int cout_stride, int co_base);
void add_launch(float*, const float*, const float*, size_t, int);
void add_relu_launch(float*, const float*, const float*, size_t, int);
void maxpool_launch(const float*, float*, int, int, int, int, int);
// mode: 1=ReLU, 2=ReLU6（clamp 到 [0,6]）
void relu_launch(float*, size_t, int, int mode);
// depthwise：cin=1, groups=cout。w 按 [channel][Kpad] 存（K=kh*kw）。
void depthwise_conv_launch(const float* x, float* out, const int8_t* w,
                           const float* wscale, const float* bias,
                           float ascale, int qmax,
                           int N, int C, int H, int W,
                           int kh, int kw, int sh, int sw, int ph, int pw,
                           int OH, int OW, int w_stride, int grid);
} // namespace ptqint8

// 统一模型类型来自 ptq_model.h（PTQW/PTQ2 分派 + 拓扑校验都在那里）。
using ptqmodel::BlockSpec;
using ptqmodel::ConvSpec;
using ptqmodel::Model;
// 层选择表在 ptqint8 命名空间里；引擎类在全局命名空间，这里引入名字。
using ptqint8::LayerImpl;
using ptqint8::LayerImplTable;

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _e = (expr);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at line %d: %s\n", __LINE__,            \
                    cudaGetErrorString(_e));                                    \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

// ================================================================ 模型格式
// Model / ConvSpec / BlockSpec 定义在 ptq_model.h：按 magic 分派 PTQW/PTQ2，
// 并在加载时做通道/尺寸/残差形状校验。这里只保留校准文件入口。

// 校准文件：每层一个 (lo, hi)。严格校验 8+8n 且 n == 模型 conv 数。
bool load_calib(const std::string& path, size_t n_conv,
                std::vector<std::pair<float, float>>* out,
                std::string* err = nullptr);

// ================================================================ 引擎

class Int8Engine {
public:
    bool build(const Model& m, int w_bits, int a_bits)
    {
        m_ = (Model*)&m;
        w_bits_ = w_bits;
        a_bits_ = a_bits;
        // ⭐ 按层混合：默认读实测生成的层选择表（layer_impl_default.h）。
        //   为什么不「全线 implicit」或「全线 cublas」：两条路都 bit-exact，但逐层
        //   各有赢面（ROOFLINE §10.2/§11）。层选择表在同 batch、100 次预热、
        //   int32 逐元素相同的条件下测出来，默认选每层更快的那个。
        //   覆盖方式：LAYER_IMPL_JSON=<json> / LAYER_IMPL=0:c,1:i,... /
        //             USE_IMPLICIT_GEMM=1|0（全局 A/B 开关，优先级最高）。
        {
            std::string ierr;
            if (!impl_.init(m.convs.size(), &ierr))
            {
                fprintf(stderr, "[FAIL] 层选择表不可用：%s\n", ierr.c_str());
                return false;
            }
        }
        // implicit 路径是否可能被用到（决定是否准备 DP4A 权重 + 覆盖率检查）。
        use_implicit_ = impl_.mode() != LayerImplTable::Mode::AllCublas;
        // 转置 GEMM 默认关闭；先验证正确性与 A/B，再决定是否默认。
        const char* tg = std::getenv("USE_TRANSPOSED_GEMM");
        use_transposed_ = (tg && std::strcmp(tg, "1") == 0);
        // 融合 epilogue：Lpos==1 的层不需要转置（直接写 NCHW，零额外开销），
        // Lpos>1 且转置 tile <=16KB 时才融合。所以默认开。
        const char* ie = std::getenv("USE_IMPLICIT_EPILOGUE");
        use_implicit_epilogue_ = !(ie && std::strcmp(ie, "0") == 0);
        // 输入 tile 缓存 vs 直接读全局：默认交给 implicit 内部的形状判据。
        const char* id = std::getenv("USE_IMPLICIT_DIRECT");
        use_implicit_direct_ = !(id && std::strcmp(id, "0") == 0);
        // IMPLICIT_STRICT=1：任何一层没被 implicit 覆盖就直接失败，
        // 防止"静默回退到 im2col"把性能结论带偏（验收脚本用它）。
        const char* ist = std::getenv("IMPLICIT_STRICT");
        implicit_strict_ = (ist && std::strcmp(ist, "1") == 0);
        if (!ctx_.init()) { fprintf(stderr, "cublas 初始化失败\n"); return false; }
        prof_on_ = (std::getenv("PROF_LAYERS") != nullptr);
        if (prof_on_)
        {
            cudaEventCreate(&prof_ev_[0]);
            cudaEventCreate(&prof_ev_[1]);
            prof_ms_.assign(m.convs.size(), 0.0);
            prof_flops_.assign(m.convs.size(), 0.0);
            prof_calls_.assign(m.convs.size(), 0);
        }

        layers_.resize(m.convs.size());
        for (size_t i = 0; i < m.convs.size(); i++)
        {
            const ConvSpec& c = m.convs[i];
            if (c.groups < 1 || c.cout % c.groups != 0)
            {
                fprintf(stderr, "[FAIL] conv[%zu]: groups=%d 非法（需 groups>=1 且整除 cout=%d）\n",
                        i, c.groups, c.cout);
                return false;
            }
            if (c.weight.size() != (size_t)c.cout * c.cin * c.kh * c.kw)
            {
                fprintf(stderr, "[FAIL] conv[%zu]: 权重数量 %zu != cout*cin*kh*kw\n",
                        i, c.weight.size());
                return false;
            }
            L& q = layers_[i];
            q.cout = c.cout; q.cin = c.cin;
            q.kh = c.kh; q.kw = c.kw;
            q.sh = c.sh; q.sw = c.sw; q.ph = c.ph; q.pw = c.pw;
            q.groups = c.groups;
            q.cin_pg = c.cin;                 // ConvSpec.cin 本身就是每组通道
            q.cout_pg = c.cout / c.groups;
            q.K = c.cin * c.kh * c.kw;        // 每组 K
            q.Kpad = ((q.K + 7) / 8) * 8;
            q.gstride = (size_t)q.Kpad * q.cout_pg;   // 每组权重矩阵元素数

            // ---- 权重量化：C++ per-channel sym，与 quant.py 口径一致 ----
            // c.weight 是 [cout][cin][kh][kw] 的 Cout-major 展平；每组每输出通道
            // 恰好是连续的 K=c.cin*kh*kw 个元素，所以 per-channel 轴仍是 dim=0。
            ptq::ChannelQuant cq = ptq::quantize_weight_per_channel_sym(
                c.weight.data(), c.cout, q.K, w_bits);
            q.wscale = cq.scale;

            // 每组权重转成 (Kpad, cout_pg) 行主序 —— at::_int_mm 的 B 形状。
            // 组间不共享，按 g 依次放在 w_dev 的 g*gstride 处。
            // depthwise（groups=cout, cout_pg=1）也走这个布局：每组就是
            // 一个通道的 [Kpad] 列，depthwise kernel 直接按 w[c*Kpad+k] 读。
            q.w_dev.assign((size_t)q.groups * q.gstride, 0);
            for (int g = 0; g < q.groups; g++)
                for (int oo = 0; oo < q.cout_pg; oo++)
                {
                    const int o = g * q.cout_pg + oo;
                    for (int k = 0; k < q.K; k++)
                        q.w_dev[g * q.gstride + (size_t)k * q.cout_pg + oo] =
                            cq.q[(size_t)o * q.K + k];
                }

            // tap-major GEMM 权重：K 顺序 (tap, c4) 与 tap-major im2col 一致。
            // 整数和与顺序无关，GEMM 的 int32 输出与旧 c-major 路径逐元素相同。
            q.Cpad = (q.cin_pg + 3) & ~3;
            q.Ktap = q.kh * q.kw * q.Cpad;
            q.KpadTap = ((q.Ktap + 7) / 8) * 8;
            {
                std::vector<int8_t> wt((size_t)q.KpadTap * q.cout_pg, 0);
                for (int g = 0; g < q.groups; g++)
                    for (int oo = 0; oo < q.cout_pg; oo++)
                        for (int t = 0; t < q.kh * q.kw; t++)
                            for (int c = 0; c < q.cin_pg; c++)
                                wt[(size_t)(t * q.Cpad + c) * q.cout_pg + oo] =
                                    cq.q[((size_t)(g * q.cout_pg + oo) * q.cin_pg +
                                          c) * (q.kh * q.kw) + t];
                q.w_tap_bytes = wt.size() * q.groups;
                CUDA_CHECK(cudaMalloc(&q.w_tap_gpu, wt.size() * q.groups));
                for (int g = 0; g < q.groups; g++)
                    CUDA_CHECK(cudaMemcpy(q.w_tap_gpu + (size_t)g * wt.size(),
                                          wt.data(), wt.size(), cudaMemcpyHostToDevice));
            }

            // implicit GEMM 的 DP4A 权重布局（tap-major × c4 × cout × 4）。
            // 旧的 (Kpad,cout) 权重仍保留，USE_IMPLICIT_GEMM=0 时走旧路径。
            if (use_implicit_ && q.groups == 1)
            {
                std::vector<int8_t> wi;
                ptqint8::repack_implicit_weight(cq.q.data(), q.cout, q.cin_pg,
                                                q.kh, q.kw, &wi);
                q.w_imp_bytes = wi.size();
                CUDA_CHECK(cudaMalloc(&q.w_imp_gpu, wi.size()));
                CUDA_CHECK(cudaMemcpy(q.w_imp_gpu, wi.data(), wi.size(),
                                      cudaMemcpyHostToDevice));
            }

            // 转置 GEMM 的 A 矩阵：(cout_pg, Kpad)，每组一份。
            {
                std::vector<int8_t> wt((size_t)q.Kpad * q.cout_pg, 0);
                for (int g = 0; g < q.groups; g++)
                    for (int oo = 0; oo < q.cout_pg; oo++)
                        for (int k = 0; k < q.Kpad; k++)
                            wt[(size_t)oo * q.Kpad + k] =
                                q.w_dev[(size_t)g * q.gstride +
                                        (size_t)k * q.cout_pg + oo];
                CUDA_CHECK(cudaMalloc(&q.wT_gpu, wt.size() * q.groups));
                for (int g = 0; g < q.groups; g++)
                    CUDA_CHECK(cudaMemcpy(q.wT_gpu + (size_t)g * wt.size(),
                                          wt.data(), wt.size(), cudaMemcpyHostToDevice));
            }

            CUDA_CHECK(cudaMalloc(&q.w_gpu, q.w_dev.size()));
            CUDA_CHECK(cudaMemcpy(q.w_gpu, q.w_dev.data(), q.w_dev.size(), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMalloc(&q.ws_gpu, c.cout * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(q.ws_gpu, q.wscale.data(), c.cout * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMalloc(&q.bias_gpu, c.cout * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(q.bias_gpu, c.bias.data(), c.cout * sizeof(float), cudaMemcpyHostToDevice));
        }

        // fc 权重转置成 (fc_in, fc_out) 行主序，供 cublasSgemm 直接用
        std::vector<float> wt((size_t)m.fc_in * m.fc_out);
        for (int o = 0; o < m.fc_out; o++)
            for (int j = 0; j < m.fc_in; j++)
                wt[(size_t)j * m.fc_out + o] = m.fc_w[(size_t)o * m.fc_in + j];
        CUDA_CHECK(cudaMalloc(&fc_w_gpu_, wt.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(fc_w_gpu_, wt.data(), wt.size() * sizeof(float), cudaMemcpyHostToDevice));

        aq_.resize(m.convs.size());

        // ---- implicit 覆盖率（构建期一次判定，不在运行期"回退"）----
        // 覆盖判据：groups==1、cout 能被 NT(64/128) 整除、且 (kh,kw,sh,sw,ph,pw)
        // 属于编译期实例化的 6 个形状之一。运行期的唯一分流是
        // 「Lpos==1 直接写 / Lpos>1 走 shared 转置 / tile 太大走 acc+requant」，
        // 那是 epilogue 的实现选择，不是算子回退。
        std::vector<char> shape_ok_vec(m.convs.size(), 0);
        int covered = 0, uncovered_unary = 0;
        for (size_t i = 0; i < m.convs.size(); i++)
        {
            const ConvSpec& c = m.convs[i];
            const bool shape_ok =
                (c.kh == 7 && c.kw == 7 && c.sh == 2 && c.sw == 2 && c.ph == 3 && c.pw == 3) ||
                (c.kh == 3 && c.kw == 3 && c.sh == 1 && c.sw == 1 && c.ph == 1 && c.pw == 1) ||
                (c.kh == 3 && c.kw == 3 && c.sh == 2 && c.sw == 2 && c.ph == 1 && c.pw == 1) ||
                (c.kh == 3 && c.kw == 3 && c.sh == 1 && c.sw == 1 && c.ph == 0 && c.pw == 0) ||
                (c.kh == 1 && c.kw == 1 && c.sh == 1 && c.sw == 1 && c.ph == 0 && c.pw == 0) ||
                (c.kh == 1 && c.kw == 1 && c.sh == 2 && c.sw == 2 && c.ph == 0 && c.pw == 0);
            shape_ok_vec[i] = shape_ok && use_implicit_ &&
                              c.groups == 1 && c.cout % 64 == 0;
            if (shape_ok_vec[i]) covered++;
            else uncovered_unary++;
        }
        implicit_covered_ = covered;
        implicit_uncovered_ = uncovered_unary;
        if (use_implicit_)
            printf("[implicit] GEMM 覆盖 %d/%d 层（分组/深度可分离 %d 层走专用 kernel）\n",
                   covered, (int)m.convs.size(), uncovered_unary);

        // ---- 层选择表：汇总 + 严格校验 ----
        // 与运行期不同，这里没有 H/W；表的选择只按层号，所以用 select_static。
        // 表要求某层走 implicit 而该形状没被覆盖 → 构建期就失败，绝不运行期静默换实现。
        impl_.print_summary(m.convs.size());
        ran_impl_.assign(m.convs.size(), 0);
        impl_warned_.assign(m.convs.size(), 0);
        for (size_t i = 0; i < m.convs.size(); i++)
        {
            const ConvSpec& c = m.convs[i];
            if (c.groups == c.cout && c.cin == 1) ran_impl_[i] = -1;   // depthwise
            if (impl_.select_static((int)i) != LayerImpl::Implicit) continue;
            if (!shape_ok_vec[i])
            {
                fprintf(stderr,
                        "[FAIL] 层选择表要求 conv[%zu] 走 implicit，但该形状不在 implicit 覆盖内"
                        "（groups=%d cout=%d k%dx%d s%dx%d pad%d,%d）\n",
                        i, c.groups, c.cout, c.kh, c.kw, c.sh, c.sw, c.ph, c.pw);
                return false;
            }
        }
        if (implicit_strict_ && impl_.mode() == LayerImplTable::Mode::AllImplicit &&
            uncovered_unary > 0)
        {
            fprintf(stderr, "[FAIL] IMPLICIT_STRICT=1 但 %d 层未被 implicit 覆盖\n",
                    uncovered_unary);
            return false;
        }
        return true;
    }

    // 激活范围 → scale 的口径。默认 AbsMax（与 09 校准一致），
    // --act-minmax 可显式切回 06 历史公式用于对照。
    void set_act_mode(ptq::ActRangeMode mode) { act_mode_ = mode; }

    // 装载 Python 侧校准出来的 (lo,hi)，并推出 scale/zp/zp 修正项。
    // lo/hi 存在 ranges_ 里直接打印 —— 不另存一份 act_lo_/act_hi_，
    // 那两份缓存上一版留过，结果 print_calib 读空 vector 直接段错误。
    //
    // ⭐ 返回 bool 而不是 exit(1)：层数/数值非法是**调用方可处理的错误**，
    //    尤其 PTQC 来自外部文件，宁可返回错误让上层打印路径并退出，
    //    也不要在库内部直接终止进程（H-20261022-002 的硬要求）。
    bool set_calib(std::vector<std::pair<float, float>> ranges,
                   std::string* err = nullptr)
    {
        auto fail = [&](const std::string& s) {
            if (err) *err = s;
            return false;
        };
        if (ranges.size() != layers_.size())
            return fail("校准层数 " + std::to_string(ranges.size()) +
                        " != conv 层数 " + std::to_string(layers_.size()));
        for (size_t i = 0; i < ranges.size(); i++)
        {
            if (!std::isfinite(ranges[i].first) || !std::isfinite(ranges[i].second))
                return fail("第 " + std::to_string(i) + " 层校准范围非有限");
            if (ranges[i].first > ranges[i].second)
                return fail("第 " + std::to_string(i) + " 层校准范围 lo > hi");
        }
        ranges_ = std::move(ranges);
        for (size_t i = 0; i < layers_.size(); i++)
        {
            // ⚠️ 必须索引 ranges_ 而不是 ranges —— 上一行已经把它 move 走了，
            //    move 之后的 vector 是空的（size 0，gdb 会如实显示 length 0）。
            //    这是「move 之后再用」的标准坑，不是显示问题。
            aq_[i] = ptq::tensor_quant_from_range(ranges_[i].first, ranges_[i].second,
                                                  a_bits_, act_mode_);
            L& q = layers_[i];
            // zp 修正：out += -zp * sa * sw[co] * Σ_k q_w[k][co]
            // 分组时每个输出通道只连自己那一组的 K 个权重（组间无交叉项）。
            std::vector<float> corr(q.cout, 0.0f);
            for (int o = 0; o < q.cout; o++)
            {
                const int g = o / q.cout_pg;
                const int oo = o % q.cout_pg;
                long long sum = 0;
                for (int k = 0; k < q.K; k++)
                    sum += q.w_dev[g * q.gstride + (size_t)k * q.cout_pg + oo];
                corr[o] = (float)(-(double)aq_[i].zero_point * (double)aq_[i].scale
                                  * (double)q.wscale[o] * (double)sum);
            }
            CUDA_CHECK(cudaMalloc(&q.zp_gpu, corr.size() * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(q.zp_gpu, corr.data(), corr.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
        }
        return true;
    }

    void print_calib() const
    {
        printf("%-4s %-11s %-11s %-11s %s\n", "层", "lo", "hi", "scale", "zp");
        for (size_t i = 0; i < aq_.size(); i++)
            printf("%-4zu %-11.5g %-11.5g %-11.5g %d\n", i, ranges_[i].first, ranges_[i].second,
                   aq_[i].scale, aq_[i].zero_point);
    }

    // ---------------- int8 前向（单 batch）----------------
    //
    // ⭐ 按**导出的拓扑**执行，不按 conv 索引顺序串联。
    //   上一版按顺序串联，残差分支整个丢失 —— 单层校验全绿但全网精度 7.6%。
    //   教训：单层对 ≠ 全网对，结构性错误逐层校验发现不了。
    // stop_after_block >= 0 时：跑到第 K 个 block 就停，把该处的激活拷到
    // mid_out（而不是继续跑 fc）。这是与numpy 参考做二分定位的接口 ——
    // 端到端比对只说「不一致」，逐层编 probe 找错层太慢，
    // 而这里两次运行（K 与 K+1）就能二分出分歧起点。
    // stop_after_block == -2 时导出 stem 之后的激活（conv1 + relu + maxpool）。
    // 单独导出 stem 的意义：conv1 是**唯一K 需要补零对齐 8 的倍数**的层
    // （K = 3*7*7 = 147），如果分歧真来自补零逻辑，这里一次比对就能判定，
    // 不需要为 conv1 单独编一个 probe。
    void forward(const float* x, int n, int c, int h, int w, float* logits,
                 int stop_after_block = -1, float* mid_out = nullptr)
    {
        // ---- 显存池策略（实测调出来的）----
        //
        // 每 batch pool_reset → 稳态 0.30 ms/img（同 batch 内 20 层反复 malloc）
        // 块级复用        → 稳态 0.157 ms/img，但**显存单调增长**，
        //                   batch=1000 跑 2000 张直接 OOM/超时。
        //
        // 根因：块级复用的 `acquire` 在「旧块不够大」时会**追加**新块
        // （不能原地 free，上层可能还持有指针）。于是尺寸波动时池会一直涨。
        //
        // 正确做法：**跨 batch 复用 + 容量超限就整体重建**。
        //   · 复用：同一 batch 尺寸下，第二批起零 malloc（拿稳态）
        //   · 重建：池块数或总字节超阈值时 reset（防 OOM）
        // 阈值取「当前 batch 估算峰值 × 2」，既能吃掉正常波动，又能兜住失控。
        // ---- 预算：精确算出这次 forward 会申请的所有块 ----
        //
        // ⚠️ 第一版按「每层都是 32×32」粗估，need 大了 4 倍多，
        // 结果容量守卫每个 batch 都触发重建 —— 稳态 0.084 → 0.178 ms/img。
        // 教训：**阈值估错比没有阈值更糟**：它让本该复用的路径反复重建。
        //
        // 精确算法：按真实拓扑走一遍形状，累加每层的 col/acc/out。
        size_t need = 0;
        {
            int ch = c, hh = h, ww2 = w;
            need += (size_t)n * ch * hh * ww2 * sizeof(float);   // d_cur
            // stem conv
            {
                need += conv_scratch_bytes(m_->convs[0], n, hh, ww2);
                const ConvSpec& cs = m_->convs[0];
                hh = out_h(cs, hh); ww2 = out_w(cs, ww2); ch = cs.cout;
            }
            if (m_->stem_pool)
            {
                hh = (hh + 2 - 3) / 2 + 1;
                ww2 = (ww2 + 2 - 3) / 2 + 1;
                need += (size_t)n * ch * hh * ww2 * sizeof(float);
            }
            for (const BlockSpec& blk : m_->blocks)
            {
                // identity（downsample 或直连）只在有残差时需要缓存
                if (blk.residual != 0)
                    need += (size_t)n * ch * hh * ww2 * sizeof(float);

                int tch = ch, thh = hh, tww = ww2;
                for (int ci : blk.convs)
                {
                    if (ci < 0) continue;   // -1 尾部补位
                    const ConvSpec& cs = m_->convs[ci];
                    need += conv_scratch_bytes(cs, n, thh, tww);
                    thh = out_h(cs, thh); tww = out_w(cs, tww); tch = cs.cout;
                }
                ch = tch; hh = thh; ww2 = tww;
            }
            need += (size_t)n * m_->fc_in * sizeof(float);
            need += (size_t)n * m_->fc_out * sizeof(float);
        }

        if (pool_batch_ != n || pool_bytes_ + need > pool_limit_)
        {
            pool_reset();
            pool_batch_ = n;
        }
        pool_cursor_ = 0;
        pool_bytes_ = 0;
        prof_batch_ = n;
        // 逐层计时要跳过第一个 forward：它包含 pool 构建（几十次 cudaMalloc）、
        // 首次 kernel 加载、cublas 算法选择 —— 实测全部记在第 0 层上，
        // 能把 conv[0] 的均值从 2.6 ms 抬到 59 ms（污染整张表）。
        if (prof_on_)
        {
            prof_skip_this_ = !prof_warmed_;
            prof_warmed_ = true;
        }
        ptqint8::set_stream(stream_);
        if (stream_) cublasSetStream(ctx_.handle, stream_);
        float* d_cur = acquire((size_t)n * c * h * w * sizeof(float));
        copy_h2d(d_cur, x, (size_t)n * c * h * w * sizeof(float));

        int ch = c, hh = h, ww = w;

        // ---- stem: conv1 → (ReLU) → maxpool ----
        {
            float* d_next = nullptr;
            run_conv(0, d_cur, n, ch, hh, ww, &d_next, &ch, &hh, &ww);
            d_cur = d_next;
            if (m_->stem_pool)
            {
                const int OH = (hh + 2 - 3) / 2 + 1;
                const int OW = (ww + 2 - 3) / 2 + 1;
                float* d_pool = acquire((size_t)n * ch * OH * OW * sizeof(float));
                ptqint8::maxpool_launch(d_cur, d_pool, n, ch, hh, ww,
                                        grid_for((size_t)n * ch * OH * OW));
                CUDA_CHECK(cudaGetLastError());
                d_cur = d_pool;
                hh = OH;
                ww = OW;
            }
        }

        if (stop_after_block == -2 && mid_out)
        {
            CUDA_CHECK(cudaMemcpy(mid_out, d_cur,
                                  (size_t)n * ch * hh * ww * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            return;
        }

        // ---- 按导出的 block 列表执行（ResNet 2/3 层、MobileNetV2 1~3 层都走这里）----
        for (int bi = 0; bi < (int)m_->blocks.size(); bi++)
        {
            const BlockSpec& blk = m_->blocks[bi];

            // identity：只有有残差时才需要（residual=0 的块是纯串联）
            float* d_id = nullptr;
            int ich = ch, ihh = hh, iww = ww;
            if (blk.residual != 0)
            {
                if (blk.ds >= 0)
                {
                    run_conv(blk.ds, d_cur, n, ch, hh, ww, &d_id, &ich, &ihh, &iww);
                }
                else
                {
                    d_id = acquire((size_t)n * ch * hh * ww * sizeof(float));
                    CUDA_CHECK(cudaMemcpy(d_id, d_cur,
                                          (size_t)n * ch * hh * ww * sizeof(float),
                                          cudaMemcpyDeviceToDevice));
                }
            }

            // 主支路：按 convs 顺序执行，-1 是尾部补位
            float* d_t = d_cur;
            int tch = ch, thh = hh, tww = ww;
            for (int ci : blk.convs)
            {
                if (ci < 0) continue;
                float* d_next = nullptr;
                run_conv(ci, d_t, n, tch, thh, tww, &d_next, &tch, &thh, &tww);
                d_t = d_next;
            }

            if (blk.residual == 0)
            {
                // 纯串联：最后一个 conv 的激活已在 run_conv 里做过
                d_cur = d_t;
            }
            else
            {
                // ⛔ 尺寸必须一致才能逐元素加。不一致就是拓扑/权重导出有问题，
                //    必须显式报错而不是让 kernel 越界读内存（那会静默出错结果）。
                const size_t sz = (size_t)n * tch * thh * tww;
                if (tch != ich || thh != ihh || tww != iww)
                {
                    fprintf(stderr,
                            "[FAIL] block[%d] 残差形状不匹配：主支路 (%d,%d,%d) vs "
                            "identity (%d,%d,%d)\n",
                            bi, tch, thh, tww, ich, ihh, iww);
                    std::exit(1);
                }
                // residual=1（ResNet）：add+ReLU 合成一个 kernel，少一遍全量读写。
                // residual=2（MobileNetV2）：project 线性输出，相加后不激活。
                if (blk.residual == 1)
                    ptqint8::add_relu_launch(d_t, d_t, d_id, sz, grid_for(sz));
                else
                    ptqint8::add_launch(d_t, d_t, d_id, sz, grid_for(sz));
                CUDA_CHECK(cudaGetLastError());
                d_cur = d_t;
            }
            ch = tch; hh = thh; ww = tww;

            if (stop_after_block >= 0 && bi == stop_after_block)
            {
                if (mid_out)
                    CUDA_CHECK(cudaMemcpy(mid_out, d_cur,
                                          (size_t)n * ch * hh * ww * sizeof(float),
                                          cudaMemcpyDeviceToHost));
                return;
            }
        }

        const int HW = hh * ww;
        const int D = m_->fc_in;
        if (ch != D)
        {
            fprintf(stderr, "[FAIL] 全局池化通道 %d != fc_in %d\n", ch, D);
            std::exit(1);
        }
        // ⛔ 这里必须做真正的 global average pool：d_cur 是 NCHW 的
        //    [n][ch][hh][ww]，不能直接 memcpy 前 n*D 个 float（那只是
        //    sample0 的 channel0 前半段），否则 FC 输入与训练口径不一致，
        //    整网 logits 随机但单层 check_layer 仍然全绿。
        // ⭐ 全局平均池化必须在 GPU 上做：
        //   旧版把 n*D*HW 个 float 拷回主机、CPU 求均值、再拷回去。
        //   R18 最后 9 层的空间是 1x1（均值=恒等），两个 memcpy 纯属白跑；
        //   更糟的是这两个同步 memcpy 会打断多流并行 —— 两条流被钉在
        //   同一个主机同步点上，跨图并行收益归零。
        if (d_pool_cap_ != (size_t)n * D)
        {
            if (d_pool_) { cudaFree(d_pool_); d_pool_ = nullptr; }
            d_pool_cap_ = (size_t)n * D;
            CUDA_CHECK(cudaMalloc(&d_pool_, d_pool_cap_ * sizeof(float)));
        }
        ptqint8::gap_launch(d_cur, d_pool_, n, ch, HW);
        CUDA_CHECK(cudaGetLastError());

        const float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(ctx_.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    m_->fc_out, n, D, &alpha,
                    (const float*)fc_w_gpu_, m_->fc_out,
                    d_pool_, D, &beta,
                    d_logits_, m_->fc_out);
        copy_d2h(logits, d_logits_, (size_t)n * m_->fc_out * sizeof(float));
        for (int i = 0; i < n * m_->fc_out; i++)
            logits[i] += m_->fc_b[i % m_->fc_out];
    }

    // ---------------- 显存池 ----------------
    //
    // ⭐ 为什么必须做池化（实测数据）：
    //   每个 conv 都 cudaMalloc/cudaFree 三个缓冲（col / acc / out）时，
    //   全网 6.99 ms/img；改成复用后降到 0.06 ms/img 量级。
    //   cudaMalloc 一次几十微秒，20 层 × 3 个 = 60 次，
    //   而单个 conv 的 GEMM 只有几毫秒 —— 分配开销占 90% 以上。
    //
    // ⚠️ 单调增长的池不能跨 batch 累积（跑 40 个 batch 就 OOM），
    //    所以用 **「按 batch 重建」** 的策略：
    //      每个 batch 开始时 free 掉上一批的缓冲，重新分配。
    //    这样池内最多只存在一个 batch 的中间量，显存占用与 batch 数无关。
    //    真正的复用发生在**同一 batch 内**的 20 层之间 ——
    //    那才是分配开销的大头（60 次 vs 40 次）。
    //    实测：这个策略下 1000 张（batch=250，4 个 batch）显存稳定。
    void pool_reset()
    {
        for (float* p : pool_) cudaFree(p);
        pool_.clear();
        pool_size_.clear();
        pool_cursor_ = 0;
        pool_bytes_ = 0;
    }

    // 块级复用：同一位置在不同 batch 里要的形状相同，
    // 第二个 batch 起直接拿现成的块，不再 cudaMalloc。
    //
    // ⛔ 尺寸不够时**绝不覆盖旧指针再 free** —— 上层可能还持有它
    //    （比如 d_cur 和 d_next 指向同一块）。
    //    正确做法：把更大的那块追加到池尾，旧块保留（浪费一点显存，
    //    但绝对安全）。池在 batch 尺寸变化时整体重建，所以不会无限增长。
    float* acquire(size_t bytes)
    {
        // 只做「复用 or 追加」，不做任何重置游标的操作 ——
        // 重置必须由 forward 在批次边界统一做，中途重置会让上层持有的
        // 指针失效（实测表现为 cudaMalloc "invalid argument"）。
        const size_t idx = pool_cursor_++;
        if (idx < pool_.size() && pool_size_[idx] >= bytes)
            return pool_[idx];

        // ⭐ 分配前的显存检查 —— 这是防OOM 的唯一有效位置。
        //
        // 为什么必须在这里查而不是在 forward 开头：
        //   forward 开头的预算是**估算**，而估算可能偏小（形状按拓扑推，
        //   但没考虑 cuBLAS/cuDNN 自己的 workspace）。
        //   实测 batch=2000 时 im2col 单层就吃掉 1.2 GB，
        //   加上 cuBLAS 的 workspace 会直接撞上 4 GB 卡的物理上限。
        //
        // 为什么不用 cudaMalloc 的返回值判断：
        //   它返回 nullptr，但我们已经检查过 ——问题是**整机显存被别的进程占用**，
        //   而不是这一块分配失败。nullptr 判断太晚（已经浪费了整轮验证）。
        //
        // 阈值取「物理显存的 90%」：
        //   留10% 给 cuBLAS workspace + 驱动 + 其他进程。
        //   实测 4 GB 卡在 batch=2000 时用掉 95%，那已经是在悬崖边上。
        size_t avail = 0;
        if (!mem_ok(bytes + 64, &avail))
        {
            fprintf(stderr,
                    "[FAIL] 显存不足：本次需要 %.0f MB，当前可用 %.0f MB\n"
                    "       当前 batch=%d。\n"
                    "       先查是不是别的进程占着显存（nvidia-smi 可能显示\n"
                    "       No running processes found，但占用者在 Windows 侧）。\n"
                    "       确认没人占再降 batch：参考 --batch 512（实测最优点）。\n",
                    (bytes + 64) / 1048576.0, avail / 1048576.0,
                    batch_hint_);
            std::exit(1);
        }

        float* p = nullptr;
        // 64 字节对齐：float4 / vectorized load 的前提
        CUDA_CHECK(cudaMalloc(&p, bytes + 64));
        p = reinterpret_cast<float*>(
            (reinterpret_cast<uintptr_t>(p) + 63) & ~uintptr_t(63));

        if (idx < pool_.size())
        {
            pool_[idx] = p;
            pool_size_[idx] = bytes;
        }
        else
        {
            pool_.push_back(p);
            pool_size_.push_back(bytes);
        }
        return p;
    }

    // 查询可用显存是否够用。
    // 判定标准：free_b - need 要留得出「上限 - need」的余量，
    // 等价于「分配后剩余不超过总显存的 90%」。
    // 简化写法：分配后 free_b' = free_b - need，若 free_b' > 0.10 * total 则放行。
    //
    // ⚠️ 用动态查询而不是启动时缓存一次：
    //    同机上可能有别的进程（浏览器、另一个 python）占显存，
    //    启动时的快照在跑起来之后就失效了。实测这台 4 GB 卡上
    //    一个后台 python 就能吃掉 1.4 GB。
    // avail_out：回填查询到的可用显存，供调用方打印错误信息。
    // 用出参而不是 static 成员：static 成员声明了还得在某个 .cpp 里定义，
    // 否则链接报 undefined reference（这一轮踩过）。
    static bool mem_ok(size_t need, size_t* avail_out = nullptr)
    {
        size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return true;  // 查不到就放行
        if (avail_out) *avail_out = free_b;
        if (total_b == 0) return true;
        const size_t headroom = total_b / 10;   // 保留 10%
        return free_b > need + headroom;
    }
    // 每个 batch 开始时调用：游标归零，块全部可复用
    void pool_begin()
    {
        pool_cursor_ = 0;
        if (pool_batch_ != batch_hint_)
        {
            pool_reset();
            pool_batch_ = batch_hint_;
        }
    }

    // ---------------- 多流（部署侧跨图并行）----------------
    // 只对 implicit 路径成立：那条路一个 kernel 就能吃下整个 batch，
    // 不依赖 at::_int_mm（它用的是 torch 自己的 current stream）。
    // 传 nullptr 恢复单流 + 同步语义（行为与改动前完全一致）。
    void set_stream(cudaStream_t s) { stream_ = s; ptqint8::set_stream(s); }
    cudaStream_t stream() const { return stream_; }

    // 逐层计时（PROF_LAYERS=1）：每层前后各记一个 event，累加到表里，
    // 结束时 print_prof() 打出「ms / TOPS / 占峰值% / 占总量%」。
    void print_prof() const
    {
        if (!prof_on_ || prof_ms_.empty()) return;
        double total = 0.0;
        for (double v : prof_ms_) total += v;
        std::vector<int> order(prof_ms_.size());
        for (size_t i = 0; i < order.size(); i++) order[i] = (int)i;
        std::sort(order.begin(), order.end(),
                  [&](int a, int b) { return prof_ms_[a] > prof_ms_[b]; });
        printf("\n──── 逐层计时（按层混合，batch=%d）────\n", prof_batch_);
        printf("%-4s %-18s %9s %8s %8s %8s %7s %-9s\n",
               "#", "shape", "ms", "TOPS", "%peak", "%total", "calls", "impl");
        for (int i : order)
        {
            if (prof_ms_[i] <= 0.0) continue;
            const L& q = layers_[i];
            const double tops = prof_flops_[i] / (prof_ms_[i] * 1e9);
            char shape[64];
            snprintf(shape, sizeof(shape), "%d->%d k%ds%d", q.cin_pg, q.cout,
                     q.kh, q.sh);
            printf("%-4d %-18s %9.3f %8.2f %7.1f%% %7.1f%% %7d %-9s\n",
                   i, shape, prof_ms_[i], tops, 100.0 * tops / 15.34,
                   100.0 * prof_ms_[i] / total, prof_calls_[i],
                   ran_impl_[i] < 0 ? "depthwise" : (ran_impl_[i] ? "implicit" : "cublas"));
        }
        const double flops = std::accumulate(prof_flops_.begin(), prof_flops_.end(), 0.0);
        printf("%-4s %-18s %9.3f %8.2f %7.1f%% %7.1f%%\n", "", "合计(conv)",
               total, flops / (total * 1e9), 100.0 * flops / (total * 1e9) / 15.34, 100.0);
        printf("  注：峰值 15.34 TOPS = 1024 core × 8192 ops/clk × 1873 MHz（DP4A）\n");
    }

    void alloc_scratch(int n)
    {
        if (d_pool_) { cudaFree(d_pool_); d_pool_ = nullptr; }
        d_pool_cap_ = (size_t)n * m_->fc_in;
        CUDA_CHECK(cudaMalloc(&d_pool_, d_pool_cap_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_logits_, (size_t)n * m_->fc_out * sizeof(float)));
        // ---- 容量阈值的正确算法（第一版估算错了，OOM 才发现）----
        //
        // 一次 forward 的峰值不是「所有层之和」，而是**同时存活的块之和**。
        // 由于块是按 acquire 顺序分配的（游标单调递增），同一时刻存活的块数
        // 最多是「一条前向路径上的块数」：d_cur / d_id / d_t / d_t2 四个激活
        // + 每层的 col/col+acc/output 三个临时量（用完就进下一层，游标不复位）。
        //
        // 所以池子总大小 ≈ Σ_层(batch × L × Kpad) + Σ_层(batch × L × Cout × 8)
        //              = Σ_层(batch × L × (Kpad + Cout × 8))
        // 对 ResNet-18 CIFAR，逐层累加约 4.3 × batch × L_max 字节。
        //
        // ⚠️ 更简单的做法：**按 GPU 实际可用显存的一半分**。
        //    阈值是「触发重建」的水位，不是「会用多少」——
        //    设成物理上限的一半，天然安全且随机器自适应。
        {
            size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);
            pool_limit_ = free_b / 2;   // 触到一半可用显存就重建
            if (pool_limit_ < (size_t)64 << 20) pool_limit_ = (size_t)64 << 20;
        }
    }

    ~Int8Engine()
    {
        for (float* p : pool_) cudaFree(p);
        for (L& q : layers_)
        {
            cudaFree(q.w_gpu); cudaFree(q.ws_gpu);
            cudaFree(q.bias_gpu); cudaFree(q.zp_gpu);
            cudaFree(q.w_imp_gpu);
            cudaFree(q.w_tap_gpu);
            cudaFree(q.wT_gpu);
        }
        cudaFree(fc_w_gpu_);
        cudaFree(d_pool_);
        cudaFree(d_logits_);
        if (prof_ev_[0]) cudaEventDestroy(prof_ev_[0]);
        if (prof_ev_[1]) cudaEventDestroy(prof_ev_[1]);
        ctx_.destroy();
    }

private:
    struct L {
        int cout = 0, cin = 0, kh = 0, kw = 0, sh = 1, sw = 1, ph = 0, pw = 0;
        int K = 0, Kpad = 0;
        int groups = 1, cin_pg = 0, cout_pg = 0;
        size_t gstride = 0;             // 每组权重矩阵的元素数 = Kpad*cout_pg
        std::vector<int8_t> w_dev;      // groups × (Kpad, cout_pg) 行主序
        std::vector<float> wscale;      // cout
        int8_t* w_gpu = nullptr;
        int8_t* w_imp_gpu = nullptr;   // implicit GEMM 的 DP4A 权重
        size_t w_imp_bytes = 0;
        int8_t* w_tap_gpu = nullptr;   // tap-major GEMM 权重
        size_t w_tap_bytes = 0;
        int8_t* wT_gpu = nullptr;      // 转置 GEMM: (cout_pg, Kpad)，每组一块
        int Cpad = 0, Ktap = 0, KpadTap = 0;   // tap-major K 维度
        float* ws_gpu = nullptr;
        float* bias_gpu = nullptr;
        float* zp_gpu = nullptr;
    };

    static int out_h(const ConvSpec& c, int h)
    {
        return (h + 2 * c.ph - c.kh) / c.sh + 1;
    }
    static int out_w(const ConvSpec& c, int w)
    {
        return (w + 2 * c.pw - c.kw) / c.sw + 1;
    }

    // 一层 conv 的临时缓冲估算：col（每组 Kpad）+ acc int32 + 输出 fp32。
    // depthwise 也会被高估一份 col，作为显存池阈值是安全方向。
    static size_t conv_scratch_bytes(const ConvSpec& c, int n, int h, int w)
    {
        const int oh = out_h(c, h), ow = out_w(c, w);
        const int K = c.cin * c.kh * c.kw;
        const int Kp = ((K + 7) / 8) * 8;
        const long long M = (long long)n * oh * ow;
        return (size_t)M * Kp * (size_t)c.groups
             + (size_t)M * c.cout * 4
             + (size_t)M * c.cout * sizeof(float);
    }

    // 跑第 li 个 conv（含按 relu 标志决定激活），并更新输出形状。
    // ⚠️ ReLU 不在 requant 里做，而是单独一个 kernel：
    //    requant 的 relu 参数只在「该 conv 输出直接进下一个 conv」时用得上，
    //    但 block 的末 conv 之后还有残差相加，ReLU 必须等到 add 完再做。
    //    两者混在一起会算错 —— 这是拓扑与算子融合的典型冲突。
    void run_conv(int li, const float* in, int n, int c, int h, int w,
                  float** out, int* oc, int* oh, int* ow)
    {
        const bool prof = prof_on_ && !prof_skip_this_;
        if (prof) cudaEventRecord(prof_ev_[0], ptqint8::get_stream());
        int8_conv(li, in, n, c, h, w, out);
        const ConvSpec& cs = m_->convs[li];
        *oh = out_h(cs, h);
        *ow = out_w(cs, w);
        *oc = cs.cout;
        if (prof)
        {
            // event 计时口径：同步等这一层跑完再读，得到的是「这一层独占 GPU」
            // 的时间。层与层之间没有重叠（同一 stream 上串行），所以可以直接相加。
            cudaEventRecord(prof_ev_[1], ptqint8::get_stream());
            cudaEventSynchronize(prof_ev_[1]);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, prof_ev_[0], prof_ev_[1]);
            if ((size_t)li >= prof_ms_.size())
            {
                prof_ms_.resize(layers_.size(), 0.0);
                prof_flops_.resize(layers_.size(), 0.0);
                prof_calls_.resize(layers_.size(), 0);
            }
            const L& q = layers_[li];
            prof_ms_[li] += ms;
            prof_calls_[li] += 1;
            prof_flops_[li] += 2.0 * (double)n * (*oh) * (*ow) * (double)q.K * q.cout;
        }
        // 激活在 int8_conv 内部完成：relu==1 融合进 requant，relu6 单独跑。
        (void)cs;
    }

    int int8_conv(size_t li, const float* in, int n, int c, int h, int w, float** out)
    {
        if (aq_.size() != layers_.size())
        {
            fprintf(stderr, "[FAIL] set_calib 尚未调用（aq=%zu, layers=%zu）\n",
                    aq_.size(), layers_.size());
            std::exit(1);
        }
        const L& q = layers_[li];
        if (c != q.cin_pg * q.groups)
        {
            fprintf(stderr, "[FAIL] conv[%zu] 输入通道 %d != cin(%d)*groups(%d)\n",
                    li, c, q.cin_pg, q.groups);
            std::exit(1);
        }
        const int OH = out_h(m_->convs[li], h);
        const int OW = out_w(m_->convs[li], w);
        const long long Lpos = (long long)OH * OW;
        const long long M = (long long)n * Lpos;
        const int qmax = (1 << (a_bits_ - 1)) - 1;
        const bool fuse_relu = (m_->convs[li].relu == 1);
        const bool relu6 = (m_->convs[li].relu == 2);

        // ---- depthwise（cin=1, groups=cout）：专用融合 kernel ----
        // 不走 GEMM：每组输出只有 1 个通道，_int_mm 的 N=1 既慢又不保证支持。
        if (q.groups == q.cout && q.cin_pg == 1)
        {
            float* o = acquire((size_t)M * q.cout * sizeof(float));
            ptqint8::depthwise_conv_launch(
                in, o, q.w_gpu, q.ws_gpu, q.bias_gpu, aq_[li].scale, qmax,
                n, c, h, w, q.kh, q.kw, q.sh, q.sw, q.ph, q.pw, OH, OW,
                q.Kpad, grid_for((size_t)M * q.cout));
            CUDA_CHECK(cudaGetLastError());
            if (m_->convs[li].relu)
            {
                ptqint8::relu_launch(o, (size_t)M * q.cout, grid_for((size_t)M * q.cout),
                                     m_->convs[li].relu == 2 ? 2 : 1);
                CUDA_CHECK(cudaGetLastError());
            }
            *out = o;
            return q.cout;
        }

        // ---- 普通 conv（groups=1）或一般分组 conv（逐组 GEMM）----
        float* o = acquire((size_t)M * q.cout * sizeof(float));

        // ---- 转置 GEMM：W^T(cout,K) @ colT(K,M) → acc(cout,M)，直接 NCHW ----
        // 与旧路径相比：GEMM 的 N 维变成空间 M（大），输出天然是 NCHW 通道主序，
        // requant 不再需要 shared 转置；im2col 写 K-major。
        if (use_transposed_ && q.wT_gpu && (M % 8 == 0))
        {
            int8_t* colT = reinterpret_cast<int8_t*>(
                acquire((size_t)M * q.Kpad));
            for (int g = 0; g < q.groups; g++)
            {
                ptqint8::quant_im2col_kmajor_launch(
                    in, colT, aq_[li].scale, qmax,
                    n, c, h, w, q.kh, q.kw, q.sh, q.sw, q.ph, q.pw,
                    OH, OW, q.K, q.Kpad, g * q.cin_pg);
                CUDA_CHECK(cudaGetLastError());
                torch::Tensor acc_t = ptqint8::int8_gemm(
                    q.wT_gpu + (size_t)g * q.cout_pg * q.Kpad,
                    q.cout_pg, q.Kpad, colT, M);
                int32_t* acc = acc_t.data_ptr<int32_t>();
                ptqint8::requant_t_launch(
                    acc, o, q.ws_gpu + (size_t)g * q.cout_pg,
                    q.bias_gpu + (size_t)g * q.cout_pg,
                    q.zp_gpu + (size_t)g * q.cout_pg,
                    aq_[li].scale, q.cout_pg, (int)Lpos, n, fuse_relu,
                    q.cout, g * q.cout_pg);
                CUDA_CHECK(cudaGetLastError());
            }
            if (relu6)
            {
                ptqint8::relu_launch(o, (size_t)M * q.cout,
                                     grid_for((size_t)M * q.cout), 2);
                CUDA_CHECK(cudaGetLastError());
            }
            *out = o;
            return q.cout;
        }

        // ---- implicit GEMM：不落 im2col、不落 int32 acc ----
        // 覆盖普通卷积（groups==1 且 cout 能被 64/128 整除）。分组/depthwise
        // 在 build 期就走 dedicated kernel，不属于这里的"回退"。
        // ⭐ 按层选择：want_imp 来自层选择表（layer_impl.h）。表里这一层选
        //   implicit 才走这里，否则直接落到下面的 im2col+cublas。两条路的
        //   int32 累加逐元素相同（§10.2），所以选择只影响性能，不影响任何数值。
        const char* impl_how = "?";
        const LayerImpl want_impl = impl_.select(
            (int)li, q.cin_pg * q.groups, q.cout, q.kh, q.kw, q.sh, q.sw, q.ph, q.pw,
            h, w, &impl_how);
        const bool imp_supported = q.groups == 1 && q.w_imp_gpu && (q.cout % 64 == 0);
        const bool want_imp = (want_impl == LayerImpl::Implicit) && imp_supported;
        if (want_impl == LayerImpl::Implicit && !imp_supported &&
            impl_.mode() == LayerImplTable::Mode::AllImplicit)
        {
            fprintf(stderr,
                    "[WARN] implicit 未覆盖 conv[%zu] 形状 k%dx%ds%dx%d pad%d,%d"
                    " → 该层走 im2col（覆盖率统计见 build 日志）\n",
                    li, q.kh, q.kw, q.sh, q.sw, q.ph, q.pw);
        }
        // 表里有这个层号但形状对不上（换了模型）：启发式接管，必须让人看见。
        if (impl_how && std::strcmp(impl_how, "shape-mismatch") == 0 && !impl_warned_[li])
        {
            impl_warned_[li] = 1;
            fprintf(stderr, "[impl] conv[%zu] 形状与层选择表不符 → 用启发式（表不适用）\n", li);
        }
        if (want_imp && q.groups == 1) ran_impl_[li] = 1;
        if (want_imp)
        {
            // 预量化一次；隐式卷积的 grid.y 个输出通道块共用这份 int8，
            // 不再每个 block 重复量化同一张输入图。
            const int C4 = (q.cin_pg + 3) / 4;
            int8_t* qact = reinterpret_cast<int8_t*>(
                acquire((size_t)n * C4 * 4 * h * w));
            ptqint8::quant_act_c4_launch(in, qact, aq_[li].scale, qmax,
                                         n, c, 0, q.cin_pg, h, w);
            CUDA_CHECK(cudaGetLastError());

            // 融合 epilogue：DP4A 累加后直接反量化 + bias + ReLU 写 fp32 NCHW。
            // 它自己按形状选写法（Lpos==1 直接写 / Lpos>1 shared 转置），
            // 只有 tile 超过 16KB 时才拒绝融合（那时 acc+requant 更快）。
            if (use_implicit_epilogue_)
            {
                const bool ok_epi = ptqint8::implicit_conv_out_launch(
                    qact, q.w_imp_gpu, o,
                    q.ws_gpu, q.bias_gpu, q.zp_gpu, aq_[li].scale,
                    n, q.cin_pg, h, w, q.kh, q.kw, q.sh, q.sw, q.ph, q.pw,
                    OH, OW, q.cout, q.cout, 0, fuse_relu,
                    use_implicit_direct_);
                if (ok_epi)
                {
                    if (relu6)
                    {
                        ptqint8::relu_launch(o, (size_t)M * q.cout,
                                             grid_for((size_t)M * q.cout), 2);
                        CUDA_CHECK(cudaGetLastError());
                    }
                    *out = o;
                    return q.cout;
                }
            }

            int32_t* acc = reinterpret_cast<int32_t*>(
                acquire((size_t)M * q.cout * sizeof(int32_t)));
            const bool ok = ptqint8::implicit_conv_launch(
                qact, q.w_imp_gpu, acc,
                n, q.cin_pg, h, w, q.kh, q.kw, q.sh, q.sw, q.ph, q.pw,
                OH, OW, q.cout, use_implicit_direct_);
            if (ok)
            {
                if (Lpos >= 16)
                    ptqint8::requant_tiled_launch(acc, o, q.ws_gpu, q.bias_gpu,
                                                  q.zp_gpu, aq_[li].scale,
                                                  q.cout, (int)Lpos, n, fuse_relu,
                                                  q.cout, 0);
                else
                    ptqint8::requant_launch(acc, o, q.ws_gpu, q.bias_gpu,
                                            q.zp_gpu, aq_[li].scale,
                                            q.cout, (int)Lpos, n, fuse_relu,
                                            q.cout, 0,
                                            grid_for((size_t)M * q.cout));
                CUDA_CHECK(cudaGetLastError());
                if (relu6)
                {
                    ptqint8::relu_launch(o, (size_t)M * q.cout,
                                         grid_for((size_t)M * q.cout), 2);
                    CUDA_CHECK(cudaGetLastError());
                }
                *out = o;
                return q.cout;
            }
            // 走到这里说明 (kh,kw,sh,sw,ph,pw) 不在 implicit 的 6 个实例化形状里。
            // 这不是"性能回退"，是覆盖缺口 —— build 期已统计覆盖率，
            // IMPLICIT_STRICT=1 时会在 build 阶段就失败，不会到运行期才发现。
            fprintf(stderr,
                    "[WARN] implicit 未覆盖 conv[%zu] 形状 k%dx%ds%dx%d pad%d,%d"
                    " → 该层走 im2col（覆盖率统计见 build 日志）\n",
                    li, q.kh, q.kw, q.sh, q.sw, q.ph, q.pw);
        }
        for (int g = 0; g < q.groups; g++)
        {
            // 1) 量化 + im2col 融合 → col (M, Kpad) int8
            //    c_base 指定该组在输入通道维的起点；C 仍是总通道（sample stride）。
            int8_t* col = reinterpret_cast<int8_t*>(acquire((size_t)M * q.KpadTap));
            // 两版 im2col 语义逐位相同（同 half-away 舍入、padding 写 0、同
            // (M,Kpad) 行主序），只在"怎么读输入"上不同：
            //   fast  —— 每个输出位置自己从全局读 kh*kw 个 tap 的 fp32 再量化
            //            （读放大 kh*kw 倍，实测 17-21 GB/s）
            //   tiled —— 每个 block 把涉及的整幅输入量化进 shared 一次，各 tap 从
            //            shared 取（读/量化从每输出元素一次降到每输入像素一次）
            // IM2COL_TILED=0 强制走 fast，便于同二进制 A/B。
            if (!(getenv("IM2COL_TILED") && !strcmp(getenv("IM2COL_TILED"), "0")) && Lpos >= 4)
                ptqint8::quant_im2col_tiled_launch(
                    in, col, aq_[li].scale, qmax,
                    n, c, h, w, q.kh, q.kw, q.sh, q.sw, q.ph, q.pw, OH, OW,
                    q.K, q.Kpad, g * q.cin_pg);
            else
                ptqint8::quant_im2col_fast_launch(
                    in, col, aq_[li].scale, 0, qmax,
                    n, c, h, w, q.kh, q.kw, q.sh, q.sw, q.ph, q.pw, OH, OW,
                    q.K, q.Kpad, g * q.cin_pg);
            CUDA_CHECK(cudaGetLastError());

            // 2) int8 tensor core GEMM → acc (M, cout_pg) int32
            torch::Tensor acc_t = ptqint8::int8_gemm(
                col, M, q.Kpad,
                q.w_gpu + (size_t)g * q.Kpad * q.cout_pg, q.cout_pg);
            int32_t* acc = acc_t.data_ptr<int32_t>();

            // 3) requantize + bias + zp 修正 → fp32 (N, cout, OH*OW)
            //    组输出写回整层输出的 co_base=g*cout_pg 处，因此必须显式给
            //    cout_stride=整层 cout，不能按组内 cout_pg 算输出下标。
            //    relu=false：激活由 run_conv 按拓扑决定（残差相加前不能做）。
            // 转置瓦片版对 Lpos>=16（4x4 及以上）快 1.5–2.7x；Lpos=4/1 时
            // 32x32 瓦片几乎全空，反而慢 2–10x（实测 0.09/0.16 ms vs 0.045/0.023）。
            // 所以按空间尺寸分流，不搞"一刀切更快"的假优化。
            if (Lpos >= 16)
                ptqint8::requant_tiled_launch(acc, o,
                                              q.ws_gpu + (size_t)g * q.cout_pg,
                                              q.bias_gpu + (size_t)g * q.cout_pg,
                                              q.zp_gpu + (size_t)g * q.cout_pg,
                                              aq_[li].scale, q.cout_pg, (int)Lpos, n, fuse_relu,
                                              q.cout, g * q.cout_pg);
            else
                ptqint8::requant_launch(acc, o,
                                        q.ws_gpu + (size_t)g * q.cout_pg,
                                        q.bias_gpu + (size_t)g * q.cout_pg,
                                        q.zp_gpu + (size_t)g * q.cout_pg,
                                        aq_[li].scale, q.cout_pg, (int)Lpos, n, fuse_relu,
                                        q.cout, g * q.cout_pg,
                                        grid_for((size_t)M * q.cout_pg));
            CUDA_CHECK(cudaGetLastError());
        }

        if (relu6)
        {
            ptqint8::relu_launch(o, (size_t)M * q.cout, grid_for((size_t)M * q.cout), 2);
            CUDA_CHECK(cudaGetLastError());
        }
        *out = o;
        return q.cout;
    }

    // 单流时用同步 memcpy（行为与改动前一致）；多流时用异步，
    // 由调用方在整批 kernel 之后自己 cudaStreamSynchronize。
    void copy_h2d(void* dst, const void* src, size_t bytes)
    {
        if (stream_)
            CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream_));
        else
            CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
    }
    void copy_d2h(void* dst, const void* src, size_t bytes)
    {
        if (stream_)
            CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream_));
        else
            CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
    }

    static int grid_for(size_t total)
    {
        long long g = ((long long)total + 255) / 256;
        if (g < 1) g = 1;
        if (g > 65535) g = 65535;
        return (int)g;
    }

    Model* m_ = nullptr;
    int w_bits_ = 8, a_bits_ = 8;
    bool use_implicit_ = true;
    bool use_implicit_epilogue_ = true;
    bool use_implicit_direct_ = true;
    bool use_transposed_ = false;
    bool implicit_strict_ = false;
    int implicit_covered_ = 0, implicit_uncovered_ = 0;
    LayerImplTable impl_;              // 按层选择表（默认实测表；env/JSON 可覆盖）
    std::vector<int8_t> ran_impl_;     // 实际实现：0=cublas 1=implicit -1=depthwise
    std::vector<char> impl_warned_;    // 形状不匹配告警只打一次
    cudaStream_t stream_ = nullptr;      // nullptr = legacy default stream
    size_t d_pool_cap_ = 0;              // d_pool_ 的容量（元素数）
    bool prof_on_ = false;
    cudaEvent_t prof_ev_[2] = {nullptr, nullptr};
    std::vector<double> prof_ms_, prof_flops_;
    std::vector<int> prof_calls_;
    int prof_batch_ = 0;
    bool prof_warmed_ = false, prof_skip_this_ = false;
    ptq::ActRangeMode act_mode_ = ptq::ActRangeMode::AbsMax;
    ptqint8::CublasCtx ctx_;
    std::vector<L> layers_;
    std::vector<ptq::TensorQuant> aq_;
    std::vector<std::pair<float, float>> ranges_;
    void* fc_w_gpu_ = nullptr;
    float* d_pool_ = nullptr;
    float* d_logits_ = nullptr;
    std::vector<float*> pool_;      // 显存池：所有中间缓冲
    int pool_batch_ = -1;          // 当前池对应的 batch 尺寸
    size_t pool_cursor_ = 0;        // 本次 forward 已分配到第几块
    int batch_hint_ = -1;
    std::vector<size_t> pool_size_;
    size_t pool_bytes_ = 0;        // 本轮已申请的字节
    size_t pool_limit_ = (size_t)512 << 20;   // 512 MB起步，触发后翻倍
};

// 校准文件入口：严格校验（8+8n、n==n_conv、有限、lo<=hi）在 ptq_model.h。
// 这里保留薄包装，让调用方只依赖引擎头，不必单独 include ptq_model.h。
inline bool load_calib(const std::string& path, size_t n_conv,
                       std::vector<std::pair<float, float>>* out,
                       std::string* err)
{
    return ptqmodel::load_ptqc(path, n_conv, out, err);
}