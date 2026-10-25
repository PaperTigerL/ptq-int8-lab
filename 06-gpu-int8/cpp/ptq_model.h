// 06 引擎的统一模型层：PTQW / PTQ2 按 magic 分派，PTQC 严格校验。
//
// ⭐ 为什么把模型结构抽成 CPU-only 头文件：
//   1. 旧 Model::load 只认 PTQW，且把解析写在 int8_engine.cuh 里 ——
//      那个头 include 了 cublas/torch，想给它写单测就得有 GPU 环境。
//   2. PTQ2 已经有独立读取器 ptq2_reader.h（T-32 验收），引擎不应再写第二份
//      解析逻辑；两份迟早漂移，而漂移只会在 GPU 上以「维度错」的形式炸。
//   所以本文件只做三件事，全部与 CUDA 无关：
//     · 把两种格式统一成 Model（ConvSpec/BlockSpec）
//     · 校验拓扑连通性（通道/尺寸/relu/residual/groups）
//     · 严格读取 PTQC（长度、层数、有限性、lo<=hi）
//   int8_engine.cuh 只保留执行逻辑；ptq_model_test.cpp 可在无 GPU 机器上跑。
//
// 分组卷积的语义（与 09 topoblocks.py 写入端逐字对齐）：
//   ConvSpec.cin 是**每组**输入通道数（PyTorch 权重第 2 维），
//   总输入通道 = cin * groups。
//   depthwise：cin=1, groups=cout；权重按 [cout][1][kh][kw] 存。
//   一般分组：每组权重 [cout/groups][cin][kh][kw]，组间不共享。
#pragma once

#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "ptq2_reader.h"

namespace ptqmodel {

inline std::string fmt_s(const char* f, ...)
{
    char buf[512];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return std::string(buf);
}

struct ConvSpec {
    int cout = 0, cin = 0, kh = 0, kw = 0;
    int sh = 1, sw = 1, ph = 0, pw = 0;
    int relu = 0;        // 0=无激活 1=ReLU 2=ReLU6
    int groups = 1;      // 分组数；cin 为每组输入通道
    std::vector<float> weight;  // [cout][cin][kh][kw]，Cout-major
    std::vector<float> bias;    // [cout]

    int in_channels() const { return cin * groups; }
};

struct BlockSpec {
    // 块内 conv 的全局索引，按执行顺序；-1 = 尾部补位。
    std::vector<int> convs;
    int ds = -1;          // identity 分支的 conv 索引；-1 = 直连
    int stride = 1;       // 块级 stride（仅用于文档/预算；真实 stride 在 conv 上）
    int residual = 1;     // 0=无残差 1=相加后 ReLU 2=相加后无激活
};

struct Model {
    std::vector<ConvSpec> convs;
    std::vector<BlockSpec> blocks;
    int stem_pool = 0;
    int fc_in = 0, fc_out = 0;
    std::vector<float> fc_w, fc_b;

    bool load(const std::string& path, std::string* err)
    {
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open()) { set_err(err, "打不开模型文件: " + path); return false; }
        char magic[4] = {0};
        f.read(magic, 4);
        if (f.gcount() != 4) { set_err(err, path + ": 文件不足 4 B，读不到 magic"); return false; }
        f.close();

        bool ok = false;
        if (memcmp(magic, "PTQW", 4) == 0)
            ok = load_ptqw(path, err);
        else if (memcmp(magic, "PTQ2", 4) == 0)
            ok = load_ptq2_file(path, err);
        else
            return set_err(err, path + ": magic 不符（期望 PTQW/PTQ2，实得 '" +
                                    printable_magic(magic) + "'）");
        if (!ok) return false;
        return validate(err);
    }

    // 兼容旧调用点：打印错误并返回 false。
    bool load(const std::string& path)
    {
        std::string err;
        if (!load(path, &err)) { fprintf(stderr, "%s\n", err.c_str()); return false; }
        return true;
    }

    // 拓扑自检：通道、空间尺寸、残差形状、fc 输入。
    // input_h/input_w 只影响空间尺寸校验；ResNet/MobileNet 的结构与输入尺寸无关，
    // 用 32×32（CIFAR）作为代表值即可。
    bool validate(std::string* err, int input_h = 32, int input_w = 32) const
    {
        if (convs.empty()) return set_err(err, "模型没有 conv");
        if (fc_in <= 0 || fc_out <= 0)
            return set_err(err, "fc_in/fc_out 必须为正（得 " +
                                    std::to_string(fc_in) + "/" + std::to_string(fc_out) + "）");

        const ConvSpec& stem = convs[0];
        int ch = stem.cout, hh = 0, ww = 0;
        conv_out_hw(stem, input_h, input_w, &hh, &ww);
        if (stem_pool)
        {
            hh = (hh + 2 - 3) / 2 + 1;
            ww = (ww + 2 - 3) / 2 + 1;
        }

        for (size_t bi = 0; bi < blocks.size(); bi++)
        {
            const BlockSpec& blk = blocks[bi];
            if (blk.residual < 0 || blk.residual > 2)
                return set_err(err, "block[" + std::to_string(bi) + "]: residual=" +
                                        std::to_string(blk.residual) + " 非法（只允许 0/1/2）");
            if (blk.stride < 1)
                return set_err(err, "block[" + std::to_string(bi) + "]: stride < 1");

            bool saw_pad = false, any = false;
            int tch = ch, thh = hh, tww = ww;
            for (size_t k = 0; k < blk.convs.size(); k++)
            {
                const int ci = blk.convs[k];
                if (ci == -1) { saw_pad = true; continue; }
                if (saw_pad)
                    return set_err(err, "block[" + std::to_string(bi) +
                                            "]: -1 补位必须集中在尾部（位置 " +
                                            std::to_string(k) + " 的 conv 索引出现在补位之后）");
                if (ci < 0 || ci >= (int)convs.size())
                    return set_err(err, "block[" + std::to_string(bi) + "]: conv 索引 " +
                                            std::to_string(ci) + " 越界");
                const ConvSpec& cs = convs[ci];
                if (k == 0 && cs.in_channels() != ch)
                    return set_err(err, "block[" + std::to_string(bi) + "] 首 conv 输入通道 " +
                                            std::to_string(cs.in_channels()) + " != 上游 " +
                                            std::to_string(ch) + "（cin=" +
                                            std::to_string(cs.cin) + ", groups=" +
                                            std::to_string(cs.groups) + "）");
                if (k > 0 && cs.in_channels() != tch)
                    return set_err(err, "block[" + std::to_string(bi) + "] conv[" +
                                            std::to_string(k) + "] 输入通道 " +
                                            std::to_string(cs.in_channels()) + " != 上游 " +
                                            std::to_string(tch));
                conv_out_hw(cs, thh, tww, &thh, &tww);
                tch = cs.cout;
                any = true;
            }
            if (!any)
                return set_err(err, "block[" + std::to_string(bi) + "] 没有任何有效 conv");

            if (blk.residual == 0)
            {
                if (blk.ds >= 0)
                    return set_err(err, "block[" + std::to_string(bi) +
                                            "]: residual=0（无残差）却给了 ds");
            }
            else
            {
                int ich = ch, ihh = hh, iww = ww;
                if (blk.ds >= 0)
                {
                    if (blk.ds >= (int)convs.size())
                        return set_err(err, "block[" + std::to_string(bi) + "]: ds 索引越界");
                    const ConvSpec& dsc = convs[blk.ds];
                    if (dsc.in_channels() != ch)
                        return set_err(err, "block[" + std::to_string(bi) + "] ds 输入通道 " +
                                                std::to_string(dsc.in_channels()) + " != 上游 " +
                                                std::to_string(ch));
                    conv_out_hw(dsc, ihh, iww, &ihh, &iww);
                    ich = dsc.cout;
                }
                if (tch != ich || thh != ihh || tww != iww)
                    return set_err(err, "block[" + std::to_string(bi) +
                                            "]: 残差形状不匹配：主支路 (" +
                                            std::to_string(tch) + "," + std::to_string(thh) + "," +
                                            std::to_string(tww) + ") vs identity (" +
                                            std::to_string(ich) + "," + std::to_string(ihh) + "," +
                                            std::to_string(iww) + ")");
            }
            ch = tch; hh = thh; ww = tww;
        }

        if (ch != fc_in)
            return set_err(err, "fc_in=" + std::to_string(fc_in) + " != 拓扑输出通道 " +
                                    std::to_string(ch));
        if ((size_t)fc_in * (size_t)fc_out != fc_w.size())
            return set_err(err, "fc 权重数量不符：期望 " +
                                    std::to_string((size_t)fc_in * fc_out) + "，实得 " +
                                    std::to_string(fc_w.size()));
        if ((size_t)fc_out != fc_b.size())
            return set_err(err, "fc bias 数量不符");
        return true;
    }

    int n_grouped() const
    {
        int n = 0;
        for (const ConvSpec& c : convs) if (c.groups != 1) n++;
        return n;
    }
    int n_relu6() const
    {
        int n = 0;
        for (const ConvSpec& c : convs) if (c.relu == 2) n++;
        return n;
    }
    int n_residual2() const
    {
        int n = 0;
        for (const BlockSpec& b : blocks) if (b.residual == 2) n++;
        return n;
    }
    int n_depthwise() const
    {
        int n = 0;
        for (const ConvSpec& c : convs)
            if (c.cin == 1 && c.groups == c.cout) n++;
        return n;
    }

private:
    static bool set_err(std::string* err, const std::string& msg)
    {
        if (err) *err = msg;
        return false;
    }

    static std::string printable_magic(const char m[4])
    {
        std::string s;
        for (int i = 0; i < 4; i++)
        {
            const unsigned char c = (unsigned char)m[i];
            if (c >= 32 && c < 127) s.push_back((char)c);
            else { char b[8]; std::snprintf(b, sizeof b, "\\x%02x", c); s += b; }
        }
        return s;
    }

    static bool conv_out_hw(const ConvSpec& c, int h, int w, int* oh, int* ow)
    {
        *oh = (h + 2 * c.ph - c.kh) / c.sh + 1;
        *ow = (w + 2 * c.pw - c.kw) / c.sw + 1;
        return *oh > 0 && *ow > 0;
    }

    bool load_ptqw(const std::string& path, std::string* err)
    {
        ptq2::detail::Stream s(path, err);
        if (!s.ok()) return false;

        char magic[4] = {0};
        if (!s.block(magic, 4, "magic", err)) return false;
        if (memcmp(magic, "PTQW", 4) != 0)
            return set_err(err, "PTQW: magic 不符");

        Model m;
        int n_conv = 0, n_block = 0;
        if (!s.i32(&n_conv, "header.n_conv", err)) return false;
        if (!s.i32(&n_block, "header.n_block", err)) return false;
        if (!s.i32(&m.stem_pool, "header.stem_pool", err)) return false;
        if (n_conv <= 0) return set_err(err, "PTQW: n_conv 必须 > 0");
        if (n_block < 0) return set_err(err, "PTQW: n_block 为负");
        if (m.stem_pool != 0 && m.stem_pool != 1)
            return set_err(err, "PTQW: stem_pool 只允许 0/1");

        m.convs.resize((size_t)n_conv);
        for (int i = 0; i < n_conv; i++)
        {
            ConvSpec& c = m.convs[i];
            c.groups = 1;
            int* meta[9] = {&c.cout, &c.cin, &c.kh, &c.kw, &c.sh,
                            &c.sw, &c.ph, &c.pw, &c.relu};
            char what[64];
            for (int k = 0; k < 9; k++)
            {
                std::snprintf(what, sizeof what, "conv[%d].meta[%d]", i, k);
                if (!s.i32(meta[k], what, err)) return false;
            }
            if (c.cout <= 0 || c.cin <= 0 || c.kh <= 0 || c.kw <= 0)
                return set_err(err, fmt("conv[%d]: cout/cin/kh/kw 必须为正（得 %d/%d/%d/%d）",
                                        i, c.cout, c.cin, c.kh, c.kw));
            if (c.sh < 1 || c.sw < 1 || c.ph < 0 || c.pw < 0)
                return set_err(err, fmt("conv[%d]: stride 必须 >=1、padding >=0", i));
            if (c.relu != 0 && c.relu != 1 && c.relu != 2)
                return set_err(err, fmt("conv[%d]: relu=%d 非法（只允许 0/1/2）", i, c.relu));

            size_t nw = 0;
            if (!safe_mul4((size_t)c.cout, (size_t)c.cin, (size_t)c.kh, (size_t)c.kw, &nw,
                           "PTQW conv 权重", err)) return false;
            std::snprintf(what, sizeof what, "conv[%d].weight", i);
            if (!s.floats(nw, &c.weight, what, err)) return false;
            std::snprintf(what, sizeof what, "conv[%d].bias", i);
            if (!s.floats((size_t)c.cout, &c.bias, what, err)) return false;
        }

        m.blocks.resize((size_t)n_block);
        for (int i = 0; i < n_block; i++)
        {
            int c1 = 0, c2 = 0, ds = -1, stride = 1;
            if (!s.i32(&c1, "block.c1", err)) return false;
            if (!s.i32(&c2, "block.c2", err)) return false;
            if (!s.i32(&ds, "block.ds", err)) return false;
            if (!s.i32(&stride, "block.stride", err)) return false;
            if (c1 < 0 || c1 >= n_conv || c2 < 0 || c2 >= n_conv)
                return set_err(err, fmt("PTQW block[%d]: conv 索引越界", i));
            if (ds < -1 || ds >= n_conv)
                return set_err(err, fmt("PTQW block[%d]: ds 越界", i));
            if (stride < 1)
                return set_err(err, fmt("PTQW block[%d]: stride < 1", i));
            BlockSpec b;
            b.convs = {c1, c2};
            b.ds = ds;
            b.stride = stride;
            b.residual = 1;     // PTQW 是 ResNet-18 的 blocks 格式，恒有残差+ReLU
            m.blocks[i] = std::move(b);
        }

        if (!s.i32(&m.fc_in, "fc.fc_in", err)) return false;
        if (!s.i32(&m.fc_out, "fc.fc_out", err)) return false;
        if (m.fc_in <= 0 || m.fc_out <= 0)
            return set_err(err, "PTQW: fc_in/fc_out 必须为正");
        size_t nfw = 0;
        if (!safe_mul4((size_t)m.fc_in, (size_t)m.fc_out, 1, 1, &nfw, "PTQW fc", err))
            return false;
        if (!s.floats(nfw, &m.fc_w, "fc.weight", err)) return false;
        if (!s.floats((size_t)m.fc_out, &m.fc_b, "fc.bias", err)) return false;

        if (s.pos() != s.size())
            return set_err(err, fmt("PTQW EOF 校验失败: 解析止于 %zu B，文件 %zu B —— "
                                    "尾部多余 %zu B，拒绝",
                                    s.pos(), s.size(), s.size() - s.pos()));
        *this = std::move(m);
        if (err) err->clear();
        return true;
    }

    bool load_ptq2_file(const std::string& path, std::string* err)
    {
        ptq2::Ptq2Model pm;
        if (!ptq2::load_ptq2(path, &pm, err)) return false;

        Model m;
        m.stem_pool = pm.stem_pool;
        m.convs.resize(pm.convs.size());
        for (size_t i = 0; i < pm.convs.size(); i++)
        {
            const ptq2::Ptq2Conv& src = pm.convs[i];
            ConvSpec& d = m.convs[i];
            d.cout = src.cout; d.cin = src.cin;
            d.kh = src.kh; d.kw = src.kw;
            d.sh = src.sh; d.sw = src.sw;
            d.ph = src.ph; d.pw = src.pw;
            d.relu = src.relu;
            d.groups = src.groups;
            d.weight = src.weight;
            d.bias = src.bias;
        }
        m.blocks.resize(pm.blocks.size());
        for (size_t i = 0; i < pm.blocks.size(); i++)
        {
            const ptq2::Ptq2Block& src = pm.blocks[i];
            BlockSpec& d = m.blocks[i];
            d.convs = src.convs;
            d.ds = src.ds;
            d.stride = src.stride;
            d.residual = src.residual;
        }
        m.fc_in = pm.fc_in;
        m.fc_out = pm.fc_out;
        m.fc_w = std::move(pm.fc_w);
        m.fc_b = std::move(pm.fc_b);

        if (m.fc_in <= 0 || m.fc_out <= 0)
            return set_err(err, "PTQ2: fc_in/fc_out 必须为正");
        *this = std::move(m);
        if (err) err->clear();
        return true;
    }

    static bool safe_mul4(size_t a, size_t b, size_t c, size_t d, size_t* out,
                          const char* what, std::string* err)
    {
        // 先除以再比较，避免 a*b*c*d 在 size_t 上溢出成一个小数字，
        // 让后面的 remaining 检查误判为「文件够长」。
        size_t v = a;
        const size_t lim = std::numeric_limits<size_t>::max();
        if (b != 0 && v > lim / b) return set_err(err, std::string(what) + ": 元素数溢出");
        v *= b;
        if (c != 0 && v > lim / c) return set_err(err, std::string(what) + ": 元素数溢出");
        v *= c;
        if (d != 0 && v > lim / d) return set_err(err, std::string(what) + ": 元素数溢出");
        v *= d;
        *out = v;
        return true;
    }

    static std::string fmt(const char* f, ...)
    {
        char buf[512];
        va_list ap;
        va_start(ap, f);
        std::vsnprintf(buf, sizeof buf, f, ap);
        va_end(ap);
        return std::string(buf);
    }
};

// PTQC：magic "PTQC" + int32 n + n 对小端 float32 (lo,hi)。
// 严格口径（H-20261022-002）：
//   · 文件长度必须恰好 8 + 8n
//   · n 必须等于模型 conv 数
//   · lo/hi 必须有限且 lo <= hi
// 任何一条不满足都返回 false 并写清原因，**不 exit、不越界**。
inline bool load_ptqc(const std::string& path, size_t n_expected,
                      std::vector<std::pair<float, float>>* out,
                      std::string* err = nullptr)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open())
    {
        if (err) *err = "打不开校准文件: " + path;
        return false;
    }
    f.seekg(0, std::ios::end);
    const std::streamoff end = f.tellg();
    if (!f || end < 0)
    {
        if (err) *err = "读取校准文件大小失败: " + path;
        return false;
    }
    const size_t size = (size_t)end;
    f.seekg(0, std::ios::beg);

    if (size < 8)
    {
        if (err) *err = fmt_s("PTQC 文件过短: %zu B（至少 8 B 的 magic+n）", size);
        return false;
    }
    char magic[4] = {0};
    int32_t n = -1;
    f.read(magic, 4);
    f.read((char*)&n, 4);
    if (f.gcount() != 4 || memcmp(magic, "PTQC", 4) != 0)
    {
        if (err) *err = "PTQC magic 不符（期望 \"PTQC\"）";
        return false;
    }
    if (n < 0)
    {
        if (err) *err = fmt_s("PTQC n=%d 为负", n);
        return false;
    }
    const size_t want = 8 + 8 * (size_t)n;
    if (size != want)
    {
        if (err) *err = fmt_s("PTQC 长度不符: 文件 %zu B，按 n=%d 应为 %zu B"
                              "（8+8n；截断或尾部多余均拒绝）",
                              size, n, want);
        return false;
    }
    if ((size_t)n != n_expected)
    {
        if (err) *err = fmt_s("PTQC 层数不符: n=%d != 模型 conv 数 %zu", n, n_expected);
        return false;
    }

    out->resize((size_t)n);
    for (int i = 0; i < n; i++)
    {
        float lo = 0, hi = 0;
        f.read((char*)&lo, 4);
        f.read((char*)&hi, 4);
        if (!f)
        {
            if (err) *err = fmt_s("PTQC 读第 %d 对 (lo,hi) 失败", i);
            return false;
        }
        if (!std::isfinite(lo) || !std::isfinite(hi))
        {
            if (err) *err = fmt_s("PTQC 第 %d 层范围非有限: lo=%g hi=%g", i, lo, hi);
            return false;
        }
        if (lo > hi)
        {
            if (err) *err = fmt_s("PTQC 第 %d 层范围非法: lo=%g > hi=%g", i, lo, hi);
            return false;
        }
        (*out)[i] = {lo, hi};
    }
    if (err) err->clear();
    return true;
}

} // namespace ptqmodel
