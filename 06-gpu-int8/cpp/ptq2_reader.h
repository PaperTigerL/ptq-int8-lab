// PTQ2 二进制解析器（loader 第一阶段：只解析、只校验，不接 kernel）。
//
// 格式权威 = 09-ptq-toolkit/ptq_toolkit/export/topoblocks.py:247-271（写入端）。
// 布局：
//   magic "PTQ2"
//   header 4×int32: n_conv, n_block, stem_pool, n_conv_per_block(ncp)
//   每 conv: 10×int32 (cout,cin,kh,kw,sh,sw,ph,pw,relu,groups)
//            + cout*cin*kh*kw 个小端 float32 权重 + cout 个 bias
//   ⚠️ cin 是**每组**输入通道数（= PyTorch 权重第 2 维），总输入通道 = cin*groups。
//      09 写入端直接取 weight.shape[1]（topoblocks.py:134-138），depthwise 的权重
//      形状是 [cout,1,kh,kw] 且 groups=cout —— 因此 cin=1、groups=32 是合法组合，
//      不能用 groups<=cin 判非法。
//   每 block: (ncp+3)×int32: ncp 个 conv 全局索引(-1 补位), ds, stride, residual
//   fc: 2×int32 (fc_in, fc_out) + fc_in*fc_out 小端 float32 [fc_out][fc_in] 行主序
//       + fc_out 个 bias
// 语义: relu 0=无/1=ReLU/2=ReLU6; residual 0=无/1=相加后ReLU/2=相加后无激活。
//
// 校验策略：宁可拒绝也不半读。任何异常返回 false 且 *err 写清原因。
// 尾部多余字节也算失败——与写入端"blob 长度 = 模型精确大小"（topoblocks.py:271
// p.write_bytes）对齐；留尾巴意味着消费端对"哪些字节属于模型"的理解已经错位，
// 静默忽略会把格式漂移拖到 kernel 里才爆。
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace ptq2 {

struct Ptq2Conv {
    int cout, cin, kh, kw, sh, sw, ph, pw, relu, groups;
    std::vector<float> weight, bias;
};

struct Ptq2Block {
    std::vector<int> convs;  // 长度恒为 ncp；-1 = 补位（见 topoblocks.py:258）
    int ds, stride, residual;
};

struct Ptq2Model {
    int n_conv, n_block, stem_pool, n_conv_per_block;
    std::vector<Ptq2Conv> convs;
    std::vector<Ptq2Block> blocks;
    int fc_in, fc_out;
    std::vector<float> fc_w, fc_b;
};

namespace detail {

class Stream {
public:
    Stream(const std::string& path, std::string* err) : f_(path, std::ios::binary)
    {
        // MinGW libstdc++ 的坑：构造 ifstream 时打不开文件**不会**让 !f_ 成立，
        // 必须显式查 is_open()/tellg，否则错误被推迟到第一次 read（信息误导）。
        if (!f_.is_open()) { *err = "打不开文件: " + path; return; }
        f_.seekg(0, std::ios::end);
        const std::streamoff end = f_.tellg();
        if (!f_ || end < 0) { *err = "读取文件大小失败: " + path; return; }
        size_ = static_cast<size_t>(end);
        f_.seekg(0, std::ios::beg);
        ok_ = true;
    }

    bool ok() const { return ok_; }
    size_t size() const { return size_; }
    size_t pos() const { return pos_; }

    bool i32(int* v, const char* what, std::string* err)
    {
        int32_t x;
        if (!raw(&x, sizeof x, what, err)) return false;
        *v = x;
        return true;
    }

    bool block(void* dst, size_t bytes, const char* what, std::string* err)
    {
        return raw(dst, bytes, what, err);
    }

    // 元素数用 size_t 计，先乘后验，杜绝 int 溢出与越界读
    bool floats(size_t n, std::vector<float>* v, const char* what, std::string* err)
    {
        if (n > (size_ - pos_) / sizeof(float)) {
            char buf[160];
            std::snprintf(buf, sizeof buf,
                          "%s: 需 %zu 个 float (%zu B)，但剩余仅 %zu B —— 文件被截断",
                          what, n, n * sizeof(float), size_ - pos_);
            *err = buf;
            return false;
        }
        v->resize(n);
        return raw(v->data(), n * sizeof(float), what, err);
    }

private:
    bool raw(void* dst, size_t bytes, const char* what, std::string* err)
    {
        if (!ok_) return false;
        if (bytes > size_ - pos_) {
            char buf[160];
            std::snprintf(buf, sizeof buf, "%s: 文件过短（缺 %zu B，文件大小 %zu，已读 %zu）",
                          what, bytes - (size_ - pos_), size_, pos_);
            *err = buf;
            ok_ = false;
            return false;
        }
        f_.read(static_cast<char*>(dst), static_cast<std::streamsize>(bytes));
        if (static_cast<size_t>(f_.gcount()) != bytes) {
            *err = std::string(what) + ": 读取失败";
            ok_ = false;
            return false;
        }
        pos_ += bytes;
        return true;
    }

    std::ifstream f_;
    size_t size_ = 0, pos_ = 0;
    bool ok_ = false;
};

inline bool fail(std::string* err, const std::string& msg)
{
    if (err) *err = msg;
    return false;
}

} // namespace detail

inline bool load_ptq2(const std::string& path, Ptq2Model* out, std::string* err)
{
    detail::Stream s(path, err);
    if (!s.ok()) return false;
    if (s.size() < 4 + 4 * sizeof(int32_t))
        return detail::fail(err, "文件过短: 不足 magic+header（< 20 B）");

    char magic[4] = {0};
    if (!s.block(magic, 4, "magic", err)) return false;
    if (memcmp(magic, "PTQ2", 4) != 0) {
        char buf[96];
        std::snprintf(buf, sizeof buf, "magic 不符: 期望 \"PTQ2\"，实得 '%c%c%c%c' (0x%02x%02x%02x%02x)",
                      magic[0], magic[1], magic[2], magic[3],
                      (uint8_t)magic[0], (uint8_t)magic[1], (uint8_t)magic[2], (uint8_t)magic[3]);
        return detail::fail(err, buf);
    }

    Ptq2Model m;
    if (!s.i32(&m.n_conv, "header.n_conv", err)) return false;
    if (!s.i32(&m.n_block, "header.n_block", err)) return false;
    if (!s.i32(&m.stem_pool, "header.stem_pool", err)) return false;
    if (!s.i32(&m.n_conv_per_block, "header.n_conv_per_block", err)) return false;

    if (m.n_conv < 0) return detail::fail(err, "header: n_conv 为负");
    if (m.n_block < 0) return detail::fail(err, "header: n_block 为负");
    if (m.stem_pool != 0 && m.stem_pool != 1)
        return detail::fail(err, "header: stem_pool 只允许 0/1");
    if (m.n_conv_per_block <= 0)
        return detail::fail(err, "header: n_conv_per_block 必须 > 0");
    if (static_cast<size_t>(m.n_block) * (m.n_conv_per_block + 3) * sizeof(int32_t)
        > s.size())
        return detail::fail(err, "header: n_block×(ncp+3) 已超文件大小（数量造假？）");

    // ---- convs ----
    m.convs.resize(static_cast<size_t>(m.n_conv));
    for (int i = 0; i < m.n_conv; i++) {
        Ptq2Conv& c = m.convs[i];
        int* meta[10] = {&c.cout, &c.cin, &c.kh,     &c.kw, &c.sh,
                         &c.sw,   &c.ph, &c.pw,     &c.relu, &c.groups};
        char what[64];
        for (int k = 0; k < 10; k++) {
            std::snprintf(what, sizeof what, "conv[%d].meta[%d]", i, k);
            if (!s.i32(meta[k], what, err)) return false;
        }
        if (c.cout <= 0 || c.cin <= 0 || c.kh <= 0 || c.kw <= 0)
            return detail::fail(err, [&] {
                char b[128];
                std::snprintf(b, sizeof b, "conv[%d]: cout/cin/kh/kw 必须为正（得 %d/%d/%d/%d）",
                              i, c.cout, c.cin, c.kh, c.kw);
                return std::string(b);
            }());
        if (c.sh < 0 || c.sw < 0 || c.ph < 0 || c.pw < 0)
            return detail::fail(err, [&] {
                char b[128];
                std::snprintf(b, sizeof b, "conv[%d]: stride/padding 为负（%d,%d,%d,%d）",
                              i, c.sh, c.sw, c.ph, c.pw);
                return std::string(b);
            }());
        if (c.relu != 0 && c.relu != 1 && c.relu != 2)
            return detail::fail(err, [&] {
                char b[96];
                std::snprintf(b, sizeof b, "conv[%d]: relu=%d 非法（只允许 0/1/2）", i, c.relu);
                return std::string(b);
            }());
        if (c.groups < 1 || c.cout % c.groups != 0)
            return detail::fail(err, [&] {
                char b[160];
                std::snprintf(b, sizeof b,
                              "conv[%d]: groups=%d 非法（需 groups>=1 且整除 cout=%d；"
                              "cin=%d 为每组输入通道，总输入=%d）",
                              i, c.groups, c.cout, c.cin, c.cin * c.groups);
                return std::string(b);
            }());
        std::snprintf(what, sizeof what, "conv[%d].weight", i);
        if (!s.floats((size_t)c.cout * c.cin * c.kh * c.kw, &c.weight, what, err)) return false;
        std::snprintf(what, sizeof what, "conv[%d].bias", i);
        if (!s.floats((size_t)c.cout, &c.bias, what, err)) return false;
    }

    // ---- blocks ----
    m.blocks.resize(static_cast<size_t>(m.n_block));
    const int ncp = m.n_conv_per_block;
    for (int i = 0; i < m.n_block; i++) {
        Ptq2Block& b = m.blocks[i];
        b.convs.resize(static_cast<size_t>(ncp));
        char what[64];
        for (int k = 0; k < ncp; k++) {
            std::snprintf(what, sizeof what, "block[%d].conv[%d]", i, k);
            if (!s.i32(&b.convs[k], what, err)) return false;
            if (b.convs[k] < -1 || b.convs[k] >= m.n_conv)
                return detail::fail(err, [&] {
                    char b2[160];
                    std::snprintf(b2, sizeof b2,
                                  "block[%d].conv[%d]=%d 越界（合法: -1 补位 或 [0,%d)）",
                                  i, k, b.convs[k], m.n_conv);
                    return std::string(b2);
                }());
        }
        std::snprintf(what, sizeof what, "block[%d].ds", i);
        if (!s.i32(&b.ds, what, err)) return false;
        if (b.ds < -1 || b.ds >= m.n_conv)
            return detail::fail(err, [&] {
                char b2[160];
                std::snprintf(b2, sizeof b2, "block[%d].ds=%d 越界（合法: -1 或 [0,%d)）",
                              i, b.ds, m.n_conv);
                return std::string(b2);
            }());
        std::snprintf(what, sizeof what, "block[%d].stride", i);
        if (!s.i32(&b.stride, what, err)) return false;
        if (b.stride < 1)
            return detail::fail(err, [&] {
                char b2[96];
                std::snprintf(b2, sizeof b2, "block[%d].stride=%d < 1", i, b.stride);
                return std::string(b2);
            }());
        std::snprintf(what, sizeof what, "block[%d].residual", i);
        if (!s.i32(&b.residual, what, err)) return false;
        if (b.residual != 0 && b.residual != 1 && b.residual != 2)
            return detail::fail(err, [&] {
                char b2[96];
                std::snprintf(b2, sizeof b2, "block[%d].residual=%d 非法（只允许 0/1/2）",
                              i, b.residual);
                return std::string(b2);
            }());
    }

    // ---- fc ----
    if (!s.i32(&m.fc_in, "fc.fc_in", err)) return false;
    if (!s.i32(&m.fc_out, "fc.fc_out", err)) return false;
    if (m.fc_in <= 0 || m.fc_out <= 0)
        return detail::fail(err, [&] {
            char b[96];
            std::snprintf(b, sizeof b, "fc: fc_in/fc_out 必须为正（得 %d/%d）",
                          m.fc_in, m.fc_out);
            return std::string(b);
        }());
    if (!s.floats((size_t)m.fc_in * m.fc_out, &m.fc_w, "fc.weight", err)) return false;
    if (!s.floats((size_t)m.fc_out, &m.fc_b, "fc.bias", err)) return false;

    if (s.pos() != s.size()) {
        char buf[160];
        std::snprintf(buf, sizeof buf,
                      "EOF 校验失败: 解析止于 %zu B，文件 %zu B —— 尾部多余 %zu B，拒绝",
                      s.pos(), s.size(), s.size() - s.pos());
        return detail::fail(err, buf);
    }

    *out = std::move(m);
    if (err) err->clear();
    return true;
}

} // namespace ptq2
