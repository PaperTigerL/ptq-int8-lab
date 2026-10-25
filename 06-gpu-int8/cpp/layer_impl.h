// 按层选择「implicit GEMM」还是「im2col + cublas GEMM」—— 数据驱动的混合实现。
//
// 为什么需要它（ROOFLINE-int8.md §10.2 的结论）：
//   全线 implicit（0.0386 ms/img）比全线 im2col+cublas（0.0261）慢，但**不是每层都慢**：
//   l1 的四层（64->64 k3s1 8x8）和末段三层（512->512 k3s1 1x1）implicit 更快，
//   其余 13 层 cublas 明显更快。既然两条路都已 bit-exact，最优解就是**按层选**。
//
// 表从哪来：
//   1) 默认：layer_impl_default.h（由 bench_layers --select --emit-header 生成，
//      同 batch、100 次预热、两路交错计时取最快、int32 逐元素比对通过后才写入）；
//   2) 文件：LAYER_IMPL_JSON=/path/to/layer_impl.json（同上格式，可手改）；
//   3) 环境变量：LAYER_IMPL=all:implicit|all:cublas|auto|0:c,1:i,17:i,...
//      （c=cublas，i=implicit；未列出的层仍按表）
//   4) 最高优先级：USE_IMPLICIT_GEMM=1/0 —— 保留为全局 A/B 开关（实验对照用）。
//
// 退化规则（表不适用时，绝不静默沿用错误的选择）：
//   · 层号在表里但形状对不上（换了模型）→ 回落到形状启发式，并由引擎打印一次告警；
//   · 层号不在表里 → 同上。
//   启发式只保留实测赢面最大的两种形状，其余一律 cublas（保守方向）。
#pragma once

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace ptqint8 {

enum class LayerImpl : int8_t { Cublas = 0, Implicit = 1 };

struct LayerImplEntry
{
    int layer = -1;
    int cin = 0, cout = 0;                   // cin = 该层总输入通道（cin_pg*groups）
    int kh = 0, kw = 0, sh = 1, sw = 1, ph = 0, pw = 0;
    int h = 0, w = 0;                        // 该层输入空间尺寸
    LayerImpl impl = LayerImpl::Cublas;
    double t_cublas_ms = 0.0, t_implicit_ms = 0.0;   // 生成时的实测（同一 batch）
    int batch = 0;
    bool any_shape = false;                  // 环境变量覆盖：只按层号生效
};

inline const char* layer_impl_name(LayerImpl v)
{
    return v == LayerImpl::Implicit ? "implicit" : "cublas";
}

inline bool layer_impl_parse_name(const std::string& s, LayerImpl* out)
{
    if (s == "implicit" || s == "i" || s == "imp") { *out = LayerImpl::Implicit; return true; }
    if (s == "cublas" || s == "c" || s == "im2col") { *out = LayerImpl::Cublas; return true; }
    return false;
}

// 表不适用时的启发式。依据（ROOFLINE §10.2 逐层实测）：
//   · l1：M=32768 很大、N=64 只有半个 cublas 瓦片、K=576 —— implicit 赢；
//   · 末段：M=512 很小、N=512、Lpos=1 —— cublas 128x128 瓦片吃不满，implicit 赢；
//   其他形状（尤其 stem 的 7x7、Lpos=16 的 4x4）cublas 赢，故默认 cublas。
inline LayerImpl layer_impl_heuristic(int cin, int cout, int kh, int kw, int sh, int sw,
                                      int h, int w)
{
    const long long lpos = (long long)h * w;
    if (cin == cout && kh == 3 && kw == 3 && sh == 1 && sw == 1)
    {
        if (cin == 64 && lpos == 64) return LayerImpl::Implicit;
        if (cin == 512 && lpos == 1) return LayerImpl::Implicit;
    }
    return LayerImpl::Cublas;
}

// 实测生成的默认表（bench_layers --select --emit-header）。
// 必须放在类定义之前：类内成员函数直接引用这个数组。
#include "layer_impl_default.h"

// ------------------------------------------------------------------ JSON
// 只解析 bench_layers 生成的那种「扁平对象数组」，不追求通用 JSON。
// 每个层对象必须包含：layer/cin/cout/kh/kw/sh/sw/ph/pw/h/w/impl；
// t_*_ms 可选（用于打印来源）。
inline bool layer_impl_json_int(const std::string& s, const char* key,
                                size_t from, size_t end, int* out)
{
    const std::string pat = std::string("\"") + key + "\"";
    size_t p = s.find(pat, from);
    if (p == std::string::npos || p >= end) return false;
    p = s.find(':', p + pat.size());
    if (p == std::string::npos || p >= end) return false;
    p++;
    while (p < end && std::isspace((unsigned char)s[p])) p++;
    char* e = nullptr;
    const long v = std::strtol(s.c_str() + p, &e, 10);
    if (e == s.c_str() + p) return false;
    *out = (int)v;
    return true;
}

inline bool layer_impl_json_str(const std::string& s, const char* key,
                                size_t from, size_t end, std::string* out)
{
    const std::string pat = std::string("\"") + key + "\"";
    size_t p = s.find(pat, from);
    if (p == std::string::npos || p >= end) return false;
    p = s.find(':', p + pat.size());
    if (p == std::string::npos || p >= end) return false;
    p = s.find('"', p + 1);
    if (p == std::string::npos || p >= end) return false;
    const size_t q = s.find('"', p + 1);
    if (q == std::string::npos || q >= end) return false;
    *out = s.substr(p + 1, q - p - 1);
    return true;
}

inline bool layer_impl_parse_json(const std::string& txt,
                                  std::vector<LayerImplEntry>* out, std::string* err)
{
    const size_t lp = txt.find("\"layers\"");
    if (lp == std::string::npos) { if (err) *err = "JSON 缺少 \"layers\""; return false; }
    const size_t lb = txt.find('[', lp);
    if (lb == std::string::npos) { if (err) *err = "\"layers\" 后缺少 '['"; return false; }

    size_t i = lb + 1;
    for (;;)
    {
        while (i < txt.size() &&
               (std::isspace((unsigned char)txt[i]) || txt[i] == ',')) i++;
        if (i >= txt.size()) { if (err) *err = "layers 数组未闭合"; return false; }
        if (txt[i] == ']') break;
        if (txt[i] != '{') { if (err) *err = "layers 元素不是对象"; return false; }
        // 层对象是扁平的：直接找闭合 '}'（生成器不写嵌套对象）
        const size_t e = txt.find('}', i);
        if (e == std::string::npos) { if (err) *err = "层对象未闭合"; return false; }

        LayerImplEntry ent;
        std::string impl_s;
        bool ok = layer_impl_json_int(txt, "layer", i, e, &ent.layer) &&
                  layer_impl_json_int(txt, "cin", i, e, &ent.cin) &&
                  layer_impl_json_int(txt, "cout", i, e, &ent.cout) &&
                  layer_impl_json_int(txt, "kh", i, e, &ent.kh) &&
                  layer_impl_json_int(txt, "kw", i, e, &ent.kw) &&
                  layer_impl_json_int(txt, "sh", i, e, &ent.sh) &&
                  layer_impl_json_int(txt, "sw", i, e, &ent.sw) &&
                  layer_impl_json_int(txt, "ph", i, e, &ent.ph) &&
                  layer_impl_json_int(txt, "pw", i, e, &ent.pw) &&
                  layer_impl_json_int(txt, "h", i, e, &ent.h) &&
                  layer_impl_json_int(txt, "w", i, e, &ent.w) &&
                  layer_impl_json_str(txt, "impl", i, e, &impl_s);
        if (!ok || !layer_impl_parse_name(impl_s, &ent.impl))
        {
            if (err) *err = "层对象字段缺失或 impl 非法（需 cublas/implicit）";
            return false;
        }
        out->push_back(ent);
        i = e + 1;
    }
    if (out->empty()) { if (err) *err = "\"layers\" 为空"; return false; }
    return true;
}

// ------------------------------------------------------------------ 表
class LayerImplTable
{
public:
    enum class Mode { Table, AllImplicit, AllCublas };

    bool init(size_t n_layers, std::string* err = nullptr)
    {
        mode_ = Mode::Table;
        entries_.clear();
        source_ = "embedded";
        use_global_env_ = false;

        // 最高优先级：全局 A/B 开关（保留历史语义）
        const char* ig = std::getenv("USE_IMPLICIT_GEMM");
        if (ig && std::strcmp(ig, "1") == 0) { mode_ = Mode::AllImplicit; use_global_env_ = true; }
        else if (ig && std::strcmp(ig, "0") == 0) { mode_ = Mode::AllCublas; use_global_env_ = true; }

        if (!use_global_env_)
        {
            const char* jp = std::getenv("LAYER_IMPL_JSON");
            if (jp && *jp && std::strcmp(jp, "auto") != 0)
            {
                std::ifstream f(jp);
                if (!f) return fail(err, std::string("打不开 LAYER_IMPL_JSON=") + jp);
                std::string txt((std::istreambuf_iterator<char>(f)),
                                std::istreambuf_iterator<char>());
                if (!layer_impl_parse_json(txt, &entries_, err)) return false;
                source_ = std::string("json:") + jp;
            }
            else
            {
                entries_.assign(kDefaultLayerImpl, kDefaultLayerImpl + kDefaultLayerImplCount);
            }

            const char* spec = std::getenv("LAYER_IMPL");
            if (spec && *spec)
            {
                if (!apply_spec(spec, err)) return false;
            }
        }

        if (!validate(n_layers, err)) return false;
        return true;
    }

    // 运行期选择：层号 + 真实形状。表里形状对不上 → 启发式 + how 说明。
    LayerImpl select(int li, int cin, int cout, int kh, int kw, int sh, int sw,
                     int ph, int pw, int h, int w, const char** how) const
    {
        if (mode_ == Mode::AllImplicit) { *how = "all-implicit"; return LayerImpl::Implicit; }
        if (mode_ == Mode::AllCublas) { *how = "all-cublas"; return LayerImpl::Cublas; }
        const LayerImplEntry* e = find(li);
        if (e)
        {
            if (e->any_shape ||
                (e->cin == cin && e->cout == cout && e->kh == kh && e->kw == kw &&
                 e->sh == sh && e->sw == sw && e->ph == ph && e->pw == pw &&
                 e->h == h && e->w == w))
            {
                *how = "table";
                return e->impl;
            }
            *how = "shape-mismatch";
            return layer_impl_heuristic(cin, cout, kh, kw, sh, sw, h, w);
        }
        *how = "no-entry";
        return layer_impl_heuristic(cin, cout, kh, kw, sh, sw, h, w);
    }

    // 构建期（没有 H/W）：只用于覆盖率严格检查与汇总。
    LayerImpl select_static(int li) const
    {
        if (mode_ == Mode::AllImplicit) return LayerImpl::Implicit;
        if (mode_ == Mode::AllCublas) return LayerImpl::Cublas;
        const LayerImplEntry* e = find(li);
        return e ? e->impl : LayerImpl::Cublas;
    }

    bool all_implicit() const
    {
        if (mode_ == Mode::AllImplicit) return true;
        if (mode_ == Mode::AllCublas) return false;
        return false;
    }

    int count(LayerImpl v) const
    {
        int c = 0;
        for (const auto& e : entries_) c += (e.impl == v);
        return c;
    }

    size_t entry_count() const { return entries_.size(); }
    Mode mode() const { return mode_; }
    const std::string& source() const { return source_; }
    const std::vector<LayerImplEntry>& entries() const { return entries_; }

    void print_summary(size_t n_layers) const
    {
        if (mode_ == Mode::AllImplicit)
            printf("[impl] 全局开关 USE_IMPLICIT_GEMM=1：%zu 层全部 implicit（A/B 对照）\n",
                   n_layers);
        else if (mode_ == Mode::AllCublas)
            printf("[impl] 全局开关 USE_IMPLICIT_GEMM=0：%zu 层全部 im2col+cublas（A/B 对照）\n",
                   n_layers);
        else
        {
            printf("[impl] 按层混合：表来源 %s，%d implicit / %d cublas（表内 %zu 条 / 共 %zu 层）\n",
                   source_.c_str(), count(LayerImpl::Implicit), count(LayerImpl::Cublas),
                   entries_.size(), n_layers);
            printf("[impl] implicit 层：");
            for (const auto& e : entries_)
                if (e.impl == LayerImpl::Implicit) printf("%d ", e.layer);
            printf("\n");
            if (entries_.size() < n_layers)
                printf("[impl] 注意：%zu 层不在表内 → 走形状启发式（默认 cublas）\n",
                       n_layers - entries_.size());
        }
    }

private:
    static bool fail(std::string* err, const std::string& msg)
    {
        if (err) *err = msg;
        return false;
    }

    const LayerImplEntry* find(int li) const
    {
        for (const auto& e : entries_) if (e.layer == li) return &e;
        return nullptr;
    }

    bool apply_spec(const std::string& spec, std::string* err)
    {
        std::string s;
        s.reserve(spec.size());
        for (char ch : spec) s.push_back((char)std::tolower((unsigned char)ch));

        size_t pos = 0;
        while (pos <= s.size())
        {
            const size_t comma = s.find(',', pos);
            const size_t end = (comma == std::string::npos) ? s.size() : comma;
            std::string tok = s.substr(pos, end - pos);
            while (!tok.empty() && std::isspace((unsigned char)tok.front())) tok.erase(tok.begin());
            while (!tok.empty() && std::isspace((unsigned char)tok.back())) tok.pop_back();
            if (tok.empty()) { pos = end + 1; if (comma == std::string::npos) break; continue; }

            if (tok == "auto")
            {
                entries_.assign(kDefaultLayerImpl, kDefaultLayerImpl + kDefaultLayerImplCount);
                source_ = "embedded(auto)";
            }
            else if (tok.rfind("all:", 0) == 0)
            {
                LayerImpl v;
                if (!layer_impl_parse_name(tok.substr(4), &v))
                    return fail(err, "LAYER_IMPL 无法解析: " + tok);
                mode_ = (v == LayerImpl::Implicit) ? Mode::AllImplicit : Mode::AllCublas;
            }
            else
            {
                const size_t colon = tok.find(':');
                if (colon == std::string::npos || colon == 0)
                    return fail(err, "LAYER_IMPL 无法解析: " + tok);
                char* e = nullptr;
                const long li = std::strtol(tok.c_str(), &e, 10);
                if (e != tok.c_str() + colon)
                    return fail(err, "LAYER_IMPL 层号非法: " + tok);
                LayerImpl v;
                if (!layer_impl_parse_name(tok.substr(colon + 1), &v))
                    return fail(err, "LAYER_IMPL 实现名非法: " + tok);
                LayerImplEntry* hit = nullptr;
                for (auto& ent : entries_) if (ent.layer == (int)li) { hit = &ent; break; }
                if (hit) { hit->impl = v; hit->any_shape = false; }
                else
                {
                    LayerImplEntry ent;
                    ent.layer = (int)li;
                    ent.impl = v;
                    ent.any_shape = true;
                    entries_.push_back(ent);
                }
                source_ += "+env";
            }
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
        return true;
    }

    bool validate(size_t n_layers, std::string* err)
    {
        for (const auto& e : entries_)
        {
            if (e.layer < 0 || (size_t)e.layer >= n_layers)
                return fail(err, "层号越界: " + std::to_string(e.layer) +
                                      "（模型只有 " + std::to_string(n_layers) + " 层）");
            if (!e.any_shape)
            {
                if (e.cin <= 0 || e.cout <= 0 || e.kh <= 0 || e.kw <= 0 || e.h <= 0 || e.w <= 0)
                    return fail(err, "层 " + std::to_string(e.layer) +
                                          " 缺少形状字段（cin/cout/kh/kw/h/w 必须 >0）");
            }
        }
        return true;
    }

    Mode mode_ = Mode::Table;
    bool use_global_env_ = false;
    std::string source_ = "embedded";
    std::vector<LayerImplEntry> entries_;
};

// main.cpp 在建引擎之前就要知道显存上限：混合模式仍可能落 im2col，
// 所以只有「全部 implicit」才放宽到 2048；其余保持 512 的旧口径。
inline int layer_impl_batch_cap()
{
    LayerImplTable t;
    std::string err;
    // 层号校验用一个足够大的上界：这里只关心「是否全部 implicit」。
    if (!t.init(1u << 20, &err)) return 512;
    return t.all_implicit() ? 2048 : 512;
}

} // namespace ptqint8

// 实测生成的默认表（bench_layers --select --emit-header）。
#include "layer_impl_default.h"