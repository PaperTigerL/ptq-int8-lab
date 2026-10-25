// ptq2_reader 的 CPU 单测 + 真实产物 dump（不依赖 CUDA）。
//
// 用法：
//   ptq2_reader_test            → 跑合成夹具（正向+反向），全过 exit 0
//   ptq2_reader_test a.bin b…   → 用解析器只读 dump 真实 PTQ2 产物统计
//
// 编译：g++ -std=c++17 -O2 -o ptq2_reader_test ptq2_reader_test.cpp
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "ptq2_reader.h"

// ================================================================ 合成夹具

static void put_i32(std::string* b, int32_t v)
{
    b->append(reinterpret_cast<char*>(&v), 4);
}
static void put_f32(std::string* b, float v)
{
    b->append(reinterpret_cast<char*>(&v), 4);
}

struct FixConv {
    int cout, cin, kh, kw, sh, sw, ph, pw, relu, groups;
};

// 造一个完整合法 blob：convs 顺序写，blocks 每块 ncp 个索引 + ds/stride/residual。
// 权重填可辨识的 pattern：w[c][k] = 1000*i + c*10 + (k%10)，便于逐位断言。
static std::string make_blob(const std::vector<FixConv>& convs,
                             int ncp,
                             const std::vector<std::pair<std::vector<int>, std::array<int, 3>>>& blocks,
                             int fc_in, int fc_out,
                             const char* magic = "PTQ2")
{
    std::string b;
    b.append(magic, 4);
    put_i32(&b, (int32_t)convs.size());
    put_i32(&b, (int32_t)blocks.size());
    put_i32(&b, 0);   // stem_pool
    put_i32(&b, ncp);
    for (size_t i = 0; i < convs.size(); i++) {
        const FixConv& c = convs[i];
        for (int v : {c.cout, c.cin, c.kh, c.kw, c.sh, c.sw, c.ph, c.pw, c.relu, c.groups})
            put_i32(&b, v);
        size_t wn = (size_t)c.cout * c.cin * c.kh * c.kw;
        for (size_t k = 0; k < wn; k++)
            put_f32(&b, 1000.0f * i + (float)(k % 10));
        for (int k = 0; k < c.cout; k++)
            put_f32(&b, -(float)k);
    }
    for (const auto& blk : blocks) {
        const auto& idx = blk.first;
        for (int k = 0; k < ncp; k++)
            put_i32(&b, k < (int)idx.size() ? idx[k] : -1);
        put_i32(&b, blk.second[0]);  // ds
        put_i32(&b, blk.second[1]);  // stride
        put_i32(&b, blk.second[2]);  // residual
    }
    put_i32(&b, fc_in);
    put_i32(&b, fc_out);
    for (size_t k = 0; k < (size_t)fc_in * fc_out; k++)
        put_f32(&b, 0.5f + (float)(k % 7));
    for (int k = 0; k < fc_out; k++)
        put_f32(&b, (float)k);
    return b;
}

static bool write_file(const std::string& path, const std::string& bytes)
{
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    size_t w = std::fwrite(bytes.data(), 1, bytes.size(), f);
    std::fclose(f);
    return w == bytes.size();
}

// ================================================================ 断言框架

static int g_fail = 0;
#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::printf("  FAIL: %s  (%s:%d)\n", msg, __FILE__, __LINE__);      \
            g_fail++;                                                           \
        }                                                                       \
    } while (0)
#define CHECK_ERR(haystack, needle, msg)                                        \
    do {                                                                        \
        if ((haystack).find(needle) == std::string::npos) {                     \
            std::printf("  FAIL: %s —— err=\"%s\" 不含 \"%s\"\n",               \
                        msg, (haystack).c_str(), needle);                       \
            g_fail++;                                                           \
        }                                                                       \
    } while (0)

static const std::vector<FixConv> kConvs = {
    {3, 3, 3, 3, 1, 1, 1, 1, 2, 1},      // 0 stem: relu6
    {8, 1, 3, 3, 1, 1, 1, 1, 2, 8},      // 1 depthwise: groups=cout, cin=1（真实 PTQ2 口径）
    {4, 8, 1, 1, 1, 1, 0, 0, 0, 1},      // 2 project: relu=0
    {8, 4, 1, 1, 2, 2, 0, 0, 1, 2},      // 3 ds 1×1 s2, 分组 conv（cin=4/组, groups=2）
    {8, 1, 3, 3, 1, 1, 1, 1, 2, 8},      // 4 第二个 depthwise
};

// 正向用例 1：ncp=2 + depthwise(groups=cin) + relu=2 + residual=2 + ds=-1
static void test_ncp2_dw_relu6_res2_nods()
{
    std::printf("[1] ncp=2 / depthwise(cin=1,groups=cout) / relu=2 / residual=2 / ds=-1\n");
    auto blob = make_blob(kConvs, 2,
                          {{{0, 1}, {-1, 1, 2}},        // block0: 无 ds、相加后无激活
                           {{2, 4}, {3, 2, 1}},         // block1: 有 ds、相加后 ReLU
                           {{2, 4}, {-1, 1, 0}}},        // block2: 纯串联
                          32, 10);
    std::string path = "ptq2_fix_ncp2.bin";
    CHECK(write_file(path, blob), "写夹具失败");
    ptq2::Ptq2Model m;
    std::string err;
    bool ok = ptq2::load_ptq2(path, &m, &err);
    CHECK(ok, "正向解析应成功");
    if (ok) {
        CHECK(m.n_conv == 5 && m.n_block == 3 && m.stem_pool == 0 && m.n_conv_per_block == 2,
              "header 字段");
        CHECK(m.convs[1].groups == 8 && m.convs[1].cin == 1,
              "depthwise groups=cout、cin=1/组（groups>cin 必须合法）");
        CHECK(m.convs[1].weight.size() == 8 * 1 * 3 * 3, "depthwise 权重元素数");
        CHECK(m.convs[3].groups == 2 && m.convs[3].cin == 4, "分组 conv（非 depthwise）保留");
        CHECK(m.convs[0].relu == 2 && m.convs[2].relu == 0, "relu=2/0 保留");
        CHECK(m.blocks[0].residual == 2 && m.blocks[0].ds == -1, "residual=2/ds=-1");
        CHECK(m.blocks[1].ds == 3 && m.blocks[1].stride == 2 && m.blocks[1].residual == 1, "ds/stride/residual");
        CHECK(m.blocks[2].residual == 0, "residual=0");
        CHECK(m.convs[0].weight.size() == 3 * 3 * 3 * 3, "conv0 权重元素数");
        CHECK(m.convs[0].weight[11] == 1.0f, "conv0 权重 pattern 逐位");
        CHECK(m.convs[4].weight[4] == 4004.0f, "conv4 权重 pattern 逐位");
        CHECK(m.convs[1].bias.size() == 8 && m.convs[1].bias[3] == -3.0f, "bias 长度+值");
        CHECK(m.fc_in == 32 && m.fc_out == 10, "fc 维度");
        CHECK(m.fc_w.size() == 320 && m.fc_w[0] == 0.5f && m.fc_w[7] == 0.5f, "fc 权重");
        CHECK(m.fc_b.size() == 10 && m.fc_b[9] == 9.0f, "fc bias");
        CHECK(m.blocks[0].convs.size() == 2, "block.convs 长度 = ncp");
    }
    std::remove(path.c_str());
}

// 正向用例 2：ncp=3（R50 风格，块内 2 conv 时 -1 补位）
static void test_ncp3_padded()
{
    std::printf("[2] ncp=3 / -1 补位\n");
    auto blob = make_blob(kConvs, 3,
                          {{{0, 1, 2}, {-1, 1, 1}},
                           {{3, 4, -1}, {3, 2, 2}}},
                          32, 10);
    std::string path = "ptq2_fix_ncp3.bin";
    CHECK(write_file(path, blob), "写夹具失败");
    ptq2::Ptq2Model m;
    std::string err;
    bool ok = ptq2::load_ptq2(path, &m, &err);
    CHECK(ok, "ncp=3 正向解析应成功");
    if (ok) {
        CHECK(m.n_conv_per_block == 3, "ncp=3");
        CHECK(m.blocks[0].convs[2] == 2 && m.blocks[1].convs[2] == -1, "-1 补位读回");
        CHECK(m.blocks[1].residual == 2, "residual=2");
    }
    std::remove(path.c_str());
}

static void expect_fail(const std::string& blob, const char* needle, const char* label)
{
    std::string path = std::string("ptq2_neg_") + label + ".bin";
    CHECK(write_file(path, blob), "写夹具失败");
    ptq2::Ptq2Model m;
    std::string err;
    bool ok = ptq2::load_ptq2(path, &m, &err);
    CHECK(!ok, "应解析失败");
    if (!ok) CHECK_ERR(err, needle, label);
    std::remove(path.c_str());
}

static void test_negative_cases()
{
    std::printf("[3] 反向用例\n");
    auto good = make_blob(kConvs, 2, {{{0, 1}, {-1, 1, 2}}}, 32, 10);

    expect_fail(good.substr(0, good.size() / 2), "截断", "truncated");          // 截一半
    expect_fail(good.substr(0, 18), "文件过短", "short");                        // 不足 magic+header
    expect_fail(good + "X", "尾部多余", "tail_byte");                            // 尾部多 1 字节
    expect_fail(std::string("PTQX") + good.substr(4), "magic 不符", "bad_magic");

    {   // block 索引越界（>= n_conv=5）
        auto b = good;
        size_t blocks_off = good.size() - (size_t)(32 * 10 + 10) * 4 - 8 - 1 * (2 + 3) * 4;
        int32_t bad = 5;
        memcpy(&b[blocks_off], &bad, 4);
        expect_fail(b, "越界", "idx_oob");
    }
    {   // n_conv_per_block = 0
        auto b = good;
        int32_t z = 0;
        memcpy(&b[4 + 3 * 4], &z, 4);
        expect_fail(b, "n_conv_per_block", "ncp0");
    }
    {   // ds 索引越界（>= n_conv=5）
        auto b = good;
        size_t blocks_off = good.size() - (size_t)(32 * 10 + 10) * 4 - 8 - 1 * (2 + 3) * 4;
        int32_t bad = 5;
        memcpy(&b[blocks_off + 2 * 4], &bad, 4);   // 跳过 2 个 conv 索引
        expect_fail(b, "ds=", "ds_oob");
    }
    {   // groups = 0
        auto b = good;
        size_t off = 4 + 16 + 9 * 4;         // conv0.groups
        int32_t v = 0;
        memcpy(&b[off], &v, 4);
        expect_fail(b, "groups=", "groups_zero");
    }
    {   // n_conv 为负
        auto b = good;
        int32_t neg = -1;
        memcpy(&b[4], &neg, 4);
        expect_fail(b, "n_conv 为负", "nconv_neg");
    }
    {   // relu = 7
        auto b = good;
        size_t conv0_relu = 4 + 16 + 8 * 4;  // magic+header+前 8 个 meta int
        int32_t v = 7;
        memcpy(&b[conv0_relu], &v, 4);
        expect_fail(b, "relu=", "bad_relu");
    }
    {   // groups 不整除 cout（conv0 cout=3, groups=2）
        auto b = good;
        size_t off = 4 + 16 + 9 * 4;         // conv0.groups
        int32_t v = 2;
        memcpy(&b[off], &v, 4);
        expect_fail(b, "groups=", "bad_groups");
    }
    {   // 打不开文件
        ptq2::Ptq2Model m;
        std::string err;
        CHECK(!ptq2::load_ptq2("ptq2_nonexistent_file.bin", &m, &err), "不存在的路径应失败");
        CHECK_ERR(err, "打不开", "missing_file");
    }
}

// ================================================================ 真实产物 dump

static int dump_real(const std::string& path)
{
    std::printf("\n==== %s ====\n", path.c_str());
    ptq2::Ptq2Model m;
    std::string err;
    if (!ptq2::load_ptq2(path, &m, &err)) {
        std::printf("  解析失败: %s\n", err.c_str());
        return 1;
    }
    FILE* f = std::fopen(path.c_str(), "rb");
    long fsz = (f ? (std::fseek(f, 0, SEEK_END), std::ftell(f)) : -1L);
    if (f) std::fclose(f);
    std::printf("  文件大小 %ld B；EOF 恰好读完: yes（校验通过即 pos==size）\n", fsz);
    std::printf("  n_conv=%d  n_block=%d  stem_pool=%d  n_conv_per_block=%d\n",
                m.n_conv, m.n_block, m.stem_pool, m.n_conv_per_block);
    std::printf("  fc: %d -> %d\n", m.fc_in, m.fc_out);

    std::map<int, int> groups_hist, relu_hist;
    for (const auto& c : m.convs) {
        groups_hist[c.groups]++;
        relu_hist[c.relu]++;
    }
    std::printf("  groups 分布:");
    for (auto& kv : groups_hist) std::printf(" g=%d×%d", kv.first, kv.second);
    std::printf("\n  relu 分布:   ");
    const char* rn[] = {"无激活", "ReLU", "ReLU6"};
    for (auto& kv : relu_hist) std::printf(" %d(%s)×%d", kv.first, rn[kv.first], kv.second);
    std::printf("\n  residual 分布:");
    std::map<int, int> res_hist;
    for (const auto& b : m.blocks) res_hist[b.residual]++;
    const char* sn[] = {"无残差", "相加后ReLU", "相加后无激活"};
    for (auto& kv : res_hist) std::printf(" %d(%s)×%d", kv.first, sn[kv.first], kv.second);
    std::printf("\n  ds 分布:");
    int nodes = 0;
    for (const auto& b : m.blocks) if (b.ds >= 0) nodes++;
    std::printf(" 有ds×%d / 无ds×%d\n", nodes, m.n_block - nodes);
    size_t wtot = 0;
    for (const auto& c : m.convs) wtot += c.weight.size() + c.bias.size();
    wtot += m.fc_w.size() + m.fc_b.size();
    std::printf(" 参数合计 %zu floats（header 推算应含 meta 共 %zu B）\n", wtot,
                4 + 16 + (size_t)m.n_conv * 40 + wtot * 4 + (size_t)m.n_block * (m.n_conv_per_block + 3) * 4 + 8);
    return 0;
}

int main(int argc, char** argv)
{
    if (argc > 1) {
        int rc = 0;
        for (int i = 1; i < argc; i++) rc |= dump_real(argv[i]);
        return rc;
    }
    std::printf("== ptq2_reader 单测 ==\n");
    test_ncp2_dw_relu6_res2_nods();
    test_ncp3_padded();
    test_negative_cases();
    std::printf(g_fail ? "\nFAILED: %d 项断言\n" : "\nALL PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
