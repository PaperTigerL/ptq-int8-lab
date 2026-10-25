// ptq_model.h 的 CPU 单测（不需要 GPU / CUDA / torch）。
//
// 覆盖：
//   1. PTQW / PTQ2 按 magic 分派、EOF 严格校验
//   2. groups 语义：cin 是每组通道，总输入 = cin*groups；depthwise = cin=1/groups=cout
//   3. relu=2 / residual=2 解析
//   4. 拓扑连通性：通道/尺寸/残差形状/fc
//   5. PTQC 严格校验：8+8n、n==n_conv、有限、lo<=hi
//   6. 激活 scale：默认 AbsMax（与 09 校准一致），MinMax 显式对照，R18 第 0 层比值 1.9356
//   7. 真实产物：R18(PTQW) / R50(PTQ2) / 52-conv MV2(PTQ2) / 35-conv MV2(PTQ2)
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

#include "ptq_model.h"
#include "ptq_quant.h"

static int g_fail = 0;
static int g_pass = 0;

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (cond) { g_pass++; }                                                 \
        else { g_fail++; fprintf(stderr, "  [FAIL] %s:%d  %s\n", __FILE__,       \
                                 __LINE__, msg); }                              \
    } while (0)

static void check_ok(bool ok, const std::string& err, const char* what)
{
    if (!ok) { g_fail++; fprintf(stderr, "  [FAIL] %s: %s\n", what, err.c_str()); }
    else g_pass++;
}

static void check_fail(bool ok, const std::string& err, const char* needle,
                       const char* what)
{
    if (ok) { g_fail++; fprintf(stderr, "  [FAIL] %s: 期望失败但成功\n", what); return; }
    if (err.find(needle) == std::string::npos)
    {
        g_fail++;
        fprintf(stderr, "  [FAIL] %s: 错误信息不含 \"%s\"，实为: %s\n",
                what, needle, err.c_str());
        return;
    }
    g_pass++;
}

// ---------------------------------------------------------------- 写夹具

static void put_i32(std::ofstream& f, int v) { f.write((const char*)&v, 4); }
static void put_f32(std::ofstream& f, float v) { f.write((const char*)&v, 4); }
static void put_magic(std::ofstream& f, const char* m) { f.write(m, 4); }

struct C
{
    int cout, cin, kh, kw, sh, sw, ph, pw, relu, groups;
};

static void put_conv(std::ofstream& f, const C& c, bool with_groups = true)
{
    const int meta[10] = {c.cout, c.cin, c.kh, c.kw, c.sh,
                          c.sw, c.ph, c.pw, c.relu, c.groups};
    const int nmeta = with_groups ? 10 : 9;
    for (int i = 0; i < nmeta; i++) put_i32(f, meta[i]);
    const size_t nw = (size_t)c.cout * c.cin * c.kh * c.kw;
    for (size_t i = 0; i < nw; i++)
    {
        // 确定性权重：被 cout/cin 分开，避免全零触发 scale 下限
        put_f32(f, 0.01f * (float)((i % 17) + 1) * ((i & 1) ? -1.0f : 1.0f));
    }
    for (int i = 0; i < c.cout; i++) put_f32(f, 0.05f * (float)((i % 5) + 1));
}

struct B { std::vector<int> convs; int ds, stride, residual; };

static void put_block(std::ofstream& f, const B& b, int ncp)
{
    for (int i = 0; i < ncp; i++)
        put_i32(f, i < (int)b.convs.size() ? b.convs[i] : -1);
    put_i32(f, b.ds);
    put_i32(f, b.stride);
    put_i32(f, b.residual);
}

// PTQW 的 block 是 4 个 int32（c1,c2,ds,stride），没有 residual/ncp 补位
static void put_block_ptqw(std::ofstream& f, const B& b)
{
    put_i32(f, b.convs[0]);
    put_i32(f, b.convs[1]);
    put_i32(f, b.ds);
    put_i32(f, b.stride);
}

static const char* kDir = ".";

static std::string p(const std::string& name) { return std::string(kDir) + "/" + name; }

// 小 PTQW：5 conv / 2 block / 4→fc10
static std::string write_ptqw(const std::string& path)
{
    std::ofstream f(path, std::ios::binary);
    put_magic(f, "PTQW");
    put_i32(f, 5); put_i32(f, 2); put_i32(f, 0);
    put_conv(f, {4, 3, 3, 3, 1, 1, 1, 1, 1, 1}, false);   // stem
    put_conv(f, {4, 4, 3, 3, 1, 1, 1, 1, 1, 1}, false);   // b0.c1
    put_conv(f, {4, 4, 3, 3, 1, 1, 1, 1, 0, 1}, false);   // b0.c2
    put_conv(f, {4, 4, 3, 3, 1, 1, 1, 1, 1, 1}, false);   // b1.c1
    put_conv(f, {4, 4, 3, 3, 1, 1, 1, 1, 0, 1}, false);   // b1.c2
    put_block_ptqw(f, {{1, 2}, -1, 1, 1});
    put_block_ptqw(f, {{3, 4}, -1, 1, 1});
    put_i32(f, 4); put_i32(f, 10);
    for (int i = 0; i < 40; i++) put_f32(f, 0.02f * (float)(i % 7));
    for (int i = 0; i < 10; i++) put_f32(f, 0.01f * i);
    f.close();
    return path;
}

// 小 PTQ2：8 conv / 3 block / ncp=3，含 depthwise + relu6 + residual2
//        block2 无残差（纯串联），末 conv 输出 8 通道 -> fc 8→10
static std::string write_ptq2(const std::string& path, bool bad_pad = false,
                              bool bad_ds = false, bool bad_cin = false)
{
    std::ofstream f(path, std::ios::binary);
    put_magic(f, "PTQ2");
    put_i32(f, 8); put_i32(f, 3); put_i32(f, 0); put_i32(f, 3);
    put_conv(f, {4, 3, 3, 3, 1, 1, 1, 1, 1, 1});    // 0 stem
    put_conv(f, {4, 4, 3, 3, 1, 1, 1, 1, 1, 1});    // 1 b0.c1
    put_conv(f, {4, 4, 3, 3, 1, 1, 1, 1, 0, 1});    // 2 b0.c2
    put_conv(f, {4, bad_cin ? 3 : 4, 3, 3, 1, 1, 1, 1, 1, 1}); // 3 b1.c1
    put_conv(f, {4, 4, 3, 3, 1, 1, 1, 1, 0, 1});    // 4 b1.c2
    put_conv(f, {4, 1, 3, 3, 1, 1, 1, 1, 2, 4});    // 5 depthwise
    put_conv(f, {8, 4, 1, 1, 1, 1, 0, 0, 1, 1});    // 6
    put_conv(f, {8, 8, 1, 1, 1, 1, 0, 0, 1, 1});    // 7
    B b0;
    b0.convs = bad_pad ? std::vector<int>{1, -1, 2} : std::vector<int>{1, 2, -1};
    b0.ds = -1; b0.stride = 1; b0.residual = 1;
    put_block(f, b0, 3);
    put_block(f, {{3, 4, -1}, -1, 1, 2}, 3);
    put_block(f, {{5, 6, 7}, bad_ds ? 0 : -1, 1, 0}, 3);
    put_i32(f, 8); put_i32(f, 10);
    for (int i = 0; i < 80; i++) put_f32(f, 0.02f * (float)((i % 11) - 5));
    for (int i = 0; i < 10; i++) put_f32(f, 0.01f * i);
    f.close();
    return path;
}

static std::string read_all(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)),
                       std::istreambuf_iterator<char>());
}

static void write_all(const std::string& path, const std::string& data)
{
    std::ofstream f(path, std::ios::binary);
    f.write(data.data(), (std::streamsize)data.size());
}

static std::string derr;

static ptqmodel::Model load_or_die(const std::string& path)
{
    ptqmodel::Model m;
    if (!m.load(path, &derr))
    {
        fprintf(stderr, "  [FATAL] 加载失败 %s: %s\n", path.c_str(), derr.c_str());
        g_fail++;
    }
    return m;
}

int main()
{
    printf("== ptq_model 单测 ==\n");

    // ---------------- [1] PTQW 分派 ----------------
    printf("[1] PTQW 分派/字段/拓扑\n");
    {
        const std::string f = write_ptqw(p("_t_model.ptqw"));
        ptqmodel::Model m = load_or_die(f);
        CHECK(m.convs.size() == 5, "PTQW n_conv 应为 5");
        CHECK(m.blocks.size() == 2, "PTQW n_block 应为 2");
        CHECK(m.stem_pool == 0, "PTQW stem_pool 应为 0");
        if (m.blocks.size() == 2 && m.convs.size() == 5)
        {
            CHECK(m.blocks[0].residual == 1, "PTQW block residual 应为 1");
            CHECK(m.convs[0].groups == 1 && m.convs[0].in_channels() == 3,
                  "PTQW 首层输入应为 3");
            CHECK(m.fc_in == 4 && m.fc_out == 10, "PTQW fc 应为 4→10");
        }
        std::remove(f.c_str());
    }

    // ---------------- [2] PTQ2 groups/relu6/residual2 ----------------
    printf("[2] PTQ2 groups/ReLU6/residual2\n");
    std::string f2 = write_ptq2(p("_t_model.ptq2"));
    {
        ptqmodel::Model m = load_or_die(f2);
        CHECK(m.convs.size() == 8 && m.blocks.size() == 3, "PTQ2 规模应为 8 conv/3 block");
        if (m.convs.size() == 8 && m.blocks.size() == 3)
        {
            const ptqmodel::ConvSpec& dw = m.convs[5];
            CHECK(dw.cin == 1 && dw.groups == 4 && dw.cout == 4,
                  "depthwise 应为 cin=1, groups=4, cout=4");
            CHECK(dw.in_channels() == 4, "depthwise 总输入 = cin*groups = 4");
            CHECK(dw.relu == 2, "depthwise relu 应为 2（ReLU6）");
            CHECK(m.n_relu6() == 1, "relu6 计数应为 1");
            CHECK(m.n_residual2() == 1, "residual2 计数应为 1");
            CHECK(m.n_depthwise() == 1, "depthwise 计数应为 1");
            CHECK(m.blocks[0].convs.size() == 3 && m.blocks[0].convs[2] == -1,
                  "block0 第三个 conv 应为 -1 补位");
            CHECK(m.blocks[1].residual == 2, "block1 residual 应为 2");
            CHECK(m.blocks[2].residual == 0 && m.blocks[2].ds == -1,
                  "block2 应为无残差、ds=-1");
            CHECK(m.fc_in == 8 && m.fc_out == 10, "PTQ2 fc 应为 8→10");
        }
    }

    // ---------------- [3] PTQ2 负例 ----------------
    printf("[3] PTQ2 负例\n");
    {
        std::string f = write_ptq2(p("_bad_pad.ptq2"), true, false, false);
        ptqmodel::Model m; std::string e;
        check_fail(m.load(f, &e), e, "-1 补位必须集中在尾部", "中间补位");
        std::remove(f.c_str());
    }
    {
        std::string f = write_ptq2(p("_bad_ds.ptq2"), false, true, false);
        ptqmodel::Model m; std::string e;
        check_fail(m.load(f, &e), e, "residual=0", "无残差却给 ds");
        std::remove(f.c_str());
    }
    {
        std::string f = write_ptq2(p("_bad_cin.ptq2"), false, false, true);
        ptqmodel::Model m; std::string e;
        check_fail(m.load(f, &e), e, "输入通道", "通道不连通");
        std::remove(f.c_str());
    }
    {
        std::string data = read_all(f2);
        write_all(p("_trunc.ptq2"), data.substr(0, data.size() - 1));
        ptqmodel::Model m; std::string e;
        check_fail(m.load(p("_trunc.ptq2"), &e), e, "文件被截断", "PTQ2 截断");
        std::remove(p("_trunc.ptq2").c_str());
    }
    {
        std::string data = read_all(f2) + "X";
        write_all(p("_extra.ptq2"), data);
        ptqmodel::Model m; std::string e;
        check_fail(m.load(p("_extra.ptq2"), &e), e, "尾部多余", "PTQ2 尾部多余");
        std::remove(p("_extra.ptq2").c_str());
    }
    {
        std::string data = read_all(f2);
        // conv0 的 relu 在 offset 4(magic)+16(header)+8*4=52
        data[52] = 9;
        write_all(p("_bad_relu.ptq2"), data);
        ptqmodel::Model m; std::string e;
        check_fail(m.load(p("_bad_relu.ptq2"), &e), e, "relu=9", "非法 relu");
        std::remove(p("_bad_relu.ptq2").c_str());
    }
    {
        std::string data = read_all(f2);
        // conv0 的 groups 在 offset 4+16+9*4=56
        data[56] = 0;
        write_all(p("_bad_groups.ptq2"), data);
        ptqmodel::Model m; std::string e;
        check_fail(m.load(p("_bad_groups.ptq2"), &e), e, "groups=0", "非法 groups");
        std::remove(p("_bad_groups.ptq2").c_str());
    }
    {
        std::string data = read_all(f2);
        data[0] = 'X';
        write_all(p("_bad_magic.ptq2"), data);
        ptqmodel::Model m; std::string e;
        check_fail(m.load(p("_bad_magic.ptq2"), &e), e, "magic", "坏 magic");
        std::remove(p("_bad_magic.ptq2").c_str());
    }
    std::remove(f2.c_str());

    // ---------------- [4] PTQC 严格校验 ----------------
    printf("[4] PTQC 严格校验\n");
    {
        const std::string calib = "../../09-ptq-toolkit/results/calib_resnet18_folded_w8a8.ptqc";
        std::vector<std::pair<float, float>> r;
        std::string e;
        check_ok(ptqmodel::load_ptqc(calib, 20, &r, &e), e, "PTQC 168B n=20");
        CHECK(r.size() == 20, "PTQC 应读出 20 对");
        if (r.size() == 20)
        {
            // 与 06 的 calib_minmax.bin 第 0 层一致
            CHECK(std::fabs(r[0].first - (-1.9894737f)) < 1e-4f,
                  "第 0 层 lo 应为 -1.98947");
            CHECK(std::fabs(r[0].second - 2.1264887f) < 1e-4f,
                  "第 0 层 hi 应为 2.12649");
        }

        std::string data = read_all(calib);
        CHECK(data.size() == 168, "PTQC 样例应为 168 B");

        write_all(p("_c_trunc.ptqc"), data.substr(0, 160));
        r.clear(); e.clear();
        check_fail(ptqmodel::load_ptqc(p("_c_trunc.ptqc"), 20, &r, &e), e,
                   "长度不符", "PTQC 截断");
        std::remove(p("_c_trunc.ptqc").c_str());

        write_all(p("_c_extra.ptqc"), data + "X");
        r.clear(); e.clear();
        check_fail(ptqmodel::load_ptqc(p("_c_extra.ptqc"), 20, &r, &e), e,
                   "长度不符", "PTQC 尾部多余");
        std::remove(p("_c_extra.ptqc").c_str());

        // n=19 + 160 B：长度自洽但层数不匹配
        {
            std::string d = data.substr(0, 160);
            int32_t n19 = 19;
            memcpy(&d[4], &n19, 4);
            write_all(p("_c_n19.ptqc"), d);
            r.clear(); e.clear();
            check_fail(ptqmodel::load_ptqc(p("_c_n19.ptqc"), 20, &r, &e), e,
                       "层数不符", "PTQC n != n_conv");
            std::remove(p("_c_n19.ptqc").c_str());
        }

        {
            std::string d = data;
            d[0] = 'X';
            write_all(p("_c_magic.ptqc"), d);
            r.clear(); e.clear();
            check_fail(ptqmodel::load_ptqc(p("_c_magic.ptqc"), 20, &r, &e), e,
                       "magic", "PTQC 坏 magic");
            std::remove(p("_c_magic.ptqc").c_str());
        }

        {
            std::string d = data;
            float nan = std::numeric_limits<float>::quiet_NaN();
            memcpy(&d[8], &nan, 4);
            write_all(p("_c_nan.ptqc"), d);
            r.clear(); e.clear();
            check_fail(ptqmodel::load_ptqc(p("_c_nan.ptqc"), 20, &r, &e), e,
                       "非有限", "PTQC NaN");
            std::remove(p("_c_nan.ptqc").c_str());
        }

        {
            std::string d = data;
            float lo = 1.0f, hi = -1.0f;
            memcpy(&d[8], &lo, 4);
            memcpy(&d[12], &hi, 4);
            write_all(p("_c_lohi.ptqc"), d);
            r.clear(); e.clear();
            check_fail(ptqmodel::load_ptqc(p("_c_lohi.ptqc"), 20, &r, &e), e,
                       "lo=", "PTQC lo>hi");
            std::remove(p("_c_lohi.ptqc").c_str());
        }
    }

    // ---------------- [5] 激活 scale 口径 ----------------
    printf("[5] 激活 scale: 默认 AbsMax / MinMax 显式对照\n");
    {
        // 真实 R18 第 0 层范围（PTQC 首层）：验证默认口径与 1.9356 比值。
        const float lo0 = -1.9894737f, hi0 = 2.1264887f;
        ptq::TensorQuant def0 = ptq::tensor_quant_from_range(lo0, hi0, 8);
        ptq::TensorQuant mm0 = ptq::tensor_quant_from_range(
            lo0, hi0, 8, ptq::ActRangeMode::MinMax);
        ptq::TensorQuant am0 = ptq::tensor_quant_from_range(
            lo0, hi0, 8, ptq::ActRangeMode::AbsMax);
        CHECK(def0.scale == am0.scale, "默认模式必须等于 AbsMax");
        const double ratio = (double)mm0.scale / (double)am0.scale;
        CHECK(std::fabs(ratio - 1.935567) < 1e-4,
              "R18 第 0 层 MinMax/AbsMax 应为 1.9356");
        CHECK(std::fabs((double)am0.scale - 2.1264887 / 127.0) < 1e-8,
              "AbsMax scale 应为 max(|lo|,|hi|)/127");
        CHECK(std::fabs((double)mm0.scale - 4.1159624 / 127.0) < 1e-8,
              "MinMax scale 应为 (hi-lo)/127");

        // 交接单指定的跨零小范围：lo=-0.0653, hi=0.0930。
        const float lo = -0.0653f, hi = 0.0930f;
        ptq::TensorQuant def = ptq::tensor_quant_from_range(lo, hi, 8);
        ptq::TensorQuant am = ptq::tensor_quant_from_range(
            lo, hi, 8, ptq::ActRangeMode::AbsMax);
        ptq::TensorQuant mm = ptq::tensor_quant_from_range(
            lo, hi, 8, ptq::ActRangeMode::MinMax);
        CHECK(def.scale == am.scale, "跨零范围默认应为 AbsMax");
        CHECK(std::fabs((double)am.scale - 0.0930 / 127.0) < 1e-9,
              "lo=-0.0653/hi=0.0930 时 scale 应为 absmax/127");
        CHECK(std::fabs((double)mm.scale - 0.1583 / 127.0) < 1e-9,
              "MinMax 应走旧公式 (hi-lo)/127");

        // 量化 + 反量化无裁剪：全区间采样，码字必须落在 int8 对称范围，
        // 反量化误差不超过半个 scale。
        for (int mode = 0; mode < 2; mode++)
        {
            const ptq::ActRangeMode md = (mode == 0)
                ? ptq::ActRangeMode::AbsMax : ptq::ActRangeMode::MinMax;
            const ptq::TensorQuant t = ptq::tensor_quant_from_range(lo, hi, 8, md);
            int qmin = 127, qmax = -128, clipped = 0;
            double worst = 0.0;
            for (int i = 0; i <= 1000; i++)
            {
                const double x = (double)lo + ((double)hi - (double)lo) * i / 1000.0;
                long q = (long)std::lround(x / (double)t.scale);
                if (q < -128 || q > 127) { clipped++; q = q < -128 ? -128 : 127; }
                if (q < qmin) qmin = (int)q;
                if (q > qmax) qmax = (int)q;
                const double err = std::fabs((double)q * (double)t.scale - x);
                if (err > worst) worst = err;
            }
            CHECK(clipped == 0, "量化码字不应裁剪（AbsMax/MinMax）");
            CHECK(qmin >= -127 && qmax <= 127, "码字应落在 [-127,127]");
            CHECK(worst <= 0.5 * (double)t.scale + 1e-9,
                  "反量化误差应 <= 0.5*scale");
        }

        ptq::TensorQuant z1 = ptq::tensor_quant_from_range(0.0f, 2.0f, 8);
        ptq::TensorQuant z2 = ptq::tensor_quant_from_range(
            0.0f, 2.0f, 8, ptq::ActRangeMode::MinMax);
        CHECK(std::fabs(z1.scale - z2.scale) < 1e-9f, "lo=0 时两种口径应相同");
    }

    // ---------------- [6] 真实产物 ----------------
    printf("[6] 真实产物加载\n");
    const std::string R = "../../09-ptq-toolkit/results/";
    {
        ptqmodel::Model m = load_or_die(R + "resnet18_folded_w8a8.bin");
        CHECK(m.convs.size() == 20, "R18 应为 20 conv");
        CHECK(m.blocks.size() == 8, "R18 应为 8 block");
        CHECK(m.stem_pool == 1, "R18 stem_pool 应为 1");
        CHECK(m.n_grouped() == 0, "R18 不应有分组 conv");
        CHECK(m.fc_in == 512 && m.fc_out == 10, "R18 fc 应为 512→10");
    }
    {
        ptqmodel::Model m = load_or_die(R + "resnet50_folded_w8a8.bin");
        CHECK(m.convs.size() == 53, "R50 应为 53 conv");
        CHECK(m.blocks.size() == 16, "R50 应为 16 block");
        CHECK(m.stem_pool == 0, "R50 无 stem pool");
        CHECK(m.n_grouped() == 0, "R50 不应有分组 conv");
        CHECK(m.n_relu6() == 0, "R50 不应有 ReLU6");
        CHECK(m.n_residual2() == 0, "R50 residual 应恒为 1");
        CHECK(m.fc_in == 2048 && m.fc_out == 10, "R50 fc 应为 2048→10");
        CHECK(m.blocks[0].convs.size() == 3, "R50 ncp 应为 3");
    }
    {
        ptqmodel::Model m = load_or_die(R + "m2_accept/mobilenet_v2_folded_w8a8.bin");
        CHECK(m.convs.size() == 52, "m2_accept 应为 52 conv");
        CHECK(m.blocks.size() == 18, "m2_accept 应为 18 block");
        CHECK(m.n_depthwise() == 17, "m2_accept 应有 17 个 depthwise");
        CHECK(m.n_grouped() == 17, "m2_accept 应有 17 个分组 conv");
        CHECK(m.n_relu6() == 35, "m2_accept 应有 35 个 ReLU6");
        CHECK(m.n_residual2() == 10, "m2_accept 应有 10 个 residual=2");
        CHECK(m.fc_in == 1280 && m.fc_out == 10, "m2_accept fc 应为 1280→10");
        for (const auto& c : m.convs)
            if (c.groups > 1)
                CHECK(c.in_channels() == c.cout, "depthwise 总输入应等于 cout");
    }
    {
        ptqmodel::Model m = load_or_die(R + "mobilenet_v2_folded_w8a8.bin");
        CHECK(m.convs.size() == 35, "旧 MV2 应为 35 conv");
        CHECK(m.blocks.size() == 17, "旧 MV2 应为 17 block");
        CHECK(m.n_depthwise() == 17, "旧 MV2 应有 17 个 depthwise");
        CHECK(m.n_grouped() == 17, "旧 MV2 应有 17 个分组 conv");
        CHECK(m.fc_in == 320 && m.fc_out == 10, "旧 MV2 fc 应为 320→10");
    }

    printf("----------------------------------------\n");
    printf("%s：通过 %d 项，失败 %d 项\n", g_fail ? "FAIL" : "ALL PASS", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
