#include "layer_impl.h"
#include <cassert>
#include <cstdio>
using namespace ptqint8;

static int fails = 0;
#define CHECK(x) do { if (!(x)) { printf("FAIL line %d: %s\n", __LINE__, #x); fails++; } } while (0)

int main()
{
    {
        LayerImplTable t;
        std::string e;
        CHECK(t.init(20, &e));
        CHECK(t.mode() == LayerImplTable::Mode::Table);
        CHECK(t.entry_count() == 0);
        const char* how = "";
        CHECK(t.select(1, 64, 64, 3, 3, 1, 1, 1, 1, 8, 8, &how) == LayerImpl::Implicit);
        CHECK(!strcmp(how, "no-entry"));
        CHECK(t.select(0, 3, 64, 7, 7, 2, 2, 3, 3, 32, 32, &how) == LayerImpl::Cublas);
        CHECK(t.select(17, 512, 512, 3, 3, 1, 1, 1, 1, 1, 1, &how) == LayerImpl::Implicit);
        CHECK(t.select(7, 128, 128, 3, 3, 1, 1, 1, 1, 16, 16, &how) == LayerImpl::Cublas);
    }
    {
        // 非法层号
        setenv("LAYER_IMPL", "99:c", 1);
        LayerImplTable t; std::string e;
        CHECK(!t.init(20, &e));
        unsetenv("LAYER_IMPL");
    }
    {
        setenv("LAYER_IMPL", "0:i,4:i,17:i", 1);
        LayerImplTable t; std::string e;
        CHECK(t.init(20, &e));
        const char* how = "";
        CHECK(t.select(0, 3, 64, 7, 7, 2, 2, 3, 3, 32, 32, &how) == LayerImpl::Implicit);
        CHECK(!strcmp(how, "table"));
        CHECK(t.select(17, 1, 2, 1, 1, 1, 1, 0, 0, 1, 1, &how) == LayerImpl::Implicit);  // 只按层号
        CHECK(!strcmp(how, "table"));
        CHECK(t.select(5, 64, 128, 1, 1, 2, 2, 0, 0, 16, 16, &how) == LayerImpl::Cublas);
        unsetenv("LAYER_IMPL");
    }
    {
        setenv("LAYER_IMPL", "all:implicit", 1);
        LayerImplTable t; std::string e;
        CHECK(t.init(20, &e));
        CHECK(t.all_implicit());
        const char* how = "";
        CHECK(t.select(0, 3, 64, 7, 7, 2, 2, 3, 3, 32, 32, &how) == LayerImpl::Implicit);
        unsetenv("LAYER_IMPL");
    }
    {
        setenv("USE_IMPLICIT_GEMM", "0", 1);
        LayerImplTable t; std::string e;
        CHECK(t.init(20, &e));
        CHECK(t.mode() == LayerImplTable::Mode::AllCublas);
        CHECK(t.count(LayerImpl::Implicit) == 0);
        unsetenv("USE_IMPLICIT_GEMM");
    }
    {
        // JSON 载入 + 形状校验
        FILE* f = fopen("/tmp/_li_test.json", "w");
        fprintf(f, "{\"format\":\"ptq-layer-impl/1\",\"layers\":[\n"
                   " {\"layer\":0,\"shape\":\"3->64 k7s2\",\"cin\":3,\"cout\":64,\"kh\":7,\"kw\":7,"
                   "\"sh\":2,\"sw\":2,\"ph\":3,\"pw\":3,\"h\":32,\"w\":32,"
                   "\"impl\":\"implicit\",\"t_cublas_ms\":1.5,\"t_implicit_ms\":4.3},\n"
                   " {\"layer\":1,\"cin\":64,\"cout\":64,\"kh\":3,\"kw\":3,\"sh\":1,\"sw\":1,"
                   "\"ph\":1,\"pw\":1,\"h\":8,\"w\":8,\"impl\":\"cublas\"}\n]}\n");
        fclose(f);
        setenv("LAYER_IMPL_JSON", "/tmp/_li_test.json", 1);
        LayerImplTable t; std::string e;
        CHECK(t.init(20, &e));
        CHECK(t.entry_count() == 2);
        const char* how = "";
        CHECK(t.select(0, 3, 64, 7, 7, 2, 2, 3, 3, 32, 32, &how) == LayerImpl::Implicit);
        CHECK(!strcmp(how, "table"));
        // 形状不匹配 → 启发式
        CHECK(t.select(0, 3, 64, 7, 7, 2, 2, 3, 3, 16, 16, &how) == LayerImpl::Cublas);
        CHECK(!strcmp(how, "shape-mismatch"));
        unsetenv("LAYER_IMPL_JSON");
    }
    {
        setenv("LAYER_IMPL_JSON", "/tmp/does-not-exist.json", 1);
        LayerImplTable t; std::string e;
        CHECK(!t.init(20, &e));
        unsetenv("LAYER_IMPL_JSON");
    }
    {
        // batch cap
        unsetenv("USE_IMPLICIT_GEMM"); unsetenv("LAYER_IMPL");
        CHECK(layer_impl_batch_cap() == 512);
        setenv("LAYER_IMPL", "all:implicit", 1);
        CHECK(layer_impl_batch_cap() == 2048);
        unsetenv("LAYER_IMPL");
    }
    printf(fails ? "TEST FAILED (%d)\n" : "TEST OK\n", fails);
    return fails ? 1 : 0;
}
