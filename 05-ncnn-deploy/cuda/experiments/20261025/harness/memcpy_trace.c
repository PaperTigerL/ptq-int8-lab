/* LD_PRELOAD interposer: aggregate cudaMemcpy* calls by kind, bytes and call site.
 * nsys on this WSL box only records the runtime API name (no size/direction), so
 * this shim supplies what CUPTI_ACTIVITY_KIND_MEMCPY would have provided.
 * Output: MEMCPY_TRACE_OUT (default /tmp/memcpy_trace.txt), one line per
 * (kind, bytes, caller), plus a TOTAL line. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef int cudaError_t;
typedef void *cudaStream_t;

typedef cudaError_t (*fn3)(void *, const void *, size_t, int);
typedef cudaError_t (*fn5)(void *, const void *, size_t, int, cudaStream_t);
typedef cudaError_t (*fn2d)(void *, size_t, const void *, size_t, size_t, size_t, int);
typedef cudaError_t (*fn2d5)(void *, size_t, const void *, size_t, size_t, size_t, int, cudaStream_t);

#define MAXENT 8192
typedef struct
{
    int kind;
    size_t bytes;
    void *caller;
    void *modbase;
    char mod[128];
    long count;
    unsigned long long total;
} Ent;

static Ent g_ent[MAXENT];
static int g_n;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static long g_calls;
static const char *g_out;

static void resolve(Ent *e, void *ra)
{
    Dl_info info;
    e->caller = ra;
    if (dladdr(ra, &info) && info.dli_fbase)
    {
        e->modbase = info.dli_fbase;
        snprintf(e->mod, sizeof(e->mod), "%s", info.dli_fname ? info.dli_fname : "?");
    }
}

static void record(int kind, size_t bytes, void *ra)
{
    pthread_mutex_lock(&g_mu);
    g_calls++;
    int i;
    for (i = 0; i < g_n; i++)
    {
        Ent *e = &g_ent[i];
        if (e->kind == kind && e->bytes == bytes && e->caller == ra)
        {
            e->count++;
            e->total += bytes;
            pthread_mutex_unlock(&g_mu);
            return;
        }
    }
    if (g_n < MAXENT)
    {
        Ent *e = &g_ent[g_n++];
        memset(e, 0, sizeof(*e));
        e->kind = kind;
        e->bytes = bytes;
        resolve(e, ra);
        e->count = 1;
        e->total = bytes;
    }
    pthread_mutex_unlock(&g_mu);
}

static void dump(void)
{
    const char *path = getenv("MEMCPY_TRACE_OUT");
    if (!path) path = g_out ? g_out : "/tmp/memcpy_trace.txt";
    FILE *f = fopen(path, "w");
    if (!f) return;
    long long tcount[8] = {0};
    unsigned long long tbytes[8] = {0};
    fprintf(f, "# kind bytes count total caller_offset module\n");
    for (int i = 0; i < g_n; i++)
    {
        Ent *e = &g_ent[i];
        long long off = e->modbase ? (long long)((char *)e->caller - (char *)e->modbase) : -1;
        fprintf(f, "%d %zu %ld %llu %lld %s\n", e->kind, e->bytes, e->count, e->total, off, e->mod);
        if (e->kind >= 0 && e->kind < 8)
        {
            tcount[e->kind] += e->count;
            tbytes[e->kind] += e->total;
        }
    }
    for (int k = 0; k < 8; k++)
        fprintf(f, "TOTAL kind=%d count=%lld bytes=%llu\n", k, tcount[k], tbytes[k]);
    fprintf(f, "TOTAL calls=%ld distinct=%d\n", g_calls, g_n);
    fclose(f);
}

__attribute__((destructor)) static void fini(void) { dump(); }

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, int kind)
{
    static fn3 real;
    if (!real) real = (fn3)dlsym(RTLD_NEXT, "cudaMemcpy");
    record(kind, count, __builtin_return_address(0));
    return real(dst, src, count, kind);
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, int kind, cudaStream_t stream)
{
    static fn5 real;
    if (!real) real = (fn5)dlsym(RTLD_NEXT, "cudaMemcpyAsync");
    record(kind, count, __builtin_return_address(0));
    return real(dst, src, count, kind, stream);
}

cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, int kind)
{
    static fn2d real;
    if (!real) real = (fn2d)dlsym(RTLD_NEXT, "cudaMemcpy2D");
    record(kind, width * height, __builtin_return_address(0));
    return real(dst, dpitch, src, spitch, width, height, kind);
}

cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, int kind, cudaStream_t stream)
{
    static fn2d5 real;
    if (!real) real = (fn2d5)dlsym(RTLD_NEXT, "cudaMemcpy2DAsync");
    record(kind, width * height, __builtin_return_address(0));
    return real(dst, dpitch, src, spitch, width, height, kind, stream);
}
