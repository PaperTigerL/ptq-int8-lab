/* H-048: redirect kernel launches onto a capture stream for CUDA Graph.
 *
 * ncnn's CUDA layer launches every kernel on the default stream (`<<<g,b>>>`),
 * which CUDA cannot capture. LD_PRELOAD this shim, call ncnn_graph_shim_set()
 * with a non-default stream, and launches are re-targeted for capture/replay.
 * Outside the redirect window calls pass through untouched.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>

typedef int cudaError_t;
typedef void* cudaStream_t;

typedef struct { unsigned int x, y, z; } dim3_h048;

typedef cudaError_t (*launch_fn)(const void*, dim3_h048, dim3_h048, void**, size_t, cudaStream_t);

static cudaStream_t g_stream;
static int g_enabled;

void ncnn_graph_shim_set(void* stream, int enabled)
{
    g_stream = (cudaStream_t)stream;
    g_enabled = enabled;
}

cudaError_t cudaLaunchKernel(const void* func, dim3_h048 grid, dim3_h048 block,
                             void** args, size_t shared, cudaStream_t stream)
{
    static launch_fn real;
    if (!real) real = (launch_fn)dlsym(RTLD_NEXT, "cudaLaunchKernel");
    return real(func, grid, block, args, shared, g_enabled ? g_stream : stream);
}
