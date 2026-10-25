// 最小验证：libtorch C++ + CUDA 能否在本机工作。
// 编译：
//   g++ -std=c++17 cuda_smoke.cpp -o cuda_smoke \
//     -I/home/user/libtorch/include \
//     -I/home/user/libtorch/include/torch/csrc/api/include \
//     -L/home/user/libtorch/lib -ltorch -ltorch_cuda -lc10 -lc10_cuda -ltorch_cpu \
//     -Wl,-rpath,/home/user/libtorch/lib
#include <chrono>
#include <cstdio>
#include <torch/torch.h>

int main()
{
    printf("cuda available: %d\n", (int)torch::cuda::is_available());
    if (!torch::cuda::is_available())
        return 1;

    printf("device count: %d\n", (int)torch::cuda::device_count());

    auto dev = torch::Device(torch::kCUDA, 0);
    auto x = torch::randn({64, 64}, dev);
    auto w = torch::randn({64, 64}, dev);

    // warmup
    for (int i = 0; i < 5; i++) auto y = torch::mm(x, w);
    torch::cuda::synchronize();

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 200; i++) auto y = torch::mm(x, w);
    torch::cuda::synchronize();
    auto t1 = std::chrono::steady_clock::now();

    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 200;
    printf("64x64 mm: %.4f ms/次\n", ms);

    // 大批量，模拟 batch 推理
    auto xb = torch::randn({256, 3, 32, 32}, dev);
    auto convw = torch::randn({64, 3, 3, 3}, dev); auto convb = torch::randn({64}, dev);
    for (int i = 0; i < 5; i++) auto yb = torch::conv2d(xb, convw, convb, {1}, {1}, {1}, 1);
    torch::cuda::synchronize();

    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 50; i++) auto yb = torch::conv2d(xb, convw, convb, {1}, {1}, {1}, 1);
    torch::cuda::synchronize();
    t1 = std::chrono::steady_clock::now();

    const double msb = std::chrono::duration<double, std::milli>(t1 - t0).count() / 50;
    printf("conv 3->64 3x3 batch=256: %.3f ms/次 = %.5f ms/img\n", msb, msb / 256);

    printf("CUDA 可用\n");
    return 0;
}
