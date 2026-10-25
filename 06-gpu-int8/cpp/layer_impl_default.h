// 自动生成：./select_layers resnet18_folded.bin calib_minmax.bin --batch 512 --json ... --header ...
// 不要手改：改判据请改 select_layers.cpp，然后重跑生成。
// t_cublas_ms / t_implicit_ms 是生成时（batch=512）的实测，供人读，不参与运行期判断。
// ⚠️ 本文件在 layer_impl.h 的 namespace ptqint8 内部被 include，不能再开 namespace。
#pragma once

inline const LayerImplEntry kDefaultLayerImpl[] = {
    {0, 3, 64, 7, 7, 2, 2, 3, 3, 32, 32, LayerImpl::Cublas, 1.6794, 5.8264, 512, false},
    {1, 64, 64, 3, 3, 1, 1, 1, 1, 8, 8, LayerImpl::Cublas, 0.8704, 1.0206, 512, false},
    {2, 64, 64, 3, 3, 1, 1, 1, 1, 8, 8, LayerImpl::Cublas, 0.8745, 1.0260, 512, false},
    {3, 64, 64, 3, 3, 1, 1, 1, 1, 8, 8, LayerImpl::Cublas, 0.8685, 1.0141, 512, false},
    {4, 64, 64, 3, 3, 1, 1, 1, 1, 8, 8, LayerImpl::Cublas, 0.8753, 1.0263, 512, false},
    {5, 64, 128, 3, 3, 2, 2, 1, 1, 8, 8, LayerImpl::Cublas, 0.3387, 0.5415, 512, false},
    {6, 128, 128, 3, 3, 1, 1, 1, 1, 4, 4, LayerImpl::Cublas, 0.4301, 0.7452, 512, false},
    {7, 64, 128, 1, 1, 2, 2, 0, 0, 8, 8, LayerImpl::Implicit, 0.2263, 0.2132, 512, false},
    {8, 128, 128, 3, 3, 1, 1, 1, 1, 4, 4, LayerImpl::Cublas, 0.4301, 0.7442, 512, false},
    {9, 128, 128, 3, 3, 1, 1, 1, 1, 4, 4, LayerImpl::Cublas, 0.4301, 0.7453, 512, false},
    {10, 128, 256, 3, 3, 2, 2, 1, 1, 4, 4, LayerImpl::Cublas, 0.2554, 0.5755, 512, false},
    {11, 256, 256, 3, 3, 1, 1, 1, 1, 2, 2, LayerImpl::Cublas, 0.3400, 0.7614, 512, false},
    {12, 128, 256, 1, 1, 2, 2, 0, 0, 4, 4, LayerImpl::Cublas, 0.1475, 0.1540, 512, false},
    {13, 256, 256, 3, 3, 1, 1, 1, 1, 2, 2, LayerImpl::Cublas, 0.3368, 0.7560, 512, false},
    {14, 256, 256, 3, 3, 1, 1, 1, 1, 2, 2, LayerImpl::Cublas, 0.3400, 0.7608, 512, false},
    {15, 256, 512, 3, 3, 2, 2, 1, 1, 2, 2, LayerImpl::Cublas, 0.1721, 0.2105, 512, false},
    {16, 512, 512, 3, 3, 1, 1, 1, 1, 1, 1, LayerImpl::Implicit, 0.3055, 0.1085, 512, false},
    {17, 256, 512, 1, 1, 2, 2, 0, 0, 2, 2, LayerImpl::Cublas, 0.0589, 0.0759, 512, false},
    {18, 512, 512, 3, 3, 1, 1, 1, 1, 1, 1, LayerImpl::Implicit, 0.2680, 0.0963, 512, false},
    {19, 512, 512, 3, 3, 1, 1, 1, 1, 1, 1, LayerImpl::Implicit, 0.2677, 0.1027, 512, false},
};
inline constexpr int kDefaultLayerImplCount = 20;
