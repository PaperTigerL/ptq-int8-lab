# Vulkan 可用性探测（2026-10-19）

目的：我在 `README.md` 第七节写过「真要 ncnn 上 GPU，正确做法是换 Vulkan 后端」。
这句话是**推测**，必须拿事实替换掉。下面是两侧的实际探测结果。

## WSL 侧：❌ 不可用

`/usr/share/vulkan/icd.d/` 下 8 个 ICD：

```
asahi_icd.json      libvulkan_asahi.so      (Apple M 系列)
gfxstream_vk_icd.json  libvulkan_gfxstream.so  (Android 模拟器)
intel_hasvk_icd.json   libvulkan_intel_hasvk.so
intel_icd.json         libvulkan_intel.so
lvp_icd.json           libvulkan_lvp.so        (lavapipe = 纯软件光栅)
nouveau_icd.json       libvulkan_nouveau.so
radeon_icd.json        libvulkan_radeon.so
virtio_icd.json        libvulkan_virtio.so     (虚拟化)
```

**没有 NVIDIA 的 ICD。** 能用的只有 `lvp`（lavapipe，CPU 软件实现，比直接跑 CPU 还慢）
和 `nouveau`（开源第三方，GTX 1650 Ti 上基本不可用）。
WSL 的 GPU 直通只给了 CUDA（`libcuda.so` / `libcudart`），没给 Vulkan。

## Windows 侧：❌ 不可用

| 检查项 | 结果 |
|---|---|
| `C:\Windows\System32\vulkan-1.dll` | 存在（1.73 MB）—— 只是 loader |
| `nvidia-smi` 驱动版本 | 616.92 / GTX 1650 Ti |
| `HKLM:\SOFTWARE\Khronos\Vulkan\Drivers` | **不存在**（64 位与 WOW6432Node 都查了） |
| `nv-vk64.dll`（DriverStore 全盘搜） | **找不到** |
| Vulkan SDK（`C:\VulkanSDK`） | **不存在** |
| `glslangValidator`（编译 ncnn shader 到 SPIR-V 必需） | **未安装** |

驱动装了，但**没带 Vulkan 组件**，也没有注册 ICD。要跑 ncnn Vulkan 至少还得装
Vulkan SDK（含 glslang），才能让 ncnn 的 `.comp` shader 编成 SPIR-V。

## 结论

Vulkan 路线在这台机器上**当前不可行**，而且就算装上了也只是"能跑"，不代表"能到 500×"。

更重要的一点：**瓶颈根本不在 CUDA 还是 Vulkan。**
ncnn 的 `Mat` 只有 `w/h/c`，图里没有 batch 维 —— 这是框架层面的限制，
换哪个 GPU 后端都一样。libtorch 的 148× 和 TensorRT 的 906× 都是 batch=512 打出来的，
ncnn 无论 CUDA 还是 Vulkan 都只能一张一张跑。

所以正确的判断不是"换 Vulkan"，而是：
**要用 GPU 吃到 batch 红利，就得用有 batch 维的推理引擎（TensorRT / libtorch / ONNXRuntime），
ncnn 的定位本来就是移动端单张低延迟推理，不是吞吐。**
