# rsmb-rs — 视频动态模糊工具（RSMB 风格）

使用 Rust 实现的命令行工具，为视频添加基于光流运动估计的高质量动态模糊，效果类似 After Effects 的 ReelSmart Motion Blur (RSMB) 插件。支持 **GPU 加速**（wgpu 计算着色器，兼容 NVIDIA / AMD / Intel 核显）。

## 架构

```
解码线程 ──[raw_rx]──▶ 主处理线程 ──[proc_tx]──▶ 编码线程
    ▲                      │
    └── [pool_tx] 帧缓冲池 ─┘ (回收复用, 避免频繁分配 6MB buffer)
```

- **解码**：调用 ffmpeg 将输入视频解码为 RGB24 帧序列（独立线程，帧缓冲池回收复用）
- **运动估计**：稠密光流（金字塔 Lucas-Kanade：高斯金字塔 + 局部窗口结构张量 + 中值滤波）
- **定向模糊**：以当前帧为曝光中心，沿运动轨迹双线性定向采样叠加
- **编码**：ffmpeg 编码为 H.264 MP4（自动携带原视频音轨）

### GPU 优化（`--gpu`）

| 优化 | 说明 |
|------|------|
| 显存驻留 | 金字塔 / 结构张量 / 流场全程留在显存，帧间仅回读最终 RGB |
| 金字塔复用 | 当前帧高斯金字塔构建一次，前后向光流共享 |
| 合并提交 | 一帧内所有 compute dispatch 合并到一个 command encoder 提交 |
| 设备端流场 | `GpuFlowPair` 保持 flow u/v 在 GPU buffer，`--flow-cache` 时 GPU 直接取反 |

### CPU 优化

| 优化 | 说明 |
|------|------|
| 流水线并行 | 解码 / 光流 / 模糊 / 编码 分离到独立线程，buffered channel |
| 光流双路并行 | 前向 / 后向光流 `rayon::join` 并发求解 |
| 灰度缓存 | 灰度转换结果与 RGB 帧同步旋转，每帧仅转换一次 |
| 帧缓冲池 | 解码线程回收 retired buffer，避免每帧 malloc 6MB |

## 构建

```bash
cargo build --release
```

需要系统安装 `ffmpeg` / `ffprobe`（可用 `--ffmpeg` 指定路径）。GPU 模式需要支持 Vulkan / DX12 / Metal 的显卡驱动。

## 用法

```bash
rsmb-rs <输入视频> -o <输出视频> [选项]
```

| 选项 | 说明 | 默认值 |
| --- | --- | --- |
| `-s, --shutter <度>` | 快门角度，控制模糊量（360° = 一帧间隔曝光；0 关闭模糊） | 180 |
| `--samples <N>` | 沿运动轨迹的采样数（质量/速度权衡） | 16 |
| `--window <R>` | Lucas-Kanade 窗口半径（越大光流越平滑） | 7 |
| `--iters <N>` | 每个金字塔层级的求解迭代次数 | 5 |
| `--levels <N>` | 光流金字塔层数（0=自动，1-6=固定，越少越快） | 0 |
| `--flow-cache` | 复用前向光流取反作为后向光流（大幅提速，质量影响极小） | 关闭 |
| `--gpu` | 启用 GPU 加速（wgpu：Vulkan / DX12 / Metal） | 关闭 |
| `--crf <0-51>` | x264 输出质量（越小越清晰） | 18 |
| `--preset <名>` | x264 编码预设 | medium |
| `--ffmpeg <路径>` | ffmpeg 可执行文件路径 | ffmpeg |

### 示例

```bash
# 标准 180° 快门模糊（GPU 加速）
rsmb-rs input.mp4 -o blurred.mp4 --gpu

# 最高画质（适合最终渲染输出）
rsmb-rs input.mp4 -o final.mp4 --gpu --samples 64 --window 9 --iters 7 --crf 14 --preset veryslow

# 快速模式（降金字塔 + 流缓存）
rsmb-rs input.mp4 -o fast.mp4 --gpu --levels 3 --flow-cache

# 极速预览
rsmb-rs input.mp4 -o preview.mp4 --gpu --levels 2 --flow-cache --samples 8 --preset veryfast

# CPU 回退（无 GPU 环境）
rsmb-rs input.mp4 -o blurred.mp4 --levels 3 --flow-cache

# 指定 ffmpeg 路径
rsmb-rs input.mp4 -o blurred.mp4 --ffmpeg "C:\ffmpeg\bin\ffmpeg.exe" --gpu
```

## 性能实测

测试素材：1920×1080 @ 60fps 游戏录屏，180 帧，shutter=180°、samples=16、preset=veryfast。GPU 为 Intel UHD 750 核显。

| 配置 | 耗时 | 吞吐 | 相对加速 |
| --- | --- | --- | --- |
| 纯 CPU | 3m49s | 0.79 fps | 1× |
| GPU（初版，含 CPU 回读） | 1m49s | 1.65 fps | 2.1× |
| GPU（优化：显存驻留 + 金字塔复用 + 合并提交） | **1m05s** | **2.79 fps** | **3.5×** |

注：测试 GPU 为核显；独立显卡上加速比显著更高。

## 性能提示

- 优先使用 `--gpu`；CPU 路径作为无 GPU 环境的回退
- `--levels 3 --flow-cache` 可再提速 50-80%，画质损失极小
- GPU 模式下流缓存（`--flow-cache`）直接在设备端取反，几乎零开销
- 大位移画面建议增大 `--samples`（24-64）避免拖影断层
- 可先用 `--gpu --levels 2 --flow-cache --samples 8 --preset veryfast` 快速预览
