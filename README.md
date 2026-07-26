# rsmb-rs — 视频动态模糊工具（RSMB 风格）

使用 Rust 实现的命令行工具，为视频添加基于光流运动估计的高质量动态模糊，效果类似 After Effects 的 ReelSmart Motion Blur (RSMB) 插件。

## 工作原理

1. **解码**：调用 ffmpeg 将输入视频解码为 RGB24 帧序列；
2. **运动估计**：对每帧计算稠密光流（金字塔 Lucas-Kanade：高斯金字塔由粗到细 + 局部窗口结构张量求解 + 中值滤波去噪），得到逐像素运动向量；
3. **定向模糊**：以当前帧为曝光中心，沿运动轨迹做双线性定向采样并叠加平均，模拟相机快门曝光；
4. **编码**：处理后的帧通过管道交给 ffmpeg 编码为 H.264 MP4（自动携带原视频音轨）。

## 性能特性

- **多线程并行**：解码、光流计算、模糊合成、编码管线化分离到独立线程，充分利用多核 CPU
- **光流并行**：前向/后向光流并发计算，金字塔构建与结构张量内部并行
- **进度条**：自动探测视频总帧数，显示百分比进度条、已用时间、预计剩余时间（ETA）、处理速度

## 构建

```bash
cargo build --release
```

需要系统安装 `ffmpeg` / `ffprobe`（可用 `--ffmpeg` 指定路径）。

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
| `--flow-cache` | 复用前向光流作为后向光流（大幅提速，质量影响极小） | 关闭 |
| `--crf <0-51>` | x264 输出质量（越小越清晰） | 18 |
| `--preset <名>` | x264 编码预设 | medium |
| `--ffmpeg <路径>` | ffmpeg 可执行文件路径 | ffmpeg |

### 示例

```bash
# 标准 180° 快门模糊
rsmb-rs input.mp4 -o blurred.mp4

# 强模糊，高质量采样
rsmb-rs input.mp4 -o blurred.mp4 --shutter 300 --samples 32

# 快速模式（降金字塔 + 流缓存）
rsmb-rs input.mp4 -o fast.mp4 --levels 3 --flow-cache

# 极速预览（少采样 + 快速编码 + 算法加速）
rsmb-rs input.mp4 -o preview.mp4 --levels 2 --flow-cache --samples 8 --preset veryfast

# 指定 ffmpeg 路径
rsmb-rs input.mp4 -o blurred.mp4 --ffmpeg "C:\ffmpeg\bin\ffmpeg.exe"
```

## 性能提示

- 光流计算量与分辨率成正比，1080p 视频处理速度约为每秒数帧
- `--levels 3 --flow-cache` 相比默认参数约提速 50-80%，画质损失极小
- 可先用 `--samples 8 --levels 2 --flow-cache --preset veryfast` 快速预览效果
