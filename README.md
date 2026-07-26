# rsmb-rs — 视频动态模糊工具（RSMB 风格）

使用 Rust 实现的命令行工具，为视频添加基于光流运动估计的高质量动态模糊，效果类似 After Effects 的 ReelSmart Motion Blur (RSMB) 插件。

## 工作原理

1. **解码**：调用 ffmpeg 将输入视频（MP4 等常见格式）解码为 RGB24 帧序列；
2. **运动估计**：对每帧计算到前一帧和后一帧的稠密光流（金字塔 Lucas-Kanade：高斯金字塔由粗到细 + 局部窗口结构张量求解 + 中值滤波去噪），得到逐像素运动向量；
3. **定向模糊**：以当前帧为曝光中心，沿运动轨迹（前半段用后向流、后半段用前向流）做双线性定向采样并叠加平均，模拟相机快门曝光；
4. **编码**：处理后的帧通过管道交给 ffmpeg 编码为 H.264 MP4（自动携带原视频音轨）。

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
| `-s, --shutter <度>` | 快门角度，控制模糊量（360° = 曝光覆盖一个完整帧间隔；0 关闭模糊） | 180 |
| `--samples <N>` | 沿运动轨迹的采样数（质量/速度权衡） | 16 |
| `--window <R>` | Lucas-Kanade 窗口半径（越大光流越平滑） | 7 |
| `--iters <N>` | 每个金字塔层级的求解迭代次数 | 5 |
| `--crf <0-51>` | x264 输出质量（越小越清晰） | 18 |
| `--preset <名>` | x264 编码预设 | medium |
| `--ffmpeg <路径>` | ffmpeg 可执行文件路径 | ffmpeg |

### 示例

```bash
# 标准 180° 快门模糊
rsmb-rs input.mp4 -o blurred.mp4

# 强模糊，高质量采样
rsmb-rs input.mp4 -o blurred.mp4 --shutter 300 --samples 32

# 快速预览（低采样 + 快速编码）
rsmb-rs input.mp4 -o preview.mp4 --samples 8 --preset veryfast
```

## 性能提示

光流计算量与分辨率成正比，1080p 视频处理速度约为每秒数帧（多核并行）。可先用 `--samples 8` 和低分辨率副本快速验证效果。
