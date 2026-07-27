//! rsmb-rs — ReelSmart-style motion blur for video files.
//!
//! Pipeline per frame:
//!   1. decode RGB24 frames from the input video via ffmpeg,
//!   2. estimate dense optical flow to the previous and next frames
//!      (pyramidal Lucas-Kanade),
//!   3. blur each pixel along its motion trajectory, shutter angle controls
//!      the exposure length,
//!   4. stream the result into an ffmpeg H.264 encoder.

mod blur;
mod flow;
mod gpu;
mod video;

use anyhow::Result;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;

#[derive(Parser, Debug)]
#[command(
    name = "rsmb-rs",
    version,
    about = "Add high-quality optical-flow motion blur (RSMB style) to a video"
)]
struct Args {
    /// Input video file (e.g. MP4).
    #[arg(value_name = "INPUT")]
    input: PathBuf,

    /// Output video file.
    #[arg(short, long, value_name = "OUTPUT")]
    output: PathBuf,

    /// Shutter angle in degrees, controls blur amount (360 = one full frame
    /// interval of exposure, like RSMB's shutter angle). 0 disables blur.
    #[arg(short, long, default_value_t = 180.0)]
    shutter: f32,

    /// Number of samples along the motion trajectory (quality knob).
    #[arg(long, default_value_t = 16)]
    samples: usize,

    /// Lucas-Kanade window radius in pixels (larger = smoother flow).
    #[arg(long, default_value_t = 7)]
    window: usize,

    /// Solver iterations per pyramid level.
    #[arg(long, default_value_t = 5)]
    iters: usize,

    /// H.264 CRF quality (lower = better).
    #[arg(long, default_value_t = 18)]
    crf: u8,

    /// x264 encoder preset.
    #[arg(long, default_value = "medium")]
    preset: String,

    /// Path to the ffmpeg binary.
    #[arg(long, default_value = "ffmpeg")]
    ffmpeg: String,

    /// Max pyramid levels for optical flow (0 = auto, 1-6 = fixed).
    /// Fewer levels = faster but less accurate for large motion.
    #[arg(long, default_value_t = 0)]
    levels: usize,

    /// Reuse forward flow as negated backward flow (faster, minor quality loss).
    #[arg(long)]
    flow_cache: bool,

    /// Use GPU acceleration (supports NVIDIA, AMD, Intel iGPU via wgpu).
    #[arg(long)]
    gpu: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if !args.input.exists() {
        anyhow::bail!("input file not found: {}", args.input.display());
    }

    let info = video::probe(&args.ffmpeg, &args.input)?;
    let (w, h) = (info.width, info.height);
    let frame_size = w * h * 3;

    let gpu_ctx: Option<gpu::GpuContext> = if args.gpu {
        Some(gpu::GpuContext::new()?)
    } else {
        None
    };

    let (raw_tx, raw_rx) = mpsc::sync_channel::<Option<Vec<u8>>>(16);
    let (proc_tx, proc_rx) = mpsc::sync_channel::<Vec<u8>>(8);
    // Buffer pool: retired frame buffers are recycled by the decoder thread
    // instead of cloning a fresh 6 MB buffer for every frame.
    let (pool_tx, pool_rx) = mpsc::sync_channel::<Vec<u8>>(8);

    let dec_ffmpeg = args.ffmpeg.clone();
    let dec_input = args.input.clone();
    let dec_handle = thread::spawn(move || -> Result<()> {
        let mut dec = video::Decoder::new(&dec_ffmpeg, &dec_input)?;
        loop {
            let mut buf = match pool_rx.try_recv() {
                Ok(b) => b,
                Err(_) => vec![0u8; frame_size],
            };
            if !dec.read_frame(&mut buf)? {
                break;
            }
            if raw_tx.send(Some(buf)).is_err() {
                break;
            }
        }
        let _ = raw_tx.send(None);
        dec.finish()
    });

    let mut encoder = video::Encoder::new(
        &args.ffmpeg,
        &args.input,
        &args.output,
        w,
        h,
        info.fps,
        args.crf,
        &args.preset,
    )?;
    let enc_handle = thread::spawn(move || -> Result<()> {
        while let Ok(frame) = proc_rx.recv() {
            encoder.write_frame(&frame)?;
        }
        encoder.finish()
    });

    let pb = if info.total_frames > 0 {
        let pb = ProgressBar::new(info.total_frames);
        pb.set_style(
            ProgressStyle::with_template(
                "{bar:40.green/black} {pos:>6}/{len:6} [{elapsed_precise} | {eta}] {per_sec}",
            )
            .unwrap()
            .progress_chars("█▉▊▋▌▍▎▏ "),
        );
        pb
    } else {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::with_template("{spinner:.green} frame {pos} | {elapsed_precise} | {per_sec}")
                .unwrap(),
        );
        pb
    };
    pb.println(format!(
        "input : {} ({}x{} @ {:.3} fps, {} frames)",
        args.input.display(),
        w,
        h,
        info.fps,
        if info.total_frames > 0 { info.total_frames.to_string() } else { "?".into() }
    ));
    pb.println(format!(
        "params: shutter={}° samples={} window={} iters={} levels={} cache={} gpu={} crf={} preset={}",
        args.shutter, args.samples, args.window, args.iters,
        if args.levels > 0 { args.levels.to_string() } else { "auto".into() },
        args.flow_cache, args.gpu, args.crf, args.preset
    ));

    let mut prev = vec![0u8; frame_size];
    let mut cur: Vec<u8>;
    let mut next: Vec<u8>;
    let has_next: bool;

    match raw_rx.recv() {
        Ok(Some(f)) => cur = f,
        Ok(None) => anyhow::bail!("input video has no frames"),
        Err(_) => anyhow::bail!("decoder failed unexpectedly"),
    }
    match raw_rx.recv() {
        Ok(Some(f)) => { next = f; has_next = true; }
        Ok(None) => { next = vec![0u8; frame_size]; next.copy_from_slice(&cur); has_next = false; }
        Err(_) => { next = vec![0u8; frame_size]; next.copy_from_slice(&cur); has_next = false; }
    }

    // Grayscale conversions are cached and rotated together with the RGB
    // frames, so each frame is converted exactly once (not 3x).
    let mut gray_cur = flow::Gray::from_rgb(&cur, w, h);
    let mut gray_next = flow::Gray::from_rgb(&next, w, h);
    let mut gray_prev = gray_cur.clone(); // frame 0: duplicated neighbour

    let mut flow_cache_cpu: Option<(Vec<f32>, Vec<f32>)> = None;
    let mut flow_cache_gpu: Option<gpu::GpuFlowPair> = None;

    let mut n: u64 = 0;
    loop {
        let processed = if args.shutter <= 0.0 {
            cur.clone()
        } else if let Some(ctx) = &gpu_ctx {
            // ---- GPU path: flow fields never leave the device -----------
            let flows = if args.flow_cache {
                if let Some(cached) = flow_cache_gpu.take() {
                    let bwd = gpu::gpu_negate_flow(ctx, &cached, w, h);
                    let fwd = gpu::gpu_flow_single_device(
                        ctx, &gray_cur.data, &gray_next.data, w, h, args.window, args.iters, args.levels,
                    )?;
                    flow_cache_gpu = Some(fwd.clone());
                    gpu::GpuFrameFlows { fwd, bwd }
                } else {
                    let f = gpu::gpu_flows_device(
                        ctx, &gray_cur.data, &gray_prev.data, &gray_next.data, w, h, args.window, args.iters, args.levels,
                    )?;
                    flow_cache_gpu = Some(f.fwd.clone());
                    f
                }
            } else {
                gpu::gpu_flows_device(
                    ctx, &gray_cur.data, &gray_prev.data, &gray_next.data, w, h, args.window, args.iters, args.levels,
                )?
            };
            gpu::gpu_motion_blur(ctx, &cur, &flows, w, h, args.shutter, args.samples)?
        } else {
            // ---- CPU path -------------------------------------------------
            let (fwd, bwd) = if args.flow_cache {
                if let Some((cu, cv)) = flow_cache_cpu.take() {
                    let neg_u: Vec<f32> = cu.iter().map(|v| -v).collect();
                    let neg_v: Vec<f32> = cv.iter().map(|v| -v).collect();
                    let bwd = flow::FlowField { w, h, u: neg_u, v: neg_v };
                    let fwd = flow::median_flow(&flow::optical_flow(
                        &gray_cur, &gray_next, args.window, args.iters, args.levels,
                    ));
                    flow_cache_cpu = Some((fwd.u.clone(), fwd.v.clone()));
                    (fwd, bwd)
                } else {
                    let (fwd, bwd) = rayon::join(
                        || flow::median_flow(&flow::optical_flow(
                            &gray_cur, &gray_next, args.window, args.iters, args.levels,
                        )),
                        || flow::median_flow(&flow::optical_flow(
                            &gray_cur, &gray_prev, args.window, args.iters, args.levels,
                        )),
                    );
                    flow_cache_cpu = Some((fwd.u.clone(), fwd.v.clone()));
                    (fwd, bwd)
                }
            } else {
                rayon::join(
                    || flow::median_flow(&flow::optical_flow(
                        &gray_cur, &gray_next, args.window, args.iters, args.levels,
                    )),
                    || flow::median_flow(&flow::optical_flow(
                        &gray_cur, &gray_prev, args.window, args.iters, args.levels,
                    )),
                )
            };

            blur::motion_blur(&cur, w, h, &fwd, &bwd, args.shutter, args.samples)
        };

        if proc_tx.send(processed).is_err() {
            anyhow::bail!("encoder channel closed unexpectedly");
        }
        n += 1;
        pb.set_position(n);

        if !has_next {
            break;
        }

        std::mem::swap(&mut prev, &mut cur);
        std::mem::swap(&mut cur, &mut next);
        std::mem::swap(&mut gray_prev, &mut gray_cur);
        std::mem::swap(&mut gray_cur, &mut gray_next);

        match raw_rx.recv() {
            Ok(Some(frame)) => {
                // Return the retired buffer (old `prev`, now scratch) to the
                // decoder's pool instead of dropping it.
                let retired = std::mem::replace(&mut next, frame);
                let _ = pool_tx.send(retired);
                gray_next = flow::Gray::from_rgb(&next, w, h);
            }
            Ok(None) => {
                // Last frame: forward side duplicates the current frame.
                let processed = if args.shutter <= 0.0 {
                    cur.clone()
                } else if let Some(ctx) = &gpu_ctx {
                    let flows = gpu::GpuFrameFlows {
                        fwd: gpu::gpu_flow_single_device(
                            ctx, &gray_cur.data, &gray_cur.data, w, h, args.window, args.iters, args.levels,
                        )?,
                        bwd: gpu::gpu_flow_single_device(
                            ctx, &gray_cur.data, &gray_prev.data, w, h, args.window, args.iters, args.levels,
                        )?,
                    };
                    gpu::gpu_motion_blur(ctx, &cur, &flows, w, h, args.shutter, args.samples)?
                } else {
                    let (fwd, bwd) = rayon::join(
                        || flow::median_flow(&flow::optical_flow(
                            &gray_cur, &gray_cur, args.window, args.iters, args.levels,
                        )),
                        || flow::median_flow(&flow::optical_flow(
                            &gray_cur, &gray_prev, args.window, args.iters, args.levels,
                        )),
                    );
                    blur::motion_blur(&cur, w, h, &fwd, &bwd, args.shutter, args.samples)
                };

                if proc_tx.send(processed).is_err() {
                    anyhow::bail!("encoder channel closed unexpectedly");
                }
                n += 1;
                pb.set_position(n);
                break;
            }
            Err(_) => {
                anyhow::bail!("decoder failed unexpectedly");
            }
        }
    }

    drop(proc_tx);
    drop(pool_tx);

    dec_handle.join().map_err(|_| anyhow::anyhow!("decoder thread panicked"))??;
    enc_handle.join().map_err(|_| anyhow::anyhow!("encoder thread panicked"))??;

    if info.total_frames > 0 && n != info.total_frames {
        pb.println(format!(
            "WARNING: processed {} frames but input reports {} — output may be truncated",
            n, info.total_frames
        ));
    }
    pb.finish_with_message(format!("done — {} frames", n));
    println!("output: {}", args.output.display());
    Ok(())
}
