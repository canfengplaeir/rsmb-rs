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
mod video;

use anyhow::{Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;

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
}

fn main() -> Result<()> {
    let args = Args::parse();
    if !args.input.exists() {
        anyhow::bail!("input file not found: {}", args.input.display());
    }

    let info = video::probe(&args.ffmpeg, &args.input)?;
    let (w, h) = (info.width, info.height);
    let frame_size = w * h * 3;
    println!(
        "input : {} ({}x{} @ {:.3} fps)",
        args.input.display(),
        w,
        h,
        info.fps
    );
    println!(
        "params: shutter={}° samples={} window={} iters={} crf={} preset={}",
        args.shutter, args.samples, args.window, args.iters, args.crf, args.preset
    );

    let mut decoder = video::Decoder::new(&args.ffmpeg, &args.input)?;
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

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::with_template("{spinner:.green} frame {pos} | {elapsed_precise} | {per_sec}")
            .unwrap(),
    );

    // Sliding three-frame window: prev / cur / next.
    let mut prev = vec![0u8; frame_size];
    let mut cur = vec![0u8; frame_size];
    let mut next = vec![0u8; frame_size];

    // Prime the pipeline with the first two frames.
    if !decoder.read_frame(&mut cur)? {
        anyhow::bail!("input video has no frames");
    }
    let has_next = decoder.read_frame(&mut next)?;

    let mut n: u64 = 0;
    loop {
        let processed = if args.shutter <= 0.0 {
            cur.clone()
        } else {
            // For the first/last frame the missing neighbour is duplicated,
            // which simply halves the shutter on that side.
            let gray_prev = flow::Gray::from_rgb(if n == 0 { &cur } else { &prev }, w, h);
            let gray_cur = flow::Gray::from_rgb(&cur, w, h);
            let gray_next = flow::Gray::from_rgb(if has_next { &next } else { &cur }, w, h);

            let fwd = flow::median_flow(&flow::optical_flow(
                &gray_cur, &gray_next, args.window, args.iters,
            ));
            let bwd = flow::median_flow(&flow::optical_flow(
                &gray_cur, &gray_prev, args.window, args.iters,
            ));

            blur::motion_blur(&cur, w, h, &fwd, &bwd, args.shutter, args.samples)
        };
        encoder.write_frame(&processed)?;
        n += 1;
        pb.set_position(n);

        if !has_next {
            break;
        }
        std::mem::swap(&mut prev, &mut cur);
        std::mem::swap(&mut cur, &mut next);
        // After the swap `next` holds the old `cur` buffer (scratch space).
        let got = decoder.read_frame(&mut next)?;
        if !got {
            // One last frame to emit with the duplicated-neighbour path.
            let gray_prev = flow::Gray::from_rgb(&prev, w, h);
            let gray_cur = flow::Gray::from_rgb(&cur, w, h);
            let fwd = flow::median_flow(&flow::optical_flow(
                &gray_cur, &gray_cur, args.window, args.iters,
            ));
            let bwd = flow::median_flow(&flow::optical_flow(
                &gray_cur, &gray_prev, args.window, args.iters,
            ));
            let processed = if args.shutter <= 0.0 {
                cur.clone()
            } else {
                blur::motion_blur(&cur, w, h, &fwd, &bwd, args.shutter, args.samples)
            };
            encoder.write_frame(&processed)?;
            n += 1;
            pb.set_position(n);
            break;
        }
    }

    decoder.finish().context("decoder error")?;
    encoder.finish().context("encoder error")?;
    pb.finish_with_message(format!("done — {} frames", n));
    println!("output: {}", args.output.display());
    Ok(())
}
