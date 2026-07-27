//! Video decoding/encoding through ffmpeg child processes.
//!
//! Decoding: ffmpeg converts the input file to raw RGB24 frames on stdout,
//! which we read frame by frame.
//! Encoding: processed RGB24 frames are written to ffmpeg's stdin and
//! encoded to H.264 (audio from the source is copied over when present).

use anyhow::{anyhow, Context, Result};
use std::io::{Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

#[derive(Debug, Clone, Copy)]
pub struct VideoInfo {
    pub width: usize,
    pub height: usize,
    pub fps: f64,
    pub total_frames: u64,
}

/// Locate ffprobe next to the chosen ffmpeg binary (same package family).
fn ffprobe_name(ffmpeg: &str) -> String {
    if ffmpeg.to_lowercase().ends_with("ffmpeg.exe") {
        ffmpeg[..ffmpeg.len() - "ffmpeg.exe".len()].to_string() + "ffprobe.exe"
    } else if ffmpeg.ends_with("ffmpeg") {
        ffmpeg[..ffmpeg.len() - "ffmpeg".len()].to_string() + "ffprobe"
    } else {
        "ffprobe".to_string()
    }
}

pub fn probe(ffmpeg: &str, input: &Path) -> Result<VideoInfo> {
    let ffprobe = ffprobe_name(ffmpeg);
    // NOTE: csv output does NOT preserve the requested field order, so parse
    // key=value pairs instead of positional columns.
    let out = Command::new(&ffprobe)
        .args([
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
            "-of", "default=noprint_wrappers=1",
        ])
        .arg(input)
        .output()
        .with_context(|| format!("failed to run {ffprobe}"))?;
    if !out.status.success() {
        return Err(anyhow!(
            "ffprobe failed: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut width: Option<usize> = None;
    let mut height: Option<usize> = None;
    let mut r_fps: Option<f64> = None;
    let mut avg_fps: Option<f64> = None;
    let mut nb_frames: Option<u64> = None;
    let mut duration: Option<f64> = None;
    for line in text.lines() {
        let Some((key, val)) = line.trim().split_once('=') else {
            continue;
        };
        match key {
            "width" => width = val.parse().ok(),
            "height" => height = val.parse().ok(),
            "r_frame_rate" => r_fps = parse_rate(val).ok().filter(|f| *f > 0.0),
            "avg_frame_rate" => avg_fps = parse_rate(val).ok().filter(|f| *f > 0.0),
            "nb_frames" => nb_frames = val.parse().ok(),
            "duration" => duration = val.parse().ok(),
            _ => {}
        }
    }
    let (width, height) = match (width, height) {
        (Some(w), Some(h)) if w > 0 && h > 0 => (w, h),
        _ => return Err(anyhow!("ffprobe returned no valid stream info")),
    };
    // avg_frame_rate tracks the real cadence of VFR sources; fall back to
    // the nominal r_frame_rate.
    let fps = match (avg_fps, r_fps) {
        (Some(a), _) => a,
        (None, Some(r)) => r,
        _ => return Err(anyhow!("ffprobe returned no frame rate")),
    };
    let total_frames = match nb_frames {
        Some(n) if n > 0 => n,
        _ => match duration {
            Some(d) if d > 0.0 => (d * fps).round() as u64,
            _ => 0,
        },
    };
    Ok(VideoInfo { width, height, fps, total_frames })
}

fn parse_rate(s: &str) -> Result<f64> {
    if let Some((num, den)) = s.split_once('/') {
        let num: f64 = num.parse().context("bad fps numerator")?;
        let den: f64 = den.parse().context("bad fps denominator")?;
        Ok(num / den)
    } else {
        Ok(s.parse()?)
    }
}

/// Streaming decoder: yields RGB24 frames from ffmpeg's stdout.
pub struct Decoder {
    child: Child,
    stdout: ChildStdout,
}

impl Decoder {
    pub fn new(ffmpeg: &str, input: &Path) -> Result<Self> {
        let mut child = Command::new(ffmpeg)
            .args([
                "-hide_banner",
                "-loglevel", "error",
                "-i",
            ])
            .arg(input)
            .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("failed to spawn {ffmpeg}"))?;
        let stdout = child.stdout.take().unwrap();
        Ok(Decoder { child, stdout })
    }

    /// Read one frame into `buf`. Returns false on end of stream.
    pub fn read_frame(&mut self, buf: &mut [u8]) -> Result<bool> {
        match self.stdout.read_exact(buf) {
            Ok(()) => Ok(true),
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(false),
            Err(e) => Err(e.into()),
        }
    }

    pub fn finish(mut self) -> Result<()> {
        let status = self.child.wait()?;
        if !status.success() {
            return Err(anyhow!("ffmpeg decoder exited with {status}"));
        }
        Ok(())
    }
}

/// Streaming encoder: accepts RGB24 frames, writes an H.264 MP4.
pub struct Encoder {
    child: Child,
    stdin: Option<ChildStdin>,
}

impl Encoder {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ffmpeg: &str,
        input: &Path,
        output: &Path,
        width: usize,
        height: usize,
        fps: f64,
        crf: u8,
        preset: &str,
    ) -> Result<Self> {
        let fps_str = format!("{fps}");
        let size = format!("{width}x{height}");
        let mut child = Command::new(ffmpeg)
            .args([
                "-hide_banner",
                "-loglevel", "error",
                "-y",
                "-thread_queue_size", "512",
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "-s", &size,
                "-r", &fps_str,
                "-i", "pipe:0",
                "-thread_queue_size", "512",
                "-i",
            ])
            .arg(input)
            .args([
                "-map", "0:v:0",
                "-map", "1:a?",
                "-c:v", "libx264",
                "-preset", preset,
                "-crf", &crf.to_string(),
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-movflags", "+faststart",
            ])
            .arg(output)
            .stdin(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("failed to spawn {ffmpeg}"))?;
        let stdin = child.stdin.take().unwrap();
        Ok(Encoder {
            child,
            stdin: Some(stdin),
        })
    }

    pub fn write_frame(&mut self, frame: &[u8]) -> Result<()> {
        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| anyhow!("encoder already finished"))?;
        stdin.write_all(frame).map_err(|e| {
            if e.kind() == std::io::ErrorKind::BrokenPipe {
                anyhow!("encoder pipe closed unexpectedly (ffmpeg error — see messages above)")
            } else {
                e.into()
            }
        })
    }

    pub fn finish(mut self) -> Result<()> {
        drop(self.stdin.take()); // close pipe so ffmpeg sees EOF
        let status = self.child.wait()?;
        if !status.success() {
            return Err(anyhow!("ffmpeg encoder exited with {status}"));
        }
        Ok(())
    }
}
