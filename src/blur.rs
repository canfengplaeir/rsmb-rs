//! Motion-blur synthesis: sample the current frame along each pixel's
//! optical-flow trajectory and accumulate the samples, mimicking a camera
//! shutter that stays open for a fraction of the frame interval.
//!
//! The exposure is centred on the current frame:
//!   * backward flow (current -> previous) covers the first half of the
//!     shutter interval,
//!   * forward flow (current -> next) covers the second half.
//! A 360-degree shutter therefore spans one full frame interval (half a
//! frame of motion in each direction), and 180 degrees half of it — exactly
//! the way shutter angle controls blur amount in AE's RSMB.

use crate::flow::FlowField;
use rayon::prelude::*;

/// Bilinear RGB sample with border clamping.
#[inline]
fn sample_rgb(frame: &[u8], w: usize, h: usize, x: f32, y: f32, out: &mut [f32; 3]) {
    let x = x.clamp(0.0, w as f32 - 1.001);
    let y = y.clamp(0.0, h as f32 - 1.001);
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let w00 = (1.0 - fx) * (1.0 - fy);
    let w10 = fx * (1.0 - fy);
    let w01 = (1.0 - fx) * fy;
    let w11 = fx * fy;
    let i00 = (y0 * w + x0) * 3;
    let i10 = i00 + 3;
    let i01 = i00 + w * 3;
    let i11 = i01 + 3;
    for c in 0..3 {
        out[c] = frame[i00 + c] as f32 * w00
            + frame[i10 + c] as f32 * w10
            + frame[i01 + c] as f32 * w01
            + frame[i11 + c] as f32 * w11;
    }
}

/// Apply motion blur to one RGB24 frame.
///
/// * `fwd`  — flow from the current frame to the next frame.
/// * `bwd`  — flow from the current frame to the previous frame.
/// * `shutter_deg` — shutter angle in degrees (0..=360).
/// * `samples` — number of taps along the trajectory (quality vs speed).
pub fn motion_blur(
    frame: &[u8],
    w: usize,
    h: usize,
    fwd: &FlowField,
    bwd: &FlowField,
    shutter_deg: f32,
    samples: usize,
) -> Vec<u8> {
    let span = (shutter_deg.clamp(0.0, 360.0) / 360.0) * 0.5;
    let samples = samples.max(2);
    let mut out = vec![0u8; w * h * 3];

    out.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
        let mut acc: [f32; 3];
        let mut tap = [0.0f32; 3];
        for x in 0..w {
            let i = y * w + x;
            let fu = fwd.u[i] * span;
            let fv = fwd.v[i] * span;
            let bu = bwd.u[i] * span;
            let bv = bwd.v[i] * span;

            // Short-circuit stationary pixels: avoids shimmer/noise.
            if fu * fu + fv * fv + bu * bu + bv * bv < 1e-4 {
                let base = i * 3;
                row[x * 3] = frame[base];
                row[x * 3 + 1] = frame[base + 1];
                row[x * 3 + 2] = frame[base + 2];
                continue;
            }

            acc = [0.0; 3];
            for s in 0..samples {
                // t in [-1, 1]: negative taps follow the backward flow,
                // positive taps follow the forward flow.
                let t = (s as f32 + 0.5) / samples as f32 * 2.0 - 1.0;
                let (ox, oy) = if t < 0.0 {
                    (t * bu, t * bv)
                } else {
                    (t * fu, t * fv)
                };
                sample_rgb(frame, w, h, x as f32 + ox, y as f32 + oy, &mut tap);
                acc[0] += tap[0];
                acc[1] += tap[1];
                acc[2] += tap[2];
            }
            let inv = 1.0 / samples as f32;
            row[x * 3] = (acc[0] * inv).round().clamp(0.0, 255.0) as u8;
            row[x * 3 + 1] = (acc[1] * inv).round().clamp(0.0, 255.0) as u8;
            row[x * 3 + 2] = (acc[2] * inv).round().clamp(0.0, 255.0) as u8;
        }
    });
    out
}
