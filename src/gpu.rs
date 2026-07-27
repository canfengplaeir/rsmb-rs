//! GPU-accelerated optical flow and motion blur via wgpu compute shaders.
//! Supports NVIDIA, AMD, and Intel iGPU (Vulkan / DX12 / Metal backends).
//!
//! Performance notes:
//! * All intermediate tensors stay on the GPU. The only host readbacks are
//!   the two final flow components, once per frame.
//! * Dispatches are recorded into one command encoder per pipeline stage and
//!   submitted together, minimising per-submit overhead.
//! * The current frame's Gaussian pyramid is built once per frame and shared
//!   by the forward and backward flow solves.

use anyhow::{Context, Result};
use rayon::prelude::*;
use wgpu::util::DeviceExt;

// ---------------------------------------------------------------------------
// WGSL compute shaders (single module, multiple entry points)
// ---------------------------------------------------------------------------

const ALL_SHADERS: &str = r#"
struct Dims {
    width: u32,
    height: u32,
}

struct BoxParams {
    width: u32,
    height: u32,
    radius: u32,
}

struct WarpParams {
    width: u32,
    height: u32,
}

struct SolveParams {
    width: u32,
    height: u32,
}

struct UpscaleParams {
    dst_w: u32,
    dst_h: u32,
    src_w: u32,
    src_h: u32,
}

// Same 4-field layout as UpscaleParams, reused for downsample.
struct DsParams {
    dst_w: u32,
    dst_h: u32,
    src_w: u32,
    src_h: u32,
}

// ---- gaussian_h (5-tap horizontal) -------------------------------------
@group(0) @binding(0) var<storage, read> src_h: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst_h: array<f32>;
@group(0) @binding(2) var<uniform> dims_h: Dims;

@compute @workgroup_size(16, 16)
fn gaussian_h(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= dims_h.width || y >= dims_h.height { return; }
    let w = dims_h.width;
    let base = y * w;
    let x0 = max(x, 2u) - 2u;
    let x1 = max(x, 1u) - 1u;
    let x3 = min(x + 1u, w - 1u);
    let x4 = min(x + 2u, w - 1u);
    dst_h[base + x] =
        0.0625 * src_h[base + x0] +
        0.25   * src_h[base + x1] +
        0.375  * src_h[base + x]  +
        0.25   * src_h[base + x3] +
        0.0625 * src_h[base + x4];
}

// ---- gaussian_v (5-tap vertical) ---------------------------------------
@group(0) @binding(0) var<storage, read> gauss_src_v: array<f32>;
@group(0) @binding(1) var<storage, read_write> gauss_dst_v: array<f32>;
@group(0) @binding(2) var<uniform> dims_v: Dims;

@compute @workgroup_size(16, 16)
fn gaussian_v(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= dims_v.width || y >= dims_v.height { return; }
    let w = dims_v.width;
    let h = dims_v.height;
    let y0 = (max(y, 2u) - 2u) * w;
    let y1 = (max(y, 1u) - 1u) * w;
    let yc = y * w;
    let y3 = min(y + 1u, h - 1u) * w;
    let y4 = min(y + 2u, h - 1u) * w;
    gauss_dst_v[yc + x] =
        0.0625 * gauss_src_v[y0 + x] +
        0.25   * gauss_src_v[y1 + x] +
        0.375  * gauss_src_v[yc + x] +
        0.25   * gauss_src_v[y3 + x] +
        0.0625 * gauss_src_v[y4 + x];
}

// ---- downsample_2x (2x2 box average) -----------------------------------
// NOTE: src_w/src_h carry the *actual* source dimensions, so odd source
// sizes are handled correctly (row stride is src_w, never 2*dst_w).
@group(0) @binding(0) var<storage, read> src_ds: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst_ds: array<f32>;
@group(0) @binding(2) var<uniform> dp: DsParams;

@compute @workgroup_size(16, 16)
fn downsample_2x(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= dp.dst_w || y >= dp.dst_h { return; }
    let sw = dp.src_w;
    let sx0 = min(x * 2u, sw - 1u);
    let sy0 = min(y * 2u, dp.src_h - 1u);
    let sx1 = min(sx0 + 1u, sw - 1u);
    let sy1 = min(sy0 + 1u, dp.src_h - 1u);
    let v00 = src_ds[sy0 * sw + sx0];
    let v10 = src_ds[sy0 * sw + sx1];
    let v01 = src_ds[sy1 * sw + sx0];
    let v11 = src_ds[sy1 * sw + sx1];
    dst_ds[y * dp.dst_w + x] = 0.25 * (v00 + v10 + v01 + v11);
}

// ---- gradients (ix, iy) ------------------------------------------------
@group(0) @binding(0) var<storage, read> img_a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_ix: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_iy: array<f32>;
@group(0) @binding(3) var<uniform> dims_g: Dims;

@compute @workgroup_size(16, 16)
fn gradients(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let w = dims_g.width;
    let h = dims_g.height;
    if x >= w || y >= h { return; }
    let i = y * w + x;
    let xm = max(x, 1u) - 1u;
    let xp = min(x + 1u, w - 1u);
    let ym = max(y, 1u) - 1u;
    let yp = min(y + 1u, h - 1u);
    out_ix[i] = 0.5 * (img_a[xp + y * w] - img_a[xm + y * w]);
    out_iy[i] = 0.5 * (img_a[x + yp * w] - img_a[x + ym * w]);
}

// ---- products (ix^2, iy^2, ix*iy) --------------------------------------
// Keeps the structure-tensor products on the GPU — no host readback.
@group(0) @binding(0) var<storage, read> in_ix: array<f32>;
@group(0) @binding(1) var<storage, read> in_iy: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_xx: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_yy: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_xy: array<f32>;
@group(0) @binding(5) var<uniform> dims_p: Dims;

@compute @workgroup_size(16, 16)
fn products(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= dims_p.width || y >= dims_p.height { return; }
    let i = y * dims_p.width + x;
    let gx = in_ix[i];
    let gy = in_iy[i];
    out_xx[i] = gx * gx;
    out_yy[i] = gy * gy;
    out_xy[i] = gx * gy;
}

// ---- box_filter_3 (horizontal, 3-array combined) -----------------------
@group(0) @binding(0) var<storage, read> in0: array<f32>;
@group(0) @binding(1) var<storage, read> in1: array<f32>;
@group(0) @binding(2) var<storage, read> in2: array<f32>;
@group(0) @binding(3) var<storage, read_write> out0: array<f32>;
@group(0) @binding(4) var<storage, read_write> out1: array<f32>;
@group(0) @binding(5) var<storage, read_write> out2: array<f32>;
@group(0) @binding(6) var<uniform> bp3: BoxParams;

@compute @workgroup_size(16, 16)
fn box3_h(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let w = bp3.width;
    if x >= w || y >= bp3.height { return; }
    let r = bp3.radius;
    let lo = max(x, r) - r;
    let hi = min(x + r, w - 1u);
    var s0 = 0.0; var s1 = 0.0; var s2 = 0.0;
    for (var xi = lo; xi <= hi; xi += 1u) {
        let si = y * w + xi;
        s0 += in0[si]; s1 += in1[si]; s2 += in2[si];
    }
    let idx = y * w + x;
    out0[idx] = s0; out1[idx] = s1; out2[idx] = s2;
}

// ---- box_filter_3 (vertical) -------------------------------------------
@group(0) @binding(0) var<storage, read> in0_v: array<f32>;
@group(0) @binding(1) var<storage, read> in1_v: array<f32>;
@group(0) @binding(2) var<storage, read> in2_v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out0_v: array<f32>;
@group(0) @binding(4) var<storage, read_write> out1_v: array<f32>;
@group(0) @binding(5) var<storage, read_write> out2_v: array<f32>;
@group(0) @binding(6) var<uniform> bp3v: BoxParams;

@compute @workgroup_size(16, 16)
fn box3_v(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let w = bp3v.width;
    let h = bp3v.height;
    if x >= w || y >= h { return; }
    let r = bp3v.radius;
    let lo = max(y, r) - r;
    let hi = min(y + r, h - 1u);
    var s0 = 0.0; var s1 = 0.0; var s2 = 0.0;
    for (var yi = lo; yi <= hi; yi += 1u) {
        let si = yi * w + x;
        s0 += in0_v[si]; s1 += in1_v[si]; s2 += in2_v[si];
    }
    let idx = y * w + x;
    out0_v[idx] = s0; out1_v[idx] = s1; out2_v[idx] = s2;
}

// ---- box_filter_2 (horizontal, 2-array combined) -----------------------
@group(0) @binding(0) var<storage, read> in0_2h: array<f32>;
@group(0) @binding(1) var<storage, read> in1_2h: array<f32>;
@group(0) @binding(2) var<storage, read_write> out0_2h: array<f32>;
@group(0) @binding(3) var<storage, read_write> out1_2h: array<f32>;
@group(0) @binding(4) var<uniform> bp2h: BoxParams;

@compute @workgroup_size(16, 16)
fn box2_h(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let w = bp2h.width;
    if x >= w || y >= bp2h.height { return; }
    let r = bp2h.radius;
    let lo = max(x, r) - r;
    let hi = min(x + r, w - 1u);
    var s0 = 0.0; var s1 = 0.0;
    for (var xi = lo; xi <= hi; xi += 1u) {
        let si = y * w + xi;
        s0 += in0_2h[si]; s1 += in1_2h[si];
    }
    let idx = y * w + x;
    out0_2h[idx] = s0; out1_2h[idx] = s1;
}

// ---- box_filter_2 (vertical) -------------------------------------------
@group(0) @binding(0) var<storage, read> in0_2v: array<f32>;
@group(0) @binding(1) var<storage, read> in1_2v: array<f32>;
@group(0) @binding(2) var<storage, read_write> out0_2v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out1_2v: array<f32>;
@group(0) @binding(4) var<uniform> bp2v: BoxParams;

@compute @workgroup_size(16, 16)
fn box2_v(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let w = bp2v.width;
    let h = bp2v.height;
    if x >= w || y >= h { return; }
    let r = bp2v.radius;
    let lo = max(y, r) - r;
    let hi = min(y + r, h - 1u);
    var s0 = 0.0; var s1 = 0.0;
    for (var yi = lo; yi <= hi; yi += 1u) {
        let si = yi * w + x;
        s0 += in0_2v[si]; s1 += in1_2v[si];
    }
    let idx = y * w + x;
    out0_2v[idx] = s0; out1_2v[idx] = s1;
}

// ---- warp_diff ---------------------------------------------------------
@group(0) @binding(0) var<storage, read> img_a_warp: array<f32>;
@group(0) @binding(1) var<storage, read> img_b: array<f32>;
@group(0) @binding(2) var<storage, read> flow_u: array<f32>;
@group(0) @binding(3) var<storage, read> flow_v: array<f32>;
@group(0) @binding(4) var<storage, read> ix: array<f32>;
@group(0) @binding(5) var<storage, read> iy: array<f32>;
@group(0) @binding(6) var<storage, read_write> out_xt: array<f32>;
@group(0) @binding(7) var<storage, read_write> out_yt: array<f32>;
@group(0) @binding(8) var<uniform> wp: WarpParams;

@compute @workgroup_size(16, 16)
fn warp_diff(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let w = wp.width;
    let h = wp.height;
    if x >= w || y >= h { return; }
    let i = y * w + x;

    // inline bilinear sample of img_b at warped position
    let px = f32(x) + flow_u[i];
    let py = f32(y) + flow_v[i];
    let cx = clamp(px, 0.0, f32(w) - 1.001);
    let cy = clamp(py, 0.0, f32(h) - 1.001);
    let x0 = u32(floor(cx));
    let y0 = u32(floor(cy));
    let fx = cx - f32(x0);
    let fy = cy - f32(y0);
    let x1 = min(x0 + 1u, w - 1u);
    let y1 = min(y0 + 1u, h - 1u);
    let wb = (img_b[y0 * w + x0] * (1.0 - fx) + img_b[y0 * w + x1] * fx) * (1.0 - fy)
           + (img_b[y1 * w + x0] * (1.0 - fx) + img_b[y1 * w + x1] * fx) * fy;

    let it = wb - img_a_warp[i];
    out_xt[i] = ix[i] * it;
    out_yt[i] = iy[i] * it;
}

// ---- solve_update ------------------------------------------------------
@group(0) @binding(0) var<storage, read> ixx: array<f32>;
@group(0) @binding(1) var<storage, read> iyy: array<f32>;
@group(0) @binding(2) var<storage, read> ixy: array<f32>;
@group(0) @binding(3) var<storage, read> sxt: array<f32>;
@group(0) @binding(4) var<storage, read> syt: array<f32>;
@group(0) @binding(5) var<storage, read_write> u: array<f32>;
@group(0) @binding(6) var<storage, read_write> v: array<f32>;
@group(0) @binding(7) var<uniform> sp: SolveParams;

@compute @workgroup_size(16, 16)
fn solve_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let w = sp.width;
    if x >= w || y >= sp.height { return; }
    let i = y * w + x;
    let a00 = ixx[i] + 1e-3;
    let a11 = iyy[i] + 1e-3;
    let a01 = ixy[i];
    let det = a00 * a11 - a01 * a01;
    if abs(det) < 1e-9 { return; }
    let bx = -sxt[i];
    let by = -syt[i];
    let du = clamp((a11 * bx - a01 * by) / det, -5.0, 5.0);
    let dv = clamp((-a01 * bx + a00 * by) / det, -5.0, 5.0);
    u[i] += du;
    v[i] += dv;
}

// ---- median_3x3 --------------------------------------------------------
@group(0) @binding(0) var<storage, read> src_m: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst_m: array<f32>;
@group(0) @binding(2) var<uniform> dims_m: Dims;

@compute @workgroup_size(16, 16)
fn median_3x3(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let w = dims_m.width;
    let h = dims_m.height;
    if x >= w || y >= h { return; }
    var win: array<f32, 9>;
    var n: u32 = 0u;
    for (var dy: i32 = -1; dy <= 1; dy += 1) {
        for (var dx: i32 = -1; dx <= 1; dx += 1) {
            let xx = clamp(i32(x) + dx, 0, i32(w) - 1);
            let yy = clamp(i32(y) + dy, 0, i32(h) - 1);
            win[n] = src_m[u32(yy) * w + u32(xx)];
            n += 1u;
        }
    }
    // 4 bubble passes leave the median (5th of 9) at index 4.
    for (var i = 0u; i < 4u; i += 1u) {
        for (var j = 0u; j < 8u - i; j += 1u) {
            if win[j] > win[j + 1u] {
                let tmp = win[j];
                win[j] = win[j + 1u];
                win[j + 1u] = tmp;
            }
        }
    }
    dst_m[y * w + x] = win[4];
}

// ---- upscale_flow_2x ---------------------------------------------------
@group(0) @binding(0) var<storage, read> src_u: array<f32>;
@group(0) @binding(1) var<storage, read> src_v: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst_u: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst_v: array<f32>;
@group(0) @binding(4) var<uniform> up: UpscaleParams;

@compute @workgroup_size(16, 16)
fn upscale_flow_2x(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= up.dst_w || y >= up.dst_h { return; }
    let sx = f32(x) * 0.5;
    let sy = f32(y) * 0.5;
    // inline bilinear interpolation
    let cx = clamp(sx, 0.0, f32(up.src_w) - 1.001);
    let cy = clamp(sy, 0.0, f32(up.src_h) - 1.001);
    let x0 = u32(floor(cx));
    let y0 = u32(floor(cy));
    let tx = cx - f32(x0);
    let ty = cy - f32(y0);
    let x1 = min(x0 + 1u, up.src_w - 1u);
    let y1 = min(y0 + 1u, up.src_h - 1u);
    let su = (src_u[y0 * up.src_w + x0] * (1.0 - tx) + src_u[y0 * up.src_w + x1] * tx) * (1.0 - ty)
           + (src_u[y1 * up.src_w + x0] * (1.0 - tx) + src_u[y1 * up.src_w + x1] * tx) * ty;
    let sv = (src_v[y0 * up.src_w + x0] * (1.0 - tx) + src_v[y0 * up.src_w + x1] * tx) * (1.0 - ty)
           + (src_v[y1 * up.src_w + x0] * (1.0 - tx) + src_v[y1 * up.src_w + x1] * tx) * ty;
    dst_u[y * up.dst_w + x] = su * 2.0;
    dst_v[y * up.dst_w + x] = sv * 2.0;
}

// ---- negate_flow (u,v) -> (-u,-v) ---------------------------------------
// Used by flow-cache mode to derive the backward flow from the cached
// forward flow of the previous frame, entirely on the GPU.
@group(0) @binding(0) var<storage, read> ng_u_in: array<f32>;
@group(0) @binding(1) var<storage, read> ng_v_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> ng_u_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> ng_v_out: array<f32>;
@group(0) @binding(4) var<uniform> ng_dims: Dims;

@compute @workgroup_size(16, 16)
fn negate_flow(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= ng_dims.width || y >= ng_dims.height { return; }
    let i = y * ng_dims.width + x;
    ng_u_out[i] = -ng_u_in[i];
    ng_v_out[i] = -ng_v_in[i];
}

// ---- motion_blur (packed RGB u32 along flow trajectories) ---------------
struct BlurParams {
    width: u32,
    height: u32,
    samples: u32,
    shutter: f32,
}

@group(0) @binding(0) var<storage, read> frame_px: array<u32>;
@group(0) @binding(1) var<storage, read> bf_u: array<f32>;
@group(0) @binding(2) var<storage, read> bf_v: array<f32>;
@group(0) @binding(3) var<storage, read> bb_u: array<f32>;
@group(0) @binding(4) var<storage, read> bb_v: array<f32>;
@group(0) @binding(5) var<storage, read_write> out_px: array<u32>;
@group(0) @binding(6) var<uniform> blur_p: BlurParams;

fn px_unpack(p: u32) -> vec3<f32> {
    return vec3<f32>(f32(p & 255u), f32((p >> 8u) & 255u), f32((p >> 16u) & 255u));
}

fn px_pack(c: vec3<f32>) -> u32 {
    let r = u32(clamp(c.x + 0.5, 0.0, 255.0));
    let g = u32(clamp(c.y + 0.5, 0.0, 255.0));
    let b = u32(clamp(c.z + 0.5, 0.0, 255.0));
    return r | (g << 8u) | (b << 16u);
}

fn px_bilin(x: f32, y: f32, w: u32, h: u32) -> vec3<f32> {
    let cx = clamp(x, 0.0, f32(w) - 1.001);
    let cy = clamp(y, 0.0, f32(h) - 1.001);
    let x0 = u32(floor(cx));
    let y0 = u32(floor(cy));
    let fx = cx - f32(x0);
    let fy = cy - f32(y0);
    let x1 = min(x0 + 1u, w - 1u);
    let y1 = min(y0 + 1u, h - 1u);
    let c00 = px_unpack(frame_px[y0 * w + x0]);
    let c10 = px_unpack(frame_px[y0 * w + x1]);
    let c01 = px_unpack(frame_px[y1 * w + x0]);
    let c11 = px_unpack(frame_px[y1 * w + x1]);
    return (c00 * (1.0 - fx) + c10 * fx) * (1.0 - fy)
         + (c01 * (1.0 - fx) + c11 * fx) * fy;
}

@compute @workgroup_size(16, 16)
fn motion_blur_px(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= blur_p.width || y >= blur_p.height { return; }
    let w = blur_p.width;
    let h = blur_p.height;
    let i = y * w + x;

    let span = clamp(blur_p.shutter, 0.0, 360.0) / 360.0 * 0.5;
    let fux = bf_u[i] * span;
    let fvx = bf_v[i] * span;
    let bux = bb_u[i] * span;
    let bvx = bb_v[i] * span;

    // stationary pixel short-circuit (same threshold as the CPU version)
    if fux * fux + fvx * fvx + bux * bux + bvx * bvx < 1e-4 {
        out_px[i] = frame_px[i];
        return;
    }

    let n = max(blur_p.samples, 2u);
    var acc = vec3<f32>(0.0, 0.0, 0.0);
    for (var s = 0u; s < n; s += 1u) {
        let t = (f32(s) + 0.5) / f32(n) * 2.0 - 1.0;
        var ox = 0.0;
        var oy = 0.0;
        if t < 0.0 {
            ox = t * bux;
            oy = t * bvx;
        } else {
            ox = t * fux;
            oy = t * fvx;
        }
        acc += px_bilin(f32(x) + ox, f32(y) + oy, w, h);
    }
    out_px[i] = px_pack(acc / f32(n));
}
"#;

// ---------------------------------------------------------------------------
// Uniform structs (must be Pod + Zeroable for bytemuck)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DimsUniform {
    width: u32,
    height: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BoxUniform {
    width: u32,
    height: u32,
    radius: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct FourDimUniform {
    a: u32,
    b: u32,
    c: u32,
    d: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BlurUniform {
    width: u32,
    height: u32,
    samples: u32,
    shutter: f32,
}

// ---------------------------------------------------------------------------
// GpuContext
// ---------------------------------------------------------------------------

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    bgl_simple: wgpu::BindGroupLayout,
    bgl_downsample: wgpu::BindGroupLayout,
    bgl_gradients: wgpu::BindGroupLayout,
    bgl_products: wgpu::BindGroupLayout,
    bgl_box3: wgpu::BindGroupLayout,
    bgl_box2: wgpu::BindGroupLayout,
    bgl_warp: wgpu::BindGroupLayout,
    bgl_solve: wgpu::BindGroupLayout,
    bgl_median: wgpu::BindGroupLayout,
    bgl_upscale: wgpu::BindGroupLayout,
    bgl_negate: wgpu::BindGroupLayout,
    bgl_blur: wgpu::BindGroupLayout,
    pipe_gaussian_h: wgpu::ComputePipeline,
    pipe_gaussian_v: wgpu::ComputePipeline,
    pipe_downsample: wgpu::ComputePipeline,
    pipe_gradients: wgpu::ComputePipeline,
    pipe_products: wgpu::ComputePipeline,
    pipe_box3_h: wgpu::ComputePipeline,
    pipe_box3_v: wgpu::ComputePipeline,
    pipe_box2_h: wgpu::ComputePipeline,
    pipe_box2_v: wgpu::ComputePipeline,
    pipe_warp_diff: wgpu::ComputePipeline,
    pipe_solve_update: wgpu::ComputePipeline,
    pipe_median: wgpu::ComputePipeline,
    pipe_upscale_flow: wgpu::ComputePipeline,
    pipe_negate: wgpu::ComputePipeline,
    pipe_blur: wgpu::ComputePipeline,
    // Keeps the bind-group-layout entry arrays alive (layouts borrow them).
    _bgl_entries: Vec<Vec<wgpu::BindGroupLayoutEntry>>,
}

impl GpuContext {
    pub fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY | wgpu::Backends::GL,
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            },
        ))
        .context("no suitable GPU adapter found -- try updating graphics drivers")?;

        let info = adapter.get_info();
        eprintln!(
            "GPU  : {} ({:?})",
            info.name,
            info.device_type
        );

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("rsmb-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        ))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rsmb-shaders"),
            source: wgpu::ShaderSource::Wgsl(ALL_SHADERS.into()),
        });

        // -- bind group layouts (entries kept alive in `_bgl_entries`) -----
        let mut keeper: Vec<Vec<wgpu::BindGroupLayoutEntry>> = Vec::new();
        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let mut bgl = |device: &wgpu::Device, label: &str, entries: Vec<wgpu::BindGroupLayoutEntry>| {
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries: &entries,
            });
            keeper.push(entries);
            layout
        };

        let bgl_simple = bgl(&device, "bgl-simple", vec![storage(0, true), storage(1, false), uniform(2)]);
        let bgl_downsample = bgl(&device, "bgl-downsample", vec![storage(0, true), storage(1, false), uniform(2)]);
        let bgl_gradients = bgl(&device, "bgl-gradients", vec![storage(0, true), storage(1, false), storage(2, false), uniform(3)]);
        let bgl_products = bgl(&device, "bgl-products", vec![
            storage(0, true), storage(1, true),
            storage(2, false), storage(3, false), storage(4, false),
            uniform(5),
        ]);
        let bgl_box3 = bgl(&device, "bgl-box3", vec![
            storage(0, true), storage(1, true), storage(2, true),
            storage(3, false), storage(4, false), storage(5, false),
            uniform(6),
        ]);
        let bgl_box2 = bgl(&device, "bgl-box2", vec![
            storage(0, true), storage(1, true),
            storage(2, false), storage(3, false),
            uniform(4),
        ]);
        let bgl_warp = bgl(&device, "bgl-warp", vec![
            storage(0, true), storage(1, true), storage(2, true), storage(3, true),
            storage(4, true), storage(5, true),
            storage(6, false), storage(7, false),
            uniform(8),
        ]);
        let bgl_solve = bgl(&device, "bgl-solve", vec![
            storage(0, true), storage(1, true), storage(2, true),
            storage(3, true), storage(4, true),
            storage(5, false), storage(6, false),
            uniform(7),
        ]);
        let bgl_median = bgl(&device, "bgl-median", vec![storage(0, true), storage(1, false), uniform(2)]);
        let bgl_upscale = bgl(&device, "bgl-upscale", vec![
            storage(0, true), storage(1, true),
            storage(2, false), storage(3, false),
            uniform(4),
        ]);
        let bgl_negate = bgl(&device, "bgl-negate", vec![
            storage(0, true), storage(1, true),
            storage(2, false), storage(3, false),
            uniform(4),
        ]);
        let bgl_blur = bgl(&device, "bgl-blur", vec![
            storage(0, true), storage(1, true), storage(2, true),
            storage(3, true), storage(4, true),
            storage(5, false),
            uniform(6),
        ]);

        let make_pipe = |label: &str, layout: &wgpu::BindGroupLayout, entry: &str| {
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[layout],
                push_constant_ranges: &[],
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                module: &shader,
                entry_point: entry,
            })
        };

        Ok(Self {
            pipe_gaussian_h: make_pipe("gaussian_h", &bgl_simple, "gaussian_h"),
            pipe_gaussian_v: make_pipe("gaussian_v", &bgl_simple, "gaussian_v"),
            pipe_downsample: make_pipe("downsample", &bgl_downsample, "downsample_2x"),
            pipe_gradients: make_pipe("gradients", &bgl_gradients, "gradients"),
            pipe_products: make_pipe("products", &bgl_products, "products"),
            pipe_box3_h: make_pipe("box3_h", &bgl_box3, "box3_h"),
            pipe_box3_v: make_pipe("box3_v", &bgl_box3, "box3_v"),
            pipe_box2_h: make_pipe("box2_h", &bgl_box2, "box2_h"),
            pipe_box2_v: make_pipe("box2_v", &bgl_box2, "box2_v"),
            pipe_warp_diff: make_pipe("warp_diff", &bgl_warp, "warp_diff"),
            pipe_solve_update: make_pipe("solve_update", &bgl_solve, "solve_update"),
            pipe_median: make_pipe("median", &bgl_median, "median_3x3"),
            pipe_upscale_flow: make_pipe("upscale_flow", &bgl_upscale, "upscale_flow_2x"),
            pipe_negate: make_pipe("negate", &bgl_negate, "negate_flow"),
            pipe_blur: make_pipe("blur", &bgl_blur, "motion_blur_px"),
            bgl_simple,
            bgl_downsample,
            bgl_gradients,
            bgl_products,
            bgl_box3,
            bgl_box2,
            bgl_warp,
            bgl_solve,
            bgl_median,
            bgl_upscale,
            bgl_negate,
            bgl_blur,
            _bgl_entries: keeper,
            device,
            queue,
        })
    }

    // -- small helpers -----------------------------------------------------

    fn storage_buf(&self, n_floats: usize, label: &str) -> wgpu::Buffer {
        let size = (n_floats.max(1) * std::mem::size_of::<f32>()) as u64;
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn upload_buf(&self, data: &[f32], label: &str) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        })
    }

    fn uniform_buf<T: bytemuck::Pod>(&self, v: &T, label: &str) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(v),
            usage: wgpu::BufferUsages::UNIFORM,
        })
    }

    fn bind_group(
        &self,
        layout: &wgpu::BindGroupLayout,
        buffers: &[&wgpu::Buffer],
        label: &str,
    ) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &entries,
        })
    }

    fn dispatch<'p>(pass: &mut wgpu::ComputePass<'p>, pipe: &'p wgpu::ComputePipeline, bg: &'p wgpu::BindGroup, w: usize, h: usize) {
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, bg, &[]);
        let gx = ((w as u32) + 15) / 16;
        let gy = ((h as u32) + 15) / 16;
        pass.dispatch_workgroups(gx.max(1), gy.max(1), 1);
    }

    /// Read back `n_bytes` from `buf` (full pipeline sync — only used once
    /// per frame, for the final blurred RGB image).
    fn read_buf_bytes(&self, buf: &wgpu::Buffer, n_bytes: usize) -> Vec<u8> {
        let size = n_bytes as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback"),
        });
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let out = data.to_vec();
        drop(data);
        staging.unmap();
        out
    }

    // -- pipeline stages (record into an encoder; caller submits) ----------

    fn enc_gaussian(&self, enc: &mut wgpu::CommandEncoder, src: &wgpu::Buffer, tmp: &wgpu::Buffer, dst: &wgpu::Buffer, w: usize, h: usize) {
        let dims = self.uniform_buf(&DimsUniform { width: w as u32, height: h as u32 }, "dims");
        {
            let bg = self.bind_group(&self.bgl_simple, &[src, tmp, &dims], "gauss-h");
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("gauss-h"), timestamp_writes: None });
            Self::dispatch(&mut pass, &self.pipe_gaussian_h, &bg, w, h);
        }
        {
            let bg = self.bind_group(&self.bgl_simple, &[tmp, dst, &dims], "gauss-v");
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("gauss-v"), timestamp_writes: None });
            Self::dispatch(&mut pass, &self.pipe_gaussian_v, &bg, w, h);
        }
    }

    fn enc_downsample(&self, enc: &mut wgpu::CommandEncoder, src: &wgpu::Buffer, dst: &wgpu::Buffer, sw: usize, sh: usize, nw: usize, nh: usize) {
        let dims = self.uniform_buf(
            &FourDimUniform { a: nw as u32, b: nh as u32, c: sw as u32, d: sh as u32 },
            "ds-dims",
        );
        let bg = self.bind_group(&self.bgl_downsample, &[src, dst, &dims], "ds");
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("downsample"), timestamp_writes: None });
        Self::dispatch(&mut pass, &self.pipe_downsample, &bg, nw, nh);
    }

    fn enc_gradients(&self, enc: &mut wgpu::CommandEncoder, a: &wgpu::Buffer, ix: &wgpu::Buffer, iy: &wgpu::Buffer, w: usize, h: usize) {
        let dims = self.uniform_buf(&DimsUniform { width: w as u32, height: h as u32 }, "dims-g");
        let bg = self.bind_group(&self.bgl_gradients, &[a, ix, iy, &dims], "grad");
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("grad"), timestamp_writes: None });
        Self::dispatch(&mut pass, &self.pipe_gradients, &bg, w, h);
    }

    fn enc_products(&self, enc: &mut wgpu::CommandEncoder, ix: &wgpu::Buffer, iy: &wgpu::Buffer, xx: &wgpu::Buffer, yy: &wgpu::Buffer, xy: &wgpu::Buffer, w: usize, h: usize) {
        let dims = self.uniform_buf(&DimsUniform { width: w as u32, height: h as u32 }, "dims-p");
        let bg = self.bind_group(&self.bgl_products, &[ix, iy, xx, yy, xy, &dims], "prod");
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("prod"), timestamp_writes: None });
        Self::dispatch(&mut pass, &self.pipe_products, &bg, w, h);
    }

    fn enc_box3(&self, enc: &mut wgpu::CommandEncoder, in_: [&wgpu::Buffer; 3], out: [&wgpu::Buffer; 3], tmp: [&wgpu::Buffer; 3], w: usize, h: usize, r: usize) {
        let params = self.uniform_buf(
            &BoxUniform { width: w as u32, height: h as u32, radius: r as u32, _pad: 0 },
            "bp3",
        );
        {
            let bg = self.bind_group(&self.bgl_box3, &[in_[0], in_[1], in_[2], tmp[0], tmp[1], tmp[2], &params], "box3h");
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("box3h"), timestamp_writes: None });
            Self::dispatch(&mut pass, &self.pipe_box3_h, &bg, w, h);
        }
        {
            let bg = self.bind_group(&self.bgl_box3, &[tmp[0], tmp[1], tmp[2], out[0], out[1], out[2], &params], "box3v");
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("box3v"), timestamp_writes: None });
            Self::dispatch(&mut pass, &self.pipe_box3_v, &bg, w, h);
        }
    }

    fn enc_box2(&self, enc: &mut wgpu::CommandEncoder, in0: &wgpu::Buffer, in1: &wgpu::Buffer, out0: &wgpu::Buffer, out1: &wgpu::Buffer, tmp0: &wgpu::Buffer, tmp1: &wgpu::Buffer, w: usize, h: usize, r: usize) {
        let params = self.uniform_buf(
            &BoxUniform { width: w as u32, height: h as u32, radius: r as u32, _pad: 0 },
            "bp2",
        );
        {
            let bg = self.bind_group(&self.bgl_box2, &[in0, in1, tmp0, tmp1, &params], "box2h");
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("box2h"), timestamp_writes: None });
            Self::dispatch(&mut pass, &self.pipe_box2_h, &bg, w, h);
        }
        {
            let bg = self.bind_group(&self.bgl_box2, &[tmp0, tmp1, out0, out1, &params], "box2v");
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("box2v"), timestamp_writes: None });
            Self::dispatch(&mut pass, &self.pipe_box2_v, &bg, w, h);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn enc_warp(&self, enc: &mut wgpu::CommandEncoder, a: &wgpu::Buffer, b: &wgpu::Buffer, u: &wgpu::Buffer, v: &wgpu::Buffer, ix: &wgpu::Buffer, iy: &wgpu::Buffer, xt: &wgpu::Buffer, yt: &wgpu::Buffer, w: usize, h: usize) {
        let dims = self.uniform_buf(&DimsUniform { width: w as u32, height: h as u32 }, "wp");
        let bg = self.bind_group(&self.bgl_warp, &[a, b, u, v, ix, iy, xt, yt, &dims], "warp");
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("warp"), timestamp_writes: None });
        Self::dispatch(&mut pass, &self.pipe_warp_diff, &bg, w, h);
    }

    #[allow(clippy::too_many_arguments)]
    fn enc_solve(&self, enc: &mut wgpu::CommandEncoder, ixx: &wgpu::Buffer, iyy: &wgpu::Buffer, ixy: &wgpu::Buffer, sxt: &wgpu::Buffer, syt: &wgpu::Buffer, u: &wgpu::Buffer, v: &wgpu::Buffer, w: usize, h: usize) {
        let dims = self.uniform_buf(&DimsUniform { width: w as u32, height: h as u32 }, "sp");
        let bg = self.bind_group(&self.bgl_solve, &[ixx, iyy, ixy, sxt, syt, u, v, &dims], "solve");
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("solve"), timestamp_writes: None });
        Self::dispatch(&mut pass, &self.pipe_solve_update, &bg, w, h);
    }

    fn enc_median(&self, enc: &mut wgpu::CommandEncoder, src: &wgpu::Buffer, dst: &wgpu::Buffer, w: usize, h: usize) {
        let dims = self.uniform_buf(&DimsUniform { width: w as u32, height: h as u32 }, "dims-m");
        let bg = self.bind_group(&self.bgl_median, &[src, dst, &dims], "median");
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("median"), timestamp_writes: None });
        Self::dispatch(&mut pass, &self.pipe_median, &bg, w, h);
    }

    #[allow(clippy::too_many_arguments)]
    fn enc_upscale_flow(&self, enc: &mut wgpu::CommandEncoder, su: &wgpu::Buffer, sv: &wgpu::Buffer, du: &wgpu::Buffer, dv: &wgpu::Buffer, tw: usize, th: usize, sw: usize, sh: usize) {
        let dims = self.uniform_buf(
            &FourDimUniform { a: tw as u32, b: th as u32, c: sw as u32, d: sh as u32 },
            "up",
        );
        let bg = self.bind_group(&self.bgl_upscale, &[su, sv, du, dv, &dims], "upscale");
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("upscale"), timestamp_writes: None });
        Self::dispatch(&mut pass, &self.pipe_upscale_flow, &bg, tw, th);
    }

    fn enc_negate(&self, enc: &mut wgpu::CommandEncoder, su: &wgpu::Buffer, sv: &wgpu::Buffer, du: &wgpu::Buffer, dv: &wgpu::Buffer, w: usize, h: usize) {
        let dims = self.uniform_buf(&DimsUniform { width: w as u32, height: h as u32 }, "ng-dims");
        let bg = self.bind_group(&self.bgl_negate, &[su, sv, du, dv, &dims], "negate");
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("negate"), timestamp_writes: None });
        Self::dispatch(&mut pass, &self.pipe_negate, &bg, w, h);
    }

    #[allow(clippy::too_many_arguments)]
    fn enc_blur(&self, enc: &mut wgpu::CommandEncoder, frame: &wgpu::Buffer, flows: &GpuFrameFlows, out: &wgpu::Buffer, w: usize, h: usize, shutter: f32, samples: usize) {
        let params = self.uniform_buf(
            &BlurUniform {
                width: w as u32,
                height: h as u32,
                samples: samples as u32,
                shutter,
            },
            "blur-p",
        );
        let bg = self.bind_group(
            &self.bgl_blur,
            &[
                frame,
                flows.fwd.u.as_ref(),
                flows.fwd.v.as_ref(),
                flows.bwd.u.as_ref(),
                flows.bwd.v.as_ref(),
                out,
                &params,
            ],
            "blur",
        );
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("blur"), timestamp_writes: None });
        Self::dispatch(&mut pass, &self.pipe_blur, &bg, w, h);
    }
}

// ---------------------------------------------------------------------------
// Host-side orchestration
// ---------------------------------------------------------------------------

/// Gaussian pyramid living entirely on the GPU.
struct GpuPyramid {
    bufs: Vec<wgpu::Buffer>,
    dims: Vec<(usize, usize)>,
}

fn pyramid_level_count(w: usize, h: usize, levels: usize) -> usize {
    if levels > 0 {
        return levels.min(6);
    }
    let mut lv = 1;
    let (mut w, mut h) = (w, h);
    while w.min(h) / 2 >= 24 && lv < 6 {
        w /= 2;
        h /= 2;
        lv += 1;
    }
    lv
}

/// Build a Gaussian pyramid on the GPU in a single submission.
fn gpu_build_pyramid(ctx: &GpuContext, gray: &[f32], w: usize, h: usize, levels: usize) -> Result<GpuPyramid> {
    let levels = pyramid_level_count(w, h, levels);

    let base = ctx.upload_buf(gray, "pyr-base");
    let mut bufs = Vec::with_capacity(levels);
    let mut dims = Vec::with_capacity(levels);
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("pyramid"),
    });

    // Level 0: blur the base image.
    let n0 = w * h;
    let blurred = ctx.storage_buf(n0, "pyr-0");
    let tmp = ctx.storage_buf(n0, "pyr-tmp");
    ctx.enc_gaussian(&mut encoder, &base, &tmp, &blurred, w, h);
    bufs.push(blurred);
    dims.push((w, h));

    // Coarser levels: 2x2 box downsample, then blur (both stay on-GPU).
    let (mut cw, mut ch) = (w, h);
    for i in 1..levels {
        let (nw, nh) = ((cw / 2).max(1), (ch / 2).max(1));
        let down = ctx.storage_buf(nw * nh, "pyr-down");
        ctx.enc_downsample(&mut encoder, bufs.last().unwrap(), &down, cw, ch, nw, nh);
        let out = ctx.storage_buf(nw * nh, "pyr-lvl");
        let tmp2 = ctx.storage_buf(nw * nh, "pyr-tmp2");
        ctx.enc_gaussian(&mut encoder, &down, &tmp2, &out, nw, nh);
        bufs.push(out);
        dims.push((nw, nh));
        cw = nw;
        ch = nh;
        let _ = i;
    }

    ctx.queue.submit(Some(encoder.finish()));
    Ok(GpuPyramid { bufs, dims })
}

/// One LK level: gradients, products, tensor accumulation, then `iters`
/// warp/box/solve iterations — all recorded into a single submission.
fn gpu_lk_level(
    ctx: &GpuContext,
    a_buf: &wgpu::Buffer,
    b_buf: &wgpu::Buffer,
    w: usize,
    h: usize,
    u_buf: &wgpu::Buffer,
    v_buf: &wgpu::Buffer,
    win_r: usize,
    iters: usize,
) -> Result<()> {
    let n = w * h;
    let ix = ctx.storage_buf(n, "ix");
    let iy = ctx.storage_buf(n, "iy");
    let ix_sq = ctx.storage_buf(n, "ix_sq");
    let iy_sq = ctx.storage_buf(n, "iy_sq");
    let ix_iy = ctx.storage_buf(n, "ix_iy");
    let ixx = ctx.storage_buf(n, "ixx");
    let iyy = ctx.storage_buf(n, "iyy");
    let ixy = ctx.storage_buf(n, "ixy");
    let t0 = ctx.storage_buf(n, "t0");
    let t1 = ctx.storage_buf(n, "t1");
    let t2 = ctx.storage_buf(n, "t2");
    let ixt = ctx.storage_buf(n, "ixt");
    let iyt = ctx.storage_buf(n, "iyt");
    let sxt = ctx.storage_buf(n, "sxt");
    let syt = ctx.storage_buf(n, "syt");
    let u0 = ctx.storage_buf(n, "u0");
    let u1 = ctx.storage_buf(n, "u1");

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("lk-level"),
    });

    ctx.enc_gradients(&mut encoder, a_buf, &ix, &iy, w, h);
    ctx.enc_products(&mut encoder, &ix, &iy, &ix_sq, &iy_sq, &ix_iy, w, h);
    ctx.enc_box3(&mut encoder, [&ix_sq, &iy_sq, &ix_iy], [&ixx, &iyy, &ixy], [&t0, &t1, &t2], w, h, win_r);

    for _ in 0..iters {
        // warp_diff writes Ix*It and Iy*It directly — no host round-trip.
        ctx.enc_warp(&mut encoder, a_buf, b_buf, u_buf, v_buf, &ix, &iy, &ixt, &iyt, w, h);
        ctx.enc_box2(&mut encoder, &ixt, &iyt, &sxt, &syt, &u0, &u1, w, h, win_r);
        ctx.enc_solve(&mut encoder, &ixx, &iyy, &ixy, &sxt, &syt, u_buf, v_buf, w, h);
    }

    ctx.queue.submit(Some(encoder.finish()));
    Ok(())
}

/// Solve the flow field across the pyramid; returns GPU buffers (u, v) at
/// full resolution (level-0 dimensions).
fn gpu_solve_flow(
    ctx: &GpuContext,
    pa: &GpuPyramid,
    pb: &GpuPyramid,
    win_r: usize,
    iters: usize,
    levels: usize,
) -> Result<(wgpu::Buffer, wgpu::Buffer)> {
    let levels = if levels == 0 {
        pa.bufs.len()
    } else {
        levels.min(pa.bufs.len())
    };
    let (mut cw, mut ch) = pa.dims[levels - 1];
    let mut u = ctx.storage_buf(cw * ch, "u-init");
    let mut v = ctx.storage_buf(cw * ch, "v-init");
    // Zero-initialise the flow at the coarsest level.
    ctx.queue.write_buffer(&u, 0, &vec![0u8; cw * ch * 4]);
    ctx.queue.write_buffer(&v, 0, &vec![0u8; cw * ch * 4]);

    for lvl in (0..levels).rev() {
        if lvl != levels - 1 {
            let (tw, th) = pa.dims[lvl];
            let u2 = ctx.storage_buf(tw * th, "u-up");
            let v2 = ctx.storage_buf(tw * th, "v-up");
            let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("upscale"),
            });
            ctx.enc_upscale_flow(&mut encoder, &u, &v, &u2, &v2, tw, th, cw, ch);
            ctx.queue.submit(Some(encoder.finish()));
            u = u2;
            v = v2;
            cw = tw;
            ch = th;
        }
        gpu_lk_level(ctx, &pa.bufs[lvl], &pb.bufs[lvl], cw, ch, &u, &v, win_r, iters)?;
    }
    Ok((u, v))
}

/// Median-filter both flow components on the GPU; results stay on-device.
fn median_on_device(ctx: &GpuContext, u: &wgpu::Buffer, v: &wgpu::Buffer, w: usize, h: usize) -> GpuFlowPair {
    let n = w * h;
    let mu = ctx.storage_buf(n, "mu");
    let mv = ctx.storage_buf(n, "mv");
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("median"),
    });
    ctx.enc_median(&mut encoder, u, &mu, w, h);
    ctx.enc_median(&mut encoder, v, &mv, w, h);
    ctx.queue.submit(Some(encoder.finish()));
    GpuFlowPair {
        u: std::sync::Arc::new(mu),
        v: std::sync::Arc::new(mv),
    }
}

/// A flow field (u, v) living on the GPU. Cloning is cheap (Arc handle).
#[derive(Clone)]
pub struct GpuFlowPair {
    pub u: std::sync::Arc<wgpu::Buffer>,
    pub v: std::sync::Arc<wgpu::Buffer>,
}

/// Forward and backward flow for one frame, both on the GPU.
pub struct GpuFrameFlows {
    pub fwd: GpuFlowPair,
    pub bwd: GpuFlowPair,
}

/// Forward (cur→next) and backward (cur→prev) flow, sharing the current
/// frame's pyramid. Everything stays on the GPU — no host readback.
#[allow(clippy::too_many_arguments)]
pub fn gpu_flows_device(
    ctx: &GpuContext,
    gray_cur: &[f32],
    gray_prev: &[f32],
    gray_next: &[f32],
    w: usize,
    h: usize,
    win_r: usize,
    iters: usize,
    levels: usize,
) -> Result<GpuFrameFlows> {
    let pa = gpu_build_pyramid(ctx, gray_cur, w, h, levels)?;
    let pb_prev = gpu_build_pyramid(ctx, gray_prev, w, h, levels)?;
    let pb_next = gpu_build_pyramid(ctx, gray_next, w, h, levels)?;

    let (fu, fv) = gpu_solve_flow(ctx, &pa, &pb_next, win_r, iters, levels)?;
    let (bu, bv) = gpu_solve_flow(ctx, &pa, &pb_prev, win_r, iters, levels)?;

    Ok(GpuFrameFlows {
        fwd: median_on_device(ctx, &fu, &fv, w, h),
        bwd: median_on_device(ctx, &bu, &bv, w, h),
    })
}

/// Single-direction flow (a→b), median-filtered, for flow-cache mode.
#[allow(clippy::too_many_arguments)]
pub fn gpu_flow_single_device(
    ctx: &GpuContext,
    gray_a: &[f32],
    gray_b: &[f32],
    w: usize,
    h: usize,
    win_r: usize,
    iters: usize,
    levels: usize,
) -> Result<GpuFlowPair> {
    let pa = gpu_build_pyramid(ctx, gray_a, w, h, levels)?;
    let pb = gpu_build_pyramid(ctx, gray_b, w, h, levels)?;
    let (u, v) = gpu_solve_flow(ctx, &pa, &pb, win_r, iters, levels)?;
    Ok(median_on_device(ctx, &u, &v, w, h))
}

/// Negate a flow field on the GPU (flow-cache backward derivation).
pub fn gpu_negate_flow(ctx: &GpuContext, flow: &GpuFlowPair, w: usize, h: usize) -> GpuFlowPair {
    let n = w * h;
    let ou = ctx.storage_buf(n, "neg-u");
    let ov = ctx.storage_buf(n, "neg-v");
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("negate"),
    });
    ctx.enc_negate(&mut encoder, flow.u.as_ref(), flow.v.as_ref(), &ou, &ov, w, h);
    ctx.queue.submit(Some(encoder.finish()));
    GpuFlowPair {
        u: std::sync::Arc::new(ou),
        v: std::sync::Arc::new(ov),
    }
}

/// Apply motion blur to an RGB24 frame on the GPU. The only host readback
/// of the whole frame pipeline is the final blurred image.
#[allow(clippy::too_many_arguments)]
pub fn gpu_motion_blur(
    ctx: &GpuContext,
    rgb: &[u8],
    flows: &GpuFrameFlows,
    w: usize,
    h: usize,
    shutter: f32,
    samples: usize,
) -> Result<Vec<u8>> {
    let n = w * h;
    // Pack RGB24 -> u32 (r | g<<8 | b<<16) on the CPU (parallel, ~2 Mpx).
    let mut packed = vec![0u32; n];
    packed.par_iter_mut().enumerate().for_each(|(i, p)| {
        *p = rgb[i * 3] as u32 | (rgb[i * 3 + 1] as u32) << 8 | (rgb[i * 3 + 2] as u32) << 16;
    });
    let frame_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("frame-packed"),
        contents: bytemuck::cast_slice(&packed),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let out_buf = ctx.storage_buf(n, "blur-out");

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("blur"),
    });
    ctx.enc_blur(&mut encoder, &frame_buf, flows, &out_buf, w, h, shutter, samples);
    ctx.queue.submit(Some(encoder.finish()));

    let bytes = ctx.read_buf_bytes(&out_buf, n * 4);
    let px: &[u32] = bytemuck::cast_slice(&bytes);
    let mut rgb_out = vec![0u8; n * 3];
    rgb_out
        .par_chunks_mut(3)
        .enumerate()
        .for_each(|(i, c)| {
            let p = px[i];
            c[0] = (p & 255) as u8;
            c[1] = ((p >> 8) & 255) as u8;
            c[2] = ((p >> 16) & 255) as u8;
        });
    Ok(rgb_out)
}
