//! Dense optical flow via coarse-to-fine (pyramidal) Lucas-Kanade.
//!
//! For each pixel of image A we estimate the motion vector pointing to the
//! corresponding location in image B. The estimation runs on a Gaussian
//! pyramid: a coarse solution is found at the smallest scale, then refined
//! level by level back up to full resolution. At every level the classic
//! Lucas-Kanade normal equations are solved over a local window, iterated a
//! few times with warping of image B by the current estimate.

use rayon::prelude::*;

/// Single-channel f32 image.
#[derive(Clone)]
pub struct Gray {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

impl Gray {
    pub fn from_rgb(rgb: &[u8], w: usize, h: usize) -> Self {
        let mut data = vec![0.0f32; w * h];
        data.par_iter_mut().enumerate().for_each(|(i, g)| {
            let r = rgb[i * 3] as f32;
            let gg = rgb[i * 3 + 1] as f32;
            let b = rgb[i * 3 + 2] as f32;
            *g = 0.299 * r + 0.587 * gg + 0.114 * b;
        });
        Gray { w, h, data }
    }

    #[inline]
    pub fn at(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.w + x]
    }

    /// Bilinear sample at fractional coordinates, clamped to the border.
    #[inline]
    pub fn sample(&self, x: f32, y: f32) -> f32 {
        let x = x.clamp(0.0, self.w as f32 - 1.001);
        let y = y.clamp(0.0, self.h as f32 - 1.001);
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let fx = x - x0 as f32;
        let fy = y - y0 as f32;
        let i00 = self.at(x0, y0);
        let i10 = self.at(x0 + 1, y0);
        let i01 = self.at(x0, y0 + 1);
        let i11 = self.at(x0 + 1, y0 + 1);
        (i00 * (1.0 - fx) + i10 * fx) * (1.0 - fy) + (i01 * (1.0 - fx) + i11 * fx) * fy
    }
}

/// Per-pixel motion vector field.
#[derive(Clone)]
pub struct FlowField {
    pub w: usize,
    pub h: usize,
    pub u: Vec<f32>, // horizontal component
    pub v: Vec<f32>, // vertical component
}

impl FlowField {
    pub fn zeros(w: usize, h: usize) -> Self {
        FlowField {
            w,
            h,
            u: vec![0.0; w * h],
            v: vec![0.0; w * h],
        }
    }
}

/// 5-tap Gaussian kernel.
const GAUSS: [f32; 5] = [0.0625, 0.25, 0.375, 0.25, 0.0625];

fn blur_horiz(src: &[f32], w: usize, _h: usize, dst: &mut [f32]) {
    dst.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let mut acc = 0.0;
            for k in 0..5isize {
                let xx = (x as isize + k - 2).clamp(0, w as isize - 1) as usize;
                acc += src[y * w + xx] * GAUSS[k as usize];
            }
            row[x] = acc;
        }
    });
}

fn blur_vert(src: &[f32], w: usize, h: usize, dst: &mut [f32]) {
    dst.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let mut acc = 0.0;
            for k in 0..5isize {
                let yy = (y as isize + k - 2).clamp(0, h as isize - 1) as usize;
                acc += src[yy * w + x] * GAUSS[k as usize];
            }
            row[x] = acc;
        }
    });
}

fn gaussian_blur(img: &Gray) -> Gray {
    let mut tmp = vec![0.0f32; img.w * img.h];
    let mut out = vec![0.0f32; img.w * img.h];
    blur_horiz(&img.data, img.w, img.h, &mut tmp);
    blur_vert(&tmp, img.w, img.h, &mut out);
    Gray {
        w: img.w,
        h: img.h,
        data: out,
    }
}

/// Downsample by 2 after Gaussian blur (proper anti-aliased pyramid level).
fn downsample(img: &Gray) -> Gray {
    let blurred = gaussian_blur(img);
    let nw = (img.w / 2).max(1);
    let nh = (img.h / 2).max(1);
    let mut data = vec![0.0f32; nw * nh];
    data.par_iter_mut().enumerate().for_each(|(i, px)| {
        let x = i % nw;
        let y = i / nw;
        *px = blurred.sample(x as f32 * 2.0, y as f32 * 2.0);
    });
    Gray {
        w: nw,
        h: nh,
        data,
    }
}

/// Build a Gaussian pyramid; level 0 is the original image.
pub fn build_pyramid(base: &Gray, levels: usize) -> Vec<Gray> {
    let mut pyr = vec![gaussian_blur(base)];
    for _ in 1..levels {
        let next = downsample(pyr.last().unwrap());
        pyr.push(next);
    }
    pyr
}

/// Pick a pyramid level count so the coarsest level is >= ~24px on its
/// smallest side.
pub fn auto_levels(w: usize, h: usize) -> usize {
    let mut levels = 1;
    let (mut w, mut h) = (w, h);
    while w.min(h) / 2 >= 24 && levels < 6 {
        w /= 2;
        h /= 2;
        levels += 1;
    }
    levels
}

/// Box-filter (sum over a clamped square window) via prefix sums.
/// Used to accumulate the Lucas-Kanade structure tensor over the window.
fn box_filter(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    // Row prefix sums: rp[y][x+1] = sum of src[y][0..=x]
    let mut rp = vec![0.0f32; h * (w + 1)];
    for y in 0..h {
        let mut acc = 0.0;
        for x in 0..w {
            acc += src[y * w + x];
            rp[y * (w + 1) + x + 1] = acc;
        }
    }
    // Horizontal window sums.
    let mut tmp = vec![0.0f32; w * h];
    tmp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let rp_row = &rp[y * (w + 1)..(y + 1) * (w + 1)];
        for x in 0..w {
            let lo = x.saturating_sub(r);
            let hi = (x + r).min(w - 1);
            row[x] = rp_row[hi + 1] - rp_row[lo];
        }
    });
    // Column prefix sums on tmp: cp[x][y+1] = sum of tmp[0..=y][x]
    let mut cp = vec![0.0f32; w * (h + 1)];
    for x in 0..w {
        let mut acc = 0.0;
        for y in 0..h {
            acc += tmp[y * w + x];
            cp[x * (h + 1) + y + 1] = acc;
        }
    }
    let mut dst = vec![0.0f32; w * h];
    dst.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let lo = y.saturating_sub(r);
        let hi = (y + r).min(h - 1);
        for x in 0..w {
            row[x] = cp[x * (h + 1) + hi + 1] - cp[x * (h + 1) + lo];
        }
    });
    dst
}

/// One level of iterative Lucas-Kanade refinement.
///
/// `a` is the reference image, `b` the image to be aligned. `flow` holds the
/// incoming estimate and is updated in place.
fn lk_level(a: &Gray, b: &Gray, flow: &mut FlowField, win_r: usize, iters: usize) {
    let (w, h) = (a.w, a.h);
    // Spatial gradients of the reference image (central differences).
    let mut ix = vec![0.0f32; w * h];
    let mut iy = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let xm = x.saturating_sub(1);
            let xp = (x + 1).min(w - 1);
            let ym = y.saturating_sub(1);
            let yp = (y + 1).min(h - 1);
            ix[y * w + x] = 0.5 * (a.at(xp, y) - a.at(xm, y));
            iy[y * w + x] = 0.5 * (a.at(x, yp) - a.at(x, ym));
        }
    }

    // Structure tensor terms that do not change between iterations.
    let ixx = box_filter(&ix.iter().map(|v| v * v).collect::<Vec<_>>(), w, h, win_r);
    let iyy = box_filter(&iy.iter().map(|v| v * v).collect::<Vec<_>>(), w, h, win_r);
    let ixy = box_filter(
        &ix.iter().zip(&iy).map(|(a, b)| a * b).collect::<Vec<_>>(),
        w,
        h,
        win_r,
    );

    for _ in 0..iters {
        // Temporal gradient: warped B minus A.
        let mut ixt = vec![0.0f32; w * h];
        let mut iyt = vec![0.0f32; w * h];
        ixt.par_chunks_mut(w)
            .zip(iyt.par_chunks_mut(w))
            .enumerate()
            .for_each(|(y, (row_xt, row_yt))| {
                for x in 0..w {
                    let i = y * w + x;
                    let wb = b.sample(x as f32 + flow.u[i], y as f32 + flow.v[i]);
                    let it = wb - a.at(x, y);
                    row_xt[x] = ix[i] * it;
                    row_yt[x] = iy[i] * it;
                }
            });
        let sxt = box_filter(&ixt, w, h, win_r);
        let syt = box_filter(&iyt, w, h, win_r);

        // Solve the 2x2 normal equations per pixel, with Tikhonov
        // regularisation to stay stable in flat regions.
        flow.u
            .par_chunks_mut(w)
            .zip(flow.v.par_chunks_mut(w))
            .enumerate()
            .for_each(|(y, (row_u, row_v))| {
                for x in 0..w {
                    let i = y * w + x;
                    let a00 = ixx[i] + 1e-3;
                    let a11 = iyy[i] + 1e-3;
                    let a01 = ixy[i];
                    let det = a00 * a11 - a01 * a01;
                    if det.abs() < 1e-9 {
                        continue;
                    }
                    let bx = -sxt[i];
                    let by = -syt[i];
                    let du = (a11 * bx - a01 * by) / det;
                    let dv = (-a01 * bx + a00 * by) / det;
                    // Clamp the update to a sane range to avoid explosions.
                    row_u[x] += du.clamp(-5.0, 5.0);
                    row_v[x] += dv.clamp(-5.0, 5.0);
                }
            });
    }
}

/// Upscale a flow field by a factor of two (values are also doubled).
fn upscale_flow(flow: &FlowField, w: usize, h: usize) -> FlowField {
    let mut u = vec![0.0f32; w * h];
    let mut v = vec![0.0f32; w * h];
    u.par_iter_mut()
        .zip(v.par_iter_mut())
        .enumerate()
        .for_each(|(i, (uu, vv))| {
            let x = i % w;
            let y = i / w;
            let sx = (x as f32 * 0.5).min(flow.w as f32 - 1.001);
            let sy = (y as f32 * 0.5).min(flow.h as f32 - 1.001);
            let x0 = sx.floor() as usize;
            let y0 = sy.floor() as usize;
            let fx = sx - x0 as f32;
            let fy = sy - y0 as f32;
            let idx = |xx: usize, yy: usize| yy * flow.w + xx;
            let x1 = (x0 + 1).min(flow.w - 1);
            let y1 = (y0 + 1).min(flow.h - 1);
            let lerp = |f: &Vec<f32>| {
                (f[idx(x0, y0)] * (1.0 - fx) + f[idx(x1, y0)] * fx) * (1.0 - fy)
                    + (f[idx(x0, y1)] * (1.0 - fx) + f[idx(x1, y1)] * fx) * fy
            };
            *uu = lerp(&flow.u) * 2.0;
            *vv = lerp(&flow.v) * 2.0;
        });
    FlowField { w, h, u, v }
}

/// 3x3 median filter on each flow component — kills outliers while keeping
/// motion boundaries reasonably sharp.
pub fn median_flow(flow: &FlowField) -> FlowField {
    let filter = |src: &Vec<f32>| -> Vec<f32> {
        let (w, h) = (flow.w, flow.h);
        let mut out = vec![0.0f32; w * h];
        out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            let mut win = [0.0f32; 9];
            for x in 0..w {
                let mut n = 0;
                for dy in -1isize..=1 {
                    for dx in -1isize..=1 {
                        let xx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                        let yy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                        win[n] = src[yy * w + xx];
                        n += 1;
                    }
                }
                win.sort_by(|a, b| a.partial_cmp(b).unwrap());
                row[x] = win[4];
            }
        });
        out
    };
    FlowField {
        w: flow.w,
        h: flow.h,
        u: filter(&flow.u),
        v: filter(&flow.v),
    }
}

/// Full pyramidal Lucas-Kanade: returns the motion of every pixel of `a`
/// toward image `b`, in pixels at full resolution.
pub fn optical_flow(a: &Gray, b: &Gray, win_r: usize, iters: usize) -> FlowField {
    let levels = auto_levels(a.w, a.h);
    let pa = build_pyramid(a, levels);
    let pb = build_pyramid(b, levels);

    let coarse = pa.last().unwrap();
    let mut flow = FlowField::zeros(coarse.w, coarse.h);

    for lvl in (0..levels).rev() {
        if lvl == levels - 1 {
            lk_level(&pa[lvl], &pb[lvl], &mut flow, win_r, iters);
        } else {
            flow = upscale_flow(&flow, pa[lvl].w, pa[lvl].h);
            lk_level(&pa[lvl], &pb[lvl], &mut flow, win_r, iters);
        }
    }
    flow
}
