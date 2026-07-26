//! Synthetic validation: a white square moving diagonally, blurred with a
//! 360-degree shutter. Expect the square to smear along its motion diagonal.

use std::path::Path;

#[path = "../src/blur.rs"]
mod blur;
#[path = "../src/flow.rs"]
mod flow;

const W: usize = 200;
const H: usize = 120;
const SQ: usize = 24;
const DX: isize = 10; // px per frame
const DY: isize = 6;

fn frame(cx: isize, cy: isize) -> Vec<u8> {
    let mut img = vec![30u8; W * H * 3];
    for y in 0..H {
        for x in 0..W {
            // subtle vertical gradient background so flat regions still have texture
            img[(y * W + x) * 3] = 30 + (y * 40 / H) as u8;
            img[(y * W + x) * 3 + 1] = 30 + (y * 40 / H) as u8;
            img[(y * W + x) * 3 + 2] = 40 + (y * 40 / H) as u8;
        }
    }
    for y in cy..cy + SQ as isize {
        for x in cx..cx + SQ as isize {
            if x >= 0 && y >= 0 && (x as usize) < W && (y as usize) < H {
                let i = (y as usize * W + x as usize) * 3;
                img[i] = 240;
                img[i + 1] = 220;
                img[i + 2] = 60;
            }
        }
    }
    img
}

fn save_png(rgb: &[u8], path: &str) {
    image::save_buffer(
        Path::new(path),
        rgb,
        W as u32,
        H as u32,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();
}

fn main() {
    let f_prev = frame(40 - DX, 30 - DY);
    let f_cur = frame(40, 30);
    let f_next = frame(40 + DX, 30 + DY);

    let g_prev = flow::Gray::from_rgb(&f_prev, W, H);
    let g_cur = flow::Gray::from_rgb(&f_cur, W, H);
    let g_next = flow::Gray::from_rgb(&f_next, W, H);

    let fwd = flow::median_flow(&flow::optical_flow(&g_cur, &g_next, 7, 5, 0));
    let bwd = flow::median_flow(&flow::optical_flow(&g_cur, &g_prev, 7, 5, 0));

    // Sanity: mean flow magnitude near the square should approach (DX, DY).
    let mut su = 0.0;
    let mut sv = 0.0;
    let mut n = 0.0;
    for y in 34..50usize {
        for x in 44..60usize {
            let i = y * W + x;
            su += fwd.u[i];
            sv += fwd.v[i];
            n += 1.0;
        }
    }
    println!(
        "mean fwd flow on square: ({:.2}, {:.2}), expected ~({}, {})",
        su / n,
        sv / n,
        DX,
        DY
    );

    let blurred = blur::motion_blur(&f_cur, W, H, &fwd, &bwd, 360.0, 24);
    save_png(&f_cur, "synth_original.png");
    save_png(&blurred, "synth_blurred.png");
    println!("wrote synth_original.png / synth_blurred.png");
}
