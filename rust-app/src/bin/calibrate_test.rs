//! Milestone 2 acceptance check: sample corner patches from a clip, run
//! calibration, print the detected bbox (to diff against the Python
//! prototype's logged `Watermark box: ...` line) and dump the soft mask as a
//! PNG for visual comparison against the Python version's `debug_mask.png`.

use std::path::PathBuf;

use anyhow::{Context, Result};
use watermark_remover::calibrate::{self, Corner};
use watermark_remover::pipeline::sample_corner_patches;
use watermark_remover::video_io;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let input: PathBuf = args
        .next()
        .context("usage: calibrate_test <input> [max_samples]")?
        .into();
    let max_samples: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(90);

    let info = video_io::probe(&input)?;
    println!(
        "probed: {}x{} @ {:.3}fps, nb_frames={:?}",
        info.width, info.height, info.fps, info.nb_frames
    );

    let (x0, y0, qw, qh) = calibrate::quadrant_bounds(info.width, info.height, Corner::BottomRight, 0.25, 0.30);
    println!("quadrant: x0={x0} y0={y0} qw={qw} qh={qh}");
    let corner_pt = calibrate::corner_point(Corner::BottomRight, qw, qh);

    let patches = sample_corner_patches(&input, &info, x0, y0, qw, qh, max_samples)?;
    println!("sampled {} patches", patches.len());

    match calibrate::calibrate(&patches, corner_pt) {
        Some(cal) => {
            let (lx1, ly1, lx2, ly2) = cal.bbox;
            let (bw, bh) = (lx2 - lx1, ly2 - ly1);
            println!("Watermark box: x={} y={} w={} h={}", x0 + lx1, y0 + ly1, bw, bh);
            let mask_u8 = image::ImageBuffer::from_fn(bw, bh, |x, y| {
                image::Luma([(cal.mask.get_pixel(x, y).0[0] * 255.0) as u8])
            });
            mask_u8.save("rust_debug_mask.png").context("failed to save mask preview")?;
            println!("wrote rust_debug_mask.png");
        }
        None => {
            println!("Could not auto-detect a static watermark in that region.");
        }
    }

    Ok(())
}
