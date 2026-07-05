//! Hand-rolled diffusion inpainting: solves for smooth interior values
//! inside a small masked hole via Gauss-Seidel relaxation on Laplace's
//! equation (each masked pixel converges to the average of its neighbours),
//! with the mask's boundary pixels held fixed as the known values.
//!
//! This exists instead of porting OpenCV's Telea fast-marching algorithm
//! because the masked region here is always small (tens of px) and mostly a
//! soft, flat overlay shape — a plain diffusion fill is much simpler to get
//! right and is expected to be roughly comparable in quality, with the
//! caveat (same one the Python/Telea version already has) that a hard
//! scene edge crossing the mask will come out somewhat blurred rather than
//! perfectly reconstructed.

use image::{GrayImage, Luma, RgbImage};

pub const DEFAULT_ITERATIONS: usize = 400;

/// In-place diffusion fill of `patch` wherever `mask` is non-zero. `mask`
/// must be the same width/height as `patch`.
pub fn diffusion_inpaint(patch: &mut RgbImage, mask: &GrayImage, iterations: usize) {
    let (w, h) = patch.dimensions();
    debug_assert_eq!(mask.dimensions(), (w, h));
    let w = w as usize;
    let h = h as usize;

    let is_masked: Vec<bool> = mask.pixels().map(|p| p.0[0] > 0).collect();
    if !is_masked.iter().any(|&m| m) {
        return; // nothing to fill
    }

    let mut channels: [Vec<f32>; 3] = [
        vec![0f32; w * h],
        vec![0f32; w * h],
        vec![0f32; w * h],
    ];
    for (i, px) in patch.pixels().enumerate() {
        channels[0][i] = px.0[0] as f32;
        channels[1][i] = px.0[1] as f32;
        channels[2][i] = px.0[2] as f32;
    }

    // Seed masked pixels with the mean of unmasked pixels, so relaxation
    // starts from a reasonable guess rather than 0 (faster convergence,
    // and a sane fallback if a channel somehow never converges).
    for c in channels.iter_mut() {
        let (sum, count) = c
            .iter()
            .zip(is_masked.iter())
            .filter(|(_, &m)| !m)
            .fold((0f32, 0u32), |(s, n), (&v, _)| (s + v, n + 1));
        let mean = if count > 0 { sum / count as f32 } else { 128.0 };
        for (v, &m) in c.iter_mut().zip(is_masked.iter()) {
            if m {
                *v = mean;
            }
        }
    }

    for _ in 0..iterations {
        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                if !is_masked[idx] {
                    continue;
                }
                for c in channels.iter_mut() {
                    let mut sum = 0f32;
                    let mut count = 0f32;
                    if x > 0 {
                        sum += c[idx - 1];
                        count += 1.0;
                    }
                    if x + 1 < w {
                        sum += c[idx + 1];
                        count += 1.0;
                    }
                    if y > 0 {
                        sum += c[idx - w];
                        count += 1.0;
                    }
                    if y + 1 < h {
                        sum += c[idx + w];
                        count += 1.0;
                    }
                    c[idx] = sum / count;
                }
            }
        }
    }

    for (i, px) in patch.pixels_mut().enumerate() {
        px.0[0] = channels[0][i].round().clamp(0.0, 255.0) as u8;
        px.0[1] = channels[1][i].round().clamp(0.0, 255.0) as u8;
        px.0[2] = channels[2][i].round().clamp(0.0, 255.0) as u8;
    }
}

/// Binarizes a soft [0,1] mask at `threshold`, then dilates by `dilate_px`
/// — the same shape of operation as the Python prototype's
/// `build_binary_mask`, used to decide which pixels actually get
/// overwritten by inpainting (the soft mask itself is reserved for the
/// final feathered alpha blend).
pub fn binary_mask_from_soft(
    soft_mask: &image::ImageBuffer<Luma<f32>, Vec<f32>>,
    threshold: f32,
    dilate_px: u8,
) -> GrayImage {
    let (w, h) = soft_mask.dimensions();
    let mut binm = GrayImage::new(w, h);
    for (px, sp) in binm.pixels_mut().zip(soft_mask.pixels()) {
        px.0[0] = if sp.0[0] > threshold { 255 } else { 0 };
    }
    if dilate_px > 0 {
        binm = imageproc::morphology::dilate(&binm, imageproc::distance_transform::Norm::L2, dilate_px);
    }
    binm
}
