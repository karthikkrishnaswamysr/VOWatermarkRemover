//! Self-calibration: find the static watermark blob in a stack of
//! same-position corner crops sampled across the clip. Direct port of the
//! validated Python `calibrate()` — see remove_watermark.py in the repo
//! root for the reference behaviour this must match.

use std::collections::HashMap;

use image::{GrayImage, ImageBuffer, Luma, RgbImage};
use imageproc::distance_transform::Norm;
use imageproc::filter::gaussian_blur_f32;
use imageproc::morphology::dilate;
use imageproc::region_labelling::{connected_components, Connectivity};

#[derive(Debug, Clone, Copy)]
pub enum Corner {
    BottomRight,
    BottomLeft,
    TopRight,
    TopLeft,
}

/// Returns (x0, y0, qw, qh): the corner region to search, in absolute frame
/// coordinates. Fraction-based (not fixed pixels) so this scales to any
/// input resolution.
pub fn quadrant_bounds(width: u32, height: u32, corner: Corner, frac_w: f64, frac_h: f64) -> (u32, u32, u32, u32) {
    let qw = (width as f64 * frac_w) as u32;
    let qh = (height as f64 * frac_h) as u32;
    let (x0, y0) = match corner {
        Corner::BottomRight => (width - qw, height - qh),
        Corner::BottomLeft => (0, height - qh),
        Corner::TopRight => (width - qw, 0),
        Corner::TopLeft => (0, 0),
    };
    (x0, y0, qw, qh)
}

pub fn corner_point(corner: Corner, qw: u32, qh: u32) -> (f32, f32) {
    match corner {
        Corner::BottomRight => (qw as f32, qh as f32),
        Corner::BottomLeft => (0.0, qh as f32),
        Corner::TopRight => (qw as f32, 0.0),
        Corner::TopLeft => (0.0, 0.0),
    }
}

pub struct Calibration {
    /// (x1, y1, x2, y2) in patch-local coordinates.
    pub bbox: (u32, u32, u32, u32),
    /// Soft mask, same size as the bbox, values in [0, 1].
    pub mask: ImageBuffer<Luma<f32>, Vec<f32>>,
}

struct LabelStats {
    count: u64,
    sum_x: u64,
    sum_y: u64,
    min_x: u32,
    min_y: u32,
    max_x: u32,
    max_y: u32,
}

/// One linear pass over a connected-components label image, accumulating
/// count/bbox/centroid per label. imageproc's `connected_components` only
/// gives us the label image itself — no OpenCV-style per-label stats — so
/// this replaces `cv2.connectedComponentsWithStats`.
fn label_stats(labels: &ImageBuffer<Luma<u32>, Vec<u32>>) -> HashMap<u32, LabelStats> {
    let mut stats: HashMap<u32, LabelStats> = HashMap::new();
    for (x, y, px) in labels.enumerate_pixels() {
        let label = px.0[0];
        if label == 0 {
            continue; // background
        }
        stats
            .entry(label)
            .and_modify(|s| {
                s.count += 1;
                s.sum_x += x as u64;
                s.sum_y += y as u64;
                s.min_x = s.min_x.min(x);
                s.min_y = s.min_y.min(y);
                s.max_x = s.max_x.max(x);
                s.max_y = s.max_y.max(y);
            })
            .or_insert(LabelStats {
                count: 1,
                sum_x: x as u64,
                sum_y: y as u64,
                min_x: x,
                min_y: y,
                max_x: x,
                max_y: y,
            });
    }
    stats
}

fn rgb_mean_to_gray_f32(patches: &[RgbImage]) -> ImageBuffer<Luma<f32>, Vec<f32>> {
    let (w, h) = patches[0].dimensions();
    let n = patches.len() as f32;
    let mut sum_r = vec![0f32; (w * h) as usize];
    let mut sum_g = vec![0f32; (w * h) as usize];
    let mut sum_b = vec![0f32; (w * h) as usize];
    for patch in patches {
        for (i, px) in patch.pixels().enumerate() {
            sum_r[i] += px.0[0] as f32;
            sum_g[i] += px.0[1] as f32;
            sum_b[i] += px.0[2] as f32;
        }
    }
    let mut gray = ImageBuffer::<Luma<f32>, Vec<f32>>::new(w, h);
    for (i, px) in gray.pixels_mut().enumerate() {
        let r = sum_r[i] / n;
        let g = sum_g[i] / n;
        let b = sum_b[i] / n;
        // Same weights as OpenCV's cv2.cvtColor(..., COLOR_BGR2GRAY).
        px.0[0] = 0.299 * r + 0.587 * g + 0.114 * b;
    }
    gray
}

/// Finds the static watermark blob in a stack of same-position corner
/// patches. Returns `None` if nothing that looks like a static overlay is
/// found (caller should fall back to a manual `--box`, matching the
/// Python CLI's behaviour).
pub fn calibrate(patches: &[RgbImage], corner: (f32, f32)) -> Option<Calibration> {
    const MIN_SIZE: u32 = 12;
    const MAX_AREA_FRAC: f32 = 0.6;
    // Real watermark shapes (a sparkle/diamond, a short text wordmark) are
    // not extremely elongated. Without this, a long, static, mildly-bright
    // scene edge (e.g. a sofa armrest's highlight) can out-score the actual
    // watermark on raw area alone — this happened on a real static-camera
    // interview clip where a 302x20px armrest highlight (aspect 15:1) beat
    // a genuine 72x72px diamond.
    const MAX_ASPECT_RATIO: f32 = 3.5;
    const THRESHOLDS: [f32; 5] = [0.55, 0.45, 0.35, 0.25, 0.15];

    let gray = rgb_mean_to_gray_f32(patches);
    let (w, h) = gray.dimensions();

    let sigma = (w.max(h) as f32) / 8.0;
    let blurred = gaussian_blur_f32(&gray, sigma);

    let mut hp: Vec<f32> = Vec::with_capacity((w * h) as usize);
    let mut hp_max = f32::MIN;
    let mut hp_min = f32::MAX;
    for (g, b) in gray.pixels().zip(blurred.pixels()) {
        let v = g.0[0] - b.0[0];
        hp_max = hp_max.max(v);
        hp_min = hp_min.min(v);
        hp.push(v);
    }
    // A static overlay can be either brighter or darker than its surroundings.
    if hp_max.abs() < hp_min.abs() {
        for v in hp.iter_mut() {
            *v = -*v;
        }
    }
    let mut hp_pos_max = 0f32;
    for v in hp.iter_mut() {
        *v = v.max(0.0);
        hp_pos_max = hp_pos_max.max(*v);
    }
    if hp_pos_max < 1e-3 {
        return None;
    }
    let hp_norm: Vec<f32> = hp.iter().map(|v| v / hp_pos_max).collect();

    let max_area = (w * h) as f32 * MAX_AREA_FRAC;

    for &thresh in THRESHOLDS.iter() {
        let mut binm = GrayImage::new(w, h);
        for (i, px) in binm.pixels_mut().enumerate() {
            px.0[0] = if hp_norm[i] > thresh { 255 } else { 0 };
        }
        let labels = connected_components(&binm, Connectivity::Eight, Luma([0u8]));
        let stats = label_stats(&labels);

        let mut best: Option<(f32, u32, u32, u32, u32, u32)> = None; // score,x,y,bw,bh,label_id
        for (&label_id, s) in stats.iter() {
            let bw = s.max_x - s.min_x + 1;
            let bh = s.max_y - s.min_y + 1;
            if bw < MIN_SIZE || bh < MIN_SIZE {
                continue;
            }
            let area = s.count as f32;
            if area > max_area {
                continue;
            }
            let compactness = area / (bw * bh) as f32;
            if compactness < 0.25 {
                continue;
            }
            let aspect_ratio = bw.max(bh) as f32 / bw.min(bh) as f32;
            if aspect_ratio > MAX_ASPECT_RATIO {
                continue;
            }
            let cx = s.sum_x as f32 / s.count as f32;
            let cy = s.sum_y as f32 / s.count as f32;
            let dist = ((cx - corner.0).powi(2) + (cy - corner.1).powi(2)).sqrt();
            let score = area * compactness / (1.0 + dist);
            if std::env::var_os("WM_DEBUG_CALIBRATE").is_some() {
                eprintln!(
                    "  thresh={thresh} label={label_id} bbox=({},{},{bw}x{bh}) area={area} compactness={compactness:.3} dist={dist:.1} score={score:.1}",
                    s.min_x, s.min_y
                );
            }
            if best.map(|b| score > b.0).unwrap_or(true) {
                best = Some((score, s.min_x, s.min_y, bw, bh, label_id));
            }
        }

        if let Some((_, x, y, bw, bh, label_id)) = best {
            let pad = (0.2 * (bw.max(bh) as f32)).round().max(6.0) as u32;
            let x1 = x.saturating_sub(pad);
            let y1 = y.saturating_sub(pad);
            let x2 = (x + bw + pad).min(w);
            let y2 = (y + bh + pad).min(h);

            // Exactly the connected component validated above — re-thresholding
            // hp_norm at a different level over the *whole quadrant* here would
            // risk pulling in a differently shaped (and halo-contaminated,
            // see below) region.
            let core_w = x2 - x1;
            let core_h = y2 - y1;
            let mut core = GrayImage::new(core_w, core_h);
            for cy in 0..core_h {
                for cx in 0..core_w {
                    let l = labels.get_pixel(x1 + cx, y1 + cy).0[0];
                    core.put_pixel(cx, cy, Luma([if l == label_id { 255 } else { 0 }]));
                }
            }

            // The validated core is correctly *located*, but the threshold
            // that found it (e.g. 0.55) only captures the high-confidence
            // middle of the shape — the anti-aliased rim/tips are dimmer and
            // get missed. That rim scales roughly with the watermark's own
            // size (bigger watermark, bigger soft edge in absolute pixels),
            // so the safety-margin dilation must scale with the detected
            // blob size too, not be a fixed pixel count — a fixed 3px
            // margin was fine at 720p but left a visible translucent
            // residue on a larger 1080p mark.
            //
            // (An earlier attempt recovered this by re-thresholding hp_norm
            // a second time at a low fixed cutoff. That reintroduced the
            // "halo" bug this same function already had to fix once —
            // hp_norm's magnitude only drops close to the true watermark
            // edge when the blur kernel used to compute it is much smaller
            // than the crop, which isn't reliably true here. Plain
            // size-proportional dilation avoids depending on that at all.)
            let dilate_px = ((0.08 * bw.max(bh) as f32).round() as u8).max(3);
            let dilated = dilate(&core, Norm::L2, dilate_px);
            let dilated_f32: ImageBuffer<Luma<f32>, Vec<f32>> = ImageBuffer::from_fn(core_w, core_h, |px, py| {
                Luma([dilated.get_pixel(px, py).0[0] as f32 / 255.0])
            });
            let feathered = gaussian_blur_f32(&dilated_f32, (dilate_px as f32 * 0.5).max(1.5));
            let max_v = feathered.pixels().fold(0f32, |acc, p| acc.max(p.0[0]));
            let mask = ImageBuffer::from_fn(core_w, core_h, |px, py| {
                Luma([(feathered.get_pixel(px, py).0[0] / (max_v + 1e-6)).clamp(0.0, 1.0)])
            });

            return Some(Calibration { bbox: (x1, y1, x2, y2), mask });
        }
    }

    None
}
