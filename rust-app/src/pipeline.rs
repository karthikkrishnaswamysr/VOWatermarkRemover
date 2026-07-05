//! Orchestrates the full removal pipeline: probe -> calibrate (or manual
//! box) -> build mask -> stream every frame through
//! decode -> inpaint the masked patch -> feather-blend -> encode, re-muxing
//! the original audio track.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{bail, Context, Result};
use image::{GrayImage, ImageBuffer, Luma, Rgb, RgbImage};

use crate::calibrate::{self, Corner};
use crate::inpaint;
use crate::video_io::{self, ChildGuard, VideoInfo};

/// How the run ended: either it processed every frame, or a cancellation
/// request (checked once per frame) cut it short. Both are `Ok` outcomes —
/// cancellation is a normal user action, not an error.
#[derive(Debug, Clone, Copy)]
pub enum RunOutcome {
    Completed { frames: u64 },
    Cancelled { frames: u64 },
}

pub struct Options {
    pub corner: Corner,
    pub region_frac: (f64, f64),
    /// Samples per segment (see `segment_seconds`) used for calibration.
    pub samples: usize,
    /// Length of each independently-recalibrated segment, in seconds. A
    /// single video-wide calibration assumes the watermark sits at one
    /// fixed position/size for the whole file — true for a single
    /// continuous shot, but false for an edited multi-clip compilation
    /// where each source clip may have been rendered (and watermarked) at
    /// a slightly different scale/position before being cut together.
    /// Recalibrating every few seconds tracks that instead of silently
    /// under- or over-covering the mark partway through.
    pub segment_seconds: f64,
    /// Manual override: (x1, y1, x2, y2) in absolute frame coordinates,
    /// skips auto-calibration entirely — same escape hatch as the Python
    /// prototype's `--box`.
    pub manual_box: Option<(u32, u32, u32, u32)>,
    pub inpaint_iterations: usize,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            corner: Corner::BottomRight,
            region_frac: (0.25, 0.30),
            samples: 30,
            segment_seconds: 5.0,
            manual_box: None,
            inpaint_iterations: inpaint::DEFAULT_ITERATIONS,
        }
    }
}

/// A resolved, ready-to-use calibration for one segment of the video.
struct SegmentCal {
    bbox_abs: (u32, u32, u32, u32),
    soft_mask: ImageBuffer<Luma<f32>, Vec<f32>>,
    binary_mask: GrayImage,
}

/// One sequential decode pass, bucketing sampled corner-quadrant patches by
/// segment (`frame_index / segment_frames`) so calibrating N segments only
/// costs one decode instead of N.
fn sample_patches_by_segment(
    input: &Path,
    info: &VideoInfo,
    x0: u32,
    y0: u32,
    qw: u32,
    qh: u32,
    segment_frames: u64,
    num_segments: u64,
    samples_per_segment: usize,
) -> Result<Vec<Vec<RgbImage>>> {
    let global_step = (segment_frames / samples_per_segment.max(1) as u64).max(1);

    let mut decoder = ChildGuard::new(video_io::spawn_decoder(input)?);
    let mut decoder_stdout = decoder.get_mut().stdout.take().context("decoder stdout not piped")?;
    let frame_size = (info.width * info.height * 3) as usize;
    let mut buf = vec![0u8; frame_size];

    let mut by_segment: Vec<Vec<RgbImage>> = (0..num_segments).map(|_| Vec::new()).collect();
    let mut idx: u64 = 0;
    loop {
        let got = video_io::read_frame(&mut decoder_stdout, &mut buf)?;
        if !got {
            break;
        }
        if idx % global_step == 0 {
            let seg = (idx / segment_frames).min(num_segments - 1) as usize;
            by_segment[seg].push(extract_patch(&buf, info.width, x0, y0, x0 + qw, y0 + qh));
        }
        idx += 1;
    }
    drop(decoder_stdout);
    let status = decoder.into_inner().wait().context("decoder failed")?;
    if !status.success() {
        bail!("ffmpeg decoder exited with {status} while sampling");
    }
    Ok(by_segment)
}

/// Calibrates every segment, then fills any segment where calibration
/// failed (e.g. too little background variation in that short window) with
/// its nearest successfully-calibrated neighbour, preferring the previous
/// segment (temporal continuity) and falling back to the next one for a
/// failed leading segment. Errors only if every segment failed.
fn calibrate_segments(
    by_segment: Vec<Vec<RgbImage>>,
    corner_pt: (f32, f32),
    quadrant_origin: (u32, u32),
) -> Result<Vec<SegmentCal>> {
    let n = by_segment.len();
    let mut resolved: Vec<Option<SegmentCal>> = Vec::with_capacity(n);
    for patches in &by_segment {
        let cal = if patches.is_empty() {
            None
        } else {
            calibrate::calibrate(patches, corner_pt)
        };
        resolved.push(cal.map(|c| {
            let (lx1, ly1, lx2, ly2) = c.bbox;
            let (x0, y0) = quadrant_origin;
            let bbox_abs = (x0 + lx1, y0 + ly1, x0 + lx2, y0 + ly2);
            let binary_mask = inpaint::binary_mask_from_soft(&c.mask, 0.3, 2);
            SegmentCal { bbox_abs, soft_mask: c.mask, binary_mask }
        }));
    }

    if resolved.iter().all(Option::is_none) {
        bail!(
            "could not auto-detect a static watermark in that region in any segment; \
             try a larger --region-frac, a different --corner, or a manual --box"
        );
    }

    // Forward-fill from the previous segment, then back-fill any still-empty
    // leading segments from the first successful one.
    let mut last_good: Option<usize> = None;
    let mut filled: Vec<Option<SegmentCal>> = (0..n).map(|_| None).collect();
    for i in 0..n {
        if let Some(cal) = resolved[i].take() {
            last_good = Some(i);
            filled[i] = Some(cal);
        } else if let Some(src) = last_good {
            filled[i] = filled[src].as_ref().map(|c| SegmentCal {
                bbox_abs: c.bbox_abs,
                soft_mask: c.soft_mask.clone(),
                binary_mask: c.binary_mask.clone(),
            });
        }
    }
    if filled[0].is_none() {
        let first_good = filled.iter().position(Option::is_some).expect("checked all-None above");
        let src = {
            let c = filled[first_good].as_ref().unwrap();
            SegmentCal { bbox_abs: c.bbox_abs, soft_mask: c.soft_mask.clone(), binary_mask: c.binary_mask.clone() }
        };
        for slot in filled.iter_mut().take(first_good) {
            *slot = Some(SegmentCal {
                bbox_abs: src.bbox_abs,
                soft_mask: src.soft_mask.clone(),
                binary_mask: src.binary_mask.clone(),
            });
        }
    }

    Ok(filled.into_iter().map(|c| c.expect("every segment filled above")).collect())
}

/// Samples up to `max_samples` frames spread across the clip, cropped to
/// the given quadrant. Mirrors the Python prototype's
/// `sample_quadrant_patches`: a step derived from the container's reported
/// frame count (falling back to a sane default if that's unavailable),
/// decoding sequentially and keeping every `step`-th frame's crop.
pub fn sample_corner_patches(
    input: &Path,
    info: &VideoInfo,
    x0: u32,
    y0: u32,
    qw: u32,
    qh: u32,
    max_samples: usize,
) -> Result<Vec<RgbImage>> {
    let total = info.nb_frames.unwrap_or(max_samples as u64 * 4).max(1);
    let step = (total / max_samples as u64).max(1);

    let mut decoder = ChildGuard::new(video_io::spawn_decoder(input)?);
    let mut decoder_stdout = decoder.get_mut().stdout.take().context("decoder stdout not piped")?;
    let frame_size = (info.width * info.height * 3) as usize;
    let mut buf = vec![0u8; frame_size];

    let mut patches = Vec::new();
    let mut idx: u64 = 0;
    loop {
        let got = video_io::read_frame(&mut decoder_stdout, &mut buf)?;
        if !got {
            break;
        }
        if idx % step == 0 {
            patches.push(extract_patch(&buf, info.width, x0, y0, x0 + qw, y0 + qh));
        }
        idx += 1;
    }
    drop(decoder_stdout);
    let status = decoder.into_inner().wait().context("decoder failed")?;
    if !status.success() {
        bail!("ffmpeg decoder exited with {status} while sampling");
    }
    if patches.is_empty() {
        bail!("no frames could be read from {}", input.display());
    }
    Ok(patches)
}

fn extract_patch(frame: &[u8], frame_w: u32, x1: u32, y1: u32, x2: u32, y2: u32) -> RgbImage {
    let bw = x2 - x1;
    let bh = y2 - y1;
    let mut out = RgbImage::new(bw, bh);
    let row_bytes = (bw * 3) as usize;
    for y in 0..bh {
        let src_start = (((y1 + y) * frame_w + x1) * 3) as usize;
        let dst_start = (y * bw * 3) as usize;
        out.as_mut()[dst_start..dst_start + row_bytes]
            .copy_from_slice(&frame[src_start..src_start + row_bytes]);
    }
    out
}

fn write_patch_back(frame: &mut [u8], frame_w: u32, x1: u32, y1: u32, patch: &RgbImage) {
    let bw = patch.width();
    let bh = patch.height();
    let row_bytes = (bw * 3) as usize;
    for y in 0..bh {
        let dst_start = (((y1 + y) * frame_w + x1) * 3) as usize;
        let src_start = (y * bw * 3) as usize;
        frame[dst_start..dst_start + row_bytes]
            .copy_from_slice(&patch.as_raw()[src_start..src_start + row_bytes]);
    }
}

fn blend(
    original: &RgbImage,
    inpainted: &RgbImage,
    mask: &ImageBuffer<Luma<f32>, Vec<f32>>,
) -> RgbImage {
    let (w, h) = original.dimensions();
    let mut out = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let a = mask.get_pixel(x, y).0[0];
            let o = original.get_pixel(x, y).0;
            let ip = inpainted.get_pixel(x, y).0;
            let mut px = [0u8; 3];
            for c in 0..3 {
                let v = a * ip[c] as f32 + (1.0 - a) * o[c] as f32;
                px[c] = v.round().clamp(0.0, 255.0) as u8;
            }
            out.put_pixel(x, y, Rgb(px));
        }
    }
    out
}

fn manual_box_mask(bw: u32, bh: u32) -> ImageBuffer<Luma<f32>, Vec<f32>> {
    // Soft-edged ellipse, matching the Python prototype's manual --box mask.
    let mut mask = ImageBuffer::<Luma<f32>, Vec<f32>>::new(bw, bh);
    let cx = bw as f32 / 2.0;
    let cy = bh as f32 / 2.0;
    let rx = bw as f32 / 2.0;
    let ry = bh as f32 / 2.0;
    for y in 0..bh {
        for x in 0..bw {
            let nx = (x as f32 - cx) / rx.max(1.0);
            let ny = (y as f32 - cy) / ry.max(1.0);
            mask.put_pixel(x, y, Luma([if nx * nx + ny * ny <= 1.0 { 1.0 } else { 0.0 }]));
        }
    }
    let sigma = (bw.min(bh) as f32 / 8.0).max(2.0);
    let blurred = imageproc::filter::gaussian_blur_f32(&mask, sigma);
    let max_v = blurred.pixels().fold(0f32, |acc, p| acc.max(p.0[0]));
    ImageBuffer::from_fn(bw, bh, |x, y| {
        Luma([(blurred.get_pixel(x, y).0[0] / (max_v + 1e-6)).clamp(0.0, 1.0)])
    })
}

pub fn run(
    input: &Path,
    output: &Path,
    opts: &Options,
    cancel: &AtomicBool,
    mut on_progress: impl FnMut(u64, u64),
) -> Result<RunOutcome> {
    let info = video_io::probe(input)?;

    let (segments, segment_frames): (Vec<SegmentCal>, u64) = if let Some((mut x1, mut y1, mut x2, mut y2)) =
        opts.manual_box
    {
        x1 = x1.min(info.width);
        y1 = y1.min(info.height);
        x2 = x2.min(info.width);
        y2 = y2.min(info.height);
        if x2 <= x1 || y2 <= y1 {
            bail!("--box resolves to an empty/invalid region for a {}x{} frame", info.width, info.height);
        }
        let soft_mask = manual_box_mask(x2 - x1, y2 - y1);
        let binary_mask = inpaint::binary_mask_from_soft(&soft_mask, 0.3, 2);
        (vec![SegmentCal { bbox_abs: (x1, y1, x2, y2), soft_mask, binary_mask }], u64::MAX)
    } else {
        let (x0, y0, qw, qh) =
            calibrate::quadrant_bounds(info.width, info.height, opts.corner, opts.region_frac.0, opts.region_frac.1);
        let corner_pt = calibrate::corner_point(opts.corner, qw, qh);

        let segment_frames = match info.nb_frames {
            // No reliable frame count to segment against — fall back to one
            // whole-video calibration (the old behaviour).
            None => u64::MAX,
            Some(_) => ((opts.segment_seconds * info.fps).round() as u64).max(1),
        };
        let num_segments = match info.nb_frames {
            None => 1,
            Some(total) => total.div_ceil(segment_frames).max(1),
        };

        let by_segment =
            sample_patches_by_segment(input, &info, x0, y0, qw, qh, segment_frames, num_segments, opts.samples)?;
        let segments = calibrate_segments(by_segment, corner_pt, (x0, y0))?;
        (segments, segment_frames)
    };

    if segments.len() > 1 {
        eprintln!("Calibrated {} segments ({}s each):", segments.len(), opts.segment_seconds);
        for (i, s) in segments.iter().enumerate() {
            let (x1, y1, x2, y2) = s.bbox_abs;
            eprintln!("  segment {i}: x={x1} y={y1} w={} h={}", x2 - x1, y2 - y1);
        }
    } else {
        let (x1, y1, x2, y2) = segments[0].bbox_abs;
        eprintln!("Watermark box: x={x1} y={y1} w={} h={}", x2 - x1, y2 - y1);
    }

    let mut decoder = ChildGuard::new(video_io::spawn_decoder(input)?);
    let mut encoder = ChildGuard::new(video_io::spawn_encoder(input, output, info)?);
    let mut decoder_stdout = decoder.get_mut().stdout.take().context("decoder stdout not piped")?;
    let mut encoder_stdin = encoder.get_mut().stdin.take().context("encoder stdin not piped")?;

    let frame_size = (info.width * info.height * 3) as usize;
    let mut buf = vec![0u8; frame_size];
    let total_frames = info.nb_frames.unwrap_or(0);

    let mut frame_count: u64 = 0;
    let mut cancelled = false;
    loop {
        if cancel.load(Ordering::Relaxed) {
            cancelled = true;
            break;
        }
        let got = video_io::read_frame(&mut decoder_stdout, &mut buf)?;
        if !got {
            break;
        }

        let seg_idx = ((frame_count / segment_frames) as usize).min(segments.len() - 1);
        let seg = &segments[seg_idx];
        let (x1, y1, x2, y2) = seg.bbox_abs;

        let orig_patch = extract_patch(&buf, info.width, x1, y1, x2, y2);
        let mut work_patch = orig_patch.clone();
        inpaint::diffusion_inpaint(&mut work_patch, &seg.binary_mask, opts.inpaint_iterations);
        let blended = blend(&orig_patch, &work_patch, &seg.soft_mask);
        write_patch_back(&mut buf, info.width, x1, y1, &blended);

        video_io::write_frame(&mut encoder_stdin, &buf)?;
        frame_count += 1;
        on_progress(frame_count, total_frames);
    }

    drop(encoder_stdin);
    drop(decoder_stdout);

    if cancelled {
        // ChildGuard::drop kills+reaps both processes; the partially written
        // output file is left behind for the caller to clean up if desired.
        return Ok(RunOutcome::Cancelled { frames: frame_count });
    }

    let decode_status = decoder.into_inner().wait().context("failed to wait on decoder")?;
    let encode_status = encoder.into_inner().wait().context("failed to wait on encoder")?;
    if !decode_status.success() {
        bail!("ffmpeg decoder exited with {decode_status}");
    }
    if !encode_status.success() {
        bail!("ffmpeg encoder exited with {encode_status}");
    }

    Ok(RunOutcome::Completed { frames: frame_count })
}
