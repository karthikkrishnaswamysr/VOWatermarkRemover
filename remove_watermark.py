#!/usr/bin/env python3
"""
Corner watermark remover for AI-generated video clips (Veo / Gemini "diamond",
Veo "text", and similar static bottom-right overlays such as the "omni" mark).

Instead of relying on a pre-baked template for one specific watermark, this
tool *self-calibrates* against the input clip: it samples frames spread across
the video, looks at a corner region, and finds the one static shape that stays
in the same place while the scene behind it keeps changing. That shape's
footprint becomes the removal mask for every frame. Because the calibration is
data-driven, the same code path handles the Veo text wordmark, the Gemini/omni
diamond, or any other static corner overlay, without per-watermark special
casing.

Removal is done with per-frame spatial inpainting (OpenCV Telea), feathered
back into the frame with the soft shape mask recovered during calibration.
This does not require a "golden" watermark-on/watermark-off reference pair;
it estimates only from the clip you give it.
"""
import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np

CORNERS = ("br", "bl", "tr", "tl")


def resolve_tool(name):
    """Resolves ffmpeg/ffprobe next to this program first (a bundled sidecar
    binary when packaged as a standalone exe), only falling back to PATH for
    dev runs where no sidecar is present. Never trusts a bare name off PATH
    when a bundled copy is available — otherwise a malicious ffmpeg.exe
    placed earlier in PATH (or, in some invocation modes, the working
    directory) could get run instead."""
    exe_name = f"{name}.exe" if os.name == "nt" else name
    base_dir = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(base_dir, exe_name)
    if os.path.isfile(candidate):
        return candidate
    return exe_name


def ffprobe_info(path):
    cmd = [
        resolve_tool("ffprobe"), "-v", "error", "-print_format", "json",
        "-show_entries", "stream=index,codec_type,width,height,r_frame_rate,avg_frame_rate,nb_frames",
        "-show_entries", "format=duration",
        str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    streams = data["streams"]
    vstream = next(s for s in streams if s["codec_type"] == "video")
    has_audio = any(s["codec_type"] == "audio" for s in streams)
    num, den = vstream["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    nb_frames = int(vstream["nb_frames"]) if str(vstream.get("nb_frames", "")).isdigit() else None
    return {
        "width": vstream["width"],
        "height": vstream["height"],
        "fps": fps,
        "has_audio": has_audio,
        "nb_frames": nb_frames,
    }


def quadrant_bounds(width, height, corner, frac_w, frac_h):
    qw = int(width * frac_w)
    qh = int(height * frac_h)
    if corner == "br":
        x0, y0 = width - qw, height - qh
    elif corner == "bl":
        x0, y0 = 0, height - qh
    elif corner == "tr":
        x0, y0 = width - qw, 0
    else:  # tl
        x0, y0 = 0, 0
    return x0, y0, qw, qh


def sample_patches_by_segment(path, x0, y0, qw, qh, segment_frames, num_segments, global_step):
    """One decode pass, bucketing sampled corner-quadrant patches by segment
    (frame_index // segment_frames) so calibrating N segments only costs one
    decode instead of N. `global_step` is computed by the caller (kept
    separate from `segment_frames`, which can be an arbitrarily huge
    "single segment" sentinel when the real frame count is unknown)."""
    cap = cv2.VideoCapture(str(path))
    by_segment = [[] for _ in range(num_segments)]
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % global_step == 0:
            seg = min(idx // segment_frames, num_segments - 1)
            by_segment[seg].append(frame[y0:y0 + qh, x0:x0 + qw].astype(np.float32))
        idx += 1
    cap.release()
    return by_segment


def calibrate(patches, min_size=12, max_size_frac=0.6, max_aspect_ratio=3.5, corner_point=None):
    """Find the static watermark blob in a stack of same-position patches.

    Returns (local_bbox, soft_mask) where local_bbox = (x1,y1,x2,y2) in patch
    coordinates and soft_mask is a float32 HxW array in [0,1] the same size as
    the bbox, giving the watermark's approximate coverage/shape.
    """
    if not patches:
        return None, None
    stack = np.stack(patches, axis=0)
    mean = stack.mean(axis=0)
    gray = cv2.cvtColor(mean.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape

    sigma = max(w, h) / 8.0
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
    hp = gray - blurred
    # a static overlay can be either brighter or darker than its surroundings
    hp_signed = hp if np.abs(hp.max()) >= np.abs(hp.min()) else -hp
    hp_pos = np.clip(hp_signed, 0, None)
    if hp_pos.max() < 1e-3:
        return None, None
    hp_norm = hp_pos / hp_pos.max()

    max_area = w * h * max_size_frac
    best = None
    if corner_point is None:
        corner_point = (w, h)  # bottom-right of patch by default

    for thresh in (0.55, 0.45, 0.35, 0.25, 0.15):
        binm = (hp_norm > thresh).astype(np.uint8)
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(binm, connectivity=8)
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if bw < min_size or bh < min_size:
                continue
            if area > max_area:
                continue
            compactness = area / float(bw * bh)
            if compactness < 0.25:
                continue
            # Real watermark shapes (a sparkle/diamond, a short text wordmark)
            # are not extremely elongated. Without this, a long, static,
            # mildly-bright scene edge (e.g. a sofa armrest's highlight) can
            # out-score the actual watermark on raw area alone.
            aspect_ratio = max(bw, bh) / float(min(bw, bh))
            if aspect_ratio > max_aspect_ratio:
                continue
            cx, cy = centroids[i]
            dist = ((cx - corner_point[0]) ** 2 + (cy - corner_point[1]) ** 2) ** 0.5
            score = area * compactness / (1.0 + dist)
            if best is None or score > best[0]:
                best = (score, x, y, bw, bh, labels, i)
        if best is not None:
            break

    if best is None:
        return None, None

    _, x, y, bw, bh, labels, label_id = best
    pad = max(6, int(0.2 * max(bw, bh)))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + bw + pad)
    y2 = min(h, y + bh + pad)

    # Use exactly the connected component that was validated above (same
    # threshold that found it) as the core shape — re-thresholding hp_norm
    # again at a different, lower level here would pull in a differently
    # shaped (and often clipped-at-the-crop-edge) region instead.
    core = (labels[y1:y2, x1:x2] == label_id).astype(np.uint8) * 255
    # The validated core is correctly *located*, but the threshold that found
    # it (e.g. 0.55) only captures the high-confidence middle of the shape —
    # the anti-aliased rim/tips are dimmer and get missed. That rim scales
    # roughly with the watermark's own size (bigger watermark, bigger soft
    # edge in absolute pixels), so the safety-margin dilation must scale with
    # the detected blob size too — a fixed 3px margin was fine at 720p but
    # left a visible translucent residue on a larger 1080p mark.
    dilate_px = max(3, int(round(0.08 * max(bw, bh))))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
    dilated = cv2.dilate(core, k)
    mask = cv2.GaussianBlur(dilated.astype(np.float32) / 255.0, (0, 0), sigmaX=max(1.5, dilate_px * 0.5))
    mask = np.clip(mask / (mask.max() + 1e-6), 0, 1)

    return (x1, y1, x2, y2), mask.astype(np.float32)


def build_binary_mask(soft_mask, threshold=0.3, dilate_px=2):
    binm = (soft_mask > threshold).astype(np.uint8) * 255
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        binm = cv2.dilate(binm, k)
    return binm


def calibrate_segments(by_segment, corner_point, quadrant_origin):
    """Calibrates every segment, then fills any segment where calibration
    failed (e.g. too little background variation in that short window) with
    its nearest successfully-calibrated neighbour — forward-filled from the
    previous segment (temporal continuity), then back-filled for a failed
    leading run from the first successful one. Raises if every segment
    failed.

    A single video-wide calibration assumes the watermark sits at one fixed
    position/size for the whole file — true for a single continuous shot,
    but false for an edited multi-clip compilation where each source clip
    may have been rendered (and watermarked) at a slightly different
    scale/position before being cut together. Recalibrating every few
    seconds tracks that instead of silently under- or over-covering the
    mark partway through.

    Returns a list of dicts: {bbox_abs, soft_mask, binary_mask}.
    """
    x0, y0 = quadrant_origin
    resolved = []
    for patches in by_segment:
        local_bbox, mask = calibrate(patches, corner_point=corner_point)
        if local_bbox is None:
            resolved.append(None)
            continue
        lx1, ly1, lx2, ly2 = local_bbox
        resolved.append({
            "bbox_abs": (x0 + lx1, y0 + ly1, x0 + lx2, y0 + ly2),
            "soft_mask": mask,
            "binary_mask": build_binary_mask(mask),
            "alpha3": np.dstack([mask] * 3),
        })

    if all(r is None for r in resolved):
        raise RuntimeError(
            "could not auto-detect a static watermark in that region in any segment; "
            "try a larger --region-frac, a different --corner, or a manual --box"
        )

    filled = [None] * len(resolved)
    last_good = None
    for i, r in enumerate(resolved):
        if r is not None:
            filled[i] = r
            last_good = r
        elif last_good is not None:
            filled[i] = last_good
    if filled[0] is None:
        first_good = next(r for r in filled if r is not None)
        for i, r in enumerate(filled):
            if r is None:
                filled[i] = first_good
            else:
                break
    return filled


DEFAULT_INPAINT_METHOD = "telea"


def inpaint_patch(frame, bbox_abs, binary_mask, inpaint_radius, method=DEFAULT_INPAINT_METHOD, context_margin=120):
    """Fills the masked region of `bbox_abs` within `frame`, returning just
    the box-sized result (ready to blend back in with the soft alpha mask).

    method: "telea" / "ns" (OpenCV's classic PDE-based inpainting — fast,
    fine on smooth/plain backgrounds, weak across a hard structured edge
    like a straight line or grain pattern) or "fsr_fast" / "fsr_best"
    (OpenCV-contrib's xphoto self-similarity inpainting — a
    PatchMatch-family algorithm that continues structured patterns across
    the hole far better, at real cost: fsr_fast is roughly 0.5-1s/frame,
    and fsr_best measured at ~19s *per call* in testing — impractical for
    per-frame use, but genuinely practical when it's only run once per
    detected-stable window (see process_video_adaptive) rather than once
    per frame.

    The xphoto methods need much more surrounding context than the tight
    box to have real material to draw an exemplar from — passing them just
    the box (as cv2.inpaint is happy with) starves them and produces
    garbage, confirmed empirically. `context_margin` controls how much
    padding around the box is given as that context.
    """
    x1, y1, x2, y2 = bbox_abs
    if method in ("telea", "ns"):
        cv_method = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
        patch = frame[y1:y2, x1:x2]
        return cv2.inpaint(patch, binary_mask, inpaint_radius, cv_method)

    if method not in ("fsr_fast", "fsr_best"):
        raise ValueError(f"unknown inpaint method: {method!r}")

    h, w = frame.shape[:2]
    px1, py1 = max(0, x1 - context_margin), max(0, y1 - context_margin)
    px2, py2 = min(w, x2 + context_margin), min(h, y2 + context_margin)
    big_patch = frame[py1:py2, px1:px2]
    bx, by = x1 - px1, y1 - py1
    bw, bh = x2 - x1, y2 - y1

    big_mask = np.zeros(big_patch.shape[:2], dtype=np.uint8)
    big_mask[by:by + bh, bx:bx + bw] = binary_mask
    inv_mask = cv2.bitwise_not(big_mask)  # xphoto convention: non-zero = keep, zero = fill

    algo = cv2.xphoto.INPAINT_FSR_BEST if method == "fsr_best" else cv2.xphoto.INPAINT_FSR_FAST
    dst = np.zeros_like(big_patch)
    cv2.xphoto.inpaint(big_patch, inv_mask, dst, algo)
    return dst[by:by + bh, bx:bx + bw]


def manual_box_segment(bbox_abs):
    """Builds a full segment dict (bbox + soft/binary masks) from a plain
    box, using a soft-edged ellipse as the blend mask. Used both for the
    CLI's --box escape hatch and for GUI review boxes (auto-detected or
    user-drawn/adjusted) — one consistent mask recipe regardless of a box's
    origin, so there's a single code path to reason about."""
    x1, y1, x2, y2 = bbox_abs
    bw, bh = x2 - x1, y2 - y1
    mask = np.zeros((bh, bw), dtype=np.float32)
    cv2.ellipse(mask, (bw // 2, bh // 2), (bw // 2, bh // 2), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(2, min(bw, bh) / 8))
    mask = mask / (mask.max() + 1e-6)
    return {
        "bbox_abs": bbox_abs,
        "soft_mask": mask,
        "binary_mask": build_binary_mask(mask),
        "alpha3": np.dstack([mask] * 3),
    }


def grab_segment_preview_frames(path, segment_frames, num_segments):
    """One sequential decode pass, grabbing a single representative frame
    (the midpoint) for each segment — for a GUI review step to show what
    that segment's calibration was run against. Cheaper than seeking
    independently per segment (which would re-decode from the start each
    time)."""
    targets = {min(i * segment_frames + segment_frames // 2, i * segment_frames + segment_frames - 1): i
               for i in range(num_segments)}
    cap = cv2.VideoCapture(str(path))
    frames = [None] * num_segments
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in targets:
            frames[targets[idx]] = frame.copy()
        idx += 1
    cap.release()
    last = None
    for i in range(num_segments):
        if frames[i] is None:
            frames[i] = last
        else:
            last = frames[i]
    return frames


def _padded_bounds(bbox_abs, frame_w, frame_h, margin):
    """The box expanded outward by `margin`, clamped to the frame — this
    padded region is what we decode/store per frame for ring-based change
    detection (small: a watermark box plus a thin border, not the full
    frame)."""
    x1, y1, x2, y2 = bbox_abs
    px1 = max(0, x1 - margin)
    py1 = max(0, y1 - margin)
    px2 = min(frame_w, x2 + margin)
    py2 = min(frame_h, y2 + margin)
    return px1, py1, px2, py2


def ring_signature(padded_patch, box_local):
    """A cheap per-frame feature vector for the ring of always-visible
    pixels surrounding the watermark box: mean BGR of each of the 8 cells
    of a 3x3 grid overlaid on the padded patch, skipping the center cell
    (which is the box itself, never visible). Splitting into cells rather
    than one overall ring average means a change localized to one side
    isn't diluted away by the rest of the ring staying put.
    """
    oh_total, ow_total = padded_patch.shape[:2]
    ox, oy, ow, oh = box_local
    col_bounds = (0, ox, ox + ow, ow_total)
    row_bounds = (0, oy, oy + oh, oh_total)
    feats = []
    for ri in range(3):
        r0, r1 = row_bounds[ri], row_bounds[ri + 1]
        for ci in range(3):
            if ri == 1 and ci == 1:
                continue
            c0, c1 = col_bounds[ci], col_bounds[ci + 1]
            if r1 <= r0 or c1 <= c0:
                feats.extend((0.0, 0.0, 0.0))
                continue
            cell = padded_patch[r0:r1, c0:c1]
            feats.extend(cell.reshape(-1, 3).mean(axis=0).tolist())
    return np.array(feats, dtype=np.float32)


def detect_stable_windows(signatures, threshold=14.0, hysteresis_frames=3, baseline_alpha=0.1,
                           max_window_drift=None):
    """Splits a segment's frames into windows where the surrounding ring
    stayed put (background behind the watermark probably didn't change
    either) versus where it moved on. Returns a list of (start, end)
    index ranges (end exclusive) covering every frame exactly once.

    Two separate triggers force a new window, because they catch different
    failure modes:
    - A *jump*: the ring vs. a rolling exponential-average baseline clears
      `threshold` for `hysteresis_frames` consecutive frames (so one noisy
      frame can't fracture a window or make the fill flip-flop on
      borderline, semi-continuous motion). The rolling baseline is
      deliberately forgiving of slow drift — a lighting fade shouldn't
      itself read as "changed."
    - Cumulative *drift*: the ring vs. the window's own STARTING signature
      exceeds `max_window_drift`. This exists because the rolling
      baseline's whole point — tolerating slow drift — is also its blind
      spot: several seconds of gradual camera/subject movement can each
      individually stay under `threshold` while the true background
      quietly ends up somewhere completely different from the window's
      representative frame, silently going stale with no jump ever
      confirmed. (Verified empirically: a locked-off-looking shot still
      drifted enough over ~9s that the fill was visibly misaligned before
      this check was added.)
    """
    n = len(signatures)
    if n == 0:
        return []
    if max_window_drift is None:
        max_window_drift = threshold * 2.5
    windows = []
    window_start = 0
    window_start_sig = signatures[0]
    baseline = signatures[0].copy()
    consec = 0
    for i in range(1, n):
        diff = float(np.abs(signatures[i] - baseline).mean())
        drift = float(np.abs(signatures[i] - window_start_sig).mean())
        if diff > threshold:
            consec += 1
        else:
            consec = 0
            baseline = baseline_alpha * signatures[i] + (1 - baseline_alpha) * baseline
        if consec >= hysteresis_frames or drift > max_window_drift:
            cut = i - hysteresis_frames + 1 if consec >= hysteresis_frames else i
            if cut > window_start:
                windows.append((window_start, cut))
                window_start = cut
            window_start_sig = signatures[window_start]
            baseline = signatures[window_start].copy()
            consec = 0
    windows.append((window_start, n))
    return windows


def representative_index(signatures, start, end):
    """The frame within [start, end) whose ring signature is closest to
    the window's mean — the most "typical" frame, avoiding a pick right at
    a noisy edge."""
    window = np.stack(signatures[start:end])
    mean_sig = window.mean(axis=0)
    dists = np.abs(window - mean_sig).mean(axis=1)
    return start + int(np.argmin(dists))


def process_video_adaptive(path, output, segments, segment_frames, inpaint_radius, denoise,
                            progress=True, on_progress=None, cancel_event=None,
                            inpaint_method=DEFAULT_INPAINT_METHOD, ring_margin=14,
                            change_threshold=14.0, hysteresis_frames=3, debug=False):
    """Two-pass variant of process_video: within each segment, detects
    stretches of frames where the ring around the watermark box hasn't
    meaningfully changed, inpaints once per stretch (on that stretch's most
    representative frame), and reuses that single fill for every frame in
    it. This removes the frame-to-frame flicker/inconsistency that
    independent per-frame inpainting can show over an otherwise-static
    background, at the cost of one extra decode pass. Wherever the
    surroundings keep genuinely changing (handheld/panning footage),
    windows collapse toward single frames and this degrades to
    (effectively) the same per-frame behaviour as process_video — not
    worse, just no extra benefit there.

    Returns (frame_count, cancelled) — same contract as process_video.
    """
    info = ffprobe_info(path)
    w, h, fps, has_audio = info["width"], info["height"], info["fps"], info["has_audio"]
    total_frames = info.get("nb_frames") or 0
    frame_size = w * h * 3
    ffmpeg = resolve_tool("ffmpeg")

    padded_bounds = [_padded_bounds(seg["bbox_abs"], w, h, ring_margin) for seg in segments]
    box_local = []
    for seg, (px1, py1, px2, py2) in zip(segments, padded_bounds):
        x1, y1, x2, y2 = seg["bbox_abs"]
        box_local.append((x1 - px1, y1 - py1, x2 - x1, y2 - y1))

    # Pass 1: one decode, collecting just the small padded patch per frame
    # (a box plus a thin border — tens of KB, not a full frame) bucketed by
    # segment, in chronological order. Reports progress against a combined
    # 0..2*total_frames range (this pass is the first half) so the caller's
    # progress bar/stall-detector sees continuous movement across both
    # passes instead of looking stuck through all of pass 1.
    patches_by_segment = [[] for _ in segments]
    decode_cmd = [ffmpeg, "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE)
    frame_idx = 0
    pass1_cancelled = False
    try:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                pass1_cancelled = True
                break
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
            seg_i = min(frame_idx // segment_frames, len(segments) - 1)
            px1, py1, px2, py2 = padded_bounds[seg_i]
            patches_by_segment[seg_i].append(frame[py1:py2, px1:px2].copy())
            frame_idx += 1
            if on_progress is not None:
                on_progress(frame_idx, total_frames * 2)
    except BaseException:
        try:
            proc.kill()
        except Exception:
            pass
        raise
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        if pass1_cancelled:
            try:
                proc.kill()
            except Exception:
                pass
        proc.wait()

    if pass1_cancelled:
        return frame_idx, True

    # Per segment: ring signatures -> stable windows -> pick each window's
    # representative (global) frame index. The actual inpainting is
    # deferred until after we've collected every window's representative
    # frame across all segments, because "fsr_fast"/"fsr_best" need far more
    # surrounding context than the small ring patches stored above (a tight
    # box-only patch starves their exemplar search) — cheaper to grab that
    # bigger context in one extra targeted decode pass over just the
    # (typically few) representative frames than to have stored it for
    # every single frame in pass 1.
    windows_by_segment = []
    for seg_idx, seg in enumerate(segments):
        patches = patches_by_segment[seg_idx]
        if not patches:
            windows_by_segment.append([])
            continue
        sigs = [ring_signature(p, box_local[seg_idx]) for p in patches]
        windows = detect_stable_windows(sigs, threshold=change_threshold, hysteresis_frames=hysteresis_frames)
        if debug:
            print(f"segment {seg_idx}: {len(windows)} window(s) over {len(patches)} frames: {windows}",
                  file=sys.stderr)
        seg_windows = []
        for (ws, we) in windows:
            rep_local = representative_index(sigs, ws, we)
            rep_global = seg_idx * segment_frames + rep_local
            seg_windows.append((ws, we, rep_global))
        windows_by_segment.append(seg_windows)

    needed_context = 120 if inpaint_method in ("fsr_fast", "fsr_best") else 0
    rep_frames = {}  # global frame index -> full frame (or None if never reached)
    for seg_windows in windows_by_segment:
        for (_, _, rep_global) in seg_windows:
            rep_frames[rep_global] = None

    if inpaint_method in ("telea", "ns"):
        # Cheap path: the small ring-margin patch already stored is plenty
        # for these — no extra decode needed.
        for seg_idx, seg_windows in enumerate(windows_by_segment):
            bx, by, bw, bh = box_local[seg_idx]
            cv_method = cv2.INPAINT_TELEA if inpaint_method == "telea" else cv2.INPAINT_NS
            binary_mask = segments[seg_idx]["binary_mask"]
            for (_, _, rep_global) in seg_windows:
                rep_local = rep_global - seg_idx * segment_frames
                box_patch = patches_by_segment[seg_idx][rep_local][by:by + bh, bx:bx + bw]
                rep_frames[rep_global] = cv2.inpaint(box_patch, binary_mask, inpaint_radius, cv_method)
    else:
        target_set = set(rep_frames.keys())
        decode_cmd2 = [ffmpeg, "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        # We deliberately stop reading once every representative frame has
        # been grabbed, well before EOF — ffmpeg then hits a broken pipe
        # writing further frames we no longer want, which is expected, not
        # a real error; silence its stderr so that doesn't look alarming.
        proc2 = subprocess.Popen(decode_cmd2, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        idx2 = 0
        try:
            while target_set:
                raw = proc2.stdout.read(frame_size)
                if len(raw) < frame_size:
                    break
                if idx2 in target_set:
                    full_frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
                    seg_idx = min(idx2 // segment_frames, len(segments) - 1)
                    bx1, by1, bx2, by2 = segments[seg_idx]["bbox_abs"]
                    rep_frames[idx2] = inpaint_patch(
                        full_frame, (bx1, by1, bx2, by2), segments[seg_idx]["binary_mask"],
                        inpaint_radius, inpaint_method, context_margin=needed_context,
                    )
                    target_set.discard(idx2)
                idx2 += 1
        finally:
            try:
                proc2.stdout.close()
            except Exception:
                pass
            proc2.wait()

    fills_by_segment = []
    for seg_idx, seg_windows in enumerate(windows_by_segment):
        seg_fills = []
        for (ws, we, rep_global) in seg_windows:
            inpainted = rep_frames[rep_global]
            if inpainted is not None and denoise:
                inpainted = cv2.bilateralFilter(inpainted, d=7, sigmaColor=45, sigmaSpace=45)
            seg_fills.append((ws, we, inpainted))
        fills_by_segment.append(seg_fills)

    def fill_for(seg_i, local_idx):
        for ws, we, filled in fills_by_segment[seg_i]:
            if ws <= local_idx < we:
                return filled
        return fills_by_segment[seg_i][-1][2]  # degenerate fallback, shouldn't happen

    # Pass 2: decode again, blend each frame against its precomputed
    # per-window fill, encode.
    decode_cmd = [ffmpeg, "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    encode_cmd = [
        ffmpeg, "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-i", str(path),
        "-map", "0:v:0",
    ]
    if has_audio:
        encode_cmd += ["-map", "1:a:0?", "-c:a", "copy"]
    encode_cmd += [
        "-c:v", "libx264", "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p",
        str(output),
    ]

    decode_proc = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE)
    encode_proc = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE)

    frame_idx = 0
    cancelled = False
    try:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            raw = decode_proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()

            seg_i = min(frame_idx // segment_frames, len(segments) - 1)
            seg = segments[seg_i]
            x1, y1, x2, y2 = seg["bbox_abs"]
            alpha3 = seg["alpha3"]
            local_idx = frame_idx - seg_i * segment_frames

            inpainted = fill_for(seg_i, local_idx)
            patch = frame[y1:y2, x1:x2]
            blended = (alpha3 * inpainted + (1 - alpha3) * patch).astype(np.uint8)
            frame[y1:y2, x1:x2] = blended

            encode_proc.stdin.write(frame.tobytes())
            frame_idx += 1
            if progress and frame_idx % 24 == 0:
                sys.stderr.write(f"\rprocessed {frame_idx} frames")
                sys.stderr.flush()
            if on_progress is not None:
                on_progress(total_frames + frame_idx, total_frames * 2)
    except BaseException:
        for p in (decode_proc, encode_proc):
            try:
                p.kill()
            except Exception:
                pass
        raise
    finally:
        try:
            decode_proc.stdout.close()
        except Exception:
            pass
        try:
            encode_proc.stdin.close()
        except Exception:
            pass
        if cancelled:
            for p in (decode_proc, encode_proc):
                try:
                    p.kill()
                except Exception:
                    pass
        decode_ret = decode_proc.wait()
        encode_ret = encode_proc.wait()

    if progress:
        sys.stderr.write(f"\rprocessed {frame_idx} frames\n")

    if cancelled:
        return frame_idx, True

    if encode_ret != 0:
        raise RuntimeError(f"ffmpeg encode failed with exit code {encode_ret}")
    if decode_ret != 0:
        raise RuntimeError(f"ffmpeg decode failed with exit code {decode_ret}")

    return frame_idx, False


def process_video(path, output, segments, segment_frames, inpaint_radius, denoise,
                   progress=True, on_progress=None, cancel_event=None, inpaint_method=DEFAULT_INPAINT_METHOD):
    """Runs the full per-frame pipeline. Returns (frame_count, cancelled).

    `on_progress(frame_idx, total_frames)`, if given, is called after every
    frame (in addition to the `progress` stderr ticker, which is CLI-only).
    `cancel_event` (a threading.Event), if given, is checked once per frame;
    when set, both ffmpeg processes are killed immediately rather than left
    to finish or orphaned — same reasoning as the Rust build's ChildGuard.
    """
    info = ffprobe_info(path)
    w, h, fps, has_audio = info["width"], info["height"], info["fps"], info["has_audio"]
    total_frames = info.get("nb_frames") or 0

    frame_size = w * h * 3
    ffmpeg = resolve_tool("ffmpeg")
    decode_cmd = [ffmpeg, "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    encode_cmd = [
        ffmpeg, "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-i", str(path),
        "-map", "0:v:0",
    ]
    if has_audio:
        encode_cmd += ["-map", "1:a:0?", "-c:a", "copy"]
    encode_cmd += [
        "-c:v", "libx264", "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p",
        str(output),
    ]

    decode_proc = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE)
    encode_proc = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE)

    frame_idx = 0
    cancelled = False
    try:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            raw = decode_proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()

            seg = segments[min(frame_idx // segment_frames, len(segments) - 1)]
            x1, y1, x2, y2 = seg["bbox_abs"]
            binary_mask = seg["binary_mask"]
            alpha3 = seg["alpha3"]

            patch = frame[y1:y2, x1:x2]
            inpainted = inpaint_patch(frame, (x1, y1, x2, y2), binary_mask, inpaint_radius, inpaint_method)
            if denoise:
                inpainted = cv2.bilateralFilter(inpainted, d=7, sigmaColor=45, sigmaSpace=45)
            blended = (alpha3 * inpainted + (1 - alpha3) * patch).astype(np.uint8)
            frame[y1:y2, x1:x2] = blended

            encode_proc.stdin.write(frame.tobytes())
            frame_idx += 1
            if progress and frame_idx % 24 == 0:
                sys.stderr.write(f"\rprocessed {frame_idx} frames")
                sys.stderr.flush()
            if on_progress is not None:
                on_progress(frame_idx, total_frames)
    except BaseException:
        # A real error (not a clean cancel) — kill both immediately rather
        # than let a wedged process hang around.
        for p in (decode_proc, encode_proc):
            try:
                p.kill()
            except Exception:
                pass
        raise
    finally:
        try:
            decode_proc.stdout.close()
        except Exception:
            pass
        try:
            encode_proc.stdin.close()
        except Exception:
            pass
        if cancelled:
            # Cancellation cuts the loop short deliberately (not an
            # exception) — kill both rather than let the encoder try to
            # finalize a file we're about to discard anyway.
            for p in (decode_proc, encode_proc):
                try:
                    p.kill()
                except Exception:
                    pass
        decode_ret = decode_proc.wait()
        encode_ret = encode_proc.wait()

    if progress:
        sys.stderr.write(f"\rprocessed {frame_idx} frames\n")

    if cancelled:
        return frame_idx, True

    if encode_ret != 0:
        raise RuntimeError(f"ffmpeg encode failed with exit code {encode_ret}")
    if decode_ret != 0:
        raise RuntimeError(f"ffmpeg decode failed with exit code {decode_ret}")

    return frame_idx, False


def parse_box(s):
    parts = [int(p) for p in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--box must be x,y,w,h")
    x, y, bw, bh = parts
    return (x, y, x + bw, y + bh)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="input video path")
    ap.add_argument("-o", "--output", help="output video path (default: <input>_processed.mp4)")
    ap.add_argument("--corner", choices=CORNERS, default="br", help="which corner the watermark sits in (default: br)")
    ap.add_argument("--region-frac", nargs=2, type=float, metavar=("W_FRAC", "H_FRAC"), default=(0.25, 0.30),
                     help="fraction of frame width/height to search within the chosen corner (default: 0.25 0.30)")
    ap.add_argument("--samples", type=int, default=30, help="number of frames to sample per segment for calibration (default: 30)")
    ap.add_argument("--segment-seconds", type=float, default=5.0,
                     help="recalibrate every this many seconds, so an edited multi-clip video whose watermark "
                          "position/size isn't constant throughout still gets tracked (default: 5.0)")
    ap.add_argument("--box", type=parse_box, default=None,
                     help="skip auto-detection and use an explicit watermark box: x,y,w,h (frame pixel coords)")
    ap.add_argument("--inpaint-radius", type=int, default=5, help="OpenCV inpaint radius (default: 5)")
    ap.add_argument("--inpaint-method", choices=("telea", "ns", "fsr_fast", "fsr_best"), default=DEFAULT_INPAINT_METHOD,
                     help="telea/ns are fast classic inpainting; fsr_fast/fsr_best (opencv-contrib) continue "
                          "structured patterns (straight edges, grain) far better but are much slower — fsr_best "
                          "(~19s/call) is only practical with --adaptive, since that runs it once per detected "
                          "stable window instead of once per frame (default: telea)")
    ap.add_argument("--adaptive", action="store_true",
                     help="two-pass mode: detect stretches of frames where the area around the watermark hasn't "
                          "changed and reuse one fill for the whole stretch, instead of inpainting every frame "
                          "independently — removes flicker on a locked-off/slow-moving camera")
    ap.add_argument("--no-denoise", action="store_true", help="skip the light bilateral smoothing pass on the patched region")
    ap.add_argument("--debug", action="store_true", help="dump calibration visuals (mean/mask/bbox) next to the output")
    args = ap.parse_args()

    input_path = args.input
    output_path = args.output
    if output_path is None:
        if input_path.lower().endswith((".mp4", ".mkv", ".mov")):
            base, ext = input_path.rsplit(".", 1)
            output_path = f"{base}_processed.mp4"
        else:
            output_path = input_path + "_processed.mp4"

    info = ffprobe_info(input_path)
    width, height = info["width"], info["height"]

    if args.box is not None:
        x1, y1, x2, y2 = args.box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            print(f"--box resolves to an empty/invalid region after clamping to the "
                  f"{width}x{height} frame; check the coordinates.", file=sys.stderr)
            sys.exit(1)
        # explicit box, applied to the whole video as a single "segment"
        segments = [manual_box_segment((x1, y1, x2, y2))]
        segment_frames = 2 ** 62  # single segment for the whole video
    else:
        x0, y0, qw, qh = quadrant_bounds(width, height, args.corner, *args.region_frac)
        corner_point = {
            "br": (qw, qh), "bl": (0, qh), "tr": (qw, 0), "tl": (0, 0),
        }[args.corner]

        nb_frames = info.get("nb_frames")
        if nb_frames:
            segment_frames = max(1, round(args.segment_seconds * info["fps"]))
            num_segments = max(1, -(-nb_frames // segment_frames))  # ceil div
            global_step = max(1, segment_frames // max(1, args.samples))
        else:
            # No reliable frame count to segment against — fall back to one
            # whole-video calibration, with a sampling step derived the same
            # way the original single-calibration version did.
            segment_frames = 2 ** 62
            num_segments = 1
            fallback_total = args.samples * 4
            global_step = max(1, fallback_total // max(1, args.samples))

        print(f"Calibrating {num_segments} segment(s) against corner={args.corner} "
              f"region=({x0},{y0},{qw}x{qh})...", file=sys.stderr)
        by_segment = sample_patches_by_segment(input_path, x0, y0, qw, qh, segment_frames, num_segments, global_step)
        try:
            segments = calibrate_segments(by_segment, corner_point, (x0, y0))
        except RuntimeError as e:
            print(f"{e}\nTry --region-frac with a larger area, a different --corner, "
                  f"or pass an explicit --box x,y,w,h.", file=sys.stderr)
            sys.exit(1)

        if args.debug:
            mean = np.stack(by_segment[0], axis=0).mean(axis=0).astype(np.uint8)
            lx1, ly1 = segments[0]["bbox_abs"][0] - x0, segments[0]["bbox_abs"][1] - y0
            lx2, ly2 = segments[0]["bbox_abs"][2] - x0, segments[0]["bbox_abs"][3] - y0
            cv2.imwrite("debug_quadrant_mean.png", mean)
            cv2.rectangle(mean, (lx1, ly1), (lx2, ly2), (0, 0, 255), 1)
            cv2.imwrite("debug_quadrant_bbox.png", mean)
            cv2.imwrite("debug_mask.png", (segments[0]["soft_mask"] * 255).astype(np.uint8))

    if len(segments) > 1:
        print(f"Calibrated {len(segments)} segments ({args.segment_seconds}s each):", file=sys.stderr)
        for i, seg in enumerate(segments):
            sx1, sy1, sx2, sy2 = seg["bbox_abs"]
            print(f"  segment {i}: x={sx1} y={sy1} w={sx2 - sx1} h={sy2 - sy1}", file=sys.stderr)
    else:
        sx1, sy1, sx2, sy2 = segments[0]["bbox_abs"]
        print(f"Watermark box: x={sx1} y={sy1} w={sx2 - sx1} h={sy2 - sy1}", file=sys.stderr)

    if os.path.exists(output_path):
        print(f"Note: {output_path} already exists and will be overwritten.", file=sys.stderr)

    process_fn = process_video_adaptive if args.adaptive else process_video
    n, cancelled = process_fn(input_path, output_path, segments, segment_frames, args.inpaint_radius,
                               denoise=not args.no_denoise, inpaint_method=args.inpaint_method)
    if cancelled:
        print(f"Cancelled after {n} frames.", file=sys.stderr)
    else:
        print(f"Wrote {n} frames to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
