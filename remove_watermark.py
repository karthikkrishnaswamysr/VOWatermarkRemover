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
import subprocess
import sys

import cv2
import numpy as np

CORNERS = ("br", "bl", "tr", "tl")


def ffprobe_info(path):
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_entries", "stream=index,codec_type,width,height,r_frame_rate,avg_frame_rate",
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
    return {
        "width": vstream["width"],
        "height": vstream["height"],
        "fps": fps,
        "has_audio": has_audio,
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


def sample_quadrant_patches(path, x0, y0, qw, qh, max_samples):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = max_samples * 4
    step = max(1, total // max_samples)
    patches = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            patches.append(frame[y0:y0 + qh, x0:x0 + qw].astype(np.float32))
        idx += 1
    cap.release()
    if not patches:
        raise RuntimeError("Could not read any frames from the input video")
    return patches


def calibrate(patches, min_size=12, max_size_frac=0.6, corner_point=None):
    """Find the static watermark blob in a stack of same-position patches.

    Returns (local_bbox, soft_mask) where local_bbox = (x1,y1,x2,y2) in patch
    coordinates and soft_mask is a float32 HxW array in [0,1] the same size as
    the bbox, giving the watermark's approximate coverage/shape.
    """
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
    dilate_px = 3
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
    dilated = cv2.dilate(core, k)
    mask = cv2.GaussianBlur(dilated.astype(np.float32) / 255.0, (0, 0), sigmaX=1.5)
    mask = np.clip(mask / (mask.max() + 1e-6), 0, 1)

    return (x1, y1, x2, y2), mask.astype(np.float32)


def build_binary_mask(soft_mask, threshold=0.3, dilate_px=2):
    binm = (soft_mask > threshold).astype(np.uint8) * 255
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        binm = cv2.dilate(binm, k)
    return binm


def process_video(path, output, bbox_abs, soft_mask, inpaint_radius, denoise, progress=True):
    info = ffprobe_info(path)
    w, h, fps, has_audio = info["width"], info["height"], info["fps"], info["has_audio"]
    x1, y1, x2, y2 = bbox_abs
    binary_mask = build_binary_mask(soft_mask)
    alpha3 = np.dstack([soft_mask] * 3)

    frame_size = w * h * 3
    decode_cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    encode_cmd = [
        "ffmpeg", "-y", "-v", "error",
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
    try:
        while True:
            raw = decode_proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()

            patch = frame[y1:y2, x1:x2]
            inpainted = cv2.inpaint(patch, binary_mask, inpaint_radius, cv2.INPAINT_TELEA)
            if denoise:
                inpainted = cv2.bilateralFilter(inpainted, d=7, sigmaColor=45, sigmaSpace=45)
            blended = (alpha3 * inpainted + (1 - alpha3) * patch).astype(np.uint8)
            frame[y1:y2, x1:x2] = blended

            encode_proc.stdin.write(frame.tobytes())
            frame_idx += 1
            if progress and frame_idx % 24 == 0:
                sys.stderr.write(f"\rprocessed {frame_idx} frames")
                sys.stderr.flush()
    finally:
        decode_proc.stdout.close()
        encode_proc.stdin.close()
        decode_ret = decode_proc.wait()
        encode_ret = encode_proc.wait()

    if progress:
        sys.stderr.write(f"\rprocessed {frame_idx} frames\n")

    if encode_ret != 0:
        raise RuntimeError(f"ffmpeg encode failed with exit code {encode_ret}")
    if decode_ret != 0:
        raise RuntimeError(f"ffmpeg decode failed with exit code {decode_ret}")

    return frame_idx


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
    ap.add_argument("--samples", type=int, default=90, help="number of frames to sample for calibration (default: 90)")
    ap.add_argument("--box", type=parse_box, default=None,
                     help="skip auto-detection and use an explicit watermark box: x,y,w,h (frame pixel coords)")
    ap.add_argument("--inpaint-radius", type=int, default=5, help="OpenCV inpaint radius (default: 5)")
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
        bbox_abs = (x1, y1, x2, y2)
        # explicit box: use a soft-edged ellipse as the blend mask
        bw, bh = x2 - x1, y2 - y1
        mask = np.zeros((bh, bw), dtype=np.float32)
        cv2.ellipse(mask, (bw // 2, bh // 2), (bw // 2, bh // 2), 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(2, min(bw, bh) / 8))
        mask = mask / (mask.max() + 1e-6)
    else:
        x0, y0, qw, qh = quadrant_bounds(width, height, args.corner, *args.region_frac)
        corner_point = {
            "br": (qw, qh), "bl": (0, qh), "tr": (qw, 0), "tl": (0, 0),
        }[args.corner]
        print(f"Calibrating against corner={args.corner} region=({x0},{y0},{qw}x{qh}) "
              f"using up to {args.samples} sampled frames...", file=sys.stderr)
        patches = sample_quadrant_patches(input_path, x0, y0, qw, qh, args.samples)
        local_bbox, mask = calibrate(patches, corner_point=corner_point)
        if local_bbox is None:
            print("Could not auto-detect a static watermark in that region.\n"
                  "Try --region-frac with a larger area, a different --corner, "
                  "or pass an explicit --box x,y,w,h.", file=sys.stderr)
            sys.exit(1)
        lx1, ly1, lx2, ly2 = local_bbox
        bbox_abs = (x0 + lx1, y0 + ly1, x0 + lx2, y0 + ly2)

        if args.debug:
            mean = np.stack(patches, axis=0).mean(axis=0).astype(np.uint8)
            cv2.imwrite("debug_quadrant_mean.png", mean)
            cv2.rectangle(mean, (lx1, ly1), (lx2, ly2), (0, 0, 255), 1)
            cv2.imwrite("debug_quadrant_bbox.png", mean)
            cv2.imwrite("debug_mask.png", (mask * 255).astype(np.uint8))

    bw, bh = bbox_abs[2] - bbox_abs[0], bbox_abs[3] - bbox_abs[1]
    print(f"Watermark box: x={bbox_abs[0]} y={bbox_abs[1]} w={bw} h={bh}", file=sys.stderr)

    n = process_video(input_path, output_path, bbox_abs, mask, args.inpaint_radius,
                       denoise=not args.no_denoise)
    print(f"Wrote {n} frames to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
