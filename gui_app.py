#!/usr/bin/env python3
"""PySide6 desktop GUI for remove_watermark.py.

Drag-and-drop a video (or use the file picker). The tool auto-calibrates a
watermark box per segment, then shows you a preview frame per segment with
the detected box overlaid — drag/resize it (or draw a new one) if it's
wrong, step through segments with Prev/Next, then start removal. Reuses the
exact same pipeline as the CLI (calibrate_segments/process_video); this is
a front end plus a manual-review step over it.
"""
import os
import subprocess
import sys
import threading
import time
import traceback

import cv2

from PySide6.QtCore import Qt, QThread, Signal, QTimer, QRectF
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QImage, QPainter, QPen, QBrush, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QProgressBar, QFileDialog, QFrame, QMessageBox, QCheckBox,
)

import remove_watermark as rw

APP_VERSION = "0.2.0"
STALL_WARNING_SECS = 30

# A --windowed build has no console, so an uncaught exception on the main
# thread would otherwise vanish silently (app just freezes or disappears,
# with nothing to see why). Log it next to the exe and show it in a message
# box instead.
LOG_PATH = os.path.join(
    os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__)),
    "error_log.txt",
)


def log_exception(kind, exc, tb_text):
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"--- {kind} ---\n{tb_text}\n")
    except Exception:
        pass


def install_excepthook():
    def hook(exc_type, exc_value, exc_tb):
        tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        log_exception("uncaught exception (main thread)", exc_value, tb_text)
        try:
            QMessageBox.critical(None, "Watermark Remover — error",
                                  f"An unexpected error occurred:\n\n{exc_value}\n\n"
                                  f"Details were written to:\n{LOG_PATH}")
        except Exception:
            pass
    sys.excepthook = hook


SEGMENT_SECONDS = 5.0
SAMPLES_PER_SEGMENT = 30


def plan_segments(info):
    """Returns (segment_frames, num_segments, global_step) for a probed
    video — shared by calibration and preview-frame grabbing so both agree
    on the same segment boundaries."""
    nb_frames = info.get("nb_frames")
    if nb_frames:
        segment_frames = max(1, round(SEGMENT_SECONDS * info["fps"]))
        num_segments = max(1, -(-nb_frames // segment_frames))
        global_step = max(1, segment_frames // max(1, SAMPLES_PER_SEGMENT))
    else:
        segment_frames = 2 ** 62
        num_segments = 1
        global_step = max(1, (SAMPLES_PER_SEGMENT * 4) // max(1, SAMPLES_PER_SEGMENT))
    return segment_frames, num_segments, global_step


class CalibrateThread(QThread):
    done = Signal(list, list, int)  # boxes (bbox_abs or None per segment), preview frames, segment_frames
    failed = Signal(str)

    def __init__(self, input_path):
        super().__init__()
        self.input_path = input_path

    def run(self):
        try:
            info = rw.ffprobe_info(self.input_path)
            width, height = info["width"], info["height"]
            x0, y0, qw, qh = rw.quadrant_bounds(width, height, "br", 0.25, 0.30)
            corner_point = (qw, qh)
            segment_frames, num_segments, global_step = plan_segments(info)

            previews = rw.grab_segment_preview_frames(self.input_path, segment_frames, num_segments)

            by_segment = rw.sample_patches_by_segment(
                self.input_path, x0, y0, qw, qh, segment_frames, num_segments, global_step)
            try:
                segments = rw.calibrate_segments(by_segment, corner_point, (x0, y0))
                boxes = [s["bbox_abs"] for s in segments]
            except RuntimeError:
                # Nothing auto-detected anywhere — hand back a small default
                # box near the corner per segment so there's still something
                # to see and drag into place, rather than a hard failure.
                default_w, default_h = min(120, qw), min(120, qh)
                default_box = (x0 + qw - default_w, y0 + qh - default_h, x0 + qw, y0 + qh)
                boxes = [default_box] * num_segments

            self.done.emit(boxes, previews, segment_frames)
        except Exception as e:
            log_exception("calibration error", e, traceback.format_exc())
            self.failed.emit(str(e))


class ProcessThread(QThread):
    progress = Signal(int, int)
    succeeded = Signal(str, int)
    cancelled = Signal(int)
    failed = Signal(str)

    def __init__(self, input_path, output_path, boxes, segment_frames, denoise=False, adaptive=False,
                 inpaint_method=rw.DEFAULT_INPAINT_METHOD):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.boxes = boxes
        self.segment_frames = segment_frames
        self.denoise = denoise
        self.adaptive = adaptive
        self.inpaint_method = inpaint_method
        self.cancel_event = threading.Event()

    def cancel(self):
        self.cancel_event.set()

    def run(self):
        try:
            segments = [rw.manual_box_segment(box) for box in self.boxes]
            process_fn = rw.process_video_adaptive if self.adaptive else rw.process_video
            frames, cancelled = process_fn(
                self.input_path, self.output_path, segments, self.segment_frames,
                inpaint_radius=5, denoise=self.denoise, progress=False,
                on_progress=lambda done, total: self.progress.emit(done, total),
                cancel_event=self.cancel_event, inpaint_method=self.inpaint_method,
            )
            if cancelled:
                self.cancelled.emit(frames)
            else:
                self.succeeded.emit(self.output_path, frames)
        except Exception as e:
            log_exception("processing error", e, traceback.format_exc())
            self.failed.emit(str(e))


class DropArea(QFrame):
    file_dropped = Signal(str)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumHeight(80)
        self.setStyleSheet("QFrame { border: 2px dashed #888; border-radius: 8px; }")
        layout = QVBoxLayout(self)
        self.label = QLabel("Drop a video file here")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            self.file_dropped.emit(urls[0].toLocalFile())


class BoxEditor(QWidget):
    """Shows a video frame with a watermark box overlaid. Drag inside the
    box to move it, drag a corner handle to resize it, or click-drag on
    empty space to draw a brand new box from scratch."""

    HANDLE_R = 6
    MIN_SIZE = 8

    box_changed = Signal()

    def __init__(self):
        super().__init__()
        self.setMinimumSize(480, 300)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)
        self._qimage = None
        self.frame_w = 0
        self.frame_h = 0
        self.box = None  # (x, y, w, h) in frame pixel coords
        self._drag_mode = None
        self._drag_start_mouse = None
        self._drag_start_box = None
        self._new_anchor = None

    def set_frame(self, bgr_frame):
        h, w = bgr_frame.shape[:2]
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()
        self._qimage = qimg
        self.frame_w, self.frame_h = w, h
        self.update()

    def set_box(self, bbox_abs):
        """Takes (x1, y1, x2, y2) — the same convention used everywhere
        outside this widget (calibration results, self.boxes, confirmed_box's
        return value) — and converts to the (x,y,w,h) this widget tracks
        internally while dragging."""
        if bbox_abs is None:
            self.box = None
        else:
            x1, y1, x2, y2 = bbox_abs
            self.box = (float(x1), float(y1), float(x2 - x1), float(y2 - y1))
        self.update()

    def _fit(self):
        if self.frame_w == 0 or self.frame_h == 0:
            return 1.0, 0.0, 0.0
        scale = min(self.width() / self.frame_w, self.height() / self.frame_h)
        disp_w, disp_h = self.frame_w * scale, self.frame_h * scale
        off_x = (self.width() - disp_w) / 2
        off_y = (self.height() - disp_h) / 2
        return scale, off_x, off_y

    def _f2w(self, x, y):
        scale, ox, oy = self._fit()
        return ox + x * scale, oy + y * scale

    def _w2f(self, x, y):
        scale, ox, oy = self._fit()
        if scale <= 0:
            return 0.0, 0.0
        fx = (x - ox) / scale
        fy = (y - oy) / scale
        return min(max(0.0, fx), float(self.frame_w)), min(max(0.0, fy), float(self.frame_h))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(35, 35, 35))
        if self._qimage is not None:
            scale, ox, oy = self._fit()
            target = QRectF(ox, oy, self.frame_w * scale, self.frame_h * scale)
            painter.drawImage(target, self._qimage)
        if self.box is not None:
            x, y, w, h = self.box
            x0, y0 = self._f2w(x, y)
            x1, y1 = self._f2w(x + w, y + h)
            painter.setPen(QPen(QColor(255, 70, 70), 2))
            painter.setBrush(QBrush(QColor(255, 70, 70, 50)))
            painter.drawRect(QRectF(x0, y0, x1 - x0, y1 - y0))
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            r = self.HANDLE_R
            for hx, hy in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
                painter.drawRect(QRectF(hx - r, hy - r, r * 2, r * 2))
        painter.end()

    def _hit_test(self, pos):
        if self.box is None:
            return "new"
        x, y, w, h = self.box
        x0, y0 = self._f2w(x, y)
        x1, y1 = self._f2w(x + w, y + h)
        px, py = pos.x(), pos.y()
        tol = self.HANDLE_R * 1.6

        def near(hx, hy):
            return abs(px - hx) <= tol and abs(py - hy) <= tol

        if near(x0, y0):
            return "tl"
        if near(x1, y0):
            return "tr"
        if near(x0, y1):
            return "bl"
        if near(x1, y1):
            return "br"
        if x0 <= px <= x1 and y0 <= py <= y1:
            return "move"
        return "new"

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        mode = self._hit_test(event.position())
        self._drag_start_mouse = event.position()
        self._drag_start_box = self.box
        if mode == "new":
            fx, fy = self._w2f(event.position().x(), event.position().y())
            self._new_anchor = (fx, fy)
            self.box = (fx, fy, 1, 1)
            self._drag_mode = "new"
        else:
            self._drag_mode = mode

    def mouseMoveEvent(self, event):
        if self._drag_mode is None:
            mode = self._hit_test(event.position())
            cursors = {
                "tl": Qt.SizeFDiagCursor, "br": Qt.SizeFDiagCursor,
                "tr": Qt.SizeBDiagCursor, "bl": Qt.SizeBDiagCursor,
                "move": Qt.SizeAllCursor, "new": Qt.CrossCursor,
            }
            self.setCursor(cursors[mode])
            return

        fx, fy = self._w2f(event.position().x(), event.position().y())

        if self._drag_mode == "new":
            ax, ay = self._new_anchor
            x0, y0 = min(ax, fx), min(ay, fy)
            x1, y1 = max(ax, fx), max(ay, fy)
            self.box = (x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0))

        elif self._drag_mode == "move":
            sx, sy, sw, sh = self._drag_start_box
            start_fx, start_fy = self._w2f(self._drag_start_mouse.x(), self._drag_start_mouse.y())
            dx, dy = fx - start_fx, fy - start_fy
            nx = min(max(0.0, sx + dx), max(0.0, self.frame_w - sw))
            ny = min(max(0.0, sy + dy), max(0.0, self.frame_h - sh))
            self.box = (nx, ny, sw, sh)

        else:  # tl / tr / bl / br resize
            sx, sy, sw, sh = self._drag_start_box
            x0, y0, x1, y1 = sx, sy, sx + sw, sy + sh
            if "l" in self._drag_mode:
                x0 = fx
            if "r" in self._drag_mode:
                x1 = fx
            if "t" in self._drag_mode:
                y0 = fy
            if "b" in self._drag_mode:
                y1 = fy
            nx0, nx1 = min(x0, x1), max(x0, x1)
            ny0, ny1 = min(y0, y1), max(y0, y1)
            if nx1 - nx0 < self.MIN_SIZE:
                nx1 = nx0 + self.MIN_SIZE
            if ny1 - ny0 < self.MIN_SIZE:
                ny1 = ny0 + self.MIN_SIZE
            self.box = (nx0, ny0, nx1 - nx0, ny1 - ny0)

        self.update()

    def mouseReleaseEvent(self, event):
        if self._drag_mode is not None:
            self._drag_mode = None
            self.box_changed.emit()

    def confirmed_box(self):
        """Returns the box as integer (x1, y1, x2, y2), clamped to the frame."""
        if self.box is None:
            return None
        x, y, w, h = self.box
        x1 = int(max(0, min(self.frame_w - 1, round(x))))
        y1 = int(max(0, min(self.frame_h - 1, round(y))))
        x2 = int(max(x1 + 1, min(self.frame_w, round(x + w))))
        y2 = int(max(y1 + 1, min(self.frame_h, round(y + h))))
        return (x1, y1, x2, y2)


class MainWindow(QMainWindow):
    IDLE, CALIBRATING, REVIEW, PROCESSING = range(4)

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Watermark Remover v{APP_VERSION}")
        self.resize(760, 560)

        self.input_path = None
        self.output_path = None
        self.calibrate_thread = None
        self.process_thread = None
        self.last_progress_at = None

        self.preview_frames = []
        self.boxes = []
        self.auto_boxes = []
        self.user_touched = set()
        self.segment_frames = 1
        self.current_segment = 0

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        heading = QLabel(f"<b>Watermark Remover</b> — v{APP_VERSION}")
        layout.addWidget(heading)
        layout.addWidget(QLabel("Removes the static Veo/Gemini corner watermark from AI-generated clips."))

        # --- Idle widgets ---
        self.drop_area = DropArea()
        self.drop_area.file_dropped.connect(self.set_input)
        layout.addWidget(self.drop_area)

        row = QHBoxLayout()
        self.choose_btn = QPushButton("Choose video…")
        self.choose_btn.clicked.connect(self.choose_file)
        row.addWidget(self.choose_btn)
        layout.addLayout(row)

        self.input_label = QLabel("")
        self.input_label.setWordWrap(True)
        layout.addWidget(self.input_label)

        self.start_btn = QPushButton("Find watermark…")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_calibration)
        layout.addWidget(self.start_btn)

        # --- Calibrating widget ---
        self.calibrating_label = QLabel("Auto-detecting the watermark…")
        self.calibrating_label.setVisible(False)
        layout.addWidget(self.calibrating_label)

        # --- Review widgets ---
        self.box_editor = BoxEditor()
        self.box_editor.setVisible(False)
        self.box_editor.box_changed.connect(self.on_box_changed)
        layout.addWidget(self.box_editor, stretch=1)

        review_row = QHBoxLayout()
        self.prev_btn = QPushButton("◀ Prev segment")
        self.prev_btn.clicked.connect(self.prev_segment)
        self.prev_btn.setVisible(False)
        review_row.addWidget(self.prev_btn)
        self.segment_label = QLabel("")
        self.segment_label.setAlignment(Qt.AlignCenter)
        self.segment_label.setVisible(False)
        review_row.addWidget(self.segment_label, stretch=1)
        self.next_btn = QPushButton("Next segment ▶")
        self.next_btn.clicked.connect(self.next_segment)
        self.next_btn.setVisible(False)
        review_row.addWidget(self.next_btn)
        layout.addLayout(review_row)

        review_actions = QHBoxLayout()
        self.review_hint = QLabel("Drag inside the box to move it, drag a corner to resize, or "
                                   "click-drag on empty space to draw a new one. Fixing one segment "
                                   "carries that box forward to the segments after it.")
        self.review_hint.setWordWrap(True)
        self.review_hint.setVisible(False)
        review_actions.addWidget(self.review_hint, stretch=1)
        self.reset_box_btn = QPushButton("Reset to auto-detected")
        self.reset_box_btn.setVisible(False)
        self.reset_box_btn.clicked.connect(self.reset_current_box)
        review_actions.addWidget(self.reset_box_btn)
        layout.addLayout(review_actions)

        self.denoise_check = QCheckBox("Reduce compression artifacts (can look slightly softer)")
        self.denoise_check.setChecked(False)
        self.denoise_check.setVisible(False)
        layout.addWidget(self.denoise_check)

        self.adaptive_check = QCheckBox(
            "Reduce flicker on static backgrounds (slower — two passes; best for a locked-off camera)")
        self.adaptive_check.setChecked(False)
        self.adaptive_check.setVisible(False)
        self.adaptive_check.toggled.connect(self.on_adaptive_toggled)
        layout.addWidget(self.adaptive_check)

        self.quality_check = QCheckBox(
            "Higher-quality fill (much slower — continues patterns like straight edges/grain far better; "
            "needs the flicker option above)")
        self.quality_check.setChecked(False)
        self.quality_check.setVisible(False)
        self.quality_check.setEnabled(False)
        layout.addWidget(self.quality_check)

        self.process_btn = QPushButton("Looks good — remove watermark")
        self.process_btn.setVisible(False)
        self.process_btn.clicked.connect(self.start_processing)
        layout.addWidget(self.process_btn)

        # --- Processing widgets ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self.cancel_processing)
        layout.addWidget(self.cancel_btn)

        self.stall_label = QLabel("")
        self.stall_label.setStyleSheet("color: #d9a300;")
        layout.addWidget(self.stall_label)

        # --- Result widgets ---
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.open_folder_btn = QPushButton("Open output folder")
        self.open_folder_btn.setVisible(False)
        self.open_folder_btn.clicked.connect(self.open_output_folder)
        layout.addWidget(self.open_folder_btn)

        self.stall_timer = QTimer(self)
        self.stall_timer.setInterval(1000)
        self.stall_timer.timeout.connect(self.check_stall)

        self._set_state(self.IDLE)

    # ---------- state ----------

    def _set_state(self, state):
        self.state = state
        idle = state == self.IDLE
        calibrating = state == self.CALIBRATING
        review = state == self.REVIEW
        processing = state == self.PROCESSING

        self.drop_area.setVisible(idle)
        self.choose_btn.setVisible(idle)
        self.input_label.setVisible(idle)
        self.start_btn.setVisible(idle)

        self.calibrating_label.setVisible(calibrating)

        self.box_editor.setVisible(review)
        self.prev_btn.setVisible(review)
        self.next_btn.setVisible(review)
        self.segment_label.setVisible(review)
        self.review_hint.setVisible(review)
        self.reset_box_btn.setVisible(review)
        self.denoise_check.setVisible(review)
        self.adaptive_check.setVisible(review)
        self.quality_check.setVisible(review)
        self.process_btn.setVisible(review)

        self.progress_bar.setVisible(processing)
        self.cancel_btn.setVisible(processing)

    # ---------- idle ----------

    def set_input(self, path):
        self.input_path = path
        self.output_path = None
        self.input_label.setText(f"Input: {path}")
        self.start_btn.setEnabled(True)
        self.status_label.setText("")
        self.open_folder_btn.setVisible(False)

    def choose_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose video", "", "Video files (*.mp4 *.mov *.mkv)")
        if path:
            self.set_input(path)

    def default_output_path(self, input_path):
        base, _ = os.path.splitext(input_path)
        return f"{base}_processed.mp4"

    # ---------- calibration ----------

    def start_calibration(self):
        if not self.input_path:
            return
        self._set_state(self.CALIBRATING)
        self.status_label.setText("")
        self.calibrate_thread = CalibrateThread(self.input_path)
        self.calibrate_thread.done.connect(self.on_calibrated)
        self.calibrate_thread.failed.connect(self.on_calibration_failed)
        self.calibrate_thread.start()

    def on_calibrated(self, boxes, preview_frames, segment_frames):
        self.boxes = list(boxes)
        self.auto_boxes = list(boxes)
        self.user_touched = set()
        self.preview_frames = preview_frames
        self.segment_frames = segment_frames
        self.current_segment = 0
        self._set_state(self.REVIEW)
        self._show_segment(0)

    def on_calibration_failed(self, message):
        self._set_state(self.IDLE)
        self.status_label.setStyleSheet("color: #c0392b;")
        self.status_label.setText(f"Error: {message}")

    def on_adaptive_toggled(self, checked):
        # The higher-quality fill (fsr_best) only runs once per detected
        # stable window, not once per frame — without the flicker-reduction
        # pass batching things into windows, it would mean ~19s of inpainting
        # for every single frame, which is impractical.
        self.quality_check.setEnabled(checked)
        if not checked:
            self.quality_check.setChecked(False)

    # ---------- review ----------

    def _show_segment(self, i):
        # A segment the user hasn't touched yet inherits the box from the
        # nearest earlier segment they *did* fix, rather than showing its
        # own (possibly wrong) auto-detected box — fixing segment 1 should
        # mean segments 2, 3, ... start from that same corrected position
        # instead of making the user redo the same fix repeatedly.
        if i not in self.user_touched:
            for j in range(i - 1, -1, -1):
                if j in self.user_touched:
                    self.boxes[i] = self.boxes[j]
                    break

        self.current_segment = i
        n = len(self.boxes)
        self.box_editor.set_frame(self.preview_frames[i])
        self.box_editor.set_box(self.boxes[i])
        start_s = i * self.segment_frames / max(1.0, self._fps())
        if n > 1:
            self.segment_label.setText(f"Segment {i + 1} of {n}  (~{start_s:.0f}s)")
        else:
            self.segment_label.setText("Whole clip")
        self.prev_btn.setEnabled(i > 0)
        self.next_btn.setEnabled(i < n - 1)

    def _fps(self):
        try:
            return rw.ffprobe_info(self.input_path)["fps"]
        except Exception:
            return 24.0

    def on_box_changed(self):
        box = self.box_editor.confirmed_box()
        if box is not None:
            self.boxes[self.current_segment] = box
            self.user_touched.add(self.current_segment)

    def reset_current_box(self):
        i = self.current_segment
        self.boxes[i] = self.auto_boxes[i]
        self.user_touched.discard(i)
        self.box_editor.set_box(self.boxes[i])

    def prev_segment(self):
        if self.current_segment > 0:
            self._show_segment(self.current_segment - 1)

    def next_segment(self):
        if self.current_segment < len(self.boxes) - 1:
            self._show_segment(self.current_segment + 1)

    # ---------- processing ----------

    def start_processing(self):
        # Make sure the currently-shown segment's edits are captured even if
        # the user didn't nudge the mouse after their last drag.
        box = self.box_editor.confirmed_box()
        if box is not None:
            self.boxes[self.current_segment] = box

        self.output_path = self.default_output_path(self.input_path)
        self._set_state(self.PROCESSING)
        self.status_label.setText("")
        self.stall_label.setText("")
        self.open_folder_btn.setVisible(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setRange(0, 0)
        self.last_progress_at = time.monotonic()
        self.stall_timer.start()

        adaptive = self.adaptive_check.isChecked()
        inpaint_method = "fsr_best" if (adaptive and self.quality_check.isChecked()) else rw.DEFAULT_INPAINT_METHOD
        self.process_thread = ProcessThread(self.input_path, self.output_path, self.boxes, self.segment_frames,
                                             denoise=self.denoise_check.isChecked(),
                                             adaptive=adaptive, inpaint_method=inpaint_method)
        self.process_thread.progress.connect(self.on_progress)
        self.process_thread.succeeded.connect(self.on_succeeded)
        self.process_thread.cancelled.connect(self.on_cancelled)
        self.process_thread.failed.connect(self.on_failed)
        self.process_thread.start()

    def cancel_processing(self):
        if self.process_thread is not None:
            self.process_thread.cancel()
            self.cancel_btn.setEnabled(False)

    def on_progress(self, done, total):
        self.last_progress_at = time.monotonic()
        self.stall_label.setText("")
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(done)
        else:
            self.progress_bar.setRange(0, 0)

    def check_stall(self):
        if self.last_progress_at is None:
            return
        if time.monotonic() - self.last_progress_at > STALL_WARNING_SECS:
            self.stall_label.setText("No progress for a while — this may be stuck. You can cancel below.")

    def _reset_after_run(self):
        self.stall_timer.stop()
        self.cancel_btn.setEnabled(True)
        self.stall_label.setText("")
        self._set_state(self.IDLE)

    def on_succeeded(self, output_path, frames):
        self._reset_after_run()
        self.status_label.setStyleSheet("color: #2e8b40;")
        self.status_label.setText(f"Done — {frames} frames written to {output_path}")
        self.open_folder_btn.setVisible(True)

    def on_cancelled(self, frames):
        self._reset_after_run()
        self.status_label.setStyleSheet("")
        self.status_label.setText(f"Cancelled after {frames} frames.")

    def on_failed(self, message):
        self._reset_after_run()
        self.status_label.setStyleSheet("color: #c0392b;")
        self.status_label.setText(f"Error: {message}")

    def open_output_folder(self):
        if self.output_path:
            folder = os.path.dirname(os.path.abspath(self.output_path))
            if os.name == "nt":
                subprocess.Popen(["explorer", folder])


def main():
    install_excepthook()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
