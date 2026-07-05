#![cfg_attr(windows, windows_subsystem = "windows")]

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Instant;

use eframe::egui;
use watermark_remover::pipeline::{self, Options, RunOutcome};

const STALL_WARNING_SECS: u64 = 30;

enum ProgressMsg {
    Progress { done: u64, total: u64 },
    Done(Result<RunOutcome, String>),
}

#[derive(PartialEq)]
enum Status {
    Idle,
    Processing,
}

struct App {
    input_path: Option<PathBuf>,
    output_path: Option<PathBuf>,
    status: Status,
    progress_rx: Option<mpsc::Receiver<ProgressMsg>>,
    cancel_flag: Option<Arc<AtomicBool>>,
    last_progress_at: Option<Instant>,
    done_frames: u64,
    total_frames: u64,
    error: Option<String>,
    result_message: Option<String>,
}

impl Default for App {
    fn default() -> Self {
        App {
            input_path: None,
            output_path: None,
            status: Status::Idle,
            progress_rx: None,
            cancel_flag: None,
            last_progress_at: None,
            done_frames: 0,
            total_frames: 0,
            error: None,
            result_message: None,
        }
    }
}

impl App {
    fn set_input(&mut self, path: PathBuf) {
        self.input_path = Some(path);
        self.output_path = None;
        self.result_message = None;
        self.error = None;
    }

    fn start_processing(&mut self, ctx: &egui::Context) {
        let Some(input) = self.input_path.clone() else { return };
        let output = self.output_path.clone().unwrap_or_else(|| default_output_path(&input));
        self.output_path = Some(output.clone());

        let (tx, rx) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        self.progress_rx = Some(rx);
        self.cancel_flag = Some(cancel.clone());
        self.error = None;
        self.result_message = None;
        self.done_frames = 0;
        self.total_frames = 0;
        self.status = Status::Processing;
        self.last_progress_at = Some(Instant::now());

        let worker_ctx = ctx.clone();
        std::thread::spawn(move || {
            let opts = Options::default();
            let progress_ctx = worker_ctx.clone();
            let progress_tx = tx.clone();
            let result = pipeline::run(&input, &output, &opts, &cancel, move |done, total| {
                let _ = progress_tx.send(ProgressMsg::Progress { done, total });
                progress_ctx.request_repaint();
            });
            let _ = tx.send(ProgressMsg::Done(result.map_err(|e| e.to_string())));
            worker_ctx.request_repaint();
        });
    }

    fn drain_progress(&mut self) {
        let Some(rx) = &self.progress_rx else { return };
        let mut finished = false;
        while let Ok(msg) = rx.try_recv() {
            match msg {
                ProgressMsg::Progress { done, total } => {
                    self.done_frames = done;
                    self.total_frames = total;
                    self.last_progress_at = Some(Instant::now());
                }
                ProgressMsg::Done(result) => {
                    finished = true;
                    match result {
                        Ok(RunOutcome::Completed { frames }) => {
                            let out = self.output_path.as_ref().map(|p| p.display().to_string()).unwrap_or_default();
                            self.result_message = Some(format!("Done — {frames} frames written to {out}"));
                        }
                        Ok(RunOutcome::Cancelled { frames }) => {
                            self.result_message = Some(format!("Cancelled after {frames} frames."));
                        }
                        Err(e) => {
                            self.error = Some(e);
                        }
                    }
                }
            }
        }
        if finished {
            self.status = Status::Idle;
            self.progress_rx = None;
            self.cancel_flag = None;
        }
    }
}

fn default_output_path(input: &std::path::Path) -> PathBuf {
    let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
    let dir = input.parent().unwrap_or_else(|| std::path::Path::new("."));
    dir.join(format!("{stem}_processed.mp4"))
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.drain_progress();

        let ctx = ui.ctx().clone();
        ctx.input(|i| {
            if let Some(file) = i.raw.dropped_files.first() {
                if let Some(path) = file.path.clone() {
                    self.set_input(path);
                }
            }
        });

        egui::CentralPanel::default().show(ui, |ui| {
            ui.heading("Watermark Remover");
            ui.label("Removes the static Veo/Gemini corner watermark from AI-generated clips.");
            ui.add_space(12.0);

            match self.status {
                Status::Idle => self.ui_idle(ui, &ctx),
                Status::Processing => self.ui_processing(ui),
            }
        });

        preview_files_being_dropped(&ctx);
    }
}

impl App {
    fn ui_idle(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.label("Drop a video file anywhere in this window, or:");
        if ui.button("Choose video…").clicked() {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("video", &["mp4", "mov", "mkv"])
                .pick_file()
            {
                self.set_input(path);
            }
        }

        if let Some(input) = self.input_path.clone() {
            ui.add_space(8.0);
            ui.label(format!("Input: {}", input.display()));
            ui.add_space(12.0);
            if ui.button("Remove watermark").clicked() {
                self.start_processing(ctx);
            }
        }

        if let Some(msg) = self.result_message.clone() {
            ui.add_space(12.0);
            ui.colored_label(egui::Color32::from_rgb(100, 220, 120), &msg);
            if ui.button("Open output folder").clicked() {
                if let Some(dir) = self.output_path.as_ref().and_then(|p| p.parent()) {
                    let _ = open_in_file_manager(dir);
                }
            }
        }
        if let Some(err) = self.error.clone() {
            ui.add_space(12.0);
            ui.colored_label(egui::Color32::from_rgb(230, 90, 90), format!("Error: {err}"));
        }
    }

    fn ui_processing(&mut self, ui: &mut egui::Ui) {
        if let Some(input) = &self.input_path {
            ui.label(format!("Processing: {}", input.display()));
        }
        ui.add_space(8.0);

        if self.total_frames > 0 {
            let frac = self.done_frames as f32 / self.total_frames as f32;
            ui.add(egui::ProgressBar::new(frac).text(format!("{}/{}", self.done_frames, self.total_frames)));
        } else {
            ui.spinner();
            ui.label(format!("{} frames processed", self.done_frames));
        }

        if let Some(last) = self.last_progress_at {
            if last.elapsed().as_secs() > STALL_WARNING_SECS {
                ui.add_space(8.0);
                ui.colored_label(
                    egui::Color32::from_rgb(230, 190, 80),
                    "No progress for a while — this may be stuck. You can cancel below.",
                );
            }
        }

        ui.add_space(12.0);
        if ui.button("Cancel").clicked() {
            if let Some(flag) = &self.cancel_flag {
                flag.store(true, Ordering::Relaxed);
            }
        }
    }
}

fn preview_files_being_dropped(ctx: &egui::Context) {
    if ctx.input(|i| !i.raw.hovered_files.is_empty()) {
        let painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Foreground,
            egui::Id::new("file_drop_overlay"),
        ));
        let screen_rect = ctx.viewport_rect();
        painter.rect_filled(screen_rect, 0.0, egui::Color32::from_black_alpha(180));
        painter.text(
            screen_rect.center(),
            egui::Align2::CENTER_CENTER,
            "Drop video to load",
            egui::FontId::proportional(24.0),
            egui::Color32::WHITE,
        );
    }
}

#[cfg(windows)]
fn open_in_file_manager(dir: &std::path::Path) -> std::io::Result<()> {
    std::process::Command::new("explorer").arg(dir).spawn()?;
    Ok(())
}
#[cfg(not(windows))]
fn open_in_file_manager(_dir: &std::path::Path) -> std::io::Result<()> {
    Ok(())
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([520.0, 340.0])
            .with_drag_and_drop(true),
        ..Default::default()
    };
    eframe::run_native("Watermark Remover", options, Box::new(|_cc| Ok(Box::new(App::default()))))
}
