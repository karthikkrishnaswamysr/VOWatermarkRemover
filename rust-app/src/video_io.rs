//! ffprobe/ffmpeg subprocess plumbing. We never link against libav* — every
//! interaction with video codecs goes through a separate ffmpeg/ffprobe
//! process, piped rawvideo in and out. That keeps this binary free of any
//! codec-library licensing/linking question (see docs/FFMPEG_LICENSING.md).

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// Kills (and reaps) the wrapped ffmpeg process on drop unless
/// [`ChildGuard::into_inner`] was called first. Protects against orphaned
/// `ffmpeg.exe` processes holding a lock on a half-written output file on
/// any early return — an error via `?`, a panic, or cooperative
/// cancellation from the GUI.
pub struct ChildGuard(Option<Child>);

impl ChildGuard {
    pub fn new(child: Child) -> Self {
        ChildGuard(Some(child))
    }

    pub fn get_mut(&mut self) -> &mut Child {
        self.0.as_mut().expect("ChildGuard used after into_inner")
    }

    /// Disarms the guard, handing back the raw `Child` for a normal
    /// `wait()` on the success path.
    pub fn into_inner(mut self) -> Child {
        self.0.take().expect("ChildGuard used after into_inner")
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if let Some(mut child) = self.0.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VideoInfo {
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub has_audio: bool,
    /// Container-reported frame count, when present. Not always available
    /// (depends on container/muxer) — callers needing a sampling step
    /// should fall back to a sane default when this is `None`, same as the
    /// Python prototype did for `cv2.CAP_PROP_FRAME_COUNT <= 0`.
    pub nb_frames: Option<u64>,
}

#[derive(Deserialize)]
struct ProbeOutput {
    streams: Vec<ProbeStream>,
}

#[derive(Deserialize)]
struct ProbeStream {
    codec_type: String,
    width: Option<u32>,
    height: Option<u32>,
    r_frame_rate: Option<String>,
    nb_frames: Option<String>,
}

/// Resolves a sidecar binary (ffmpeg/ffprobe) next to our own exe first,
/// falling back to PATH only if no sidecar is bundled (dev builds run from
/// `cargo run` before `dist/` is assembled). Never trusts a bare name off
/// PATH when a bundled copy is available, closing the PATH-hijack gap the
/// Python prototype had.
pub fn resolve_tool(name: &str) -> Result<PathBuf> {
    let exe_name = if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    };

    if let Ok(exe_dir) = std::env::current_exe() {
        if let Some(dir) = exe_dir.parent() {
            let candidate = dir.join(&exe_name);
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
    }

    Ok(PathBuf::from(exe_name))
}

fn base_command(exe: &Path) -> Command {
    let mut cmd = Command::new(exe);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}

pub fn probe(path: &Path) -> Result<VideoInfo> {
    let ffprobe = resolve_tool("ffprobe")?;
    let mut cmd = base_command(&ffprobe);
    cmd.args([
        "-v", "error",
        "-print_format", "json",
        "-show_entries", "stream=codec_type,width,height,r_frame_rate,nb_frames",
    ])
    .arg(path)
    .stdin(Stdio::null())
    .stdout(Stdio::piped())
    .stderr(Stdio::piped());

    let output = cmd.output().context("failed to run ffprobe")?;
    if !output.status.success() {
        bail!(
            "ffprobe exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let parsed: ProbeOutput = serde_json::from_slice(&output.stdout)
        .context("failed to parse ffprobe JSON output")?;

    let vstream = parsed
        .streams
        .iter()
        .find(|s| s.codec_type == "video")
        .ok_or_else(|| anyhow!("no video stream found"))?;
    let has_audio = parsed.streams.iter().any(|s| s.codec_type == "audio");

    let width = vstream.width.ok_or_else(|| anyhow!("video stream missing width"))?;
    let height = vstream.height.ok_or_else(|| anyhow!("video stream missing height"))?;
    let rate_str = vstream
        .r_frame_rate
        .as_deref()
        .ok_or_else(|| anyhow!("video stream missing r_frame_rate"))?;
    let (num, den) = rate_str
        .split_once('/')
        .ok_or_else(|| anyhow!("unexpected r_frame_rate format: {rate_str}"))?;
    let fps = num.parse::<f64>()? / den.parse::<f64>()?;
    let nb_frames = vstream.nb_frames.as_deref().and_then(|s| s.parse::<u64>().ok());

    Ok(VideoInfo { width, height, fps, has_audio, nb_frames })
}

/// Decodes `path` to raw rgb24 frames on stdout (rgb24, not bgr24, so bytes
/// map directly onto `image::RgbImage`'s channel order with no reshuffling).
/// Caller reads exactly `width*height*3` bytes per frame via `read_exact` (a
/// plain `read()` call is NOT guaranteed to fill the buffer in one go on a
/// pipe).
pub fn spawn_decoder(path: &Path) -> Result<Child> {
    let ffmpeg = resolve_tool("ffmpeg")?;
    let mut cmd = base_command(&ffmpeg);
    cmd.args(["-v", "error", "-i"])
        .arg(path)
        .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    cmd.spawn().context("failed to spawn ffmpeg decoder")
}

/// Encodes raw rgb24 frames written to stdin into `output`, re-muxing the
/// original audio track from `source` via stream copy (no re-encode).
pub fn spawn_encoder(
    source: &Path,
    output: &Path,
    info: VideoInfo,
) -> Result<Child> {
    let ffmpeg = resolve_tool("ffmpeg")?;
    let mut cmd = base_command(&ffmpeg);
    cmd.args(["-y", "-v", "error"])
        .args(["-f", "rawvideo", "-pix_fmt", "rgb24"])
        .arg("-s")
        .arg(format!("{}x{}", info.width, info.height))
        .arg("-r")
        .arg(info.fps.to_string())
        .args(["-i", "-"])
        .arg("-i")
        .arg(source)
        .args(["-map", "0:v:0"]);
    if info.has_audio {
        cmd.args(["-map", "1:a:0?", "-c:a", "copy"]);
    }
    cmd.args(["-c:v", "libx264", "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p"])
        .arg(output)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    cmd.spawn().context("failed to spawn ffmpeg encoder")
}

/// Reads exactly one frame's worth of bytes, returning `Ok(None)` on clean
/// EOF (no partial frame). Any short read that isn't a clean EOF is an
/// error, not a silently-dropped/desynced frame.
pub fn read_frame(child_stdout: &mut impl Read, buf: &mut [u8]) -> Result<bool> {
    match child_stdout.read_exact(buf) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(false),
        Err(e) => Err(e).context("error reading frame from decoder pipe"),
    }
}

pub fn write_frame(child_stdin: &mut impl Write, buf: &[u8]) -> Result<()> {
    child_stdin.write_all(buf).context("error writing frame to encoder pipe")
}
