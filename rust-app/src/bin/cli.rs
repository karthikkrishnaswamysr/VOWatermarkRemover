use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

use anyhow::{bail, Context, Result};
use watermark_remover::calibrate::Corner;
use watermark_remover::pipeline::{self, Options, RunOutcome};

fn parse_box(s: &str) -> Result<(u32, u32, u32, u32)> {
    let parts: Vec<u32> = s
        .split(',')
        .map(|p| p.trim().parse::<u32>())
        .collect::<std::result::Result<_, _>>()
        .context("--box must be x,y,w,h (non-negative integers)")?;
    if parts.len() != 4 {
        bail!("--box must be x,y,w,h");
    }
    let (x, y, w, h) = (parts[0], parts[1], parts[2], parts[3]);
    Ok((x, y, x + w, y + h))
}

fn print_usage() {
    eprintln!(
        "usage: watermark-remover <input> [-o output] [--corner br|bl|tr|tl] \
         [--region-frac W H] [--samples N] [--box x,y,w,h] [--inpaint-iterations N]"
    );
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1).peekable();
    let input: PathBuf = match args.next() {
        Some(a) => a.into(),
        None => {
            print_usage();
            std::process::exit(1);
        }
    };

    let mut output: Option<PathBuf> = None;
    let mut opts = Options::default();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-o" | "--output" => {
                output = Some(args.next().context("--output needs a value")?.into());
            }
            "--corner" => {
                let v = args.next().context("--corner needs a value")?;
                opts.corner = match v.as_str() {
                    "br" => Corner::BottomRight,
                    "bl" => Corner::BottomLeft,
                    "tr" => Corner::TopRight,
                    "tl" => Corner::TopLeft,
                    other => bail!("unknown --corner value: {other} (expected br|bl|tr|tl)"),
                };
            }
            "--region-frac" => {
                let w: f64 = args.next().context("--region-frac needs W H")?.parse()?;
                let h: f64 = args.next().context("--region-frac needs W H")?.parse()?;
                opts.region_frac = (w, h);
            }
            "--samples" => {
                opts.samples = args.next().context("--samples needs a value")?.parse()?;
            }
            "--box" => {
                let v = args.next().context("--box needs a value")?;
                opts.manual_box = Some(parse_box(&v)?);
            }
            "--inpaint-iterations" => {
                opts.inpaint_iterations = args.next().context("--inpaint-iterations needs a value")?.parse()?;
            }
            "-h" | "--help" => {
                print_usage();
                return Ok(());
            }
            other => bail!("unknown argument: {other}"),
        }
    }

    let output = output.unwrap_or_else(|| {
        let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
        let dir = input.parent().unwrap_or_else(|| std::path::Path::new("."));
        dir.join(format!("{stem}_processed.mp4"))
    });

    if output.exists() {
        eprintln!("Note: {} already exists and will be overwritten.", output.display());
    }

    let cancel = AtomicBool::new(false); // CLI has no interactive cancel path (yet)
    let outcome = pipeline::run(&input, &output, &opts, &cancel, |done, total| {
        if total > 0 {
            eprint!("\rprocessed {done}/{total} frames");
        } else {
            eprint!("\rprocessed {done} frames");
        }
    })?;
    eprintln!();

    match outcome {
        RunOutcome::Completed { frames } => {
            println!("Wrote {frames} frames to {}", output.display());
        }
        RunOutcome::Cancelled { frames } => {
            println!("Cancelled after {frames} frames.");
        }
    }
    Ok(())
}
