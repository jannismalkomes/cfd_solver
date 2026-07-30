//! Visualisation is delegated to the bundled Python script `visualize.py`
//! (numpy + matplotlib), invoked automatically after a run. The script is
//! embedded in the binary at compile time and written to a temp file before
//! running, so it works regardless of the current directory.
//!
//! This keeps the Rust crate's only dependency to the multithreading library;
//! Python/matplotlib is an optional external tool, not a cargo dependency. If
//! `python3` (or the required packages) is unavailable, figures are skipped
//! with a helpful message rather than failing the run.

use std::process::Command;

const SCRIPT: &str = include_str!("python_helper/visualize.py");

/// Render the figures for `outdir` by running the bundled Python visualiser.
/// With `show = true`, the figures are also displayed interactively.
pub fn generate(outdir: &str, show: bool) {
    let script_path = std::env::temp_dir().join("cfd_solver_visualize.py");
    if let Err(e) = std::fs::write(&script_path, SCRIPT) {
        eprintln!("visualisation: could not write temp script ({e}); skipping figures.");
        return;
    }

    // Try `python3`, then `python`, then the Windows `py` launcher.
    for exe in ["python3", "python", "py"] {
        let mut cmd = Command::new(exe);
        cmd.arg(&script_path).arg(outdir);
        if show {
            cmd.arg("--show");
        }
        match cmd.status() {
            Ok(status) if status.success() => return,
            Ok(status) => {
                eprintln!("visualisation: {exe} exited with {status}.");
                return;
            }
            Err(_) => continue, // this interpreter not found; try the next
        }
    }

    eprintln!(
        "visualisation: could not run python3/python. Figures were skipped.\n\
         Install numpy + matplotlib, then run:  python3 visualize.py {outdir}"
    );
}
