//! Default xtask entry point — runs every check sequentially.
//!
//! Usage:
//!   cargo run -p xtask                       # runs all checks
//!   cargo run -p xtask --bin check-no-fma    # one specific check
//!
//! Each individual check is a separate [[bin]] in Cargo.toml.
//!
//! Exit codes:
//!   0 — all checks passed
//!   2 — at least one check failed (see stderr for the failing check name)

use std::process::{Command, ExitCode};

fn main() -> ExitCode {
    let checks = [
        "check-no-fma",
        "check-forbidden-paths",
        "check-catch-unwind",
        "check-dependency-wall",
        "check-cubecl-pin",
        "check-orphan-modules",
    ];
    let current_exe = std::env::current_exe().ok();
    let bin_dir = current_exe.as_ref().and_then(|p| p.parent());

    let mut all_passed = true;
    for check in &checks {
        eprintln!("--- xtask: running {check} ---");
        let mut cmd = if let Some(dir) = bin_dir {
            let bin_path = dir.join(check);
            if bin_path.exists() {
                Command::new(bin_path)
            } else {
                let mut c = Command::new("cargo");
                c.args(["run", "--quiet", "-p", "xtask", "--bin", check]);
                c
            }
        } else {
            let mut c = Command::new("cargo");
            c.args(["run", "--quiet", "-p", "xtask", "--bin", check]);
            c
        };

        let status = cmd.status();
        let ok = matches!(status, Ok(s) if s.success());
        if !ok {
            eprintln!("--- xtask: {check} FAILED ---");
            all_passed = false;
        } else {
            eprintln!("--- xtask: {check} OK ---");
        }
    }
    if all_passed {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(2)
    }
}
