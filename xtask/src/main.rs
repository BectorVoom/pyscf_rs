//! Default xtask entry point — runs all five checks sequentially.
//!
//! Usage:
//!   cargo run -p xtask                       # runs all 5 checks
//!   cargo run -p xtask --bin check-no-fma    # one specific check
//!
//! Each individual check is a separate [[bin]] in Cargo.toml.
//!
//! Exit codes:
//!   0 — all 5 checks passed
//!   2 — at least one check failed (see stderr for the failing check name)

use std::process::{Command, ExitCode};

fn main() -> ExitCode {
    let checks = [
        "check-no-fma",
        "check-forbidden-paths",
        "check-catch-unwind",
        "check-dependency-wall",
        "check-cubecl-pin",
    ];
    let mut all_passed = true;
    for check in &checks {
        eprintln!("--- xtask: running {check} ---");
        let status = Command::new("cargo")
            .args(["run", "--quiet", "-p", "xtask", "--bin", check])
            .status();
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
