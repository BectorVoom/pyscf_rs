//! BIND-06 lint: forbid `lazy_static!` in crates/pyscf-py/.
//!
//! pyscf-rs has no `lazy_static!` today, so this lint is preventive — under
//! free-threaded Python (3.13t), `lazy_static!` deadlocks because it uses
//! `std::sync::Once` without PyO3-aware coordination. Phase 3 plan 03-07 uses
//! `pyo3::sync::PyOnceLock` instead.
//!
//! Source: BIND-06 + RESEARCH §"Pitfall (NEW): abi3 + Python 3.13t ABI
//! incompatibility" + §"Pattern 1" PyOverrideBridge using `PyOnceLock`.
//!
//! Exit codes:
//!   0 — PASS (no `lazy_static!` found, or pyscf-py crate not yet present)
//!   2 — FAIL (lazy_static! found in crates/pyscf-py/**)
use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn main() -> Result<ExitCode> {
    let root = workspace_root().join("crates/pyscf-py");
    if !root.exists() {
        eprintln!(
            "check-forbid-lazy-static: pyscf-py crate not yet present at {} — PASS",
            root.display()
        );
        return Ok(ExitCode::from(0));
    }
    let hits = scan(&root)?;
    if hits.is_empty() {
        eprintln!(
            "check-forbid-lazy-static: PASS — no `lazy_static!` in {} (BIND-06)",
            root.display()
        );
        Ok(ExitCode::from(0))
    } else {
        eprintln!("check-forbid-lazy-static: FAIL — `lazy_static!` invocations found:");
        for h in &hits {
            eprintln!("  {}", h);
        }
        eprintln!("Use `pyo3::sync::PyOnceLock` instead (BIND-06, free-thread-safe).");
        Ok(ExitCode::from(2))
    }
}

/// Walk up from CWD until we find a Cargo.toml whose contents declare
/// `[workspace]` — that's the repo root. Falls back to CWD if nothing matches
/// so the binary still works when invoked via `cargo run -p xtask --bin ...`.
fn workspace_root() -> PathBuf {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut cur: &Path = &cwd;
    loop {
        let candidate = cur.join("Cargo.toml");
        if candidate.exists() {
            if let Ok(s) = fs::read_to_string(&candidate) {
                if s.contains("[workspace]") {
                    return cur.to_path_buf();
                }
            }
        }
        match cur.parent() {
            Some(p) => cur = p,
            None => return cwd,
        }
    }
}

fn scan(dir: &Path) -> Result<Vec<String>> {
    let mut hits = vec![];
    for entry in fs::read_dir(dir).with_context(|| format!("read_dir {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().and_then(|s| s.to_str()) == Some("target") {
                continue;
            }
            hits.extend(scan(&path)?);
        } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
            let content = fs::read_to_string(&path)
                .with_context(|| format!("reading {}", path.display()))?;
            for (i, line) in content.lines().enumerate() {
                let trimmed = line.trim_start();
                if trimmed.starts_with("//") {
                    continue;
                }
                if line.contains("lazy_static!") {
                    hits.push(format!("{}:{}: {}", path.display(), i + 1, line.trim()));
                }
            }
        }
    }
    Ok(hits)
}
