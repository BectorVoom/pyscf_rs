//! GRAD-10 workspace gate: exactly ONE CPHF `pub fn solve` may exist.
//!
//! `crates/pyscf-grad/src/cphf.rs::solve` is the single matrix-free Krylov
//! CPHF/CPKS solver. Phase 19 is exactly when a second would be added — by
//! someone writing a periodic response and finding the molecular one
//! inconveniently shaped. 19-03 supplies a k-aware `fvind` and reuses
//! `cphf::solve`; it must not add a second solver, and this lint (19-02 Task 3)
//! makes that machine-checked rather than remembered.
//!
//! Rule: across every `crates/*/src/**/*.rs`, the token `pub fn solve(`
//! (bare `solve` — `solve_linear`/`solve_lambda`/`solve_cphf_rhf` and friends
//! are different names and unaffected) may appear in exactly one file:
//! `crates/pyscf-grad/src/cphf.rs`.
//!
//! Exit codes:
//!   0 — PASS (exactly one site, in the right file)
//!   1 — invocation / workspace-discovery error
//!   2 — FAIL: zero sites, or any site outside the allowlist

use anyhow::{Context, Result, bail};
use std::process::ExitCode;

/// The single file allowed to define `pub fn solve(`.
const ALLOWED: &str = "crates/pyscf-grad/src/cphf.rs";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("check-single-cphf: FAIL: {e:#}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<()> {
    let root = workspace_root()?;
    let mut sites: Vec<String> = Vec::new();
    visit(&root.join("crates"), &mut sites)?;
    if sites.is_empty() {
        bail!("no `pub fn solve(` site found anywhere — the solver itself is missing");
    }
    let bad: Vec<_> = sites.iter().filter(|s| !s.ends_with(ALLOWED)).collect();
    if !bad.is_empty() {
        bail!(
            "second CPHF solver site(s) outside {ALLOWED}:\n  {}",
            bad.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("\n  ")
        );
    }
    if sites.len() != 1 {
        bail!("expected exactly one `pub fn solve(` site, found {}: {sites:?}", sites.len());
    }
    println!("check-single-cphf: PASS (sole site: {ALLOWED})");
    Ok(())
}

fn workspace_root() -> Result<std::path::PathBuf> {
    let out = std::process::Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .context("git rev-parse failed")?;
    if !out.status.success() {
        bail!("git rev-parse --show-toplevel failed");
    }
    Ok(std::path::PathBuf::from(
        String::from_utf8_lossy(&out.stdout).trim().to_string(),
    ))
}

fn visit(dir: &std::path::Path, sites: &mut Vec<String>) -> Result<()> {
    for entry in std::fs::read_dir(dir).with_context(|| format!("read_dir {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            // Skip build artefacts and test trees: the rule governs shipped
            // `src/` code (mirrors check-dependency-wall's scope note).
            if name == "target" || name == "tests" || name == ".git" {
                continue;
            }
            visit(&path, sites)?;
            continue;
        }
        if path.extension().map(|e| e == "rs").unwrap_or(false) {
            let text = std::fs::read_to_string(&path)
                .with_context(|| format!("read {}", path.display()))?;
            for line in text.lines() {
                let t = line.trim_start();
                // Match `pub fn solve(` exactly: `solve` followed by `(` or
                // generic params — never `solve_foo`.
                if t.starts_with("pub fn solve(") || t.starts_with("pub fn solve<") {
                    sites.push(path.to_string_lossy().into_owned());
                    break;
                }
            }
        }
    }
    Ok(())
}
