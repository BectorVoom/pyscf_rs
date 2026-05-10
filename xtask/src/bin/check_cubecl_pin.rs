//! FOUND-04: cubecl 0.10.0 lockstep verification across siblings.
//!
//! Walks `cargo metadata --format-version 1` (full graph) and asserts:
//!   * cubecl, cubecl-cpu, cubecl-cuda, cubecl-hip, cubecl-runtime,
//!     cubecl-wgpu  →  exactly `0.10.0`
//!   * cubecl-matmul, cubecl-reduce  →  exactly `0.9.0-pre.5`  (the
//!     pre-release ABI that interoperates with cubecl 0.10.0; the
//!     0.10.0 publishes for these two crates do not exist on
//!     crates.io as of 2026-05-10 — RESEARCH §"Standard Stack" /
//!     Pitfall 1).
//!
//! Adapted from ~/Documents/workspace/xcfun_rs/xtask/src/bin/check_cubecl_pin.rs
//! (PATTERNS row "check_cubecl_pin"). Extends the PINNED list with cubecl-runtime
//! and adds the PRE_PINNED bucket for cubecl-matmul / cubecl-reduce.
//!
//! Crates not yet pulled into the resolved dep graph are silently skipped — the
//! gate enforces lockstep only on what's present. Once a sibling crate is added,
//! it joins the enforced set automatically (matches the xcfun_rs precedent so
//! Plan 02 / Plan 06 GPU additions Just Work).
//!
//! Exit codes:
//!   0 — PASS
//!   1 — `cargo metadata` invocation / parse error (anyhow bail)
//!   2 — FAIL: any cubecl-family crate at the wrong pinned version

use anyhow::{Context, Result, bail};
use serde_json::Value;
use std::path::PathBuf;
use std::process::{Command, ExitCode};

/// Crates that MUST equal `REQUIRED_VERSION`.
const REQUIRED_VERSION: &str = "0.10.0";
const PINNED_CRATES: &[&str] = &[
    "cubecl",
    "cubecl-cpu",
    "cubecl-cuda",
    "cubecl-hip",
    "cubecl-runtime",
    "cubecl-wgpu",
];

/// cubecl-matmul / cubecl-reduce 0.10.0 unpublished as of 2026-05-10;
/// pinned at the pre-release with the matching ABI per RESEARCH
/// §"Standard Stack" / Pitfall 1.
const PRE_REQUIRED_VERSION: &str = "0.9.0-pre.5";
const PRE_PINNED_CRATES: &[&str] = &["cubecl-matmul", "cubecl-reduce"];

fn main() -> Result<ExitCode> {
    let root = workspace_root()?;
    let output = Command::new("cargo")
        .current_dir(&root)
        .args(["metadata", "--format-version", "1"])
        .output()
        .context("failed to spawn `cargo metadata`")?;
    if !output.status.success() {
        bail!(
            "cargo metadata failed (exit {:?}): {}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let metadata: Value =
        serde_json::from_slice(&output.stdout).context("parse cargo metadata JSON")?;

    let empty: Vec<Value> = Vec::new();
    let packages = metadata["packages"].as_array().unwrap_or(&empty);

    let mut violations = Vec::new();
    let mut seen_pinned = 0usize;
    let mut seen_pre = 0usize;

    for pkg in packages {
        let name = pkg["name"].as_str().unwrap_or("");
        let version = pkg["version"].as_str().unwrap_or("");
        if PINNED_CRATES.contains(&name) {
            seen_pinned += 1;
            if version != REQUIRED_VERSION {
                violations.push(format!(
                    "{name}: version {version} (expected {REQUIRED_VERSION})"
                ));
            }
        }
        if PRE_PINNED_CRATES.contains(&name) {
            seen_pre += 1;
            if version != PRE_REQUIRED_VERSION {
                violations.push(format!(
                    "{name}: version {version} (expected {PRE_REQUIRED_VERSION})"
                ));
            }
        }
    }

    if violations.is_empty() {
        eprintln!(
            "check-cubecl-pin: PASS — {seen_pinned} crate(s) at {REQUIRED_VERSION}, \
             {seen_pre} crate(s) at {PRE_REQUIRED_VERSION} (FOUND-04)"
        );
        Ok(ExitCode::SUCCESS)
    } else {
        eprintln!("check-cubecl-pin: FAIL — cubecl version drift detected (FOUND-04 / Pitfall 1):");
        for v in &violations {
            eprintln!("  - {v}");
        }
        eprintln!(
            "\nFix: align workspace [workspace.dependencies] cubecl-* pins, or update"
        );
        eprintln!(
            "the sibling-crate revs in [patch.crates-io] to ones that use the matching pin."
        );
        Ok(ExitCode::from(2))
    }
}

/// Walk parents from `cwd` until we find a `Cargo.toml` containing a
/// `[workspace]` table; return that directory as the workspace root.
fn workspace_root() -> Result<PathBuf> {
    let mut p = std::env::current_dir().context("getcwd")?;
    loop {
        let cargo = p.join("Cargo.toml");
        if cargo.is_file() {
            let s = std::fs::read_to_string(&cargo)?;
            if s.contains("[workspace]") {
                return Ok(p);
            }
        }
        if !p.pop() {
            bail!("no [workspace] root found from {:?}", std::env::current_dir()?);
        }
    }
}
