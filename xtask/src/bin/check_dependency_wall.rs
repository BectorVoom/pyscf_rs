//! ALG-06: only `pyscf-algebra` and `pyscf-runtime` may declare normal
//! `cubecl-*` deps. Adapted from
//! ~/Documents/workspace/xcfun_rs/xtask/src/bin/check_boundaries.rs
//! (PATTERNS row "check_dependency_wall") — switches the semantics from
//! allowlist to denylist per ALG-06.
//!
//! Walks `cargo metadata --format-version 1 --no-deps` and fails if any
//! workspace member whose name starts with `pyscf-` (other than the
//! ALG-06 carve-out crates) declares a normal dep on any
//! `cubecl-*`-family crate.
//!
//! Note: dev-dependencies are NOT checked (they don't affect the
//! shipped wheel). RESEARCH Pitfall 4 documents the gap; future work
//! may extend the lint to cover them.
//!
//! Exit codes:
//!   0 — PASS
//!   1 — `cargo metadata` invocation / parse error (anyhow bail)
//!   2 — FAIL: a non-carve-out pyscf-* crate names cubecl-* in [dependencies]

use anyhow::{Context, Result, bail};
use serde_json::Value;
use std::path::PathBuf;
use std::process::{Command, ExitCode};

/// Crates that may NOT appear as a normal dep of any pyscf-rs workspace
/// crate other than the carve-out crates listed in `ALLOWED_CRATES`.
const FORBIDDEN_DEPS: &[&str] = &[
    "cubecl",
    "cubecl-cpu",
    "cubecl-cuda",
    "cubecl-hip",
    "cubecl-matmul",
    "cubecl-reduce",
    "cubecl-runtime",
    "cubecl-std",
    "cubecl-wgpu",
];

/// Crates permitted to consume cubecl-* (ALG-06 carve-out).
///
/// pyscf-kernels added in Phase 2 D-04: the eval_gto cubecl kernel lives
/// here per the cintx-cubecl / xcfun-kernels split established in Phase 1.
/// Phase 4 (DFT) will land grid loops + libxc/xcfun bridges in the same
/// crate. Method crates (pyscf-gto, pyscf-scf, pyscf-dft, …) still go
/// through pyscf-algebra; this carve-out is for the kernel home only.
const ALLOWED_CRATES: &[&str] = &["pyscf-algebra", "pyscf-runtime", "pyscf-kernels"];

fn main() -> Result<ExitCode> {
    let root = workspace_root()?;
    let output = Command::new("cargo")
        .current_dir(&root)
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .output()
        .context("failed to spawn `cargo metadata --no-deps`")?;
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
    for pkg in packages {
        let name = pkg["name"].as_str().unwrap_or("");
        // Skip the carve-out crates and non-pyscf packages.
        if ALLOWED_CRATES.contains(&name) {
            continue;
        }
        // Only check pyscf-rs workspace crates (skip xtask, etc.).
        if !name.starts_with("pyscf-") {
            continue;
        }

        let deps = pkg["dependencies"].as_array().unwrap_or(&empty);
        for dep in deps {
            let dep_name = dep["name"].as_str().unwrap_or("");
            // `kind` is null/absent for normal deps; "dev" / "build" otherwise.
            let kind = dep["kind"].as_str();
            let is_normal = kind.is_none() || kind == Some("");
            if !is_normal {
                continue;
            }
            if FORBIDDEN_DEPS.contains(&dep_name) {
                violations.push(format!(
                    "{name}: declares normal dep on `{dep_name}` — ALG-06 forbids; \
                     only {ALLOWED_CRATES:?} may consume cubecl-*"
                ));
            }
        }
    }

    if violations.is_empty() {
        eprintln!("check-dependency-wall: PASS — cubecl-* containment intact (ALG-06)");
        Ok(ExitCode::SUCCESS)
    } else {
        eprintln!("check-dependency-wall: FAIL — ALG-06 violation:");
        for v in &violations {
            eprintln!("  - {v}");
        }
        eprintln!(
            "\nFix: route through pyscf-algebra's public surface (Tensor + free fns)."
        );
        eprintln!("Reference: docs/manual/Cubecl/ + RESEARCH.md Architecture Patterns.");
        Ok(ExitCode::from(2))
    }
}

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
