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
//! D-PBC-14 / PyO3 wall (Phase 20 plan 20-16): a SECOND, independent rule.
//! No workspace member other than `PYO3_ALLOWED_CRATES` may name `pyo3`,
//! `pyo3-build-config` or `numpy` (rust-numpy) as a dependency of ANY kind
//! (normal, dev or build) — method crates, and every `pyscf-pbc-*` crate in
//! particular, stay pyo3-free; only the binding crate touches Python. The
//! cubecl rule above is unchanged by this addition.
//!
//! Exit codes:
//!   0 — PASS (both rules)
//!   1 — `cargo metadata` invocation / parse error (anyhow bail)
//!   2 — FAIL: a non-carve-out pyscf-* crate names cubecl-* in [dependencies],
//!       and/or a non-allowlisted workspace member names pyo3/numpy

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
    // `cube-math` is a cubecl crate wearing a libm's name: its public surface is
    // `#[cube]` DEVICE functions, and it re-exports cubecl to every dependent.
    // Letting a method crate depend on it both breaches ALG-06 transitively and
    // invites the failure that motivated this entry — every entry point
    // launders its argument through `bits::opaque64`, whose `RuntimeCell` has
    // no native implementation, so a HOST call panics at runtime with
    // "Unexpanded Cube functions should not be called" rather than failing to
    // compile. Kernel crates use it inside kernels; host code uses `std`.
    "cube-math",
];

/// Crates permitted to consume cubecl-* (ALG-06 carve-out).
///
/// pyscf-kernels added in Phase 2 D-04: the eval_gto cubecl kernel lives
/// here per the cintx-cubecl / xcfun-kernels split established in Phase 1.
/// Phase 4 (DFT) will land grid loops + libxc/xcfun bridges in the same
/// crate. Method crates (pyscf-gto, pyscf-scf, pyscf-dft, …) still go
/// through pyscf-algebra; this carve-out is for the kernel home only.
const ALLOWED_CRATES: &[&str] = &[
    "pyscf-algebra",
    "pyscf-runtime",
    "pyscf-kernels",
    "pyscf-bench",
];

/// Python-binding crates that may NOT appear as a dependency (of any kind) of
/// any workspace member other than those listed in `PYO3_ALLOWED_CRATES`
/// (D-PBC-14; 04-09-PLAN "pyscf-dft stays pyo3-free; only pyscf-py names
/// pyo3"). `numpy` is the rust-numpy crate, which re-exports pyo3.
const PYO3_FORBIDDEN_DEPS: &[&str] = &["pyo3", "pyo3-build-config", "numpy"];

/// Crates permitted to name the `PYO3_FORBIDDEN_DEPS`.
const PYO3_ALLOWED_CRATES: &[&str] = &[
    // The PyO3 wheel (BIND-01) — the one crate that binds Python.
    "pyscf-py",
    // Pre-existing, legitimate exception found when the rule was added (20-16):
    // the upstream-PySCF test oracle drives Python in-process through
    // `pyo3 = { optional = true, features = ["auto-initialize"] }`, gated
    // behind its non-default `python` feature. It is a TEST harness, never a
    // normal dependency of `pyscf-py` or of any method crate (ORACLE-01:
    // release wheels never link Python through it), so it does not breach
    // the wall. Method crates reach it via dev-dependencies only.
    "pyscf-oracle",
];

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
    let mut pyo3_violations = Vec::new();
    for pkg in packages {
        let name = pkg["name"].as_str().unwrap_or("");
        // `--no-deps` lists workspace members only, so this covers every crate
        // in the tree (including non-`pyscf-*` ones such as xtask).
        if PYO3_ALLOWED_CRATES.contains(&name) {
            continue;
        }
        let deps = pkg["dependencies"].as_array().unwrap_or(&empty);
        for dep in deps {
            // `name` is the package name even when the dep is renamed.
            let dep_name = dep["name"].as_str().unwrap_or("");
            if !PYO3_FORBIDDEN_DEPS.contains(&dep_name) {
                continue;
            }
            let kind = match dep["kind"].as_str() {
                None | Some("") => "normal",
                Some(k) => k,
            };
            pyo3_violations.push(format!(
                "{name}: declares {kind} dep on `{dep_name}` — D-PBC-14 PyO3 wall forbids; \
                 only {PYO3_ALLOWED_CRATES:?} may name {PYO3_FORBIDDEN_DEPS:?}"
            ));
        }
    }

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
    } else {
        eprintln!("check-dependency-wall: FAIL — ALG-06 violation:");
        for v in &violations {
            eprintln!("  - {v}");
        }
        eprintln!("\nFix: route through pyscf-algebra's public surface (Tensor + free fns).");
        eprintln!("Reference: docs/manual/Cubecl/ + RESEARCH.md Architecture Patterns.");
    }

    if pyo3_violations.is_empty() {
        eprintln!("check-dependency-wall: PASS — PyO3 wall intact (D-PBC-14)");
    } else {
        eprintln!("check-dependency-wall: FAIL — D-PBC-14 PyO3 wall violation:");
        for v in &pyo3_violations {
            eprintln!("  - {v}");
        }
        eprintln!(
            "\nFix: keep the method crate pyo3-free; put #[pyclass]/#[pyfunction] \
             wrappers in crates/pyscf-py."
        );
    }

    if violations.is_empty() && pyo3_violations.is_empty() {
        Ok(ExitCode::SUCCESS)
    } else {
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
            bail!(
                "no [workspace] root found from {:?}",
                std::env::current_dir()?
            );
        }
    }
}
