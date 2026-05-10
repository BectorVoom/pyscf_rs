//! FOUND-08 + Pitfall 21 (scope creep): refuse imports from out-of-scope
//! upstream PySCF modules. Walkdir over `crates/**/*.rs`; per-file string
//! match against `FORBIDDEN_IMPORT_NEEDLES`.
//!
//! Adapted from ~/Documents/workspace/xcfun_rs/xtask/src/bin/check_no_anyhow.rs
//! (PATTERNS row "check_forbidden_paths") — replaces the toml-parse
//! shell with a plain string grep over `.rs` files (we're checking
//! source `use`-statements, not Cargo.toml manifests).
//!
//! The needle list mirrors REQUIREMENTS.md "Out of Scope" — pbc, x2c,
//! mcscf, mcpdft, mrpt, tdscf, tddft, adc, gw, eom, nac, eph. Every
//! module in that list defers to v1.x or later; importing from any of
//! them would silently widen the scope of the rewrite and explode the
//! Phase 1 test surface.
//!
//! Exit codes:
//!   0 — PASS
//!   1 — IO error (anyhow bail)
//!   2 — FAIL: an in-scope crate references an out-of-scope upstream module

use anyhow::{Context, Result, bail};
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;
use walkdir::WalkDir;

/// Out-of-scope upstream-PySCF module imports per REQUIREMENTS.md
/// "Out of Scope" + ROADMAP "Out of Scope". The `use pyscf::` prefix
/// is unique enough to avoid collisions with same-named modules from
/// other crates (e.g. our own pyscf-* workspace crates use
/// `use pyscf_runtime::` / `use pyscf_algebra::`, never
/// `use pyscf::`).
const FORBIDDEN_IMPORT_NEEDLES: &[&str] = &[
    "use pyscf::pbc",
    "use pyscf::x2c",
    "use pyscf::mcscf",
    "use pyscf::mcpdft",
    "use pyscf::mrpt",
    "use pyscf::tdscf",
    "use pyscf::tddft",
    "use pyscf::adc",
    "use pyscf::gw",
    "use pyscf::eom",
    "use pyscf::nac",
    "use pyscf::eph",
];

fn main() -> Result<ExitCode> {
    let root = workspace_root()?;
    let crates_dir = root.join("crates");
    let mut violations = Vec::new();
    let mut scanned = 0usize;

    if !crates_dir.is_dir() {
        eprintln!(
            "check-forbidden-paths: WARN — {} not a directory; nothing to scan.",
            crates_dir.display()
        );
        return Ok(ExitCode::SUCCESS);
    }

    for entry in WalkDir::new(&crates_dir).into_iter().filter_map(Result::ok) {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        // Skip target/ and .git/ subdirs (defensive — they shouldn't appear
        // inside crates/ but a stray symlink or build artefact must not
        // poison the scan).
        if path.components().any(|c| {
            let s = c.as_os_str().to_string_lossy();
            s == "target" || s == ".git"
        }) {
            continue;
        }
        scanned += 1;
        let content = fs::read_to_string(path)
            .with_context(|| format!("read {}", path.display()))?;
        for (lineno, line) in content.lines().enumerate() {
            for needle in FORBIDDEN_IMPORT_NEEDLES {
                if line.contains(needle) {
                    violations.push(format!(
                        "{}:{}: forbidden import `{needle}`",
                        path.strip_prefix(&root).unwrap_or(path).display(),
                        lineno + 1,
                    ));
                }
            }
        }
    }

    if violations.is_empty() {
        eprintln!(
            "check-forbidden-paths: PASS — {scanned} .rs file(s); no out-of-scope upstream PySCF imports (FOUND-08)"
        );
        Ok(ExitCode::SUCCESS)
    } else {
        eprintln!("check-forbidden-paths: FAIL — out-of-scope import detected (Pitfall 21):");
        for v in &violations {
            eprintln!("  - {v}");
        }
        eprintln!(
            "\nReference: REQUIREMENTS.md \"Out of Scope\" — pbc/x2c/mcscf/tdscf/adc/\n\
             gw/eom/nac/eph defer to v1.x or later."
        );
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
            bail!("no [workspace] root found");
        }
    }
}
