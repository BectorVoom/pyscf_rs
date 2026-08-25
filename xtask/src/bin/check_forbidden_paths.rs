//! FOUND-08 + Pitfall 21 (scope creep): refuse imports from out-of-scope
//! upstream PySCF modules. Walkdir over `crates/**/*.rs`; per-file string
//! match against `FORBIDDEN_IMPORT_NEEDLES`.
//!
//! Exemption: files under `crates/pyscf-pbc-*` are exempt because periodic
//! boundary conditions are in-scope for v2.0 milestone (PBC-FOUND-01 / PBC-FOUND-02).
//!
//! Exit codes:
//!   0 — PASS
//!   1 — IO error (anyhow bail)
//!   2 — FAIL: an in-scope crate references an out-of-scope upstream module

use anyhow::{Context, Result, bail};
use std::path::PathBuf;
use std::process::ExitCode;
use xtask::forbidden_paths::scan_crates_dir;

fn main() -> Result<ExitCode> {
    let root = workspace_root()?;
    let crates_dir = root.join("crates");

    let (scanned, violations) = scan_crates_dir(&crates_dir, &root)?;

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
            "\nReference: .planning/pbc/PBC-MASTER-PLAN.md §1 — x2c/mcscf/tdscf/adc/\n\
             gw/eom/nac/eph defer to molecular scope; periodic variants are under crates/pyscf-pbc-*."
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
