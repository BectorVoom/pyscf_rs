//! Forbidden paths lint implementation and needle definitions.

use anyhow::{Context, Result};
use std::fs;
use std::path::Path;
use walkdir::WalkDir;

/// Out-of-scope upstream-PySCF module imports per .planning/pbc/PBC-MASTER-PLAN.md §1.
/// The `use pyscf::` prefix is unique enough to avoid collisions with same-named
/// modules from other crates.
/// Note: `use pyscf::pbc` is removed in v2.0 as periodic calculations are now in scope.
pub const FORBIDDEN_IMPORT_NEEDLES: &[&str] = &[
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

/// Returns true if the given path belongs to a periodic (`pyscf-pbc-*`) crate.
pub fn is_pbc_exempt_path(path: &Path) -> bool {
    let s = path.to_string_lossy();
    s.contains("crates/pyscf-pbc-") || s.contains("crates\\pyscf-pbc-") || s.starts_with("pyscf-pbc-")
}

/// Check content of a single file against forbidden needles.
/// Returns a list of violation messages.
pub fn check_file_content(path: &Path, content: &str) -> Vec<String> {
    if is_pbc_exempt_path(path) {
        return Vec::new();
    }
    let mut violations = Vec::new();
    for (lineno, line) in content.lines().enumerate() {
        for needle in FORBIDDEN_IMPORT_NEEDLES {
            if line.contains(needle) {
                violations.push(format!(
                    "{}:{}: forbidden import `{needle}`",
                    path.display(),
                    lineno + 1,
                ));
            }
        }
    }
    violations
}

/// Scan crates directory for forbidden imports.
pub fn scan_crates_dir(crates_dir: &Path, root: &Path) -> Result<(usize, Vec<String>)> {
    let mut violations = Vec::new();
    let mut scanned = 0usize;

    if !crates_dir.is_dir() {
        return Ok((0, violations));
    }

    for entry in WalkDir::new(crates_dir).into_iter().filter_map(Result::ok) {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        // Skip target/ and .git/ subdirs.
        if path.components().any(|c| {
            let s = c.as_os_str().to_string_lossy();
            s == "target" || s == ".git"
        }) {
            continue;
        }
        let rel_path = path.strip_prefix(root).unwrap_or(path);
        if is_pbc_exempt_path(rel_path) {
            continue;
        }

        scanned += 1;
        let content =
            fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        violations.extend(check_file_content(rel_path, &content));
    }

    Ok((scanned, violations))
}
