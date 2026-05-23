//! FOUND-05 + Pitfall 1 SHOWSTOPPER: ban FMA contraction in oracle-profile
//! machine code. Compiles target crates with `cargo rustc --profile
//! release-oracle -- --emit=asm`, scans `target/release-oracle/deps/*.s`
//! for forbidden FMA mnemonics, and exits non-zero on any match.
//!
//! Adapted from ~/Documents/workspace/xcfun_rs/xtask/src/bin/check_no_fma.rs
//! (PATTERNS row "check_no_fma"). RESEARCH §6 + §14: this is the
//! `cargo-llvm-ir | grep llvm.fmuladd` replacement (cargo-llvm-ir does not
//! exist on crates.io — Pitfall 1).
//!
//! Operation:
//!   1. For each `(crate_name, asm_stem)` in `SCAN_TARGETS`, run
//!      `cargo rustc --profile release-oracle -p <crate_name> --lib -- --emit=asm`.
//!   2. Locate every `.s` file under `target/release-oracle/deps/`.
//!   3. For each `.s` file, split into function blocks by symbol label.
//!      Demangle each label via `rustc_demangle::demangle`.
//!      Walk the body up to the next symbol label and grep for forbidden
//!      mnemonics (`vfmadd*`, `fmadd`, `fma213`, `fma231`, …).
//!   4. If any match → print symbol + line + exit 2.
//!      Else → exit 0.
//!
//! Phase 1 SCAN_TARGETS = [pyscf-algebra, pyscf-core]. Plan 04 lands the
//! algebra crate; Plan 06 wires this gate into CI. Other crates (pyscf-kernels
//! etc.) join the scan list as they accrete oracle-relevant numeric paths.
//!
//! Exit codes:
//!   0 — PASS (no FMA mnemonics on the scanned crates' guarded symbols)
//!   1 — build / IO error (anyhow bail or non-zero ExitCode)
//!   2 — FAIL: forbidden mnemonic detected (Pitfall 1 SHOWSTOPPER)

use anyhow::{Context, Result, bail};
use rustc_demangle::demangle;
use std::fs;
use std::path::PathBuf;
use std::process::{Command, ExitCode};

/// Forbidden FMA mnemonics — verbatim from
/// xcfun_rs/xtask/src/bin/check_no_fma.rs (PATTERNS row "check_no_fma").
const FORBIDDEN_MNEMONICS: &[&str] = &[
    // x86_64 packed double + scalar double FMA
    "vfmadd132pd",
    "vfmadd213pd",
    "vfmadd231pd",
    "vfmadd132sd",
    "vfmadd213sd",
    "vfmadd231sd",
    "vfmsub132pd",
    "vfmsub213pd",
    "vfmsub231pd",
    "vfmsub132sd",
    "vfmsub213sd",
    "vfmsub231sd",
    "vfnmadd132pd",
    "vfnmadd213pd",
    "vfnmadd231pd",
    "vfnmadd132sd",
    "vfnmadd213sd",
    "vfnmadd231sd",
    "vfnmsub132pd",
    "vfnmsub213pd",
    "vfnmsub231pd",
    "vfnmsub132sd",
    "vfnmsub213sd",
    "vfnmsub231sd",
    // aarch64 + generic spellings
    "fmadd",
    "fmsub",
    "fnmadd",
    "fnmsub",
    // LLVM-intrinsic-style (belt-and-suspenders)
    "fma213",
    "fma231",
];

/// `(crate-name, asm-filename-stem)` pairs to scan. The asm stem is the
/// crate name with `-` → `_` (cargo's filename convention).
///
/// Phase 1 covers pyscf-algebra (lands in Plan 04) and pyscf-core. Future
/// plans add pyscf-kernels (Phase 4) and any other crate that materially
/// contributes to oracle-comparison numerics.
const SCAN_TARGETS: &[(&str, &str)] = &[
    ("pyscf-algebra", "pyscf_algebra"),
    ("pyscf-core", "pyscf_core"),
];

fn main() -> Result<ExitCode> {
    let root = workspace_root()?;
    let mut asm_files: Vec<PathBuf> = Vec::new();

    for (crate_name, asm_stem) in SCAN_TARGETS {
        if !is_in_workspace(&root, crate_name)? {
            eprintln!(
                "check-no-fma: {crate_name} not in workspace yet; skipping (forward-compatible per PATTERNS)."
            );
            continue;
        }

        eprintln!("check-no-fma: building {crate_name} under release-oracle...");
        let status = Command::new("cargo")
            .current_dir(&root)
            .args([
                "rustc",
                "--profile",
                "release-oracle",
                "-p",
                crate_name,
                "--lib",
                "--",
                "--emit=asm",
            ])
            .status()
            .with_context(|| format!("cargo rustc failed for {crate_name}"))?;
        if !status.success() {
            eprintln!("check-no-fma: build failed for {crate_name}");
            return Ok(ExitCode::from(1));
        }

        // Find all <stem>-*.s files under target/release-oracle/deps/.
        let deps_dir = root.join("target").join("release-oracle").join("deps");
        let mut found = 0usize;
        for entry in
            fs::read_dir(&deps_dir).with_context(|| format!("read_dir {}", deps_dir.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            let ext_ok = path.extension().and_then(|s| s.to_str()) == Some("s");
            let stem_ok = path
                .file_stem()
                .and_then(|s| s.to_str())
                .map(|s| s.starts_with(asm_stem))
                .unwrap_or(false);
            if ext_ok && stem_ok {
                asm_files.push(path);
                found += 1;
            }
        }
        if found == 0 {
            bail!(
                "no {asm_stem}-*.s files under {} — did `cargo rustc --emit=asm` actually run?",
                deps_dir.display()
            );
        }
    }

    if asm_files.is_empty() {
        eprintln!(
            "check-no-fma: no SCAN_TARGETS resolved — workspace is missing all of [{}].",
            SCAN_TARGETS
                .iter()
                .map(|(n, _)| *n)
                .collect::<Vec<_>>()
                .join(", ")
        );
        return Ok(ExitCode::SUCCESS);
    }
    eprintln!("check-no-fma: scanning {} asm file(s)", asm_files.len());

    let mut violations: Vec<String> = Vec::new();
    for path in &asm_files {
        let content =
            fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        scan_asm_file(&content, &path.display().to_string(), &mut violations);
    }

    if violations.is_empty() {
        eprintln!("check-no-fma: PASS — no FMA mnemonics in release-oracle asm (FOUND-05)");
        Ok(ExitCode::SUCCESS)
    } else {
        eprintln!("check-no-fma: FAIL — FMA contraction detected (Pitfall 1 SHOWSTOPPER):");
        for v in violations.iter().take(20) {
            eprintln!("  - {v}");
        }
        if violations.len() > 20 {
            eprintln!("  ... ({} more)", violations.len() - 20);
        }
        eprintln!(
            "\nThe 1e-12 PySCF-oracle parity contract cannot be trusted with FMA contraction\n\
             on hot numeric paths. Inspect the offending crate's loops; insert\n\
             `core::hint::black_box` or rewrite to suppress contraction; or escalate\n\
             via PLANNING INCONCLUSIVE per RESEARCH §6 / Pitfall 1."
        );
        Ok(ExitCode::from(2))
    }
}

/// Cheap substring check on `cargo metadata --no-deps` JSON to decide whether
/// `crate_name` is a workspace member. Avoids pulling in serde_json here.
fn is_in_workspace(root: &PathBuf, crate_name: &str) -> Result<bool> {
    let output = Command::new("cargo")
        .current_dir(root)
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .output()
        .context("spawning cargo metadata")?;
    if !output.status.success() {
        return Ok(false);
    }
    let text = String::from_utf8_lossy(&output.stdout);
    Ok(text.contains(&format!("\"name\":\"{crate_name}\"")))
}

/// The no-FMA contract is enforceable only on pyscf-rs's OWN code. Third-party
/// SIMD backends (`pulp` — faer's vectorize layer — and `gemm`) annotate their
/// kernels `#[target_feature(enable="...,fma")]`, which force-emits FMA
/// regardless of the workspace `-C target-feature=-fma` (that's how their
/// runtime ISA dispatch works). pyscf-rs cannot suppress that, and PySCF's own
/// LAPACK/BLAS uses FMA too — so the 1e-12 parity is between two FMA-using
/// dense-linalg backends either way. A symbol is therefore a violation ONLY
/// when its defining crate is a pyscf-rs crate (`pyscf_*`).
fn symbol_is_pyscf_owned(sym: Option<&str>) -> bool {
    match sym {
        // Strip a leading trait-impl wrapper (`<Type as Trait>::method`) so the
        // defining type's crate prefix is what we test.
        Some(s) => s.trim_start_matches('<').starts_with("pyscf_"),
        None => false,
    }
}

/// Scan a single asm file: split by symbol labels (lines ending in `:` that
/// are not assembler directives or empty), demangle each, and grep the
/// symbol body for forbidden mnemonics. Only pyscf-rs-owned symbols are
/// flagged — see [`symbol_is_pyscf_owned`].
fn scan_asm_file(contents: &str, path: &str, violations: &mut Vec<String>) {
    let mut current_sym: Option<String> = None;

    for (lineno, line) in contents.lines().enumerate() {
        let trimmed = line.trim_start();

        // Detect a new symbol label.
        if let Some(stripped) = trimmed.strip_suffix(':')
            && !stripped.starts_with('.')
            && !stripped.is_empty()
        {
            let demangled = demangle(stripped).to_string();
            current_sym = Some(demangled);
            continue;
        }

        // Inside a symbol body — grep for FMA mnemonics. Whole-word match
        // anchored at the trimmed line start so we don't false-match on
        // operand text or comment fragments.
        for mnemonic in FORBIDDEN_MNEMONICS {
            if trimmed.starts_with(mnemonic) {
                let next = trimmed.as_bytes().get(mnemonic.len()).copied();
                let is_word_boundary = matches!(next, None | Some(b' ') | Some(b'\t') | Some(b','));
                // Only pyscf-rs's own code is in scope; third-party SIMD
                // (pulp/gemm) FMA is unavoidable and out of contract.
                if is_word_boundary && symbol_is_pyscf_owned(current_sym.as_deref()) {
                    violations.push(format!(
                        "{path}:{} [{}] {}",
                        lineno + 1,
                        current_sym.as_deref().unwrap_or("<unknown>"),
                        line.trim()
                    ));
                }
            }
        }
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
