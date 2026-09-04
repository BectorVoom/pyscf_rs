//! Fail when a `.rs` file sits in a crate's source tree but no `mod`
//! declaration reaches it — so it is never compiled.
//!
//! # Why this check exists
//!
//! Phase 12 shipped ~5 000 lines across three crates that were never compiled:
//! `crates/pyscf-pbc-dft/src/lib.rs` declared only `mod error`, and the twelve
//! sibling modules — `numint.rs`, `krks.rs`, `kuks.rs`, … — were simply files on
//! disk. `pyscf-gto/src/basis/pydict.rs` and `pyscf-pbc-gto/src/exxdiv_vcut.rs`
//! were in the same state.
//!
//! Nothing in the toolchain catches this. `cargo build`, `cargo check`,
//! `cargo test` and `cargo clippy` all start from the crate root and walk the
//! `mod` tree; a file nothing declares is not an error, a warning, or dead code
//! — it is not part of the crate at all. So the crate compiled clean, the test
//! suite passed, and a verification document was written against code that had
//! never been built.
//!
//! Exit codes:
//!   0 — every source file is reachable from its crate root
//!   2 — at least one orphan (listed on stderr)

use anyhow::{Context, Result};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// Files that are legitimately not reachable through a `mod` chain.
///
/// `build.rs` is run by cargo, not compiled into the crate. `main.rs` and files
/// under `bin/`, `tests/`, `benches/` and `examples/` are their own compilation
/// roots. `mod.rs` and `lib.rs` ARE the declaration points.
fn is_own_root(rel: &Path) -> bool {
    let s = rel.to_string_lossy().replace('\\', "/");
    s == "build.rs"
        || s == "src/main.rs"
        || s.starts_with("src/bin/")
        || s.starts_with("tests/")
        || s.starts_with("benches/")
        || s.starts_with("examples/")
}

/// Collect every `mod <name>;` declared anywhere in `text`.
///
/// Deliberately crude: it does not parse Rust. A declaration is a line whose
/// trimmed form matches `[pub[(...)] ]mod <ident>;`. That covers every real
/// declaration and cannot produce a FALSE ORPHAN (the failure that matters) —
/// at worst an inline `mod x { }` block is also counted, which only makes the
/// check more permissive.
fn declared_mods(text: &str) -> HashSet<String> {
    let mut out = HashSet::new();
    for line in text.lines() {
        let t = line.trim();
        let t = t.strip_prefix("pub ").unwrap_or(t);
        // `pub(crate) mod x;`, `pub(super) mod x;`
        let t = if t.starts_with("pub(") {
            match t.find(')') {
                Some(i) => t[i + 1..].trim_start(),
                None => t,
            }
        } else {
            t
        };
        let Some(rest) = t.strip_prefix("mod ") else {
            continue;
        };
        let name: String = rest
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if !name.is_empty() {
            out.insert(name);
        }
    }
    out
}

/// Walk `dir`, collecting `.rs` files.
fn rs_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }
    for entry in std::fs::read_dir(dir).with_context(|| format!("read_dir {}", dir.display()))? {
        let p = entry?.path();
        if p.is_dir() {
            rs_files(&p, out)?;
        } else if p.extension().is_some_and(|e| e == "rs") {
            out.push(p);
        }
    }
    Ok(())
}

fn main() -> Result<ExitCode> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .context("workspace root")?
        .to_path_buf();
    let crates = root.join("crates");
    if !crates.is_dir() {
        eprintln!("check-orphan-modules: no crates/ directory; nothing to check");
        return Ok(ExitCode::SUCCESS);
    }

    let mut orphans: Vec<String> = Vec::new();
    let mut scanned = 0usize;

    for entry in std::fs::read_dir(&crates)? {
        let krate = entry?.path();
        let src = krate.join("src");
        if !src.is_dir() {
            continue;
        }

        // Every `mod` declared anywhere in the crate, in one set. Matching by
        // NAME rather than by path is intentional: it keeps the check simple and
        // errs toward permissive, and a module name declared somewhere in the
        // crate but pointing at a different directory is a far rarer mistake
        // than a file nobody declared at all.
        let mut files = Vec::new();
        rs_files(&src, &mut files)?;
        let mut declared: HashSet<String> = HashSet::new();
        for f in &files {
            declared.extend(declared_mods(&std::fs::read_to_string(f)?));
        }

        for f in &files {
            let rel = f.strip_prefix(&krate).unwrap_or(f);
            if is_own_root(rel) {
                continue;
            }
            let stem = f.file_stem().unwrap_or_default().to_string_lossy();
            // `lib.rs` is the root; `mod.rs` is declared by its DIRECTORY name.
            let name = if stem == "lib" {
                continue;
            } else if stem == "mod" {
                f.parent()
                    .and_then(Path::file_name)
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string()
            } else {
                stem.to_string()
            };
            scanned += 1;
            if !declared.contains(&name) {
                orphans.push(format!(
                    "{}: `{name}` is never declared with `mod {name};` — it is NOT compiled",
                    rel.display()
                ));
            }
        }
    }

    if orphans.is_empty() {
        eprintln!("check-orphan-modules: PASS — {scanned} source files, all reachable");
        return Ok(ExitCode::SUCCESS);
    }
    eprintln!(
        "check-orphan-modules: FAIL — {} orphaned source file(s):",
        orphans.len()
    );
    for o in &orphans {
        eprintln!("  {o}");
    }
    eprintln!(
        "\nAn undeclared .rs file is not a warning and not dead code — it is not part of \n\
         the crate. cargo build/check/test/clippy all pass while compiling none of it.\n\
         Add `mod <name>;` (or `pub mod <name>;`) to the parent module, or delete the file."
    );
    Ok(ExitCode::from(2))
}
