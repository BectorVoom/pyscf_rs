//! FOUND-07 + Pitfall 14: every `.rs` file containing an `extern "C"`
//! block must also contain `catch_unwind`. Phase 1 has zero such blocks
//! — the gate is forward-looking for Phase 3+ FFI surfaces (PyO3 +
//! cintx FFI shims). The pairing is the minimum-viable safeguard
//! against panics escaping across the C ABI boundary causing UB
//! (RESEARCH §"FFI panic safety", Pitfall 14).
//!
//! The check is per-file, not per-block: any file that exposes an
//! `extern "C"` surface must locally know about `catch_unwind` at some
//! site. Pre-merge code review (Plan 06 reviewer checklist) catches
//! the rare case where a separate file imports the extern declarations
//! but the catch_unwind lives elsewhere.
//!
//! Single-line `// extern "C"` doc comments are stripped before
//! matching, so doc/comment occurrences don't trigger.
//!
//! Exit codes:
//!   0 — PASS
//!   1 — IO error (anyhow bail)
//!   2 — FAIL: at least one file has `extern "C"` without `catch_unwind`

use anyhow::{Context, Result, bail};
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;
use walkdir::WalkDir;

fn main() -> Result<ExitCode> {
    let root = workspace_root()?;
    let crates_dir = root.join("crates");
    let mut violations = Vec::new();
    let mut scanned = 0usize;

    if !crates_dir.is_dir() {
        eprintln!(
            "check-catch-unwind: WARN — {} not a directory; nothing to scan.",
            crates_dir.display()
        );
        return Ok(ExitCode::SUCCESS);
    }

    for entry in WalkDir::new(&crates_dir).into_iter().filter_map(Result::ok) {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        if path.components().any(|c| {
            let s = c.as_os_str().to_string_lossy();
            s == "target" || s == ".git"
        }) {
            continue;
        }
        scanned += 1;
        let content = fs::read_to_string(path)
            .with_context(|| format!("read {}", path.display()))?;

        // Strip line comments so `// extern "C"` in doc text doesn't
        // trigger. Block comments are left as-is — they're rare in the
        // codebase, and pre-merge review handles the edge case.
        let non_comment: String = content
            .lines()
            .map(|l| {
                if let Some(idx) = l.find("//") { &l[..idx] } else { l }
            })
            .collect::<Vec<_>>()
            .join("\n");

        let has_extern_c = non_comment.contains("extern \"C\"");
        let has_catch_unwind = non_comment.contains("catch_unwind");
        if has_extern_c && !has_catch_unwind {
            violations.push(format!(
                "{}: contains `extern \"C\"` but no `catch_unwind` — Pitfall 14 unguarded FFI",
                path.strip_prefix(&root).unwrap_or(path).display(),
            ));
        }
    }

    if violations.is_empty() {
        eprintln!(
            "check-catch-unwind: PASS — {scanned} .rs file(s); every `extern \"C\"` site pairs with `catch_unwind` (FOUND-07)"
        );
        Ok(ExitCode::SUCCESS)
    } else {
        eprintln!("check-catch-unwind: FAIL — unguarded FFI surface (Pitfall 14):");
        for v in &violations {
            eprintln!("  - {v}");
        }
        eprintln!(
            "\nFix: wrap the extern \"C\" body in `std::panic::catch_unwind` so panics\n\
             don't escape across the FFI boundary causing UB."
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
