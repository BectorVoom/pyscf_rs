//! FOUND-04: cubecl 0.10.0 lockstep verification across siblings.
//!
//! Walks `cargo metadata --format-version 1` (full graph) and asserts:
//!   * cubecl, cubecl-cpu, cubecl-cuda, cubecl-hip, cubecl-runtime,
//!     cubecl-wgpu  →  exactly `0.10.0` (top-level pins)
//!   * cubecl-matmul, cubecl-reduce  →  exactly `0.9.0-pre.5` (the
//!     pre-release ABI that interoperates with cubecl 0.10.0; the
//!     0.10.0 publishes for these two crates do not exist on
//!     crates.io as of 2026-05-10 — RESEARCH §"Standard Stack" /
//!     Pitfall 1).
//!
//! Transitive carve-out (Plan 09 gap closure for VERIFICATION BLOCKER 2):
//! cubecl-runtime (and any other cubecl-* family member) at 0.9.0-pre.5
//! is ALLOWED in the resolve graph — but ONLY when:
//!   (a) the workspace [workspace.dependencies] still pins
//!       cubecl-matmul = "=0.9.0-pre.5" AND cubecl-reduce = "=0.9.0-pre.5"
//!       (i.e., the documented version skew has not yet been retired); AND
//!   (b) every reverse-dep edge of the 0.9.0-pre.5 cubecl-* node is
//!       to a crate that is itself reachable from {cubecl-matmul,
//!       cubecl-reduce} only.
//! When (a) ceases (matmul/reduce publish at 0.10.0 and the workspace
//! pin moves), the carve-out auto-disengages and the lint tightens back
//! to a unified 0.10.0 graph. When (b) is violated (e.g., a method
//! crate accidentally pulls cubecl-runtime 0.9.0-pre.5 directly), the
//! lint FAILS with a precise message naming the offending crate.
//!
//! Adapted from ~/Documents/workspace/xcfun_rs/xtask/src/bin/check_cubecl_pin.rs
//! (PATTERNS row "check_cubecl_pin"). Plan 09 extends with reverse-dep-aware
//! relaxation per VERIFICATION.md missing Option (a).
//!
//! Exit codes:
//!   0 — PASS
//!   1 — `cargo metadata` invocation / parse error (anyhow bail)
//!   2 — FAIL: any cubecl-family crate at the wrong pinned version
//!            and not covered by the matmul/reduce transitive carve-out

use anyhow::{Context, Result, bail};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

/// Crates that MUST equal `REQUIRED_VERSION` at every node in the resolve graph.
/// (No carve-out applies to these — they are the canonical 0.10.0 family.)
const REQUIRED_VERSION: &str = "0.10.0";
const PINNED_CRATES: &[&str] = &[
    "cubecl",
    "cubecl-cpu",
    "cubecl-cuda",
    "cubecl-hip",
    "cubecl-runtime",
    "cubecl-wgpu",
];

/// Crates whose top-level pin is the pre-release `0.9.0-pre.5` ABI.
/// These are the ROOTS of the carve-out: any 0.9.0-pre.5 cubecl-*
/// crate transitively reachable ONLY from these is allowed.
const PRE_REQUIRED_VERSION: &str = "0.9.0-pre.5";
const PRE_PINNED_CRATES: &[&str] = &["cubecl-matmul", "cubecl-reduce"];

/// Family prefix used to identify "any cubecl-* crate" for the
/// transitive carve-out reachability check.
const CUBECL_FAMILY_PREFIX: &str = "cubecl";

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
    audit(&metadata, &root)
}

/// Reads `[workspace.dependencies]` from the workspace root Cargo.toml
/// and returns the version strings for cubecl-matmul and cubecl-reduce.
/// Returns None if either pin is missing — the carve-out is gated on
/// BOTH being present.
///
/// **Supported shape (single-line inline-table form ONLY):**
/// ```toml
/// cubecl-matmul = { version = "=0.9.0-pre.5" }
/// cubecl-reduce = "=0.9.0-pre.5"
/// ```
///
/// Multi-line table forms (`[workspace.dependencies.cubecl-matmul]` with
/// a separate `version = "=…"` line) and array-of-table forms are NOT
/// recognized by this parser and will return None — which causes the
/// carve-out to disengage and the lint to fail LOUDLY rather than
/// silently allowing relaxation. This is intentional: a Cargo.toml
/// refactor that changes the shape MUST be accompanied by an update
/// to this parser. The `parser_returns_none_on_multiline_table_form`
/// unit test enforces this invariant.
fn workspace_pre_pinned_versions(root: &Path) -> Result<Option<(String, String)>> {
    let cargo_toml =
        std::fs::read_to_string(root.join("Cargo.toml")).context("read workspace Cargo.toml")?;
    let mut matmul: Option<String> = None;
    let mut reduce: Option<String> = None;
    for line in cargo_toml.lines() {
        let trimmed = line.trim_start();
        // Match lines like:  cubecl-matmul   = { version = "=0.9.0-pre.5" }
        // and:                cubecl-matmul   = "=0.9.0-pre.5"
        for (crate_name, slot) in [
            ("cubecl-matmul", &mut matmul),
            ("cubecl-reduce", &mut reduce),
        ] {
            if let Some(after) = trimmed.strip_prefix(crate_name) {
                // Make sure the next char is whitespace or '=' — i.e. we
                // don't match `cubecl-matmul-extra` accidentally.
                let next = after.chars().next();
                if !matches!(next, Some(c) if c.is_whitespace() || c == '=') {
                    continue;
                }
                let rest = after.trim_start_matches(|c: char| c.is_whitespace() || c == '=');
                // Find the first `version` field (inline table) or a bare
                // version string.
                let v = if let Some(start) = rest.find("version") {
                    let rest = &rest[start..];
                    rest.split('"').nth(1).map(str::to_string)
                } else if let Some(start) = rest.find('"') {
                    rest[start + 1..].split('"').next().map(str::to_string)
                } else {
                    None
                };
                if let Some(v) = v {
                    // Strip a leading `=` so we can compare against
                    // PRE_REQUIRED_VERSION raw.
                    let v = v.trim_start_matches('=').to_string();
                    *slot = Some(v);
                }
            }
        }
    }
    match (matmul, reduce) {
        (Some(m), Some(r)) => Ok(Some((m, r))),
        _ => Ok(None),
    }
}

/// Build a reverse-dep map: for every package id, the set of package ids
/// that depend on it.
fn build_reverse_deps(metadata: &Value) -> BTreeMap<String, BTreeSet<String>> {
    let mut rdeps: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let empty = Vec::new();
    let nodes = metadata
        .get("resolve")
        .and_then(|r| r.get("nodes"))
        .and_then(|n| n.as_array())
        .unwrap_or(&empty);
    for node in nodes {
        let from = node["id"].as_str().unwrap_or("").to_string();
        let deps = node
            .get("dependencies")
            .and_then(|d| d.as_array())
            .unwrap_or(&empty);
        for dep_id in deps {
            if let Some(to) = dep_id.as_str() {
                rdeps
                    .entry(to.to_string())
                    .or_default()
                    .insert(from.clone());
            }
        }
    }
    rdeps
}

/// Returns Ok(()) if every reverse-dep path from `pkg_id` (the
/// 0.9.0-pre.5 cubecl-* node we're auditing) eventually terminates
/// at one of the carve-out root names (cubecl-matmul, cubecl-reduce).
/// Returns Err(leaked_sources) if any reverse-dep path leaves the
/// cubecl-* family without first being absorbed by a carve-out root,
/// OR terminates at a non-carve-out root.
///
/// Walks the rdeps map breadth-first. Any frontier node whose name is
/// in `carve_out_root_names` is "absorbed" (do not expand). Any frontier
/// node whose name is NOT in `carve_out_root_names` AND has no further
/// reverse-deps is a "leaked" root → violation. Any frontier node that
/// is itself a CUBECL_FAMILY_PREFIX-prefixed crate continues the walk
/// (the carve-out covers the matmul/reduce sub-DAG of cubecl internals).
fn reachable_only_from_carve_out_roots(
    pkg_id: &str,
    rdeps: &BTreeMap<String, BTreeSet<String>>,
    id_to_name: &BTreeMap<String, String>,
    carve_out_root_names: &[&str],
) -> std::result::Result<(), Vec<String>> {
    let mut violations: Vec<String> = Vec::new();
    let mut visited: BTreeSet<String> = BTreeSet::new();
    let mut frontier: Vec<String> = vec![pkg_id.to_string()];
    while let Some(cur) = frontier.pop() {
        if !visited.insert(cur.clone()) {
            continue;
        }
        let parents: Vec<String> = rdeps
            .get(&cur)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect();
        if parents.is_empty() {
            // Reached a root with no further reverse-deps. Is it a
            // carve-out root, or some other crate (the latter would
            // be a real violation — a method crate at the top of the
            // graph has no parents but IS a leaked source)?
            let name = id_to_name.get(&cur).cloned().unwrap_or_default();
            if !carve_out_root_names.contains(&name.as_str()) {
                violations.push(name);
            }
            continue;
        }
        for parent_id in parents {
            let parent_name = id_to_name.get(&parent_id).cloned().unwrap_or_default();
            if carve_out_root_names.contains(&parent_name.as_str()) {
                // Absorbed; do not expand further.
                continue;
            }
            if parent_name.starts_with(CUBECL_FAMILY_PREFIX) {
                // Internal cubecl-* crate; continue the walk.
                frontier.push(parent_id);
            } else {
                // Non-cubecl, non-carve-out parent — this is a leaked source.
                violations.push(parent_name);
            }
        }
    }
    if violations.is_empty() {
        Ok(())
    } else {
        Err(violations)
    }
}

/// Audit a parsed `cargo metadata` blob against the cubecl 0.10.0
/// lockstep + matmul/reduce 0.9.0-pre.5 carve-out invariants.
/// Separated from `main` so the regression-guard unit tests can
/// call it with synthetic fixtures.
fn audit(metadata: &Value, root: &Path) -> Result<ExitCode> {
    let empty = Vec::new();
    let packages = metadata["packages"].as_array().unwrap_or(&empty);
    let id_to_name: BTreeMap<String, String> = packages
        .iter()
        .filter_map(|p| {
            let id = p["id"].as_str()?.to_string();
            let name = p["name"].as_str()?.to_string();
            Some((id, name))
        })
        .collect();
    let rdeps = build_reverse_deps(metadata);

    // Read the live workspace pins to determine if the carve-out is active.
    let carve_out_active = matches!(
        workspace_pre_pinned_versions(root)?,
        Some((m, r)) if m == PRE_REQUIRED_VERSION && r == PRE_REQUIRED_VERSION
    );

    let mut violations: Vec<String> = Vec::new();
    let mut count_pinned: usize = 0;
    let mut count_pre: usize = 0;
    let mut count_pre_transitive: usize = 0;

    // Track whether we saw a 0.10.0 top-level cubecl-runtime — required.
    let mut saw_runtime_010 = false;

    for pkg in packages {
        let name = pkg["name"].as_str().unwrap_or("");
        let version = pkg["version"].as_str().unwrap_or("");
        let id = pkg["id"].as_str().unwrap_or("");

        if PINNED_CRATES.contains(&name) {
            if name == "cubecl-runtime" && version == REQUIRED_VERSION {
                saw_runtime_010 = true;
            }
            if version == REQUIRED_VERSION {
                count_pinned += 1;
                continue;
            }
            // Wrong version. Check the carve-out: is this a 0.9.0-pre.5
            // node reachable only from {cubecl-matmul, cubecl-reduce}?
            if version == PRE_REQUIRED_VERSION
                && carve_out_active
                && name.starts_with(CUBECL_FAMILY_PREFIX)
            {
                match reachable_only_from_carve_out_roots(
                    id,
                    &rdeps,
                    &id_to_name,
                    PRE_PINNED_CRATES,
                ) {
                    Ok(()) => {
                        count_pre_transitive += 1;
                        continue;
                    }
                    Err(leaked_sources) => {
                        violations.push(format!(
                            "{name} {version}: reachable from non-allowed source(s) {leaked_sources:?} (transitive carve-out limited to cubecl-matmul/cubecl-reduce)"
                        ));
                        continue;
                    }
                }
            }
            violations.push(format!(
                "{name}: version {version} (expected {REQUIRED_VERSION})"
            ));
            continue;
        }
        if PRE_PINNED_CRATES.contains(&name) {
            if version == PRE_REQUIRED_VERSION {
                count_pre += 1;
            } else {
                violations.push(format!(
                    "{name}: version {version} (expected {PRE_REQUIRED_VERSION})"
                ));
            }
            continue;
        }
        // `-sys` crates (e.g. cubecl-hip-sys) are FFI bindings versioned
        // against the native library they wrap (HIP SDK 7.1.x), NOT against
        // cubecl, so they are exempt from the 0.10.0 family lockstep.
        if name.starts_with(CUBECL_FAMILY_PREFIX) && name.ends_with("-sys") {
            continue;
        }
        // Other cubecl-* crates (cubecl-common, cubecl-core, cubecl-ir,
        // cubecl-macros, cubecl-macros-internal, cubecl-std, …) are
        // not in PINNED_CRATES because they're internal to the cubecl
        // family and not directly pinned by pyscf-rs. They MUST be at
        // 0.10.0 OR be carve-out-allowed transitives.
        if name.starts_with(CUBECL_FAMILY_PREFIX) && version != REQUIRED_VERSION {
            if version == PRE_REQUIRED_VERSION && carve_out_active {
                match reachable_only_from_carve_out_roots(
                    id,
                    &rdeps,
                    &id_to_name,
                    PRE_PINNED_CRATES,
                ) {
                    Ok(()) => {
                        count_pre_transitive += 1;
                        continue;
                    }
                    Err(leaked_sources) => {
                        violations.push(format!(
                            "{name} {version}: reachable from non-allowed source(s) {leaked_sources:?} (transitive carve-out limited to cubecl-matmul/cubecl-reduce)"
                        ));
                        continue;
                    }
                }
            }
            violations.push(format!(
                "{name}: version {version} (cubecl-* family member outside transitive carve-out)"
            ));
        }
    }

    // The 0.10.0 cubecl-runtime top-level pin MUST be present — losing
    // it would mean every consumer fell back to the carve-out version.
    // Only enforce when at least one PINNED_CRATES member is in the
    // metadata (otherwise this would falsely fail on synthetic fixtures
    // without cubecl-runtime entirely — but the dedicated test feeds a
    // matmul-only graph, where missing top-level 0.10.0 IS the bug.)
    let has_any_pinned_in_graph = packages.iter().any(|p| {
        let n = p["name"].as_str().unwrap_or("");
        PINNED_CRATES.contains(&n)
    });
    if has_any_pinned_in_graph && !saw_runtime_010 {
        violations.push(
            "cubecl-runtime: top-level pin 0.10.0 missing (only 0.9.0-pre.5 transitives present)"
                .to_string(),
        );
    }

    if violations.is_empty() {
        eprintln!(
            "check-cubecl-pin: PASS — {count_pinned} crate(s) at {REQUIRED_VERSION}, \
             {count_pre} crate(s) at {PRE_REQUIRED_VERSION}, \
             {count_pre_transitive} crate(s) at {PRE_REQUIRED_VERSION} transitively from cubecl-matmul/reduce (FOUND-04)"
        );
        Ok(ExitCode::SUCCESS)
    } else {
        eprintln!("check-cubecl-pin: FAIL — cubecl version drift detected (FOUND-04 / Pitfall 1):");
        for v in &violations {
            eprintln!("  - {v}");
        }
        eprintln!("\nFix: align workspace [workspace.dependencies] cubecl-* pins, or update");
        eprintln!("the sibling-crate revs in [patch.crates-io] to ones that use the matching pin.");
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
            bail!(
                "no [workspace] root found from {:?}",
                std::env::current_dir()?
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Build a synthetic minimal cargo metadata blob with the given
    /// (name, version, id, dependencies[]) tuples. The id format must
    /// match what `audit()` expects to look up in id_to_name.
    fn synth_metadata(packages: &[(&str, &str, &str, &[&str])]) -> Value {
        let pkg_objs: Vec<Value> = packages
            .iter()
            .map(|(n, v, id, _)| {
                json!({
                    "name": n,
                    "version": v,
                    "id": id,
                })
            })
            .collect();
        let nodes: Vec<Value> = packages
            .iter()
            .map(|(_, _, id, deps)| {
                json!({
                    "id": id,
                    "dependencies": deps.iter().map(|d| json!(d)).collect::<Vec<_>>(),
                })
            })
            .collect();
        json!({
            "packages": pkg_objs,
            "resolve": { "nodes": nodes },
        })
    }

    /// A throwaway temp directory carrying a stub Cargo.toml with the
    /// given matmul/reduce pin strings. Used so workspace_pre_pinned_versions
    /// returns the expected (m, r) tuple inside the test.
    fn temp_workspace(matmul_pin: &str, reduce_pin: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        let cargo = format!(
            "[workspace]\nmembers=[]\n[workspace.dependencies]\ncubecl-matmul = {{ version = \"={matmul_pin}\" }}\ncubecl-reduce = {{ version = \"={reduce_pin}\" }}\n"
        );
        std::fs::write(dir.path().join("Cargo.toml"), cargo).expect("write");
        dir
    }

    #[test]
    fn passes_when_all_pinned_at_010_with_transitive_009() {
        // Top-level: cubecl 0.10.0, cubecl-{cpu,cuda,hip,wgpu,runtime} 0.10.0
        // Transitive carve-out: cubecl-runtime 0.9.0-pre.5 reachable only from cubecl-matmul/reduce
        let m = synth_metadata(&[
            (
                "cubecl",
                "0.10.0",
                "cubecl 0.10.0",
                &["cubecl-runtime 0.10.0"],
            ),
            ("cubecl-runtime", "0.10.0", "cubecl-runtime 0.10.0", &[]),
            (
                "cubecl-cpu",
                "0.10.0",
                "cubecl-cpu 0.10.0",
                &["cubecl-runtime 0.10.0"],
            ),
            (
                "cubecl-cuda",
                "0.10.0",
                "cubecl-cuda 0.10.0",
                &["cubecl-runtime 0.10.0"],
            ),
            (
                "cubecl-hip",
                "0.10.0",
                "cubecl-hip 0.10.0",
                &["cubecl-runtime 0.10.0"],
            ),
            (
                "cubecl-wgpu",
                "0.10.0",
                "cubecl-wgpu 0.10.0",
                &["cubecl-runtime 0.10.0"],
            ),
            (
                "cubecl-matmul",
                "0.9.0-pre.5",
                "cubecl-matmul 0.9.0-pre.5",
                &["cubecl-runtime 0.9.0-pre.5"],
            ),
            (
                "cubecl-reduce",
                "0.9.0-pre.5",
                "cubecl-reduce 0.9.0-pre.5",
                &["cubecl-runtime 0.9.0-pre.5"],
            ),
            (
                "cubecl-runtime",
                "0.9.0-pre.5",
                "cubecl-runtime 0.9.0-pre.5",
                &[],
            ),
        ]);
        let dir = temp_workspace("0.9.0-pre.5", "0.9.0-pre.5");
        assert_eq!(audit(&m, dir.path()).unwrap(), ExitCode::SUCCESS);
    }

    #[test]
    fn fails_when_cubecl_runtime_010_missing() {
        // Only matmul/reduce + 0.9.0-pre.5 cubecl-runtime present. No 0.10.0
        // top-level cubecl-* pins exist; the lint must FAIL because the
        // canonical 0.10.0 lockstep is broken.
        // Note: this fixture deliberately ALSO contains a 0.10.0 placeholder
        // for `cubecl` so the has-any-pinned-in-graph guard fires. The
        // missing 0.10.0 is specifically cubecl-runtime.
        let m = synth_metadata(&[
            ("cubecl", "0.10.0", "cubecl 0.10.0", &[]),
            (
                "cubecl-matmul",
                "0.9.0-pre.5",
                "cubecl-matmul 0.9.0-pre.5",
                &["cubecl-runtime 0.9.0-pre.5"],
            ),
            (
                "cubecl-reduce",
                "0.9.0-pre.5",
                "cubecl-reduce 0.9.0-pre.5",
                &["cubecl-runtime 0.9.0-pre.5"],
            ),
            (
                "cubecl-runtime",
                "0.9.0-pre.5",
                "cubecl-runtime 0.9.0-pre.5",
                &[],
            ),
        ]);
        let dir = temp_workspace("0.9.0-pre.5", "0.9.0-pre.5");
        assert_eq!(audit(&m, dir.path()).unwrap(), ExitCode::from(2));
    }

    #[test]
    fn fails_when_pyscf_kernels_pulls_old_cubecl() {
        // The 0.9.0-pre.5 cubecl-runtime is reachable from BOTH cubecl-matmul (allowed)
        // AND pyscf-kernels (NOT allowed) — the latter is the leaked source.
        let m = synth_metadata(&[
            ("cubecl", "0.10.0", "cubecl 0.10.0", &[]),
            ("cubecl-runtime", "0.10.0", "cubecl-runtime 0.10.0", &[]),
            (
                "cubecl-cpu",
                "0.10.0",
                "cubecl-cpu 0.10.0",
                &["cubecl-runtime 0.10.0"],
            ),
            (
                "cubecl-cuda",
                "0.10.0",
                "cubecl-cuda 0.10.0",
                &["cubecl-runtime 0.10.0"],
            ),
            (
                "cubecl-hip",
                "0.10.0",
                "cubecl-hip 0.10.0",
                &["cubecl-runtime 0.10.0"],
            ),
            (
                "cubecl-wgpu",
                "0.10.0",
                "cubecl-wgpu 0.10.0",
                &["cubecl-runtime 0.10.0"],
            ),
            (
                "cubecl-matmul",
                "0.9.0-pre.5",
                "cubecl-matmul 0.9.0-pre.5",
                &["cubecl-runtime 0.9.0-pre.5"],
            ),
            (
                "cubecl-reduce",
                "0.9.0-pre.5",
                "cubecl-reduce 0.9.0-pre.5",
                &["cubecl-runtime 0.9.0-pre.5"],
            ),
            (
                "cubecl-runtime",
                "0.9.0-pre.5",
                "cubecl-runtime 0.9.0-pre.5",
                &[],
            ),
            (
                "pyscf-kernels",
                "0.1.0",
                "pyscf-kernels 0.1.0",
                &["cubecl-runtime 0.9.0-pre.5"],
            ),
        ]);
        let dir = temp_workspace("0.9.0-pre.5", "0.9.0-pre.5");
        let exit = audit(&m, dir.path()).unwrap();
        assert_eq!(exit, ExitCode::from(2));
    }

    #[test]
    fn fails_when_matmul_pin_moved_but_runtime_skew_persists() {
        // Future state: matmul/reduce now at 0.10.0 (workspace pin matches),
        // but a stray 0.9.0-pre.5 cubecl-runtime is still in the resolve graph
        // (e.g., from a different crate). The carve-out must NOT engage.
        let m = synth_metadata(&[
            ("cubecl", "0.10.0", "cubecl 0.10.0", &[]),
            ("cubecl-runtime", "0.10.0", "cubecl-runtime 0.10.0", &[]),
            (
                "cubecl-cpu",
                "0.10.0",
                "cubecl-cpu 0.10.0",
                &["cubecl-runtime 0.10.0"],
            ),
            (
                "cubecl-cuda",
                "0.10.0",
                "cubecl-cuda 0.10.0",
                &["cubecl-runtime 0.10.0"],
            ),
            (
                "cubecl-hip",
                "0.10.0",
                "cubecl-hip 0.10.0",
                &["cubecl-runtime 0.10.0"],
            ),
            (
                "cubecl-wgpu",
                "0.10.0",
                "cubecl-wgpu 0.10.0",
                &["cubecl-runtime 0.10.0"],
            ),
            ("cubecl-matmul", "0.10.0", "cubecl-matmul 0.10.0", &[]),
            ("cubecl-reduce", "0.10.0", "cubecl-reduce 0.10.0", &[]),
            (
                "cubecl-runtime",
                "0.9.0-pre.5",
                "cubecl-runtime 0.9.0-pre.5",
                &[],
            ),
        ]);
        let dir = temp_workspace("0.10.0", "0.10.0");
        assert_eq!(audit(&m, dir.path()).unwrap(), ExitCode::from(2));
    }

    #[test]
    fn parser_returns_none_on_multiline_table_form() {
        // A legitimate Cargo.toml refactor moves cubecl-matmul to the
        // multi-line table form. The parser does NOT recognize this
        // shape and returns None — which causes the carve-out to
        // disengage and the lint to FAIL LOUD on the next 0.9.0-pre.5
        // transitive, rather than silently allowing the relaxation to
        // persist against an unrecognized pin.
        let dir = tempfile::tempdir().expect("tempdir");
        let cargo = "[workspace]\nmembers=[]\n\
                     [workspace.dependencies.cubecl-matmul]\n\
                     version = \"=0.9.0-pre.5\"\n\
                     [workspace.dependencies.cubecl-reduce]\n\
                     version = \"=0.9.0-pre.5\"\n";
        std::fs::write(dir.path().join("Cargo.toml"), cargo).expect("write");
        // The string-grep parser is keyed off `crate-name <whitespace> =`
        // at the start of a line; the table-form has no such line, so
        // both slots remain None and we return Ok(None).
        let result = workspace_pre_pinned_versions(dir.path()).expect("read ok");
        assert!(
            result.is_none(),
            "multi-line table form must return None (got {result:?}); \
             see parser doc comment for the invariant"
        );
    }
}
