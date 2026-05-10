//! Integration test for BackendKind env-var parsing — the CPU-only
//! subset. Plan 04's `pyscf-algebra/tests/select_backend.rs` covers the
//! GPU-feature-gated cases that need an AlgebraClient.
//!
//! These tests use `--test-threads=1` because they mutate process env
//! vars; nextest's per-test isolation would also work but cargo test
//! defaults are sufficient here.

use pyscf_runtime::{BackendKind, DType};
use std::env;

/// Per RESEARCH Pitfall 6: install a tracing subscriber so info!/warn!
/// lines emitted by the parser would be visible if assertions failed.
/// `try_init` is idempotent — multiple calls are safe.
fn setup_tracing() {
    let _ = tracing_subscriber::fmt::try_init();
}

/// FOUND-03: `BackendKind::default()` returns `Cpu` always.
#[test]
fn test_default_is_cpu() {
    setup_tracing();
    assert_eq!(BackendKind::default(), BackendKind::Cpu);
}

/// FOUND-03 + ALG-04 row "unset": no PYSCF_BACKEND env => parser
/// returns Some(Cpu) when called with the empty/default fallback string.
/// (Plan 04's select_backend() fully wires "unset → Cpu"; here we
/// verify the building block.)
#[test]
fn test_from_env_str_cpu() {
    setup_tracing();
    assert_eq!(BackendKind::from_env_str("cpu"), Some(BackendKind::Cpu));
    assert_eq!(BackendKind::from_env_str("CPU"), Some(BackendKind::Cpu));
    assert_eq!(BackendKind::from_env_str("CpU"), Some(BackendKind::Cpu));
}

/// ALG-04: unrecognised value returns None. Plan 04's select_backend()
/// maps None → tracing::warn! + fallback to Cpu (full ALG-04 contract).
#[test]
fn test_from_env_str_bogus() {
    setup_tracing();
    assert_eq!(BackendKind::from_env_str("bogus"), None);
    assert_eq!(BackendKind::from_env_str(""), None);
    assert_eq!(BackendKind::from_env_str("xxx"), None);
    // "auto" returns None from the parser — caller must treat it as
    // a separate token (D-07 priority chain).
    assert_eq!(BackendKind::from_env_str("auto"), None);
}

/// D-07: `is_auto_token` correctly identifies the auto sentinel.
#[test]
fn test_is_auto_token() {
    setup_tracing();
    assert!(BackendKind::is_auto_token("auto"));
    assert!(BackendKind::is_auto_token("AUTO"));
    assert!(BackendKind::is_auto_token("Auto"));
    assert!(!BackendKind::is_auto_token("cpu"));
    assert!(!BackendKind::is_auto_token(""));
}

/// D-08: DType default is F64.
#[test]
fn test_dtype_default_is_f64() {
    setup_tracing();
    assert_eq!(DType::default(), DType::F64);
}

/// D-08: PYSCF_DTYPE parsing is case-insensitive; default F64.
/// Test isolates env mutation by setting/restoring the var explicitly.
#[test]
fn test_dtype_from_env_unset_defaults_f64() {
    setup_tracing();
    // Snapshot to restore.
    let saved = env::var("PYSCF_DTYPE").ok();
    // SAFETY: this test runs with --test-threads=1; no other thread
    // observes the env in this process.
    unsafe {
        env::remove_var("PYSCF_DTYPE");
    }
    assert_eq!(DType::from_env(), DType::F64);
    if let Some(v) = saved {
        unsafe {
            env::set_var("PYSCF_DTYPE", v);
        }
    }
}

/// FOUND-03: BackendKind::Cpu has a stable name.
#[test]
fn test_name_cpu() {
    setup_tracing();
    assert_eq!(BackendKind::Cpu.name(), "cpu");
}
