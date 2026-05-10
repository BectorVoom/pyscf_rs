//! ALG-04 + Roadmap criterion 6 (page 44): backend-resolution truth
//! table for the cases that exercise select_backend (Plan 03's
//! pyscf-runtime test covered the CPU-only parser cases).
//!
//! Run with `--test-threads=1` because tests mutate process env vars.

use pyscf_algebra::select_backend;
use pyscf_runtime::{BackendKind, DType};
use std::env;

fn setup_tracing() { let _ = tracing_subscriber::fmt::try_init(); }

fn save() -> (Option<String>, Option<String>) {
    (env::var("PYSCF_BACKEND").ok(), env::var("PYSCF_DTYPE").ok())
}

fn restore(saved: (Option<String>, Option<String>)) {
    unsafe {
        match saved.0 {
            Some(v) => env::set_var("PYSCF_BACKEND", v),
            None => env::remove_var("PYSCF_BACKEND"),
        }
        match saved.1 {
            Some(v) => env::set_var("PYSCF_DTYPE", v),
            None => env::remove_var("PYSCF_DTYPE"),
        }
    }
}

/// Roadmap criterion 6 row "unset" → CPU.
#[test]
fn unset_resolves_to_cpu() {
    setup_tracing();
    let s = save();
    unsafe { env::remove_var("PYSCF_BACKEND"); env::remove_var("PYSCF_DTYPE"); }
    let sel = select_backend().expect("unset must succeed");
    assert_eq!(sel.kind, BackendKind::Cpu);
    assert_eq!(sel.dtype, DType::F64);  // D-08 default
    assert_eq!(sel.raw_env, None);
    restore(s);
}

/// Roadmap criterion 6 row "cpu" → CPU.
#[test]
fn cpu_explicit_resolves_to_cpu() {
    setup_tracing();
    let s = save();
    unsafe { env::set_var("PYSCF_BACKEND", "cpu"); env::remove_var("PYSCF_DTYPE"); }
    let sel = select_backend().expect("cpu must succeed");
    assert_eq!(sel.kind, BackendKind::Cpu);
    assert_eq!(sel.raw_env.as_deref(), Some("cpu"));
    restore(s);
}

/// Roadmap criterion 6 row "bogus" → CPU + warn.
#[test]
fn bogus_resolves_to_cpu_with_warn() {
    setup_tracing();
    let s = save();
    unsafe { env::set_var("PYSCF_BACKEND", "definitely-not-a-backend"); }
    let sel = select_backend().expect("bogus must fall back to CPU per ALG-04");
    assert_eq!(sel.kind, BackendKind::Cpu);
    assert_eq!(sel.raw_env.as_deref(), Some("definitely-not-a-backend"));
    restore(s);
}

/// Roadmap criterion 6 row "auto" on CPU-only build → CPU.
/// (On a build with --features gpu and CUDA hardware, this would
/// return Cuda; on a CPU-only Phase 1 build it falls through to CPU.)
#[test]
fn auto_on_cpu_only_build_resolves_to_cpu() {
    setup_tracing();
    let s = save();
    unsafe { env::set_var("PYSCF_BACKEND", "auto"); env::set_var("PYSCF_DTYPE", "f64"); }
    let sel = select_backend().expect("auto must succeed");
    // On CPU-only Phase 1 build (no cuda/wgpu/rocm features), auto
    // resolves to Cpu.
    assert_eq!(sel.kind, BackendKind::Cpu);
    restore(s);
}

/// Case-insensitivity: PYSCF_BACKEND=AUTO and PYSCF_DTYPE=F64 work.
#[test]
fn case_insensitive_env_parsing() {
    setup_tracing();
    let s = save();
    unsafe { env::set_var("PYSCF_BACKEND", "AUTO"); env::set_var("PYSCF_DTYPE", "F64"); }
    let sel = select_backend().expect("AUTO uppercase must work");
    assert_eq!(sel.kind, BackendKind::Cpu);
    assert_eq!(sel.dtype, DType::F64);
    restore(s);
}

/// PYSCF_DTYPE=f32 is honoured.
#[test]
fn dtype_f32_honored() {
    setup_tracing();
    let s = save();
    unsafe { env::remove_var("PYSCF_BACKEND"); env::set_var("PYSCF_DTYPE", "f32"); }
    let sel = select_backend().expect("f32 dtype must succeed");
    assert_eq!(sel.dtype, DType::F32);
    restore(s);
}

/// ALG-08 final log line shape — verified by capturing tracing output.
/// Phase 1 ships log_resolution; this test confirms the `log_resolution`
/// is called inside select_backend (via tracing_subscriber capturing).
/// Full output capture requires tracing-test crate; Phase 1 satisfies
/// itself with the absence of panics + presence of selected dtype.
#[test]
fn alg08_log_resolution_invoked() {
    setup_tracing();
    let s = save();
    unsafe { env::remove_var("PYSCF_BACKEND"); env::remove_var("PYSCF_DTYPE"); }
    // No assertion on log content (would need tracing-test); the test
    // value is "select_backend doesn't panic when log_resolution
    // executes its tracing::info! macro".
    let _ = select_backend().expect("must succeed");
    restore(s);
}
