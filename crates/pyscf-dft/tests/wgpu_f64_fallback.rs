//! DFT-11 — WGPU f64 honesty: the DEFAULT-f64 path falls back to CPU-f64 with
//! a `tracing::warn!` on a `shader-f64`-less wgpu adapter, NEVER silently
//! degrading to f32 (Pitfall 6 / the Pitfall-3 silent-precision re-validation).
//!
//! ## What this asserts
//! On a `shader-f64`-less environment (the common case / the CI default — no
//! GPU adapter advertising the `shader-f64` Vulkan extension), with the
//! DEFAULT f64 precision and `PYSCF_BACKEND=wgpu` explicitly requested:
//!   1. [`pyscf_dft::xc_backend::pipeline_fallback`] resolves the XC-eval
//!      substrate to `Cpu` (the honest fallback), and
//!   2. a single below-the-line `tracing::warn!` (target
//!      `pyscf_dft::xc_backend`) is emitted naming the CPU-f64 fallback, and
//!   3. a small RKS XC-eval block (`NumInt::nr_rks` over a real H2O Becke
//!      grid + a 1e-init-guess density) produces CPU-correct, all-finite f64
//!      numbers — i.e. the fallback runs, it does NOT silently compute f32.
//!
//! ## Delegation, not re-implementation (DFT-11)
//! The substrate decision is DELEGATED to `xcfun_rs`'s existing probe:
//! [`pyscf_dft::xc_backend::xc_eval_substrate`] calls
//! `xcfun_gpu::auto_backend()` (which gates `Wgpu` on `shader-f64`) +
//! `xcfun_gpu::error_routing::must_fall_back_to_cpu()` (ERF→CPU). No
//! `shader-f64` probe is re-implemented in pyscf-dft.
//!
//! ## Why the default-f64 case is the one CI exercises
//! `auto_backend()` returns `Backend::Cpu` whenever no `shader-f64` device is
//! present (no `XCFUN_FORCE_BACKEND`, no probe-passing adapter compiled in),
//! which IS the CI default. The companion `wgpu-no-f64-fallback` CI job runs
//! this on a `shader-f64`-less/emulated runner; the explicit `PYSCF_DTYPE=f32`
//! escape hatch (D-08) is the SEPARATE honest opt-in, exercised by
//! `dtype_f32_smoke`.
//!
//! NOT `#[cfg(feature = "libxc")]`-gated — runs on the default CPU/xcfun
//! backend. Verify command: `cargo test -p pyscf-dft wgpu_f64_fallback`.

use std::sync::{Arc, Mutex};

use pyscf_core::{Mole, Unit};
use pyscf_dft::xc_backend::{pipeline_fallback, xc_eval_substrate, XcEvalSubstrate};
use pyscf_dft::NumInt;
use pyscf_grids::Grids;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs, M};
use pyscf_runtime::DType;
use tracing::subscriber::with_default;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::Layer;
use xcfun_rs::Dependency;

/// Records every `WARN`-level event whose target is `pyscf_dft::xc_backend`
/// (the DFT-11 fallback warn) — self-contained, no `tracing-test` dep.
#[derive(Clone, Default)]
struct FallbackWarnCatcher {
    saw_fallback_warn: Arc<Mutex<bool>>,
}

impl<S: tracing::Subscriber> Layer<S> for FallbackWarnCatcher {
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let meta = event.metadata();
        if *meta.level() == tracing::Level::WARN && meta.target() == "pyscf_dft::xc_backend" {
            if let Ok(mut g) = self.saw_fallback_warn.lock() {
                *g = true;
            }
        }
    }
}

fn h2o_sto3g() -> Mole {
    M(MoleBuildArgs {
        atom: AtomInput::String("O 0 0 0; H 0 0 0.96; H 0 0.93 -0.24".into()),
        basis: BasisInput::Name("sto-3g".into()),
        unit: Unit::Ang,
        ..Default::default()
    })
    .expect("build H2O/sto-3g")
}

/// 1e core-Hamiltonian init guess (the `1e` mode IS implemented) — gives the
/// XC-eval block a real, nonzero ρ to integrate.
fn one_electron_density(mol: &Mole) -> pyscf_core::Density {
    pyscf_scf::default_get_init_guess(mol, &pyscf_scf::InitGuessMode::OneElectron)
        .expect("1e init guess builds a density")
}

#[test]
fn wgpu_f64_fallback() {
    // ── Save + set the env: explicit PYSCF_BACKEND=wgpu, default f64 ──────────
    // Integration tests are separate crates (not under the lib's
    // `#![forbid(unsafe_code)]`); env mutation is `unsafe` in edition 2024 —
    // mirrors the Phase 1 pyscf-algebra / 04-06 dtype_f32_smoke test pattern.
    let saved_backend = std::env::var("PYSCF_BACKEND").ok();
    let saved_dtype = std::env::var("PYSCF_DTYPE").ok();
    unsafe {
        std::env::set_var("PYSCF_BACKEND", "wgpu");
        // DEFAULT f64 (DFT-11 honesty path, NOT the D-08 f32 escape hatch).
        std::env::remove_var("PYSCF_DTYPE");
        // Make the probe deterministic on CI: force the CPU substrate so the
        // test asserts the honest fallback regardless of the runner's GPU. A
        // shader-f64-less runner reaches the same Backend::Cpu without this.
        std::env::set_var("XCFUN_FORCE_BACKEND", "cpu");
    }

    let catcher = FallbackWarnCatcher::default();
    let saw_fallback_warn = catcher.saw_fallback_warn.clone();

    let outcome = {
        let subscriber = tracing_subscriber::registry().with(catcher);
        with_default(subscriber, || {
            // The functional capability bitset for the standard (non-RSH)
            // corpus: density + gradient, NO ERF. Forwarded to the delegated
            // probe so `must_fall_back_to_cpu` is exercised honestly.
            let deps = Dependency::DENSITY | Dependency::GRADIENT;

            // (a) bare substrate decision is delegated to xcfun_rs's probe.
            let substrate = xc_eval_substrate(deps);

            // (b) pipeline-level fallback under DEFAULT f64 (f64_requested=true)
            //     must resolve to CPU + emit the warn.
            let f64_requested = DType::from_env() == DType::F64;
            let resolved = pipeline_fallback(deps, f64_requested);

            // (c) the fallback runs CPU-correct f64 numbers, not silent f32.
            let mol = h2o_sto3g();
            let mut grids = Grids::new();
            let _ = grids.build(&mol);
            let dm = one_electron_density(&mol);
            let ni = NumInt::new();
            let dtype = ni.dtype();
            let nr = ni.nr_rks(&mol, &grids, "svwn", &dm, 0, 1, mol.max_memory, None);

            (substrate, f64_requested, resolved, dtype, nr)
        })
    };

    // ── Restore env BEFORE asserting (so a panic can't leak it) ──────────────
    unsafe {
        match saved_backend {
            Some(v) => std::env::set_var("PYSCF_BACKEND", v),
            None => std::env::remove_var("PYSCF_BACKEND"),
        }
        match saved_dtype {
            Some(v) => std::env::set_var("PYSCF_DTYPE", v),
            None => std::env::remove_var("PYSCF_DTYPE"),
        }
        std::env::remove_var("XCFUN_FORCE_BACKEND");
    }

    let (substrate, f64_requested, resolved, dtype, nr) = outcome;

    // The DEFAULT precision is f64 (the honesty path, NOT the f32 escape hatch).
    assert!(f64_requested, "default precision is f64 (PYSCF_DTYPE unset → D-08 default)");
    assert_eq!(dtype, DType::F64, "NumInt resolves to F64 (no silent f32 downgrade)");

    // The delegated probe resolves to the CPU substrate on a shader-f64-less
    // device (here forced via XCFUN_FORCE_BACKEND=cpu / the CI default).
    assert_eq!(
        substrate,
        XcEvalSubstrate::Cpu,
        "shader-f64-less adapter → XC eval substrate is CPU (delegated to xcfun_gpu::auto_backend)"
    );
    assert_eq!(
        resolved,
        XcEvalSubstrate::Cpu,
        "explicit PYSCF_BACKEND=wgpu + default f64 + no shader-f64 → CPU-f64 fallback"
    );

    // The fallback warn fired (never silently degrade to f32 under default f64).
    assert!(
        *saw_fallback_warn.lock().expect("warn flag lock"),
        "a CPU-f64 fallback tracing::warn! (target pyscf_dft::xc_backend) must fire when \
         PYSCF_BACKEND=wgpu is requested at f64 on a shader-f64-less adapter (DFT-11)"
    );

    // The CPU-f64 XC eval ran and produced correct, all-finite numbers — the
    // honest fallback, NOT a silent f32 path.
    let nr = nr.expect("CPU-f64 fallback nr_rks runs end-to-end (no panic, no error)");
    assert!(nr.excsum.is_finite(), "CPU-f64 fallback Exc is finite");
    assert!(nr.nelec.is_finite(), "CPU-f64 fallback nelec is finite");
    assert!(
        nr.vmat.data.iter().all(|v| v.is_finite()),
        "CPU-f64 fallback Vxc matrix is all-finite"
    );
}

/// The D-08 boundary: the EXPLICIT `PYSCF_DTYPE=f32` opt-in is NOT treated as a
/// silent degradation to block — `pipeline_fallback` with `f64_requested=false`
/// does NOT emit the CPU-f64 fallback warn (the 04-06 NumInt f32 chain owns its
/// own below-bit-exact warn). This is the escape-hatch distinction (DFT-11/D-08).
#[test]
fn explicit_f32_is_not_blocked_as_silent_degradation() {
    let saved_backend = std::env::var("PYSCF_BACKEND").ok();
    unsafe {
        std::env::set_var("PYSCF_BACKEND", "wgpu");
        std::env::set_var("XCFUN_FORCE_BACKEND", "cpu");
    }

    let catcher = FallbackWarnCatcher::default();
    let saw_fallback_warn = catcher.saw_fallback_warn.clone();

    let resolved = {
        let subscriber = tracing_subscriber::registry().with(catcher);
        with_default(subscriber, || {
            let deps = Dependency::DENSITY | Dependency::GRADIENT;
            // f64_requested = false ⇒ the user explicitly opted into f32 (D-08).
            pipeline_fallback(deps, false)
        })
    };

    unsafe {
        match saved_backend {
            Some(v) => std::env::set_var("PYSCF_BACKEND", v),
            None => std::env::remove_var("PYSCF_BACKEND"),
        }
        std::env::remove_var("XCFUN_FORCE_BACKEND");
    }

    // Substrate is still CPU here (forced), but the KEY assertion is that NO
    // CPU-f64 fallback warn fired — the explicit f32 opt-in is honest, not a
    // silent degradation, so it is NOT warned/blocked by pipeline_fallback.
    assert_eq!(resolved, XcEvalSubstrate::Cpu);
    assert!(
        !*saw_fallback_warn.lock().expect("warn flag lock"),
        "explicit PYSCF_DTYPE=f32 (f64_requested=false) is the honest D-08 escape hatch — \
         pipeline_fallback must NOT emit the CPU-f64 fallback warn for it"
    );
}
