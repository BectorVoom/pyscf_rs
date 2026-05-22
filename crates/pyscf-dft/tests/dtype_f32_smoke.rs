//! DFT-11 / D-08: `PYSCF_DTYPE=f32` DFT grid loop runs END-TO-END.
//!
//! This is a runs-to-completion SMOKE ONLY: it asserts that the D-08 f32
//! dispatch path runs end-to-end (AO eval → f32 ρ contraction → f64 XC
//! boundary → f32 Vxc back-contraction → finite Exc/Vxc/nelec) AND that the
//! below-bit-exact `tracing::warn!` fires at grid-loop entry. It performs
//! **NO oracle comparison** and carries **NO numeric-parity tolerance gate**
//! (D-08 contract — a loose chemical-accuracy f32 bound is a Deferred Idea,
//! not a v1 gate; the oracle/bit-exact rows stay f64-only).
//!
//! ### Why drive `NumInt::nr_rks` directly (not `RKS::kernel`)
//! The full RKS SCF cycle cannot converge in the dev sandbox: the `minao`
//! init guess and the arity-3/4 2-electron integrals (`int2e_sph` /
//! `int3c2e_sph`) are Phase-2/3 deferred gaps (`NotYetImplemented`). So a
//! "run `RKS::kernel()` to completion" smoke would fail on those gaps, NOT on
//! anything D-08. The honest D-08 smoke exercises the EXACT f32 code path the
//! requirement is about — the grid-loop matmul chain dispatched over the
//! active scalar — by driving `NumInt::nr_rks` directly over a real molecule's
//! Becke grid + a real (1e-init-guess) density. This is the same
//! "exercise the real seam, not the blocked outer driver" approach 04-04/04-05
//! used for the grid + XC oracles.
//!
//! Reference: D-08 precision switch (`DType::from_env` threaded through NumInt).
//! NOT `#[cfg(feature = "libxc")]`-gated — runs on the default CPU/xcfun backend.
//! Verify command: `cargo test -p pyscf-dft dtype_f32_smoke`.

use std::sync::{Arc, Mutex};

use pyscf_core::{Mole, Unit};
use pyscf_dft::NumInt;
use pyscf_grids::Grids;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs, M};
use pyscf_runtime::DType;
use tracing::subscriber::with_default;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::Layer;

/// A minimal in-process tracing layer that records whether ANY `WARN`-level
/// event was emitted (self-contained — no `tracing-test` dep).
#[derive(Clone, Default)]
struct WarnCatcher {
    saw_warn: Arc<Mutex<bool>>,
}

impl<S: tracing::Subscriber> Layer<S> for WarnCatcher {
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        if *event.metadata().level() == tracing::Level::WARN {
            if let Ok(mut g) = self.saw_warn.lock() {
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

/// Build a physically-reasonable density matrix from the 1e core-Hamiltonian
/// init guess (the `1e` mode IS implemented — init_guess.rs:19), so the f32
/// grid loop runs over a real, nonzero ρ.
fn one_electron_density(mol: &Mole) -> pyscf_core::Density {
    pyscf_scf::default_get_init_guess(mol, &pyscf_scf::InitGuessMode::OneElectron)
        .expect("1e init guess builds a density")
}

#[test]
fn dtype_f32_smoke() {
    // ── Save + set PYSCF_DTYPE=f32 (restore after) ─────────────────────────
    // Integration tests are separate crates (not under the lib's
    // `#![forbid(unsafe_code)]`); env mutation is `unsafe` in edition 2024 —
    // mirrors the Phase 1 pyscf-algebra select_backend test pattern.
    let saved = std::env::var("PYSCF_DTYPE").ok();
    unsafe {
        std::env::set_var("PYSCF_DTYPE", "f32");
    }

    let warn = WarnCatcher::default();
    let saw_warn = warn.saw_warn.clone();

    // Run the f32 grid loop under a capturing subscriber.
    let result = {
        let subscriber = tracing_subscriber::registry().with(warn);
        with_default(subscriber, || {
            let mol = h2o_sto3g();
            let mut grids = Grids::new();
            let _ = grids.build(&mol);
            let dm = one_electron_density(&mol);

            // NumInt reads PYSCF_DTYPE=f32 at construction (D-08).
            let ni = NumInt::new();
            assert_eq!(ni.dtype(), DType::F32, "PYSCF_DTYPE=f32 → NumInt::dtype()==F32");

            // The D-08 f32 dispatch path, end-to-end (SVWN = LDA).
            ni.nr_rks(&mol, &grids, "svwn", &dm, 0, 1, mol.max_memory, None)
        })
    };

    // ── Restore env BEFORE asserting (so a panic doesn't leak the env) ─────
    unsafe {
        match saved {
            Some(v) => std::env::set_var("PYSCF_DTYPE", v),
            None => std::env::remove_var("PYSCF_DTYPE"),
        }
    }

    // ── Assertions: runs-to-completion + finite + warned. NO oracle, NO tol ─
    let nr = result.expect("f32 nr_rks runs end-to-end (no panic, no error)");
    assert!(nr.excsum.is_finite(), "f32 Exc is finite (runs end-to-end)");
    assert!(nr.nelec.is_finite(), "f32 nelec is finite");
    assert!(
        nr.vmat.data.iter().all(|v| v.is_finite()),
        "f32 Vxc matrix is all-finite"
    );
    assert!(
        *saw_warn.lock().expect("warn flag lock"),
        "a below-bit-exact tracing::warn! must fire when PYSCF_DTYPE=f32 (D-08)"
    );
    // NOTE (D-08): there is intentionally NO comparison of `nr.excsum` to any
    // oracle value and NO numeric-parity tolerance — the f32 path is never
    // compared to upstream in v1.
}
