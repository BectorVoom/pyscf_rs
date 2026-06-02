//! quick-260530-p40 (FOUND-06, D-07): lock the numint **device AO-eval path**
//! explicitly and pin the device AO block itself.
//!
//! ## Why this test exists
//!
//! The DFT grid loop never names a device kernel directly. It evaluates AOs via
//! `pyscf_gto::eval_gto(mol, name, coords)` (the exact public fn
//! `numint::eval_gto_block` calls), which routes through
//! `pyscf_algebra::select_backend` → the cubecl `eval_gto` kernels (Phase-8
//! ljv/mlg/oms: value + l ≥ 1 + deriv1). With `PYSCF_BACKEND` unset the resolved
//! client is `CpuRuntime` — still the **device** `#[cube]` kernel, not a separate
//! host path. This file makes that path a regression-gated invariant on a real
//! `maxl = 2` (s/p/d) molecule, so the general `l ≥ 1` device kernel is exercised.
//!
//! The lock is about correctness of the **path** the numint AO loop uses, not
//! about which physical backend runs. If `select_backend`'s default ever moved
//! off Cpu, this test still passes: every device backend agrees with the
//! CpuRuntime kernel within the tolerances below (1e-9 / 1e-12).
//!
//! ### Assertion A (PRIMARY): eval_rho over a device AO block is correct
//!   ρ from `NumInt::eval_rho` (which contracts the device-produced AO block via
//!   its bit-exact `oracle_sum` pairwise tree) matches an INDEPENDENT hand-written
//!   triple sum `Σ_μν ao[g,μ]·D[μν]·ao[g,ν]` within 1e-9. Non-circular: the
//!   reference is a plain nested f64 loop in this test that shares NO code with
//!   `oracle_sum`. A small ordering Δ vs the pairwise tree is expected and bounded
//!   well under 1e-9 (correctness tolerance, NOT bit-identity).
//!
//! ### Assertion B: the device AO block itself is correct (cross-kernel lock)
//!   `GTOval_sph` (value-only kernel) and `GTOval_sph_deriv1` comp-0 (the value
//!   slice of the 4-component kernel) are TWO independently-implemented device
//!   kernels. They MUST produce identical value numerics; here they agree
//!   elementwise within 1e-12. If either kernel's monomial·radial·c2s math drifts,
//!   this fails — pinning that the AO block numint consumes is itself correct, not
//!   just that eval_rho contracts whatever it is fed.

use pyscf_core::{Density, Unit};
use pyscf_dft::{NumInt, XcType};
use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};

/// H2O / cc-pVDZ (maxl = 2 → s/p/d → exercises the general l ≥ 1 device kernel).
/// Same fixture as the df_dft_match structural layer.
fn h2o_ccpvdz() -> pyscf_core::Mole {
    M(MoleBuildArgs {
        atom: AtomInput::String("O 0 0 0; H 0 0 0.96; H 0 0.93 -0.24".into()),
        basis: BasisInput::Name("cc-pvdz".into()),
        unit: Unit::Ang,
        ..Default::default()
    })
    .expect("build H2O/cc-pVDZ")
}

/// A small deterministic grid near the three atoms (Ångström → bohr is handled
/// internally by the Mole geometry; these are AO eval coords in the same frame
/// `eval_gto` expects). Hand-listed so no `Grids::build()` is needed — the test
/// stays self-contained and cheap. Points are placed close to nuclei so AO
/// values are nonzero (a non-degenerate lock).
fn grid_coords() -> Vec<[f64; 3]> {
    vec![
        [0.05, 0.05, 0.05], // near O
        [0.0, 0.0, 0.96],   // on H1
        [0.0, 0.93, -0.24], // on H2
        [0.1, -0.1, 0.2],   // off-center
        [0.0, 0.0, 0.5],    // O–H1 midpoint-ish
        [0.0, 0.45, -0.1],  // O–H2 midpoint-ish
        [-0.2, 0.3, 0.4],   // generic point
        [0.3, 0.3, 0.3],    // generic point
    ]
}

/// A fixed, reproducible, symmetric density matrix — NO SCF run. `D[μν]` is a
/// deterministic seeded value, then symmetrized `D = (D + Dᵀ)/2`. This is purely
/// an algebra fixture: the lock is on the AO-eval + contraction path, not on any
/// physical density.
fn fixed_symmetric_dm(nao: usize) -> Density {
    let mut data = vec![0.0_f64; nao * nao];
    // Deterministic seed: a smooth bounded function of (mu, nu). Avoids all-zero
    // and avoids large magnitudes that would dwarf the tolerance.
    for mu in 0..nao {
        for nu in 0..nao {
            let raw = 0.5 + 0.3 * (((mu * 7 + nu * 3) % 11) as f64 / 11.0)
                - 0.15 * (((mu * 5 + nu) % 13) as f64 / 13.0);
            data[mu * nao + nu] = raw;
        }
    }
    // Symmetrize D = (D + Dᵀ)/2.
    let mut sym = vec![0.0_f64; nao * nao];
    for mu in 0..nao {
        for nu in 0..nao {
            sym[mu * nao + nu] = 0.5 * (data[mu * nao + nu] + data[nu * nao + mu]);
        }
    }
    Density { nao, data: sym }
}

/// Independent longhand reference: ρ[g] = Σ_μ Σ_ν ao[g,μ]·D[μν]·ao[g,ν].
/// F-order AO index `ao[g + mu*ngrids]` (matches `EvalGtoOutput` / eval_rho's
/// `ao_at`). A plain nested f64 loop — NOT a call to eval_rho, NOT oracle_sum.
fn rho_longhand(ao: &[f64], dm: &Density, ngrids: usize, nao: usize) -> Vec<f64> {
    let mut rho = vec![0.0_f64; ngrids];
    for g in 0..ngrids {
        let mut acc = 0.0_f64;
        for mu in 0..nao {
            let a_mu = ao[g + mu * ngrids];
            for nu in 0..nao {
                let a_nu = ao[g + nu * ngrids];
                acc += a_mu * dm.data[mu * nao + nu] * a_nu;
            }
        }
        rho[g] = acc;
    }
    rho
}

#[test]
fn numint_device_ao_path_eval_rho_matches_longhand() {
    let mol = h2o_ccpvdz();
    let nao = mol.nao_nr;
    assert!(nao > 0, "H2O/cc-pVDZ must have AOs");

    let coords = grid_coords();
    let ngrids = coords.len();

    // ── ASSERTION A ────────────────────────────────────────────────────────
    // THE device AO block numint::eval_gto_block produces (GTOval_sph → value
    // kernel via select_backend; PYSCF_BACKEND unset → CpuRuntime device kernel).
    let ao = pyscf_gto::eval_gto(&mol, "GTOval_sph", &coords)
        .expect("eval_gto GTOval_sph")
        .values;
    assert_eq!(
        ao.len(),
        ngrids * nao,
        "GTOval_sph value block is [ngrids, nao] F-order"
    );

    let dm = fixed_symmetric_dm(nao);
    let (rho, grad) = NumInt::eval_rho(&ao, &dm, ngrids, XcType::Lda)
        .expect("eval_rho(device AO, fixed dm, LDA)");
    assert!(grad.is_none(), "LDA eval_rho returns no gradient");
    assert_eq!(rho.len(), ngrids, "one ρ per grid point");

    let want = rho_longhand(&ao, &dm, ngrids, nao);
    for g in 0..ngrids {
        let diff = (rho[g] - want[g]).abs();
        assert!(
            diff < 1e-9,
            "Assertion A: ρ[{g}] = {} vs independent longhand {} (|Δ| = {:.3e}, tol 1e-9)",
            rho[g],
            want[g],
            diff
        );
    }

    // ── ASSERTION B ────────────────────────────────────────────────────────
    // GTOval_sph_deriv1 comp-0 (value slice of the [4,ngrids,nao] buffer) is a
    // SEPARATE device kernel; its value numerics must equal GTOval_sph exactly.
    let ao_deriv1 = pyscf_gto::eval_gto(&mol, "GTOval_sph_deriv1", &coords)
        .expect("eval_gto GTOval_sph_deriv1")
        .values;
    assert_eq!(
        ao_deriv1.len(),
        4 * ngrids * nao,
        "GTOval_sph_deriv1 is [4, ngrids, nao]"
    );
    // comp-0 occupies values[0 .. ngrids*nao].
    let value_slice = &ao_deriv1[0..ngrids * nao];
    for i in 0..ngrids * nao {
        let diff = (value_slice[i] - ao[i]).abs();
        assert!(
            diff < 1e-12,
            "Assertion B: deriv1 comp-0[{i}] = {} vs GTOval_sph value {} (|Δ| = {:.3e}, tol 1e-12)",
            value_slice[i],
            ao[i],
            diff
        );
    }
}
// NOTE (rocm variant): a `#[cfg(feature = "rocm")]` sibling running Assertion A
// on a device-resolved backend was considered, but pyscf-dft declares no `rocm`
// feature (only `default`, `libxc`, `python`), so gating on it emits an
// unexpected-cfg warning — NOT a clean addition. The always-on default-backend
// test above IS the gate: it already drives the device `eval_gto` kernel path
// (CpuRuntime `#[cube]` kernel via select_backend) end-to-end. Selecting a
// physical device backend is an env concern (`PYSCF_BACKEND`) over the SAME
// public `eval_gto` surface — covered by the pyscf-gto / pyscf-kernels device
// tests, not duplicated here.
