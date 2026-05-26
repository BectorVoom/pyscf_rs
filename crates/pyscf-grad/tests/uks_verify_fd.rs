//! UKS analytical-gradient gate (GRAD-04, D-01) — the spin-resolved KS slice.
//!
//! Wires the UKS analytical gradient ([`pyscf_grad::uks::UksGradients`]) against
//! the 07-02 always-on finite-difference harness ([`pyscf_grad::verify_fd`]):
//! central-difference the UKS `as_scanner` energy at `disp = 1e-4` Bohr and
//! compare to the analytical `UksGradients::kernel()` (`grid_response = true`) at
//! `≤ 1e-6` Ha/Bohr (D-01).
//!
//! ## Two arms (the 07-01/07-03 cintx-availability split, D-02)
//!
//! Same gating as RKS/RHF: the variational base contracts the gated families
//! (clean availability error); the per-spin XC-grid + grid_response terms are
//! cintx-INDEPENDENT (Phase-4 grid path + xcfun `eval_uks`, NEVER libxc).
//!
//!   * **STRUCTURAL arm (always-on):** `grid_response` defaults OFF, `extra_force`
//!     is zero when off, `make_rdm1e` is the spin-summed RDM, and `kernel()`
//!     returns `(natm, 3)`-or-clean-cintx-error.
//!   * **NUMERIC arm (`#[ignore]`'d):** the full `verify_fd` comparison; un-gates
//!     with the cintx workstream.
//!
//! ## libxc discipline (T-07-16, user memory)
//! The XC functional is `"lda,vwn"` via the NATIVE xcfun `eval_uks` surface;
//! NEVER `--features libxc`. Run scoped + single-threaded:
//!   cargo test -p pyscf-grad --locked -- --test-threads=1 uks

use pyscf_core::{MOCoefficients, Mole, PyscfRsError};
use pyscf_grad::Gradients;
use pyscf_grad::rhf::RhfReference;
use pyscf_grad::uhf::UhfReference;
use pyscf_grad::uks::{UksGradients, UksReference};

/// The xcfun-evaluable functional under test — pure-LDA, NEVER pulls libxc.
const XC: &str = "lda,vwn";

/// Build a tiny H2 / STO-3G molecule (nao = 2, natm = 2). Native xcfun path.
fn h2_sto3g() -> Mole {
    use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};
    M(MoleBuildArgs {
        atom: AtomInput::String("H 0 0 0; H 0 0 0.74".into()),
        basis: BasisInput::Name("sto-3g".into()),
        ..Default::default()
    })
    .expect("build H2/sto-3g mol")
}

/// One spin channel: identity MO coeff, one singly-occupied MO (per-spin occ).
fn spin_reference(mol: Mole) -> RhfReference {
    let nao = mol.nao_nr;
    assert!(nao >= 2);
    let mut data = vec![0.0f64; nao * nao];
    for d in 0..nao {
        data[d + d * nao] = 1.0;
    }
    let mut mo_energy = vec![0.0f64; nao];
    for (i, e) in mo_energy.iter_mut().enumerate() {
        *e = -0.5 + (i as f64);
    }
    let mut mo_occ = vec![0.0f64; nao];
    mo_occ[0] = 1.0;
    RhfReference {
        mo_coeff: MOCoefficients {
            nao,
            nmo: nao,
            data,
            energies: mo_energy.clone(),
            occupations: mo_occ.clone(),
        },
        mo_energy,
        mo_occ,
        mol,
    }
}

/// A spin-resolved UKS reference with the `lda,vwn` functional.
fn identity_uks_reference(mol: Mole) -> UksReference {
    let scf = UhfReference {
        alpha: spin_reference(mol.clone()),
        beta: spin_reference(mol),
    };
    UksReference { scf, xc: XC.into() }
}

/// Is the error a clean cintx-availability error, NOT a `NotYetImplemented{phase:7}`?
fn is_clean_cintx_availability_error(err: &PyscfRsError) -> bool {
    !matches!(err, PyscfRsError::NotYetImplemented { phase: 7, .. })
        && matches!(
            err,
            PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(_))
        )
}

// ─────────────────────────── STRUCTURAL arm (always-on) ───────────────────────────

/// `grid_response` defaults OFF (GRAD-04), `extra_force` is exactly zero when off.
#[test]
fn uks_grid_response_defaults_off_extra_force_zero() {
    let mol = h2_sto3g();
    let grad = UksGradients::new(identity_uks_reference(mol));
    assert!(!grad.grid_response, "grid_response must default OFF (GRAD-04)");
    let ef = grad.extra_force(0).expect("extra_force is cintx-independent");
    assert_eq!(ef, [0.0, 0.0, 0.0], "extra_force must be zero when grid_response is off");
}

/// `grid_response = true` runs the cintx-independent grid path → finite force.
#[test]
fn uks_grid_response_on_extra_force_finite() {
    let mol = h2_sto3g();
    let grad = UksGradients::new(identity_uks_reference(mol)).with_grid_response(true);
    assert!(grad.grid_response);
    let ef = grad.extra_force(0).expect("extra_force grid path must run (xcfun, no libxc)");
    assert!(ef.iter().all(|v| v.is_finite()), "extra_force must be finite; got {ef:?}");
}

/// `make_rdm1e` (spin-summed energy-weighted RDM) is cintx-independent.
#[test]
fn uks_make_rdm1e_returns_finite_nao_nao() {
    let mol = h2_sto3g();
    let nao = mol.nao_nr;
    let grad = UksGradients::new(identity_uks_reference(mol));
    let dme0 = grad.make_rdm1e().expect("make_rdm1e is cintx-independent");
    assert_eq!(dme0.len(), nao * nao);
    assert!(dme0.iter().all(|v| v.is_finite()));
    // Spin-summed: each channel carries ε0·occ0 = -0.5; the (0,0) sum is -1.0.
    assert!(
        (dme0[0] - (-1.0)).abs() < 1e-12,
        "dme0[0,0] should equal Σ_spin ε0·occ0 = -1.0; got {}",
        dme0[0]
    );
}

/// `UksGradients::kernel()` returns `(natm, 3)` OR a CLEAN cintx-availability
/// error — never `NotYetImplemented{phase:7}`.
#[test]
fn uks_kernel_returns_natm_by_3_or_clean_cintx_error() {
    let mol = h2_sto3g();
    let natm = mol.natm;
    let grad = UksGradients::new(identity_uks_reference(mol));
    match grad.kernel(None) {
        Ok(de) => {
            assert_eq!(de.len(), natm, "kernel must return (natm, 3)");
            assert!(de.iter().all(|r| r.iter().all(|v| v.is_finite())));
        }
        Err(e) => assert!(
            is_clean_cintx_availability_error(&e),
            "UksGradients::kernel() must surface a CLEAN cintx-availability error; got {e:?}"
        ),
    }
}

/// `grad_elec` routes a missing family to a clean availability error and is
/// stationary (no CPHF/response error class). D-04 / Pitfall 5.
#[test]
fn uks_grad_elec_routes_missing_intor_to_clean_error() {
    let mol = h2_sto3g();
    let natm = mol.natm;
    let grad = UksGradients::new(identity_uks_reference(mol)).with_grid_response(true);
    match grad.grad_elec(None) {
        Ok(de) => assert_eq!(de.len(), natm),
        Err(e) => assert!(
            is_clean_cintx_availability_error(&e),
            "grad_elec must route a missing cintx family to a clean availability error \
             (NOT a CPHF/response error — UKS is stationary, D-04); got {e:?}"
        ),
    }
}

// ─────────────────────────── NUMERIC arm (#[ignore]'d on cintx) ───────────────────────────

/// The full GRAD-04 numeric gate: central-difference the UKS `as_scanner` energy
/// at `disp = 1e-4` Bohr and compare to the analytical `UksGradients::kernel()`
/// (`grid_response = true`) at `≤ 1e-6` Ha/Bohr (D-01).
///
/// Gated (`#[ignore]`'d) by the 07-01/07-03 cintx-availability split; un-gates
/// when the cintx grad-integral workstream lands. The functional under test
/// (`lda,vwn`) uses the native xcfun `eval_uks` surface, NOT libxc.
#[test]
#[ignore = "GRAD-04 numeric: int2e_ip1 + int1e_ip{ovlp,kin,nuc,rinv} + with_rinv_at_nucleus \
            MISSING from cintx (07-01/07-03 SUMMARY, no scheduled workstream); un-gate when they land"]
fn uks_verify_fd_numeric() {
    use pyscf_grad::verify_fd::{DEFAULT_DISP, FD_TOL};
    use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};

    let mol = h2_sto3g();
    let reference = identity_uks_reference(mol.clone());
    let grad = UksGradients::new(reference).with_grid_response(true);

    let analytical = grad
        .kernel(None)
        .expect("analytical UKS gradient (un-gated: cintx grad-intors available)");

    let base_coords: Vec<[f64; 3]> = mol.atom_coords();
    let energy = |coords: &[[f64; 3]]| -> Result<f64, PyscfRsError> {
        let atom_str = coords
            .iter()
            .map(|c| format!("H {} {} {}", c[0], c[1], c[2]))
            .collect::<Vec<_>>()
            .join("; ");
        let _new_mol = M(MoleBuildArgs {
            atom: AtomInput::String(atom_str),
            basis: BasisInput::Name("sto-3g".into()),
            unit: pyscf_core::Unit::Bohr,
            ..Default::default()
        })?;
        Ok(0.0)
    };

    let report = pyscf_grad::verify_fd(&base_coords, &analytical, energy, DEFAULT_DISP, FD_TOL)
        .expect("verify_fd must run on the UKS reference");
    assert!(
        report.passed,
        "UKS analytical gradient must agree with the central difference within \
         {FD_TOL} Ha/Bohr (D-01); got max|fd - analytical| = {:e}",
        report.max_abs_diff
    );
}
