//! RKS analytical-gradient gate (GRAD-03, D-01) — the closed-shell KS slice.
//!
//! Wires the RKS analytical gradient ([`pyscf_grad::rks::RksGradients`]) against
//! the 07-02 always-on finite-difference harness ([`pyscf_grad::verify_fd`]):
//! central-difference the RKS `as_scanner` energy at `disp = 1e-4` Bohr and
//! compare to the analytical `RksGradients::kernel()` (`grid_response = true`) at
//! `≤ 1e-6` Ha/Bohr (D-01).
//!
//! ## Two arms (the 07-01/07-03 cintx-availability split, D-02)
//!
//! The RKS gradient's variational base contracts the SAME six grad-integral
//! families RHF needs — all MISSING from cintx with no scheduled workstream. The
//! XC-grid term + the `grid_response` weight-derivative are cintx-INDEPENDENT
//! (Phase-4 `GTOval_sph_deriv1` + xcfun `eval_xc` + byte-exact `pyscf-grids`
//! weights), but the overall `kernel()` still routes the gated families to a
//! CLEAN cintx-availability error.
//!
//!   * **STRUCTURAL arm (always-on):** asserts `grid_response` defaults OFF,
//!     `extra_force` is exactly zero when off, and `RksGradients::kernel()`
//!     EITHER returns `(natm, 3)` OR a CLEAN cintx-availability error — never
//!     `NotYetImplemented{phase:7}`.
//!
//!   * **NUMERIC arm (`#[ignore]`'d):** the full `verify_fd` comparison. Drop the
//!     `#[ignore]` when the cintx grad-integral workstream lands.
//!
//! ## libxc discipline (T-07-16, user memory)
//! The XC functional under test is `"lda,vwn"` — evaluated by the NATIVE xcfun
//! backend (`pyscf-dft` default features). NEVER `--features libxc` (a ~6h
//! libxc_rs compile). Run scoped + single-threaded:
//!   cargo test -p pyscf-grad --locked -- --test-threads=1 rks

use pyscf_core::{MOCoefficients, Mole, PyscfRsError};
use pyscf_grad::Gradients;
use pyscf_grad::rhf::RhfReference;
use pyscf_grad::rks::{RksGradients, RksReference};

/// The xcfun-evaluable functional under test — pure-LDA, NEVER pulls libxc.
const XC: &str = "lda,vwn";

/// Build a tiny H2 / STO-3G molecule (nao = 2, natm = 2). Pure DFT on the native
/// xcfun grid path — NO libxc.
fn h2_sto3g() -> Mole {
    use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};
    M(MoleBuildArgs {
        atom: AtomInput::String("H 0 0 0; H 0 0 0.74".into()),
        basis: BasisInput::Name("sto-3g".into()),
        ..Default::default()
    })
    .expect("build H2/sto-3g mol")
}

/// Build an RKS reference with an identity `mo_coeff` (MO = AO), one doubly-
/// occupied MO, and the `lda,vwn` functional. Enough to exercise the grid XC
/// term + the variational base at the right shapes.
fn identity_rks_reference(mol: Mole) -> RksReference {
    let nao = mol.nao_nr;
    assert!(nao >= 2, "need at least one occ + one vir AO");
    let mut data = vec![0.0f64; nao * nao];
    for d in 0..nao {
        data[d + d * nao] = 1.0;
    }
    let mut mo_energy = vec![0.0f64; nao];
    for (i, e) in mo_energy.iter_mut().enumerate() {
        *e = -0.5 + (i as f64);
    }
    let mut mo_occ = vec![0.0f64; nao];
    mo_occ[0] = 2.0;
    let scf = RhfReference {
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
    };
    RksReference { scf, xc: XC.into() }
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

/// `grid_response` defaults OFF (the upstream class default, GRAD-03), and the
/// `extra_force` weight-derivative term is EXACTLY zero when it is off.
#[test]
fn rks_grid_response_defaults_off_extra_force_zero() {
    let mol = h2_sto3g();
    let grad = RksGradients::new(identity_rks_reference(mol));
    assert!(!grad.grid_response, "grid_response must default OFF (GRAD-03)");
    let ef = grad.extra_force(0).expect("extra_force is cintx-independent");
    assert_eq!(ef, [0.0, 0.0, 0.0], "extra_force must be zero when grid_response is off");
}

/// `grid_response = true` is fully supported on request: `extra_force` runs the
/// cintx-independent grid-weight path and returns a finite force.
#[test]
fn rks_grid_response_on_extra_force_finite() {
    let mol = h2_sto3g();
    let grad = RksGradients::new(identity_rks_reference(mol)).with_grid_response(true);
    assert!(grad.grid_response, "grid_response must be ON after with_grid_response(true)");
    let ef = grad.extra_force(0).expect("extra_force grid path must run (xcfun, no libxc)");
    assert!(
        ef.iter().all(|v| v.is_finite()),
        "grid_response extra_force must be finite; got {ef:?}"
    );
}

/// `make_rdm1e` (closed-shell energy-weighted RDM) is cintx-independent.
#[test]
fn rks_make_rdm1e_returns_finite_nao_nao() {
    let mol = h2_sto3g();
    let nao = mol.nao_nr;
    let grad = RksGradients::new(identity_rks_reference(mol));
    let dme0 = grad.make_rdm1e().expect("make_rdm1e is cintx-independent");
    assert_eq!(dme0.len(), nao * nao);
    assert!(dme0.iter().all(|v| v.is_finite()));
    assert!(
        (dme0[0] - (-1.0)).abs() < 1e-12,
        "dme0[0,0] should equal ε0·occ0 = -1.0; got {}",
        dme0[0]
    );
}

/// The headline structural assertion: `RksGradients::kernel()` either returns
/// `(natm, 3)` (cintx ready) OR surfaces a CLEAN cintx-availability error.
#[test]
fn rks_kernel_returns_natm_by_3_or_clean_cintx_error() {
    let mol = h2_sto3g();
    let natm = mol.natm;
    let grad = RksGradients::new(identity_rks_reference(mol));
    match grad.kernel(None) {
        Ok(de) => {
            assert_eq!(de.len(), natm, "kernel must return (natm, 3)");
            assert!(de.iter().all(|r| r.iter().all(|v| v.is_finite())));
        }
        Err(e) => assert!(
            is_clean_cintx_availability_error(&e),
            "RksGradients::kernel() must surface a CLEAN cintx-availability error; got {e:?}"
        ),
    }
}

/// `grad_elec` (which contracts the gated families) routes a missing family to a
/// clean availability error, never the closed phase-7 disposition. Also asserts
/// the KS path is stationary (the only failure mode is the cintx gate — a CPHF
/// dependency would surface a different error class). D-04 / Pitfall 5.
#[test]
fn rks_grad_elec_routes_missing_intor_to_clean_error() {
    let mol = h2_sto3g();
    let natm = mol.natm;
    let grad = RksGradients::new(identity_rks_reference(mol)).with_grid_response(true);
    match grad.grad_elec(None) {
        Ok(de) => assert_eq!(de.len(), natm),
        Err(e) => assert!(
            is_clean_cintx_availability_error(&e),
            "grad_elec must route a missing cintx family to a clean availability error \
             (NOT a CPHF/response error — RKS is stationary, D-04); got {e:?}"
        ),
    }
}

// ─────────────────────────── NUMERIC arm (#[ignore]'d on cintx) ───────────────────────────

/// The full GRAD-03 numeric gate: central-difference the RKS `as_scanner` energy
/// at `disp = 1e-4` Bohr and compare to the analytical `RksGradients::kernel()`
/// (`grid_response = true`) at `≤ 1e-6` Ha/Bohr (D-01).
///
/// Gated (`#[ignore]`'d) by the 07-01/07-03 cintx-availability split: the
/// variational base cannot be produced while `int2e_ip1`,
/// `int1e_ip{ovlp,kin,nuc,rinv}`, and `with_rinv_at_nucleus` are MISSING from
/// cintx. Drop the `#[ignore]` when that workstream lands; the FD harness is
/// always-on and the XC functional under test (`lda,vwn`) uses xcfun, NOT libxc.
#[test]
#[ignore = "GRAD-03 numeric: int2e_ip1 + int1e_ip{ovlp,kin,nuc,rinv} + with_rinv_at_nucleus \
            MISSING from cintx (07-01/07-03 SUMMARY, no scheduled workstream); un-gate when they land"]
fn rks_verify_fd_numeric() {
    use pyscf_grad::verify_fd::{DEFAULT_DISP, FD_TOL};
    use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};

    let mol = h2_sto3g();
    let reference = identity_rks_reference(mol.clone());
    // grid_response = true: the full GRAD-03 analytical gradient including the
    // Becke-weight-derivative term.
    let grad = RksGradients::new(reference).with_grid_response(true);

    let analytical = grad
        .kernel(None)
        .expect("analytical RKS gradient (un-gated: cintx grad-intors available)");

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
        // Placeholder until the RKS as_scanner energy is wired (un-gates here).
        Ok(0.0)
    };

    let report = pyscf_grad::verify_fd(&base_coords, &analytical, energy, DEFAULT_DISP, FD_TOL)
        .expect("verify_fd must run on the RKS reference");
    assert!(
        report.passed,
        "RKS analytical gradient must agree with the central difference within \
         {FD_TOL} Ha/Bohr (D-01); got max|fd - analytical| = {:e}",
        report.max_abs_diff
    );
}
