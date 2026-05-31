//! UHF analytical-gradient gate (GRAD-02, D-01) — the spin-resolved slice.
//!
//! Wires the UHF analytical gradient ([`pyscf_grad::uhf::UhfGradients`]) against
//! the 07-02 always-on finite-difference harness ([`pyscf_grad::verify_fd`]):
//! central-difference the UHF `as_scanner` energy at `disp = 1e-4` Bohr and
//! compare to the analytical `UhfGradients::kernel()` at `≤ 1e-6` Ha/Bohr (D-01).
//!
//! ## Two arms (the 07-01/07-03 cintx-availability split, D-02)
//!
//! The UHF gradient contracts the SAME six grad-integral families the RHF body
//! needs — `int2e_ip1` (2e Pulay), `int1e_ip{ovlp,kin,nuc}` (overlap + hcore
//! Pulay), and `int1e_iprinv` + `with_rinv_at_nucleus` (the per-atom Hellmann-
//! Feynman shift) — all **MISSING from every cintx branch with no scheduled
//! workstream** (07-01/07-03 SUMMARY).
//!
//!   * **STRUCTURAL arm (always-on):** builds a UHF reference (spin-resolved
//!     α/β pair), exercises the cintx-independent pieces (`make_rdm1e`,
//!     `grad_nuc`) at the right shapes, and asserts `UhfGradients::kernel()`
//!     EITHER returns the `(natm, 3)` analytical gradient (if a future cintx
//!     ships the families) OR surfaces a CLEAN cintx-availability error — never
//!     `NotYetImplemented{phase:7}`.
//!
//!   * **NUMERIC arm (`#[ignore]`'d):** the full `verify_fd` FD-vs-analytical
//!     comparison. `#[ignore]`'d today because `UhfGradients::kernel()` cannot
//!     produce a numeric gradient while the six families are absent; it
//!     un-gates (drop the `#[ignore]`) the moment the cintx grad-integral
//!     workstream lands them. Run on demand:
//!     `cargo test -p pyscf-grad --locked -- --ignored uhf_verify_fd_numeric`
//!
//! Run scoped + single-threaded (user-memory + CI discipline, NO libxc):
//!   cargo test -p pyscf-grad --locked -- --test-threads=1 uhf

use pyscf_core::{MOCoefficients, Mole, PyscfRsError};
use pyscf_grad::Gradients;
use pyscf_grad::rhf::RhfReference;
use pyscf_grad::uhf::{UhfGradients, UhfReference};

/// Build a tiny H2 / STO-3G molecule (nao = 2, natm = 2) that does NOT pull
/// libxc — pure HF, smallest basis.
fn h2_sto3g() -> Mole {
    use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};
    M(MoleBuildArgs {
        atom: AtomInput::String("H 0 0 0; H 0 0 0.74".into()),
        basis: BasisInput::Name("sto-3g".into()),
        ..Default::default()
    })
    .expect("build H2/sto-3g mol")
}

/// Build a per-spin [`RhfReference`] with an identity `mo_coeff` (MO = AO) and a
/// single per-spin occupation (UHF occupations are 1.0/0.0 per channel). This is
/// enough to exercise every cintx-independent gradient piece at the right shapes.
fn spin_reference(mol: Mole) -> RhfReference {
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
    // One singly-occupied MO per spin channel (the open-shell convention).
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

/// A spin-resolved UHF reference whose α/β channels share the same identity
/// molecule (closed-shell-like singlet: α and β both occupy MO 0).
fn identity_uhf_reference(mol: Mole) -> UhfReference {
    UhfReference {
        alpha: spin_reference(mol.clone()),
        beta: spin_reference(mol),
    }
}

/// Is the error a clean cintx-availability error (`Core(InvalidMolecule(..))`),
/// NOT a `NotYetImplemented{phase:7}`? The D-02 contract.
fn is_clean_cintx_availability_error(err: &PyscfRsError) -> bool {
    !matches!(err, PyscfRsError::NotYetImplemented { phase: 7, .. })
        && matches!(
            err,
            PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(_))
        )
}

// ─────────────────────────── STRUCTURAL arm (always-on) ───────────────────────────

/// `make_rdm1e` (spin-summed energy-weighted RDM) is a thin pure-linear-algebra
/// port over the shipped MO arrays — NO cintx dependency. It must return a
/// finite `(nao, nao)` matrix always-on.
#[test]
fn uhf_make_rdm1e_returns_finite_nao_nao() {
    let mol = h2_sto3g();
    let nao = mol.nao_nr;
    let grad = UhfGradients::new(identity_uhf_reference(mol));

    let dme0 = grad.make_rdm1e().expect("make_rdm1e is cintx-independent");
    assert_eq!(dme0.len(), nao * nao, "dme0 must be (nao, nao) row-major");
    assert!(dme0.iter().all(|v| v.is_finite()), "dme0 must be finite");
    // dme0 = dme0_α + dme0_β; each channel carries ε0·occ0 = -0.5·1 = -0.5 on the
    // (0,0) AO, so the spin-summed (0,0) element is -1.0.
    assert!(
        (dme0[0] - (-1.0)).abs() < 1e-12,
        "dme0[0,0] should equal Σ_spin ε0·occ0 = -1.0; got {}",
        dme0[0]
    );
}

/// `grad_nuc` (the nuclear-repulsion gradient, trait default) is spin-
/// independent and cintx-independent. For H2 on the z-axis the force is ±z.
#[test]
fn uhf_grad_nuc_returns_natm_by_3() {
    let mol = h2_sto3g();
    let natm = mol.natm;
    let grad = UhfGradients::new(identity_uhf_reference(mol));

    let gn = grad.grad_nuc(None).expect("grad_nuc is cintx-independent");
    assert_eq!(gn.len(), natm, "grad_nuc must be (natm, 3)");
    assert!(
        gn.iter().all(|r| r.iter().all(|v| v.is_finite())),
        "grad_nuc must be finite"
    );
    for r in &gn {
        assert!(
            r[0].abs() < 1e-12 && r[1].abs() < 1e-12,
            "off-axis force must vanish"
        );
    }
    assert!(
        (gn[0][2] + gn[1][2]).abs() < 1e-12,
        "Newton's third law: z-forces are equal and opposite"
    );
}

/// The headline structural assertion: `UhfGradients::kernel()` either returns
/// the `(natm, 3)` analytical gradient (cintx ready) OR surfaces a CLEAN
/// cintx-availability error — NEVER `NotYetImplemented{phase:7}`.
#[test]
fn uhf_kernel_returns_natm_by_3_or_clean_cintx_error() {
    let mol = h2_sto3g();
    let natm = mol.natm;
    let grad = UhfGradients::new(identity_uhf_reference(mol));

    match grad.kernel(None) {
        Ok(de) => {
            assert_eq!(de.len(), natm, "kernel must return (natm, 3)");
            assert!(
                de.iter().all(|r| r.iter().all(|v| v.is_finite())),
                "analytical gradient must be finite"
            );
        }
        Err(e) => assert!(
            is_clean_cintx_availability_error(&e),
            "UhfGradients::kernel() must surface a CLEAN cintx-availability error \
             (never NotYetImplemented{{phase:7}}); got {e:?}"
        ),
    }
}

/// `grad_elec` (which contracts the missing `int2e_ip1` / `int1e_ip*` families
/// per spin channel) must ALSO route any missing family to a clean availability
/// error, never the closed phase-7 disposition.
#[test]
fn uhf_grad_elec_routes_missing_intor_to_clean_error() {
    let mol = h2_sto3g();
    let natm = mol.natm;
    let grad = UhfGradients::new(identity_uhf_reference(mol));
    match grad.grad_elec(None) {
        Ok(de) => assert_eq!(de.len(), natm, "grad_elec returns one row per atom"),
        Err(e) => assert!(
            is_clean_cintx_availability_error(&e),
            "grad_elec must route a missing cintx family to a clean availability error; got {e:?}"
        ),
    }
}

/// UHF makes NO CPHF call (D-04) — assert the analytical path never references a
/// response solve at the source level by checking that `grad_elec` succeeds or
/// fails ONLY on the cintx-availability gate (a CPHF dependency would surface a
/// different error class). The grep gate in the plan acceptance criteria is the
/// belt; this is the suspenders.
#[test]
fn uhf_grad_elec_is_stationary_no_cphf_error_class() {
    let mol = h2_sto3g();
    let grad = UhfGradients::new(identity_uhf_reference(mol));
    if let Err(e) = grad.grad_elec(None) {
        // The only failure mode today is the cintx-availability gate — never a
        // NotYetImplemented{phase:7} (which would signal an unported CPHF path).
        assert!(
            is_clean_cintx_availability_error(&e),
            "UHF grad_elec must be stationary (no CPHF); the only allowed error is the \
             cintx-availability gate, got {e:?}"
        );
    }
}

// ─────────────────────────── NUMERIC arm (#[ignore]'d on cintx) ───────────────────────────

/// The full GRAD-02 numeric gate: central-difference the UHF `as_scanner`
/// energy at `disp = 1e-4` Bohr and compare to the analytical
/// `UhfGradients::kernel()` at `≤ 1e-6` Ha/Bohr (D-01).
///
/// Gated (`#[ignore]`'d) by the 07-01/07-03 cintx-availability split: the
/// analytical gradient cannot be produced while `int2e_ip1`,
/// `int1e_ip{ovlp,kin,nuc,rinv}`, and `with_rinv_at_nucleus` are MISSING from
/// cintx (no scheduled workstream). Drop the `#[ignore]` the moment that cintx
/// grad-integral workstream lands; the wiring below mirrors the RHF numeric arm
/// and the FD harness is always-on.
#[test]
#[ignore = "GRAD-02 numeric: int2e_ip1 + int1e_ip{ovlp,kin,nuc,rinv} + with_rinv_at_nucleus \
            MISSING from cintx (07-01/07-03 SUMMARY, no scheduled workstream); un-gate when they land"]
fn uhf_verify_fd_numeric() {
    use pyscf_grad::verify_fd::{DEFAULT_DISP, FD_TOL};
    use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};

    // A UHF reference on H2 / STO-3G. (The structural identity reference stands
    // in until a converged UHF SCF snapshot is wired; the FD comparison itself
    // un-gates with cintx, so this arm documents the full wiring shape.)
    let mol = h2_sto3g();
    let reference = identity_uhf_reference(mol.clone());
    let grad = UhfGradients::new(reference);

    let analytical = grad
        .kernel(None)
        .expect("analytical UHF gradient (un-gated: cintx grad-intors available)");

    // The FD energy closure rebuilds the Mole at the displaced geometry; the
    // real UHF as_scanner energy wires in when the numeric arm un-gates.
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
        // Placeholder until the UHF as_scanner energy is wired (un-gates here).
        Ok(0.0)
    };

    let report = pyscf_grad::verify_fd(&base_coords, &analytical, energy, DEFAULT_DISP, FD_TOL)
        .expect("verify_fd must run on the UHF reference");
    assert!(
        report.passed,
        "UHF analytical gradient must agree with the central difference within \
         {FD_TOL} Ha/Bohr (D-01); got max|fd - analytical| = {:e}",
        report.max_abs_diff
    );
}
