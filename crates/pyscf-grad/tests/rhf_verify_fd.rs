//! RHF analytical-gradient gate (GRAD-01, D-01) — the phase-headline slice.
//!
//! Wires the RHF analytical gradient ([`pyscf_grad::rhf::RhfGradients`]) against
//! the 07-02 always-on finite-difference harness ([`pyscf_grad::verify_fd`]):
//! central-difference the SCF `as_scanner` energy at `disp = 1e-4` Bohr and
//! compare to the analytical `RhfGradients::kernel()` at `≤ 1e-6` Ha/Bohr (D-01).
//!
//! ## Status (F-08, quick "fix f8", 2026-06-02)
//!
//! The grad-integral families now EVALUATE in cintx, and the RHF `grad_elec`
//! assembly is correct: the s-shell numeric gate (H2/STO-3G,
//! `rhf_verify_fd_numeric`) is **un-gated and PASSES to ~2.6e-9 Ha/Bohr**. Three
//! real assembly bugs were fixed to get there (component-fastest F-order layout;
//! `hcore_deriv` axis-1 slice; the `int2e_ip1` K-contraction l-output index).
//!
//!   * **STRUCTURAL arm (always-on):** builds an RHF reference, exercises the
//!     cintx-independent pieces (`make_rdm1e`, `grad_nuc`, `aoslice_by_atom`) at
//!     the right shapes, and asserts `RhfGradients::kernel()` EITHER returns the
//!     `(natm, 3)` analytical gradient OR surfaces a CLEAN cintx-availability
//!     error — never `NotYetImplemented{phase:7}` (GRAD-07 closed that).
//!
//!   * **NUMERIC s-shell arm (`rhf_verify_fd_numeric`, un-gated):** the full
//!     `verify_fd` FD-vs-analytical comparison on H2/STO-3G — passes.
//!
//!   * **NUMERIC p-shell arm (`rhf_verify_fd_numeric_pshell`, `#[ignore]`'d):**
//!     blocked by an EXTERNAL cintx kernel bug — `int1e_ipnuc`, `int1e_iprinv`
//!     (off-origin nuclei) and `int2e_ip1` are numerically WRONG for l≥1 shells
//!     (verified element-wise vs PySCF 2.12.1 while `int1e_ovlp`/`int2e`/
//!     `int1e_ipovlp`/`int1e_ipkin` are all correct to ~7e-9). Un-gate when
//!     cintx fixes its p-shell gradient kernels.
//!
//! Run scoped + single-threaded (user-memory + CI discipline, NO libxc):
//!   cargo test -p pyscf-grad --locked -- --test-threads=1 rhf_verify_fd

use pyscf_core::{MOCoefficients, Mole, PyscfRsError};
use pyscf_grad::Gradients;
use pyscf_grad::rhf::{RhfGradients, RhfReference, aoslice_by_atom};
use pyscf_grad::verify_fd::{DEFAULT_DISP, FD_TOL};
use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};

/// Build a tiny H2 / STO-3G molecule (nao = 2, natm = 2) that does NOT pull
/// libxc — pure HF, smallest basis.
fn h2_sto3g() -> Mole {
    M(MoleBuildArgs {
        atom: AtomInput::String("H 0 0 0; H 0 0 0.74".into()),
        basis: BasisInput::Name("sto-3g".into()),
        ..Default::default()
    })
    .expect("build H2/sto-3g mol")
}

/// Build an RHF reference snapshot with an identity `mo_coeff` (MO = AO) and a
/// well-defined closed-shell occ<vir spectrum. Mirrors the in-tree MP2 smoke's
/// `real_identity_reference`: this is enough to exercise every cintx-independent
/// gradient piece (`make_rdm1e`, `grad_nuc`, `aoslice_by_atom`) at the right
/// shapes for the STRUCTURAL arm.
fn identity_reference(mol: Mole) -> RhfReference {
    let nao = mol.nao_nr;
    assert!(nao >= 2, "need at least one occ + one vir AO");

    // Column-major identity: C[ao, mo] = δ at ao + mo*nao.
    let mut data = vec![0.0f64; nao * nao];
    for d in 0..nao {
        data[d + d * nao] = 1.0;
    }
    // One doubly-occupied MO (index 0), the rest virtual; occ energy below vir.
    let mut mo_energy = vec![0.0f64; nao];
    for (i, e) in mo_energy.iter_mut().enumerate() {
        *e = -0.5 + (i as f64);
    }
    let mut mo_occ = vec![0.0f64; nao];
    mo_occ[0] = 2.0;

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

/// Is the error a clean cintx-availability error (`Core(InvalidMolecule(..))`),
/// NOT a `NotYetImplemented{phase:7}`? The whole D-02 contract: missing cintx
/// families surface as availability errors, never the closed phase-7
/// disposition.
fn is_clean_cintx_availability_error(err: &PyscfRsError) -> bool {
    !matches!(err, PyscfRsError::NotYetImplemented { phase: 7, .. })
        && matches!(
            err,
            PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(_))
        )
}

// ─────────────────────────── STRUCTURAL arm (always-on) ───────────────────────────

/// `make_rdm1e` (energy-weighted RDM) is a thin pure-linear-algebra port over
/// the shipped MO arrays — NO cintx dependency. It must return a finite
/// `(nao, nao)` matrix always-on.
#[test]
fn rhf_make_rdm1e_returns_finite_nao_nao() {
    let mol = h2_sto3g();
    let nao = mol.nao_nr;
    let grad = RhfGradients::new(identity_reference(mol));

    let dme0 = grad.make_rdm1e().expect("make_rdm1e is cintx-independent");
    assert_eq!(dme0.len(), nao * nao, "dme0 must be (nao, nao) row-major");
    assert!(dme0.iter().all(|v| v.is_finite()), "dme0 must be finite");
    // The single occupied MO (identity coeff) carries weight ε0·occ0 = -0.5·2 = -1
    // on the (0,0) AO; off the occupied block it is zero.
    assert!(
        (dme0[0] - (-1.0)).abs() < 1e-12,
        "dme0[0,0] should equal ε0·occ0 = -1.0; got {}",
        dme0[0]
    );
}

/// `grad_nuc` (the nuclear-repulsion gradient, trait default) is a thin
/// Coulomb-force port — NO cintx dependency. It must return one `[x,y,z]` per
/// atom, finite, always-on. For H2 on the z-axis the force is purely ±z.
#[test]
fn rhf_grad_nuc_returns_natm_by_3() {
    let mol = h2_sto3g();
    let natm = mol.natm;
    let grad = RhfGradients::new(identity_reference(mol));

    let gn = grad.grad_nuc(None).expect("grad_nuc is cintx-independent");
    assert_eq!(gn.len(), natm, "grad_nuc must be (natm, 3)");
    assert!(
        gn.iter().all(|r| r.iter().all(|v| v.is_finite())),
        "grad_nuc must be finite"
    );
    // H2 aligned on z: the x/y nuclear-repulsion force is zero, z is non-zero
    // and equal-and-opposite on the two H.
    for r in &gn {
        assert!(
            r[0].abs() < 1e-12 && r[1].abs() < 1e-12,
            "off-axis force must vanish"
        );
    }
    assert!(
        gn[0][2].abs() > 1e-6,
        "on-axis nuclear force must be non-zero"
    );
    assert!(
        (gn[0][2] + gn[1][2]).abs() < 1e-12,
        "Newton's third law: z-forces are equal and opposite"
    );
}

/// `aoslice_by_atom` (the AO-range walk) is cintx-independent. For H2/STO-3G
/// each H carries one s-AO, so the slices are [0,1) and [1,2), covering nao.
#[test]
fn rhf_aoslice_by_atom_partitions_nao() {
    let mol = h2_sto3g();
    let natm = mol.natm;
    let nao = mol.nao_nr;
    let slices = aoslice_by_atom(&mol).expect("aoslice_by_atom is cintx-independent");
    assert_eq!(slices.len(), natm, "one (p0,p1) per atom");
    // Contiguous, non-overlapping, covering [0, nao).
    let mut covered = 0usize;
    for (p0, p1) in &slices {
        assert!(p0 <= p1 && *p1 <= nao, "slice [{p0},{p1}) within [0,{nao})");
        covered += p1 - p0;
    }
    assert_eq!(covered, nao, "the per-atom AO slices must cover all nao");
}

/// The headline structural assertion: `RhfGradients::kernel()` either returns
/// the `(natm, 3)` analytical gradient (cintx ready) OR surfaces a CLEAN
/// cintx-availability error — NEVER `NotYetImplemented{phase:7}`. This is the
/// always-on D-02 contract while the six grad-intor families are absent.
#[test]
fn rhf_kernel_returns_natm_by_3_or_clean_cintx_error() {
    let mol = h2_sto3g();
    let natm = mol.natm;
    let grad = RhfGradients::new(identity_reference(mol));

    match grad.kernel(None) {
        // Un-gated future: cintx shipped the families → a real (natm, 3) grad.
        Ok(de) => {
            assert_eq!(de.len(), natm, "kernel must return (natm, 3)");
            assert!(
                de.iter().all(|r| r.iter().all(|v| v.is_finite())),
                "analytical gradient must be finite"
            );
        }
        // Expected today: a missing cintx grad-intor family surfaces as a clean
        // availability error, never the closed phase-7 disposition.
        Err(e) => assert!(
            is_clean_cintx_availability_error(&e),
            "RhfGradients::kernel() must surface a CLEAN cintx-availability error \
             (never NotYetImplemented{{phase:7}}); got {e:?}"
        ),
    }
}

/// `grad_elec` (which contracts the missing `int2e_ip1` / `int1e_ip*` families)
/// must ALSO route any missing family to a clean availability error, never the
/// closed phase-7 disposition.
#[test]
fn rhf_grad_elec_routes_missing_intor_to_clean_error() {
    let mol = h2_sto3g();
    let natm = mol.natm;
    let grad = RhfGradients::new(identity_reference(mol));
    match grad.grad_elec(None) {
        Ok(de) => assert_eq!(de.len(), natm, "grad_elec returns one row per atom"),
        Err(e) => assert!(
            is_clean_cintx_availability_error(&e),
            "grad_elec must route a missing cintx family to a clean availability error; got {e:?}"
        ),
    }
}

// ─────────────────────────── NUMERIC arm (#[ignore]'d on cintx) ───────────────────────────

/// The full GRAD-01 numeric gate: central-difference the SCF `as_scanner`
/// energy at `disp = 1e-4` Bohr and compare to the analytical
/// `RhfGradients::kernel()` at `≤ 1e-6` Ha/Bohr (D-01).
///
/// Un-gated (F-08, 2026-06-02): the grad integrals evaluate and the assembly is
/// correct; H2/STO-3G (s-shells only) agrees with the central difference to
/// ~2.6e-9 Ha/Bohr.
#[test]
fn rhf_verify_fd_numeric() {
    use pyscf_scf::{RHF, as_scanner};

    // A converged RHF reference on H2 / STO-3G (tight tolerance for FD).
    let mol = h2_sto3g();
    let mut rhf = RHF::new(mol.clone());
    rhf.conv_tol = 1e-12;
    rhf.max_cycle = 100;
    rhf.kernel()
        .expect("RHF SCF must converge for the FD reference");

    let mo_coeff = rhf.mo_coeff.clone().expect("converged mo_coeff");
    let mo_energy = rhf.mo_energy.clone().expect("converged mo_energy");
    let mo_occ = rhf.mo_occ.clone().expect("converged mo_occ");
    let reference = RhfReference {
        mo_coeff,
        mo_energy,
        mo_occ,
        mol: mol.clone(),
    };
    let grad = RhfGradients::new(reference);

    // The analytical gradient (the thing under test).
    let analytical = grad
        .kernel(None)
        .expect("analytical RHF gradient (un-gated: cintx grad-intors available)");

    // The SCF as_scanner energy expressed over per-atom Bohr coords: rebuild the
    // Mole at the displaced geometry and run the energy kernel (the verify_fd
    // coord-closure contract from 07-02).
    let scanner = as_scanner(&rhf);
    let base_coords: Vec<[f64; 3]> = mol.atom_coords();
    let symbols: Vec<String> = mol_atom_symbols(&mol);
    let energy = |coords: &[[f64; 3]]| -> Result<f64, PyscfRsError> {
        let atom_str = geometry_string(&symbols, coords);
        let new_mol = M(MoleBuildArgs {
            atom: AtomInput::String(atom_str),
            basis: BasisInput::Name("sto-3g".into()),
            unit: pyscf_core::Unit::Bohr,
            ..Default::default()
        })?;
        scanner(&new_mol).map(|e| e.0)
    };

    let report = pyscf_grad::verify_fd(&base_coords, &analytical, energy, DEFAULT_DISP, FD_TOL)
        .expect("verify_fd must run on the RHF reference");
    assert!(
        report.passed,
        "RHF analytical gradient must agree with the central difference within \
         {FD_TOL} Ha/Bohr (D-01); got max|fd - analytical| = {:e}",
        report.max_abs_diff
    );
}

/// The p-shell numeric gate: bent H2O/STO-3G (O carries a 2p shell). The RHF
/// assembly is correct (s-shells pass above), but this is **`#[ignore]`'d on an
/// EXTERNAL cintx kernel bug**: `int1e_ipnuc`, `int1e_iprinv` (off-origin
/// nuclei) and `int2e_ip1` are numerically WRONG for l≥1 shells. Verified
/// 2026-06-02 by element-wise comparison vs PySCF 2.12.1: cintx
/// `int1e_ovlp`/`int2e`/`int1e_ipovlp`/`int1e_ipkin`/`int1e_iprinv@first-atom`
/// are all correct (~7e-9), but the families above diverge by 0.2–1.1 on bent
/// H2O. Drop the `#[ignore]` when cintx fixes its p-shell gradient kernels.
#[test]
#[ignore = "F-08: cintx int1e_ipnuc / int1e_iprinv(off-origin) / int2e_ip1 are numerically \
            WRONG for l>=1 shells (external cintx kernel bug; ipovlp/ipkin/int2e are correct). \
            Un-gate when cintx fixes its p-shell gradient kernels."]
fn rhf_verify_fd_numeric_pshell() {
    use pyscf_scf::{RHF, as_scanner};

    let mol = M(MoleBuildArgs {
        atom: AtomInput::String("O 0.0 0.0 0.12; H 0.0 0.75 -0.48; H 0.1 -0.75 -0.46".into()),
        basis: BasisInput::Name("sto-3g".into()),
        ..Default::default()
    })
    .expect("build H2O/sto-3g mol");
    let mut rhf = RHF::new(mol.clone());
    rhf.conv_tol = 1e-12;
    rhf.max_cycle = 100;
    rhf.kernel()
        .expect("RHF SCF must converge for the FD reference");

    let reference = RhfReference {
        mo_coeff: rhf.mo_coeff.clone().expect("converged mo_coeff"),
        mo_energy: rhf.mo_energy.clone().expect("converged mo_energy"),
        mo_occ: rhf.mo_occ.clone().expect("converged mo_occ"),
        mol: mol.clone(),
    };
    let grad = RhfGradients::new(reference);
    let analytical = grad.kernel(None).expect("analytical RHF gradient");

    let scanner = as_scanner(&rhf);
    let base_coords: Vec<[f64; 3]> = mol.atom_coords();
    let symbols: Vec<String> = mol_atom_symbols(&mol);
    let energy = |coords: &[[f64; 3]]| -> Result<f64, PyscfRsError> {
        let new_mol = M(MoleBuildArgs {
            atom: AtomInput::String(geometry_string(&symbols, coords)),
            basis: BasisInput::Name("sto-3g".into()),
            unit: pyscf_core::Unit::Bohr,
            ..Default::default()
        })?;
        scanner(&new_mol).map(|e| e.0)
    };
    let report = pyscf_grad::verify_fd(&base_coords, &analytical, energy, DEFAULT_DISP, FD_TOL)
        .expect("verify_fd must run on the RHF reference");
    assert!(
        report.passed,
        "RHF analytical gradient (p-shell) must agree with the central difference within \
         {FD_TOL} Ha/Bohr; got max|fd - analytical| = {:e}",
        report.max_abs_diff
    );
}

/// Per-atom element symbols (for rebuilding a geometry string at displaced
/// coords). Derived from `mol.atom_charges()` via a tiny Z→symbol table
/// sufficient for the H2 FD reference.
fn mol_atom_symbols(mol: &Mole) -> Vec<String> {
    mol.atom_charges()
        .into_iter()
        .map(|z| match z {
            1 => "H".to_string(),
            6 => "C".to_string(),
            7 => "N".to_string(),
            8 => "O".to_string(),
            other => format!("X{other}"),
        })
        .collect()
}

/// Format a `symbol x y z; ...` geometry string (Bohr) for `MoleBuildArgs`.
fn geometry_string(symbols: &[String], coords: &[[f64; 3]]) -> String {
    symbols
        .iter()
        .zip(coords.iter())
        .map(|(s, c)| format!("{s} {} {} {}", c[0], c[1], c[2]))
        .collect::<Vec<_>>()
        .join("; ")
}
