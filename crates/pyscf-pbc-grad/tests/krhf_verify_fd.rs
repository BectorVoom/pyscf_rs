//! Gate B for KRHF — analytic gradient vs this port's `verify_fd` (18-05).
//!
//! Diamond at gamma (`gth-szv`/`gth-pade`, 2 atoms, 8 AOs) with atom 1
//! displaced +0.02 Bohr in x so the gradient is O(10⁻²) Ha/Bohr and the gate
//! cannot pass vacuously. Thirteen KRHF runs (one central + twelve
//! displaced) at `conv_tol = 1e-10` — upstream's own `test_krhf.py` tightness
//! — with the harness half-step `5e-6` (= upstream's full `1e-5` disp).
//!
//! Tolerance is `FD_TOL = 1e-6` Ha/Bohr. If this fails, the measured number
//! is reported and the tolerance is NOT loosened (18-CONTEXT §5.6).
//!
//! # Geometry is specified in BOHR

use pyscf_pbc_grad::{KrhfGradients, verify_fd};
use pyscf_pbc_scf::{KScfConfig, Krhf};

const DISP: f64 = 5e-6;
const TOL: f64 = 1e-6;

fn tight_config(cell: &pyscf_pbc_gto::Cell) -> KScfConfig {
    KScfConfig {
        conv_tol: 1e-12,
        conv_tol_grad: Some(1e-10),
        max_cycle: 100,
        ..KScfConfig::for_cell(cell)
    }
}

fn krhf_energy(
    cell: &pyscf_pbc_gto::Cell,
    kpts: &[[f64; 3]],
) -> Result<f64, pyscf_core::PyscfRsError> {
    let mf = Krhf::new(cell.clone(), kpts).expect("KRHF builds on a displaced cell");
    let cfg = tight_config(cell);
    let result = mf.kernel(&cfg)?;
    assert!(
        result.converged,
        "KRHF did not converge after {} cycles — the FD number is meaningless",
        result.cycles
    );
    Ok(result.e_tot)
}

/// Gate B: `max|verify_fd − analytic| <= 1e-6`, with the measured residual
/// printed on every run.
#[test]
fn krhf_gamma_gradient_passes_verify_fd() {
    // Displaced diamond in Bohr (scf `common::diamond` geometry: fcc
    // a0 = 6.74064, second C at (q,q,q), q = 1.68516), atom 1 pushed +0.02
    // Bohr in x so the gradient is O(10⁻²) and the gate cannot pass
    // vacuously. Built, never mutated: `CellBuildArgs`, not field surgery.
    use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
    use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs};
    let h = 3.37032;
    let q = 1.68516;
    let cell = Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![
                ("C".into(), [0.0, 0.0, 0.0]),
                ("C".into(), [q + 0.02, q, q]),
            ]),
            basis: BasisInput::Name("gth-szv".into()),
            unit: pyscf_core::Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]),
        pseudo: Some("gth-pade".into()),
        // Pinned mesh, exactly like upstream's own test_krhf.py
        // (`cell.mesh = [13] * 3`): Gate B is self-consistency of the
        // analytic gradient against this port's FD of this port's energy,
        // valid at any mesh; the default precision-derived mesh (47³ on
        // diamond) would make the thirteen SCF runs gratuitously slow.
        mesh: Some([13, 13, 13]),
        ..Default::default()
    })
    .expect("displaced diamond must build");

    let kpts = [[0.0_f64; 3]];
    let analytic = {
        let mf = Krhf::new(cell.clone(), &kpts).expect("central KRHF");
        let cfg = tight_config(&cell);
        let result = mf.kernel(&cfg).expect("central SCF");
        assert!(result.converged, "central SCF did not converge");
        KrhfGradients::new(&mf, result.mo_energy, result.mo_coeff, result.mo_occ)
            .expect("gradient object")
            .kernel()
            .expect("analytic gradient")
    };
    let peak = analytic
        .iter()
        .flat_map(|r| r.iter())
        .fold(0.0_f64, |a, v| a.max(v.abs()));
    assert!(
        peak > 1e-4,
        "gate is vacuous: max|analytic| = {peak:e} on the displaced cell"
    );

    let report = verify_fd(&cell, &analytic, |c| krhf_energy(c, &kpts), DISP, TOL)
        .expect("finite-difference harness");
    println!(
        "Gate B (KRHF gamma diamond): max|fd − analytic| = {:.3e} Ha/Bohr",
        report.max_abs_diff
    );
    assert!(
        report.passed,
        "Gate B FAILED: max|fd − analytic| = {:.3e} > {TOL:e}\nanalytic = {analytic:?}\nfd = {:?}",
        report.max_abs_diff, report.fd_grad,
    );
}
