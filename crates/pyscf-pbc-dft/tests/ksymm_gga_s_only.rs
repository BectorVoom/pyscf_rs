//! 20-13 D4 regression: k-symmetric GGA on an s-only basis.
//!
//! The default symmetrized IBZ quadrature (S-03) rotates the GGA density
//! gradient with the `l = 1` Wigner-D matrices. `Symmetry` builds `Dmats` only
//! up to the basis' highest angular momentum (`symmetry.py:83-84`), so on an
//! s-only cell `KPoints::symmetrize_density_vec` indexed `dmats()[iop][1]` out
//! of bounds — a panic, and under the release profile's `panic = "abort"` a
//! killed process (measured through the Python binding on He sto-3g PBE).
//!
//! Kept in its own test binary: `ksymm_symmetrize_rho.rs` toggles
//! `PYSCF_PBC_KSYMM_RHO` process-wide, which would race the route this file
//! must exercise.

use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs, Unit};
use pyscf_pbc_dft::krks::Krks;
use pyscf_pbc_dft::krks_ksymm::{KsymAdaptedKrks, KsymAdaptedKuks};
use pyscf_pbc_dft::kuks::Kuks;
use pyscf_pbc_gto::make_kpts_default;
use pyscf_pbc_gto::types::{ALattice, CellBuildArgs};
use pyscf_pbc_scf::{KInitGuess, KScfConfig};
use pyscf_pbc_symm::basis::{self, SymmAdaptedBasisInput};
use pyscf_pbc_symm::kpts::make_kpts;

/// Gate C's KS bound (`krks_ksymm.rs::KRKS_E_TOL`), unchanged.
const E_TOL: f64 = 1e-9;

/// He fcc sto-3g (1 s AO), 2x2x2, PBE, at the cell's OWN mesh — 20-13 D5
/// measured that a pinned coarse mesh aliases the IBZ and full-BZ arms apart
/// (8.3e-7 at 15^3), so the default mesh is the valid Gate-C fixture.
#[test]
fn ksymm_gga_on_an_s_only_basis_runs_and_matches_the_full_bz() {
    assert!(
        !std::env::var("PYSCF_PBC_KSYMM_RHO").is_ok_and(|v| v.eq_ignore_ascii_case("unfold")),
        "this gate must run the symmetrized route"
    );
    let h = 2.834589;
    let mut cell = pyscf_pbc_gto::Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![("He".into(), [0.0, 0.0, 0.0])]),
            basis: BasisInput::Name("sto-3g".into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]),
        ..Default::default()
    })
    .expect("He cell");
    let kpts_abs = make_kpts_default(&cell, [2, 2, 2]).expect("make_kpts_default");
    let kp = make_kpts(&cell, &kpts_abs, true, false).expect("make_kpts");
    assert!(kp.nkpts_ibz() < kp.nkpts(), "the fixture must fold");
    // The precondition the defect needs: no l = 1 Wigner-D block exists.
    assert!(
        kp.dmats().iter().all(|d| d.len() == 1),
        "the fixture must be s-only (Dmats stop at l = 0)"
    );
    basis::build_symmetry(
        &mut cell,
        &SymmAdaptedBasisInput {
            kpts_scaled_ibz: kp.kpts_scaled_ibz.clone(),
            little_cogroup_ops: kp.little_cogroup_ops.clone(),
            ops: kp.symmetry.ops.clone(),
            dmats: kp.symmetry.dmats.clone(),
        },
    )
    .expect("build_symmetry");

    let cfg = KScfConfig {
        conv_tol: 1e-12,
        conv_tol_grad: Some(1e-8),
        max_cycle: 60,
        init_guess: KInitGuess::Minao,
        ..KScfConfig::default()
    };

    let krks_ibz = KsymAdaptedKrks::new(cell.clone(), kp.clone(), "pbe")
        .expect("KsymAdaptedKrks")
        .kernel(&cfg)
        .expect("IBZ KRKS PBE");
    let krks_bz = Krks::new(cell.clone(), &kp.kpts, "pbe")
        .expect("Krks")
        .kernel(&cfg)
        .expect("full-BZ KRKS PBE");
    let kuks_ibz = KsymAdaptedKuks::new(cell.clone(), kp.clone(), "pbe")
        .expect("KsymAdaptedKuks")
        .kernel(&cfg)
        .expect("IBZ KUKS PBE");
    let kuks_bz = Kuks::new(cell.clone(), &kp.kpts, "pbe")
        .expect("Kuks")
        .kernel(&cfg)
        .expect("full-BZ KUKS PBE");

    for (label, ibz, bz) in [("KRKS", &krks_ibz, &krks_bz), ("KUKS", &kuks_ibz, &kuks_bz)] {
        let de = (ibz.e_tot - bz.e_tot).abs();
        println!(
            "{label} He sto-3g PBE mesh {:?}: e_ibz {:.15}  e_bz {:.15}  |dE| = {de:e} ({} of {} k)",
            cell.mesh,
            ibz.e_tot,
            bz.e_tot,
            kp.nkpts_ibz(),
            kp.nkpts()
        );
        assert!(ibz.converged && bz.converged, "{label}: both arms converge");
        assert!(ibz.e_tot.is_finite() && ibz.e_tot < 0.0);
        assert!(de < E_TOL, "{label}: ksymm vs full BZ |dE| = {de:e}");
    }
}
