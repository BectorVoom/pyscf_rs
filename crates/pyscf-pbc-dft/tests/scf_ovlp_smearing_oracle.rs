//! Plan 20-19 item D — `examples/pbc/23-smearing.py`'s σ = 0.1 Fermi-smeared
//! KRKS/PBE on Al (gth-dzvp / gth-pbe, 4x4x4) against live upstream PySCF
//! **2.12.1**.
//!
//! Before item D the port integrated the SCF overlap at plain `cell.precision`
//! while upstream's `pbc/scf/hf.py:get_ovlp` uses `cell.precision * 1e-5`. The
//! Γ overlap here is near-singular (λ_min ≈ 3e-9); the 2.0e-9 overlap error
//! moved the ill-conditioned Γ p band from 0.474 to 0.278 Ha, its σ = 0.1
//! occupation from 0.20 to 0.77, and the free energy by **5.04e-3 Ha**
//! (20-18-PRE-SUMMARY, same cell at `conv_tol` 1e-7). The element-level
//! overlap gate is `crates/pyscf-pbc-scf/tests/scf_ovlp_oracle.rs`.
//!
//! The lattice is in BOHR (2.02 Å at CODATA-2010) so the CODATA gap of an
//! Angstrom cell does not enter. Both sides use the cell's default mesh for the
//! FFTDF and the XC grid, as the example does; the mesh is asserted equal.
//!
//! ```bash
//! PYSCF_ORACLE_VENV=1 cargo test -p pyscf-pbc-dft --release \
//!     --test scf_ovlp_smearing_oracle -- --ignored --nocapture
//! ```

mod common;

use common::{GATE, cell_args, oracle_python, run_python};
use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_dft::krks::Krks;
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs, make_kpts_default};
use pyscf_pbc_scf::{KScfConfig, Smearing};

const NK: [usize; 3] = [4, 4, 4];
const SIGMA: f64 = 0.1;
/// Energy gate, fixed BEFORE the first post-fix measurement: ~10x the
/// no-smearing agreement 20-18 measured on this cell (8.7e-9 at Γ), since the
/// overlap's condition number (≈ 3e8) amplifies SCF-level noise; still 4.7
/// orders below the pre-fix σ = 0.1 error (5.04e-3).
const TOL: f64 = 1e-7;

fn al_cell() -> Cell {
    let h = 2.02 / 0.529_177_210_92;
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![("Al".into(), [0.0, 0.0, 0.0])]),
            basis: BasisInput::Name("gth-dzvp".into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]),
        pseudo: Some("gth-pbe".into()),
        ..Default::default()
    })
    .expect("Al cell must build")
}

const ORACLE_PY: &str = r#"
import json, sys
import numpy as np
from pyscf.pbc import gto, dft

a_json, xyz_json, sym_json, spin, charge, nk_json, sigma = sys.argv[1:8]
c = gto.Cell()
c.a = json.loads(a_json)
c.atom = [(s, tuple(r)) for s, r in zip(json.loads(sym_json), json.loads(xyz_json))]
c.basis = 'gth-dzvp'
c.pseudo = 'gth-pbe'
c.unit = 'Bohr'
c.spin = int(spin)
c.charge = int(charge)
c.verbose = 0
c.build()
mf = dft.KRKS(c, c.make_kpts(json.loads(nk_json)), xc='pbe')
from pyscf.dft import libxc
mf._numint.libxc = libxc
mf = mf.smearing(sigma=float(sigma), method='fermi')
mf.conv_tol = 1e-10
mf.conv_tol_grad = 1e-6
mf.max_cycle = 100
e = mf.kernel()
print(json.dumps({
    'version': __import__('pyscf').__version__,
    'converged': bool(mf.converged),
    'mesh': [int(x) for x in c.mesh],
    'e_nuc': float(c.energy_nuc()),
    'e_tot': float(e),
    'e_free': float(mf.e_free),
    'entropy': float(mf.entropy),
    'mo_occ_gamma': np.asarray(mf.mo_occ[0]).tolist(),
    'mo_energy_gamma': np.asarray(mf.mo_energy[0]).tolist(),
}))
"#;

#[test]
#[ignore = "oracle: set PYSCF_ORACLE_VENV (upstream PySCF 2.12.1)"]
fn al_smearing_sigma_0p1_free_energy_matches_upstream() {
    let Some(py) = oracle_python() else {
        eprintln!("SKIP: {GATE} unset — skipping");
        return;
    };
    let cell = al_cell();
    let args = cell_args(
        &cell,
        &[
            serde_json::to_string(&NK.to_vec()).expect("json"),
            SIGMA.to_string(),
        ],
    );
    let want = run_python(&py, ORACLE_PY, &args);
    assert_eq!(
        want["version"].as_str(),
        Some("2.12.1"),
        "oracle must be vendored 2.12.1"
    );
    assert_eq!(
        want["converged"].as_bool(),
        Some(true),
        "upstream did not converge"
    );
    let mesh: Vec<usize> = want["mesh"]
        .as_array()
        .expect("mesh")
        .iter()
        .map(|v| v.as_u64().expect("mesh int") as usize)
        .collect();
    assert_eq!(
        mesh,
        cell.try_mesh().expect("mesh").to_vec(),
        "cell.mesh differs"
    );

    let kpts = make_kpts_default(&cell, NK).expect("k-mesh");
    let mut mf = Krks::new(cell, &kpts, "pbe").expect("KRKS");
    mf.smearing = Some(Smearing::fermi(SIGMA));
    let got = mf
        .kernel(&KScfConfig {
            conv_tol: 1e-10,
            conv_tol_grad: Some(1e-6),
            max_cycle: 100,
            ..KScfConfig::default()
        })
        .expect("smeared KRKS");
    assert!(
        got.converged,
        "port did not converge in {} cycles",
        got.cycles
    );

    let e_nuc = want["e_nuc"].as_f64().expect("e_nuc");
    assert!(
        (got.e_nuc - e_nuc).abs() < 1e-12,
        "e_nuc differs — not the same cell"
    );

    let e_tot = want["e_tot"].as_f64().expect("e_tot");
    let e_free = want["e_free"].as_f64().expect("e_free");
    let entropy = want["entropy"].as_f64().expect("entropy");
    let got_free = got.e_free.expect("smearing must report e_free");
    let got_entropy = (got.e_tot - got_free) / SIGMA;
    let occ_w: Vec<f64> = want["mo_occ_gamma"]
        .as_array()
        .expect("occ")
        .iter()
        .map(|v| v.as_f64().expect("f64"))
        .collect();
    let occ_dev = got.mo_occ[0]
        .iter()
        .zip(&occ_w)
        .fold(0.0_f64, |a, (g, w)| a.max((g - w).abs()));
    let de_tot = (got.e_tot - e_tot).abs();
    let de_free = (got_free - e_free).abs();
    let ds = (got_entropy - entropy).abs();
    println!(
        "Al 4x4x4 PBE fermi σ={SIGMA}: e_tot rust {:.12} upstream {e_tot:.12} |Δ| {de_tot:.3e}; \
         e_free rust {got_free:.12} upstream {e_free:.12} |Δ| {de_free:.3e}; entropy |Δ| {ds:.3e}; \
         Γ mo_occ max|Δ| {occ_dev:.3e}; cycles {}  (tol {TOL:.0e})",
        got.e_tot, got.cycles
    );
    assert!(de_tot < TOL, "e_tot |Δ| {de_tot:e} exceeds {TOL:e}");
    assert!(de_free < TOL, "e_free |Δ| {de_free:e} exceeds {TOL:e}");
}
