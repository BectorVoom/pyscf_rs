//! Plan 20-19 item D — the SCF overlap against live upstream PySCF **2.12.1**
//! at the ELEMENT level, on `examples/pbc/23-smearing.py`'s Al cell.
//!
//! Upstream's `KSCF.get_ovlp` (`khf.py:52-63`) is `pbc/scf/hf.py:get_ovlp`,
//! which integrates the lattice sum at `cell.precision * 1e-5` with a widened
//! `rcut` and `hermi=0` (`hf.py:47-55`). The Al Γ overlap is near-singular
//! (λ_min ≈ 3e-9), so the plain-precision `pbc_intor` differs from it by
//! 2.0e-9 — which moved a fractionally occupied band and cost 5.0e-3 Ha of
//! σ = 0.1 free energy (20-18-PRE-SUMMARY).
//!
//! The gate is on the driver HOOK (`KOverrideHooks::get_ovlp`, what `kscf`
//! consumes) for KRHF and KUHF, and on `get_bands`' explicit-k overlap route
//! (`get_ovlp_scf` at an arbitrary k). Non-vacuity: the plain
//! [`pyscf_pbc_gto::get_ovlp`] must MISS the same gate by orders of magnitude,
//! which is the pre-fix state of every driver (the RED of this plan).
//!
//! ```bash
//! PYSCF_ORACLE_VENV=1 cargo test -p pyscf-pbc-scf --release \
//!     --test scf_ovlp_oracle -- --ignored --nocapture
//! ```

mod common;

use common::{GATE, cell_args, max_dev, oracle_python, run_python};
use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_df::Fftdf;
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs, make_kpts_default};
use pyscf_pbc_scf::{KOverrideHooks, Krhf, Kuhf};

/// `23-smearing.py`'s k-mesh.
const NK: [usize; 3] = [4, 4, 4];
/// The overlap never touches the DF mesh; keep the FFTDF object cheap.
const MESH: [usize; 3] = [9, 9, 9];
/// Element-level gate. `pbc_intor('int1e_ovlp')` agrees between the codes to
/// 7e-14 at every precision (20-18-PRE-SUMMARY item 7).
const TOL: f64 = 1e-12;
/// The plain-precision overlap must miss upstream's by at least this much, or
/// the fixture does not exercise the precision bump.
const PLAIN_MIN_DEV: f64 = 1e-10;

/// 2.02 Å in Bohr at upstream's CODATA-2010 `BOHR = 0.52917721092`.
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
import json, sys, inspect
import numpy as np
import scipy.linalg
from pyscf.pbc import gto, scf
from pyscf.pbc.scf import hf as pbchf

a_json, xyz_json, sym_json, nk_json, kband_json = sys.argv[1:6]
c = gto.Cell()
c.a = json.loads(a_json)
c.atom = [(s, tuple(r)) for s, r in zip(json.loads(sym_json), json.loads(xyz_json))]
c.basis = 'gth-dzvp'
c.pseudo = 'gth-pbe'
c.unit = 'Bohr'
c.verbose = 0
c.build()
kpts = c.make_kpts(json.loads(nk_json))

def flat(s):
    s = np.asarray(s, dtype=np.complex128)
    return {'re': s.real.ravel().tolist(), 'im': s.imag.ravel().tolist()}

s_krhf = np.asarray(scf.KRHF(c, kpts).get_ovlp())
s_kuhf = np.asarray(scf.KUHF(c, kpts).get_ovlp())
kband = np.asarray(json.loads(kband_json), dtype=float)
s_band = np.asarray(scf.KRHF(c, kpts).get_ovlp(c, kband))
lmin_gamma = float(scipy.linalg.eigvalsh(s_krhf[0]).min())

print(json.dumps({
    'version': __import__('pyscf').__version__,
    'rule_pinned': 'cell.precision * 1e-5' in inspect.getsource(pbchf.get_ovlp),
    'nao': int(c.nao_nr()),
    'krhf': flat(s_krhf),
    'kuhf': flat(s_kuhf),
    'band': flat(s_band),
    'lmin_gamma': lmin_gamma,
}))
"#;

/// A few off-mesh band k-points, absolute 1/Bohr, computed on the Rust side.
fn band_kpts(cell: &Cell) -> Vec<[f64; 3]> {
    let b = cell.reciprocal_vectors_2pi().expect("reciprocal vectors");
    [[0.1, 0.2, 0.3], [0.5, 0.0, 0.5], [0.0, 0.0, 0.0]]
        .iter()
        .map(|f| {
            let mut k = [0.0; 3];
            for (i, fi) in f.iter().enumerate() {
                for (x, kx) in k.iter_mut().enumerate() {
                    *kx += fi * b[i][x];
                }
            }
            k
        })
        .collect()
}

#[test]
#[ignore = "oracle: set PYSCF_ORACLE_VENV (upstream PySCF 2.12.1)"]
fn scf_overlap_matches_upstream_kscf_get_ovlp_on_al_smearing_cell() {
    let Some(py) = oracle_python() else {
        eprintln!("SKIP: {GATE} unset — skipping");
        return;
    };
    let cell = al_cell();
    let nao = cell.mol.nao_nr;
    let kpts = make_kpts_default(&cell, NK).expect("k-mesh");
    let kband = band_kpts(&cell);

    let args = cell_args(
        &cell,
        &[
            serde_json::to_string(&NK.to_vec()).expect("json"),
            serde_json::to_string(&kband.iter().map(|k| k.to_vec()).collect::<Vec<_>>())
                .expect("json"),
        ],
    );
    let want = run_python(&py, ORACLE_PY, &args);
    assert_eq!(
        want["version"].as_str(),
        Some("2.12.1"),
        "oracle must be vendored 2.12.1"
    );
    assert_eq!(
        want["rule_pinned"].as_bool(),
        Some(true),
        "upstream hf.get_ovlp no longer carries the precision*1e-5 rule this port mirrors"
    );
    assert_eq!(want["nao"].as_u64(), Some(nao as u64), "nao");

    let krhf = Krhf::from_df(Box::new(
        Fftdf::with_mesh(cell.clone(), &kpts, MESH).expect("FFTDF"),
    ));
    let kuhf = Kuhf::from_df(Box::new(
        Fftdf::with_mesh(cell.clone(), &kpts, MESH).expect("FFTDF"),
    ));

    let s_krhf = krhf.get_ovlp().expect("KRHF get_ovlp");
    let s_kuhf = kuhf.get_ovlp().expect("KUHF get_ovlp");
    let s_band = pyscf_pbc_scf::krhf::to_row_major(
        pyscf_pbc_gto::get_ovlp_scf(&cell, &kband).expect("get_ovlp_scf"),
        nao,
    );
    let s_plain = pyscf_pbc_scf::krhf::to_row_major(
        pyscf_pbc_gto::get_ovlp(&cell, &kpts).expect("get_ovlp"),
        nao,
    );

    let d_krhf = max_dev(&s_krhf, &want["krhf"]);
    let d_kuhf = max_dev(&s_kuhf, &want["kuhf"]);
    let d_band = max_dev(&s_band, &want["band"]);
    let d_plain = max_dev(&s_plain, &want["krhf"]);
    println!(
        "Al gth-dzvp 4x4x4 (Γ λ_min upstream {:.3e}): KRHF hook {d_krhf:.3e}, KUHF hook \
         {d_kuhf:.3e}, band-k {d_band:.3e}  (tol {TOL:.0e}); plain cell.precision \
         pbc_intor {d_plain:.3e} (must exceed {PLAIN_MIN_DEV:.0e})",
        want["lmin_gamma"].as_f64().unwrap_or(f64::NAN)
    );
    assert!(
        d_plain > PLAIN_MIN_DEV,
        "vacuous fixture: the plain overlap already matches upstream to {d_plain:e}"
    );
    assert!(d_krhf < TOL, "KRHF get_ovlp vs upstream: {d_krhf:e}");
    assert!(d_kuhf < TOL, "KUHF get_ovlp vs upstream: {d_kuhf:e}");
    assert!(d_band < TOL, "band-k SCF overlap vs upstream: {d_band:e}");
}
