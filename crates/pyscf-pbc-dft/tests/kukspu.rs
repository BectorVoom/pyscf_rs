//! `KUKSpU` — the unrestricted Hubbard term (`pyscf/pbc/dft/kukspu.py`).
//!
//! 20-13 D3 found `kspu::add_vhubbard(_weighted)` applying the RESTRICTED
//! expressions `E_U = w (U/2)(Tr P - Tr P^2 / 2)`, `V = (1 - P) U/2`
//! (`krkspu.py:109-111`) to each channel of a two-channel density, where
//! upstream `kukspu.py:98-99` uses `Tr P - Tr P^2` and `(1 - 2P) U/2` per spin.
//! Measured on He sto-3g 2x2x2, U = 5 eV: converged `E_U` 9.187e-02 against
//! upstream 5.6e-14.
//!
//! ```bash
//! PYSCF_ORACLE_VENV=1 cargo test -p pyscf-pbc-dft --release --test kukspu -- --include-ignored --nocapture
//! ```

mod common;

use common::{GATE, he_all_electron, oracle_python, run_python};
use pyscf_algebra::CTensor;
use pyscf_core::PyscfRsError;
use pyscf_pbc_df::Fftdf;
use pyscf_pbc_dft::gen_grid::PeriodicGrids;
use pyscf_pbc_dft::kspu::{HubbardU, USite, add_vhubbard};
use pyscf_pbc_dft::kuks::Kuks;
use pyscf_pbc_gto::{Cell, make_kpts_default};
use pyscf_pbc_scf::types::{KDms, KMats};
use pyscf_pbc_scf::{KInitGuess, KOverrideHooks, KScfConfig};
use std::cell::Cell as StdCell;

/// 20-13's fixture mesh (`test_pbc_dft.py::MESH_HE`), pinned on both sides.
const MESH_HE: [usize; 3] = [15, 15, 15];

fn he_1s_u5() -> HubbardU {
    HubbardU {
        sites: vec![USite::Shell {
            element: "He".into(),
            l: 0,
            contraction: Some(0),
        }],
        u_val: vec![5.0], // eV
        ..HubbardU::default()
    }
}

/// A Hermitian, k-dependent, FRACTIONALLY occupied density (so `Tr P != Tr P^2`).
fn hermitian_dm(nk: usize, nao: usize, diag: f64, seed: usize) -> KMats {
    (0..nk)
        .map(|k| {
            let mut m = CTensor::zeros(nao * nao);
            for i in 0..nao {
                for j in 0..i {
                    let re = (((k + seed) * 31 + i * 7 + j * 13) as f64).sin() * 0.05;
                    let im = (((k + seed + 1) * 17 + i * 11 + j * 5) as f64).cos() * 0.02;
                    m.re[i * nao + j] = re;
                    m.re[j * nao + i] = re;
                    m.im[i * nao + j] = im;
                    m.im[j * nao + i] = -im;
                }
                m.re[i * nao + i] = diag + 0.01 * ((k + i) as f64).cos();
            }
            m
        })
        .collect()
}

fn scaled(dm: &KMats, s: f64) -> KMats {
    dm.iter()
        .map(|m| {
            CTensor::from_planes(
                m.re.iter().map(|x| x * s).collect(),
                m.im.iter().map(|x| x * s).collect(),
            )
        })
        .collect()
}

/// **Oracle-free.** For a closed-shell density `D = 2 d`, the unrestricted
/// expression summed over the two spins equals the restricted one:
/// `2 (U/2)(Tr p - Tr p^2) = (U/2)(Tr 2p - Tr (2p)^2 / 2)`, and the per-spin
/// potential `(1 - 2p) U/2` equals the restricted `(1 - D) U/2`. The old
/// per-channel restricted formula gives `2 (U/2)(Tr p - Tr p^2 / 2)` instead,
/// which differs whenever `Tr p^2 != 0`.
#[test]
fn kukspu_e_u_and_potential_equal_krkspu_on_a_closed_shell_density() {
    for basis in ["sto-3g", "6-31g"] {
        let mut cell = he_all_electron();
        if basis != "sto-3g" {
            cell = common::spin_cell(
                cell.a,
                vec![("He".into(), [0.0, 0.0, 0.0])],
                basis,
                None,
                0,
                0,
            );
        }
        let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("k-mesh");
        let nk = kpts.len();
        let nao = cell.mol.nao_nr;
        let cfg = he_1s_u5();

        let d_total = hermitian_dm(nk, nao, 0.7, 3);
        let d_spin = scaled(&d_total, 0.5);

        let mut v_r: Vec<KMats> = vec![vec![CTensor::zeros(nao * nao); nk]];
        let e_r = add_vhubbard(&mut v_r, &cell, &kpts, &vec![d_total], &cfg).expect("KRKSpU E_U");
        let mut v_u: Vec<KMats> = vec![vec![CTensor::zeros(nao * nao); nk]; 2];
        let e_u = add_vhubbard(&mut v_u, &cell, &kpts, &vec![d_spin.clone(), d_spin], &cfg)
            .expect("KUKSpU E_U");

        let de = (e_r - e_u).abs();
        let mut dv = 0.0_f64;
        for s in 0..2 {
            for k in 0..nk {
                for i in 0..nao * nao {
                    dv = dv
                        .max((v_u[s][k].re[i] - v_r[0][k].re[i]).abs())
                        .max((v_u[s][k].im[i] - v_r[0][k].im[i]).abs());
                }
            }
        }
        println!(
            "{basis}: E_U restricted {e_r:.15}  unrestricted {e_u:.15}  |dE_U| = {de:e}  max|dV| = {dv:e}"
        );
        assert!(
            e_r.abs() > 1e-4,
            "{basis}: the Hubbard term must be active (E_U = {e_r:e})"
        );
        assert!(
            de < 1e-14,
            "{basis}: closed-shell KUKSpU E_U differs from KRKSpU by {de:e}"
        );
        assert!(
            dv < 1e-14,
            "{basis}: closed-shell KUKSpU V_U differs from KRKSpU by {dv:e}"
        );
    }
}

// ---------------------------------------------------------------------------
// The upstream oracle
// ---------------------------------------------------------------------------

/// `KUKSpU` over a full-BZ [`Kuks`] — upstream's class is `kuks.KUKS` with
/// `get_veff`/`energy_elec` replaced (`kukspu.py:143-146`). The Rust
/// `Kukspu` carries no `KOverrideHooks`, so this test-local driver makes the
/// same two calls `Kukspu::get_veff_tagged` does, and adds `E_U` in
/// `energy_elec` (`kukspu.py:127`: `e1 + ecoul + exc + E_U`).
struct KukspuScf {
    ks: Kuks,
    u: HubbardU,
    e_u: StdCell<f64>,
}

impl KOverrideHooks for KukspuScf {
    fn cell(&self) -> &Cell {
        self.ks.cell()
    }
    fn kpts(&self) -> &[[f64; 3]] {
        self.ks.kpts()
    }
    fn nset(&self) -> usize {
        self.ks.nset()
    }
    fn nfock(&self) -> usize {
        self.ks.nfock()
    }
    fn nao(&self) -> usize {
        self.ks.nao()
    }
    fn get_ovlp(&self) -> Result<KMats, PyscfRsError> {
        self.ks.get_ovlp()
    }
    fn get_hcore(&self) -> Result<KMats, PyscfRsError> {
        self.ks.get_hcore()
    }
    fn get_init_guess(&self, mode: &KInitGuess, s1e: &KMats) -> Result<KDms, PyscfRsError> {
        self.ks.get_init_guess(mode, s1e)
    }
    fn get_veff(&self, dms: &KDms) -> Result<KDms, PyscfRsError> {
        let mut v = self.ks.get_veff(dms)?;
        let e_u =
            add_vhubbard(&mut v, self.ks.cell(), self.ks.kpts(), dms, &self.u).map_err(|e| {
                PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(format!(
                    "add_vhubbard: {e}"
                )))
            })?;
        self.e_u.set(e_u);
        Ok(v)
    }
    fn get_fock(&self, h1e: &KMats, vhf: &KDms, dms: &KDms) -> Result<KDms, PyscfRsError> {
        self.ks.get_fock(h1e, vhf, dms)
    }
    fn diis_dms(&self, dms: &KDms) -> KDms {
        self.ks.diis_dms(dms)
    }
    fn eig(&self, fock: &KDms, s1e: &KMats) -> Result<(Vec<Vec<f64>>, Vec<CTensor>), PyscfRsError> {
        self.ks.eig(fock, s1e)
    }
    fn get_occ(&self, mo_energy: &[Vec<f64>]) -> Result<(Vec<Vec<f64>>, Vec<f64>), PyscfRsError> {
        self.ks.get_occ(mo_energy)
    }
    fn make_rdm1(&self, mo_coeff: &[CTensor], mo_occ: &[Vec<f64>]) -> Result<KDms, PyscfRsError> {
        self.ks.make_rdm1(mo_coeff, mo_occ)
    }
    fn energy_elec(&self, dms: &KDms, h1e: &KMats, vhf: &KDms) -> Result<(f64, f64), PyscfRsError> {
        let (e, e2) = self.ks.energy_elec(dms, h1e, vhf)?;
        let e_u = self.e_u.get();
        Ok((e + e_u, e2 + e_u))
    }
    fn energy_nuc(&self) -> Result<f64, PyscfRsError> {
        self.ks.energy_nuc()
    }
    fn get_grad(
        &self,
        mo_coeff: &[CTensor],
        mo_occ: &[Vec<f64>],
        h1e: &KMats,
        vhf: &KDms,
    ) -> Vec<f64> {
        self.ks.get_grad(mo_coeff, mo_occ, h1e, vhf)
    }
    fn free_energy(&self) -> Option<f64> {
        self.ks.free_energy()
    }
}

const ORACLE_PY: &str = r#"
import json, sys
import numpy as np
from pyscf.pbc import gto
from pyscf.pbc.dft import kukspu

a, mesh = json.loads(sys.argv[1])
he = gto.Cell()
he.a = a
he.atom = [('He', (0.0, 0.0, 0.0))]
he.basis = 'sto-3g'
he.unit = 'Bohr'
he.verbose = 0
he.build()
mf = kukspu.KUKSpU(he, he.make_kpts([2, 2, 2]), xc='lda,vwn', U_idx=['He 1s'], U_val=[5.0])
mf.with_df.mesh = mesh
mf.grids.mesh = mesh
mf.init_guess_breaksym = 1
mf.conv_tol = 1e-12
mf.conv_tol_grad = 1e-8
mf.max_cycle = 60
e = mf.kernel()
e_u_conv = float(mf.get_veff(he, mf.make_rdm1()).E_U.real)
frac = np.array([[np.eye(1) * 0.35] * 8] * 2, dtype=complex)
e_u_frac = float(mf.get_veff(he, frac).E_U.real)
pol = np.array([[np.eye(1) * 0.5] * 8, [np.eye(1) * 0.2] * 8], dtype=complex)
e_u_pol = float(mf.get_veff(he, pol).E_U.real)
print(json.dumps({'version': __import__('pyscf').__version__,
                  'xclib': mf._numint.libxc.__name__,
                  'e_tot': float(e), 'converged': bool(mf.converged),
                  'e_u_conv': e_u_conv, 'e_u_frac': e_u_frac, 'e_u_pol': e_u_pol}))
"#;

/// A `(2, nk)` constant-diagonal density for the 1-AO He cell.
fn diag_dm(nk: usize, a: f64, b: f64) -> KDms {
    let one = |x: f64| {
        let mut t = CTensor::zeros(1);
        t.re[0] = x;
        vec![t; nk]
    };
    vec![one(a), one(b)]
}

/// **Gate: KUKSpU He sto-3g 2x2x2, mesh 15, U = 5 eV on `He 1s`, against
/// vendored 2.12.1.** `e_tot` at the all-electron KRKS gate (1e-12, the bound
/// `test_pbc_dft.py::test_krkspu_matches_upstream` holds KRKSpU to). `E_U` at
/// the converged density and at two fractional densities (one spin-balanced,
/// one polarised) is bounded at 1e-9 — 20-13 D7's KRKSpU bound, set by the
/// MINAO local-orbital construction (`zsolve_linear` + `zeigh_gen` here,
/// `cho_solve` + `vec_lowdin` upstream), not a floor.
#[test]
#[ignore = "T1: needs PYSCF_ORACLE_VENV + the vendored upstream PySCF"]
fn kukspu_he_matches_upstream() {
    let cell = he_all_electron();
    let Some(py) = oracle_python() else {
        eprintln!("SKIP: {GATE} is not set");
        return;
    };
    let a: Vec<Vec<f64>> = cell.a.iter().map(|r| r.to_vec()).collect();
    let want = run_python(
        &py,
        ORACLE_PY,
        &[serde_json::to_string(&(a, MESH_HE.to_vec())).expect("json")],
    );
    assert_eq!(want["version"].as_str().expect("version"), "2.12.1");
    assert!(
        want["xclib"].as_str().expect("xclib").ends_with("libxc"),
        "upstream must run its libxc default"
    );
    assert!(want["converged"].as_bool().expect("converged"));

    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("k-mesh");
    let nk = kpts.len();
    let df = Fftdf::with_mesh(cell, &kpts, MESH_HE).expect("FFTDF");
    let mut ks = Kuks::from_df(Box::new(df), "lda,vwn").expect("KUKS");
    ks.grids = PeriodicGrids::uniform(ks.cell(), Some(MESH_HE)).expect("XC grid");
    let mf = KukspuScf {
        ks,
        u: he_1s_u5(),
        e_u: StdCell::new(0.0),
    };
    let cfg = KScfConfig {
        conv_tol: 1e-12,
        conv_tol_grad: Some(1e-8),
        max_cycle: 60,
        ..KScfConfig::default()
    };
    let res = pyscf_pbc_scf::kernel(&mf, &cfg).expect("KUKSpU kernel");
    assert!(res.converged, "KUKSpU did not converge");

    let e_u_at = |dms: &KDms| {
        mf.get_veff(dms).expect("get_veff");
        mf.e_u.get()
    };
    let rows = [
        (
            "e_tot",
            res.e_tot,
            want["e_tot"].as_f64().expect("e_tot"),
            1e-12,
        ),
        (
            "E_U converged",
            e_u_at(&res.dm),
            want["e_u_conv"].as_f64().expect("e_u_conv"),
            1e-9,
        ),
        (
            "E_U 0.35/0.35",
            e_u_at(&diag_dm(nk, 0.35, 0.35)),
            want["e_u_frac"].as_f64().expect("e_u_frac"),
            1e-9,
        ),
        (
            "E_U 0.50/0.20",
            e_u_at(&diag_dm(nk, 0.5, 0.2)),
            want["e_u_pol"].as_f64().expect("e_u_pol"),
            1e-9,
        ),
    ];
    for (label, got, up, tol) in rows {
        let d = (got - up).abs();
        println!("{label:<14} rust {got:.15e}  upstream {up:.15e}  |d| = {d:.3e}  (tol {tol:.0e})");
    }
    for (label, got, up, tol) in rows {
        let d = (got - up).abs();
        assert!(d < tol, "{label}: |rust - upstream| = {d:e} >= {tol:e}");
    }
}
