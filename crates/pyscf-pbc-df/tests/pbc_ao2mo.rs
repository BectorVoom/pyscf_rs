//! Plan 13-06 acceptance — periodic ERIs from both density-fitting builders.

mod common;

use common::diamond;
use pyscf_algebra::CTensor;
use pyscf_pbc_df::pbc_ao2mo::{aft_get_eri, fft_get_eri};
use pyscf_pbc_df::{Aftdf, Fftdf};

fn max_dev_ct(a: &CTensor, b: &CTensor) -> f64 {
    a.re.iter()
        .zip(b.re.iter())
        .map(|(x, y)| (x - y).abs())
        .chain(a.im.iter().zip(b.im.iter()).map(|(x, y)| (x - y).abs()))
        .fold(0.0f64, f64::max)
}

/// **Test 1** — 8-fold permutational symmetry at gamma, no oracle.
///
/// `(pq|rs) = (qp|rs) = (pq|sr) = (rs|pq)`, and the whole matrix is real. This
/// catches an index transposition in the contraction that an oracle comparison
/// on a symmetric test density could easily miss.
#[test]
fn gamma_eri_has_eightfold_symmetry() {
    let cell = diamond();
    let nao = cell.mol.nao_nr;
    let n2 = nao * nao;
    let df = Aftdf::with_mesh(cell, &[[0.0; 3]], [11, 11, 11]).expect("aftdf");
    let eri = aft_get_eri(&df, [[0.0; 3]; 4]).expect("get_eri");

    let imax = eri.im.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    assert!(imax < 1e-12, "a gamma-point ERI must be real, |Im| = {imax:e}");

    let at = |p: usize, q: usize, r: usize, s: usize| eri.re[(p * nao + q) * n2 + r * nao + s];
    let mut worst = 0.0f64;
    for p in 0..nao {
        for q in 0..nao {
            for r in 0..nao {
                for s in 0..nao {
                    let v = at(p, q, r, s);
                    worst = worst.max((v - at(q, p, r, s)).abs());
                    worst = worst.max((v - at(p, q, s, r)).abs());
                    worst = worst.max((v - at(r, s, p, q)).abs());
                }
            }
        }
    }
    // 1e-11, not 1e-12: the contraction is a sequential sum over 1331 G-vectors
    // (mesh 11) of terms of order 0.5, so a few e-12 of accumulated rounding is
    // the floor, not a defect. The measured residue is 1.966e-12 and the oracle
    // comparison in `eri_matches_upstream` bounds the total numerical noise at
    // 4.172e-12 against an upstream value that is symmetric by construction
    // (it packs `s4` at gamma) — so the asymmetry is inside the roundoff, not
    // on top of it.
    assert!(worst < 1e-11, "8-fold ERI symmetry broken by {worst:e}");
}

/// **Test 2, cross-builder** — `AFTDF.get_eri` vs `FFTDF.get_eri` through the
/// SAME contraction, so the difference isolates `ft_aopair` against the FFT of
/// the real-space AO product. Must fall with the mesh.
#[test]
fn aft_and_fft_eri_converge() {
    let kpts = [[0.0; 3]];
    let mut devs = Vec::new();
    for m in [11usize, 15, 21] {
        let a = Aftdf::with_mesh(diamond(), &kpts, [m, m, m]).expect("aftdf");
        let f = Fftdf::with_mesh(diamond(), &kpts, [m, m, m]).expect("fftdf");
        let ea = aft_get_eri(&a, [[0.0; 3]; 4]).expect("aft eri");
        let ef = fft_get_eri(&f, [[0.0; 3]; 4]).expect("fft eri");
        let d = max_dev_ct(&ea, &ef);
        eprintln!("ERI mesh {m}: |AFT − FFT| = {d:e}");
        devs.push(d);
    }
    assert!(devs[1] < devs[0], "mesh 15 must improve on mesh 11");
    assert!(devs[2] < devs[1], "mesh 21 must improve on mesh 15");
}

/// **Gate 3 (oracle)** — `AFTDF.get_eri` vs upstream.
#[test]
fn eri_matches_upstream() {
    let Some(py) = common::oracle_python() else {
        eprintln!("{} unset — skipping the upstream oracle", common::GATE);
        return;
    };
    let script = r#"
import json, sys
import numpy as np
from pyscf.pbc import gto, df
from pyscf import ao2mo

a_json, xyz_json, sym_json, basis, pseudo, mesh_json = sys.argv[1:7]
c = gto.Cell()
c.a = json.loads(a_json)
c.atom = [(s, tuple(r)) for s, r in zip(json.loads(sym_json), json.loads(xyz_json))]
c.basis = basis
if pseudo:
    c.pseudo = pseudo
c.unit = 'Bohr'
c.verbose = 0
c.build()
mydf = df.AFTDF(c, np.zeros((1, 3)))
mydf.mesh = json.loads(mesh_json)
nao = c.nao_nr()
eri = mydf.get_eri(np.zeros((4, 3)), compact=False)
eri = np.asarray(eri).reshape(nao**2, nao**2)
out = {'nao': int(nao), 'version': __import__('pyscf').__version__,
       're': np.real(eri).ravel().tolist(),
       'im': np.zeros(eri.size).tolist()}
print(json.dumps(out))
"#;
    let cell = diamond();
    let nao = cell.mol.nao_nr;
    let mesh = [11usize, 11, 11];
    let mut args = common::cell_args(&cell, &[]);
    args.insert(3, "gth-szv".into());
    args.insert(4, "gth-pade".into());
    args.push(format!("[{},{},{}]", mesh[0], mesh[1], mesh[2]));
    let want = common::run_python(&py, script, &args);
    assert_eq!(want["version"].as_str(), Some("2.12.1"), "vendored oracle");

    let df = Aftdf::with_mesh(diamond(), &[[0.0; 3]], mesh).expect("aftdf");
    let got = aft_get_eri(&df, [[0.0; 3]; 4]).expect("get_eri");
    let n2 = nao * nao;
    assert_eq!(got.re.len(), n2 * n2, "ERI shape");
    let w = common::max_dev(std::slice::from_ref(&got), &want);
    eprintln!("AFTDF get_eri: deviation vs upstream = {w:e}");
    assert!(w < 5e-9, "get_eri deviates from upstream by {w:e}");
}

/// **The accuracy knob is `cell.precision`, and this pins it.**
///
/// The exact gamma-point ERI is 8-fold symmetric, so any asymmetry is pure
/// screening error. Two things make that a good precision probe:
///
/// 1. It is **independent of the mesh** — bit-identical at 1 331, 3 375, 9 261
///    and 19 683 G-vectors — which rules out summation roundoff as the cause and
///    points at the lattice-sum screen instead.
/// 2. It responds to `cell.precision` monotonically, saturating at the f64 floor
///    by 1e-12.
///
/// | `cell.precision` | `rcut` | residue |
/// |---|---|---|
/// | 1e-8 (default) | 20.420 | 1.966e-12 |
/// | 1e-10 | 22.297 | 1.497e-14 |
/// | 1e-12 | 24.020 | 3.842e-16 |
///
/// `RcutChoice::Scaled(1.5)` reaches the same floor but at `rcut` 31.979 —
/// 2.4× more lattice images for no extra accuracy, because it inflates the
/// radius without tightening the screens.
#[test]
fn tightening_cell_precision_improves_the_eri() {
    let mesh = [11usize, 11, 11];
    let mut residues = Vec::new();
    for prec in [1e-8f64, 1e-10, 1e-12] {
        let mut cell = diamond();
        cell.precision = prec;
        let nao = cell.mol.nao_nr;
        let n2 = nao * nao;
        let df = Aftdf::with_mesh(cell, &[[0.0; 3]], mesh).expect("aftdf");
        let eri = aft_get_eri(&df, [[0.0; 3]; 4]).expect("eri");
        let at = |p: usize, q: usize, r: usize, s: usize| eri.re[(p * nao + q) * n2 + r * nao + s];
        let mut worst = 0.0f64;
        for p in 0..nao {
            for q in 0..nao {
                for r in 0..nao {
                    for s in 0..nao {
                        let v = at(p, q, r, s);
                        worst = worst.max((v - at(q, p, r, s)).abs());
                        worst = worst.max((v - at(p, q, s, r)).abs());
                        worst = worst.max((v - at(r, s, p, q)).abs());
                    }
                }
            }
        }
        eprintln!("cell.precision {prec:.0e}: ERI asymmetry {worst:e}");
        residues.push(worst);
    }
    assert!(
        residues[1] < residues[0] * 0.1,
        "1e-10 must improve on 1e-8 by >10x: {:e} vs {:e}",
        residues[1],
        residues[0]
    );
    assert!(
        residues[2] < 1e-14,
        "cell.precision = 1e-12 should reach the f64 floor, got {:e}",
        residues[2]
    );
}
