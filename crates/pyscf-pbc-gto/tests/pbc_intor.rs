//! Plan 10-03 — `pbc_intor`, the periodic 1-electron lattice-sum driver.
//!
//! # The Phase-10 gate
//!
//! [`ovlp_matches_upstream_on_diamond_222`] compares
//! `pbc_intor("int1e_ovlp", make_kpts([2,2,2]))` element-by-element against
//! live upstream PySCF 2.12.1 at **1e-12**. It is `#[ignore]`d and additionally
//! short-circuits unless `PYSCF_ORACLE_VENV` is set, exactly as
//! `oracle_phase9.rs` does, so `cargo test --workspace` never needs Python:
//!
//! ```bash
//! PYSCF_ORACLE_VENV=1 cargo test -p pyscf-pbc-gto --test pbc_intor --release -- --ignored
//! ```
//!
//! Everything else in this file is oracle-free (D-PBC-19) — Hermiticity,
//! positive-definiteness, Bloch consistency, `rcut` convergence, and the
//! screened-vs-unscreened agreement.
//!
//! # Geometry is specified in BOHR
//!
//! `pyscf_core::Unit::Ang` is CODATA-2014 and upstream is CODATA-2010, so an
//! Angstrom cell would differ in the 8th digit of every lattice vector before a
//! single integral is evaluated. See `oracle_phase9.rs` for the same note.

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_gto::{
    ALattice, Cell, CellBuildArgs, PbcIntorOpts, PbcIntorOutput, kpts_mesh::make_kpts_default,
    lattice::get_lattice_ls, pbc_intor,
};
use std::path::PathBuf;
use std::process::Command;

// ---------------------------------------------------------------------------
// Reference systems, in Bohr
// ---------------------------------------------------------------------------

/// Diamond, fcc `a0 = 6.74064` Bohr (= 3.5668 Angstrom at CODATA-2010),
/// `gth-szv` / `gth-pade`. 8 AOs, 4 shells.
fn diamond() -> Cell {
    let h = 3.37032;
    let q = 1.68516;
    bohr_cell(
        [[0.0, h, h], [h, 0.0, h], [h, h, 0.0]],
        vec![("C".into(), [0.0, 0.0, 0.0]), ("C".into(), [q, q, q])],
    )
}

/// He on an fcc lattice — one atom, one shell; the cheapest possible lattice sum.
fn he_fcc() -> Cell {
    let h = 2.834589;
    bohr_cell(
        [[0.0, h, h], [h, 0.0, h], [h, h, 0.0]],
        vec![("He".into(), [0.0, 0.0, 0.0])],
    )
}

fn bohr_cell(a: [[f64; 3]; 3], atoms: Vec<(String, [f64; 3])>) -> Cell {
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(atoms),
            basis: BasisInput::Name("gth-szv".into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix(a),
        pseudo: Some("gth-pade".into()),
        ..Default::default()
    })
    .expect("reference cell must build")
}

// ---------------------------------------------------------------------------
// Oracle-free gates (D-PBC-19)
// ---------------------------------------------------------------------------

/// The gamma-point overlap is real, symmetric and positive definite, and the
/// diagonal is the on-site self-overlap plus its periodic images.
#[test]
fn gamma_overlap_is_real_symmetric_and_positive_definite() {
    let cell = diamond();
    let s = pbc_intor(&cell, "int1e_ovlp", &[[0.0; 3]], PbcIntorOpts::default())
        .expect("pbc_intor int1e_ovlp");

    assert_eq!(s.nkpts(), 1);
    assert_eq!((s.ni, s.nj, s.comp), (8, 8, 1));
    assert!(s.gamma[0], "kpts = [0,0,0] must be flagged gamma");
    assert_eq!(s.max_abs_imag(), 0.0, "gamma-point matrix must be real");

    assert_hermitian(&s, 0, 1e-12);
    assert_positive_definite(&s, 0);
}

/// `S^k` is Hermitian and positive definite at every k of a 2x2x2 mesh —
/// the property periodic SCF depends on and the one a sign error in the Bloch
/// phase destroys.
#[test]
fn overlap_is_hermitian_and_positive_definite_at_every_k() {
    let cell = diamond();
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("make_kpts");
    assert_eq!(kpts.len(), 8);

    let s = pbc_intor(&cell, "int1e_ovlp", &kpts, PbcIntorOpts::default()).expect("pbc_intor");
    assert_eq!(s.nkpts(), 8);

    for k in 0..8 {
        assert_hermitian(&s, k, 1e-12);
        assert_positive_definite(&s, k);
    }
    // Exactly one of the eight is the gamma point.
    assert_eq!(s.gamma.iter().filter(|g| **g).count(), 1);
}

/// `S^{-k} == conj(S^{k})`, i.e. time-reversal symmetry. A wrong phase sign
/// would still be Hermitian and positive definite, but this would fail.
#[test]
fn overlap_obeys_time_reversal_symmetry() {
    let cell = diamond();
    let k = [0.13, -0.07, 0.21];
    let mk = [-k[0], -k[1], -k[2]];
    let s = pbc_intor(&cell, "int1e_ovlp", &[k, mk], PbcIntorOpts::default()).expect("pbc_intor");

    let (a, b) = (s.at(0), s.at(1));
    let mut worst = 0.0_f64;
    for p in 0..a.len() {
        worst = worst.max((a.re[p] - b.re[p]).abs());
        worst = worst.max((a.im[p] + b.im[p]).abs());
    }
    assert!(
        worst < 1e-13,
        "S(-k) != conj(S(k)): max deviation {worst:e}"
    );
}

/// The lattice sum is converged: widening `rcut` by 50% must not move any
/// element. This is a complete correctness gate for the truncation and needs no
/// oracle (PBC-MASTER-PLAN plan 10-03, "self-consistency gate").
#[test]
fn lattice_sum_is_converged_in_rcut() {
    let cell = diamond();
    let k = [0.1, 0.2, -0.05];

    let ls_default = get_lattice_ls(&cell, None, None, true).expect("Ls");
    let ls_wide = get_lattice_ls(&cell, Some(cell.rcut * 1.5), None, true).expect("wide Ls");
    assert!(
        ls_wide.len() > ls_default.len(),
        "1.5*rcut must widen the image list ({} vs {})",
        ls_wide.len(),
        ls_default.len()
    );

    let a = pyscf_pbc_gto::intor_cross_with_images(
        "int1e_ovlp",
        &cell,
        &cell,
        &[k],
        PbcIntorOpts::default(),
        &ls_default,
        None,
    )
    .expect("default rcut");
    let b = pyscf_pbc_gto::intor_cross_with_images(
        "int1e_ovlp",
        &cell,
        &cell,
        &[k],
        PbcIntorOpts::default(),
        &ls_wide,
        None,
    )
    .expect("1.5x rcut");

    let worst = max_deviation(&a, &b);
    assert!(
        worst < 1e-9,
        "overlap moved by {worst:e} when rcut grew 50% — the lattice sum is not converged"
    );
}

/// Screening (D-PBC-08) must not change the answer. `cell.precision` is 1e-8,
/// but the per-shell radii used to screen sum to ~28 Bohr while `cell.rcut`
/// truncates the image list at ~21 Bohr, so on this system screening is
/// strictly weaker than the truncation already applied and the two agree far
/// below the nominal precision.
#[test]
fn screened_and_unscreened_agree() {
    let cell = diamond();
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("make_kpts");

    let plain = pbc_intor(&cell, "int1e_ovlp", &kpts, PbcIntorOpts::default()).expect("plain");
    let screened = pbc_intor(
        &cell,
        "int1e_ovlp",
        &kpts,
        PbcIntorOpts {
            screen: true,
            ..Default::default()
        },
    )
    .expect("screened");

    let worst = max_deviation(&plain, &screened);
    assert!(
        worst < 1e-12,
        "screening changed the overlap by {worst:e} (> 1e-12)"
    );
}

/// The neighbor list is exact: it keeps every on-site pair, and every pair it
/// drops really is beyond the sum of the two shell radii.
///
/// Note what it does NOT screen on diamond/`gth-szv`: the per-shell radii sum to
/// ~27.8 Bohr while `cell.rcut` already truncates the image list at 21.3 Bohr,
/// so with default radii nothing is dropped — the truncation is strictly
/// tighter than the screen. That is why `screened_and_unscreened_agree` passes
/// at 1e-12 rather than at `cell.precision`. Tightening the radii turns
/// screening on, which is what the second half of this test exercises.
#[test]
fn neighbor_list_is_exact() {
    let cell = diamond();
    let ls = get_lattice_ls(&cell, None, None, true).expect("Ls");
    let nl = pyscf_pbc_gto::build_neighbor_list_for_shlpairs(&cell, &ls).expect("neighbor list");

    assert_eq!(nl.nish, cell.mol.nbas);
    assert_eq!(nl.njsh, cell.mol.nbas);
    assert_eq!(nl.nimgs, ls.len());

    // The zero image must keep every on-site pair.
    let zero = ls
        .iter()
        .position(|l| l.iter().all(|c| c.abs() < 1e-12))
        .expect("Ls contains the origin");
    for ish in 0..nl.nish {
        for jsh in 0..nl.njsh {
            assert!(
                nl.contains(ish, jsh, zero),
                "pair ({ish},{jsh}) dropped at L = 0"
            );
        }
    }

    // Tight radii: screening must bite, and every decision must match the
    // `|R_j + L − R_i| < r_i + r_j` criterion exactly (neighbor_list.c:116-120).
    let tight = vec![3.0_f64; cell.mol.nbas];
    let nl =
        pyscf_pbc_gto::build_neighbor_list(&cell, None, &ls, Some(&tight), Some(&tight), 0, None)
            .expect("tight neighbor list");
    assert!(
        nl.len() < nl.dense_len() / 10,
        "3-Bohr radii kept {} of {} triples",
        nl.len(),
        nl.dense_len()
    );

    let coords = cell.mol.atom_coords();
    let bas_atom = |ib: usize| -> usize {
        use pyscf_core::raw_layout::{ATOM_OF, BAS_SLOTS};
        cell.mol._bas[ib * BAS_SLOTS + ATOM_OF] as usize
    };
    for (l_idx, l) in ls.iter().enumerate() {
        for ish in 0..nl.nish {
            for jsh in 0..nl.njsh {
                let ri = coords[bas_atom(ish)];
                let rj = coords[bas_atom(jsh)];
                let d = ((rj[0] + l[0] - ri[0]).powi(2)
                    + (rj[1] + l[1] - ri[1]).powi(2)
                    + (rj[2] + l[2] - ri[2]).powi(2))
                .sqrt();
                assert_eq!(
                    nl.contains(ish, jsh, l_idx),
                    d < 6.0,
                    "({ish},{jsh},L{l_idx}): d = {d} vs rmax = 6.0"
                );
            }
        }
    }
}

/// Loosening the screening precision degrades the answer gracefully and in
/// proportion — the guarantee D-PBC-08 rests on.
///
/// `rcut_by_shells(p)` returns the radius at which a shell's most diffuse
/// primitive has decayed below `p`, so screening a lattice sum with those radii
/// perturbs each element by a small multiple of `p`. Two decades of `p` are
/// checked so a screening bug that ignores the radii entirely (deviation flat)
/// or truncates far too early (deviation huge) both fail.
#[test]
fn screening_error_tracks_the_precision_it_was_built_for() {
    let cell = diamond();
    let ls = get_lattice_ls(&cell, None, None, true).expect("Ls");
    let k = [0.1, -0.2, 0.05];

    let plain = pyscf_pbc_gto::intor_cross_with_images(
        "int1e_ovlp",
        &cell,
        &cell,
        &[k],
        PbcIntorOpts::default(),
        &ls,
        None,
    )
    .expect("unscreened");

    let mut previous = f64::INFINITY;
    for precision in [1e-2_f64, 1e-4, 1e-6] {
        let radii = cell.rcut_by_shells(Some(precision));
        let nl = pyscf_pbc_gto::build_neighbor_list(
            &cell,
            None,
            &ls,
            Some(&radii),
            Some(&radii),
            0,
            None,
        )
        .expect("neighbor list");
        assert!(
            nl.len() < nl.dense_len(),
            "precision {precision:e} (radii {radii:?}) screened nothing"
        );

        let screened = pyscf_pbc_gto::intor_cross_with_images(
            "int1e_ovlp",
            &cell,
            &cell,
            &[k],
            PbcIntorOpts::default(),
            &ls,
            Some(&nl),
        )
        .expect("screened");

        let worst = max_deviation(&plain, &screened);
        println!(
            "precision {precision:e}: radii max {:.3} -> max |delta| {worst:e}",
            radii.iter().cloned().fold(0.0_f64, f64::max)
        );
        assert!(
            worst < 100.0 * precision,
            "screening at precision {precision:e} moved the overlap by {worst:e}"
        );
        assert!(
            worst <= previous,
            "tightening the precision to {precision:e} made the error WORSE \
             ({worst:e} vs {previous:e})"
        );
        previous = worst;
    }
}

/// `hermi = 1` evaluates only the lower triangle and mirrors it; the result must
/// equal the full `hermi = 0` matrix.
#[test]
fn hermi_triu_reproduces_the_full_matrix() {
    let cell = diamond();
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("make_kpts");

    let full = pbc_intor(&cell, "int1e_ovlp", &kpts, PbcIntorOpts::default()).expect("hermi=0");
    let tri = pbc_intor(
        &cell,
        "int1e_ovlp",
        &kpts,
        PbcIntorOpts {
            hermi: 1,
            ..Default::default()
        },
    )
    .expect("hermi=1");

    let worst = max_deviation(&full, &tri);
    assert!(worst < 1e-13, "hermi=1 disagrees with hermi=0 by {worst:e}");
}

/// The kinetic energy matrix is Hermitian with a positive, real diagonal, and
/// `int1e_ipovlp` returns the 3 component blocks its layout advertises.
#[test]
fn kinetic_and_derivative_families_have_the_right_shapes() {
    let cell = diamond();
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("make_kpts");

    let t = pbc_intor(&cell, "int1e_kin", &kpts, PbcIntorOpts::default()).expect("int1e_kin");
    assert_eq!(t.comp, 1);
    for k in 0..t.nkpts() {
        assert_hermitian(&t, k, 1e-12);
        for i in 0..t.ni {
            let (re, im) = t.element(k, 0, i, i);
            assert!(re > 0.0, "T[{k}][{i},{i}] = {re} is not positive");
            assert!(im.abs() < 1e-12);
        }
    }

    let dip =
        pbc_intor(&cell, "int1e_ipovlp", &kpts, PbcIntorOpts::default()).expect("int1e_ipovlp");
    assert_eq!(dip.comp, 3);
    assert_eq!(dip.at(0).len(), 3 * dip.ni * dip.nj);
}

/// A family Phase 10 does not cover must fail loudly, not silently produce a
/// plausible number.
#[test]
fn unsupported_family_is_refused() {
    let cell = he_fcc();
    let err = pbc_intor(&cell, "int2e", &[[0.0; 3]], PbcIntorOpts::default())
        .expect_err("int2e is not a Phase-10 family");
    assert!(
        matches!(
            err,
            pyscf_core::PyscfRsError::NotYetImplemented { phase: 13, .. }
        ),
        "unexpected error: {err}"
    );
}

/// One atom, one shell: `S^k` is the scalar `Σ_L exp(ikL) <s|s(L)>`, so it must
/// be real (the image set is inversion-symmetric) and its k = 0 value the largest.
#[test]
fn single_shell_cell_sums_to_a_real_scalar() {
    let cell = he_fcc();
    assert_eq!(cell.mol.nao_nr, 1);
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("make_kpts");
    let s = pbc_intor(&cell, "int1e_ovlp", &kpts, PbcIntorOpts::default()).expect("pbc_intor");

    let s0 = s.element(0, 0, 0, 0).0;
    for k in 0..s.nkpts() {
        let (re, im) = s.element(k, 0, 0, 0);
        assert!(im.abs() < 1e-12, "S({k}) has imaginary part {im:e}");
        assert!(re > 0.0, "S({k}) = {re} must stay positive definite");
        assert!(
            re <= s0 + 1e-12,
            "S({k}) = {re} exceeds the gamma value {s0}"
        );
    }
}

// ---------------------------------------------------------------------------
// The Phase-10 gate — live upstream comparison
// ---------------------------------------------------------------------------

const GATE: &str = "PYSCF_ORACLE_VENV";

const ORACLE_PY: &str = r#"
import json, sys
import numpy as np
from pyscf.pbc import gto

a_json, xyz_json, sym_json, nk_json, intor = sys.argv[1:6]
c = gto.Cell()
c.a = json.loads(a_json)
c.atom = [(s, tuple(r)) for s, r in zip(json.loads(sym_json), json.loads(xyz_json))]
c.basis = 'gth-szv'
c.pseudo = 'gth-pade'
c.unit = 'Bohr'
c.verbose = 0
c.build()
kpts = c.make_kpts(json.loads(nk_json))
mats = c.pbc_intor(intor, kpts=kpts)
out = {'nao': int(c.nao_nr()), 'nkpts': len(kpts), 'rcut': float(c.rcut),
       'nls': int(len(c.get_lattice_Ls())),
       'kpts': np.asarray(kpts).ravel().tolist(),
       'atom_charges': np.asarray(c.atom_charges()).tolist(),
       're': [], 'im': []}
for m in mats:
    m = np.asarray(m)
    out['re'].append(m.real.ravel(order='F').tolist())
    out['im'].append(np.zeros_like(m.real).ravel().tolist()
                     if m.dtype != np.complex128 else m.imag.ravel(order='F').tolist())
print(json.dumps(out))
"#;

/// **THE PHASE-10 GATE.** `pbc_intor('int1e_ovlp', kpts)` on diamond with a
/// 2x2x2 Monkhorst-Pack mesh, element-by-element against upstream, at 1e-12.
#[test]
#[ignore = "needs PYSCF_ORACLE_VENV + an upstream PySCF"]
fn ovlp_matches_upstream_on_diamond_222() {
    compare_with_upstream("int1e_ovlp", [2, 2, 2], 1e-12);
}

/// The same gate for the kinetic-energy matrix — a different cintx operator
/// through the identical lattice-sum path.
#[test]
#[ignore = "needs PYSCF_ORACLE_VENV + an upstream PySCF"]
fn kin_matches_upstream_on_diamond_222() {
    compare_with_upstream("int1e_kin", [2, 2, 2], 1e-12);
}

/// A 3x2x1 mesh — an unequal, non-power-of-two grid, so the k-point ordering
/// and the `cartesian_prod` axis order are exercised too.
#[test]
#[ignore = "needs PYSCF_ORACLE_VENV + an upstream PySCF"]
fn ovlp_matches_upstream_on_diamond_321() {
    compare_with_upstream("int1e_ovlp", [3, 2, 1], 1e-12);
}

fn compare_with_upstream(intor: &str, nk: [usize; 3], tol: f64) {
    let Some(py) = oracle_python() else {
        eprintln!("SKIP: {GATE} is not set — upstream oracle not run");
        return;
    };

    let cell = diamond();
    let a: Vec<Vec<f64>> = cell.a.iter().map(|r| r.to_vec()).collect();
    let xyz: Vec<Vec<f64>> = cell.mol.atom_coords().iter().map(|r| r.to_vec()).collect();
    let sym: Vec<String> = cell.mol._atom.iter().map(|(s, _)| s.clone()).collect();

    let want = run_python(
        &py,
        ORACLE_PY,
        &[
            serde_json::to_string(&a).unwrap(),
            serde_json::to_string(&xyz).unwrap(),
            serde_json::to_string(&sym).unwrap(),
            serde_json::to_string(&nk.to_vec()).unwrap(),
            intor.to_string(),
        ],
    );

    // Preconditions: the same cell, the same truncation, the same k-points.
    assert_eq!(want["nao"].as_u64().unwrap() as usize, cell.mol.nao_nr);
    assert!(
        (want["rcut"].as_f64().unwrap() - cell.rcut).abs() < 1e-10,
        "rcut differs: upstream {} vs {}",
        want["rcut"],
        cell.rcut
    );
    let ls = get_lattice_ls(&cell, None, None, true).expect("Ls");
    assert_eq!(
        want["nls"].as_u64().unwrap() as usize,
        ls.len(),
        "|Ls| differs"
    );
    assert_eq!(
        want["atom_charges"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect::<Vec<_>>(),
        cell.atom_charges(),
        "PP valence charges differ"
    );

    let kpts = make_kpts_default(&cell, nk).expect("make_kpts");
    let want_k: Vec<f64> = want["kpts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let got_k: Vec<f64> = kpts.iter().flatten().copied().collect();
    assert_eq!(want_k.len(), got_k.len(), "k-point count differs");
    for (i, (w, g)) in want_k.iter().zip(got_k.iter()).enumerate() {
        assert!((w - g).abs() < 1e-12, "kpt component {i}: {w} vs {g}");
    }

    let got = pbc_intor(&cell, intor, &kpts, PbcIntorOpts::default()).expect("pbc_intor");

    let mut worst = 0.0_f64;
    let mut worst_at = (0usize, 0usize);
    for k in 0..kpts.len() {
        let wre: Vec<f64> = want["re"][k]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let wim: Vec<f64> = want["im"][k]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        assert_eq!(wre.len(), got.at(k).len(), "element count differs at k={k}");
        for p in 0..wre.len() {
            let dre = (wre[p] - got.at(k).re[p]).abs();
            let dim = (wim[p] - got.at(k).im[p]).abs();
            let d = dre.max(dim);
            if d > worst {
                worst = d;
                worst_at = (k, p);
            }
        }
    }
    println!(
        "pbc_intor('{intor}', {nk:?}) vs upstream: max |delta| = {worst:e} at k={} element={}",
        worst_at.0, worst_at.1
    );
    assert!(
        worst < tol,
        "pbc_intor('{intor}') differs from upstream by {worst:e} (tolerance {tol:e}) \
         at k={} element={}",
        worst_at.0,
        worst_at.1
    );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn assert_hermitian(s: &PbcIntorOutput, k: usize, tol: f64) {
    let n = s.ni;
    assert_eq!(s.ni, s.nj);
    for c in 0..s.comp {
        for i in 0..n {
            for j in 0..n {
                let (re_ij, im_ij) = s.element(k, c, i, j);
                let (re_ji, im_ji) = s.element(k, c, j, i);
                assert!(
                    (re_ij - re_ji).abs() < tol && (im_ij + im_ji).abs() < tol,
                    "k={k} comp={c}: ({i},{j}) = {re_ij}+{im_ij}i vs \
                     conj({j},{i}) = {re_ji}-{im_ji}i"
                );
            }
        }
    }
}

/// Positive definiteness via the leading-principal-minor test on the real
/// `2n x 2n` embedding `[[Re, -Im], [Im, Re]]` (D-PBC-04's fallback shape),
/// done with an unpivoted Cholesky: it succeeds iff the matrix is PD.
fn assert_positive_definite(s: &PbcIntorOutput, k: usize) {
    let n = s.ni;
    let m = 2 * n;
    let mut a = vec![0.0_f64; m * m];
    for i in 0..n {
        for j in 0..n {
            let (re, im) = s.element(k, 0, i, j);
            a[i * m + j] = re;
            a[i * m + (j + n)] = -im;
            a[(i + n) * m + j] = im;
            a[(i + n) * m + (j + n)] = re;
        }
    }
    // In-place lower Cholesky.
    for i in 0..m {
        for j in 0..=i {
            let mut sum = a[i * m + j];
            for p in 0..j {
                sum -= a[i * m + p] * a[j * m + p];
            }
            if i == j {
                assert!(
                    sum > 0.0,
                    "S(k={k}) is not positive definite: pivot {i} = {sum:e}"
                );
                a[i * m + j] = sum.sqrt();
            } else {
                a[i * m + j] = sum / a[j * m + j];
            }
        }
    }
}

fn max_deviation(a: &PbcIntorOutput, b: &PbcIntorOutput) -> f64 {
    assert_eq!(a.nkpts(), b.nkpts());
    let mut worst = 0.0_f64;
    for k in 0..a.nkpts() {
        for p in 0..a.at(k).len() {
            worst = worst.max((a.at(k).re[p] - b.at(k).re[p]).abs());
            worst = worst.max((a.at(k).im[p] - b.at(k).im[p]).abs());
        }
    }
    worst
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

fn oracle_python() -> Option<PathBuf> {
    let raw = std::env::var(GATE).ok()?.trim().to_string();
    if raw.is_empty() {
        return None;
    }
    let p = if matches!(raw.as_str(), "1" | "true" | "auto" | "yes") {
        workspace_root().join(".venv/bin/python")
    } else {
        let c = PathBuf::from(&raw);
        if c.is_dir() { c.join("bin/python") } else { c }
    };
    assert!(
        p.exists(),
        "{GATE} = {raw:?} resolved to {p:?}, which does not exist"
    );
    Some(p)
}

static SCRIPT_SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn run_python(py: &PathBuf, script: &str, args: &[String]) -> serde_json::Value {
    let n = SCRIPT_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("pbc_intor_oracle_{}_{n}.py", std::process::id()));
    std::fs::write(&path, script).expect("write oracle script");
    let out = Command::new(py)
        .arg(&path)
        .args(args)
        .current_dir(workspace_root())
        .output()
        .expect("spawn upstream python");
    let _ = std::fs::remove_file(&path);
    assert!(
        out.status.success(),
        "upstream oracle failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    // PySCF prints a plugin banner before our JSON; take the last non-empty line.
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout
        .lines()
        .rev()
        .find(|l| l.trim_start().starts_with('{'))
        .expect("oracle produced no JSON line");
    serde_json::from_str(line).expect("oracle JSON parses")
}
