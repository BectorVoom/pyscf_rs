//! DIAGNOSTIC PROBE — not a gate. See
//! `.planning/phases/17-ksymm-multigrid/17-04-MEASUREMENT.md`, the document
//! this probe produced, and `17-04-SUMMARY.md`.
//!
//! # What it settled (the question is CLOSED)
//!
//! `tests/basis.rs::check_fock_block_diagonal` asserts the converged Fock is
//! block-diagonal in the symmetry-adapted basis at `BLOCK_DIAG_TOL = 1e-11`,
//! a tolerance 17-04-PLAN.md fixed BEFORE anything measured the phase's
//! actual floors. Its first `--release` run failed, with a true maximum
//! off-block element of 3.99e-10 — 40x the gate.
//!
//! This probe swept `cell.precision` x `conv_tol_grad` and REPORTED the
//! maximum off-block magnitude (rather than asserting a guessed number),
//! carrying the off-block OVERLAP alongside as a projector-only control that
//! does not depend on the SCF at all. The conclusion, in full in
//! `17-04-MEASUREMENT.md` §3:
//!
//! * **`S` is integral-precision-limited and nothing else** — 9.99e-12 at
//!   `precision = 1e-8`, 4.14e-14 at 1e-10, and *bit-identical* across every
//!   `conv_tol_grad` at fixed precision. That invariance is the control: it
//!   says the probe measures what it claims.
//! * **`F` is limited by BOTH axes, and neither alone is enough.** Tightening
//!   only the integrals leaves it at 4.18e-10; tightening only the
//!   convergence plateaus it at ~1.92e-11 (the integral floor showing
//!   through). Tightening both drops it to **5.48e-13**.
//!
//! So there is **no algebraic defect in `basis.rs`**: the residual was a
//! fixture-configuration floor. `BLOCK_DIAG_TOL` stayed at 1e-11 — reachable
//! with ~18x margin — and `tests/basis.rs` was rebuilt at
//! `FIXTURE_PRECISION = 1e-10` / `FIXTURE_CONV_TOL_GRAD = 1e-10` instead.
//!
//! # Why it is kept
//!
//! It documents the floor and can re-derive it if the gate ever moves, or if
//! a later change to `basis.rs`/the integrals makes `tests/basis.rs` fail
//! again — in which case run this FIRST, before touching any tolerance. The
//! sweep below is trimmed to the two decisive points: the old loose fixture
//! and the fixed one. Add rows to re-measure a different question.
//!
//! ```text
//! cargo test -p pyscf-pbc-symm --release --test basis_precision_probe -- --ignored --nocapture
//! ```

use num_complex::Complex64;

use pyscf_algebra::CTensor;
use pyscf_pbc_df::JkOpts;
use pyscf_pbc_gto::test_systems::si_precision;
use pyscf_pbc_gto::{Cell, make_kpts_default};
use pyscf_pbc_scf::krhf::to_row_major;
use pyscf_pbc_scf::{KInitGuess, KScfConfig, Krhf};
use pyscf_pbc_symm::basis::{self, SymmAdaptedBasisInput};
use pyscf_pbc_symm::space_group::{SPGElement, SYMPREC};
use pyscf_pbc_symm::symmetry::Symmetry;

fn col_at(t: &CTensor, nrows: usize, row: usize, col: usize) -> Complex64 {
    Complex64::new(t.re[col * nrows + row], t.im[col * nrows + row])
}

fn row_at(t: &CTensor, ncols: usize, row: usize, col: usize) -> Complex64 {
    Complex64::new(t.re[row * ncols + col], t.im[row * ncols + col])
}

/// Same test-local little co-group as `tests/basis.rs` (17-05 owns the
/// production version).
fn little_cogroup(cell: &Cell, ops: &[SPGElement], k_scaled: [f64; 3], tol: f64) -> Vec<usize> {
    ops.iter()
        .enumerate()
        .filter_map(|(i, op)| {
            let op_b = op.a2b(cell).ok()?;
            let mapped = op_b.dot_rot(&k_scaled);
            let fixes_k = (0..3).all(|d| {
                let diff = mapped[d] - k_scaled[d];
                (diff - diff.round()).abs() < tol
            });
            fixes_k.then_some(i)
        })
        .collect()
}

fn converged_fock(mf: &Krhf, dm: &pyscf_pbc_scf::KDms) -> Vec<CTensor> {
    let nao = mf.cell().mol.nao_nr;
    let mut fock = to_row_major(
        pyscf_pbc_df::get_hcore(mf.with_df.as_ref(), mf.kpts()).expect("get_hcore"),
        nao,
    );
    let r = mf
        .with_df
        .get_jk(
            dm,
            mf.kpts(),
            JkOpts {
                hermi: 1,
                kpts_band: None,
                with_j: true,
                with_k: true,
                exxdiv: mf.exxdiv,
                omega: None,
                kk_symmetry: false,
            },
        )
        .expect("get_jk");
    let vj = r.vj.expect("vj");
    let vk = r.vk.expect("vk");
    for (k, f) in fock.iter_mut().enumerate() {
        for i in 0..f.re.len() {
            f.re[i] += vj[0][k].re[i] - 0.5 * vk[0][k].re[i];
            f.im[i] += vj[0][k].im[i] - 0.5 * vk[0][k].im[i];
        }
    }
    fock
}

/// Largest off-block |Fock| element over all k-points, and the largest
/// off-block |S| element for comparison (the projector-only control).
///
/// Si, `gth-szv`/`gth-pade`, `[2,2,2]`, `symmorphic = true` — the same
/// system `tests/basis.rs` gates, built through
/// [`si_precision`] so `cell.precision` can be varied. (Mutating
/// `cell.precision` on an already-built cell and calling the `build()`
/// METHOD instead silently drops the pseudopotential: `Nocc 112 > Nmo 64`.
/// `si_precision` goes through `CellBuildArgs`.)
fn max_off_block_grad(precision: f64, conv_tol: f64, conv_tol_grad: f64) -> (f64, f64) {
    let mut cell = si_precision(precision);

    let nao = cell.mol.nao_nr;
    let kpts_abs = make_kpts_default(&cell, [2, 2, 2]).expect("make_kpts_default");
    let kpts_scaled = cell.get_scaled_kpts(&kpts_abs);

    let sym = Symmetry::build(&cell, true, true, false).expect("Symmetry::build");
    let little_cogroups: Vec<Vec<usize>> = kpts_scaled
        .iter()
        .map(|&k| little_cogroup(&cell, &sym.ops, k, SYMPREC))
        .collect();

    let input = SymmAdaptedBasisInput {
        kpts_scaled_ibz: kpts_scaled.clone(),
        little_cogroup_ops: little_cogroups,
        ops: sym.ops.clone(),
        dmats: sym.dmats.clone(),
    };
    basis::build_symmetry(&mut cell, &input).expect("build_symmetry");
    let symm_orb = cell.symm_orb.clone().expect("symm_orb");
    let irrep_id = cell.irrep_id.clone().expect("irrep_id");

    // Projector-only control: S in the adapted basis.
    let ovlp = pyscf_pbc_gto::get_ovlp(&cell, &kpts_abs).expect("get_ovlp");
    let mut worst_s: f64 = 0.0;
    for (k, s) in ovlp.iter().enumerate() {
        let (c, ids) = (&symm_orb[k], &irrep_id[k]);
        for p in 0..nao {
            for q in 0..nao {
                if ids[p] == ids[q] {
                    continue;
                }
                let mut acc = Complex64::new(0.0, 0.0);
                for i in 0..nao {
                    let ci = col_at(c, nao, i, p).conj();
                    for j in 0..nao {
                        acc += ci * col_at(s, nao, i, j) * col_at(c, nao, j, q);
                    }
                }
                worst_s = worst_s.max(acc.norm());
            }
        }
    }

    let mf = Krhf::new(cell.clone(), &kpts_abs).expect("Krhf::new");
    let cfg = KScfConfig {
        conv_tol,
        conv_tol_grad: Some(conv_tol_grad),
        max_cycle: 50,
        init_guess: KInitGuess::Minao,
        ..KScfConfig::default()
    };
    let r = mf.kernel(&cfg).expect("KRHF");
    assert!(r.converged, "KRHF did not converge");
    let fock = converged_fock(&mf, &r.dm);

    let mut worst_f: f64 = 0.0;
    for (k, f) in fock.iter().enumerate() {
        let (c, ids) = (&symm_orb[k], &irrep_id[k]);
        for p in 0..nao {
            for q in 0..nao {
                if ids[p] == ids[q] {
                    continue;
                }
                let mut acc = Complex64::new(0.0, 0.0);
                for i in 0..nao {
                    let ci = col_at(c, nao, i, p).conj();
                    for j in 0..nao {
                        acc += ci * row_at(f, nao, i, j) * col_at(c, nao, j, q);
                    }
                }
                worst_f = worst_f.max(acc.norm());
            }
        }
    }
    (worst_f, worst_s)
}

#[test]
#[ignore = "diagnostic probe; runs several converged KRHFs"]
fn fock_block_diagonality_floor() {
    use std::io::Write;
    // The two decisive points of `17-04-MEASUREMENT.md` §3's table:
    // the OLD loose fixture, and the one `tests/basis.rs` now uses.
    // (precision, conv_tol, conv_tol_grad, what it shows)
    let points: [(f64, f64, f64, &str); 2] = [
        (
            1e-8,
            1e-11,
            1e-8,
            "the OLD fixture: expect max|F| ~ 3.99e-10, 40x OVER the 1e-11 gate",
        ),
        (
            1e-10,
            1e-11,
            1e-10,
            "the FIXED fixture: expect max|F| ~ 5.5e-13, ~18x INSIDE the gate",
        ),
    ];
    for &(prec, conv_tol, grad, note) in &points {
        let (wf, ws) = max_off_block_grad(prec, conv_tol, grad);
        println!(
            "precision={prec:e} conv_tol={conv_tol:e} grad={grad:e}  \
             max|off-block F|={wf:e}   max|off-block S|={ws:e}   ({note})"
        );
        std::io::stdout().flush().ok();
    }
}
