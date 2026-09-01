//! Integration tests for `pyscf_pbc_symm::basis` — 17-04-PLAN.md Task 4.
//!
//! No oracle: every identity here is a property a symmetry-adapted basis
//! must satisfy by CONSTRUCTION, not a number pinned against upstream. Run
//! on `si`/`diamond` at `[2,2,2]` and `[3,3,3]` k-meshes (17-CONTEXT §1.2 /
//! 17-04-PLAN.md).
//!
//! # `KPoints` is 17-05, not this plan
//!
//! There is no `KPoints`/IBZ-folding machinery yet, so this file computes a
//! little co-group DIRECTLY for every k-point in the full mesh (not just a
//! reduced IBZ set): for each space-group op, keep it iff it maps the
//! k-point back to itself modulo a reciprocal lattice vector. That is
//! exactly what a real `little_cogroup_ops` entry would contain for an IBZ
//! representative — [`pyscf_pbc_symm::basis::symm_adapted_basis`] does not
//! care whether the k-points it is handed are a genuine irreducible set, only
//! that each one's own little co-group is correct — so testing every mesh
//! k-point (not just a reduced subset) is a STRONGER exercise of the same
//! code path, not a different one. This helper is deliberately test-local
//! (17-04-PLAN.md's own instruction): production `little_cogroup_ops` is
//! 17-05's job.

use num_complex::Complex64;

use pyscf_algebra::CTensor;
use pyscf_pbc_df::JkOpts;
use pyscf_pbc_gto::Cell;
use pyscf_pbc_gto::test_systems::{diamond_precision, si_precision};
use pyscf_pbc_gto::{get_ovlp, make_kpts_default};
use pyscf_pbc_scf::krhf::to_row_major;
use pyscf_pbc_scf::{KInitGuess, KScfConfig, Krhf};
use pyscf_pbc_symm::basis::{self, IrrepBlock, SymmAdaptedBasisInput, TOL};
use pyscf_pbc_symm::group::{PgElement, PointGroup};
use pyscf_pbc_symm::space_group::{SPGElement, SYMPREC};
use pyscf_pbc_symm::symmetry::{DmatSet, Symmetry, get_rotation_mat};

const ORTHO_TOL: f64 = 1e-12;
const BLOCK_DIAG_TOL: f64 = 1e-11;
const INVARIANCE_TOL: f64 = 1e-8;

/// # The fixture is deliberately TIGHT — do not loosen it
///
/// `cell.precision` here is 1e-10, NOT the `test_systems` default 1e-8, and
/// [`check_fock_block_diagonal`] runs its KRHF at `conv_tol_grad = 1e-10`,
/// NOT the usual 1e-8. Both axes are load-bearing and neither alone is
/// enough. From the 2-D sweep in
/// `.planning/phases/17-ksymm-multigrid/17-04-MEASUREMENT.md` §3 (Si,
/// `[2,2,2]`, max |off-block F| over every k and every (p,q)):
///
/// | `precision` | `conv_tol_grad` | max &#124;off-block F&#124; |
/// |---|---|---|
/// | 1e-8 (default) | 1e-8 | 3.99e-10 — 40x OVER the gate |
/// | 1e-10 | 1e-8 | 4.18e-10 — integrals alone do nothing |
/// | 1e-8 | 1e-12 | 1.92e-11 — convergence alone plateaus |
/// | **1e-10** | **1e-10** | **5.48e-13** — ~18x INSIDE the gate |
///
/// The measurement's control is the overlap: `max |off-block S|` is
/// *bit-identical* across every `conv_tol_grad` at fixed precision, as it
/// must be for a quantity that does not depend on the SCF. So the residual
/// is a fixture-configuration floor, NOT an algebraic defect in `basis.rs`,
/// and [`BLOCK_DIAG_TOL`] stays at 1e-11 with margin to spare. If this file
/// ever fails again, tighten the fixture or re-run
/// `tests/basis_precision_probe.rs` — do NOT relax a tolerance here.
const FIXTURE_PRECISION: f64 = 1e-10;

/// See [`FIXTURE_PRECISION`]: 1e-8 leaves the Fock residual at 4.0e-10.
const FIXTURE_CONV_TOL_GRAD: f64 = 1e-10;

// ---------------------------------------------------------------------
// Worst-element tracking.
//
// Every check below asserts on the LARGEST residual over all k-points and
// all (p, q), never on the first element to exceed the tolerance, and
// PRINTS that maximum under `--nocapture`. Reporting the measured floor
// (rather than only whether it cleared a threshold) is what
// `17-04-MEASUREMENT.md` needed and could not read off the original
// first-violation assert: the first violating element was 1.58e-11 while
// the true maximum was 3.99e-10, a 25x difference that changed the
// diagnosis. See [`FIXTURE_PRECISION`].
// ---------------------------------------------------------------------

/// The largest residual seen, and where.
struct Worst {
    val: f64,
    at: (usize, usize, usize),
}

impl Worst {
    fn new() -> Self {
        Worst { val: 0.0, at: (0, 0, 0) }
    }

    fn see(&mut self, val: f64, k: usize, p: usize, q: usize) {
        if val > self.val {
            self.val = val;
            self.at = (k, p, q);
        }
    }

    /// Print the measured maximum, then assert it against `tol`.
    fn report(&self, fixture: &str, what: &str, tol: f64) {
        let (k, p, q) = self.at;
        println!(
            "  {fixture:<14} max {what:<34} = {:e}   (tol {tol:e}, at k={k} p={p} q={q})",
            self.val
        );
        assert!(
            self.val < tol,
            "{fixture}: max {what} = {:e} exceeds {tol:e} at k={k} p={p} q={q}. \
             Do NOT relax the tolerance - see FIXTURE_PRECISION and \
             17-04-MEASUREMENT.md, and re-run tests/basis_precision_probe.rs first.",
            self.val
        );
    }
}

// ---------------------------------------------------------------------
// Column-major / row-major complex accessors — see `basis.rs`'s module doc:
// `symm_orb`/`so` are COLUMN-MAJOR, `S`/`F` (from `get_ovlp` / the Fock
// build below) are the F-order/row-major conventions `pyscf-pbc-gto` /
// `pyscf-pbc-scf` already use.
// ---------------------------------------------------------------------

fn col_at(t: &CTensor, nrows: usize, row: usize, col: usize) -> Complex64 {
    let idx = col * nrows + row;
    Complex64::new(t.re[idx], t.im[idx])
}

fn row_at(t: &CTensor, ncols: usize, row: usize, col: usize) -> Complex64 {
    let idx = row * ncols + col;
    Complex64::new(t.re[idx], t.im[idx])
}

/// `get_ovlp`'s output is F-order (column-major) — see `crate` doc /
/// `tests/symmetry.rs`'s module doc. Same layout as `so`, so no conversion
/// is needed: read it with [`col_at`].

// ---------------------------------------------------------------------
// Test-local little co-group (see the module doc — this is NOT production
// code; 17-05 owns the real `little_cogroup_ops`).
// ---------------------------------------------------------------------

/// Indices into `ops` of every operation that maps `k_scaled` back to itself
/// modulo a reciprocal lattice vector.
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

/// Mirrors `symm_adapted_basis`'s internal little-co-group assembly
/// (`basis.py:115-125`): sort the co-group's elements, carry `ops`/`dmats`
/// along in lockstep. Duplicated here (rather than exposed from
/// `pyscf_pbc_symm::basis`) because it is exactly the piece 17-05's
/// `KPoints` will own — see the module doc.
fn sorted_little_pg(
    ops: &[SPGElement],
    dmats: &[DmatSet],
    cogroup_idx: &[usize],
) -> (PointGroup, Vec<SPGElement>, Vec<DmatSet>) {
    let mut triples: Vec<(PgElement, usize)> = cogroup_idx
        .iter()
        .map(|&iop| {
            let op = &ops[iop];
            let rot: [[i32; 3]; 3] =
                std::array::from_fn(|r| std::array::from_fn(|c| op.rot[r][c].round() as i32));
            (PgElement::new(rot), iop)
        })
        .collect();
    triples.sort_by(|a, b| a.0.cmp(&b.0));
    let elements: Vec<PgElement> = triples.iter().map(|(e, _)| *e).collect();
    let spg_ops: Vec<SPGElement> = triples.iter().map(|&(_, iop)| ops[iop]).collect();
    let dmats_small: Vec<DmatSet> = triples.iter().map(|&(_, iop)| dmats[iop].clone()).collect();
    let pg = PointGroup::new(elements).expect("a little co-group must itself be a group");
    (pg, spg_ops, dmats_small)
}

// ---------------------------------------------------------------------
// Fixture: one system at one k-mesh, with everything the four checks need.
// ---------------------------------------------------------------------

struct Fixture {
    cell: Cell,
    kpts_abs: Vec<[f64; 3]>,
    kpts_scaled: Vec<[f64; 3]>,
    ops: Vec<SPGElement>,
    dmats: Vec<DmatSet>,
    little_cogroups: Vec<Vec<usize>>,
    /// `blocks[k]` — [`symm_adapted_basis`]'s per-irrep blocks at k-point `k`.
    blocks: Vec<Vec<IrrepBlock>>,
    /// The flattened `Cell`-ready form (Task 3's `build_symmetry`), built
    /// from the SAME `blocks`.
    symm_orb: Vec<CTensor>,
    irrep_id: Vec<Vec<i32>>,
}

fn build_fixture(mut cell: Cell, mesh: [usize; 3]) -> Fixture {
    let kpts_abs = make_kpts_default(&cell, mesh).expect("make_kpts_default");
    let kpts_scaled = cell.get_scaled_kpts(&kpts_abs);

    // `symmorphic = true` (restrict to the zero-fractional-translation
    // subgroup) — NOT for lack of exercising the phase: even a
    // zero-TRANSLATION op picks up a non-trivial per-atom phase in
    // `_get_phase` (`Lshift` is generally non-zero even when `op.trans ==
    // 0`, since atoms sit at generic positions), so this still exercises
    // Task 1's phase threading. The full non-symmorphic group
    // (`symmorphic = false`, diamond/si's actual space group Fd-3m) is
    // NOT used here: verified directly against live upstream PySCF 2.12.1
    // that `pyscf.pbc.symm.basis.symm_adapted_basis` itself hits
    // `assert nso == cell.nao` (`basis.py:90`) for BOTH `diamond` and `si`
    // at BOTH `[2,2,2]` and `[3,3,3]` with `symmorphic=False`, at the
    // little co-groups of order 16 that occur at k-points like
    // `(0.5, 0.5, 0.0)` — a genuine upstream limitation of the
    // non-symmorphic-glide + special-k-point combination, not a defect
    // this port introduced (see `17-04-SUMMARY.md`). `symmorphic = true`
    // is also upstream's own recommended/working configuration here.
    let sym = Symmetry::build(&cell, true, true, false).expect("Symmetry::build");

    let little_cogroups: Vec<Vec<usize>> = kpts_scaled
        .iter()
        .map(|&k| little_cogroup(&cell, &sym.ops, k, SYMPREC))
        .collect();
    for (i, cg) in little_cogroups.iter().enumerate() {
        assert!(
            !cg.is_empty(),
            "little co-group at k-point {i} ({:?}) must at least contain the identity",
            kpts_scaled[i]
        );
    }

    let blocks = basis::symm_adapted_basis(
        &cell,
        &kpts_scaled,
        &little_cogroups,
        &sym.ops,
        &sym.dmats,
        TOL,
    )
    .expect("symm_adapted_basis");

    let input = SymmAdaptedBasisInput {
        kpts_scaled_ibz: kpts_scaled.clone(),
        little_cogroup_ops: little_cogroups.clone(),
        ops: sym.ops.clone(),
        dmats: sym.dmats.clone(),
    };
    basis::build_symmetry(&mut cell, &input).expect("build_symmetry");
    let symm_orb = cell.symm_orb.clone().expect("symm_orb must be Some after build_symmetry");
    let irrep_id = cell.irrep_id.clone().expect("irrep_id must be Some after build_symmetry");

    Fixture {
        cell,
        kpts_abs,
        kpts_scaled,
        ops: sym.ops,
        dmats: sym.dmats,
        little_cogroups,
        blocks,
        symm_orb,
        irrep_id,
    }
}

// ---------------------------------------------------------------------
// Check 1 — orthonormality, on `symm_orb` (the flattened, Cell-ready form).
//
// **Correction to 17-04-PLAN.md's literal wording** ("symm_orb[k]ᴴ · S(k) ·
// symm_orb[k] == I"), recorded here and in 17-04-SUMMARY.md: verified
// directly against live upstream PySCF 2.12.1 that this is NOT what
// `pyscf.pbc.symm.basis` produces. `_gram_schmidt` (`basis.py:93-108`)
// orthonormalizes with the PLAIN (Euclidean/Hermitian) inner product
// `np.dot(u.conj(), v)` — no `S` anywhere in it — so the identity that
// actually holds, both upstream and here, is `symm_orbᴴ symm_orb == I`.
// Confirmed on live upstream diamond Γ-point:
// `so.conj().T @ S @ so` is `[[2.366, 2.209],[2.209, 2.366]]` (NOT `I`;
// `S[0,0] = 2.366` itself, since this basis's AOs are not `S`-normalized
// to begin with), while `so.conj().T @ so` IS exactly `I`. This is
// mathematically consistent: the AO-space rotation `R(g)`
// ([`get_rotation_mat`]) is built from atom PERMUTATIONS and per-shell
// Wigner-D matrices — both unitary under the plain Hermitian inner
// product — so the projector is unitary (and Gram-Schmidt without `S` is
// the right orthonormalization) even though the underlying AOs are not
// `S`-orthonormal.
//
// The SPIRIT of the plan's "S metric" requirement — that per-irrep
// generalized eigenproblems in `khf_ksymm.eig` are well-posed — is instead
// exactly [`check_s_block_diagonal`]: `symm_orbᴴ S symm_orb` need not be
// `I`, but it DOES need to be block-diagonal by irrep (group theory: `S`
// commutes with every group operation, so it has no matrix elements
// between distinct irreps), which is what makes solving `H_ir c = S_ir c E`
// SEPARATELY, one irrep at a time, correct in the first place.
// ---------------------------------------------------------------------

fn check_orthonormal(fx: &Fixture, name: &str) {
    let nao = fx.cell.mol.nao_nr;
    let mut worst = Worst::new();
    for (k, c) in fx.symm_orb.iter().enumerate() {
        for p in 0..nao {
            for q in 0..nao {
                let mut acc = Complex64::new(0.0, 0.0);
                for i in 0..nao {
                    acc += col_at(c, nao, i, p).conj() * col_at(c, nao, i, q);
                }
                let want = if p == q { 1.0 } else { 0.0 };
                worst.see((acc - Complex64::new(want, 0.0)).norm(), k, p, q);
            }
        }
    }
    worst.report(name, "|symm_orbᴴ symm_orb - I|", ORTHO_TOL);
}

/// See [`check_orthonormal`]'s doc: `symm_orbᴴ S symm_orb` is block-diagonal
/// by irrep (off-block elements zero), even though it is not `I`.
fn check_s_block_diagonal(fx: &Fixture, name: &str) {
    let nao = fx.cell.mol.nao_nr;
    let s_all = get_ovlp(&fx.cell, &fx.kpts_abs).expect("get_ovlp");
    let mut worst = Worst::new();
    for (k, s) in s_all.iter().enumerate() {
        let c = &fx.symm_orb[k];
        let ids = &fx.irrep_id[k];
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
                worst.see(acc.norm(), k, p, q);
            }
        }
    }
    worst.report(name, "|off-block symm_orbᴴ S symm_orb|", BLOCK_DIAG_TOL);
}

// ---------------------------------------------------------------------
// Check 2 — completeness: columns across every surviving irrep sum to nao,
// at every k-point (both on the raw `blocks` and on the flattened form).
// ---------------------------------------------------------------------

fn check_completeness(fx: &Fixture) {
    let nao = fx.cell.mol.nao_nr;
    for (k, blocks) in fx.blocks.iter().enumerate() {
        let total: usize = blocks.iter().map(|b| b.ncol).sum();
        assert_eq!(total, nao, "k={k}: irrep columns sum to {total}, want nao={nao}");
        assert_eq!(
            fx.irrep_id[k].len(),
            nao,
            "k={k}: flattened irrep_id has {} entries, want nao={nao}",
            fx.irrep_id[k].len()
        );
    }
}

// ---------------------------------------------------------------------
// Check 3 — block-diagonality of a converged full-BZ Fock, on the
// flattened `symm_orb`/`irrep_id`.
// ---------------------------------------------------------------------

/// `khf.py:670-695`'s `get_bands` Fock assembly (`hcore + vj - vk/2`), minus
/// the final `eig` — this test needs the MATRIX, not its eigen-decomposition.
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

fn check_fock_block_diagonal(fx: &Fixture, name: &str) {
    let nao = fx.cell.mol.nao_nr;
    let mf = Krhf::new(fx.cell.clone(), &fx.kpts_abs).expect("Krhf::new");
    let cfg = KScfConfig {
        conv_tol: 1e-11,
        conv_tol_grad: Some(FIXTURE_CONV_TOL_GRAD),
        max_cycle: 50,
        init_guess: KInitGuess::Minao,
        ..KScfConfig::default()
    };
    let r = mf.kernel(&cfg).expect("full-BZ KRHF must run");
    assert!(r.converged, "full-BZ KRHF did not converge in {} cycles", r.cycles);

    let fock = converged_fock(&mf, &r.dm);

    let mut worst = Worst::new();
    for (k, f) in fock.iter().enumerate() {
        let c = &fx.symm_orb[k];
        let ids = &fx.irrep_id[k];
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
                worst.see(acc.norm(), k, p, q);
            }
        }
    }
    worst.report(name, "|off-block symm_orbᴴ F symm_orb|", BLOCK_DIAG_TOL);
}

// ---------------------------------------------------------------------
// Check 4 — invariance: for every little-co-group op g and every surviving
// irrep block, R(g) maps the block into itself up to the irrep's
// character: Tr(C_irᴴ R(g) C_ir) == (ncol_ir / dim_ir) * chi_ir(g).
// (`ncol_ir / dim_ir` is the irrep's MULTIPLICITY in this k-point's AO
// representation — an exact integer whenever the decomposition is correct.)
// ---------------------------------------------------------------------

fn check_invariance(fx: &Fixture, name: &str) {
    let nao = fx.cell.mol.nao_nr;
    let mut worst = Worst::new();
    for (k, blocks) in fx.blocks.iter().enumerate() {
        let kpt_scaled = fx.kpts_scaled[k];
        let (pg, spg_ops, dmats_small) =
            sorted_little_pg(&fx.ops, &fx.dmats, &fx.little_cogroups[k]);
        let chartab = pg.character_table(true); // [nirrep][order]

        for block in blocks {
            if block.ncol == 0 {
                continue;
            }
            let ir = block.irrep_id as usize;
            let dim_ir = chartab[ir][0].re;
            assert!(dim_ir > 0.5, "k={k} ir={ir}: irrep dimension must be >= 1, got {dim_ir}");
            let multiplicity = block.ncol as f64 / dim_ir;

            for iop in 0..pg.order() {
                let r = get_rotation_mat(
                    &fx.cell,
                    kpt_scaled,
                    nao,
                    &spg_ops[iop],
                    &dmats_small[iop],
                    false,
                    SYMPREC,
                )
                .expect("R(g)");

                let mut trace = Complex64::new(0.0, 0.0);
                for p in 0..block.ncol {
                    // (R C)[:,p] = sum_j R[:,j] * C[j,p]
                    let mut rc_p = vec![Complex64::new(0.0, 0.0); nao];
                    for i in 0..nao {
                        let mut acc = Complex64::new(0.0, 0.0);
                        for j in 0..nao {
                            acc += r[i * nao + j] * col_at(&block.so, nao, j, p);
                        }
                        rc_p[i] = acc;
                    }
                    // (C^H (R C))[p,p]
                    let mut dot = Complex64::new(0.0, 0.0);
                    for i in 0..nao {
                        dot += col_at(&block.so, nao, i, p).conj() * rc_p[i];
                    }
                    trace += dot;
                }

                let expected = chartab[ir][iop] * multiplicity;
                let d = (trace - expected).norm();
                assert!(
                    d < INVARIANCE_TOL,
                    "k={k} ir={ir} op={iop}: Tr(C^H R C) = {trace:?}, want {expected:?} \
                     (multiplicity {multiplicity}, chi={:?}): diff {d:e}",
                    chartab[ir][iop]
                );
                worst.see(d, k, ir, iop);
            }
        }
    }
    worst.report(name, "|Tr(Cᴴ R C) - mult*chi|", INVARIANCE_TOL);
}

// ---------------------------------------------------------------------
// The four (system, mesh) combinations 17-04-PLAN.md Task 4 asks for.
// ---------------------------------------------------------------------

/// Runs all four checks and, under `--nocapture`, prints each one's
/// MEASURED maximum residual next to its tolerance - see [`Worst`].
fn run_all_checks(name: &str, cell: Cell, mesh: [usize; 3]) {
    let fx = build_fixture(cell, mesh);
    println!("{name}: measured maxima (recorded in 17-04-SUMMARY.md)");
    check_orthonormal(&fx, name);
    check_s_block_diagonal(&fx, name);
    check_completeness(&fx);
    check_invariance(&fx, name);
    check_fock_block_diagonal(&fx, name);
}

#[test]
fn si_2x2x2() {
    run_all_checks("si_2x2x2", si_precision(FIXTURE_PRECISION), [2, 2, 2]);
}

#[test]
fn diamond_2x2x2() {
    run_all_checks("diamond_2x2x2", diamond_precision(FIXTURE_PRECISION), [2, 2, 2]);
}

#[test]
fn si_3x3x3() {
    run_all_checks("si_3x3x3", si_precision(FIXTURE_PRECISION), [3, 3, 3]);
}

#[test]
fn diamond_3x3x3() {
    run_all_checks("diamond_3x3x3", diamond_precision(FIXTURE_PRECISION), [3, 3, 3]);
}

// ---------------------------------------------------------------------
// `Cell::build_symmetry` (the `basis::build_symmetry` free function) input
// validation — mirrors upstream's refusal-when-not-a-KPoints guard in spirit
// (17-04-PLAN.md's verification list).
// ---------------------------------------------------------------------

#[test]
fn build_symmetry_refuses_mismatched_lengths() {
    let mut cell = pyscf_pbc_gto::test_systems::diamond();
    let input = SymmAdaptedBasisInput {
        kpts_scaled_ibz: vec![[0.0, 0.0, 0.0]],
        little_cogroup_ops: vec![], // length mismatch with kpts_scaled_ibz
        ops: vec![SPGElement::default()],
        dmats: vec![],
    };
    let err = basis::build_symmetry(&mut cell, &input)
        .expect_err("mismatched kpts_scaled_ibz/little_cogroup_ops lengths must be refused");
    assert!(matches!(err, pyscf_pbc_symm::PbcSymmError::KptsSymmInputMismatch(_)));
    assert!(cell.symm_orb.is_none(), "a refused build must not touch cell.symm_orb");
    assert!(cell.irrep_id.is_none(), "a refused build must not touch cell.irrep_id");
}

#[test]
fn build_symmetry_refuses_out_of_range_op_index() {
    let mut cell = pyscf_pbc_gto::test_systems::diamond();
    let sym = Symmetry::build(&cell, true, false, false).expect("Symmetry::build");
    let input = SymmAdaptedBasisInput {
        kpts_scaled_ibz: vec![[0.0, 0.0, 0.0]],
        little_cogroup_ops: vec![vec![sym.ops.len() + 5]], // out of range
        ops: sym.ops.clone(),
        dmats: sym.dmats.clone(),
    };
    let err = basis::build_symmetry(&mut cell, &input)
        .expect_err("an out-of-range little-co-group op index must be refused");
    assert!(matches!(err, pyscf_pbc_symm::PbcSymmError::KptsSymmInputMismatch(_)));
}

#[test]
fn build_symmetry_refuses_ops_dmats_length_mismatch() {
    let mut cell = pyscf_pbc_gto::test_systems::diamond();
    let sym = Symmetry::build(&cell, true, false, false).expect("Symmetry::build");
    let input = SymmAdaptedBasisInput {
        kpts_scaled_ibz: vec![[0.0, 0.0, 0.0]],
        little_cogroup_ops: vec![vec![0]],
        ops: sym.ops.clone(),
        dmats: sym.dmats[..sym.dmats.len() - 1].to_vec(), // one short
    };
    let err = basis::build_symmetry(&mut cell, &input)
        .expect_err("ops/dmats length mismatch must be refused");
    assert!(matches!(err, pyscf_pbc_symm::PbcSymmError::KptsSymmInputMismatch(_)));
}
