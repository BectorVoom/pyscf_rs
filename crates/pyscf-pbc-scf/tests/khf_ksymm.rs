//! `KsymAdaptedKrhf` — plan 17-07.
//!
//! # What is oracle-free here, and why that is the strong test
//!
//! `use_ao_symmetry = true` block-diagonalises the Fock matrix by irrep and
//! solves each block separately. By Schur's lemma that is **exact, not an
//! approximation** — `S` and `F` both commute with every little-co-group
//! operation, so they have no matrix elements between distinct irreps. The two
//! branches must therefore agree to solver precision, and *disagreeing* is a
//! defect rather than a tolerance question.
//!
//! Two routes to the same number inside one process is a stronger test than
//! either against a third implementation — the same idiom 17-10 used for the
//! MO-factorised `get_k_kpts` and 17-07 Task 6 uses for the fast `get_jk`.
//!
//! **17-CONTEXT §3.1: `mo_coeff` is NEVER compared elementwise.** It is
//! defined only up to a unitary rotation within each degenerate subspace, and
//! every symmetric cell has degeneracies. Compare eigenvalues, and compare
//! coefficients only through the density matrix or the total energy they
//! produce.

use pyscf_pbc_gto::test_systems::si;
use pyscf_pbc_gto::{Cell, make_kpts_default};
use pyscf_pbc_scf::{KInitGuess, KScfConfig, KsymAdaptedKrhf};
use pyscf_pbc_symm::basis::{self, SymmAdaptedBasisInput};
use pyscf_pbc_symm::kpts::{KPoints, make_kpts};

/// Eigenvalue agreement between the two `eig` routes. Schur's lemma says the
/// block-diagonalisation is exact, so this is a solver-precision comparison.
const EIG_TOL: f64 = 1e-10;

/// `time_reversal_symmetry = false`, and that is an UPSTREAM limitation, not
/// a convenience.
///
/// `little_cogroup_ops` is filled from `np.where(kpts.k2opk[ki] == ki)[0]`
/// (`kpts.py:112`) — indices into `k2opk`'s **columns**, of which there are
/// `nop * (time_reversal + 1)`. But its consumer indexes the op list directly:
/// `kpts.ops[iop]` (`basis.py:113`, and this port's
/// `basis::symm_adapted_basis`). With time reversal on, `k2opk` has `2*nop`
/// columns and the second half is reachable — at Γ and at every other
/// time-reversal-invariant momentum, `-op·k == k` matches too — so upstream
/// would raise `IndexError` there. This port refuses with a typed
/// `KptsSymmInputMismatch` instead (`little_cogroup_ops[0] references op index
/// 24, but ops has 24 entries`), which is how the mismatch was found.
///
/// Plan 17-05 already recorded this: *"`little_cogroups` refuses where
/// upstream raises `IndexError` (time-reversal + `k2opk`'s `2*nop`
/// columns)."* So `use_ao_symmetry = true` is exercised on the space-group
/// fold alone, which is the part `symm_adapted_basis` actually consumes.
/// Reconciling the two index spaces is carried over — see `17-07-SUMMARY.md`.
const TIME_REVERSAL: bool = false;

fn build(mesh: [usize; 3]) -> (Cell, KPoints) {
    let cell = si();
    let kpts_abs = make_kpts_default(&cell, mesh).expect("make_kpts_default");
    let kpts = make_kpts(&cell, &kpts_abs, true, TIME_REVERSAL).expect("make_kpts");

    // 17-04's symmetry-adapted basis, which `use_ao_symmetry = true` reads.
    let mut cell = cell;
    let input = SymmAdaptedBasisInput {
        kpts_scaled_ibz: kpts.kpts_scaled_ibz.clone(),
        little_cogroup_ops: kpts.little_cogroup_ops.clone(),
        ops: kpts.symmetry.ops.clone(),
        dmats: kpts.symmetry.dmats.clone(),
    };
    basis::build_symmetry(&mut cell, &input).expect("build_symmetry");
    (cell, kpts)
}

fn adapter(cell: &Cell, kpts: &KPoints, use_ao_symmetry: bool) -> KsymAdaptedKrhf {
    // The DF object is built over the FULL BZ — the reference `get_veff`
    // route calls it there, while the one-electron hooks pass `kpts_ibz`.
    let with_df = pyscf_pbc_df::Fftdf::new(cell.clone(), &kpts.kpts).expect("Fftdf");
    let mut mf = KsymAdaptedKrhf::from_df(Box::new(with_df), kpts.clone());
    mf.use_ao_symmetry = use_ao_symmetry;
    mf
}

fn cfg() -> KScfConfig {
    KScfConfig {
        conv_tol: 1e-10,
        conv_tol_grad: Some(1e-8),
        max_cycle: 50,
        init_guess: KInitGuess::Minao,
        ..KScfConfig::default()
    }
}

/// The IBZ k-set is what the driver actually iterates: `kpts()` must return
/// `kpts_ibz`, and for `si [2,2,2]` that is strictly fewer than the 8 BZ
/// points. If this ever returns the full BZ the adapter has silently become a
/// plain `Krhf` and every other test here would still pass.
#[test]
fn kpts_hook_returns_the_ibz_set() {
    use pyscf_pbc_scf::KOverrideHooks;
    let (cell, kpts) = build([2, 2, 2]);
    let mf = adapter(&cell, &kpts, false);
    assert_eq!(kpts.nkpts(), 8, "si [2,2,2] has 8 BZ points");
    assert_eq!(
        mf.kpts().len(),
        kpts.nkpts_ibz(),
        "the driver must iterate the IBZ set"
    );
    assert!(
        mf.kpts().len() < kpts.nkpts(),
        "si [2,2,2] must fold: got {} IBZ of {} BZ points",
        mf.kpts().len(),
        kpts.nkpts()
    );
    // The electron count is a FULL-BZ quantity (17-CONTEXT §3.4) even though
    // the k-loop is over the IBZ — the single most consequential asymmetry in
    // this adapter.
    assert_eq!(
        mf.nelectron(),
        cell.tot_electrons(kpts.nkpts()),
        "nelectron must count over the BZ, not the IBZ"
    );
}

/// `weights_ibz` sums to 1 and reproduces each star's multiplicity. An
/// `energy_elec` written against `1/nkpts` instead would silently drop every
/// star multiplicity — invisible on any cell whose stars are all the same
/// size, which is why this asserts the multiplicities themselves.
#[test]
fn ibz_weights_carry_the_star_multiplicities() {
    let (_cell, kpts) = build([2, 2, 2]);
    let total: f64 = kpts.weights_ibz.iter().sum();
    assert!(
        (total - 1.0).abs() < 1e-15,
        "weights_ibz must sum to 1, got {total:e}"
    );
    let nkpts = kpts.nkpts() as f64;
    for (i, star) in kpts.stars.iter().enumerate() {
        let expect = star.len() as f64 / nkpts;
        assert!(
            (kpts.weights_ibz[i] - expect).abs() < 1e-15,
            "weights_ibz[{i}] = {:e} but star has {} members of {nkpts}",
            kpts.weights_ibz[i],
            star.len()
        );
    }
}

/// The two `eig` routes must agree. Eigenvalues are compared directly;
/// `mo_coeff` is compared only through the converged total energy (§3.1).
#[test]
fn ao_symmetry_eig_matches_the_plain_route() {
    let (cell, kpts) = build([2, 2, 2]);

    let plain = adapter(&cell, &kpts, false).kernel(&cfg()).expect("plain SCF");
    let symm = adapter(&cell, &kpts, true).kernel(&cfg()).expect("symm SCF");

    assert!(plain.converged, "plain-eig SCF did not converge");
    assert!(symm.converged, "symmetry-adapted SCF did not converge");

    let de = (plain.e_tot - symm.e_tot).abs();
    println!("e_tot plain = {:.15}", plain.e_tot);
    println!("e_tot symm  = {:.15}", symm.e_tot);
    println!("|dE| = {de:e}");
    assert!(
        de < EIG_TOL,
        "the two eig routes disagree on e_tot by {de:e} (> {EIG_TOL:e}); \
         Schur's lemma makes the block-diagonalisation exact, so a difference \
         here is a defect, not a tolerance question"
    );

}

/// The `eig` algebra itself, isolated from SCF convergence.
///
/// **This runs ONE SCF and calls both `eig` routes on its converged Fock and
/// overlap.** Comparing two *independently converged* SCFs instead would
/// measure convergence noise, not algebra — plan 17-05 states the rule for
/// Gate B in as many words (*"never two SCFs, because then the residual is
/// convergence noise, not algebra"*), and doing it the wrong way first here
/// produced a 4.4e-9 eigenvalue spread against a 1.7e-11 total-energy
/// agreement, which is that noise and nothing else.
#[test]
fn ao_symmetry_eig_matches_the_plain_route_on_identical_inputs() {
    use pyscf_pbc_scf::KOverrideHooks;

    let (cell, kpts) = build([2, 2, 2]);
    let plain = adapter(&cell, &kpts, false);
    let symm = adapter(&cell, &kpts, true);

    // ONE SCF. Its converged density defines the Fock both routes diagonalise.
    let r = plain.kernel(&cfg()).expect("SCF");
    assert!(r.converged, "SCF did not converge");

    let s1e = plain.get_ovlp().expect("get_ovlp");
    let h1e = plain.get_hcore().expect("get_hcore");
    let vhf = plain.get_veff(&r.dm).expect("get_veff");
    let fock: Vec<Vec<_>> = vec![
        h1e.iter()
            .zip(vhf[0].iter())
            .map(|(h, v)| pyscf_algebra::CTensor {
                re: h.re.iter().zip(v.re.iter()).map(|(a, b)| a + b).collect(),
                im: h.im.iter().zip(v.im.iter()).map(|(a, b)| a + b).collect(),
            })
            .collect(),
    ];

    let (e_plain, _c_plain) = plain.eig(&fock, &s1e).expect("plain eig");
    let (e_symm, _c_symm) = symm.eig(&fock, &s1e).expect("symmetry-adapted eig");

    // Report the MAXIMUM residual over all k and all orbitals, not the first
    // violation — 17-04-MEASUREMENT.md records why that distinction mattered
    // materially in this phase.
    let mut worst = 0.0_f64;
    let mut worst_at = (0usize, 0usize);
    for (k, (a, b)) in e_plain.iter().zip(e_symm.iter()).enumerate() {
        assert_eq!(a.len(), b.len(), "orbital count differs at IBZ k = {k}");
        for (p, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            let d = (x - y).abs();
            if d > worst {
                worst = d;
                worst_at = (k, p);
            }
        }
    }
    println!(
        "max |d eigenvalue| = {worst:e} at IBZ k = {}, orbital {}",
        worst_at.0, worst_at.1
    );
    assert!(
        worst < EIG_TOL,
        "the two eig routes disagree by {worst:e} (> {EIG_TOL:e}) on IDENTICAL \
         inputs, at IBZ k = {}, orbital {}. Schur's lemma makes the \
         block-diagonalisation exact, so this is a defect — most likely a \
         transposed symm_orb block or a wrong irrep grouping — not a tolerance \
         question.",
        worst_at.0,
        worst_at.1
    );
}
