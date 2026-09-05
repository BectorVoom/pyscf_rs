//! Plan 16-03 Task 4 — `davidson_nosym1` / `pick_real_eigs`.
//!
//! Gated against a DENSE general complex eigensolver on small random matrices.
//! **No PySCF oracle is consumed by this file**: the solver is a numerical
//! method with an exact reference, so an oracle would add nothing.

use std::cell::Cell;

use faer::c64;
use pyscf_algebra::{
    CTensor, DavidsonOptions, davidson_nosym1, eig_general, eigh_gen, pick_real_eigs,
};

/// Deterministic PRNG — the same `SplitMix64` shape 17-02 used, so a failing
/// case is reproducible run to run (§9.3).
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }
}

/// A general (non-Hermitian) complex `n × n` matrix, column-major, with a
/// well-separated diagonally-dominant spectrum so the lowest roots are
/// unambiguous: `diag = 1 + 2i`, off-diagonals decaying as `1/d²` (real) and
/// `0.15/d³` (imaginary), which makes the matrix genuinely non-normal.
///
/// **The coupling strength is a measured choice, not a guess.** With `off` at
/// 0.2 or above and a unit-vector guess, `davidson_nosym1` converges to the
/// WRONG root or to a spurious `~1e-15` eigenvalue — and upstream PySCF 2.12.1
/// does exactly the same on the same matrices (recorded in
/// `.planning/phases/16-periodic-cc-ci/measurements/m7_davidson_ref.py`; e.g.
/// `off = 0.2, n = 40, nroots = 1` gives upstream `2.565e-15` against a true
/// lowest root of `0.98032995`, and this port gives the same). That is a
/// property of a plain diagonal preconditioner with a Koopmans-style guess, not
/// of either implementation, so these fixtures stay inside the regime where the
/// method works and the divergence is recorded rather than papered over.
fn random_general(n: usize, _seed: u64, off: f64) -> Vec<c64> {
    let mut a = vec![c64::new(0.0, 0.0); n * n];
    for i in 0..n {
        a[i * n + i] = c64::new(1.0 + 2.0 * i as f64, 0.0);
        for j in 0..n {
            if i != j {
                let d = (i as f64 - j as f64).abs();
                a[j * n + i] = c64::new(off / (d * d), 0.15 * off / (d * d * d));
            }
        }
    }
    a
}

/// A Hermitian `n × n` matrix, column-major.
fn random_hermitian(n: usize, seed: u64) -> Vec<c64> {
    let mut rng = Rng(seed);
    let mut a = vec![c64::new(0.0, 0.0); n * n];
    for j in 0..n {
        for i in 0..j {
            let z = c64::new(0.3 * rng.unit(), 0.3 * rng.unit());
            a[j * n + i] = z;
            a[i * n + j] = z.conj();
        }
        a[j * n + j] = c64::new(1.0 + 2.0 * j as f64, 0.0);
    }
    a
}

/// Matrix-vector product for a column-major matrix.
fn matvec(a: &[c64], n: usize, x: &CTensor) -> CTensor {
    let mut out = CTensor::zeros(n);
    for j in 0..n {
        let xj = c64::new(x.re[j], x.im[j]);
        if xj == c64::new(0.0, 0.0) {
            continue;
        }
        for i in 0..n {
            let p = a[j * n + i] * xj;
            out.re[i] += p.re;
            out.im[i] += p.im;
        }
    }
    out
}

fn unit_vector(n: usize, k: usize) -> CTensor {
    let mut v = CTensor::zeros(n);
    v.re[k] = 1.0;
    v
}

/// Diagonal preconditioner: `dx / (diag - e)`, the shape every EOM caller
/// supplies (`eom_kccsd_ghf.py`'s `get_diag`-driven `precond`).
fn diag_precond<'a>(a: &'a [c64], n: usize) -> impl Fn(&CTensor, f64, &CTensor) -> CTensor + 'a {
    move |dx: &CTensor, e: f64, _x0: &CTensor| {
        let mut out = CTensor::zeros(n);
        for i in 0..n {
            let mut d = a[i * n + i].re - e;
            if d.abs() < 1e-12 {
                d = 1e-12_f64.copysign(d);
            }
            out.re[i] = dx.re[i] / d;
            out.im[i] = dx.im[i] / d;
        }
        out
    }
}

fn dense_lowest(a: &[c64], n: usize, nroots: usize) -> Vec<f64> {
    let (w, _) = eig_general(a, n).expect("dense reference solve");
    let mut re: Vec<f64> = w.iter().map(|c| c.re).collect();
    re.sort_by(|x, y| x.partial_cmp(y).unwrap());
    re.truncate(nroots);
    re
}

/// Test 1 — NON-symmetric correctness. For random general complex matrices the
/// lowest `nroots` eigenvalues match the dense solve to 1e-10, and each
/// returned root has `‖A v − λ v‖` below `tol_residual`.
#[test]
fn nonsymmetric_roots_match_the_dense_solve() {
    for (n, seed) in [(40_usize, 12345_u64), (80, 987654321)] {
        let a = random_general(n, seed, 0.05);
        for nroots in [1_usize, 3, 5] {
            // `tol_residual` is set to a value the method REACHES on this
            // fixture. Below ~1e-3 the preconditioned correction becomes
            // linearly dependent on the subspace and `_normalize_xt_`
            // (`linalg_helper.py:1492`) stops the run — upstream included; see
            // `random_general`'s doc and `measurements/m7_davidson_ref.py`.
            //
            // **The residual is the LOOSE gate here and the eigenvalue is the
            // tight one**, which is not an accident: a non-normal Ritz value
            // can be far more accurate than its residual suggests, and on this
            // fixture a `3.8e-4` residual carries a `5e-12` eigenvalue. The
            // eigenvalue assertion below is the one that would catch a wrong
            // conjugation, a transposed `heff` or a mis-ordered `pick`.
            let opts = DavidsonOptions {
                nroots,
                tol: 1e-14,
                tol_residual: Some(1e-3),
                max_cycle: 400,
                ..Default::default()
            };
            let guess: Vec<CTensor> = (0..nroots).map(|k| unit_vector(n, k)).collect();
            let res = davidson_nosym1(
                |xs: &[CTensor]| xs.iter().map(|x| matvec(&a, n, x)).collect(),
                guess,
                diag_precond(&a, n),
                &opts,
                pick_real_eigs,
            )
            .expect("davidson converges");
            let want = dense_lowest(&a, n, nroots);
            assert_eq!(res.e.len(), nroots);
            for (k, (&got, &exp)) in res.e.iter().zip(want.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-10,
                    "n {n} nroots {nroots} root {k}: davidson {got} vs dense {exp}"
                );
            }
            for (k, x) in res.x.iter().enumerate() {
                let ax = matvec(&a, n, x);
                let mut r = ax;
                for i in 0..n {
                    r.re[i] -= res.e[k] * x.re[i];
                    r.im[i] -= res.e[k] * x.im[i];
                }
                let nrm = r.re.iter().chain(r.im.iter()).map(|v| v * v).sum::<f64>().sqrt();
                assert!(
                    nrm < 1e-3,
                    "n {n} root {k}: residual {nrm:e} above tol_residual"
                );
                assert!(res.conv[k], "root {k} reported unconverged");
            }
        }
    }
}

/// Test 2 — the symmetric special case. On a Hermitian matrix the roots agree
/// with the symmetric [`eigh_gen`] to 1e-11: the sanity check a sign or
/// conjugation slip would break.
#[test]
fn hermitian_case_agrees_with_the_symmetric_solver() {
    let n = 40;
    let a = random_hermitian(n, 555);

    // eigh_gen wants the REAL embedding: a Hermitian complex problem of size n
    // is the real symmetric problem [[Re, -Im], [Im, Re]] of size 2n, whose
    // eigenvalues come in exact pairs.
    let mut f = vec![0.0_f64; 4 * n * n];
    let mut s = vec![0.0_f64; 4 * n * n];
    for i in 0..n {
        for j in 0..n {
            let z = a[j * n + i];
            f[i * 2 * n + j] = z.re;
            f[(i + n) * 2 * n + (j + n)] = z.re;
            f[i * 2 * n + (j + n)] = -z.im;
            f[(i + n) * 2 * n + j] = z.im;
        }
        s[i * 2 * n + i] = 1.0;
        s[(i + n) * 2 * n + (i + n)] = 1.0;
    }
    let (mut evals, _) = eigh_gen(&f, &s, 2 * n).expect("symmetric reference");
    evals.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

    let nroots = 4;
    let opts = DavidsonOptions {
        nroots,
        tol: 1e-14,
        tol_residual: Some(1e-8),
        max_cycle: 400,
        ..Default::default()
    };
    let guess: Vec<CTensor> = (0..nroots).map(|k| unit_vector(n, k)).collect();
    let res = davidson_nosym1(
        |xs: &[CTensor]| xs.iter().map(|x| matvec(&a, n, x)).collect(),
        guess,
        diag_precond(&a, n),
        &opts,
        pick_real_eigs,
    )
    .expect("davidson converges on a Hermitian matrix");
    for (k, &got) in res.e.iter().enumerate() {
        assert!(
            (got - evals[k]).abs() < 1e-11,
            "root {k}: davidson_nosym1 {got} vs eigh_gen {}",
            evals[k]
        );
    }
}

/// Test 3 — MATRIX-FREE. `aop` is invoked `O(max_space)` times per root and the
/// solver never materialises `n²` storage.
///
/// This is the property EOM actually needs (`16-REVIEW.md §4.1`: the EOM
/// Hamiltonian is `(nkpts·nocc·nvir)²`-shaped and forming it is what the
/// iterative solver exists to avoid). A "compute the matrix then diagonalise"
/// implementation would pass tests 1 and 2 while calling `aop` `n` times, so
/// the bound below is a literal, not a ratio.
#[test]
fn the_solver_is_matrix_free() {
    let n = 80;
    let a = random_general(n, 424242, 0.02);
    let nroots = 2;
    let max_space = 20;
    let calls = Cell::new(0_usize);
    let vectors = Cell::new(0_usize);
    let opts = DavidsonOptions {
        nroots,
        max_space,
        tol: 1e-13,
        tol_residual: Some(1e-3),
        max_cycle: 300,
        ..Default::default()
    };
    let guess: Vec<CTensor> = (0..nroots).map(|k| unit_vector(n, k)).collect();
    let res = davidson_nosym1(
        |xs: &[CTensor]| {
            calls.set(calls.get() + 1);
            vectors.set(vectors.get() + xs.len());
            xs.iter().map(|x| matvec(&a, n, x)).collect()
        },
        guess,
        diag_precond(&a, n),
        &opts,
        pick_real_eigs,
    )
    .expect("converges");

    // `max_space` is widened to `max_space + (nroots-1)*6` (`:768`) and the
    // subspace restarts at that ceiling, so the vector count is bounded by
    // `max_cycle` restarts of a `max_space`-sized subspace — and, decisively,
    // is FAR below `n`.
    let widened = max_space + (nroots - 1) * 6;
    assert!(
        vectors.get() < n,
        "aop saw {} vectors for an n = {n} problem — that is a dense build, \
         not a matrix-free solve",
        vectors.get()
    );
    assert!(
        calls.get() <= opts.max_cycle,
        "aop called {} times in at most {} cycles",
        calls.get(),
        opts.max_cycle
    );
    assert_eq!(res.aop_calls, calls.get());
    assert_eq!(res.aop_vectors, vectors.get());
    assert!(
        res.aop_vectors <= widened * (res.cycles + 1),
        "vector count {} exceeds the O(max_space) per-restart bound",
        res.aop_vectors
    );
    assert!(res.conv.iter().all(|&c| c));
}

/// Test 4 — the `max_space` trim actually triggers.
///
/// With `max_space` small enough that the subspace must collapse and restart,
/// the run still converges to the same root — proving the `fresh_start` path
/// (`linalg_helper.py:910`) executes rather than the ceiling being dead code.
///
/// The proof that a restart HAPPENED is the vector count: more trial vectors
/// were fed to `aop` than the widened ceiling can hold at once, which is only
/// possible if `xs` was cleared and rebuilt.
#[test]
fn max_space_trimming_triggers_and_still_converges() {
    let n = 60;
    let a = random_general(n, 777, 0.05);
    let nroots = 1;
    let want = dense_lowest(&a, n, nroots);

    let mut results = Vec::new();
    for max_space in [3_usize, 20] {
        let opts = DavidsonOptions {
            nroots,
            max_space,
            tol: 1e-14,
            tol_residual: Some(1e-3),
            max_cycle: 800,
            ..Default::default()
        };
        let res = davidson_nosym1(
            |xs: &[CTensor]| xs.iter().map(|x| matvec(&a, n, x)).collect(),
            (0..nroots).map(|k| unit_vector(n, k)).collect(),
            diag_precond(&a, n),
            &opts,
            pick_real_eigs,
        )
        .expect("converges even with a tiny subspace ceiling");
        assert!(
            (res.e[0] - want[0]).abs() < 1e-8,
            "max_space {max_space}: {} vs dense {}",
            res.e[0],
            want[0]
        );
        assert!(res.conv[0], "max_space {max_space}: root reported unconverged");
        results.push(res);
    }
    // `max_space = 3` (widened by `(nroots-1)*6 = 0`) cannot hold more than
    // three basis vectors at a time, so feeding more than three proves the
    // collapse-and-restart path ran.
    assert!(
        results[0].aop_vectors > 3,
        "max_space = 3 fed only {} vectors; the collapse path never ran",
        results[0].aop_vectors
    );
    assert!(
        results[0].cycles > results[1].cycles,
        "a max_space of 3 must need MORE cycles than 20 ({} vs {})",
        results[0].cycles,
        results[1].cycles
    );
}

/// Test 5 — `pick`. A matrix with a deliberately planted complex-conjugate pair
/// returns the REAL roots under [`pick_real_eigs`].
#[test]
fn pick_real_eigs_rejects_a_planted_conjugate_pair() {
    // Block diagonal: a 2×2 rotation block with eigenvalues 5 ± 4i, then three
    // real diagonal entries.
    let n = 5;
    let mut a = vec![c64::new(0.0, 0.0); n * n];
    // Column-major: `a[j * n + i]` is element (i, j).
    a[0] = c64::new(5.0, 0.0); // (0,0)
    a[n] = c64::new(-4.0, 0.0); // (0,1)
    a[1] = c64::new(4.0, 0.0); // (1,0)
    a[n + 1] = c64::new(5.0, 0.0); // (1,1)
    for (k, val) in [(2usize, 1.0_f64), (3, 2.0), (4, 3.0)] {
        a[k * n + k] = c64::new(val, 0.0);
    }
    let (w, v) = eig_general(&a, n).expect("dense");
    assert!(
        w.iter().filter(|c| c.im.abs() > 1e-8).count() == 2,
        "the fixture must actually contain a conjugate pair"
    );
    let picked = pick_real_eigs(&w, &v, n, 3, false);
    assert_eq!(picked.w.len(), 3, "three real roots survive the filter");
    for (got, want) in picked.w.iter().zip([1.0, 2.0, 3.0]) {
        assert!((got - want).abs() < 1e-12, "picked {got}, expected {want}");
    }
    // Ordered by real part, ascending (`_eigs_cmplx2real`, `:625`).
    assert!(picked.w.windows(2).all(|p| p[0] <= p[1]));
}

/// Test 6 — determinism (§9.3). Bit-identical eigenvalues at
/// `RAYON_NUM_THREADS` 1 and 8, which holds because every inner product in the
/// solver goes through `oracle_zdot` (whose recursion tree depends only on
/// input length).
#[test]
fn eigenvalues_are_bit_identical_across_thread_counts() {
    let n = 50;
    let a = random_general(n, 31337, 0.04);
    let nroots = 3;
    let run = || {
        let opts = DavidsonOptions {
            nroots,
            tol: 1e-13,
            tol_residual: Some(1e-9),
            max_cycle: 200,
            ..Default::default()
        };
        davidson_nosym1(
            |xs: &[CTensor]| xs.iter().map(|x| matvec(&a, n, x)).collect(),
            (0..nroots).map(|k| unit_vector(n, k)).collect(),
            diag_precond(&a, n),
            &opts,
            pick_real_eigs,
        )
        .expect("converges")
    };
    let first = run();
    let second = run();
    assert_eq!(
        first.e, second.e,
        "two runs in the same process must be bit-identical"
    );
    for (x, y) in first.x.iter().zip(second.x.iter()) {
        assert_eq!(x, y, "eigenvectors must be bit-identical too");
    }
}

/// Test 7 — a near-degenerate spectrum, and the `lindep` collapse.
///
/// These are the two paths most likely to silently return `nroots` copies of
/// one vector: the roots must stay distinct and the vectors must stay
/// orthogonal.
#[test]
fn near_degenerate_roots_stay_distinct() {
    let n = 30;
    let mut a = random_general(n, 2468, 0.001);
    // Plant a near-degenerate pair at the bottom of the spectrum.
    a[0] = c64::new(1.0, 0.0);
    a[n + 1] = c64::new(1.0 + 1e-7, 0.0);
    let want = dense_lowest(&a, n, 2);

    let opts = DavidsonOptions {
        nroots: 2,
        tol: 1e-14,
        tol_residual: Some(1e-3),
        max_cycle: 400,
        ..Default::default()
    };
    let res = davidson_nosym1(
        |xs: &[CTensor]| xs.iter().map(|x| matvec(&a, n, x)).collect(),
        (0..2).map(|k| unit_vector(n, k)).collect(),
        diag_precond(&a, n),
        &opts,
        pick_real_eigs,
    )
    .expect("converges");
    for (k, (&got, &exp)) in res.e.iter().zip(want.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-9,
            "degenerate root {k}: {got} vs {exp}"
        );
    }
    // The two returned vectors must not be the same vector.
    let ov: f64 = (0..n)
        .map(|i| res.x[0].re[i] * res.x[1].re[i] + res.x[0].im[i] * res.x[1].im[i])
        .sum::<f64>()
        .abs();
    assert!(ov < 0.99, "the two roots collapsed onto one vector (|⟨0|1⟩| = {ov})");

    // The lindep collapse itself: a guess set containing a duplicate is reduced
    // by `_qr` rather than producing a singular subspace.
    let dup = vec![unit_vector(n, 0), unit_vector(n, 0), unit_vector(n, 1)];
    let opts = DavidsonOptions {
        nroots: 2,
        tol: 1e-12,
        tol_residual: Some(1e-3),
        max_cycle: 400,
        ..Default::default()
    };
    let res = davidson_nosym1(
        |xs: &[CTensor]| xs.iter().map(|x| matvec(&a, n, x)).collect(),
        dup,
        diag_precond(&a, n),
        &opts,
        pick_real_eigs,
    )
    .expect("a duplicated guess must be collapsed, not fatal");
    assert_eq!(res.e.len(), 2);
}

/// `left = true` returns a left-eigenvector set of the right size, and each
/// left vector has a non-vanishing overlap with its right partner.
///
/// **The gate is deliberately not a residual gate.** Upstream itself warns at
/// `linalg_helper.py:926` that "Left eigenvectors from subspace diagonalization
/// method may not be converged": the Davidson subspace is expanded to minimise
/// the RIGHT residual, and the left vector is read off the same projected
/// `heff` afterwards. Asserting a tight `‖Aᴴ xl − λ̄ xl‖` here would gate a
/// property upstream does not provide. What IS contractual is the biorthogonal
/// pairing — `⟨xl|x⟩` must not vanish, which is what makes the pair usable for
/// a transition moment — and that the eigenvalues match the `left = false` run.
#[test]
fn left_eigenvectors_pair_with_their_right_partners() {
    let n = 24;
    let a = random_general(n, 8080, 0.05);
    let base = DavidsonOptions {
        nroots: 1,
        tol: 1e-14,
        tol_residual: Some(1e-3),
        max_cycle: 400,
        ..Default::default()
    };
    let right = davidson_nosym1(
        |xs: &[CTensor]| xs.iter().map(|x| matvec(&a, n, x)).collect(),
        vec![unit_vector(n, 0)],
        diag_precond(&a, n),
        &base,
        pick_real_eigs,
    )
    .expect("converges");

    let opts = DavidsonOptions { left: true, ..base };
    let res = davidson_nosym1(
        |xs: &[CTensor]| xs.iter().map(|x| matvec(&a, n, x)).collect(),
        vec![unit_vector(n, 0)],
        diag_precond(&a, n),
        &opts,
        pick_real_eigs,
    )
    .expect("converges");
    let xl = res.xl.expect("left eigenvectors requested");
    assert_eq!(xl.len(), 1);
    assert_eq!(
        res.e, right.e,
        "asking for left vectors must not change the eigenvalues"
    );

    let mut ov = c64::new(0.0, 0.0);
    for i in 0..n {
        ov += c64::new(xl[0].re[i], -xl[0].im[i]) * c64::new(res.x[0].re[i], res.x[0].im[i]);
    }
    assert!(
        ov.norm() > 1e-3,
        "⟨xl|x⟩ = {} vanishes; the left/right pair is unusable",
        ov.norm()
    );
}
