//! Plan 09-02 Task 8 — `zeigh_gen` / `zcholesky` / `zsolve_linear` (D-PBC-04,
//! PBC-MASTER-PLAN §5.3).
//!
//! Verified here:
//!   * random 8x8 Hermitian with `S = I` — eigenvalues match the independent
//!     `2n x 2n` real-embedding reference to 1e-12, `Cᴴ H C` is diagonal to
//!     1e-12, and `Cᴴ S C = I`;
//!   * repeated calls return bit-identical coefficients (stable phases);
//!   * a non-trivial positive-definite `S` (the actual PBC k-point shape);
//!   * both `zcholesky` routes agree and reproduce `A = L·Lᴴ`;
//!   * both `zsolve_linear` routes agree and satisfy `A·z = b`.
//!
//! Not verified here: degenerate spectra beyond the small explicit case below;
//! linear-dependency removal thresholds (inherited from `eigh_gen`).

use pyscf_algebra::{
    CTensor, zcholesky, zcholesky_crout, zcholesky_faer, zeigh_gen, zeigh_gen_embedding,
    zeigh_gen_faer, zsolve_linear, zsolve_linear_embedding, zsolve_linear_faer,
};

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
        u * 2.0 - 1.0
    }
}

const TOL: f64 = 1e-12;

/// Random Hermitian `n x n`, row-major: `H = (X + Xᴴ)/2`.
fn random_hermitian(rng: &mut Lcg, n: usize) -> CTensor {
    let re: Vec<f64> = (0..n * n).map(|_| rng.next_f64()).collect();
    let im: Vec<f64> = (0..n * n).map(|_| rng.next_f64()).collect();
    let mut h = CTensor::zeros(n * n);
    for i in 0..n {
        for j in 0..n {
            h.re[i * n + j] = 0.5 * (re[i * n + j] + re[j * n + i]);
            h.im[i * n + j] = 0.5 * (im[i * n + j] - im[j * n + i]);
        }
        h.im[i * n + i] = 0.0;
    }
    h
}

/// Hermitian positive-definite `n x n`: `A = XᴴX + n·I`.
fn random_hpd(rng: &mut Lcg, n: usize) -> CTensor {
    let x = {
        let re: Vec<f64> = (0..n * n).map(|_| rng.next_f64()).collect();
        let im: Vec<f64> = (0..n * n).map(|_| rng.next_f64()).collect();
        CTensor::from_planes(re, im)
    };
    let mut a = CTensor::zeros(n * n);
    for i in 0..n {
        for j in 0..n {
            let (mut ar, mut ai) = (0.0_f64, 0.0_f64);
            for k in 0..n {
                // conj(X[k][i]) * X[k][j]
                let (xr, xi) = (x.re[k * n + i], -x.im[k * n + i]);
                let (yr, yi) = (x.re[k * n + j], x.im[k * n + j]);
                ar += xr * yr - xi * yi;
                ai += xr * yi + xi * yr;
            }
            a.re[i * n + j] = ar;
            a.im[i * n + j] = ai;
        }
        a.re[i * n + i] += n as f64;
        a.im[i * n + i] = 0.0;
    }
    a
}

fn identity(n: usize) -> CTensor {
    let mut s = CTensor::zeros(n * n);
    for i in 0..n {
        s.re[i * n + i] = 1.0;
    }
    s
}

/// `Cᴴ M C` for row-major `M` and F-order `C`, both `n x n`.
fn ch_m_c(m: &CTensor, c: &CTensor, n: usize) -> CTensor {
    // t = M · C  (t is F-order: t[i + j*n])
    let mut t = CTensor::zeros(n * n);
    for j in 0..n {
        for i in 0..n {
            let (mut ar, mut ai) = (0.0_f64, 0.0_f64);
            for k in 0..n {
                let (mr, mi) = (m.re[i * n + k], m.im[i * n + k]);
                let (cr, ci) = (c.re[k + j * n], c.im[k + j * n]);
                ar += mr * cr - mi * ci;
                ai += mr * ci + mi * cr;
            }
            t.re[i + j * n] = ar;
            t.im[i + j * n] = ai;
        }
    }
    // out[a][b] = Σ_i conj(C[i + a*n]) * t[i + b*n]
    let mut out = CTensor::zeros(n * n);
    for a in 0..n {
        for b in 0..n {
            let (mut ar, mut ai) = (0.0_f64, 0.0_f64);
            for i in 0..n {
                let (ur, ui) = (c.re[i + a * n], -c.im[i + a * n]);
                let (vr, vi) = (t.re[i + b * n], t.im[i + b * n]);
                ar += ur * vr - ui * vi;
                ai += ur * vi + ui * vr;
            }
            out.re[a * n + b] = ar;
            out.im[a * n + b] = ai;
        }
    }
    out
}

fn max_off_diagonal(m: &CTensor, n: usize) -> f64 {
    let mut worst = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            if i != j {
                worst = worst.max(m.re[i * n + j].abs()).max(m.im[i * n + j].abs());
            }
        }
    }
    worst
}

#[test]
fn zeigh_gen_random_8x8_hermitian_with_identity_overlap() {
    let n = 8;
    let mut rng = Lcg::new(0x0902_0501);
    let h = random_hermitian(&mut rng, n);
    let s = identity(n);

    let (evals, c) = zeigh_gen(&h, &s, n).expect("zeigh_gen");
    assert_eq!(evals.len(), n);
    assert_eq!(c.len(), n * n);

    // (a) Eigenvalues match the independent real 2n x 2n embedding reference.
    let (ref_vals, _) = zeigh_gen_embedding(&h, &s, n).expect("zeigh_gen_embedding");
    for (i, (got, want)) in evals.iter().zip(ref_vals.iter()).enumerate() {
        assert!(
            (got - want).abs() < TOL,
            "eigenvalue {i}: faer c64 {got} vs 2n x 2n embedding {want} (tol {TOL:e})"
        );
    }
    // Ascending.
    for i in 1..n {
        assert!(evals[i - 1] <= evals[i], "eigenvalues are not ascending");
    }

    // (b) Cᴴ H C is diagonal, with the eigenvalues on the diagonal.
    let d = ch_m_c(&h, &c, n);
    let off = max_off_diagonal(&d, n);
    assert!(off < TOL, "Cᴴ H C max off-diagonal = {off:e} (tol {TOL:e})");
    for (i, eps) in evals.iter().enumerate() {
        assert!(
            (d.re[i * n + i] - eps).abs() < TOL,
            "diagonal {i} = {} but eigenvalue = {eps}",
            d.re[i * n + i]
        );
        assert!(d.im[i * n + i].abs() < TOL);
    }

    // (c) Cᴴ S C = I.
    let ovlp = ch_m_c(&s, &c, n);
    assert!(max_off_diagonal(&ovlp, n) < TOL);
    for i in 0..n {
        assert!((ovlp.re[i * n + i] - 1.0).abs() < TOL);
        assert!(ovlp.im[i * n + i].abs() < TOL);
    }
}

#[test]
fn zeigh_gen_repeated_calls_are_bit_identical() {
    let n = 8;
    let mut rng = Lcg::new(0x0902_0502);
    let h = random_hermitian(&mut rng, n);
    let s = identity(n);

    let (v1, c1) = zeigh_gen(&h, &s, n).expect("zeigh_gen");
    for _ in 0..3 {
        let (v2, c2) = zeigh_gen(&h, &s, n).expect("zeigh_gen");
        for i in 0..n {
            assert_eq!(v1[i].to_bits(), v2[i].to_bits(), "eigenvalue {i} drifted");
        }
        for i in 0..n * n {
            assert_eq!(c1.re[i].to_bits(), c2.re[i].to_bits(), "C.re[{i}] drifted");
            assert_eq!(c1.im[i].to_bits(), c2.im[i].to_bits(), "C.im[{i}] drifted");
        }
    }
}

/// D-PBC-04: the two routes are independent implementations (native faer `c64`
/// vs a real `2n x 2n` embedding through the molecular `eigh_gen`) and must
/// agree on eigenvalues. Eigenvectors are only defined up to a global phase, so
/// the vector-level check is the route-independent `Cᴴ H C` diagonality.
#[test]
fn both_zeigh_routes_agree_with_non_trivial_overlap() {
    let n = 6;
    let mut rng = Lcg::new(0x0902_0503);
    let h = random_hermitian(&mut rng, n);
    let s = random_hpd(&mut rng, n);

    let (v_faer, c_faer) = zeigh_gen_faer(&h, &s, n).expect("zeigh_gen_faer");
    let (v_emb, c_emb) = zeigh_gen_embedding(&h, &s, n).expect("zeigh_gen_embedding");

    for i in 0..n {
        assert!(
            (v_faer[i] - v_emb[i]).abs() < 1e-11 * (1.0 + v_faer[i].abs()),
            "eigenvalue {i}: faer {} vs embedding {}",
            v_faer[i],
            v_emb[i]
        );
    }
    for c in [&c_faer, &c_emb] {
        let d = ch_m_c(&h, c, n);
        assert!(max_off_diagonal(&d, n) < 1e-11, "Cᴴ H C not diagonal");
        let o = ch_m_c(&s, c, n);
        assert!(max_off_diagonal(&o, n) < 1e-11, "Cᴴ S C not diagonal");
        for i in 0..n {
            assert!((o.re[i * n + i] - 1.0).abs() < 1e-11, "Cᴴ S C not unit");
        }
    }
}

/// The mandated embedding route must survive a genuinely degenerate spectrum —
/// `H = diag(1, 1, 3)` has a doubly-degenerate eigenvalue, which multiplies to
/// four-fold in the `2n x 2n` embedding.
#[test]
fn zeigh_embedding_handles_a_degenerate_eigenvalue() {
    let n = 3;
    let mut h = CTensor::zeros(n * n);
    h.re[0] = 1.0;
    h.re[4] = 1.0;
    h.re[8] = 3.0;
    let s = identity(n);

    let (evals, c) = zeigh_gen_embedding(&h, &s, n).expect("zeigh_gen_embedding");
    assert!((evals[0] - 1.0).abs() < TOL);
    assert!((evals[1] - 1.0).abs() < TOL);
    assert!((evals[2] - 3.0).abs() < TOL);
    // The three vectors must still be S-orthonormal — the degeneracy fallback's
    // whole purpose.
    let o = ch_m_c(&s, &c, n);
    assert!(
        max_off_diagonal(&o, n) < 1e-10,
        "degenerate set not orthogonal"
    );
    for i in 0..n {
        assert!((o.re[i * n + i] - 1.0).abs() < 1e-10);
    }
}

#[test]
fn zcholesky_reproduces_a_and_both_routes_agree() {
    let n = 7;
    let mut rng = Lcg::new(0x0902_0504);
    let a = random_hpd(&mut rng, n);

    let l = zcholesky(&a, n).expect("zcholesky");
    let l_crout = zcholesky_crout(&a, n).expect("zcholesky_crout");
    let l_faer = zcholesky_faer(&a, n).expect("zcholesky_faer");

    // Strictly-upper triangle is zero in both routes.
    for i in 0..n {
        for j in i + 1..n {
            assert_eq!(l.re[i * n + j], 0.0);
            assert_eq!(l.im[i * n + j], 0.0);
            assert_eq!(l_crout.re[i * n + j], 0.0);
        }
    }

    // L·Lᴴ == A.
    for l in [&l_crout, &l_faer] {
        let mut worst = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                let (mut ar, mut ai) = (0.0_f64, 0.0_f64);
                for k in 0..n {
                    let (xr, xi) = (l.re[i * n + k], l.im[i * n + k]);
                    let (yr, yi) = (l.re[j * n + k], -l.im[j * n + k]);
                    ar += xr * yr - xi * yi;
                    ai += xr * yi + xi * yr;
                }
                worst = worst
                    .max((ar - a.re[i * n + j]).abs())
                    .max((ai - a.im[i * n + j]).abs());
            }
        }
        assert!(worst < 1e-10, "L·Lᴴ - A max = {worst:e}");
    }

    // The Cholesky factor of an HPD matrix with a real positive diagonal is
    // unique, so the two independent routes must agree closely.
    let d = (0..n * n)
        .map(|i| {
            (l_crout.re[i] - l_faer.re[i])
                .abs()
                .max((l_crout.im[i] - l_faer.im[i]).abs())
        })
        .fold(0.0_f64, f64::max);
    assert!(d < 1e-10, "Crout vs faer c64 Cholesky max|diff| = {d:e}");
}

#[test]
fn zcholesky_rejects_non_positive_definite_input() {
    let n = 2;
    let mut a = CTensor::zeros(n * n);
    a.re[0] = -1.0;
    a.re[3] = 1.0;
    assert!(zcholesky_crout(&a, n).is_err());
    assert!(zcholesky(&a, n).is_err());
}

#[test]
fn zsolve_linear_satisfies_the_system_on_both_routes() {
    let n = 9;
    let mut rng = Lcg::new(0x0902_0505);
    let a = random_hpd(&mut rng, n); // well-conditioned, not required to be HPD
    let b = CTensor::from_planes(
        (0..n).map(|_| rng.next_f64()).collect(),
        (0..n).map(|_| rng.next_f64()).collect(),
    );

    let z_faer = zsolve_linear_faer(&a, &b, n).expect("zsolve_linear_faer");
    let z_emb = zsolve_linear_embedding(&a, &b, n).expect("zsolve_linear_embedding");
    let z = zsolve_linear(&a, &b, n).expect("zsolve_linear");

    for z in [&z_faer, &z_emb, &z] {
        let mut worst = 0.0_f64;
        for i in 0..n {
            let (mut ar, mut ai) = (0.0_f64, 0.0_f64);
            for j in 0..n {
                let (xr, xi) = (a.re[i * n + j], a.im[i * n + j]);
                let (yr, yi) = (z.re[j], z.im[j]);
                ar += xr * yr - xi * yi;
                ai += xr * yi + xi * yr;
            }
            worst = worst.max((ar - b.re[i]).abs()).max((ai - b.im[i]).abs());
        }
        assert!(worst < 1e-10, "residual max = {worst:e}");
    }

    let d = (0..n)
        .map(|i| {
            (z_faer.re[i] - z_emb.re[i])
                .abs()
                .max((z_faer.im[i] - z_emb.im[i]).abs())
        })
        .fold(0.0_f64, f64::max);
    assert!(d < 1e-10, "faer c64 vs 2n x 2n embedding max|diff| = {d:e}");
}

#[test]
fn zeigh_family_rejects_bad_shapes() {
    let a = CTensor::zeros(4);
    let b = CTensor::zeros(2);
    assert!(zeigh_gen(&a, &a, 3).is_err());
    assert!(zeigh_gen_embedding(&a, &a, 3).is_err());
    assert!(zcholesky(&a, 3).is_err());
    assert!(zsolve_linear(&a, &b, 3).is_err());
    assert!(
        zsolve_linear(&a, &a, 2).is_err(),
        "b must be a length-n vector"
    );
}
