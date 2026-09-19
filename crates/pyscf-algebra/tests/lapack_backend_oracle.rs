//! Differential oracle tests for [`pyscf_algebra::lapack_backend`] — the
//! `lapack_rs` (OpenBLAS-track DSYEVD/ZHEEVD) bridge that replaced faer
//! `SelfAdjointEigen` on the standard-eigh paths.
//!
//! Strategy: assert each routine's DEFINING PROPERTY with plain row-major
//! `Vec<f64>` reference math (no LAPACK/faer call), over deterministic
//! LCG-generated symmetric/Hermitian inputs at several sizes:
//!
//!   * `sym_eigh`: `A·V = V·diag(w)`, `Vᵀ·V = I`, eigenvalues ascending.
//!   * `hermitian_eigh`: `A·V = V·diag(w)`, `Vᴴ·V = I`, eigenvalues ascending.
//!   * `eigh_gen` (generalized, L\"owdin glue over the new backend):
//!     `F·c_j = ε_j·S·c_j` per kept column, eigenvalues nondecreasing.
//!   * `zeigh_gen` (complex generalized): `F·c_j = ε_j·S·c_j` per kept column.
//!
//! Tolerances are numeric (1e-9), not bitwise: DSYEVD and faer EVD agree only
//! to round-off.

use pyscf_algebra::{CTensor, eigh_gen, hermitian_eigh, sym_eigh, zeigh_gen};

/// Deterministic LCG (Knuth/MMIX constants) → reproducible inputs without
/// pulling in the `rand` crate.
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
        let u = (self.0 >> 11) as f64 / (1u64 << 53) as f64; // [0, 1)
        u * 2.0 - 1.0 // [-1, 1)
    }
}

const TOL: f64 = 1e-9;
const SIZES: &[usize] = &[1, 2, 3, 5, 8];

/// Symmetric `A = M·Mᵀ + n·I` (PD, so all decompositions are well-posed).
fn symmetric_pd(rng: &mut Lcg, n: usize) -> Vec<f64> {
    let m: Vec<f64> = (0..n * n).map(|_| rng.next_f64()).collect();
    let mut a = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for k in 0..n {
                acc += m[i * n + k] * m[j * n + k];
            }
            a[i * n + j] = acc;
        }
        a[i * n + i] += n as f64;
    }
    a
}

/// Hermitian `A = M·Mᴴ + n·I` as planar row-major parts.
fn hermitian_pd(rng: &mut Lcg, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mr: Vec<f64> = (0..n * n).map(|_| rng.next_f64()).collect();
    let mi: Vec<f64> = (0..n * n).map(|_| rng.next_f64()).collect();
    let mut re = vec![0.0_f64; n * n];
    let mut im = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let (mut ar, mut ai) = (0.0, 0.0);
            for k in 0..n {
                // M[i,k] * conj(M[j,k])
                ar += mr[i * n + k] * mr[j * n + k] + mi[i * n + k] * mi[j * n + k];
                ai += mi[i * n + k] * mr[j * n + k] - mr[i * n + k] * mi[j * n + k];
            }
            re[i * n + j] = ar;
            im[i * n + j] = ai;
        }
        re[i * n + i] += n as f64;
    }
    (re, im)
}

fn assert_same_bits(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(actual.to_bits(), expected.to_bits(), "entry {index}");
    }
}

#[test]
fn sym_eigh_matches_openblas_track_bits() {
    let mut rng = Lcg::new(0x05ee_d1a7);
    for &n in SIZES {
        let a = symmetric_pd(&mut rng, n);
        let (w, v) = sym_eigh(&a, n).expect("sym_eigh");
        let mut col = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                col[i + j * n] = a[i * n + j];
            }
        }
        let mut expected = vec![0.0; n];
        lapack_rs::openblas::dsyevd_host(
            lapack_rs::Job::Vectors,
            lapack_rs::Uplo::Lower,
            n,
            &mut col,
            &mut expected,
        )
        .expect("OpenBLAS host dsyevd");
        assert_same_bits(&w, &expected);
        assert_same_bits(&v, &col);
    }
}

#[test]
fn hermitian_eigh_matches_openblas_track_bits() {
    let mut rng = Lcg::new(0x05ee_d1a7);
    for &n in SIZES {
        let (re, im) = hermitian_pd(&mut rng, n);
        let (w, vr, vi) = hermitian_eigh(&re, &im, n).expect("hermitian_eigh");
        let mut re_col = vec![0.0; n * n];
        let mut im_col = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                re_col[i + j * n] = re[i * n + j];
                im_col[i + j * n] = im[i * n + j];
            }
        }
        let mut expected = vec![0.0; n];
        lapack_rs::openblas::zheevd_host(
            lapack_rs::Job::Vectors,
            lapack_rs::Uplo::Lower,
            n,
            &mut re_col,
            &mut im_col,
            &mut expected,
        )
        .expect("OpenBLAS host zheevd");
        assert_same_bits(&w, &expected);
        assert_same_bits(&vr, &re_col);
        assert_same_bits(&vi, &im_col);
    }
}

#[test]
fn sym_eigh_defining_property() {
    let mut rng = Lcg::new(0x05ee_d1a7);
    for &n in SIZES {
        let a = symmetric_pd(&mut rng, n);
        let (w, v_col) = sym_eigh(&a, n).expect("sym_eigh must converge on PD input");
        assert_eq!(w.len(), n);
        assert_eq!(v_col.len(), n * n);
        // Eigenvalues ascending.
        for k in 1..n {
            assert!(w[k - 1] <= w[k], "n={n}: eigenvalues not ascending: {w:?}");
        }
        // V is column-major: V[(i,j)] at v_col[i + j*n].
        // A·V = V·diag(w).
        for j in 0..n {
            for i in 0..n {
                let mut av = 0.0;
                for k in 0..n {
                    av += a[i * n + k] * v_col[k + j * n];
                }
                let vw = v_col[i + j * n] * w[j];
                assert!(
                    (av - vw).abs() <= TOL * (1.0 + vw.abs()),
                    "n={n}: (A·V)[{i},{j}]={av} vs (V·diag)[{i},{j}]={vw}"
                );
            }
        }
        // Vᵀ·V = I.
        for j in 0..n {
            for k in 0..n {
                let mut dot = 0.0;
                for i in 0..n {
                    dot += v_col[i + j * n] * v_col[i + k * n];
                }
                let want = if j == k { 1.0 } else { 0.0 };
                assert!(
                    (dot - want).abs() <= TOL,
                    "n={n}: (VᵀV)[{j},{k}]={dot}, want {want}"
                );
            }
        }
    }
}

#[test]
fn sym_eigh_shape_mismatch_errors() {
    let bad = vec![1.0_f64; 3];
    let r = sym_eigh(&bad, 2);
    assert!(matches!(
        r,
        Err(pyscf_algebra::AlgebraError::ShapeMismatch { .. })
    ));
}

#[test]
fn hermitian_eigh_defining_property() {
    let mut rng = Lcg::new(0x6e91);
    for &n in SIZES {
        let (re, im) = hermitian_pd(&mut rng, n);
        let (w, vr_col, vi_col) =
            hermitian_eigh(&re, &im, n).expect("hermitian_eigh must converge on PD input");
        assert_eq!(w.len(), n);
        for k in 1..n {
            assert!(w[k - 1] <= w[k], "n={n}: eigenvalues not ascending: {w:?}");
        }
        // (A·V)[i,j] = V[i,j]·w[j] with full complex products.
        for j in 0..n {
            for i in 0..n {
                let (mut avr, mut avi) = (0.0, 0.0);
                for k in 0..n {
                    let (ar, ai) = (re[i * n + k], im[i * n + k]);
                    let (br, bi) = (vr_col[k + j * n], vi_col[k + j * n]);
                    avr += ar * br - ai * bi;
                    avi += ar * bi + ai * br;
                }
                let (wr, wi) = (vr_col[i + j * n] * w[j], vi_col[i + j * n] * w[j]);
                assert!(
                    (avr - wr).abs() <= TOL * (1.0 + wr.abs())
                        && (avi - wi).abs() <= TOL * (1.0 + wi.abs()),
                    "n={n}: (A·V)[{i},{j}]=({avr},{avi}) vs ({wr},{wi})"
                );
            }
        }
        // Vᴴ·V = I.
        for j in 0..n {
            for k in 0..n {
                let (mut dr, mut di) = (0.0, 0.0);
                for i in 0..n {
                    // conj(V[i,j]) * V[i,k]
                    let (ar, ai) = (vr_col[i + j * n], -vi_col[i + j * n]);
                    let (br, bi) = (vr_col[i + k * n], vi_col[i + k * n]);
                    dr += ar * br - ai * bi;
                    di += ar * bi + ai * br;
                }
                let (wr, wi) = (if j == k { 1.0 } else { 0.0 }, 0.0);
                assert!(
                    (dr - wr).abs() <= TOL && (di - wi).abs() <= TOL,
                    "n={n}: (VᴴV)[{j},{k}]=({dr},{di})"
                );
            }
        }
    }
}

#[test]
fn eigh_gen_equation_holds_over_lapack_backend() {
    // Hand-crafted F and S (both symmetric, S positive-definite).
    let f = vec![2.0, 1.0, 1.0, 2.0];
    let s = vec![1.0, 0.1, 0.1, 1.0];
    let (vals, c) = eigh_gen(&f, &s, 2).expect("eigh_gen");
    assert!(vals.windows(2).all(|pair| pair[0] <= pair[1]));
    // C is F-order with n=2: consecutive 2-element chunks are the columns.
    for (val, col) in vals.iter().zip(c.as_chunks::<2>().0.iter()) {
        for i in 0..2 {
            let fc: f64 = f[i * 2..i * 2 + 2].iter().zip(col.iter()).map(|(&a, &b)| a * b).sum();
            let sc: f64 = s[i * 2..i * 2 + 2].iter().zip(col.iter()).map(|(&a, &b)| a * b).sum();
            assert!(
                (fc - val * sc).abs() < 1e-9,
                "row {i}: F·c={fc} vs ε·S·c={} (ε={val})",
                val * sc,
            );
        }
    }
}

/// Complex row·column dot product with full complex arithmetic.
fn zdot_row(re_row: &[f64], im_row: &[f64], col_re: &[f64], col_im: &[f64]) -> (f64, f64) {
    re_row
        .iter()
        .zip(im_row.iter())
        .zip(col_re.iter().zip(col_im.iter()))
        .fold((0.0, 0.0), |(ar, ai), ((&mr, &mi), (&vr, &vi))| {
            (ar + mr * vr - mi * vi, ai + mr * vi + mi * vr)
        })
}

#[test]
fn zeigh_gen_equation_holds_over_lapack_backend() {
    // 2×2 Hermitian F and PD Hermitian S, planar row-major.
    let f = CTensor::from_planes(vec![2.0, 1.0, 1.0, 3.0], vec![0.0, 0.5, -0.5, 0.0]);
    let s = CTensor::from_planes(vec![1.0, 0.1, 0.1, 1.0], vec![0.0, 0.05, -0.05, 0.0]);
    let (vals, c) = zeigh_gen(&f, &s, 2).expect("zeigh_gen");
    assert!(vals.windows(2).all(|pair| pair[0] <= pair[1]));
    // C is F-order planar: consecutive 2-element chunks are the columns.
    let cols = c.re.as_chunks::<2>().0.iter().zip(c.im.as_chunks::<2>().0.iter());
    for (val, (re_col, im_col)) in vals.iter().zip(cols) {
        for i in 0..2 {
            let (fcr, fci) = zdot_row(&f.re[i * 2..i * 2 + 2], &f.im[i * 2..i * 2 + 2], re_col, im_col);
            let (scr, sci) = zdot_row(&s.re[i * 2..i * 2 + 2], &s.im[i * 2..i * 2 + 2], re_col, im_col);
            assert!(
                (fcr - val * scr).abs() < 1e-9 && (fci - val * sci).abs() < 1e-9,
                "row {i}: F·c=({fcr},{fci}) vs ε·S·c=({},{}) (ε={val})",
                val * scr,
                val * sci,
            );
        }
    }
}
