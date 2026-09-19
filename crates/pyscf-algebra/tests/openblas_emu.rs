//! Oracle-free checks of the Barcelona OpenBLAS summation-order emulation
//! (`pyscf_algebra::openblas_emu`). The bit-for-bit comparison against the
//! real `dgemm_`/`zgemm_` lives in `pyscf-pbc-df/tests/get_nuc_bitexact.rs`
//! (it needs the upstream wheel's OpenBLAS).

use pyscf_algebra::openblas_emu::{dgemm_nt, k_blocks, zgemm, zgemm_nt};

/// The level-3 driver's K blocking: `GEMM_Q = 224`, the last two blocks
/// halved and rounded up to the M-unroll (4 for `dgemm`, 2 for `zgemm`).
#[test]
fn k_blocks_follow_the_level3_driver() {
    assert_eq!(k_blocks(1, 4), vec![1]);
    assert_eq!(k_blocks(224, 4), vec![224]);
    assert_eq!(k_blocks(225, 4), vec![112, 113]);
    assert_eq!(k_blocks(448, 4), vec![224, 224]);
    assert_eq!(k_blocks(1331, 4), vec![224, 224, 224, 224, 220, 215]);
    assert_eq!(k_blocks(1331, 2), vec![224, 224, 224, 224, 218, 217]);
    for k in 1..2000 {
        assert_eq!(k_blocks(k, 4).iter().sum::<usize>(), k, "k = {k}");
        assert_eq!(k_blocks(k, 2).iter().sum::<usize>(), k, "k = {k}");
    }
}

/// Small integers make every summation order exact, so the emulation must
/// then equal the plain matrix product — this checks the indexing and the
/// tile bookkeeping independently of the rounding model.
#[test]
fn integer_inputs_give_the_exact_product() {
    let val = |s: usize| ((s * 7919) % 13) as f64 - 6.0;
    for m in 1..9 {
        for n in 1..9 {
            for k in [1, 3, 4, 7, 230, 500] {
                let a: Vec<f64> = (0..m * k).map(val).collect();
                let b: Vec<f64> = (0..n * k).map(|s| val(s + 3)).collect();
                let ai: Vec<f64> = (0..m * k).map(|s| val(s + 5)).collect();
                let bi: Vec<f64> = (0..n * k).map(|s| val(s + 11)).collect();
                let mut c = vec![1.0; m * n];
                dgemm_nt(m, n, k, &a, &b, &mut c);
                let (mut zr, mut zi) = (vec![0.0; m * n], vec![0.0; m * n]);
                zgemm_nt(m, n, k, &a, &ai, &b, &bi, &mut zr, &mut zi);
                for i in 0..m {
                    for j in 0..n {
                        let (mut d, mut r, mut im) = (1.0, 0.0, 0.0);
                        for kk in 0..k {
                            let (x, xi) = (a[i + kk * m], ai[i + kk * m]);
                            let (y, yi) = (b[j + kk * n], bi[j + kk * n]);
                            d += x * y;
                            r += x * y - xi * yi;
                            im += x * yi + xi * y;
                        }
                        assert_eq!(c[i + j * m], d, "dgemm m={m} n={n} k={k} ({i},{j})");
                        assert_eq!(zr[i + j * m], r, "zgemm.re m={m} n={n} k={k} ({i},{j})");
                        assert_eq!(zi[i + j * m], im, "zgemm.im m={m} n={n} k={k} ({i},{j})");
                    }
                }
            }
        }
    }
}

/// Integer inputs for the general [`zgemm`] (`('N','T')`, real `alpha`,
/// `beta` 0/1, padded leading dims): every partial sum is a small exact
/// integer, so the expected value is order-independent — one rounding by
/// `alpha`, one addition of `beta * c`.
#[test]
fn integer_inputs_give_the_exact_product_with_alpha() {
    let val = |s: usize| ((s * 7919) % 13) as f64 - 6.0;
    for m in [1, 2, 3, 5, 16, 17] {
        for n in [1, 2, 3, 7, 100] {
            for k in [1, 3, 17] {
                for (alpha, beta) in [((1.0 / 17.0, 0.0), (0.0, 0.0)), ((2.0, 0.0), (1.0, 0.0))] {
                    // Padded leading dims exercise the lda/ldb/ldc indexing.
                    let (lda, ldb, ldc) = (m + 1, n + 2, m + 3);
                    let a: Vec<f64> = (0..lda * k).map(val).collect();
                    let ai: Vec<f64> = (0..lda * k).map(|s| val(s + 5)).collect();
                    let b: Vec<f64> = (0..ldb * k).map(|s| val(s + 3)).collect();
                    let bi: Vec<f64> = (0..ldb * k).map(|s| val(s + 11)).collect();
                    let (mut zr, mut zi) = (vec![0.5; ldc * n], vec![-0.25; ldc * n]);
                    let (c0r, c0i) = (zr.clone(), zi.clone());
                    zgemm(
                        'N', 'T', m, n, k, alpha, &a, &ai, lda, &b, &bi, ldb, beta, &mut zr,
                        &mut zi, ldc,
                    );
                    for i in 0..m {
                        for j in 0..n {
                            let (mut sr, mut si) = (0.0, 0.0);
                            for kk in 0..k {
                                let (x, xi) = (a[i + kk * lda], ai[i + kk * lda]);
                                let (y, yi) = (b[j + kk * ldb], bi[j + kk * ldb]);
                                sr += x * y - xi * yi;
                                si += x * yi + xi * y;
                            }
                            // Single K block here (k <= 17 < 224): one
                            // rounding by alpha, one beta accumulation.
                            let (er, ei) = if beta == (0.0, 0.0) {
                                (alpha.0 * sr, alpha.0 * si)
                            } else {
                                (
                                    c0r[i + j * ldc] + alpha.0 * sr,
                                    c0i[i + j * ldc] + alpha.0 * si,
                                )
                            };
                            assert_eq!(zr[i + j * ldc], er, "re m={m} n={n} k={k} ({i},{j})");
                            assert_eq!(zi[i + j * ldc], ei, "im m={m} n={n} k={k} ({i},{j})");
                        }
                    }
                }
            }
        }
    }
}
