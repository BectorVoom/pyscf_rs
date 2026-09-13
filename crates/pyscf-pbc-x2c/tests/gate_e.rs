//! Gate E: X2C one-electron energies vs upstream at 8 decimals (19-18).
//!
//! Fixture `fixtures/x2c_h2_gamma_c4.json` was generated from live upstream
//! PySCF **2.12.1** (H2/`sto-3g` cell, Γ, `light_speed(4)` — exaggerated
//! relativity for test sensitivity): the `(t, v, w, s)` xcell blocks, the
//! contraction coefficients, and upstream's own `h1x`/`h1`. This test feeds
//! the SAME blocks into this port's transform and asserts 1e-8 — Gate E.
//!
//! Oracle-free arms: nonrelativistic limit (`c → ∞` gives `T + V`), `R → I`
//! at `s1 == s`, Hermiticity, the positive-branch eigenvalue identity
//! (independent path: full `(h, m)` spectrum vs FW-assembled `h1`), and the
//! `x2c1e == doubled sfx2c1e` spin structure.
//!
//! Run scoped: `cargo test -p pyscf-pbc-x2c --test gate_e`

use pyscf_algebra::eigh_gen;
use pyscf_pbc_x2c::sfx2c1e::{hcore_fw, renorm_r, sfx2c1e_hcore_at_c, xmatrix};
use pyscf_pbc_x2c::x2c1e::{x2c1e_hcore_at_c, x2c1e_hcore_so};
use serde_json::Value;

fn fixture() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, usize, usize, f64) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/x2c_h2_gamma_c4.json");
    let v: Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("fixture must exist")).unwrap();
    let get = |k: &str| -> Vec<f64> {
        v[k].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
    };
    (
        get("t"), get("v"), get("w"), get("s"), get("contr"), get("h1x"), get("h1"),
        v["nao_x"].as_u64().unwrap() as usize,
        v["nao"].as_u64().unwrap() as usize,
        v["c"].as_f64().unwrap(),
    )
}

fn contract(hx: &[f64], contr: &[f64], nx: usize, n: usize) -> Vec<f64> {
    // h = Cᵀ·hx·C, contr is nx × n row-major.
    let mut out = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0f64;
            for k in 0..nx {
                for l in 0..nx {
                    acc += contr[k * n + i] * hx[k * nx + l] * contr[l * n + j];
                }
            }
            out[i * n + j] = acc;
        }
    }
    out
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

/// Gate E: this port's transform on upstream's blocks matches upstream's h1.
#[test]
fn gate_e_sfx2c1e_matches_upstream_1e8() {
    let (t, v, w, s, contr, h1x_ref, h1_ref, nx, n, c) = fixture();
    let h1x = sfx2c1e_hcore_at_c(&t, &v, &w, &s, nx, c).expect("transform must run");
    assert!(
        max_abs_diff(&h1x, &h1x_ref) < 1e-8,
        "xcell-basis h1 deviates: {:e}",
        max_abs_diff(&h1x, &h1x_ref)
    );
    let h1 = contract(&h1x, &contr, nx, n);
    assert!(
        max_abs_diff(&h1, &h1_ref) < 1e-8,
        "contracted h1 deviates: {:e}",
        max_abs_diff(&h1, &h1_ref)
    );
}

/// Nonrelativistic limit: c → ∞ recovers T + V (O(1/c²) corrections vanish).
///
/// c = 1000 (not 1e6: at 1e6 the small-component metric block T/2c² falls
/// below `S_LINEAR_DEP_TOL` and the Dirac problem itself is numerically
/// singular — upstream's `eigh` degrades identically there).
#[test]
fn nonrelativistic_limit_recovers_t_plus_v() {
    let (t, v, w, s, _c, _h1x, _h1, nx, _n, _cc) = fixture();
    let h1 = sfx2c1e_hcore_at_c(&t, &v, &w, &s, nx, 1000.0).expect("transform must run");
    let tv: Vec<f64> = t.iter().zip(v.iter()).map(|(a, b)| a + b).collect();
    let d = max_abs_diff(&h1, &tv);
    assert!(d < 1e-4, "c=1000 still {d:e} from T+V");
}

/// R → I when the metric needs no renormalization.
#[test]
fn renorm_is_identity_on_equal_metrics() {
    let (_t, _v, _w, s, _c, _h1x, _h1, nx, _n, _cc) = fixture();
    let r = renorm_r(&s, &s, nx).expect("renorm must run");
    for i in 0..nx {
        for j in 0..nx {
            let want = if i == j { 1.0 } else { 0.0 };
            assert!((r[i * nx + j] - want).abs() < 1e-10, "R[{i},{j}] = {}", r[i * nx + j]);
        }
    }
}

/// The picture-changed h1 is Hermitian.
#[test]
fn hcore_fw_is_hermitian() {
    let (t, v, w, s, _c, _h1x, _h1, nx, _n, c) = fixture();
    let x = xmatrix(&t, &v, &w, &s, nx, c).expect("xmatrix must run");
    let h1 = hcore_fw(&t, &v, &w, &s, &x, nx, c).expect("fw must run");
    for i in 0..nx {
        for j in 0..nx {
            assert!((h1[i * nx + j] - h1[j * nx + i]).abs() < 1e-12);
        }
    }
}

/// Positive-branch identity (independent path): the eigenvalues of the
/// FW-assembled h1 equal the positive branch of the full (h, m) spectrum.
#[test]
fn fw_eigenvalues_match_dirac_positive_branch() {
    let (t, v, w, s, _c, _h1x, _h1, nx, _n, c) = fixture();
    let n2 = 2 * nx;
    let (mut h, mut m) = (vec![0.0f64; n2 * n2], vec![0.0f64; n2 * n2]);
    for i in 0..nx {
        for j in 0..nx {
            h[i * n2 + j] = v[i * nx + j];
            h[i * n2 + nx + j] = t[i * nx + j];
            h[(nx + i) * n2 + j] = t[i * nx + j];
            h[(nx + i) * n2 + nx + j] = w[i * nx + j] * (0.25 / (c * c)) - t[i * nx + j];
            m[i * n2 + j] = s[i * nx + j];
            m[(nx + i) * n2 + nx + j] = t[i * nx + j] * (0.5 / (c * c));
        }
    }
    let (e_full, _) = eigh_gen(&h, &m, n2).expect("dirac spectrum must solve");
    // Positive branch: nao largest eigenvalues (eigh_gen returns ascending).
    let pos = &e_full[nx..];
    let x = xmatrix(&t, &v, &w, &s, nx, c).expect("xmatrix must run");
    let h1 = hcore_fw(&t, &v, &w, &s, &x, nx, c).expect("fw must run");
    let ident: Vec<f64> = {
        let mut q = vec![0.0f64; nx * nx];
        for i in 0..nx {
            q[i * nx + i] = 1.0;
        }
        q
    };
    // h1 is S-orthonormal (Rᵀh1R with Rᵀs1R = s... FW h1 is standard-eigenvalue
    // only in the s metric; compare in the s-metric generalized problem.
    let mut s1 = vec![0.0f64; nx * nx];
    {
        let c2 = c * c;
        for i in 0..nx {
            for j in 0..nx {
                let mut xtx = 0.0f64;
                for k in 0..nx {
                    for l in 0..nx {
                        xtx += x[k * nx + i] * t[k * nx + l] * x[l * nx + j];
                    }
                }
                s1[i * nx + j] = s[i * nx + j] + xtx * (0.5 / c2);
            }
        }
    }
    // `R` satisfies Rᵀ·s1·R = s (`_get_r` comment), so the returned
    // h1 = Rᵀ·h1raw·R carries its spectrum in the s-metric generalized
    // problem (h1, s) — NOT (h1, s1). Löwdin-orthogonalize with s.
    let (sw, sv_f) = eigh_gen(&s, &ident, nx).expect("s spectrum must solve");
    // A = S^{-1/2} = V·diag(w^{-1/2})·Vᵀ, built explicitly (folding the
    // eigenvector indices into the matrix indices directly is a same-shape
    // plausible-wrong-number — caught here during development).
    let mut ainv = vec![0.0f64; nx * nx];
    for i in 0..nx {
        for j in 0..nx {
            let mut acc = 0.0f64;
            for k in 0..nx {
                acc += sv_f[k * nx + i] * sv_f[k * nx + j] / sw[k].sqrt();
            }
            ainv[i * nx + j] = acc;
        }
    }
    // y = A·H·A, ordinary eigh.
    let mut y = vec![0.0f64; nx * nx];
    for i in 0..nx {
        for j in 0..nx {
            let mut acc = 0.0f64;
            for k in 0..nx {
                for l in 0..nx {
                    acc += ainv[i * nx + k] * h1[k * nx + l] * ainv[l * nx + j];
                }
            }
            y[i * nx + j] = acc;
        }
    }
    let (e_fw, _) = eigh_gen(&y, &ident, nx).expect("fw spectrum must solve");
    for (a, b) in e_fw.iter().zip(pos.iter()) {
        assert!((a - b).abs() < 1e-8, "FW {a:e} vs Dirac-positive {b:e}");
    }
}

/// Full X2C on real spin blocks is the doubled spin-free transform.
#[test]
fn x2c1e_is_doubled_sfx2c1e_on_real_blocks() {
    let (t, v, w, s, _c, _h1x, _h1, nx, _n, c) = fixture();
    let h1 = sfx2c1e_hcore_at_c(&t, &v, &w, &s, nx, c).expect("sfx2c1e must run");
    let h2 = x2c1e_hcore_at_c(&t, &v, &w, &s, nx, c).expect("x2c1e must run");
    assert_eq!(h2.len(), 4 * nx * nx);
    for i in 0..nx {
        for j in 0..nx {
            assert_eq!(h2[i * 2 * nx + j].to_bits(), h1[i * nx + j].to_bits());
            assert_eq!(h2[(nx + i) * 2 * nx + nx + j].to_bits(), h1[i * nx + j].to_bits());
            assert_eq!(h2[i * 2 * nx + nx + j].to_bits(), 0.0f64.to_bits());
            assert_eq!(h2[(nx + i) * 2 * nx + j].to_bits(), 0.0f64.to_bits());
        }
    }
}

/// Explicit spin-orbit blocks are refused, never silently dropped.
#[test]
fn x2c1e_so_refuses() {
    let (t, v, w, s, _c, _h1x, _h1, nx, _n, _cc) = fixture();
    assert!(x2c1e_hcore_so(&t, &v, &w, &s, nx).is_err());
}

/// Determinism at 1 vs 8 rayon workers inside one process.
#[test]
fn x2c_deterministic_across_thread_counts() {
    let run = || {
        let (t, v, w, s, _c, _h1x, _h1, nx, _n, c) = fixture();
        sfx2c1e_hcore_at_c(&t, &v, &w, &s, nx, c).expect("transform must run")
    };
    let pool1 = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let pool8 = rayon::ThreadPoolBuilder::new().num_threads(8).build().unwrap();
    let (a, b) = (pool1.install(&run), pool8.install(&run));
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x.to_bits(), y.to_bits());
    }
}
