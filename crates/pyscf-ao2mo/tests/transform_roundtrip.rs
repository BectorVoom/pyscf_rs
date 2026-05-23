//! AO→MO synthetic-ERI transform correctness — the ONE numeric assertion that
//! ships UN-GATED this phase (per RESEARCH Validation Architecture).
//!
//! Both arms build a SYNTHETIC `nao^4` AO ERI by hand (NOT from `intor` — no
//! cintx dependency), build a small non-trivial `MOCoefficients`, call
//! `general`/`full`, and assert the result against an independent longhand
//! 4-loop reference computed inline. This proves the contraction math without
//! any cintx / arity-4 `int2e` dependency.

use pyscf_ao2mo::{full, general};
use pyscf_core::MOCoefficients;

/// Build a deterministic, non-symmetric synthetic ERI `[nao,nao,nao,nao]`
/// (F-order): element `(p,q,r,s)` at `p + q*nao + r*nao^2 + s*nao^3`.
fn synthetic_eri(nao: usize) -> Vec<f64> {
    let n = nao;
    let mut eri = vec![0.0_f64; n * n * n * n];
    for s in 0..n {
        for r in 0..n {
            for q in 0..n {
                for p in 0..n {
                    let v = (p as f64) * 1.0
                        + (q as f64) * 0.25
                        + (r as f64) * 0.0625
                        + (s as f64) * 0.015_625
                        + 0.5;
                    eri[p + q * n + r * n * n + s * n * n * n] = v;
                }
            }
        }
    }
    eri
}

/// Build an `MOCoefficients` with a column-major `[nao, nmo]` `data` block.
/// `seed` makes distinct blocks; energies/occupations are irrelevant here.
fn mk_mo(nao: usize, nmo: usize, seed: f64) -> MOCoefficients {
    let mut data = vec![0.0_f64; nao * nmo];
    for mo in 0..nmo {
        for ao in 0..nao {
            data[ao + mo * nao] = seed + (ao as f64) * 0.5 - (mo as f64) * 0.3;
        }
    }
    MOCoefficients {
        nao,
        nmo,
        data,
        energies: vec![0.0; nmo],
        occupations: vec![0.0; nmo],
    }
}

/// Independent longhand reference: staged 4-index transform with strict
/// scalar folds (never `oracle_sum`). All tensors F-order.
#[allow(clippy::too_many_arguments)]
fn reference(
    eri: &[f64],
    nao: usize,
    cp: &[f64],
    np: usize,
    cq: &[f64],
    nq: usize,
    cr: &[f64],
    nr: usize,
    cs: &[f64],
    ns: usize,
) -> Vec<f64> {
    let mut o1 = vec![0.0_f64; np * nao * nao * nao];
    for s in 0..nao {
        for r in 0..nao {
            for q in 0..nao {
                let base = q * nao + r * nao * nao + s * nao * nao * nao;
                for i in 0..np {
                    let mut acc = 0.0_f64;
                    for p in 0..nao {
                        acc += eri[p + base] * cp[p + i * nao];
                    }
                    o1[i + q * np + r * np * nao + s * np * nao * nao] = acc;
                }
            }
        }
    }
    let mut o2 = vec![0.0_f64; np * nq * nao * nao];
    for s in 0..nao {
        for r in 0..nao {
            for j in 0..nq {
                for i in 0..np {
                    let mut acc = 0.0_f64;
                    for q in 0..nao {
                        acc += o1[i + q * np + r * np * nao + s * np * nao * nao] * cq[q + j * nao];
                    }
                    o2[i + j * np + r * np * nq + s * np * nq * nao] = acc;
                }
            }
        }
    }
    let mut o3 = vec![0.0_f64; np * nq * nr * nao];
    for s in 0..nao {
        for k in 0..nr {
            for j in 0..nq {
                for i in 0..np {
                    let mut acc = 0.0_f64;
                    for r in 0..nao {
                        acc += o2[i + j * np + r * np * nq + s * np * nq * nao] * cr[r + k * nao];
                    }
                    o3[i + j * np + k * np * nq + s * np * nq * nr] = acc;
                }
            }
        }
    }
    let mut out = vec![0.0_f64; np * nq * nr * ns];
    for l in 0..ns {
        for k in 0..nr {
            for j in 0..nq {
                for i in 0..np {
                    let mut acc = 0.0_f64;
                    for s in 0..nao {
                        acc += o3[i + j * np + k * np * nq + s * np * nq * nr] * cs[s + l * nao];
                    }
                    out[i + j * np + k * np * nq + l * np * nq * nr] = acc;
                }
            }
        }
    }
    out
}

/// Always-on compile-smoke: `general`/`full` are exported and callable with
/// the real `MOCoefficients` surface.
#[test]
fn ao2mo_symbols_exist() {
    let nao = 1;
    let eri = synthetic_eri(nao);
    let mo = mk_mo(nao, nao, 1.0);
    let _ = general(&eri, nao, [&mo, &mo, &mo, &mo]);
    let _ = full(&eri, nao, &mo);
}

/// Always-on numeric assertion (THE un-gated correctness check this phase):
/// `general` on a synthetic `nao=3` ERI with four distinct non-trivial MO
/// blocks equals the independent longhand reference (bit-exact for this small
/// case — every per-index sum is length `nao ≤ 128`, so `oracle_sum`'s base
/// case is itself a strict left fold matching the reference).
#[test]
fn ao2mo_general_matches_longhand_reference() {
    let nao = 3;
    let eri = synthetic_eri(nao);
    let (np, nq, nr, ns) = (2, 3, 2, 1);
    let cp = mk_mo(nao, np, 1.1);
    let cq = mk_mo(nao, nq, 0.7);
    let cr = mk_mo(nao, nr, -0.4);
    let cs = mk_mo(nao, ns, 0.9);

    let got = general(&eri, nao, [&cp, &cq, &cr, &cs]).expect("general transform");
    let want = reference(
        &eri, nao, &cp.data, np, &cq.data, nq, &cr.data, nr, &cs.data, ns,
    );
    assert_eq!(got.len(), np * nq * nr * ns);
    assert_eq!(got.len(), want.len());
    for (idx, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            a, b,
            "general mismatch at flat index {idx}: got {a}, want {b}"
        );
    }
}

/// Always-on: `full` (all-same-coeff symmetric case) on a synthetic ERI
/// equals the longhand reference computed with the same coefficient block on
/// all four indices.
#[test]
fn ao2mo_full_matches_longhand_reference() {
    let nao = 3;
    let nmo = 3;
    let eri = synthetic_eri(nao);
    let mo = mk_mo(nao, nmo, 0.6);

    let got = full(&eri, nao, &mo).expect("full transform");
    let want = reference(
        &eri, nao, &mo.data, nmo, &mo.data, nmo, &mo.data, nmo, &mo.data, nmo,
    );
    assert_eq!(got.len(), nmo * nmo * nmo * nmo);
    assert_eq!(got.len(), want.len());
    for (idx, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(a, b, "full mismatch at flat index {idx}: got {a}, want {b}");
    }
}

/// Identity-transform roundtrip: with an identity MO coefficient block on all
/// four indices, `full` returns the AO ERI element-wise unchanged — the
/// un-gated synthetic-ERI correctness check (replaces the 05-01 `#[ignore]`d
/// placeholder).
#[test]
fn ao2mo_identity_transform_roundtrips() {
    let nao = 3;
    let eri = synthetic_eri(nao);
    let mut data = vec![0.0_f64; nao * nao];
    for i in 0..nao {
        data[i + i * nao] = 1.0; // column-major identity
    }
    let ident = MOCoefficients {
        nao,
        nmo: nao,
        data,
        energies: vec![0.0; nao],
        occupations: vec![0.0; nao],
    };
    let out = full(&eri, nao, &ident).expect("identity full transform");
    assert_eq!(out.len(), eri.len());
    for (a, b) in out.iter().zip(eri.iter()) {
        assert_eq!(a, b, "identity transform must return the ERI unchanged");
    }
}

/// Shape-mismatch guard (T-05-02-SHAPE): a coefficient block whose `nao` does
/// not match the ERI's `nao` returns `ShapeMismatch`, never panics/OOB.
#[test]
fn ao2mo_shape_mismatch_errors() {
    let nao = 2;
    let eri = synthetic_eri(nao);
    let bad = mk_mo(nao + 1, 2, 1.0); // nao=3 != ERI nao=2
    let res = full(&eri, nao, &bad);
    assert!(res.is_err(), "mismatched nao must error, not panic");
}
