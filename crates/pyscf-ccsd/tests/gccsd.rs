//! Plan 16-07 Task 4 test 4 — the physicists'-vs-chemists' transposition,
//! asserted rather than assumed.
//!
//! `crates/pyscf-ccsd/src/eris.rs` holds chemist `(pq|rs)`;
//! `gccsd::PhysicistsEris` names the ANTISYMMETRISED physicist `<pq||rs>`.
//! The two differ by an index transposition AND a subtraction, and this
//! project has already paid `+6 306 866.73 Ha` once for exactly that class of
//! misread (14-05's `decompose_j2c`, `16-CONTEXT §3.4`).

use pyscf_ccsd::gccsd::PhysicistsEris;

fn chem(n: usize) -> Vec<f64> {
    // A block with the 8-fold real-integral symmetry of a genuine `(pq|rs)`:
    // symmetric in p<->q, in r<->s, and under the bra/ket swap.
    let at = |p: usize, q: usize, r: usize, s: usize| ((p * n + q) * n + r) * n + s;
    let f = |a: usize, b: usize| ((a * 7 + b * 3) as f64).sin();
    let mut v = vec![0.0_f64; n * n * n * n];
    for p in 0..n {
        for q in 0..n {
            for r in 0..n {
                for s in 0..n {
                    let bra = f(p.min(q), p.max(q));
                    let ket = f(r.min(s), r.max(s));
                    v[at(p, q, r, s)] = bra * ket + 0.25 * (bra + ket);
                }
            }
        }
    }
    v
}

/// `<pq||rs> = (pr|qs) - (ps|qr)`, element by element.
#[test]
fn antisymmetrise_is_the_documented_formula() {
    let n = 5;
    let c = chem(n);
    let a = PhysicistsEris::antisymmetrise(&c, n);
    let at = |p: usize, q: usize, r: usize, s: usize| ((p * n + q) * n + r) * n + s;
    for p in 0..n {
        for q in 0..n {
            for r in 0..n {
                for s in 0..n {
                    let want = c[at(p, r, q, s)] - c[at(p, s, q, r)];
                    assert!(
                        (a[at(p, q, r, s)] - want).abs() < 1e-15,
                        "<{p}{q}||{r}{s}> is not (pr|qs) - (ps|qr)"
                    );
                }
            }
        }
    }
}

/// The defining antisymmetries: `<pq||rs> = -<pq||sr> = -<qp||rs> = <qp||sr>`.
///
/// These hold for ANY chemist block with the ordinary 8-fold symmetry, so they
/// gate the transposition without depending on the fixture's particular values.
#[test]
fn the_antisymmetries_hold() {
    let n = 5;
    let a = PhysicistsEris::antisymmetrise(&chem(n), n);
    let at = |p: usize, q: usize, r: usize, s: usize| ((p * n + q) * n + r) * n + s;
    for p in 0..n {
        for q in 0..n {
            for r in 0..n {
                for s in 0..n {
                    let x = a[at(p, q, r, s)];
                    assert!((x + a[at(p, q, s, r)]).abs() < 1e-14, "ket antisymmetry");
                    assert!((x + a[at(q, p, r, s)]).abs() < 1e-14, "bra antisymmetry");
                    assert!((x - a[at(q, p, s, r)]).abs() < 1e-14, "double swap");
                }
            }
        }
    }
    // And the diagonal vanishes, which a wrong transposition would not give.
    for p in 0..n {
        for r in 0..n {
            assert!(a[at(p, p, r, r)].abs() < 1e-14, "<pp||rr> must vanish");
        }
    }
}

/// It is NOT the chemist block — a port that forgot the transposition would
/// otherwise pass everything above by accident on a symmetric fixture.
#[test]
fn it_is_not_the_chemist_block() {
    let n = 4;
    let c = chem(n);
    let a = PhysicistsEris::antisymmetrise(&c, n);
    let differs = c.iter().zip(a.iter()).any(|(x, y)| (x - y).abs() > 1e-9);
    assert!(differs, "the physicist block is identical to the chemist one");
}
