//! Plan 16-02 Task 4, tests 6-9 — `build_symm_map` / `transform_symm`.
//!
//! The map itself shipped with Phase 15 (`src/khelper.rs`, gated by
//! `tests/khelper.rs`). What Phase 16 adds is the property set KCCSD depends on
//! and KMP2 did not: KMP2 asked only for `(ov|ov)`, so `15-REVIEW.md
//! D-15-R-04` correctly ruled the saving there at ≤2×; KCCSD's `_ERIS` wants
//! the full general `(pq|rs)` block (`kccsd_rhf.py:789-794`), so ALL FOUR
//! operations land inside the set it wants and the orbit structure has to be
//! exactly right (`16-CONTEXT §3.1`).
//!
//! Every assertion is an exact integer or a bit-identity. No tolerances.

use pyscf_algebra::CTensor;
use pyscf_pbc_lib::KptsHelper;

fn lattice() -> [[f64; 3]; 3] {
    [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
}

/// A 1-D chain of `n` k-points along z, the cheapest mesh with a non-trivial
/// `kconserv`.
fn kpts(n: usize) -> Vec<[f64; 3]> {
    (0..n)
        .map(|i| [0.0, 0.0, i as f64 * std::f64::consts::PI / n as f64])
        .collect()
}

/// Test 6 — completeness. Every one of the `nkpts³` triples is claimed by
/// **exactly one representative**. Exact integers, no tolerance.
///
/// `kpts_helper.py:589-612`: the `completed` sweep claims each triple once, so
/// a triple claimed by two representatives means one integral transform is
/// being done twice and a triple claimed by none means a block is never filled.
///
/// **A triple may appear MORE THAN ONCE inside its own orbit, and that is
/// upstream's behaviour, not a defect.** When a triple is a fixed point of one
/// of the four operations — every triple at `nkpts = 1`, and the Γ-containing
/// ones generally — several of `(kp,kq,kr)`, `(kr,ks,kp)`, `(kq,kp,ks)`,
/// `(ks,kr,kq)` coincide, and `:597-612` appends each unconditionally while
/// `completed` only ever guards the OUTER loop. `_operation` is likewise
/// overwritten in candidate order, so a fixed point ends on the LAST matching
/// operation. This port reproduces both, which is why the assertion below
/// counts distinct representatives rather than orbit-membership multiplicity.
#[test]
fn every_triple_is_claimed_by_exactly_one_representative() {
    for n in [1_usize, 2, 4, 8] {
        let h = KptsHelper::new(&lattice(), &kpts(n));
        let mut owners = vec![0_usize; n * n * n];
        for (_, orbit) in h.symm_map.as_ref().expect("eager map").entries() {
            let mut distinct: Vec<[usize; 3]> = orbit.to_vec();
            distinct.sort_unstable();
            distinct.dedup();
            for [p, q, r] in distinct {
                owners[(p * n + q) * n + r] += 1;
            }
        }
        assert!(
            owners.iter().all(|&c| c == 1),
            "nkpts {n}: triples claimed by {:?} representatives, expected all 1",
            {
                let mut v: Vec<usize> = owners.clone();
                v.sort_unstable();
                v.dedup();
                v
            }
        );
    }
}

/// Test 7 — orbit size is at most 4, and the DISTINCT triples across all
/// orbits number exactly `nkpts³`.
///
/// The generator set is four operations (`kpts_helper.py:614-630`), so an orbit
/// longer than 4 is impossible; the orbit COLLAPSES at fixed points (see test
/// 6), so the distinct count per orbit is 1, 2 or 4 and the realised average is
/// under 4. This test records the realised representative count per `nkpts`
/// rather than asserting a ratio — `16-01` Task 6 measures that ratio against
/// upstream, and `15-REVIEW.md D-15-R-04` is the precedent for a corollary
/// whose arithmetic was backwards even though its conclusion survived.
#[test]
fn orbits_are_at_most_four_and_partition_the_cube() {
    for n in [1_usize, 2, 4, 8] {
        let h = KptsHelper::new(&lattice(), &kpts(n));
        let map = h.symm_map.as_ref().expect("eager map");
        let mut distinct_total = 0_usize;
        for (key, orbit) in map.entries() {
            assert!(
                orbit.len() <= 4,
                "nkpts {n}: orbit of {key:?} has {} members, the four operations \
                 of kpts_helper.py:614-630 can generate at most 4",
                orbit.len()
            );
            assert!(!orbit.is_empty(), "nkpts {n}: empty orbit for {key:?}");
            assert_eq!(orbit[0], *key, "the representative is its own first member");
            let mut d: Vec<[usize; 3]> = orbit.to_vec();
            d.sort_unstable();
            d.dedup();
            distinct_total += d.len();
        }
        assert_eq!(
            distinct_total,
            n * n * n,
            "nkpts {n}: distinct orbit members must cover nkpts^3 exactly once"
        );
        // The realised saving, recorded (not asserted): n^3 / representatives.
        println!(
            "nkpts {n}: {} representatives for {} triples, ratio {:.4}",
            map.entries().len(),
            n * n * n,
            (n * n * n) as f64 / map.entries().len() as f64
        );
    }
}

/// Reference implementations of `transform_symm`'s four operations on a
/// hypercubic `[m,m,m,m]` block, written out index by index from
/// `kpts_helper.py:615-632`.
///
/// **The conjugation on ops 2 and 3 is not decoration** (`16-CONTEXT §3.2`):
/// dropping it produces a plausible wrong ERI that only the final `e_corr`
/// would catch.
fn op_ref(x: &CTensor, m: usize, op: u8) -> CTensor {
    let at = |a: usize, b: usize, c: usize, d: usize| ((a * m + b) * m + c) * m + d;
    let mut out = CTensor::zeros(x.re.len());
    for a in 0..m {
        for b in 0..m {
            for c in 0..m {
                for d in 0..m {
                    let (src, conj) = match op {
                        0 => (at(a, b, c, d), false),
                        // eri.transpose(2,3,0,1) — y[a,b,c,d] = x[c,d,a,b]
                        1 => (at(c, d, a, b), false),
                        // eri.transpose(1,0,3,2).conj()
                        2 => (at(b, a, d, c), true),
                        // eri.transpose(3,2,1,0).conj()
                        3 => (at(d, c, b, a), true),
                        _ => unreachable!("only four operations exist"),
                    };
                    out.re[at(a, b, c, d)] = x.re[src];
                    out.im[at(a, b, c, d)] = if conj { -x.im[src] } else { x.im[src] };
                }
            }
        }
    }
    out
}

fn sample(m: usize) -> CTensor {
    let n = m * m * m * m;
    CTensor::from_planes(
        (0..n).map(|i| (i as f64) * 0.5 - 3.0).collect(),
        (0..n).map(|i| 1.0 - (i as f64) * 0.25).collect(),
    )
}

/// Test 8 — the group property, bit-identically.
///
/// `op1∘op1 == id`, `op2∘op2 == id`, and `op3 == op1∘op2`. These three hold
/// only if the conjugations are where upstream puts them: `op1` is
/// conjugation-free and `op2`/`op3` conjugate, so `op1∘op2` picks up exactly
/// one conjugation and matches `op3`. A port that conjugated `op1` too, or
/// dropped it from `op3`, fails here rather than at `e_corr`.
#[test]
fn the_four_operations_are_mutually_consistent() {
    let m = 3;
    let x = sample(m);
    for op in 0..4_u8 {
        assert_eq!(op_ref(&x, m, op).re.len(), x.re.len());
    }
    assert_eq!(op_ref(&op_ref(&x, m, 1), m, 1), x, "op1 is an involution");
    assert_eq!(op_ref(&op_ref(&x, m, 2), m, 2), x, "op2 is an involution");
    assert_eq!(op_ref(&op_ref(&x, m, 3), m, 3), x, "op3 is an involution");
    assert_eq!(
        op_ref(&op_ref(&x, m, 2), m, 1),
        op_ref(&x, m, 3),
        "op3 == op1 ∘ op2 (kpts_helper.py:620-630)"
    );
}

/// Test 8b — `KptsHelper::transform_symm` applies exactly the operation it
/// recorded for the triple, bit-identically to [`op_ref`].
///
/// This is what connects the algebra above to the shipped code path: the map
/// decides WHICH operation, and this asserts the operation itself is right.
#[test]
fn transform_symm_matches_the_reference_operation_bitwise() {
    let n = 4;
    let m = 3;
    let h = KptsHelper::new(&lattice(), &kpts(n));
    let x = sample(m);
    let mut seen = [false; 4];
    for p in 0..n {
        for q in 0..n {
            for r in 0..n {
                let op = h.operation(p, q, r).expect("eager map records operations");
                seen[op as usize] = true;
                let got = h
                    .transform_symm(&x, [m; 4], p, q, r)
                    .expect("hypercubic block");
                assert_eq!(
                    got,
                    op_ref(&x, m, op),
                    "triple ({p},{q},{r}) op {op}: transform_symm diverges from \
                     the kpts_helper.py:615-632 reference"
                );
            }
        }
    }
    assert!(
        seen.iter().all(|&s| s),
        "a 4-point mesh must exercise all four operations; saw {seen:?}"
    );
}

/// Test 9 — the map is LAZY. `kccsd_rhf.py:512` constructs its helper with
/// `init_symm_map=False` and `_ERIS` builds the map at `:783`; a `KRCCSD` that
/// is constructed but never run must not pay `O(nkpts³)`.
#[test]
fn the_map_is_not_built_until_asked_for() {
    let mut lazy = KptsHelper::without_symm_map(&lattice(), &kpts(4));
    assert!(lazy.symm_map.is_none(), "kccsd_rhf.py:512 passes init_symm_map=False");
    assert!(lazy.operation(0, 0, 0).is_none());
    assert_eq!(
        lazy.transform_symm(&CTensor::zeros(1), [1; 4], 0, 0, 0),
        Err("KptsHelper symmetry map was not built")
    );

    // `_ERIS` builds it lazily, and the result equals the eager one.
    lazy.build_symm_map(None);
    let eager = KptsHelper::new(&lattice(), &kpts(4));
    assert_eq!(lazy.symm_map, eager.symm_map);
}
