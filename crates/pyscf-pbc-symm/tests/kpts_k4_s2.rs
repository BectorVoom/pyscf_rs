//! `KPoints::make_k4_ibz(sym = "s2")` — 17-09's k-quartet set.
//!
//! # Why this one is gated against upstream and not only against invariants
//!
//! 17-05 shipped `sym = "s1"` and left `"s2"` refusing, because at the time
//! its only named consumer was 17-09. `"s2"` folds the dummy-index
//! interchange `(ki, kj, ka, kb) <-> (kj, ki, kb, ka)` on top of the s1
//! stars, in TWO passes (`kpts.py:219-235` then the "refine" pass
//! `:236-273`) — and the second pass is the one that is easy to get subtly
//! wrong, because it does not compare representatives, it walks a whole s1
//! star looking for the interchanged tuple.
//!
//! The oracle-free invariants that DO exist here — the weights summing to 1,
//! `bz2ibz` covering `0..n_s2`, every representative appearing in its own
//! class — are all necessary and none is close to sufficient: a refine pass
//! that merges one pair too many still satisfies every one of them, and it
//! is precisely the *count* that 17-09's speed claim rests on. So the class
//! list, the integer class sizes and the full `bz2ibz` table are pinned
//! against upstream PySCF 2.12.1
//! (`measurements/gate_k4_s2.py` / `gate_k4_s2.out`).
//!
//! The oracle also measured that the s2 tables travel with the space-group
//! TYPE and not with the lattice constant or the time-reversal flag: `si`
//! (a = 5.4306 Å) and `diamond` (a = 3.5668 Å) at `[2,2,2]`, with
//! `time_reversal` on and off, give the **same 36 classes, the same class
//! sizes and the same 512-entry `bz2ibz`**. Both cells are asserted here, so
//! a fixture that silently loses the non-symmorphic operations shows up as a
//! changed count rather than as a slightly wrong energy three plans later.

#![allow(clippy::needless_range_loop)]

use pyscf_pbc_gto::Cell;
use pyscf_pbc_gto::test_systems::{diamond, si};
use pyscf_pbc_symm::PbcSymmError;
use pyscf_pbc_symm::kpts::{KPoints, make_kpts};
use pyscf_pbc_symm::symmetry::build_lattice_symmetry;

// ---------------------------------------------------------------------
// upstream PySCF 2.12.1, `measurements/gate_k4_s2.out`
// ---------------------------------------------------------------------

/// `si` and `diamond` at `[2,2,2]`, `time_reversal` either way: 8 BZ points,
/// 3 IBZ points, 512 k-triples, **50** s1 classes, **36** s2 classes.
const SI222_N_S1: usize = 50;

const SI222_K4_S2: [[usize; 4]; 36] = [
    [0, 0, 0, 0],
    [0, 0, 6, 6],
    [0, 0, 7, 7],
    [0, 6, 0, 6],
    [0, 6, 5, 3],
    [0, 6, 6, 0],
    [0, 6, 7, 1],
    [0, 7, 0, 7],
    [0, 7, 4, 3],
    [0, 7, 6, 1],
    [0, 7, 7, 0],
    [6, 5, 0, 3],
    [6, 5, 5, 6],
    [6, 5, 6, 5],
    [6, 5, 7, 4],
    [6, 6, 0, 0],
    [6, 6, 5, 5],
    [6, 6, 6, 6],
    [6, 6, 7, 7],
    [6, 7, 0, 1],
    [6, 7, 1, 0],
    [6, 7, 4, 5],
    [6, 7, 5, 4],
    [6, 7, 6, 7],
    [6, 7, 7, 6],
    [7, 0, 4, 3],
    [7, 0, 6, 1],
    [7, 4, 0, 3],
    [7, 4, 2, 1],
    [7, 4, 4, 7],
    [7, 4, 6, 5],
    [7, 4, 7, 4],
    [7, 7, 0, 0],
    [7, 7, 4, 4],
    [7, 7, 6, 6],
    [7, 7, 7, 7],
];

/// `weight * nkpts^3` — the exact number of BZ k-triples in each s2 class.
/// Integers, so this is an EXACT comparison, not a tolerance.
const SI222_W_S2_X_N3: [usize; 36] = [
    1, 3, 4, 6, 12, 6, 24, 8, 12, 12, 8, 12, 6, 6, 24, 3, 6, 3, 12, 24, 24, 48, 48, 24, 24, 12, 12,
    24, 24, 12, 24, 12, 4, 12, 12, 4,
];

const SI222_BZ2IBZ_S2: [usize; 512] = [
    0, 2, 2, 1, 2, 1, 1, 2, 7, 10, 8, 9, 8, 9, 9, 8, 7, 8, 10, 9, 8, 9, 9, 8, 3, 6, 6, 5, 6, 4, 4,
    6, 7, 8, 8, 9, 10, 9, 9, 8, 3, 6, 6, 4, 6, 5, 4, 6, 3, 6, 6, 4, 6, 4, 5, 6, 7, 8, 8, 9, 8, 9,
    9, 10, 10, 7, 25, 26, 25, 26, 26, 25, 32, 35, 33, 34, 33, 34, 34, 33, 27, 31, 29, 27, 28, 30,
    30, 28, 20, 23, 19, 24, 22, 21, 21, 22, 27, 31, 28, 30, 29, 27, 30, 28, 20, 23, 22, 21, 19, 24,
    21, 22, 20, 23, 22, 21, 22, 21, 24, 19, 27, 31, 28, 30, 28, 30, 27, 29, 10, 25, 7, 26, 25, 26,
    26, 25, 27, 29, 31, 27, 28, 30, 30, 28, 32, 33, 35, 34, 33, 34, 34, 33, 20, 19, 23, 24, 22, 21,
    21, 22, 27, 28, 31, 30, 29, 30, 27, 28, 20, 22, 23, 21, 22, 24, 21, 19, 20, 22, 23, 21, 19, 21,
    24, 22, 27, 28, 31, 30, 28, 27, 30, 29, 5, 6, 6, 3, 6, 4, 4, 6, 19, 24, 20, 23, 21, 22, 22, 21,
    19, 20, 24, 23, 21, 22, 22, 21, 15, 18, 18, 17, 18, 16, 16, 18, 19, 21, 21, 23, 24, 22, 22, 20,
    11, 14, 14, 13, 14, 12, 11, 14, 11, 14, 14, 13, 14, 11, 12, 14, 19, 21, 21, 23, 20, 22, 22, 24,
    10, 25, 25, 26, 7, 26, 26, 25, 27, 29, 28, 30, 31, 27, 30, 28, 27, 28, 29, 30, 31, 30, 27, 28,
    20, 22, 22, 24, 23, 21, 21, 19, 32, 33, 33, 34, 35, 34, 34, 33, 20, 19, 22, 21, 23, 24, 21, 22,
    20, 22, 19, 21, 23, 21, 24, 22, 27, 28, 28, 27, 31, 30, 30, 29, 5, 6, 6, 4, 6, 3, 4, 6, 19, 24,
    21, 22, 20, 23, 22, 21, 19, 21, 24, 22, 21, 23, 22, 20, 11, 14, 14, 12, 14, 13, 11, 14, 19, 20,
    21, 22, 24, 23, 22, 21, 15, 18, 18, 16, 18, 17, 16, 18, 11, 14, 14, 11, 14, 13, 12, 14, 19, 21,
    20, 22, 21, 23, 22, 24, 5, 6, 6, 4, 6, 4, 3, 6, 19, 24, 21, 22, 21, 22, 23, 20, 19, 21, 24, 22,
    20, 22, 23, 21, 11, 14, 14, 12, 14, 11, 13, 14, 19, 21, 20, 22, 24, 22, 23, 21, 11, 14, 14, 11,
    14, 12, 13, 14, 15, 18, 18, 16, 18, 16, 17, 18, 19, 20, 21, 22, 21, 22, 23, 24, 10, 25, 25, 26,
    25, 26, 26, 7, 27, 29, 28, 30, 28, 30, 27, 31, 27, 28, 29, 30, 28, 27, 30, 31, 20, 22, 22, 24,
    19, 21, 21, 23, 27, 28, 28, 27, 29, 30, 30, 31, 20, 22, 19, 21, 22, 24, 21, 23, 20, 19, 22, 21,
    22, 21, 24, 23, 32, 33, 33, 34, 33, 34, 34, 35,
];

/// `si` at `[1,1,2]`: 2 BZ points, 8 triples, 8 s1 classes, 6 s2 classes.
/// The `[1,1,2]` mesh keeps NO point-group operation that mixes k-points
/// (every s1 class is a singleton), so the whole reduction 8 -> 6 comes from
/// the dummy-index interchange alone. It is the one fixture in this file that
/// isolates the first pass from the refine pass.
const SI112_K4_S2: [[usize; 4]; 6] = [
    [0, 0, 0, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 1],
    [0, 1, 1, 0],
    [1, 1, 0, 0],
    [1, 1, 1, 1],
];
const SI112_W_S2_X_N3: [usize; 6] = [1, 1, 2, 2, 1, 1];
const SI112_BZ2IBZ_S2: [usize; 8] = [0, 1, 2, 3, 3, 2, 4, 5];

fn fixture(mut cell: Cell, mesh: [usize; 3], time_reversal: bool) -> (Cell, KPoints) {
    cell.space_group_symmetry = true;
    cell.symmorphic = false;
    let check_mesh_symmetry = !cell._mesh_from_build;
    build_lattice_symmetry(&mut cell, check_mesh_symmetry).expect("build_lattice_symmetry");
    let kmesh = pyscf_pbc_gto::make_kpts_default(&cell, mesh).expect("make_kpts_default");
    let kpts = make_kpts(&cell, &kmesh, true, time_reversal).expect("make_kpts");
    (cell, kpts)
}

fn assert_against_oracle(
    cell: &Cell,
    kpts: &KPoints,
    k4_ref: &[[usize; 4]],
    w_ref: &[usize],
    bz2ibz_ref: &[usize],
    what: &str,
) {
    let k4 = kpts.make_k4_ibz(cell, "s2").expect("make_k4_ibz s2");
    let n3 = kpts.nkpts().pow(3);

    assert_eq!(
        k4.k4.len(),
        k4_ref.len(),
        "{what}: s2 class COUNT — upstream {} classes, this port {}",
        k4_ref.len(),
        k4.k4.len()
    );
    assert_eq!(k4.k4, k4_ref, "{what}: s2 class representatives");

    let w_int: Vec<usize> = k4
        .weight
        .iter()
        .map(|&w| {
            let x = w * n3 as f64;
            assert!(
                (x - x.round()).abs() < 1e-9,
                "{what}: s2 weight {w} x nkpts^3 = {x} is not an integer"
            );
            x.round() as usize
        })
        .collect();
    assert_eq!(w_int, w_ref, "{what}: s2 class SIZES (weight x nkpts^3)");
    assert_eq!(
        w_int.iter().sum::<usize>(),
        n3,
        "{what}: class sizes must partition the k-triples"
    );

    assert_eq!(k4.bz2ibz.len(), n3, "{what}: bz2ibz length");
    assert_eq!(k4.bz2ibz, bz2ibz_ref, "{what}: bz2ibz");

    // `sym = "s2"` returns only (k4, weight, bz2ibz) upstream (`kpts.py:299`);
    // the three star-op fields have no honest value and come back empty.
    assert!(k4.ibz2bz.is_empty(), "{what}: s2 must not claim an ibz2bz");
    assert!(
        k4.stars_ops.is_empty(),
        "{what}: s2 must not claim stars_ops"
    );
    assert!(
        k4.stars_ops_bz.is_empty(),
        "{what}: s2 must not claim stars_ops_bz"
    );
}

#[test]
fn si_222_s2_matches_upstream() {
    for tr in [false, true] {
        let (cell, kpts) = fixture(si(), [2, 2, 2], tr);
        assert_eq!(kpts.nkpts(), 8);
        assert_eq!(kpts.nkpts_ibz(), 3);
        let s1 = kpts.make_k4_ibz(&cell, "s1").expect("s1");
        assert_eq!(
            s1.k4.len(),
            SI222_N_S1,
            "s1 class count (time_reversal={tr})"
        );
        assert_against_oracle(
            &cell,
            &kpts,
            &SI222_K4_S2,
            &SI222_W_S2_X_N3,
            &SI222_BZ2IBZ_S2,
            &format!("si [2,2,2] time_reversal={tr}"),
        );
    }
}

/// The same 36 classes, sizes and 512-entry table on a cell with a DIFFERENT
/// lattice constant and different atoms. The s2 tables are a property of the
/// space group, not of the geometry — so a fixture that quietly loses the
/// non-symmorphic operations fails here, loudly, on an integer.
#[test]
fn diamond_222_s2_matches_si_222() {
    for tr in [false, true] {
        let (cell, kpts) = fixture(diamond(), [2, 2, 2], tr);
        assert_against_oracle(
            &cell,
            &kpts,
            &SI222_K4_S2,
            &SI222_W_S2_X_N3,
            &SI222_BZ2IBZ_S2,
            &format!("diamond [2,2,2] time_reversal={tr}"),
        );
    }
}

#[test]
fn si_112_s2_isolates_the_dummy_index_fold() {
    for tr in [false, true] {
        let (cell, kpts) = fixture(si(), [1, 1, 2], tr);
        assert_eq!(kpts.nkpts(), 2);
        let s1 = kpts.make_k4_ibz(&cell, "s1").expect("s1");
        assert_eq!(
            s1.k4.len(),
            8,
            "[1,1,2] keeps no k-mixing operation: every s1 class must be a singleton"
        );
        assert_against_oracle(
            &cell,
            &kpts,
            &SI112_K4_S2,
            &SI112_W_S2_X_N3,
            &SI112_BZ2IBZ_S2,
            &format!("si [1,1,2] time_reversal={tr}"),
        );
    }
}

/// The oracle-free half — and the place where the s2 fold's REAL contract is
/// written down.
///
/// The obvious invariant, "a triple and its dummy-index partner share an s2
/// class", is **false, and it is false upstream too**: the refine pass
/// (`kpts.py:236-273`) only searches later representatives whose four
/// k-indices form the right multiset and then walks that representative's s1
/// star, so a partner living in a star whose representative has a different
/// multiset is never found. Measured on `si`/`diamond` `[2,2,2]`,
/// `time_reversal` either way: **48 of 512** triples are separated from their
/// partner (`measurements/gate_k4_s2.out`). An earlier draft of this test
/// asserted the completeness and failed against a port that reproduces
/// upstream exactly — the assertion was wrong, not the port.
///
/// What the fold must satisfy, and what is asserted here, is SOUNDNESS: every
/// member of a class is genuinely equivalent to that class's representative,
/// under the s1 star relation composed with at most one dummy-index
/// interchange. Incompleteness costs k-quartets; unsoundness would put a
/// wrong number in the energy, because the kernel evaluates the
/// representative once and multiplies by the class size.
#[test]
fn s2_classes_are_sound_and_measurably_incomplete() {
    for (name, cell0, mesh, expect_split) in [
        ("si", si(), [2usize, 2, 2], 48usize),
        ("diamond", diamond(), [2, 2, 2], 48),
        ("si", si(), [1, 1, 2], 0),
    ] {
        let (cell, kpts) = fixture(cell0, mesh, false);
        let nk = kpts.nkpts();
        let k4 = kpts.make_k4_ibz(&cell, "s2").expect("s2");
        let s1 = kpts.make_ktuples_ibz(3);
        let kconserv = kpts.get_kconserv(&cell);

        for w in k4.k4.windows(2) {
            assert!(
                w[0] < w[1],
                "{name} {mesh:?}: s2 classes must be ascending lexicographic — \
                 17-09's kernel groups CONSECUTIVE equal (ki, kj)"
            );
        }

        for (i, q) in k4.k4.iter().enumerate() {
            let flat = kpts.ktuple_to_index(&[q[0], q[1], q[2]]);
            assert_eq!(
                k4.bz2ibz[flat], i,
                "{name} {mesh:?}: representative {q:?} is not in its own class"
            );
            assert_eq!(
                kconserv.get(q[0], q[2], q[1]) as usize,
                q[3],
                "{name} {mesh:?}: kb must come from momentum conservation"
            );
        }

        let mut split = 0usize;
        for ki in 0..nk {
            for kj in 0..nk {
                for ka in 0..nk {
                    let kb = kconserv.get(ki, ka, kj) as usize;
                    let t = kpts.ktuple_to_index(&[ki, kj, ka]);
                    let tsw = kpts.ktuple_to_index(&[kj, ki, kb]);
                    let rep = k4.k4[k4.bz2ibz[t]];
                    let trep = kpts.ktuple_to_index(&[rep[0], rep[1], rep[2]]);
                    assert!(
                        s1.bz2ibz[t] == s1.bz2ibz[trep] || s1.bz2ibz[tsw] == s1.bz2ibz[trep],
                        "{name} {mesh:?}: ({ki},{kj},{ka},{kb}) is in class {rep:?} but is \
                         reachable from it by neither the s1 star nor the dummy-index \
                         interchange — the class is UNSOUND and its weight would multiply \
                         the wrong contribution"
                    );
                    if k4.bz2ibz[t] != k4.bz2ibz[tsw] {
                        split += 1;
                    }
                }
            }
        }
        assert_eq!(
            split, expect_split,
            "{name} {mesh:?}: partner-split count — upstream measures {expect_split} \
             (measurements/gate_k4_s2.out); a DIFFERENT number here means the refine \
             pass diverged from upstream's, in either direction"
        );
    }
}

/// `"s4"` stays refusing: `kpts.py:284-292` has NO caller anywhere in
/// upstream's tree, so there is no oracle for it and this port ships no
/// number it cannot check. Anything else keeps upstream's own
/// `NotImplementedError` (`kpts.py:301`).
#[test]
fn s4_and_nonsense_still_refuse() {
    let (cell, kpts) = fixture(si(), [1, 1, 2], false);
    for sym in ["s4", "s8", ""] {
        match kpts.make_k4_ibz(&cell, sym) {
            Err(PbcSymmError::UnsupportedK4Symmetry(got)) => assert_eq!(got, sym),
            other => panic!("make_k4_ibz({sym:?}) must refuse, got {other:?}"),
        }
    }
}

// =====================================================================
// D-17-09-02 — `little_cogroup_ops` can name operations the k-mesh does not
// respect, and that makes `use_ao_symmetry = true` unsound
// =====================================================================

/// `make_kpts_ibz` snapshots `k2opk` **before** wiping the columns of
/// operations that move a k-point out of the mesh (`kpts.py:60-64`), and
/// `little_cogroup_ops` is then filled from that UNWIPED table
/// (`kpts.py:109-113`). So on a k-mesh with lower symmetry than the lattice,
/// `little_cogroup_ops[i]` carries the full little co-group of the LATTICE at
/// `kpts_ibz[i]` — including operations the mesh does not respect.
///
/// `symm_adapted_basis` builds `symm_orb` from that group and `eig` then solves
/// `F c = S c e` one irrep block at a time. **Schur's lemma justifies that only
/// if `F` has no matrix elements BETWEEN the blocks, and it does have them**:
/// `v_J` is built from `rho(r) = Σ_k rho_k(r)` over a k-mesh that is not
/// point-group invariant, so neither `rho` nor `v_J` is invariant.
///
/// The consequence is subtler than "the SCF lands somewhere else". A per-block
/// solve of a non-block-diagonal Fock still spans the right OCCUPIED SUBSPACE,
/// so the density and `E_scf` are unaffected (measured **5.329e-15**). What is
/// lost is CANONICALITY — the vectors are eigenvectors of the projected Fock,
/// not of `F` — and every post-SCF method whose denominators assume `F` is
/// diagonal in the MO basis is then wrong. Measured downstream: `si [1,1,2]`
/// moves `KMP2`'s `e_corr` by **3.629e-04** against the full-BZ run and by
/// **0e0** with `use_ao_symmetry = false`, while `si [2,2,2]` — whose mesh IS
/// closed under the cubic group — lands at **1.138e-10** with it on
/// (`crates/pyscf-pbc-mp/tests/kmp2_ksymm.rs`).
///
/// **This test needs no SCF**: the whole statement is a property of the
/// k-point tables, so it costs milliseconds and cannot be starved by a
/// fixture's cost.
#[test]
fn little_cogroup_ops_can_name_operations_the_kmesh_does_not_respect() {
    // `[2,2,2]` on fcc IS closed under the cubic group: nothing is wiped, and
    // `use_ao_symmetry = true` is sound there.
    let (_cell, k222) = fixture(si(), [2, 2, 2], false);
    let wiped222 = k222.ops_outside_kmesh_subgroup();
    let bad222 = k222.little_cogroup_ops_outside_kmesh_subgroup();
    println!(
        "si [2,2,2]: {} of {} operations outside the k-mesh subgroup, {} little-co-group hits",
        wiped222.iter().filter(|b| **b).count(),
        wiped222.len(),
        bad222.len()
    );
    assert!(
        bad222.is_empty(),
        "si [2,2,2] must be a symmetry-closed mesh — every k-symmetric fixture \
         in this phase relies on it: {bad222:?}"
    );

    // `[1,1,2]` is NOT: the point group maps `b3/2` onto `b1/2` and `b2/2`,
    // which are not in the mesh.
    let (_cell, k112) = fixture(si(), [1, 1, 2], false);
    let wiped112 = k112.ops_outside_kmesh_subgroup();
    let bad112 = k112.little_cogroup_ops_outside_kmesh_subgroup();
    println!(
        "si [1,1,2]: {} of {} operations outside the k-mesh subgroup, {} little-co-group hits",
        wiped112.iter().filter(|b| **b).count(),
        wiped112.len(),
        bad112.len()
    );
    assert!(
        wiped112.iter().any(|b| *b),
        "si [1,1,2] must be a symmetry-BREAKING mesh; if it is not, this test's \
         premise is gone and the finding needs a different fixture"
    );
    assert!(
        !bad112.is_empty(),
        "si [1,1,2] must have at least one little-co-group operation outside the \
         k-mesh subgroup — that is the whole of D-17-09-02"
    );

    // Every reported hit must really be both: in the little co-group, and
    // outside the mesh subgroup. A detector that over-reports would make the
    // warning noise.
    for &(i, io) in &bad112 {
        assert!(
            k112.little_cogroup_ops[i].contains(&io),
            "op {io} is not in little_cogroup_ops[{i}]"
        );
        assert!(wiped112[io], "op {io} is not outside the k-mesh subgroup");
        assert_eq!(
            k112.k2opk[k112.ibz2bz[i]][io], k112.ibz2bz[i] as i64,
            "little_cogroup_ops[{i}] must only contain ops that FIX that k-point"
        );
    }
}
