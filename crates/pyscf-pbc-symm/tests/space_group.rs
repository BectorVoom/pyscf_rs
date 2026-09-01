//! Integration tests for `pyscf_pbc_symm::space_group` — 17-03-PLAN.md
//! Tasks 1/2.
//!
//! Fixtures come from `pyscf_pbc_gto::test_systems` (PBC-MASTER-PLAN §9.2),
//! reused via the `test-systems` feature — not redefined here, per 17-02's
//! precedent (`tests/geom.rs`, `tests/group.rs`).

use pyscf_pbc_gto::Cell;
use pyscf_pbc_gto::test_systems::{diamond, graphene, he_fcc, lif, si};
use pyscf_pbc_symm::error::PbcSymmError;
use pyscf_pbc_symm::space_group::{SPGElement, SpaceGroup, XYZ};

fn max_abs_diff(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> f64 {
    let mut m = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            m = m.max((a[i][j] - b[i][j]).abs());
        }
    }
    m
}

fn max_abs_diff3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    (0..3).map(|i| (a[i] - b[i]).abs()).fold(0.0, f64::max)
}

// ---------------------------------------------------------------------
// Task 1 — SPGElement
// ---------------------------------------------------------------------

/// `XYZ` is the Cartesian identity basis, `3x3`.
#[test]
fn xyz_is_identity() {
    assert_eq!(XYZ, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
}

/// `a2b` then `b2a` round-trips to 1e-13, for EVERY operation, on both `si`
/// (cubic) and `graphene` (hexagonal) — this pair never hits the `b2r`
/// asymmetry (see `space_group.rs`'s `b2r` doc comment), so it must always
/// succeed.
fn assert_a2b_b2a_roundtrips(cell: &Cell) {
    let sg = SpaceGroup::build(cell, 1e-6).expect("space group");
    for op in &sg.ops {
        let b = op.a2b(cell).expect("a2b must succeed for a genuine point-group op");
        let back = b.b2a(cell).expect("b2a must succeed");
        assert!(
            max_abs_diff(&back.rot, &op.rot) < 1e-13,
            "a2b/b2a rot round-trip: {:?} != {:?}",
            back.rot,
            op.rot
        );
        assert!(
            max_abs_diff3(&back.trans, &op.trans) < 1e-13,
            "a2b/b2a trans round-trip: {:?} != {:?}",
            back.trans,
            op.trans
        );
    }
}

#[test]
fn a2b_b2a_roundtrip_si() {
    assert_a2b_b2a_roundtrips(&si());
}

#[test]
fn a2b_b2a_roundtrip_graphene() {
    assert_a2b_b2a_roundtrips(&graphene());
}

/// `a2r` then `r2a` round-trips to 1e-13 for EVERY op, on both fixtures —
/// `a2r` is explicitly `allow_non_integer = true`, and `r2a` of a genuine
/// Cartesian point-group rotation is always exactly integer by construction.
fn assert_a2r_r2a_roundtrips(cell: &Cell) {
    let sg = SpaceGroup::build(cell, 1e-6).expect("space group");
    for op in &sg.ops {
        let r = op.a2r(cell).expect("a2r must always succeed (allow_non_integer=true)");
        let back = r.r2a(cell).expect("r2a of a genuine op must round-trip to integers");
        assert!(
            max_abs_diff(&back.rot, &op.rot) < 1e-13,
            "a2r/r2a rot round-trip: {:?} != {:?}",
            back.rot,
            op.rot
        );
        assert!(
            max_abs_diff3(&back.trans, &op.trans) < 1e-13,
            "a2r/r2a trans round-trip: {:?} != {:?}",
            back.trans,
            op.trans
        );
    }
}

#[test]
fn a2r_r2a_roundtrip_si() {
    assert_a2r_r2a_roundtrips(&si());
}

#[test]
fn a2r_r2a_roundtrip_graphene() {
    assert_a2r_r2a_roundtrips(&graphene());
}

/// `b2r` then `r2b` round-trips to 1e-13 EVERYWHERE it succeeds. On `si`
/// (cubic Fd-3m) every op's Cartesian representation is a signed permutation
/// matrix, so `b2r` succeeds for all 48 ops — ported verbatim, this is
/// upstream's own (undocumented) `allow_non_integer=False` default on `b2r`
/// (`space_group.py:225-229`) happening to be harmless on a cubic lattice.
#[test]
fn b2r_r2b_roundtrip_si() {
    let cell = si();
    let sg = SpaceGroup::build(&cell, 1e-6).expect("space group");
    let mut n_ok = 0;
    for op in &sg.ops {
        let b = op.a2b(&cell).expect("a2b");
        let r = b.b2r(&cell).expect("b2r must succeed on every op of a cubic lattice");
        let back = r.r2b(&cell).expect("r2b");
        assert!(max_abs_diff(&back.rot, &b.rot) < 1e-13);
        assert!(max_abs_diff3(&back.trans, &b.trans) < 1e-13);
        n_ok += 1;
    }
    assert_eq!(n_ok, sg.ops.len());
}

/// On `graphene` (hexagonal), `b2r`'s upstream-inherited `allow_non_integer
/// = false` default (see `space_group.rs`'s doc comment on `SPGElement::b2r`)
/// genuinely RAISES for the 3-/6-fold rotations, whose Cartesian
/// representation has irrational entries (`cos(120 deg) = -0.5`). This test
/// pins that this is EXPECTED — ported faithfully from upstream (RULE 2),
/// verified against live upstream 2.12.1 — and that every op where it DOES
/// succeed (the identity, the mirrors, and the 2-fold axis) round-trips.
#[test]
fn b2r_r2b_graphene_partial_by_design() {
    let cell = graphene();
    let sg = SpaceGroup::build(&cell, 1e-6).expect("space group");
    let mut n_ok = 0;
    let mut n_non_integer = 0;
    for op in &sg.ops {
        let b = op.a2b(&cell).expect("a2b");
        match b.b2r(&cell) {
            Ok(r) => {
                let back = r.r2b(&cell).expect("r2b of a genuine op must succeed");
                assert!(max_abs_diff(&back.rot, &b.rot) < 1e-13);
                assert!(max_abs_diff3(&back.trans, &b.trans) < 1e-13);
                n_ok += 1;
            }
            Err(PbcSymmError::NonIntegerRotation) => {
                n_non_integer += 1;
            }
            Err(e) => panic!("unexpected error from b2r: {e}"),
        }
    }
    assert!(n_ok > 0, "identity/mirror ops must still succeed");
    assert!(
        n_non_integer > 0,
        "graphene's 3-/6-fold rotations must hit NonIntegerRotation via b2r"
    );
    assert_eq!(n_ok + n_non_integer, sg.ops.len());
}

/// Diamond's non-symmorphic glide: at least one op has `trans_is_zero() ==
/// false`. LiF (symmorphic Fm-3m): every op has `trans_is_zero() == true`.
/// This single pair of assertions is what distinguishes symmorphic from not.
#[test]
fn diamond_has_a_glide_lif_does_not() {
    let dcell = diamond();
    let dsg = SpaceGroup::build(&dcell, 1e-6).expect("diamond space group");
    assert!(
        dsg.ops.iter().any(|op| !op.trans_is_zero()),
        "diamond must have at least one non-symmorphic op (a glide)"
    );

    let lcell = lif();
    let lsg = SpaceGroup::build(&lcell, 1e-6).expect("lif space group");
    assert!(
        lsg.ops.iter().all(|op| op.trans_is_zero()),
        "lif is symmorphic — every op must have trans_is_zero() == true"
    );
}

/// `SPGElement::default()` is the identity: `rot = I`, `trans = 0`.
#[test]
fn default_is_identity() {
    let e = SPGElement::default();
    assert!(e.is_eye());
    assert!(!e.is_inversion());
}

/// `dot`/`inv` satisfy the group axioms on a concrete op: `op.dot(&op.inv())`
/// is the identity, and `op.dot(&e) == op == e.dot(&op)`.
#[test]
fn dot_and_inv_satisfy_group_axioms() {
    let cell = diamond();
    let sg = SpaceGroup::build(&cell, 1e-6).expect("space group");
    let e = SPGElement::default();
    for op in &sg.ops {
        let inv = op.inv().expect("inv");
        let should_be_eye = op.dot(&inv);
        assert!(
            should_be_eye.is_eye(),
            "op * op^-1 != e for rot={:?} trans={:?}",
            op.rot,
            op.trans
        );
        assert_eq!(op.dot(&e), *op);
        assert_eq!(e.dot(op), *op);
    }
}

/// Total order (`Ord`) round-trips through a sort: sorting twice is a no-op,
/// and `SpaceGroup::build` always returns its `ops` already sorted.
#[test]
fn ops_are_sorted_and_order_is_idempotent() {
    let cell = si();
    let sg = SpaceGroup::build(&cell, 1e-6).expect("space group");
    let mut resorted = sg.ops.clone();
    resorted.sort();
    assert_eq!(resorted, sg.ops, "SpaceGroup::build must return sorted ops");
}

// ---------------------------------------------------------------------
// Task 2 — SpaceGroup
// ---------------------------------------------------------------------

/// `groupname` (point-group symbol) for all five §9.2 reference cells,
/// against upstream's `dump_info` output (`.venv` PySCF 2.12.1, same
/// geometry/basis/pseudo as `test_systems.rs`):
///
/// | cell     | point group |
/// |----------|-------------|
/// | diamond  | `m-3m`      |
/// | si       | `m-3m`      |
/// | lif      | `m-3m`      |
/// | he_fcc   | `m-3m`      |
/// | graphene | `6mm`       |
#[test]
fn point_group_symbol_matches_upstream() {
    let cases: &[(&str, fn() -> Cell, &str)] = &[
        ("diamond", diamond, "m-3m"),
        ("si", si, "m-3m"),
        ("lif", lif, "m-3m"),
        ("he_fcc", he_fcc, "m-3m"),
        ("graphene", graphene, "6mm"),
    ];
    for (name, build, want) in cases {
        let cell = build();
        let sg = SpaceGroup::build(&cell, 1e-6).expect("space group");
        assert_eq!(sg.point_group_symbol, *want, "point group mismatch for {name}");
    }
}

/// `nop` for all five §9.2 reference cells against upstream, and against
/// `search_point_group_ops`'s rotation count — every rotation this fixture
/// set admits pairs with EXACTLY one fractional translation (no rotation is
/// ever dropped or duplicated by the translation search).
#[test]
fn nop_matches_upstream_and_rotation_count() {
    use pyscf_pbc_symm::geom::search_point_group_ops;

    let cases: &[(&str, fn() -> Cell, usize)] = &[
        ("diamond", diamond, 48),
        ("si", si, 48),
        ("lif", lif, 48),
        ("he_fcc", he_fcc, 48),
        ("graphene", graphene, 12),
    ];
    for (name, build, want_nop) in cases {
        let cell = build();
        let sg = SpaceGroup::build(&cell, 1e-6).expect("space group");
        assert_eq!(sg.nop, *want_nop, "nop mismatch for {name}");
        assert_eq!(sg.nop, sg.ops.len());

        let rots = search_point_group_ops(&cell, 1e-6).expect("point group ops");
        assert_eq!(
            sg.nop,
            rots.len(),
            "{name}: nop must equal the point-group rotation count \
             (one translation per rotation for these fixtures)"
        );
    }
}

/// Diamond's split between symmorphic and non-symmorphic ops: 24 have
/// `trans_is_zero() == true`, 24 have `trans_is_zero() == false` (verified
/// against upstream 2.12.1 above).
#[test]
fn diamond_op_split_matches_upstream() {
    let cell = diamond();
    let sg = SpaceGroup::build(&cell, 1e-6).expect("space group");
    let n_zero = sg.ops.iter().filter(|op| op.trans_is_zero()).count();
    let n_nonzero = sg.ops.iter().filter(|op| !op.trans_is_zero()).count();
    assert_eq!(n_zero, 24);
    assert_eq!(n_nonzero, 24);
}

/// `graphene`'s split: 6 symmorphic, 6 non-symmorphic (verified above).
#[test]
fn graphene_op_split_matches_upstream() {
    let cell = graphene();
    let sg = SpaceGroup::build(&cell, 1e-6).expect("space group");
    let n_zero = sg.ops.iter().filter(|op| op.trans_is_zero()).count();
    let n_nonzero = sg.ops.iter().filter(|op| !op.trans_is_zero()).count();
    assert_eq!(n_zero, 6);
    assert_eq!(n_nonzero, 6);
}

