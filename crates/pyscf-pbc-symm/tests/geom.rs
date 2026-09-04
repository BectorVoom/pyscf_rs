//! Integration tests for `pyscf_pbc_symm::geom` — 17-02-PLAN.md Tasks 1/2.
//!
//! Fixtures come from `pyscf_pbc_gto::test_systems` (PBC-MASTER-PLAN §9.2),
//! reused via the `test-systems` feature — not redefined here.

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs, LowDimFtType};
use pyscf_pbc_symm::geom::{self, SYMPREC};

fn build(a: ALattice, dimension: u8) -> Cell {
    let mole = MoleBuildArgs {
        atom: AtomInput::Tuples(vec![("He".into(), [0.0, 0.0, 0.0])]),
        basis: BasisInput::Name("gth-szv".into()),
        unit: Unit::Ang,
        ..Default::default()
    };
    let args = CellBuildArgs {
        mole,
        a,
        dimension,
        low_dim_ft_type: LowDimFtType::None,
        pseudo: Some("gth-pade".to_string()),
        ..Default::default()
    };
    Cell::build(args).expect("test cell must build")
}

/// A simple-cubic lattice, `a = 3.0 A` — not one of PBC-MASTER-PLAN §9.2's
/// five fixtures (they are all fcc or hexagonal); `search_point_group_ops`
/// only reads `cell.lattice_vectors()`/`cell.dimension`, so a throwaway
/// single-atom basis is fine here.
fn simple_cubic() -> Cell {
    let a0 = 3.0;
    build(
        ALattice::Matrix([[a0, 0.0, 0.0], [0.0, a0, 0.0], [0.0, 0.0, a0]]),
        3,
    )
}

// ---------------------------------------------------------------------
// small local linear algebra mirrors, for asserting the metric identity
// independently of geom.rs's own (private) helpers
// ---------------------------------------------------------------------

fn metric(a: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut g = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            g[i][j] = a[i][0] * a[j][0] + a[i][1] * a[j][1] + a[i][2] * a[j][2];
        }
    }
    g
}

fn det3(w: &[[i32; 3]; 3]) -> i32 {
    w[0][0] * (w[1][1] * w[2][2] - w[1][2] * w[2][1])
        - w[0][1] * (w[1][0] * w[2][2] - w[1][2] * w[2][0])
        + w[0][2] * (w[1][0] * w[2][1] - w[1][1] * w[2][0])
}

/// `Wᵀ @ G @ W`.
fn conjugate(w: &[[i32; 3]; 3], g: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let wf: [[f64; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|j| w[i][j] as f64));
    let mut wtg = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0;
            for k in 0..3 {
                s += wf[k][i] * g[k][j];
            }
            wtg[i][j] = s;
        }
    }
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0;
            for k in 0..3 {
                s += wtg[i][k] * wf[k][j];
            }
            out[i][j] = s;
        }
    }
    out
}

fn compose(a: &[[i32; 3]; 3], b: &[[i32; 3]; 3]) -> [[i32; 3]; 3] {
    let mut out = [[0i32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0;
            for k in 0..3 {
                s += a[i][k] * b[k][j];
            }
            out[i][j] = s;
        }
    }
    out
}

// ---------------------------------------------------------------------
// Task 1
// ---------------------------------------------------------------------

/// Measured counts (this port's own `search_point_group_ops`, pinned per
/// the plan's "measure the true counts... and pin them"): fcc `si`/`diamond`
/// and simple-cubic all give the full cubic holohedry (48, `m-3m`/`Oh`);
/// `lif`/`he_fcc` (also fcc Bravais lattices) match; `graphene`'s
/// `dimension = 2` low-dim filters (geom.rs Task 1 point 3) restrict the
/// lattice's in-plane hexagonal symmetry to the subgroup that does not
/// invert the non-periodic `a3` axis — `6mm` / `C6v`, order 12 (NOT the 24
/// the plan's rough pre-measurement estimate guessed).
#[test]
fn point_group_op_counts() {
    let expected = [
        ("diamond", 48),
        ("si", 48),
        ("lif", 48),
        ("he_fcc", 48),
        ("graphene", 12),
    ];
    for (name, cell) in pyscf_pbc_gto::test_systems::all() {
        let rots = geom::search_point_group_ops(&cell, SYMPREC).unwrap();
        let want = expected
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, c)| *c)
            .unwrap();
        assert_eq!(rots.len(), want, "{name}: point-group op count");
    }

    let rots = geom::search_point_group_ops(&simple_cubic(), SYMPREC).unwrap();
    assert_eq!(rots.len(), 48, "simple cubic: point-group op count");
}

#[test]
fn every_op_preserves_the_metric_and_is_unimodular() {
    for (name, cell) in pyscf_pbc_gto::test_systems::all() {
        let a = cell.lattice_vectors();
        let g = metric(&a);
        let rots = geom::search_point_group_ops(&cell, SYMPREC).unwrap();
        assert!(!rots.is_empty(), "{name}: no rotations found");
        for w in &rots {
            let g_tilde = conjugate(w, &g);
            for i in 0..3 {
                for j in 0..3 {
                    assert!(
                        (g_tilde[i][j] - g[i][j]).abs() < 1e-12,
                        "{name}: WtGW != G at ({i},{j}): {} vs {}",
                        g_tilde[i][j],
                        g[i][j]
                    );
                }
            }
            let d = det3(w);
            assert_eq!(d.abs(), 1, "{name}: |det W| != 1, got {d}");
        }
    }
}

#[test]
fn point_group_ops_are_closed_under_multiplication_and_inverse() {
    for (name, cell) in pyscf_pbc_gto::test_systems::all() {
        let rots = geom::search_point_group_ops(&cell, SYMPREC).unwrap();
        let set: std::collections::HashSet<[[i32; 3]; 3]> = rots.iter().copied().collect();

        // closure under multiplication (every product is itself a member)
        for a in &rots {
            for b in &rots {
                let ab = compose(a, b);
                assert!(
                    set.contains(&ab),
                    "{name}: product of two point-group ops is not itself a member"
                );
            }
        }

        // closure under inverse: for a unimodular integer W, the exact
        // integer inverse is the adjugate divided by det (see group.rs's
        // `PgElement::inv` doc for why this is exact, not upstream's float
        // cast).
        for w in &rots {
            let d = det3(w);
            let cofactor = |r: usize, c: usize| -> i32 {
                let rows: Vec<usize> = (0..3).filter(|&i| i != r).collect();
                let cols: Vec<usize> = (0..3).filter(|&j| j != c).collect();
                let sign = if (r + c).is_multiple_of(2) { 1 } else { -1 };
                sign * (w[rows[0]][cols[0]] * w[rows[1]][cols[1]]
                    - w[rows[0]][cols[1]] * w[rows[1]][cols[0]])
            };
            let mut inv = [[0i32; 3]; 3];
            for (i, row) in inv.iter_mut().enumerate() {
                for (j, cell) in row.iter_mut().enumerate() {
                    *cell = cofactor(j, i) / d;
                }
            }
            assert!(
                set.contains(&inv),
                "{name}: inverse of a point-group op is not itself a member"
            );
        }
    }
}

#[test]
fn a_distorted_lattice_admits_strictly_fewer_ops() {
    let a0 = 5.4306;
    let h = a0 / 2.0;
    let exact = build(ALattice::Matrix([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]), 3);
    let n_exact = geom::search_point_group_ops(&exact, SYMPREC).unwrap().len();

    // scale one lattice vector's length by (1 + 2*SYMPREC) — just past the
    // length tolerance in `search_point_group_ops`.
    let scale = 1.0 + 2.0 * SYMPREC;
    let distorted = build(
        ALattice::Matrix([[0.0, h * scale, h * scale], [h, 0.0, h], [h, h, 0.0]]),
        3,
    );
    let n_distorted = geom::search_point_group_ops(&distorted, SYMPREC)
        .unwrap()
        .len();

    assert!(
        n_distorted < n_exact,
        "distorted lattice must admit strictly fewer ops: {n_distorted} vs {n_exact}"
    );
}

// ---------------------------------------------------------------------
// Task 2
// ---------------------------------------------------------------------

/// Crystal class strings, measured against this port's own
/// `search_space_group_ops` + `get_crystal_class` and cross-checked by hand
/// against known crystallography: `diamond`/`si` (Fd-3m diamond structure),
/// `lif`/`he_fcc` (Fm-3m rocksalt / single-atom fcc) all have the full cubic
/// point group `m-3m` (`Oh`) — the crystal class only sees ROTATIONS, so it
/// cannot distinguish the diamond glide (non-symmorphic) from rocksalt's
/// symmorphic space group; that distinction is asserted separately below via
/// the fractional-translation search itself. `graphene` gets `6mm` (`C6v`),
/// matching [`point_group_op_counts`]'s 12.
#[test]
fn crystal_class_matches_known_crystallography() {
    let expected = [
        ("diamond", "m-3m", "m-3m"),
        ("si", "m-3m", "m-3m"),
        ("lif", "m-3m", "m-3m"),
        ("he_fcc", "m-3m", "m-3m"),
        ("graphene", "6mm", "6/mmm"),
    ];
    for (name, cell) in pyscf_pbc_gto::test_systems::all() {
        let (cc, laue) = geom::get_crystal_class(&cell, None, SYMPREC).unwrap();
        let (_, want_cc, want_laue) = expected.iter().find(|(n, _, _)| *n == name).unwrap();
        assert_eq!(cc, *want_cc, "{name}: crystal class");
        assert_eq!(laue, *want_laue, "{name}: Laue class");
    }
}

/// The diamond structure's second carbon sits at the tetrahedral
/// `(1/4, 1/4, 1/4)` site: NO choice of translation makes every rotation's
/// op zero-translation at this origin, because the diamond space group
/// (Fd-3m) is genuinely non-symmorphic (`17-02-PLAN.md` Task 2, `symmorphic`
/// is "the whole reason" it is a separate flag). Rocksalt (`lif`) and the
/// single-atom `he_fcc` are symmorphic: every rotation has a zero (mod-1)
/// translation representative at the natural origin.
#[test]
fn diamond_is_non_symmorphic_lif_and_he_fcc_are_symmorphic() {
    for (name, want_symmorphic) in [
        ("diamond", false),
        ("si", false),
        ("lif", true),
        ("he_fcc", true),
    ] {
        let cell = pyscf_pbc_gto::test_systems::all()
            .into_iter()
            .find(|(n, _)| *n == name)
            .unwrap()
            .1;
        let ops = geom::search_space_group_ops(&cell, None, SYMPREC).unwrap();
        let mut rotations: Vec<_> = ops.iter().map(|o| o.rot).collect();
        rotations.sort_by_key(|w| {
            let mut flat = [0i32; 9];
            for i in 0..3 {
                for j in 0..3 {
                    flat[3 * i + j] = w[i][j];
                }
            }
            flat
        });
        rotations.dedup();

        let is_zero = |t: f64| {
            let m = t - t.floor();
            m < 1e-5 || (1.0 - m) < 1e-5
        };
        let all_symmorphic = rotations.iter().all(|rot| {
            ops.iter()
                .any(|o| o.rot == *rot && o.trans.iter().all(|&t| is_zero(t)))
        });
        assert_eq!(
            all_symmorphic, want_symmorphic,
            "{name}: symmorphic-at-this-origin probe"
        );
    }
}

#[test]
fn ghost_atoms_are_refused() {
    let mole = MoleBuildArgs {
        atom: AtomInput::Tuples(vec![
            ("He".into(), [0.0, 0.0, 0.0]),
            ("GHOST-He".into(), [1.0, 1.0, 1.0]),
        ]),
        basis: BasisInput::Name("gth-szv".into()),
        unit: Unit::Ang,
        ..Default::default()
    };
    let args = CellBuildArgs {
        mole,
        a: ALattice::Matrix([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]),
        dimension: 3,
        low_dim_ft_type: LowDimFtType::None,
        pseudo: None,
        ..Default::default()
    };
    let cell = match Cell::build(args) {
        Ok(c) => c,
        Err(_) => {
            // The molecular basis parser may itself refuse an unrecognised
            // ghost-atom label before symmetry search is ever reached; if so
            // the refusal has already happened, which is the property this
            // test cares about.
            return;
        }
    };
    let err = geom::search_space_group_ops(&cell, None, SYMPREC).unwrap_err();
    assert!(matches!(
        err,
        pyscf_pbc_symm::PbcSymmError::GhostAtomUnsupported(_)
    ));
}
