//! Plan 09-07 acceptance gate — `make_kpts`, `kpts_helper`, `get_kconserv`.
//!
//! Tier 1 (invariants, no upstream needed) comes first and must pass
//! unconditionally. Tier 2 pins hard-coded upstream numbers (D-PBC-19); every
//! literal lives in `tests/common/kpts_reference.rs` and came from the snippet
//! in [`UPSTREAM_SNIPPET`].
//!
//! [`diamond_bohr`] specifies the lattice DIRECTLY in Bohr, so the 4.95e-9
//! Angstrom -> Bohr gap (09-03) never enters and absolute k-points can be
//! compared at 1e-12 instead of a loosened bound. The `kconserv` tables are
//! integer and depend only on the k-point topology, so they are asserted on the
//! §9.2 systems too.

mod common;

use common::systems;
use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_gto::kpts_mesh::{
    KIdx, WITH_GAMMA, WRAP_AROUND, get_kconserv, get_kconserv3, intersection, is_gamma_point,
    is_zero, is_trim, make_kpts, make_kpts_default, member, unique,
};
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs};

/// The exact generating snippet for every tier-2 literal (PySCF 2.12.1,
/// `.venv/bin/python`).
///
/// ```python
/// import numpy as np
/// from pyscf.pbc import gto
/// from pyscf.pbc.lib import kpts_helper as kh
/// H = 3.3701375705493315
/// c = gto.Cell()
/// c.a = [[0., H, H], [H, 0., H], [H, H, 0.]]
/// c.atom = [('C', (0., 0., 0.)), ('C', (H/2, H/2, H/2))]
/// c.basis = 'gth-szv'; c.pseudo = 'gth-pade'; c.unit = 'Bohr'; c.verbose = 0
/// c.build()
/// c.make_kpts([2, 2, 2])
/// c.make_kpts([2, 2, 2], with_gamma_point=False)
/// c.make_kpts([2, 2, 2], wrap_around=True)
/// c.make_kpts([2, 2, 2], with_gamma_point=False, wrap_around=True)
/// c.make_kpts([3, 2, 1])
/// c.make_kpts([3, 3, 3], wrap_around=True)
/// c.make_kpts([2, 2, 2], scaled_center=[0.1, 0.2, 0.3])
/// kh.get_kconserv(c, c.make_kpts([2, 2, 2])).ravel()
/// kh.get_kconserv(c, c.make_kpts([3, 2, 1])).ravel()
/// r = np.arange(4)
/// kh.get_kconserv3(c, c.make_kpts([2, 2, 1]), [r, 1, r, r, r]).ravel()
/// kh.unique(np.array([[0.,0.,0.],[0.5,0.,0.],[0.,0.,0.],[0.5,0.,0.],[0.25,0.25,0.25]]))
/// ```
const UPSTREAM_SNIPPET: &str = "see the doc comment above";

/// Diamond lattice in BOHR — upstream's own converted value, so no Angstrom
/// conversion enters. Matches `pyscf_pbc_gto::test_systems::diamond` in every
/// other respect.
const A_BOHR: [[f64; 3]; 3] = [
    [0.0, 3.3701375705493315, 3.3701375705493315],
    [3.3701375705493315, 0.0, 3.3701375705493315],
    [3.3701375705493315, 3.3701375705493315, 0.0],
];

fn diamond_bohr() -> Cell {
    let q = 3.3701375705493315 / 2.0;
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![("C".into(), [0.0; 3]), ("C".into(), [q, q, q])]),
            basis: BasisInput::Name("gth-szv".into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix(A_BOHR),
        pseudo: Some("gth-pade".into()),
        ..Default::default()
    })
    .expect("diamond_bohr must build")
}

/// One row of `make_kpts_matches_upstream`'s table:
/// `(tag, nks, wrap_around, with_gamma_point, scaled_center, expected)`.
type MakeKptsCase = (
    &'static str,
    [usize; 3],
    bool,
    bool,
    Option<[f64; 3]>,
    &'static [[f64; 3]],
);

fn assert_kpts_eq(got: &[[f64; 3]], want: &[[f64; 3]], tol: f64, tag: &str) {
    assert_eq!(got.len(), want.len(), "{tag}: nkpts");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        for k in 0..3 {
            assert!(
                (g[k] - w[k]).abs() < tol,
                "{tag}: kpt {i} axis {k}: {} != {}",
                g[k],
                w[k]
            );
        }
    }
}

// =========================================================================
// TIER 1 — invariants.
// =========================================================================

#[test]
fn make_kpts_count_order_and_gamma_placement() {
    let cell = systems::diamond();
    for nks in [[1, 1, 1], [2, 2, 2], [3, 2, 1], [4, 4, 4], [2, 1, 3]] {
        let kpts = make_kpts_default(&cell, nks).expect("make_kpts");
        assert_eq!(kpts.len(), nks[0] * nks[1] * nks[2], "{nks:?}");
        // cell.py:852-853 — "Gamma point is placed at the first place".
        assert!(is_gamma_point(&kpts[0]), "{nks:?}: kpts[0] = {:?}", kpts[0]);
        assert_eq!(kpts[0], [0.0, 0.0, 0.0], "{nks:?}");
    }
}

#[test]
fn make_kpts_scaled_grid_is_the_expected_cartesian_product() {
    // Round-tripping through get_scaled_kpts must recover i/n on each axis,
    // last index fastest.
    let cell = systems::diamond();
    let nks = [3, 2, 4];
    let kpts = make_kpts_default(&cell, nks).expect("make_kpts");
    let scaled = cell.get_scaled_kpts(&kpts);
    let mut it = scaled.iter();
    for i in 0..nks[0] {
        for j in 0..nks[1] {
            for k in 0..nks[2] {
                let s = it.next().expect("enough kpts");
                let want = [
                    i as f64 / nks[0] as f64,
                    j as f64 / nks[1] as f64,
                    k as f64 / nks[2] as f64,
                ];
                for ax in 0..3 {
                    assert!((s[ax] - want[ax]).abs() < 1e-12, "({i},{j},{k}) axis {ax}");
                }
            }
        }
    }
}

#[test]
fn make_kpts_without_gamma_point_has_no_gamma() {
    let cell = systems::diamond();
    let kpts = make_kpts(&cell, [2, 2, 2], false, false, None).expect("make_kpts");
    assert_eq!(kpts.len(), 8);
    assert!(kpts.iter().all(|k| !is_zero(k)));
    // Every scaled coordinate is (i + 0.5)/n - 0.5, i.e. +/-0.25 for n = 2.
    for s in cell.get_scaled_kpts(&kpts) {
        for x in s {
            assert!((x.abs() - 0.25).abs() < 1e-12, "scaled {x}");
        }
    }
    // The shifted grid is symmetric about the origin.
    let sum: f64 = kpts.iter().flat_map(|k| k.iter()).sum();
    assert!(sum.abs() < 1e-12, "shifted grid must be centred, got {sum}");
}

#[test]
fn make_kpts_wrap_around_folds_each_axis_into_the_first_bz() {
    let cell = systems::diamond();
    let kpts = make_kpts(&cell, [3, 3, 3], true, true, None).expect("make_kpts");
    for s in cell.get_scaled_kpts(&kpts) {
        for x in s {
            assert!(
                (-0.5..0.5).contains(&(x + 1e-12)),
                "scaled {x} outside [-0.5, 0.5)"
            );
        }
    }
    // wrap_around only shifts a point by a whole reciprocal lattice vector, so
    // the SET of scaled coordinates is unchanged modulo 1. Compared as an
    // unordered multiset: a lexicographic sort is not stable here, because the
    // fold reintroduces last-ulp differences (0.33333333333333326 vs
    // 0.3333333333333333) that reorder otherwise-equal rows.
    let plain = make_kpts(&cell, [3, 3, 3], false, true, None).expect("make_kpts");
    let fold = |ks: &[[f64; 3]]| -> Vec<[f64; 3]> {
        cell.get_scaled_kpts(ks)
            .iter()
            .map(|s| {
                [
                    s[0].rem_euclid(1.0),
                    s[1].rem_euclid(1.0),
                    s[2].rem_euclid(1.0),
                ]
            })
            .collect()
    };
    let a = fold(&kpts);
    let mut b = fold(&plain);
    for x in &a {
        let hit = b
            .iter()
            .position(|y| (0..3).all(|k| (x[k] - y[k]).abs() < 1e-9))
            .unwrap_or_else(|| panic!("{x:?} has no partner in the un-wrapped grid"));
        b.swap_remove(hit);
    }
    assert!(b.is_empty(), "leftover un-wrapped points: {b:?}");
}

#[test]
fn make_kpts_scaled_center_is_the_zeroth_point() {
    // cell.py:842-845 — "Shift all points ... to be centered on scaled_center,
    // given as the zeroth index of the returned kpts."
    let cell = systems::diamond();
    let center = [0.1, 0.2, 0.3];
    let kpts = make_kpts(&cell, [2, 2, 2], false, true, Some(center)).expect("make_kpts");
    let s0 = cell.get_scaled_kpts(&kpts[..1])[0];
    for k in 0..3 {
        assert!((s0[k] - center[k]).abs() < 1e-12, "{s0:?} != {center:?}");
    }
    // cell.py:862 — a scaled_center forces the arange(n)/n grid, so
    // with_gamma_point becomes irrelevant.
    let same = make_kpts(&cell, [2, 2, 2], false, false, Some(center)).expect("make_kpts");
    assert_eq!(kpts, same);
}

#[test]
fn make_kpts_rejects_a_zero_axis() {
    let cell = systems::diamond();
    assert!(make_kpts_default(&cell, [2, 0, 2]).is_err());
}

/// 17-05 Task 6 closed the `NotYetImplemented` Phase-17 stub that used
/// to live at `kpts_mesh.rs:112-121`: k-point symmetry now returns a real
/// `pyscf_pbc_symm::kpts::KPoints`, and the constructor moved to
/// `pyscf_pbc_symm::kpts::make_kpts` because `Cell` sits BELOW
/// `pyscf-pbc-symm` (D-PBC-25) and cannot name that type. Nothing to refuse
/// here any more — this crate only builds the plain k-mesh.
#[test]
fn kpoint_symmetry_is_no_longer_refused_here() {
    let cell = systems::diamond();
    // The plain mesh still builds, and that is all this crate owes the caller.
    assert_eq!(make_kpts_default(&cell, [2, 2, 2]).expect("mesh").len(), 8);
}

/// `is_trim` — `kpts_helper.py:39-63` (17-05 Task 2). On a gamma-centred
/// `[2,2,2]` mesh EVERY point is a TRIM (each scaled coordinate is 0 or 1/2,
/// so `2k` is an integer vector); on `[3,3,3]` only Gamma is.
///
/// `2052 = 4096/2 + 4` — 17-CONTEXT §2.2's Gate A decomposition — is the
/// `[16,16,16]` case: 8 TRIM points, so
/// `nkpts_ibz = (4096 - 8)/2 + 8 = 2052 = 4096/2 + 8/2`.
#[test]
fn is_trim_counts_the_time_reversal_invariant_momenta() {
    let cell = systems::diamond();
    let tol = pyscf_pbc_gto::KPT_DIFF_TOL;

    let k222 = make_kpts_default(&cell, [2, 2, 2]).expect("mesh");
    assert_eq!(is_trim(&cell, &k222, tol).iter().filter(|b| **b).count(), 8);

    let k333 = make_kpts_default(&cell, [3, 3, 3]).expect("mesh");
    let mask333 = is_trim(&cell, &k333, tol);
    assert_eq!(mask333.iter().filter(|b| **b).count(), 1);
    assert!(mask333[0], "Gamma is the zeroth point and is always a TRIM");

    // The Gate A decomposition, asserted explicitly: it pins the TRIM count
    // independently of the fold itself.
    let k16 = make_kpts_default(&cell, [16, 16, 16]).expect("mesh");
    let ntrim = is_trim(&cell, &k16, tol).iter().filter(|b| **b).count();
    assert_eq!(ntrim, 8, "[16,16,16] has 2^3 TRIM points");
    let nkpts = k16.len();
    assert_eq!(nkpts, 4096);
    assert_eq!((nkpts - ntrim) / 2 + ntrim, 2052);
    assert_eq!(nkpts / 2 + ntrim / 2, 2052);
}

#[test]
fn cell_make_kpts_method_matches_the_free_function() {
    let cell = systems::diamond();
    assert_eq!(
        cell.make_kpts([3, 2, 2]).expect("method"),
        make_kpts(&cell, [3, 2, 2], WRAP_AROUND, WITH_GAMMA, None).expect("free fn")
    );
    // Upstream's documented defaults (cell.py:42-43).
    const { assert!(!WRAP_AROUND) };
    const { assert!(WITH_GAMMA) };
}

#[test]
fn is_zero_uses_kpt_diff_tol_not_1e_9() {
    // kpts_helper.py:32 — `abs(kpt).sum() < KPT_DIFF_TOL` with KPT_DIFF_TOL = 1e-6.
    // PBC-MASTER-PLAN §8.1 plan 09-07 step 2 quotes 1e-9; the Python wins (RULE 2).
    assert!(is_zero(&[0.0, 0.0, 0.0]));
    assert!(is_zero(&[1e-7, 0.0, 0.0]));
    assert!(!is_zero(&[1e-6, 0.0, 0.0]));
    // The test is on the SUM of absolute values, not the max.
    assert!(!is_zero(&[4e-7, 4e-7, 4e-7]));
    assert!(is_zero(&[3e-7, 3e-7, 3e-7]));
    assert!(is_zero(&[-1e-7, 1e-7, -1e-7]));
}

#[test]
fn kconserv_satisfies_its_defining_congruence() {
    // (k_K - k_L + k_M - k_N) . a must be an integer multiple of 2*pi.
    let cell = diamond_bohr();
    let a = cell.lattice_vectors();
    for nks in [[2, 2, 2], [3, 2, 1], [2, 2, 1]] {
        let kpts = make_kpts_default(&cell, nks).expect("make_kpts");
        let nk = kpts.len();
        let kc = get_kconserv(&cell, &kpts);
        assert_eq!(kc.nkpts, nk);
        assert_eq!(kc.data.len(), nk * nk * nk);
        for k in 0..nk {
            for l in 0..nk {
                for m in 0..nk {
                    let n = kc.get(k, l, m) as usize;
                    assert!(n < nk);
                    let d = [
                        kpts[k][0] - kpts[l][0] + kpts[m][0] - kpts[n][0],
                        kpts[k][1] - kpts[l][1] + kpts[m][1] - kpts[n][1],
                        kpts[k][2] - kpts[l][2] + kpts[m][2] - kpts[n][2],
                    ];
                    for (w, row) in a.iter().enumerate() {
                        let t = (d[0] * row[0] + d[1] * row[1] + d[2] * row[2])
                            / (2.0 * std::f64::consts::PI);
                        assert!(
                            (t - t.round()).abs() < 1e-9,
                            "{nks:?} ({k},{l},{m}) -> {n}: axis {w} gives {t}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn kconserv_has_the_symmetries_momentum_conservation_implies() {
    let cell = systems::diamond();
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("make_kpts");
    let nk = kpts.len();
    let kc = get_kconserv(&cell, &kpts);
    for k in 0..nk {
        // k - k + m - n = 0  =>  n = m.
        for m in 0..nk {
            assert_eq!(kc.get(k, k, m) as usize, m, "kconserv[{k},{k},{m}]");
        }
        // K and M enter with the same sign, so the table is symmetric in them.
        for l in 0..nk {
            for m in 0..nk {
                assert_eq!(kc.get(k, l, m), kc.get(m, l, k), "({k},{l},{m})");
            }
        }
    }
    // Gamma is index 0 and is its own inverse: kconserv[k,0,0] == k.
    for k in 0..nk {
        assert_eq!(kc.get(k, 0, 0) as usize, k);
    }
}

#[test]
fn kconserv3_satisfies_its_defining_congruence_and_squeezes_pinned_axes() {
    let cell = diamond_bohr();
    let a = cell.lattice_vectors();
    let kpts = make_kpts_default(&cell, [2, 2, 1]).expect("make_kpts");
    let nk = kpts.len();
    let all = KIdx::Many((0..nk).collect());
    let full = get_kconserv3(
        &cell,
        &kpts,
        &[
            all.clone(),
            all.clone(),
            all.clone(),
            all.clone(),
            all.clone(),
        ],
    );
    assert_eq!(full.shape, vec![nk; 5]);
    assert_eq!(full.data.len(), nk.pow(5));

    let mut idx = 0;
    for i in 0..nk {
        for j in 0..nk {
            for k in 0..nk {
                for x in 0..nk {
                    for y in 0..nk {
                        let c = full.data[idx] as usize;
                        idx += 1;
                        assert!(c < nk);
                        let mut d = [0.0_f64; 3];
                        for (w, dw) in d.iter_mut().enumerate() {
                            *dw = kpts[i][w] + kpts[j][w] + kpts[k][w]
                                - kpts[x][w]
                                - kpts[y][w]
                                - kpts[c][w];
                        }
                        for row in a.iter() {
                            let t = (d[0] * row[0] + d[1] * row[1] + d[2] * row[2])
                                / (2.0 * std::f64::consts::PI);
                            assert!((t - t.round()).abs() < 1e-9, "({i},{j},{k},{x},{y}) -> {c}");
                        }
                    }
                }
            }
        }
    }

    // kpts_helper.py:436-438 — a pinned (integer) axis is dropped from the shape.
    let pinned = get_kconserv3(
        &cell,
        &kpts,
        &[all.clone(), KIdx::One(1), all.clone(), all.clone(), all],
    );
    assert_eq!(pinned.shape, vec![nk, nk, nk, nk]);
    assert_eq!(pinned.data.len(), nk.pow(4));
    // ... and its entries are the kj = 1 slice of the full table.
    for i in 0..nk {
        for k in 0..nk {
            for x in 0..nk {
                for y in 0..nk {
                    let f = full.data[((((i * nk + 1) * nk + k) * nk + x) * nk) + y];
                    let p = pinned.data[(((i * nk + k) * nk + x) * nk) + y];
                    assert_eq!(f, p, "({i},1,{k},{x},{y})");
                }
            }
        }
    }
}

#[test]
fn member_and_intersection_find_every_duplicate() {
    let probe = [
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.25, 0.25, 0.25],
    ];
    assert_eq!(member(&[0.5, 0.0, 0.0], &probe), vec![1, 3]);
    assert_eq!(member(&[0.0, 0.0, 0.0], &probe), vec![0, 2]);
    assert_eq!(member(&[0.75, 0.0, 0.0], &probe), Vec::<usize>::new());
    // The comparison is Chebyshev against KPT_DIFF_TOL, so a 1e-7 wobble matches.
    assert_eq!(member(&[0.5 + 1e-7, 0.0, 0.0], &probe), vec![1, 3]);
    assert_eq!(member(&[0.5 + 1e-5, 0.0, 0.0], &probe), Vec::<usize>::new());
    assert_eq!(intersection(&probe, &[[0.5, 0.0, 0.0]]), vec![1, 3]);
    assert_eq!(intersection(&probe, &probe), vec![0, 1, 2, 3, 4]);
    assert_eq!(intersection(&probe, &[]), Vec::<usize>::new());
}

#[test]
fn unique_returns_first_occurrence_order() {
    let probe = [
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.25, 0.25, 0.25],
    ];
    let u = unique(&probe);
    // Tier 2 as well: upstream prints exactly these three arrays.
    assert_eq!(
        u.kpts,
        vec![[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.25, 0.25, 0.25]]
    );
    assert_eq!(u.index, vec![0, 1, 4]);
    assert_eq!(u.inverse, vec![0, 1, 0, 1, 2]);
    // The contract: index and inverse reconstruct the input.
    for (i, k) in probe.iter().enumerate() {
        assert_eq!(u.kpts[u.inverse[i]], *k);
        assert_eq!(probe[u.index[u.inverse[i]]], *k);
    }
    // A full MP mesh has no duplicates.
    let cell = systems::diamond();
    let kpts = make_kpts_default(&cell, [3, 2, 2]).expect("make_kpts");
    let u = unique(&kpts);
    assert_eq!(u.kpts.len(), kpts.len());
    assert_eq!(u.index, (0..kpts.len()).collect::<Vec<_>>());
}

// =========================================================================
// TIER 2 — hard-coded upstream values (D-PBC-19). See `UPSTREAM_SNIPPET`.
// =========================================================================

#[test]
fn make_kpts_matches_upstream() {
    assert!(!UPSTREAM_SNIPPET.is_empty());
    let cell = diamond_bohr();
    let cases: [MakeKptsCase; 7] = [
        ("222", [2, 2, 2], false, true, None, &KPTS_222),
        (
            "222_nogamma",
            [2, 2, 2],
            false,
            false,
            None,
            &KPTS_222_NOGAMMA,
        ),
        ("222_wrap", [2, 2, 2], true, true, None, &KPTS_222_WRAP),
        (
            "222_nogamma_wrap",
            [2, 2, 2],
            true,
            false,
            None,
            &KPTS_222_NOGAMMA_WRAP,
        ),
        ("321", [3, 2, 1], false, true, None, &KPTS_321),
        ("333_wrap", [3, 3, 3], true, true, None, &KPTS_333_WRAP),
        (
            "222_center",
            [2, 2, 2],
            false,
            true,
            Some([0.1, 0.2, 0.3]),
            &KPTS_222_CENTER,
        ),
    ];
    for (tag, nks, wrap, gamma, center, want) in cases {
        let got = make_kpts(&cell, nks, wrap, gamma, center).expect("make_kpts");
        assert_kpts_eq(&got, want, 1e-12, tag);
    }
}

#[test]
fn kconserv_matches_the_upstream_tables() {
    // The tables are pure integer topology, so they are identical for the
    // Angstrom-built §9.2 diamond and for `diamond_bohr`.
    for cell in [diamond_bohr(), systems::diamond()] {
        let k222 = make_kpts_default(&cell, [2, 2, 2]).expect("make_kpts");
        assert_eq!(get_kconserv(&cell, &k222).data, KCONSERV_222);
        let k321 = make_kpts_default(&cell, [3, 2, 1]).expect("make_kpts");
        assert_eq!(get_kconserv(&cell, &k321).data, KCONSERV_321);
    }
    // ... and it does not depend on where the mesh sits, only on its topology.
    let cell = diamond_bohr();
    let shifted = make_kpts(&cell, [2, 2, 2], false, false, None).expect("make_kpts");
    assert_eq!(get_kconserv(&cell, &shifted).data, KCONSERV_222);
}

#[test]
fn kconserv3_matches_the_upstream_table() {
    let cell = diamond_bohr();
    let kpts = make_kpts_default(&cell, [2, 2, 1]).expect("make_kpts");
    let all = KIdx::Many((0..kpts.len()).collect());
    let got = get_kconserv3(
        &cell,
        &kpts,
        &[all.clone(), KIdx::One(1), all.clone(), all.clone(), all],
    );
    assert_eq!(got.shape, vec![4, 4, 4, 4]);
    assert_eq!(got.data, KCONSERV3_221_KJ1);
}

include!("common/kpts_reference.rs");
