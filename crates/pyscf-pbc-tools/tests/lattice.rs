//! Plan 09-06 acceptance gate — lattice sums and supercell geometry.
//!
//! Tier 1 (no upstream needed) comes first and must pass unconditionally.
//! Tier 2 pins hard-coded upstream numbers (D-PBC-19); every literal below was
//! produced ONCE by the snippet in [`UPSTREAM_SNIPPET`] and committed.
//!
//! The cell used throughout is diamond **specified directly in Bohr**, so no
//! Angstrom -> Bohr constant enters and the comparison against upstream is
//! exact rather than 1e-7 (see plan 09-04's note on the 4.95e-9 relative gap
//! between this workspace's conversion factor and PySCF's).

use pyscf_pbc_tools::lattice::{
    check_lattice_sum_range, get_lattice_ls, max_atom_pair_distance,
    monkhorst_pack_size_from_scaled, qr_row2, round_to_cell0_default,
};
use pyscf_pbc_tools::mat3::{cross3, dot3, inv3, norm3};
use pyscf_pbc_tools::mesh::{qr_r22_abs, qr_r22_abs_closed_form};
use pyscf_pbc_tools::supercell::{
    cell_plus_imgs_translations, image_atom_coords, scale_lattice, super_cell_translations,
};

/// The exact generating snippet for every tier-2 literal in this file
/// (PySCF 2.12.1, `.venv/bin/python`).
///
/// ```python
/// import numpy as np
/// from pyscf.pbc import gto, tools
/// H = 3.3701375705493315
/// c = gto.Cell()
/// c.a = [[0., H, H], [H, 0., H], [H, H, 0.]]
/// c.atom = [('C', (0., 0., 0.)), ('C', (H/2, H/2, H/2))]
/// c.basis = 'gth-szv'; c.pseudo = 'gth-pade'; c.unit = 'Bohr'; c.verbose = 0
/// c.build()
/// for rcut in (10.0, 5.0, c.rcut):
///     for d in (True, False):
///         Ls = tools.pbc.get_lattice_Ls(c, rcut=rcut, discard=d)
///         print(rcut, d, len(Ls), Ls[0].tolist(),
///               np.linalg.norm(Ls, axis=1).max())
/// print(tools.pbc.check_lattice_sum_range(c, tools.pbc.get_lattice_Ls(c)))
/// print(tools.pbc.get_monkhorst_pack_size(c, c.make_kpts([3, 2, 1])))
/// print(tools.pbc.round_to_cell0(np.array(
///     [[0.2, -0.3, 1.7], [-1e-7, 0.5, 0.9999999],
///      [1.0, -1.0, 2.5], [0.3333333333, 0.6666666667, -0.25]])))
/// sc = tools.super_cell(c, [2, 2, 2]); print(sc.atom_coords())
/// ```
const UPSTREAM_SNIPPET: &str = "see the doc comment above";

/// Diamond lattice in BOHR, one vector per row.
const A: [[f64; 3]; 3] = [
    [0.0, 3.3701375705493315, 3.3701375705493315],
    [3.3701375705493315, 0.0, 3.3701375705493315],
    [3.3701375705493315, 3.3701375705493315, 0.0],
];
/// The two carbons, Cartesian Bohr.
const COORDS: [[f64; 3]; 2] = [
    [0.0, 0.0, 0.0],
    [1.6850687852746657, 1.6850687852746657, 1.6850687852746657],
];
/// The same two carbons, scaled by the lattice.
const SCALED: [[f64; 3]; 2] = [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]];
/// `cell.rcut` for the cell above (upstream and this port agree exactly, since
/// the lattice is given in Bohr).
const RCUT: f64 = 21.31940052177759;

fn diamond_ls(rcut: f64, discard: bool) -> Vec<[f64; 3]> {
    get_lattice_ls(&A, &SCALED, &COORDS, rcut, 3, discard)
}

/// `v . inv(a)` — the fractional (lattice) coordinates of a Cartesian vector.
fn frac(v: &[f64; 3], inv_a: &[[f64; 3]; 3]) -> [f64; 3] {
    let mut o = [0.0_f64; 3];
    for (j, oj) in o.iter_mut().enumerate() {
        *oj = v[0] * inv_a[0][j] + v[1] * inv_a[1][j] + v[2] * inv_a[2][j];
    }
    o
}

// ---------------------------------------------------------------------------
// Tier 1 — invariants. No upstream reference needed; these must hold for any
// correct implementation.
// ---------------------------------------------------------------------------

#[test]
fn qr_row2_matches_the_3x3_twin_on_its_first_three_columns() {
    // The first three entries of `qr_row2` over a 3 x n matrix must agree with
    // plan 09-04's `qr_r22_abs` over the leading 3 x 3 block — same Householder
    // reflections, same order.
    let dr = [[1.5, 0.0, 0.0], [0.0, 2.5, 0.0], [0.0, 0.0, 0.75]];
    for perm in [[1usize, 2, 0], [2, 0, 1], [0, 1, 2]] {
        let cols = [A[perm[0]], A[perm[1]], A[perm[2]], dr[0], dr[1], dr[2]];
        let r2 = qr_row2(&cols);
        assert_eq!(r2.len(), 6);
        let head = [cols[0], cols[1], cols[2]];
        assert!(
            (r2[2].abs() - qr_r22_abs(&head)).abs() < 1e-13,
            "|R[2,2]| = {} disagrees with qr_r22_abs = {}",
            r2[2].abs(),
            qr_r22_abs(&head)
        );
        // ... and with the closed form |det| / ||c0 x c1||.
        assert!((r2[2].abs() - qr_r22_abs_closed_form(&head)).abs() < 1e-12);
    }
}

#[test]
fn qr_row2_tail_is_the_perpendicular_component() {
    // Row 2 of R is <q2, col_j> with q2 the unit normal of span(c0, c1), so
    // |R[2,j]| must equal |col_j . n_hat| for EVERY column, tail included.
    let dr = [[1.5, 0.0, 0.0], [0.0, 2.5, 0.0], [0.0, 0.0, 0.75]];
    let cols = [A[1], A[2], A[0], dr[0], dr[1], dr[2]];
    let n = cross3(&cols[0], &cols[1]);
    let nn = norm3(&n);
    let nhat = [n[0] / nn, n[1] / nn, n[2] / nn];
    let r2 = qr_row2(&cols);
    for (j, r) in r2.iter().enumerate() {
        let expect = dot3(&cols[j], &nhat).abs();
        assert!(
            (r.abs() - expect).abs() < 1e-12,
            "column {j}: |R[2,j]| = {} but perpendicular component = {expect}",
            r.abs()
        );
    }
}

#[test]
fn lattice_ls_contains_the_origin_exactly_once() {
    for rcut in [5.0, 10.0, RCUT] {
        for discard in [true, false] {
            let ls = diamond_ls(rcut, discard);
            let zeros = ls.iter().filter(|l| norm3(l) == 0.0).count();
            assert_eq!(zeros, 1, "rcut = {rcut}, discard = {discard}");
        }
    }
}

#[test]
fn every_ls_is_an_integer_lattice_translation() {
    let inv_a = inv3(&A).expect("diamond lattice is invertible");
    for l in diamond_ls(10.0, true) {
        for (j, t) in frac(&l, &inv_a).iter().enumerate() {
            assert!(
                (t - t.round()).abs() < 1e-9,
                "L = {l:?} has non-integer coefficient {t} on axis {j}"
            );
        }
    }
}

#[test]
fn discard_keeps_a_subset_and_respects_its_own_bound() {
    let full = diamond_ls(10.0, false);
    let kept = diamond_ls(10.0, true);
    assert!(kept.len() < full.len());
    let limit = 10.0 + max_atom_pair_distance(&COORDS);
    for l in &kept {
        assert!(norm3(l) < limit, "kept L = {l:?} exceeds rcut + dist_max");
        assert!(
            full.iter().any(|f| f == l),
            "kept L = {l:?} is not in the un-discarded list"
        );
    }
    // ... and nothing inside the bound was dropped.
    let inside = full.iter().filter(|l| norm3(l) < limit).count();
    assert_eq!(inside, kept.len());
}

#[test]
fn undiscarded_ls_is_a_full_cartesian_product() {
    // Ls = cartesian_prod(-b..=b per axis), so the count is a product of three
    // odd numbers, and the origin sits exactly in the middle.
    for rcut in [5.0, 10.0, RCUT] {
        let ls = diamond_ls(rcut, false);
        assert_eq!(ls.len() % 2, 1, "rcut = {rcut}: count must be odd");
        let mid = ls.len() / 2;
        assert_eq!(ls[mid], [0.0, 0.0, 0.0], "rcut = {rcut}");
        // Antisymmetry: Ls[i] == -Ls[n-1-i].
        for (i, l) in ls.iter().enumerate() {
            let m = ls[ls.len() - 1 - i];
            for k in 0..3 {
                assert!((l[k] + m[k]).abs() < 1e-12);
            }
        }
    }
}

#[test]
fn degenerate_inputs_return_the_origin_only() {
    // pbc.py:619-620 — dimension == 0, rcut <= 0, or an empty cell.
    assert_eq!(
        get_lattice_ls(&A, &SCALED, &COORDS, 10.0, 0, true),
        vec![[0.0; 3]]
    );
    assert_eq!(
        get_lattice_ls(&A, &SCALED, &COORDS, 0.0, 3, true),
        vec![[0.0; 3]]
    );
    assert_eq!(
        get_lattice_ls(&A, &SCALED, &COORDS, -1.0, 3, true),
        vec![[0.0; 3]]
    );
    assert_eq!(get_lattice_ls(&A, &[], &[], 10.0, 3, true), vec![[0.0; 3]]);
}

#[test]
fn low_dimensional_sums_stay_in_their_plane() {
    // dimension = 2 uses only a[0], a[1]; dimension = 1 only a[0].
    let ortho = [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 20.0]];
    let scaled = [[0.0, 0.0, 0.0]];
    let coords = [[0.0, 0.0, 0.0]];
    let ls2 = get_lattice_ls(&ortho, &scaled, &coords, 6.0, 2, false);
    assert!(ls2.iter().all(|l| l[2] == 0.0));
    let ls1 = get_lattice_ls(&ortho, &scaled, &coords, 6.0, 1, false);
    assert!(ls1.iter().all(|l| l[1] == 0.0 && l[2] == 0.0));
    assert_eq!(ls1.len() % 2, 1);
}

#[test]
fn check_lattice_sum_range_is_infinite_when_covered() {
    let full = diamond_ls(10.0, false);
    assert_eq!(
        check_lattice_sum_range(&full, &full, &COORDS),
        f64::INFINITY
    );
    // Dropping images can only bring an outside atom closer.
    let kept = diamond_ls(10.0, true);
    let d = check_lattice_sum_range(&full, &kept, &COORDS);
    assert!(d.is_finite() && d > 0.0, "got {d}");
}

#[test]
fn monkhorst_pack_size_counts_distinct_scaled_values() {
    // A gamma-centred 2x2x2 mesh in scaled coordinates.
    let mut skpts = Vec::new();
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                skpts.push([i as f64 / 2.0, j as f64 / 2.0, k as f64 / 2.0]);
            }
        }
    }
    assert_eq!(
        monkhorst_pack_size_from_scaled(&skpts, 1e-5).expect("valid"),
        [2, 2, 2]
    );
    // A single k-point is [1,1,1] whatever its value (pbc.py:591-592).
    assert_eq!(
        monkhorst_pack_size_from_scaled(&[[0.3, -0.2, 0.1]], 1e-5).expect("valid"),
        [1, 1, 1]
    );
    // pbc.py:590 — the assertion `nkpts < 1/min_tol`.
    assert!(monkhorst_pack_size_from_scaled(&skpts, 0.2).is_err());
}

#[test]
fn round_to_cell0_lands_in_the_unit_cell_and_is_idempotent() {
    let r = [
        [0.2, -0.3, 1.7],
        [-1e-7, 0.5, 0.9999999],
        [1.0, -1.0, 2.5],
        [0.3333333333, 0.6666666667, -0.25],
    ];
    let once = round_to_cell0_default(&r);
    for k in &once {
        for x in k {
            assert!((0.0..1.0).contains(x), "{x} is outside [0, 1)");
        }
    }
    let twice = round_to_cell0_default(&once);
    assert_eq!(once, twice);
}

#[test]
fn supercell_translations_are_the_expected_cartesian_products() {
    let ls = super_cell_translations(&A, &[2, 2, 2], false);
    assert_eq!(ls.len(), 8);
    // ncopy = [1,1,1] is a no-op.
    assert_eq!(
        super_cell_translations(&A, &[1, 1, 1], false),
        vec![[0.0; 3]]
    );
    // wrap_around shifts index (n+1)/2.. by -n; for n = 2 that is index 1 -> -1.
    let wrapped = super_cell_translations(&A, &[2, 2, 2], true);
    assert_eq!(wrapped.len(), 8);
    assert_eq!(wrapped[0], [0.0; 3]);
    // Every wrapped translation is the un-wrapped one minus a lattice vector,
    // so the SET of images modulo the supercell is unchanged.
    let inv_a = inv3(&A).expect("invertible");
    for l in &wrapped {
        for t in frac(l, &inv_a) {
            assert!((t - t.round()).abs() < 1e-9);
            assert!(t.round() == 0.0 || t.round() == -1.0, "t = {t}");
        }
    }
    // +/- images: (2n+1)^3 of them.
    assert_eq!(cell_plus_imgs_translations(&A, &[1, 1, 1]).len(), 27);
    assert_eq!(cell_plus_imgs_translations(&A, &[0, 0, 0]), vec![[0.0; 3]]);
}

#[test]
fn scale_lattice_scales_each_row() {
    let s = scale_lattice(&A, &[3, 1, 2]);
    for j in 0..3 {
        assert_eq!(s[0][j], 3.0 * A[0][j]);
        assert_eq!(s[1][j], A[1][j]);
        assert_eq!(s[2][j], 2.0 * A[2][j]);
    }
}

#[test]
fn image_atom_coords_is_image_major() {
    let ls = [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]];
    let out = image_atom_coords(&ls, &COORDS);
    assert_eq!(out.len(), 4);
    assert_eq!(out[0], COORDS[0]);
    assert_eq!(out[1], COORDS[1]);
    assert_eq!(out[2], [1.0, 2.0, 3.0]);
    assert_eq!(
        out[3],
        [COORDS[1][0] + 1.0, COORDS[1][1] + 2.0, COORDS[1][2] + 3.0]
    );
}

// ---------------------------------------------------------------------------
// Tier 2 — hard-coded upstream values (D-PBC-19). See `UPSTREAM_SNIPPET`.
// ---------------------------------------------------------------------------

#[test]
fn lattice_ls_matches_upstream_counts_and_first_row() {
    assert!(!UPSTREAM_SNIPPET.is_empty());
    // (rcut, discard, len, Ls[0], max |L|)
    let cases: [(f64, bool, usize, [f64; 3], f64); 6] = [
        (
            10.0,
            true,
            135,
            [3.3701375705493315, -6.740275141098662, -10.110412711647994],
            12.60990013529029,
        ),
        (
            10.0,
            false,
            729,
            [
                -26.961100564394652,
                -26.961100564394652,
                -26.961100564394652,
            ],
            46.69799600550547,
        ),
        (
            5.0,
            true,
            19,
            [0.0, 0.0, -6.740275141098663],
            6.740275141098663,
        ),
        (
            5.0,
            false,
            343,
            [
                -20.220825423295988,
                -20.220825423295988,
                -20.220825423295988,
            ],
            35.023497004129105,
        ),
        (
            RCUT,
            true,
            767,
            [10.110412711647994, -13.480550282197324, -16.850687852746656],
            23.830471296669888,
        ),
        (
            RCUT,
            false,
            3375,
            [-47.18192598769064, -47.18192598769064, -47.18192598769064],
            81.72149300963457,
        ),
    ];
    for (rcut, discard, n, first, max_norm) in cases {
        let ls = diamond_ls(rcut, discard);
        assert_eq!(ls.len(), n, "rcut = {rcut}, discard = {discard}");
        for k in 0..3 {
            assert!(
                (ls[0][k] - first[k]).abs() < 1e-12,
                "rcut = {rcut}, discard = {discard}: Ls[0] = {:?} != {first:?}",
                ls[0]
            );
        }
        let got = ls.iter().fold(0.0_f64, |m, l| m.max(norm3(l)));
        assert!(
            (got - max_norm).abs() < 1e-12,
            "rcut = {rcut}, discard = {discard}: max |L| = {got} != {max_norm}"
        );
    }
}

#[test]
fn check_lattice_sum_range_matches_upstream() {
    // upstream: tools.pbc.check_lattice_sum_range(c, get_lattice_Ls(c))
    //           == 22.035133643781318
    let ls_full = diamond_ls(RCUT * 1.5, false);
    let ls = diamond_ls(RCUT, true);
    let d = check_lattice_sum_range(&ls_full, &ls, &COORDS);
    assert!(
        (d - 22.035133643781318).abs() < 1e-9,
        "check_lattice_sum_range = {d}"
    );
}

#[test]
fn round_to_cell0_matches_upstream() {
    let r = [
        [0.2, -0.3, 1.7],
        [-1e-7, 0.5, 0.9999999],
        [1.0, -1.0, 2.5],
        [0.3333333333, 0.6666666667, -0.25],
    ];
    let expect = [
        [0.2, 0.7, 0.7],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [0.333333, 0.666667, 0.75],
    ];
    let got = round_to_cell0_default(&r);
    for (g, e) in got.iter().zip(expect.iter()) {
        for k in 0..3 {
            assert!((g[k] - e[k]).abs() < 1e-12, "got {got:?}, want {expect:?}");
        }
    }
}

#[test]
fn supercell_translations_match_upstream_atom_layout() {
    // upstream: tools.super_cell(c, [2,2,2]).atom_coords()
    const SUPER_COORDS: [[f64; 3]; 16] = [
        [0.0, 0.0, 0.0],
        [1.6850687852746657, 1.6850687852746657, 1.6850687852746657],
        [3.3701375705493315, 3.3701375705493315, 0.0],
        [5.055206355823997, 5.055206355823997, 1.6850687852746657],
        [3.3701375705493315, 0.0, 3.3701375705493315],
        [5.055206355823997, 1.6850687852746657, 5.055206355823997],
        [6.740275141098663, 3.3701375705493315, 3.3701375705493315],
        [8.425343926373328, 5.055206355823997, 5.055206355823997],
        [0.0, 3.3701375705493315, 3.3701375705493315],
        [1.6850687852746657, 5.055206355823997, 5.055206355823997],
        [3.3701375705493315, 6.740275141098663, 3.3701375705493315],
        [5.055206355823997, 8.425343926373328, 5.055206355823997],
        [3.3701375705493315, 3.3701375705493315, 6.740275141098663],
        [5.055206355823997, 5.055206355823997, 8.425343926373328],
        [6.740275141098663, 6.740275141098663, 6.740275141098663],
        [8.425343926373328, 8.425343926373328, 8.425343926373328],
    ];
    let ls = super_cell_translations(&A, &[2, 2, 2], false);
    let coords = image_atom_coords(&ls, &COORDS);
    assert_eq!(coords.len(), 16);
    for (g, e) in coords.iter().zip(SUPER_COORDS.iter()) {
        for k in 0..3 {
            assert!((g[k] - e[k]).abs() < 1e-12, "got {coords:?}");
        }
    }
}
