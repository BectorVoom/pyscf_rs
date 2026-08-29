//! Plan 09-05 Task 3 — G-vectors, structure factors, uniform grids (PBC-GTO-04).
//!
//! Tier 1 (invariants, no upstream needed — these must pass unconditionally):
//!   * `fftfreq_scaled(n)` folds at `(n-1)/2` and is congruent to `i` mod `n`
//!     for every `n` in `1..=32`, and `fftfreq(n) == fftfreq_scaled(n)/n`;
//!   * `Gv[g] . a[i] == 2*pi * r_i[idx_i(g)]` — the DEFINING property of a
//!     reciprocal-lattice grid, which pins the row order, the axis order and
//!     the `2*pi` normalisation all at once;
//!   * `Gv[0] == [0,0,0]` and, for an all-odd mesh, the G-vector set is closed
//!     under negation;
//!   * `weights == 1/cell.vol`;
//!   * `|SI[a,g]| == 1` and `SI[a,0] == 1+0i` for every atom;
//!   * the separable `get_SI` branch and the K-02 device branch agree;
//!   * `SI` for an atom at the origin is `1+0i` everywhere, and `SI[a,-g]` is
//!     the conjugate of `SI[a,g]`;
//!   * the uniform grid is the lattice-fraction product grid, sums to zero for
//!     `wrap_around = true` on an odd mesh, and the two `wrap_around` variants
//!     differ only by whole lattice translations;
//!   * the `inf_vacuum` branch (D-PBC-20, plan 12-08) uses the non-uniform
//!     Gauss-Chebyshev base: the vacuum axes are reduced to `2 * (n // 2)`
//!     points, the scalar weight becomes a per-grid array, and the periodic
//!     axes are untouched — while the 3-D path stays a scalar on the requested
//!     mesh;
//!   * a zero-axis mesh is an error, and `atmlst` selects the right rows.
//!
//! Tier 2 (hard-coded upstream references, D-PBC-19) — the `n = 1..8`
//! `fftfreq` tables, the full 125x3 `Gv` array for diamond at mesh `[5,5,5]`,
//! the weights, `SI` rows, and both uniform grids. Generated once from live
//! PySCF 2.12.1; the generating snippet is in [`UPSTREAM_SNIPPET`].
//!
//! # Two diamonds, two tolerances
//!
//! [`diamond_bohr`] gives the lattice DIRECTLY in Bohr, using upstream's own
//! `cell.lattice_vectors()` literals, so no Angstrom conversion happens and the
//! plan's stated **1e-12** tolerance is met with ~1 ULP to spare (the only
//! residual is closed-form `inv3` vs numpy's LU inverse). The §9.2
//! `systems::diamond()` builds the same lattice from Angstrom and therefore
//! carries the 4.95e-9 `pyscf_core::Unit::Ang` gap documented in
//! `tests/cell_build.rs`; it is checked at a relative 1e-7 with the residual
//! pinned below 2e-8, so the loosened bound cannot hide a real bug.

mod common;

use common::systems;
use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_gto::gv::{
    fftfreq, fftfreq_scaled, get_gv, get_gv_weights, get_si, get_uniform_grids,
};
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs, LowDimFtType};
use std::f64::consts::PI;

/// The exact snippet that produced every literal below (D-PBC-19). Run with
/// `.venv/bin/python` against PySCF 2.12.1.
///
/// ```python
/// import numpy as np
/// from pyscf.pbc import gto as pbcgto
///
/// def fcc(a0):
///     h = a0 / 2.
///     return np.array([[0., h, h], [h, 0., h], [h, h, 0.]])
///
/// c = pbcgto.Cell()
/// c.a = fcc(3.5668)
/// c.atom = [('C', (0, 0, 0)), ('C', (0.8917, 0.8917, 0.8917))]
/// c.basis = 'gth-szv'; c.pseudo = 'gth-pade'; c.unit = 'Angstrom'
/// c.precision = 1e-8; c.verbose = 0
/// c.build()
///
/// [list(np.fft.fftfreq(n, 1./n)) for n in range(1, 9)]   # -> FFTFREQ_SCALED_REF
/// [list(np.fft.fftfreq(n)) for n in range(1, 9)]         # -> FFTFREQ_REF
///
/// mesh = [5, 5, 5]
/// Gv, Gvbase, w = c.get_Gv_weights(mesh)                 # -> GV_DIAMOND_555, WEIGHTS_555
/// c.get_SI(mesh=mesh)                                    # -> SI_ROW0/SI_ROW1
/// c.get_uniform_grids(mesh, wrap_around=True)            # -> GRID_WRAP_*
/// c.get_uniform_grids(mesh, wrap_around=False)           # -> GRID_NOWRAP_*
/// ```
pub const UPSTREAM_SNIPPET: &str = "see the doc comment above";

/// `cell.lattice_vectors()` in Bohr for the §9.2 diamond, from live PySCF.
const A_BOHR: [[f64; 3]; 3] = [
    [0.0, 3.3701375705493315, 3.3701375705493315],
    [3.3701375705493315, 0.0, 3.3701375705493315],
    [3.3701375705493315, 3.3701375705493315, 0.0],
];

/// The §9.2 diamond with the lattice given DIRECTLY in Bohr, so there is no
/// Angstrom conversion and no `Unit::Ang` gap. See the module docs.
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

const MESH: [usize; 3] = [5, 5, 5];

fn rel(a: f64, b: f64) -> f64 {
    let s = a.abs().max(b.abs());
    if s == 0.0 { 0.0 } else { (a - b).abs() / s }
}

// =========================================================================
// TIER 1 — invariants.
// =========================================================================

/// PBC-MASTER-PLAN §8.1 plan 09-05 step 1 is emphatic: getting this fold wrong
/// "silently corrupts every FFT downstream". Two independent characterisations
/// for every `n` in `1..=32`: the frequency is congruent to the index mod `n`,
/// and it lies in the half-open window numpy uses.
#[test]
fn fftfreq_scaled_folds_correctly_for_n_up_to_32() {
    for n in 1..=32_usize {
        let f = fftfreq_scaled(n);
        assert_eq!(f.len(), n, "n = {n}");
        assert_eq!(f[0], 0.0, "n = {n}: zero frequency first");
        for (i, fi) in f.iter().enumerate() {
            // Integral, and congruent to i modulo n.
            assert_eq!(fi.fract(), 0.0, "n = {n} i = {i}: {fi} is not an integer");
            let k = (*fi as i64 - i as i64) % n as i64;
            assert_eq!(
                k, 0,
                "n = {n} i = {i}: {fi} is not congruent to {i} mod {n}"
            );
            // numpy's window: -n/2 <= f < n/2 for even n, |f| <= (n-1)/2 for odd.
            assert!(
                *fi >= -(n as f64) / 2.0 && *fi < n as f64 / 2.0,
                "n = {n} i = {i}: {fi} outside [-n/2, n/2)"
            );
        }
        // Every residue class appears exactly once.
        let mut seen: Vec<i64> = f.iter().map(|x| *x as i64).collect();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), n, "n = {n}: frequencies are not distinct");
    }
}

/// `np.fft.fftfreq(n)` is `np.fft.fftfreq(n, 1./n) / n`.
#[test]
fn fftfreq_is_the_scaled_table_over_n() {
    for n in 1..=32_usize {
        let s = fftfreq_scaled(n);
        let f = fftfreq(n);
        assert_eq!(f.len(), n);
        for i in 0..n {
            assert_eq!(f[i], s[i] / n as f64, "n = {n} i = {i}");
            assert!(
                (-0.5..0.5).contains(&f[i]),
                "n = {n} i = {i}: {} outside [-0.5, 0.5)",
                f[i]
            );
        }
    }
}

/// The DEFINING property of the G-vector grid: `Gv[g] . a[i] == 2*pi*r_i`,
/// where `r_i` is the integer frequency of `g` along axis `i`. This pins the
/// row order (C-order over `(x, y, z)`), the axis assignment, and the `2*pi`
/// normalisation simultaneously — a transposed `b`, a swapped axis or a missing
/// `2*pi` all fail here without any reference value.
#[test]
fn gv_dot_lattice_is_two_pi_times_the_integer_frequency() {
    for (name, cell) in systems::all() {
        let w = get_gv_weights(&cell, Some(MESH)).expect("Gv");
        assert_eq!(w.gv.len(), MESH[0] * MESH[1] * MESH[2], "{name}");
        let a = cell.lattice_vectors();
        for (g, gvec) in w.gv.iter().enumerate() {
            let idx = [
                g / (MESH[1] * MESH[2]),
                (g / MESH[2]) % MESH[1],
                g % MESH[2],
            ];
            for i in 0..3 {
                let dot = gvec[0] * a[i][0] + gvec[1] * a[i][1] + gvec[2] * a[i][2];
                let want = 2.0 * PI * w.gvbase[i][idx[i]];
                assert!(
                    (dot - want).abs() < 1e-9,
                    "{name} g = {g} axis {i}: Gv.a = {dot}, want 2*pi*{} = {want}",
                    w.gvbase[i][idx[i]]
                );
            }
        }
    }
}

/// `Gv[0]` is the zero vector, and for an all-odd mesh every G-vector's
/// negative is also in the set (the frequency table `[0, 1, 2, -2, -1]` is
/// symmetric).
#[test]
fn gv_row_zero_is_the_origin_and_the_set_is_symmetric() {
    for (name, cell) in systems::all() {
        let gv = get_gv(&cell, Some(MESH)).expect("Gv");
        assert_eq!(gv[0], [0.0, 0.0, 0.0], "{name}: Gv row 0");
        // Index of the negated frequency along an odd axis n: 0 -> 0, i -> n-i.
        let neg = |i: usize, n: usize| if i == 0 { 0 } else { n - i };
        for (g, gvec) in gv.iter().enumerate() {
            let (x, y, z) = (
                g / (MESH[1] * MESH[2]),
                (g / MESH[2]) % MESH[1],
                g % MESH[2],
            );
            let gneg =
                neg(x, MESH[0]) * MESH[1] * MESH[2] + neg(y, MESH[1]) * MESH[2] + neg(z, MESH[2]);
            for c in 0..3 {
                assert!(
                    (gv[gneg][c] + gvec[c]).abs() < 1e-12,
                    "{name} g = {g}: Gv[{gneg}] is not -Gv[{g}]"
                );
            }
        }
    }
}

/// `weights == |det(b)|/(2*pi)^3 == 1/cell.vol` — upstream says so in its own
/// comment at `cell.py:600`.
#[test]
fn weights_equal_one_over_the_cell_volume() {
    for (name, cell) in systems::all() {
        let w = get_gv_weights(&cell, Some(MESH)).expect("Gv");
        assert!(
            rel(w.weights, 1.0 / cell.vol()) < 1e-14,
            "{name}: weights {} vs 1/vol {}",
            w.weights,
            1.0 / cell.vol()
        );
        // Independent of the mesh — it is a property of the lattice alone.
        let w2 = get_gv_weights(&cell, Some([3, 4, 5])).expect("Gv");
        assert_eq!(w.weights, w2.weights, "{name}");
    }
}

/// The plan's TEST block: `|SI[a,g]| == 1` everywhere (to 1e-14) and
/// `SI[a, 0] == 1 + 0i` for every atom, because `Gv[0]` is the zero vector.
#[test]
fn structure_factors_are_unit_modulus_and_one_at_g_zero() {
    for (name, cell) in systems::all() {
        let ngrids = MESH[0] * MESH[1] * MESH[2];
        let natm = cell.mol.natm;
        for use_kernel in [false, true] {
            let gv = get_gv(&cell, Some(MESH)).expect("Gv");
            let si = if use_kernel {
                get_si(&cell, Some(&gv), None, None).expect("SI")
            } else {
                get_si(&cell, None, Some(MESH), None).expect("SI")
            };
            assert_eq!(si.re.len(), natm * ngrids, "{name} kernel = {use_kernel}");
            assert_eq!(si.im.len(), natm * ngrids);
            for a in 0..natm {
                let r0 = si.re[a * ngrids];
                let i0 = si.im[a * ngrids];
                assert!(
                    (r0 - 1.0).abs() < 1e-14 && i0.abs() < 1e-14,
                    "{name} atom {a} kernel = {use_kernel}: SI at G=0 is {r0}{i0:+}i"
                );
            }
            for (r, i) in si.re.iter().zip(si.im.iter()) {
                let m = (r * r + i * i).sqrt();
                assert!(
                    (m - 1.0).abs() < 1e-14,
                    "{name} kernel = {use_kernel}: |SI| = {m}"
                );
            }
        }
    }
}

/// The separable branch (`Gv = None`, `natm*(mx+my+mz)` transcendentals) and
/// the K-02 device branch (`Gv = Some`, `natm*ngrids` transcendentals) compute
/// the same thing. Upstream's own two branches agree to 1.8e-15 on this cell.
#[test]
fn separable_and_kernel_structure_factors_agree() {
    for (name, cell) in systems::all() {
        let gv = get_gv(&cell, Some(MESH)).expect("Gv");
        let sep = get_si(&cell, None, Some(MESH), None).expect("SI separable");
        let dev = get_si(&cell, Some(&gv), None, None).expect("SI kernel");
        let worst = sep
            .re
            .iter()
            .zip(dev.re.iter())
            .chain(sep.im.iter().zip(dev.im.iter()))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            worst < 1e-13,
            "{name}: separable vs K-02 differ by {worst:.3e}"
        );
    }
}

/// An atom at the origin has `SI == 1+0i` at every G-vector, and negating a
/// G-vector conjugates the structure factor.
#[test]
fn structure_factor_origin_atom_and_conjugate_symmetry() {
    let cell = diamond_bohr();
    let ngrids = MESH[0] * MESH[1] * MESH[2];
    let gv = get_gv(&cell, Some(MESH)).expect("Gv");
    let si = get_si(&cell, Some(&gv), None, None).expect("SI");

    // Atom 0 sits at the origin.
    assert_eq!(cell.mol.atom_coords()[0], [0.0, 0.0, 0.0]);
    for g in 0..ngrids {
        assert!(
            (si.re[g] - 1.0).abs() < 1e-15,
            "atom 0 g = {g}: re = {}",
            si.re[g]
        );
        assert!(si.im[g].abs() < 1e-15, "atom 0 g = {g}: im = {}", si.im[g]);
    }

    // SI[a, -g] == conj(SI[a, g]).
    let neg = |i: usize, n: usize| if i == 0 { 0 } else { n - i };
    for a in 0..cell.mol.natm {
        for g in 0..ngrids {
            let (x, y, z) = (
                g / (MESH[1] * MESH[2]),
                (g / MESH[2]) % MESH[1],
                g % MESH[2],
            );
            let gn =
                neg(x, MESH[0]) * MESH[1] * MESH[2] + neg(y, MESH[1]) * MESH[2] + neg(z, MESH[2]);
            let (r, i) = (si.re[a * ngrids + g], si.im[a * ngrids + g]);
            let (rn, inn) = (si.re[a * ngrids + gn], si.im[a * ngrids + gn]);
            assert!((r - rn).abs() < 1e-14, "atom {a} g = {g}: re mismatch");
            assert!((i + inn).abs() < 1e-14, "atom {a} g = {g}: im mismatch");
        }
    }
}

/// `atmlst` selects rows, in the order given, and `None` means all of them.
#[test]
fn atmlst_selects_and_reorders_atom_rows() {
    let cell = diamond_bohr();
    let ngrids = MESH[0] * MESH[1] * MESH[2];
    let all = get_si(&cell, None, Some(MESH), None).expect("SI");
    let one = get_si(&cell, None, Some(MESH), Some(&[1])).expect("SI[1]");
    assert_eq!(one.re.len(), ngrids);
    assert_eq!(&one.re[..], &all.re[ngrids..]);
    assert_eq!(&one.im[..], &all.im[ngrids..]);

    let swapped = get_si(&cell, None, Some(MESH), Some(&[1, 0])).expect("SI swapped");
    assert_eq!(&swapped.re[..ngrids], &all.re[ngrids..]);
    assert_eq!(&swapped.re[ngrids..], &all.re[..ngrids]);

    assert!(get_si(&cell, None, Some(MESH), Some(&[7])).is_err());
}

/// The uniform grid is exactly `q_i*a[0] + q_j*a[1] + q_k*a[2]` in C-order, the
/// `wrap_around = true` grid sums to zero on an odd mesh (it is centred on the
/// origin), and the two variants differ only by whole lattice translations.
#[test]
fn uniform_grids_are_the_lattice_fraction_product_grid() {
    for (name, cell) in systems::all() {
        let a = cell.lattice_vectors();
        let ngrids = MESH[0] * MESH[1] * MESH[2];
        let wrapped = get_uniform_grids(&cell, Some(MESH), true).expect("grids");
        let plain = get_uniform_grids(&cell, Some(MESH), false).expect("grids");
        assert_eq!(wrapped.len(), ngrids, "{name}");
        assert_eq!(plain.len(), ngrids, "{name}");
        assert_eq!(wrapped[0], [0.0, 0.0, 0.0], "{name}: origin first");
        assert_eq!(plain[0], [0.0, 0.0, 0.0], "{name}: origin first");

        let fx = fftfreq(MESH[0]);
        let fy = fftfreq(MESH[1]);
        let fz = fftfreq(MESH[2]);
        for g in 0..ngrids {
            let (x, y, z) = (
                g / (MESH[1] * MESH[2]),
                (g / MESH[2]) % MESH[1],
                g % MESH[2],
            );
            for c in 0..3 {
                let want = fx[x] * a[0][c] + fy[y] * a[1][c] + fz[z] * a[2][c];
                assert!(
                    (wrapped[g][c] - want).abs() < 1e-12,
                    "{name} g = {g} c = {c}: {} vs {want}",
                    wrapped[g][c]
                );
            }
            // plain[g] - wrapped[g] is an integer combination of lattice vectors.
            let d = [
                plain[g][0] - wrapped[g][0],
                plain[g][1] - wrapped[g][1],
                plain[g][2] - wrapped[g][2],
            ];
            let inv = pyscf_pbc_gto::inv3(&a).expect("non-singular");
            for (i, _) in inv.iter().enumerate() {
                let n = d[0] * inv[0][i] + d[1] * inv[1][i] + d[2] * inv[2][i];
                assert!(
                    (n - n.round()).abs() < 1e-9,
                    "{name} g = {g}: wrap/no-wrap differ by a non-lattice vector ({n})"
                );
            }
        }
        // Centred on the origin: the wrapped grid sums to zero.
        for c in 0..3 {
            let s: f64 = wrapped.iter().map(|r| r[c]).sum();
            assert!(
                s.abs() < 1e-10,
                "{name} component {c}: wrapped grid sums to {s}"
            );
        }
    }
}

/// D-PBC-20, plan 12-08 — the `inf_vacuum` branch uses the NON-UNIFORM
/// Gauss-Chebyshev base, which changes three things at once.
///
/// 1. The mesh SHRINKS on every vacuum axis: `_non_uniform_Gv_base(n // 2)`
///    returns `2 * (n // 2)` points, so an odd axis comes back one smaller.
///    Upstream says so in a comment (`cell.py:598`, "mesh can be different from
///    the input mesh") and this is the observable consequence.
/// 2. The single scalar weight becomes a PER-GRID array — there is no one
///    weight for a non-uniform rule.
/// 3. The periodic axes keep the uniform frequencies, so a `dimension = 2` cell
///    reduces only `z`.
#[test]
fn inf_vacuum_gv_weights_use_the_non_uniform_base() {
    let mut cell = systems::graphene();
    assert_eq!(cell.dimension, 2);

    // The uniform branch, for comparison.
    let uni = get_gv_weights(&cell, Some(MESH)).expect("uniform Gv");
    assert!(
        uni.weights_per_grid.is_none(),
        "a 3-D-style uniform grid has ONE weight, not an array"
    );
    assert_eq!(uni.mesh, MESH, "the uniform branch does not resize the mesh");

    cell.low_dim_ft_type = LowDimFtType::InfVacuum;
    let vac = get_gv_weights(&cell, Some(MESH)).expect("inf_vacuum Gv");

    // (3) xy are periodic and keep the uniform frequencies; (1) z is reduced.
    assert_eq!(vac.mesh[0], MESH[0], "x is periodic and must not be reduced");
    assert_eq!(vac.mesh[1], MESH[1], "y is periodic and must not be reduced");
    assert_eq!(
        vac.mesh[2],
        2 * (MESH[2] / 2),
        "the vacuum axis takes 2 * (n // 2) Gauss-Chebyshev points"
    );

    // (2) per-grid weights, one per point, all positive and finite.
    let w = vac
        .weights_per_grid
        .as_ref()
        .expect("the inf_vacuum branch produces per-grid weights");
    assert_eq!(w.len(), vac.gv.len(), "one weight per G-vector");
    assert!(
        w.iter().all(|x| x.is_finite() && *x > 0.0),
        "every Gauss-Chebyshev weight is positive and finite"
    );
    // `GvWeights::weight` is the accessor that hides the scalar/array split.
    for g in [0usize, w.len() / 2, w.len() - 1] {
        assert_eq!(vac.weight(g), w[g]);
    }
    assert_eq!(uni.weight(0), uni.weights, "the scalar path still reads back");

    // `get_SI`'s separable branch runs through `get_Gv_weights`, so it now
    // WORKS on the reduced mesh rather than erroring.
    let si = get_si(&cell, None, Some(MESH), None).expect("get_SI on the reduced mesh");
    assert_eq!(
        si.len(),
        cell.mol._atom.len() * vac.gv.len(),
        "get_SI must be sized by the REDUCED mesh, not the requested one"
    );
}

/// The 3-D path is untouched by plan 12-08: still one scalar weight, still the
/// requested mesh.
#[test]
fn three_dimensional_weights_are_still_a_scalar() {
    let cell = diamond_bohr();
    let w = get_gv_weights(&cell, Some(MESH)).expect("Gv");
    assert!(w.weights_per_grid.is_none());
    assert_eq!(w.mesh, MESH);
    assert!(rel(w.weights, 1.0 / cell.vol()) < 1e-14);
}

/// A zero-length mesh axis is an error rather than an empty grid.
#[test]
fn zero_mesh_axis_is_an_error() {
    let cell = diamond_bohr();
    assert!(get_gv_weights(&cell, Some([0, 5, 5])).is_err());
    assert!(get_uniform_grids(&cell, Some([5, 0, 5]), true).is_err());
}

/// `mesh = None` falls back to `cell.mesh` (diamond: `[47,47,47]`).
#[test]
fn default_mesh_comes_from_the_cell() {
    let cell = diamond_bohr();
    let mesh = cell.try_mesh().expect("mesh");
    assert_eq!(mesh, [47, 47, 47]);
    let w = get_gv_weights(&cell, None).expect("Gv");
    assert_eq!(w.mesh, mesh);
    assert_eq!(w.gv.len(), mesh[0] * mesh[1] * mesh[2]);
    assert_eq!(w.gv[0], [0.0, 0.0, 0.0]);
    assert_eq!(
        get_uniform_grids(&cell, None, true).expect("grids").len(),
        mesh[0] * mesh[1] * mesh[2]
    );
}

/// A non-cubic mesh exercises the kernel's `(x, y, z)` index inversion — a
/// swapped `my`/`mz` would still pass on `[5,5,5]`.
#[test]
fn non_cubic_mesh_indexes_correctly() {
    let cell = diamond_bohr();
    let mesh = [3_usize, 4, 5];
    let w = get_gv_weights(&cell, Some(mesh)).expect("Gv");
    assert_eq!(w.gv.len(), 60);
    let b = cell.reciprocal_vectors_2pi().expect("b");
    for (g, gvec) in w.gv.iter().enumerate() {
        let (x, y, z) = (
            g / (mesh[1] * mesh[2]),
            (g / mesh[2]) % mesh[1],
            g % mesh[2],
        );
        for c in 0..3 {
            let want =
                w.gvbase[0][x] * b[0][c] + w.gvbase[1][y] * b[1][c] + w.gvbase[2][z] * b[2][c];
            assert!(
                (gvec[c] - want).abs() < 1e-13,
                "g = {g} c = {c}: {} vs {want}",
                gvec[c]
            );
        }
    }
}

// =========================================================================
// TIER 2 — hard-coded upstream references (D-PBC-19). See UPSTREAM_SNIPPET.
// =========================================================================

/// `np.fft.fftfreq(n, 1./n)` for n = 1..8 — the plan's TEST block.
const FFTFREQ_SCALED_REF: [&[f64]; 8] = [
    &[0.0],
    &[0.0, -1.0],
    &[0.0, 1.0, -1.0],
    &[0.0, 1.0, -2.0, -1.0],
    &[0.0, 1.0, 2.0, -2.0, -1.0],
    &[0.0, 1.0, 2.0, -3.0, -2.0, -1.0],
    &[0.0, 1.0, 2.0, 3.0, -3.0, -2.0, -1.0],
    &[0.0, 1.0, 2.0, 3.0, -4.0, -3.0, -2.0, -1.0],
];

/// `np.fft.fftfreq(n)` for n = 1..8.
const FFTFREQ_REF: [&[f64]; 8] = [
    &[0.0],
    &[0.0, -0.5],
    &[0.0, 0.3333333333333333, -0.3333333333333333],
    &[0.0, 0.25, -0.5, -0.25],
    &[0.0, 0.2, 0.4, -0.4, -0.2],
    &[
        0.0,
        0.16666666666666666,
        0.3333333333333333,
        -0.5,
        -0.3333333333333333,
        -0.16666666666666666,
    ],
    &[
        0.0,
        0.14285714285714285,
        0.2857142857142857,
        0.42857142857142855,
        -0.42857142857142855,
        -0.2857142857142857,
        -0.14285714285714285,
    ],
    &[0.0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125],
];

#[test]
fn fftfreq_tables_match_numpy_for_n_1_to_8() {
    for (k, want) in FFTFREQ_SCALED_REF.iter().enumerate() {
        assert_eq!(
            &fftfreq_scaled(k + 1)[..],
            *want,
            "fftfreq_scaled({})",
            k + 1
        );
    }
    for (k, want) in FFTFREQ_REF.iter().enumerate() {
        // Exact: numpy computes the same `i/n` divisions in the same order.
        assert_eq!(&fftfreq(k + 1)[..], *want, "fftfreq({})", k + 1);
    }
}

/// `c.get_Gv_weights([5,5,5])[2]` for the §9.2 diamond.
const WEIGHTS_555: f64 = 0.013062524449620905;

/// `SI[0, :4]` and `SI[1, :6]` from `c.get_SI(mesh=[5,5,5])`. Atom 0 is at the
/// origin, so its row is `1+0i` throughout.
const SI_ROW1_RE: [f64; 6] = [
    1.0,
    6.123233995736766e-17,
    -1.0,
    -1.0,
    6.123233995736766e-17,
    6.123233995736766e-17,
];
const SI_ROW1_IM: [f64; 6] = [
    -0.0,
    -1.0,
    -1.2246467991473532e-16,
    1.2246467991473532e-16,
    1.0,
    -1.0,
];

/// `c.get_uniform_grids([5,5,5], wrap_around=...)` rows 0, 1, 7, 31 and 124.
const GRID_WRAP_REF: [(usize, [f64; 3]); 5] = [
    (0, [0.0, 0.0, 0.0]),
    (1, [0.6740275141098664, 0.6740275141098664, 0.0]),
    (
        7,
        [2.022082542329599, 1.3480550282197328, 0.6740275141098664],
    ),
    (
        31,
        [1.3480550282197328, 1.3480550282197328, 1.3480550282197328],
    ),
    (
        124,
        [
            -1.3480550282197328,
            -1.3480550282197328,
            -1.3480550282197328,
        ],
    ),
];
const GRID_NOWRAP_REF: [(usize, [f64; 3]); 5] = [
    (0, [0.0, 0.0, 0.0]),
    (1, [0.6740275141098664, 0.6740275141098664, 0.0]),
    (
        7,
        [2.022082542329599, 1.3480550282197328, 0.6740275141098664],
    ),
    (
        31,
        [1.3480550282197328, 1.3480550282197328, 1.3480550282197328],
    ),
    (
        124,
        [5.392220112878931, 5.392220112878931, 5.392220112878931],
    ),
];

/// `np.abs(grids).sum()` for both variants.
const GRID_WRAP_ABS_SUM: f64 = 404.41650846591983;
const GRID_NOWRAP_ABS_SUM: f64 = 1011.0412711647996;

// The 125-row upstream table. Lives under `tests/common/` (not `tests/`) so
// cargo does not treat it as an integration-test target of its own.
include!("common/gv_reference.rs");

/// The plan's headline gate: the full 125x3 `Gv` array for diamond at mesh
/// `[5,5,5]`, to **1e-12**. Uses [`diamond_bohr`] so no Angstrom conversion
/// enters — the only residual is closed-form `inv3` versus numpy's LU inverse.
#[test]
fn diamond_gv_matches_upstream_to_1e_12() {
    let cell = diamond_bohr();
    let w = get_gv_weights(&cell, Some(MESH)).expect("Gv");
    assert_eq!(w.gv.len(), 125);
    assert_eq!(w.gv[0], [0.0, 0.0, 0.0], "Gv row 0 must be the origin");
    assert_eq!(w.gvbase[0], vec![0.0, 1.0, 2.0, -2.0, -1.0]);
    let mut worst = 0.0_f64;
    for (g, (got, want)) in w.gv.iter().zip(GV_DIAMOND_555.iter()).enumerate() {
        for c in 0..3 {
            let d = (got[c] - want[c]).abs();
            worst = worst.max(d);
            assert!(
                d < 1e-12,
                "Gv[{g}][{c}]: got {:.17e}, upstream {:.17e}, |d| = {d:.3e}",
                got[c],
                want[c]
            );
        }
    }
    // The residual is ~1 ULP of the largest |G| (~1.86), not merely under 1e-12.
    assert!(
        worst < 1e-15,
        "Gv residual {worst:.3e} is larger than 1 ULP"
    );
    assert!(rel(w.weights, WEIGHTS_555) < 1e-14, "weights {}", w.weights);
}

/// The §9.2 diamond (built from ANGSTROM) reproduces the same array, but only
/// to a relative 1e-7 — it carries the `pyscf_core::Unit::Ang` gap documented
/// in `tests/cell_build.rs`. The second assertion pins the residual to exactly
/// that gap so the loosened bound cannot hide a real error.
#[test]
fn angstrom_diamond_gv_matches_upstream_within_the_bohr_gap() {
    let cell = systems::diamond();
    let gv = get_gv(&cell, Some(MESH)).expect("Gv");
    let mut worst_rel = 0.0_f64;
    for (g, (got, want)) in gv.iter().zip(GV_DIAMOND_555.iter()).enumerate() {
        for c in 0..3 {
            let r = rel(got[c], want[c]);
            worst_rel = worst_rel.max(r);
            assert!(r < 1e-7, "Gv[{g}][{c}]: relative {r:.3e}");
        }
    }
    assert!(
        worst_rel < 2.0e-8,
        "Angstrom-diamond Gv deviates by {worst_rel:.3e}, more than the known \
         Unit::Ang gap (~4.95e-9); the reciprocal lattice should shrink by exactly that"
    );
}

/// `SI` rows against upstream. `|SI| == 1`, so an ABSOLUTE tolerance is the
/// meaningful one: the near-zero entries are `cos` of an argument near `pi/2`
/// and carry no significant digits of their own.
#[test]
fn structure_factor_rows_match_upstream() {
    let cell = diamond_bohr();
    let ngrids = MESH[0] * MESH[1] * MESH[2];
    let si = get_si(&cell, None, Some(MESH), None).expect("SI");
    // Atom 0 sits at the origin: the whole row is 1+0i.
    for g in 0..4 {
        assert!((si.re[g] - 1.0).abs() < 1e-15);
        assert!(si.im[g].abs() < 1e-15);
    }
    for g in 0..6 {
        let (r, i) = (si.re[ngrids + g], si.im[ngrids + g]);
        assert!(
            (r - SI_ROW1_RE[g]).abs() < 1e-14,
            "SI[1,{g}].re: got {r:.17e}, upstream {:.17e}",
            SI_ROW1_RE[g]
        );
        assert!(
            (i - SI_ROW1_IM[g]).abs() < 1e-14,
            "SI[1,{g}].im: got {i:.17e}, upstream {:.17e}",
            SI_ROW1_IM[g]
        );
    }
}

/// Both `get_uniform_grids` variants against upstream rows and their
/// `abs().sum()` (a whole-array digest that no single row can fake).
#[test]
fn uniform_grids_match_upstream() {
    let cell = diamond_bohr();
    let wrapped = get_uniform_grids(&cell, Some(MESH), true).expect("grids");
    let plain = get_uniform_grids(&cell, Some(MESH), false).expect("grids");

    for (g, want) in GRID_WRAP_REF {
        for c in 0..3 {
            assert!(
                (wrapped[g][c] - want[c]).abs() < 1e-13,
                "wrap grid[{g}][{c}]: got {:.17e}, upstream {:.17e}",
                wrapped[g][c],
                want[c]
            );
        }
    }
    for (g, want) in GRID_NOWRAP_REF {
        for c in 0..3 {
            assert!(
                (plain[g][c] - want[c]).abs() < 1e-13,
                "no-wrap grid[{g}][{c}]: got {:.17e}, upstream {:.17e}",
                plain[g][c],
                want[c]
            );
        }
    }
    let s_w: f64 = wrapped.iter().flat_map(|r| r.iter()).map(|v| v.abs()).sum();
    let s_p: f64 = plain.iter().flat_map(|r| r.iter()).map(|v| v.abs()).sum();
    assert!(rel(s_w, GRID_WRAP_ABS_SUM) < 1e-13, "wrap abs sum {s_w}");
    assert!(
        rel(s_p, GRID_NOWRAP_ABS_SUM) < 1e-13,
        "no-wrap abs sum {s_p}"
    );
}
