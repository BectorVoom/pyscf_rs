//! Plan 09-04 Task 3 — cutoffs, `rcut`, `mesh` (PBC-GTO-03).
//!
//! Tier 1 (invariants, no upstream needed — these must pass unconditionally):
//!   * `estimate_rcut` equals `max_shell bas_rcut`, because `_estimate_rcut` is
//!     monotone decreasing in the exponent and `_extract_pgto_params('min')`
//!     hands it the most diffuse primitive of each shell;
//!   * `rcut` grows as `precision` tightens;
//!   * `pgf_rcut` really solves `c*r^(l+2)*exp(-alpha*r^2) = precision`;
//!   * `pgf_rcut_c`'s early return is exactly `rmin` and only fires when the
//!     primitive is already below `precision` there;
//!   * `rcut_by_shells` (loose) never exceeds `bas_rcut` (tight);
//!   * `get_bounding_sphere` zeroes every axis at or beyond `cell.dimension`
//!     and is monotone in `rcut`;
//!   * `error_for_ke_cutoff(estimate_ke_cutoff(p)) ~ p` — the two are inverses;
//!   * every `mesh` axis is odd and >= 1, and `mesh_to_cutoff(cutoff_to_mesh)`
//!     never under-resolves the requested cutoff;
//!   * a built `Cell` no longer carries the `RCUT_UNSET` / `MESH_UNSET`
//!     sentinels (the plan-09-03 gap this plan closes).
//!
//! Tier 2 (hard-coded upstream references, D-PBC-19) — `estimate_rcut`,
//! `estimate_ke_cutoff`, `cell.mesh`, `cutoff_to_mesh(a, 100)`, `nimgs`,
//! `rcut_by_shells` and `bas_rcut` for all five §9.2 systems, generated ONCE
//! from live PySCF 2.12.1 and pasted as literals. The generating snippet is
//! recorded in [`UPSTREAM_SNIPPET`].
//!
//! # Deviation from the plan's stated acceptance numbers
//!
//! PBC-MASTER-PLAN §8.1 plan 09-04 guesses `rcut ~ 15.6` Bohr and
//! `cutoff_to_mesh(a, 100) == [15,15,15]` for diamond/gth-szv, and instructs
//! "regenerate and hard-code these before writing the test". Regenerated: the
//! true upstream values are `rcut = 21.31940052177759` and
//! `cutoff_to_mesh(a, 100.0) == [23,23,23]`. Those are what is asserted.
//!
//! # Tolerances
//!
//! Everything that depends ONLY on the basis (`rcut`, `ke_cutoff`,
//! `rcut_by_shells`, `bas_rcut`) is compared at a relative 1e-12 — exponents
//! and contraction coefficients are unit-independent, so the
//! `pyscf_core::Unit::Ang` gap documented in `tests/cell_build.rs` cannot reach
//! them. Anything that touches the LATTICE (`mesh`, `nimgs`, `mesh_to_cutoff`)
//! inherits that 4.95e-9 relative gap; the integer results are unaffected
//! because they are `ceil`-ed far from a boundary, and the one float result
//! (`mesh_to_cutoff`) is checked at a relative 1e-7.

mod common;

use common::systems;
use pyscf_pbc_gto::cutoff::{
    self, PgtoOp, bas_rcut, error_for_ke_cutoff, estimate_ke_cutoff, estimate_ke_cutoff_pgto,
    estimate_rcut, estimate_rcut_pgto, get_bounding_sphere, get_nimgs, pgf_rcut, pgf_rcut_c,
    rcut_by_shells, rcut_by_shells_with_pgf,
};
use pyscf_pbc_gto::{Cell, LowDimFtType};
use pyscf_pbc_tools::mesh::{cutoff_to_gs, cutoff_to_mesh, gs_to_cutoff, mesh_to_cutoff};

/// The exact snippet that produced every literal in [`UPSTREAM`] (D-PBC-19).
/// Run with `.venv/bin/python` against PySCF 2.12.1; the cells are the §9.2
/// definitions of `pyscf_pbc_gto::test_systems`.
///
/// ```python
/// import numpy as np
/// from pyscf.pbc import gto as pbcgto
/// from pyscf.pbc.gto import cell as cellmod
/// from pyscf.pbc.tools import pbc as pbctools
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
/// a = c.lattice_vectors()
/// cellmod.estimate_rcut(c, 1e-8)          # -> rcut
/// cellmod.estimate_ke_cutoff(c, 1e-8)     # -> ke_cutoff
/// list(c.mesh)                            # -> mesh
/// list(pbctools.cutoff_to_mesh(a, 100.0)) # -> mesh_at_ke100
/// list(c.get_bounding_sphere(c.rcut))     # -> nimgs
/// c.rcut_by_shells(1e-8)                  # -> shell_radii
/// [cellmod.bas_rcut(c, i, 1e-8) for i in range(c.nbas)]   # -> bas_radii
/// cellmod.error_for_ke_cutoff(c, 100.0)   # -> err_at_ke100
/// ```
pub const UPSTREAM_SNIPPET: &str = "see the doc comment above";

/// One row of tier-2 upstream reference values.
struct Upstream {
    name: &'static str,
    /// `estimate_rcut(cell, 1e-8)`, Bohr.
    rcut: f64,
    /// `estimate_ke_cutoff(cell, 1e-8)`, Hartree.
    ke_cutoff: f64,
    /// `cell.mesh` after `build()` with `precision = 1e-8`.
    mesh: [usize; 3],
    /// `pbctools.cutoff_to_mesh(cell.lattice_vectors(), 100.0)`.
    mesh_at_ke100: [usize; 3],
    /// `cell.get_bounding_sphere(cell.rcut)`.
    nimgs: [usize; 3],
    /// `cell.rcut_by_shells(1e-8)`, one per shell.
    shell_radii: &'static [f64],
    /// `bas_rcut(cell, i, 1e-8)`, one per shell.
    bas_radii: &'static [f64],
    /// `error_for_ke_cutoff(cell, 100.0)`.
    err_at_ke100: f64,
}

/// Tier-2 references, generated once from live PySCF 2.12.1 — see
/// [`UPSTREAM_SNIPPET`].
const UPSTREAM: [Upstream; 5] = [
    Upstream {
        name: "diamond",
        rcut: 21.31940052177759,
        ke_cutoff: 422.9075470012404,
        mesh: [47, 47, 47],
        mesh_at_ke100: [23, 23, 23],
        nimgs: [6, 6, 6],
        shell_radii: &[
            13.623847854932555,
            14.179038796017261,
            13.623847854932555,
            14.179038796017261,
        ],
        bas_radii: &[
            19.41306156682725,
            21.31940052177759,
            19.41306156682725,
            21.31940052177759,
        ],
        err_at_ke100: 0.6106383773613504,
    },
    Upstream {
        name: "si",
        rcut: 29.960198598827567,
        ke_cutoff: 108.20807384760171,
        mesh: [35, 35, 35],
        mesh_at_ke100: [35, 35, 35],
        nimgs: [6, 6, 6],
        shell_radii: &[
            19.314077317497432,
            20.39973389173315,
            19.314077317497432,
            20.39973389173315,
        ],
        bas_radii: &[
            26.36134731856624,
            29.960198598827567,
            26.36134731856624,
            29.960198598827567,
        ],
        err_at_ke100: 5.390674657688163e-8,
    },
    Upstream {
        name: "lif",
        rcut: 38.46107083110416,
        ke_cutoff: 1077.5424956328934,
        mesh: [81, 81, 81],
        mesh_at_ke100: [27, 27, 27],
        nimgs: [9, 9, 9],
        shell_radii: &[27.89835539597565, 8.981194551634834, 9.34129612175436],
        bas_radii: &[38.46107083110416, 13.11367284835766, 14.356709660631212],
        err_at_ke100: 25.889538387952545,
    },
    Upstream {
        name: "he_fcc",
        rcut: 16.808894871965055,
        ke_cutoff: 979.7661855059968,
        mesh: [59, 59, 59],
        mesh_at_ke100: [21, 21, 21],
        nimgs: [6, 6, 6],
        shell_radii: &[11.878300882506261],
        bas_radii: &[16.808894871965055],
        err_at_ke100: 0.6197924011412241,
    },
    Upstream {
        name: "graphene",
        rcut: 21.31940052177759,
        ke_cutoff: 422.9075470012404,
        mesh: [45, 45, 351],
        mesh_at_ke100: [23, 23, 173],
        // dimension = 2: the third axis is zeroed by `get_bounding_sphere`.
        nimgs: [6, 6, 0],
        shell_radii: &[
            13.623847854932555,
            14.179038796017261,
            13.623847854932555,
            14.179038796017261,
        ],
        bas_radii: &[
            19.41306156682725,
            21.31940052177759,
            19.41306156682725,
            21.31940052177759,
        ],
        err_at_ke100: 0.6106383773613504,
    },
];

fn upstream(name: &str) -> &'static Upstream {
    UPSTREAM
        .iter()
        .find(|u| u.name == name)
        .expect("every §9.2 system has an upstream row")
}

/// Relative difference, safe at zero.
fn rel(a: f64, b: f64) -> f64 {
    let scale = a.abs().max(b.abs());
    if scale == 0.0 {
        0.0
    } else {
        (a - b).abs() / scale
    }
}

fn assert_rel(actual: f64, expected: f64, tol: f64, what: &str) {
    let r = rel(actual, expected);
    assert!(
        r <= tol,
        "{what}: got {actual:.17e}, upstream {expected:.17e}, relative {r:.3e} > {tol:.1e}"
    );
}

// =========================================================================
// TIER 1 — invariants. No upstream needed; these must pass unconditionally.
// =========================================================================

/// `estimate_rcut` picks the most DIFFUSE primitive of each shell and
/// `_estimate_rcut` is monotone decreasing in `alpha`, so the system-wide
/// cutoff must equal the largest per-shell `bas_rcut`, which maximises over
/// ALL primitives. If the `argmin` tie-break or the `axis=1` coefficient
/// reduction were wrong, these two would disagree.
#[test]
fn estimate_rcut_equals_the_largest_bas_rcut() {
    for (name, cell) in systems::all() {
        let whole = estimate_rcut(&cell, 1e-8);
        let per_shell = (0..cell.mol.nbas)
            .map(|i| bas_rcut(&cell, i, 1e-8))
            .fold(f64::NEG_INFINITY, f64::max);
        assert_rel(
            whole,
            per_shell,
            1e-15,
            &format!("{name}: rcut vs max bas_rcut"),
        );
    }
}

/// A tighter `precision` can only ask for a LARGER lattice sum.
#[test]
fn rcut_grows_as_precision_tightens() {
    for (name, cell) in systems::all() {
        let mut last = f64::NEG_INFINITY;
        for p in [1e-4, 1e-6, 1e-8, 1e-10, 1e-12] {
            let r = estimate_rcut(&cell, p);
            assert!(r.is_finite() && r > 0.0, "{name}: rcut at {p:e} is {r}");
            assert!(
                r > last,
                "{name}: rcut must grow as precision tightens ({r} <= {last})"
            );
            last = r;
        }
    }
}

/// Same for the planewave cutoff.
#[test]
fn ke_cutoff_grows_as_precision_tightens() {
    for (name, cell) in systems::all() {
        let mut last = f64::NEG_INFINITY;
        for p in [1e-4, 1e-6, 1e-8, 1e-10, 1e-12] {
            let ke = estimate_ke_cutoff(&cell, p);
            assert!(ke.is_finite() && ke > 0.0, "{name}: ke at {p:e} is {ke}");
            assert!(
                ke > last,
                "{name}: ke_cutoff must grow as precision tightens"
            );
            last = ke;
        }
    }
}

/// `_estimate_ke_cutoff` is the inverse of `error_for_ke_cutoff`: feeding the
/// estimated cutoff back must reproduce the requested precision. Upstream's
/// two-sweep fixed point lands ~3% high, so the window is deliberately loose
/// on the upper side and tight on the lower.
#[test]
fn ke_cutoff_and_error_estimate_are_inverses() {
    for (name, cell) in systems::all() {
        for p in [1e-6, 1e-8, 1e-10] {
            let ke = estimate_ke_cutoff(&cell, p);
            let err = error_for_ke_cutoff(&cell, ke, None);
            assert!(
                err >= 0.5 * p && err <= 1.5 * p,
                "{name}: error_for_ke_cutoff({ke}) = {err:e} is not within [0.5p, 1.5p] of {p:e}"
            );
        }
    }
}

/// The residual error falls monotonically as the cutoff rises.
#[test]
fn error_for_ke_cutoff_decreases_with_cutoff() {
    for (name, cell) in systems::all() {
        let mut last = f64::INFINITY;
        for ke in [50.0, 100.0, 200.0, 400.0, 800.0] {
            let e = error_for_ke_cutoff(&cell, ke, None);
            assert!(e.is_finite() && e > 0.0, "{name}: error at ke={ke} is {e}");
            assert!(
                e < last,
                "{name}: error must fall as ke rises ({e} >= {last})"
            );
            last = e;
        }
    }
}

/// `pgf_rcut` solves `c*r^(l+2)*exp(-alpha*r^2) = precision`. Check the
/// residual of the defining equation directly, for a spread of `(l, alpha, c)`
/// — this is the one assertion that would catch a transcription slip in the
/// fixed-point body without any reference value.
#[test]
fn pgf_rcut_satisfies_its_defining_equation() {
    let precision = 1e-8;
    for l in 0..=3_i32 {
        for alpha in [0.05, 0.2, 1.0, 5.0] {
            for coeff in [0.05, 1.0, 5.0] {
                let r = pgf_rcut(l, alpha, coeff, precision, 0.0, 20, 1e-9);
                assert!(
                    r.is_finite() && r > 0.0,
                    "l={l} a={alpha} c={coeff}: r = {r}"
                );
                // c * r^(l+2) * exp(-alpha r^2) should be ~ precision.
                let lhs = coeff * r.powf(l as f64 + 2.0) * (-alpha * r * r).exp();
                assert!(
                    rel(lhs, precision) < 1e-6,
                    "l={l} alpha={alpha} c={coeff}: g({r}) = {lhs:e}, wanted {precision:e}"
                );
            }
        }
    }
}

/// The C twin's early return (`lib/pbc/cell.c:36-38`) fires exactly when the
/// primitive is already below `precision` at its own maximum `rmin`, and then
/// returns `rmin` itself.
#[test]
fn pgf_rcut_c_early_return_is_rmin() {
    let precision = 1e-8;
    for l in 0..=3_i32 {
        for alpha in [0.05, 0.2, 1.0, 5.0, 50.0] {
            for coeff in [1e-12, 1e-6, 1.0, 100.0] {
                let ll = l as f64 + 2.0;
                let rmin = (0.5 * ll / alpha).sqrt() * 2.0;
                let gmax = coeff * rmin.powf(ll) * (-alpha * rmin * rmin).exp();
                let r = pgf_rcut_c(l, alpha, coeff, precision, 0.0);
                if gmax < precision {
                    assert_eq!(
                        r, rmin,
                        "l={l} alpha={alpha} c={coeff}: gmax {gmax:e} < precision, expected rmin"
                    );
                } else {
                    assert!(
                        r >= rmin,
                        "l={l} alpha={alpha} c={coeff}: iterated r {r} below rmin {rmin}"
                    );
                }
            }
        }
    }
}

/// `rcut_by_shells` is the LOOSE estimate (`c*r^(l+2)*exp(-alpha r^2)`, the
/// function's own value) and `bas_rcut` the TIGHT one (the overlap of a
/// function with its image, plus a kinetic penalty). The loose one must never
/// exceed the tight one, shell by shell — that is why `use_loose_rcut` is an
/// opt-in speed knob.
#[test]
fn rcut_by_shells_is_looser_than_bas_rcut() {
    for (name, cell) in systems::all() {
        let loose = rcut_by_shells(&cell, 1e-8, 0.0);
        assert_eq!(loose.len(), cell.mol.nbas, "{name}: one radius per shell");
        for (i, r) in loose.iter().enumerate() {
            let tight = bas_rcut(&cell, i, 1e-8);
            assert!(r.is_finite() && *r > 0.0, "{name} shell {i}: radius {r}");
            assert!(
                *r <= tight,
                "{name} shell {i}: loose {r} exceeds tight {tight}"
            );
        }
    }
}

/// The `return_pgf_radius=True` branch: one entry per primitive, and the shell
/// radius is their maximum.
#[test]
fn rcut_by_shells_with_pgf_maxes_over_primitives() {
    for (name, cell) in systems::all() {
        let (shells, pgf) = rcut_by_shells_with_pgf(&cell, 1e-8, 0.0);
        assert_eq!(shells.len(), pgf.len());
        for (i, (s, prims)) in shells.iter().zip(pgf.iter()).enumerate() {
            assert_eq!(
                prims.len(),
                cutoff::bas_nprim(&cell, i),
                "{name} shell {i}: one radius per primitive"
            );
            let m = prims.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            assert_eq!(
                *s, m,
                "{name} shell {i}: shell radius is the max over primitives"
            );
        }
    }
}

/// `get_bounding_sphere` zeroes every axis at or beyond `cell.dimension`
/// (`cell.py:541-542`) and grows monotonically with `rcut`.
#[test]
fn bounding_sphere_respects_dimension_and_grows_with_rcut() {
    for (name, cell) in systems::all() {
        let dim = cell.dimension as usize;
        let n = get_bounding_sphere(&cell, cell.rcut).expect("non-singular lattice");
        for (i, v) in n.iter().enumerate().skip(dim) {
            assert_eq!(*v, 0, "{name}: axis {i} >= dimension {dim} must be zero");
        }
        for (i, v) in n.iter().enumerate().take(dim) {
            assert!(
                *v > 0,
                "{name}: periodic axis {i} must span at least one image"
            );
        }
        let small = get_bounding_sphere(&cell, cell.rcut * 0.5).expect("ok");
        for i in 0..3 {
            assert!(
                small[i] <= n[i],
                "{name}: axis {i} must not shrink as rcut grows"
            );
        }
        // rcut = 0 gives the home cell only.
        assert_eq!(get_bounding_sphere(&cell, 0.0).expect("ok"), [0, 0, 0]);
    }
}

/// `get_nimgs(cell, p)` is `get_bounding_sphere(cell, estimate_rcut(cell, p))`.
#[test]
fn get_nimgs_composes_rcut_and_bounding_sphere() {
    for (name, cell) in systems::all() {
        let via = get_bounding_sphere(&cell, estimate_rcut(&cell, 1e-8)).expect("ok");
        assert_eq!(get_nimgs(&cell, 1e-8).expect("ok"), via, "{name}");
    }
}

/// `cutoff_to_mesh` produces `2*ceil(Gmax) + 1` — always odd, always >= 1 —
/// and never shrinks when the cutoff rises.
#[test]
fn mesh_is_odd_and_monotone_in_cutoff() {
    for (name, cell) in systems::all() {
        let a = cell.lattice_vectors();
        let mut last = [0_usize; 3];
        for ke in [0.0, 10.0, 50.0, 100.0, 500.0] {
            let m = cutoff_to_mesh(&a, ke).expect("non-singular lattice");
            for i in 0..3 {
                assert_eq!(m[i] % 2, 1, "{name}: mesh axis {i} = {} is not odd", m[i]);
                assert!(m[i] >= 1);
                assert!(
                    m[i] >= last[i],
                    "{name}: axis {i} shrank as the cutoff rose"
                );
            }
            last = m;
        }
        assert_eq!(cutoff_to_mesh(&a, 0.0).expect("ok"), [1, 1, 1]);
    }
}

/// A mesh built for `ke` must resolve at least `ke` — otherwise
/// `Cell::build`'s grid would silently under-converge every planewave
/// integral.
#[test]
fn mesh_never_under_resolves_the_requested_cutoff() {
    for (name, cell) in systems::all() {
        let a = cell.lattice_vectors();
        for ke in [10.0, 50.0, 100.0, 422.9, 1000.0] {
            let m = cutoff_to_mesh(&a, ke).expect("ok");
            let back = mesh_to_cutoff(&a, m).expect("ok");
            for i in 0..3 {
                assert!(
                    back[i] >= ke,
                    "{name}: axis {i} mesh {} resolves only {} < {ke}",
                    m[i],
                    back[i]
                );
            }
        }
    }
}

/// The deprecated `gs` spellings are exactly `mesh // 2` and `2*gs + 1`
/// (`pbc.py:830-836`).
#[test]
fn gs_helpers_agree_with_the_mesh_helpers() {
    for (name, cell) in systems::all() {
        let a = cell.lattice_vectors();
        let mesh = cutoff_to_mesh(&a, 100.0).expect("ok");
        let gs = cutoff_to_gs(&a, 100.0).expect("ok");
        for i in 0..3 {
            assert_eq!(gs[i], mesh[i] / 2, "{name}: gs axis {i}");
        }
        let via_gs = gs_to_cutoff(&a, gs).expect("ok");
        let via_mesh =
            mesh_to_cutoff(&a, [2 * gs[0] + 1, 2 * gs[1] + 1, 2 * gs[2] + 1]).expect("ok");
        for i in 0..3 {
            assert_eq!(via_gs[i], via_mesh[i], "{name}: gs_to_cutoff axis {i}");
        }
    }
}

/// The plan-09-03 gap this plan closes: a BUILT cell no longer carries the
/// sentinels, and `try_rcut` / `try_mesh` return the same values as the fields.
/// (This test replaces `unset_rcut_and_mesh_report_the_plan_09_04_gap`.)
#[test]
fn a_built_cell_has_real_rcut_and_mesh() {
    for (name, cell) in systems::all() {
        assert_ne!(
            cell.rcut,
            pyscf_pbc_gto::cell::RCUT_UNSET,
            "{name}: rcut still unset"
        );
        assert_ne!(
            cell.mesh,
            pyscf_pbc_gto::cell::MESH_UNSET,
            "{name}: mesh still unset"
        );
        assert!(
            cell.rcut.is_finite() && cell.rcut > 0.0,
            "{name}: rcut = {}",
            cell.rcut
        );
        assert_eq!(cell.try_rcut().expect("rcut"), cell.rcut, "{name}");
        assert_eq!(cell.try_mesh().expect("mesh"), cell.mesh, "{name}");
        // build() must have used the estimators, not left the user's None.
        assert!(cell._rcut_from_build && cell._mesh_from_build, "{name}");
    }
}

/// `Cell::build` computes `mesh` from the estimated `ke_cutoff` — the same
/// number the free functions give.
#[test]
fn build_mesh_matches_the_free_estimators() {
    for (name, cell) in systems::all() {
        let ke = estimate_ke_cutoff(&cell, cell.precision);
        let a = cell.lattice_vectors();
        assert_eq!(cell.mesh, cutoff_to_mesh(&a, ke).expect("ok"), "{name}");
        assert_eq!(
            cell.mesh,
            cutoff::estimate_mesh(&cell).expect("ok"),
            "{name}"
        );
        assert_eq!(cell.rcut, estimate_rcut(&cell, cell.precision), "{name}");
    }
}

/// `use_loose_rcut` switches `estimate_rcut` to the per-shell radii — a
/// strictly smaller (looser) cutoff for every system here.
#[test]
fn use_loose_rcut_selects_the_shell_radii() {
    for (name, cell) in systems::all() {
        let mut loose = cell.clone();
        loose.use_loose_rcut = true;
        let tight_r = estimate_rcut(&cell, 1e-8);
        let loose_r = estimate_rcut(&loose, 1e-8);
        let by_shells = rcut_by_shells(&cell, 1e-8, 0.0)
            .into_iter()
            .fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(
            loose_r, by_shells,
            "{name}: loose rcut is max(rcut_by_shells)"
        );
        assert!(
            loose_r < tight_r,
            "{name}: loose {loose_r} should be below tight {tight_r}"
        );
    }
}

/// `nbas == 0` short-circuits to upstream's literal `0.01` (`cell.py:426-427`)
/// and `estimate_ke_cutoff` to `0.` (`cell.py:454-455`).
#[test]
fn empty_basis_returns_the_upstream_literals() {
    let cell = Cell::default();
    assert_eq!(cell.mol.nbas, 0);
    assert_eq!(estimate_rcut(&cell, 1e-8), 0.01);
    assert_eq!(estimate_ke_cutoff(&cell, 1e-8), 0.0);
}

/// `omega` is read from `_env[PTR_RANGE_OMEGA]` and picks the range-separated
/// branch of `_estimate_ke_cutoff`, which needs a LOWER cutoff because the
/// short-range operator decays faster.
#[test]
fn nonzero_omega_lowers_the_ke_cutoff() {
    let mut cell = systems::diamond();
    assert_eq!(cutoff::omega(&cell), 0.0, "a fresh cell has omega = 0");
    let plain = estimate_ke_cutoff(&cell, 1e-8);
    cell.mol._env[pyscf_gto::PTR_RANGE_OMEGA] = 0.3;
    assert_eq!(cutoff::omega(&cell), 0.3);
    let screened = estimate_ke_cutoff(&cell, 1e-8);
    assert!(
        screened < plain,
        "omega = 0.3 should lower the cutoff: {screened} vs {plain}"
    );
    // The explicit-omega argument of error_for_ke_cutoff overrides the cell's.
    let e_cell = error_for_ke_cutoff(&cell, 100.0, None);
    let e_arg = error_for_ke_cutoff(&cell, 100.0, Some(0.3));
    assert_eq!(e_cell, e_arg);
    assert_ne!(e_cell, error_for_ke_cutoff(&cell, 100.0, Some(0.0)));
}

/// `_extract_pgto_params` selects the smallest / largest exponent of each
/// shell and the largest absolute contraction coefficient of THAT primitive.
#[test]
fn extract_pgto_params_selects_the_extremal_primitive() {
    for (name, cell) in systems::all() {
        let (emin, cmin) = cutoff::extract_pgto_params(&cell, PgtoOp::Min);
        let (emax, cmax) = cutoff::extract_pgto_params(&cell, PgtoOp::Max);
        assert_eq!(emin.len(), cell.mol.nbas, "{name}");
        assert_eq!(emax.len(), cell.mol.nbas, "{name}");
        for i in 0..cell.mol.nbas {
            let es = cutoff::bas_exp(&cell, i);
            let cs = cutoff::libcint_ctr_coeff_max(&cell, i);
            let lo = es.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = es.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            assert_eq!(emin[i], lo, "{name} shell {i}: min exponent");
            assert_eq!(emax[i], hi, "{name} shell {i}: max exponent");
            let ilo = es.iter().position(|e| *e == lo).expect("present");
            let ihi = es.iter().position(|e| *e == hi).expect("present");
            assert_eq!(
                cmin[i], cs[ilo],
                "{name} shell {i}: coefficient of the min-exp primitive"
            );
            assert_eq!(
                cmax[i], cs[ihi],
                "{name} shell {i}: coefficient of the max-exp primitive"
            );
        }
    }
}

/// `_estimate_rcut` / `_estimate_ke_cutoff` are monotone in the exponent:
/// diffuse primitives need a bigger box, compact ones a finer grid.
#[test]
fn per_primitive_estimators_are_monotone_in_alpha() {
    for l in 0..=3_i32 {
        let mut last_r = f64::INFINITY;
        let mut last_ke = f64::NEG_INFINITY;
        for alpha in [0.05, 0.1, 0.5, 1.0, 5.0, 20.0] {
            let r = estimate_rcut_pgto(alpha, l, 1.0, 1e-8);
            let ke = estimate_ke_cutoff_pgto(alpha, l, 1.0, 1e-8, 0.0);
            assert!(r.is_finite() && r > 0.0 && ke.is_finite() && ke > 0.0);
            assert!(r < last_r, "l={l}: rcut must fall as alpha rises");
            assert!(ke > last_ke, "l={l}: ke_cutoff must rise as alpha rises");
            last_r = r;
            last_ke = ke;
        }
    }
}

/// The `dimension <= 2 && inf_vacuum` branch of `estimate_mesh`
/// (`cell.py:1765-1766`): the non-periodic axes take `_mesh_inf_vaccum`, which
/// is always EVEN (the `z+`/`z-` symmetry requires it).
#[test]
fn inf_vacuum_replaces_the_nonperiodic_mesh_axes() {
    let mut cell = systems::graphene();
    let plain = cutoff::estimate_mesh(&cell).expect("ok");
    cell.low_dim_ft_type = LowDimFtType::InfVacuum;
    let vac = cutoff::estimate_mesh(&cell).expect("ok");
    let meshz = cutoff::mesh_inf_vacuum(&cell);
    assert_eq!(meshz % 2, 0, "meshz must be even, got {meshz}");
    assert!(meshz > 0);
    // Periodic axes unchanged, the vacuum axis replaced.
    assert_eq!(vac[0], plain[0]);
    assert_eq!(vac[1], plain[1]);
    assert_eq!(vac[2], meshz);
    assert_ne!(vac[2], plain[2]);
}

// =========================================================================
// TIER 2 — hard-coded upstream references (D-PBC-19). See UPSTREAM_SNIPPET.
// =========================================================================

/// The plan's headline acceptance gate, with the REGENERATED numbers (the
/// plan's own `rcut ~ 15.6` / `mesh == [15,15,15]` guess is wrong — see the
/// module docs).
#[test]
fn diamond_matches_the_upstream_rcut_and_mesh() {
    let cell = systems::diamond();
    let a = cell.lattice_vectors();
    assert_rel(
        estimate_rcut(&cell, 1e-8),
        21.31940052177759,
        1e-12,
        "diamond rcut",
    );
    assert_eq!(cutoff_to_mesh(&a, 100.0).expect("ok"), [23, 23, 23]);
}

/// `estimate_rcut` and `estimate_ke_cutoff` for all five systems. Both depend
/// on the BASIS only (exponents and libcint contraction coefficients are
/// unit-independent), so the `Unit::Ang` gap of `tests/cell_build.rs` cannot
/// reach them and the tolerance is a tight relative 1e-12.
#[test]
fn rcut_and_ke_cutoff_match_upstream() {
    for (name, cell) in systems::all() {
        let u = upstream(name);
        assert_rel(
            estimate_rcut(&cell, 1e-8),
            u.rcut,
            1e-12,
            &format!("{name} rcut"),
        );
        assert_rel(
            estimate_ke_cutoff(&cell, 1e-8),
            u.ke_cutoff,
            1e-12,
            &format!("{name} ke_cutoff"),
        );
        // build() stored the same rcut.
        assert_rel(cell.rcut, u.rcut, 1e-12, &format!("{name} cell.rcut"));
    }
}

/// `cell.mesh` after `build()`, and `cutoff_to_mesh(a, 100)`. These DO touch
/// the lattice, but the results are integers produced by `ceil` far from a
/// boundary, so the 4.95e-9 `Unit::Ang` gap cannot move them — an exact match
/// is the right assertion, and a failure here would mean a real port bug.
#[test]
fn mesh_matches_upstream() {
    for (name, cell) in systems::all() {
        let u = upstream(name);
        let a = cell.lattice_vectors();
        assert_eq!(cell.mesh, u.mesh, "{name}: cell.mesh");
        assert_eq!(
            cutoff_to_mesh(&a, 100.0).expect("ok"),
            u.mesh_at_ke100,
            "{name}: cutoff_to_mesh(a, 100)"
        );
    }
}

/// `get_bounding_sphere(cell.rcut)`, including graphene's zeroed third axis.
#[test]
fn nimgs_matches_upstream() {
    for (name, cell) in systems::all() {
        let u = upstream(name);
        assert_eq!(
            get_bounding_sphere(&cell, cell.rcut).expect("ok"),
            u.nimgs,
            "{name}: nimgs"
        );
        assert_eq!(
            get_nimgs(&cell, 1e-8).expect("ok"),
            u.nimgs,
            "{name}: get_nimgs"
        );
    }
}

/// Per-shell radii from both estimators. `rcut_by_shells` goes through the C
/// twin `pgf_rcut_c` (early return included), which is what upstream's
/// `cell.rcut_by_shells` calls via `libpbc`.
#[test]
fn per_shell_radii_match_upstream() {
    for (name, cell) in systems::all() {
        let u = upstream(name);
        let loose = rcut_by_shells(&cell, 1e-8, 0.0);
        assert_eq!(loose.len(), u.shell_radii.len(), "{name}: shell count");
        for (i, (got, want)) in loose.iter().zip(u.shell_radii.iter()).enumerate() {
            assert_rel(*got, *want, 1e-12, &format!("{name} rcut_by_shells[{i}]"));
        }
        for (i, want) in u.bas_radii.iter().enumerate() {
            assert_rel(
                bas_rcut(&cell, i, 1e-8),
                *want,
                1e-12,
                &format!("{name} bas_rcut[{i}]"),
            );
        }
        // The Cell methods are the same thing with `precision = cell.precision`.
        assert_eq!(
            cell.rcut_by_shells(None),
            loose,
            "{name}: Cell::rcut_by_shells"
        );
        assert_eq!(
            cell.bas_rcut(0, None),
            bas_rcut(&cell, 0, 1e-8),
            "{name}: Cell::bas_rcut"
        );
    }
}

/// `error_for_ke_cutoff(cell, 100.0)` — spans 25.9 (LiF, badly under-converged
/// at 100 Ha) down to 5.4e-8 (Si), so a transcription slip in the `l - 0.5`
/// exponent or the `(2*alpha)^(2l+0.5)` denominator cannot hide.
#[test]
fn error_for_ke_cutoff_matches_upstream() {
    for (name, cell) in systems::all() {
        let u = upstream(name);
        assert_rel(
            error_for_ke_cutoff(&cell, 100.0, None),
            u.err_at_ke100,
            1e-12,
            &format!("{name} error_for_ke_cutoff(100)"),
        );
    }
}

/// `mesh_to_cutoff` on a fixed `[15,15,15]` mesh. This one IS a float that
/// rides on the lattice, so it carries the documented `Unit::Ang` gap squared
/// (~9.9e-9 relative) and is checked at 1e-7 — the same bound
/// `tests/cell_build.rs` uses, for the same reason.
#[test]
fn mesh_to_cutoff_matches_upstream_within_the_bohr_gap() {
    // Upstream `pbctools.mesh_to_cutoff(cell.lattice_vectors(), [15,15,15])`,
    // first axis (all three agree to ~1e-15 for these cells).
    const WANT: [(&str, f64); 5] = [
        ("diamond", 42.579500924406574),
        ("si", 18.36802459047572),
        ("lif", 33.35400506797271),
        ("he_fcc", 60.188784545382006),
        ("graphene", 44.756680952842046),
    ];
    for (name, cell) in systems::all() {
        let want = WANT.iter().find(|(n, _)| *n == name).expect("row").1;
        let ke = mesh_to_cutoff(&cell.lattice_vectors(), [15, 15, 15]).expect("ok");
        assert_rel(ke[0], want, 1e-7, &format!("{name} mesh_to_cutoff[0]"));
        // The residual must be the Bohr-constant gap (~9.9e-9), not something
        // larger hiding behind the loosened bound.
        assert!(
            rel(ke[0], want) < 2.0e-8,
            "{name}: mesh_to_cutoff deviates by {:.3e}, more than the known \
             Unit::Ang gap squared (~9.9e-9)",
            rel(ke[0], want)
        );
    }
}
