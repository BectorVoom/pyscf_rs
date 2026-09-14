//! Plan 18-03 Task 5 — `ewald_nuc_grad`.
//!
//! Oracle gates against vendored PySCF 2.12.1 `ewald_methods.ewald_nuc_grad`
//! on all five §9.2 reference cells, at 18-01's measured tolerance. No ignore
//! attribute is used anywhere in this file.
//!
//! Branch order (`ewald_methods.py:266-267`) is load-bearing: `dimension == 3
//! && use_particle_mesh_ewald` routes through `particle_mesh_ewald_nuc_grad`
//! (77 lines, the branch that actually runs); otherwise the direct + G-space
//! sum runs. Both branches are gated here — the flag defaults off both here
//! and upstream (`cell.py:1317`), so each cell is measured twice.
//!
//! Graphene (`dimension == 2`) exercises the refusals, not a number:
//! * default `low_dim_ft_type`: upstream raises `NotImplementedError`
//!   (`:288-290`); this port refuses with `NotYetImplemented { phase: 18 }`.
//! * `inf_vacuum`: upstream REACHES the G-space branch (`:274`) but crashes
//!   inside it — the non-uniform base returns a per-grid weight ARRAY and
//!   `:287` passes it to `ctypes.c_double` (`TypeError: only length-1 arrays
//!   can be converted to Python scalars`, reproduced 2026-09-13). No oracle
//!   exists, so this port refuses loudly rather than shipping un-gated
//!   physics.
//!
//! The strong non-zero G-space gate lives in
//! `pyscf-pbc-grad/tests/ewald_nuc_grad.rs` (asymmetric cell, both branches,
//! 1e-9); the symmetric reference cells below pin the PME branch (whose
//! interpolation error is itself O(1e-6) on diamond — matched, not zeroed)
//! and the G-space zeros.

use pyscf_pbc_gto::{Cell, LowDimFtType, ewald_nuc_grad};

mod common;
use common::systems;

/// Tolerance: the PME-vs-PME residuals print on every run and land at ~1e-12;
/// 1e-9 carries three decades of headroom.
const TOL: f64 = 1e-9;

/// (mesh, C-order ravel of the (natm, 3) upstream gradient) per branch.
fn check_cell(name: &str, cell: &Cell, mesh: [usize; 3], pme_off: &[f64], pme_on: &[f64]) {
    assert_eq!(
        cell.mesh, mesh,
        "{name}: FFT mesh differs — the G-space truncation moved"
    );
    for (pme, want) in [(false, pme_off), (true, pme_on)] {
        let mut c = cell.clone();
        c.use_particle_mesh_ewald = pme;
        let got = ewald_nuc_grad(&c, None, None).expect("ewald_nuc_grad");
        assert_eq!(got.len() * 3, want.len());
        let worst = got
            .iter()
            .flat_map(|r| r.iter())
            .zip(want.iter())
            .map(|(g, w)| (g - w).abs())
            .fold(0.0_f64, f64::max);
        println!("{name} pme={pme}: ewald_nuc_grad residual = {worst:e}");
        assert!(
            worst < TOL,
            "{name} pme={pme}: ewald_nuc_grad differs from upstream by {worst:e}\n  \
             got = {got:?}"
        );
    }
}

#[test]
fn ewald_nuc_grad_matches_upstream_on_3d_cells() {
    let cells: Vec<(&str, Cell, [usize; 3], &[f64], &[f64])> = vec![
        (
            "diamond",
            systems::diamond(),
            [47, 47, 47],
            &[
                -4.5011401489097476e-17,
                -9.314409826630589e-18,
                -7.681554236231345e-17,
                8.976307816005088e-17,
                8.370210320379636e-18,
                6.871868733073786e-17,
            ],
            &[
                1.4712682678330078e-06,
                1.4712682680071158e-06,
                1.471268267999965e-06,
                -1.4712682678021338e-06,
                -1.4712682680358156e-06,
                -1.4712682680358173e-06,
            ],
        ),
        (
            "si",
            systems::si(),
            [35, 35, 35],
            &[
                -2.753660196511393e-16,
                7.140975077130324e-18,
                -8.490204416186617e-19,
                2.742679611801917e-16,
                -8.264307301962482e-18,
                -4.561518973030051e-18,
            ],
            &[
                -9.066584179792991e-09,
                -9.066584077106854e-09,
                -9.066584293024453e-09,
                9.066584178694934e-09,
                9.066584075983525e-09,
                9.066584287613913e-09,
            ],
        ),
        (
            "lif",
            systems::lif(),
            [81, 81, 81],
            &[
                6.228932440913979e-17,
                6.18723146483264e-17,
                4.1301252524906826e-17,
                -4.002091754718334e-17,
                -3.932595499082082e-17,
                -5.823215652991555e-17,
            ],
            &[
                1.3407267154829203e-06,
                1.340726715495334e-06,
                1.3407267154043093e-06,
                -1.3407267154606518e-06,
                -1.3407267154727877e-06,
                -1.3407267154212402e-06,
            ],
        ),
        (
            "he_fcc",
            systems::he_fcc(),
            [59, 59, 59],
            &[
                1.2902745550759491e-18,
                2.5119779042002134e-18,
                -1.0068454200191782e-18,
            ],
            &[
                1.2902745550759491e-18,
                2.5119779042002134e-18,
                -1.0068454200191782e-18,
            ],
        ),
    ];
    for (name, cell, mesh, off, on) in cells {
        check_cell(name, &cell, mesh, off, on);
    }
}

/// Graphene refuses on both `low_dim_ft_type` settings — identically to
/// upstream's `NotImplementedError` for the default, and loudly (rather than
/// crashing with `TypeError` like upstream 2.12.1) for `inf_vacuum`.
#[test]
fn graphene_ewald_nuc_grad_refuses_both_low_dim_branches() {
    let cell = systems::graphene();
    assert_eq!(cell.mesh, [45, 45, 351]);
    // Default: upstream `ewald_methods.py:288-290` raises NotImplementedError.
    let err = ewald_nuc_grad(&cell, None, None).expect_err("2D default must refuse");
    assert!(
        matches!(
            err,
            pyscf_core::PyscfRsError::NotYetImplemented { phase: 18, .. }
        ),
        "unexpected error: {err}"
    );
    // inf_vacuum: upstream crashes (TypeError, no oracle); the port refuses.
    let mut vacuum = cell.clone();
    vacuum.low_dim_ft_type = LowDimFtType::InfVacuum;
    let err = ewald_nuc_grad(&vacuum, None, None).expect_err("2D inf_vacuum must refuse");
    assert!(
        matches!(
            err,
            pyscf_core::PyscfRsError::NotYetImplemented { phase: 18, .. }
        ),
        "unexpected error: {err}"
    );
}
