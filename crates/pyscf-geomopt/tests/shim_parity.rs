//! GEOMOPT-02/03 + D-07: `geometric_solver` / `berny_solver` shim parity over
//! the ONE native engine, plus the clear-error `constraints` guard.
//!
//! This proves the Rust-side shim surface (the pyo3-free layer in
//! `pyscf-geomopt`; the Python `pyscf.geomopt.geometric_solver.optimize` /
//! `berny_solver.optimize` entry points land in the 07-09 PyO3 bridge):
//!
//! 1. **Parity (always-on):** both `geometric_solver` and `berny_solver` shim
//!    entry points accept the upstream kwargs (`conv_params`, `callback`,
//!    `maxsteps`, `constraints`) and delegate to the SAME native `optimize`
//!    engine (07-04). `kernel` returns `(conv, mol)`; `optimize` returns the
//!    optimized `Mole`. The two solvers produce identical results (berny is a
//!    thin ALIAS — there is NO second optimizer, D-06).
//! 2. **constraints clear-error:** passing a non-`None` `constraints` raises a
//!    clear, descriptive [`GeomError::ConstraintsUnsupported`] (GEOMOPT-EXT-01
//!    deferred, D-07) — NEVER a silent no-op (T-07-17).
//! 3. **maxsteps DoS cap (T-07-18):** an out-of-range `maxsteps` is rejected at
//!    the shim boundary with a clear error.
//!
//! Self-contained: it reuses the internal-only analytic harmonic `GradScanner`
//! (no SCF, no cintx grad integral), so it runs in every `cargo test`.

use pyscf_core::{Energy, Mole};
use pyscf_geomopt::shims::{ShimParams, berny_solver, geometric_solver};
use pyscf_geomopt::{GeomError, NATIVE_ENGINE_NAME};
use pyscf_grad::GradScanner;
use pyscf_grad::scanner::{EnergyClosure, GradClosure};
use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};

const EQ_ROH: f64 = 1.81;
const EQ_THETA: f64 = 104.5; // degrees

fn perturbed_h2o() -> Mole {
    M(MoleBuildArgs {
        atom: AtomInput::String("O 0.0 0.0 0.0; H 1.30 1.30 0.0; H -1.30 1.30 0.0".into()),
        basis: BasisInput::Name("sto-3g".into()),
        unit: pyscf_core::Unit::Bohr,
        ..Default::default()
    })
    .expect("build perturbed H2O/STO-3G")
}

fn harmonic_scanner(kb: f64, ka: f64) -> GradScanner {
    let theta_eq = EQ_THETA.to_radians();
    let e_of = move |coords: &[[f64; 3]]| -> f64 {
        let r1 = dist(coords, 0, 1);
        let r2 = dist(coords, 0, 2);
        let th = angle_at(coords, 1, 0, 2);
        0.5 * kb * ((r1 - EQ_ROH).powi(2) + (r2 - EQ_ROH).powi(2)) + 0.5 * ka * (th - theta_eq).powi(2)
    };
    let e_energy = e_of;
    let energy: EnergyClosure = Box::new(move |mol: &Mole| Ok(Energy(e_energy(&mol.atom_coords()))));
    let grad: GradClosure = Box::new(move |mol: &Mole, _atmlst: Option<&[usize]>| {
        let coords = mol.atom_coords();
        let h = 1e-6;
        let mut de = vec![[0.0_f64; 3]; coords.len()];
        for a in 0..coords.len() {
            for c in 0..3 {
                let mut cp = coords.clone();
                let mut cm = coords.clone();
                cp[a][c] += h;
                cm[a][c] -= h;
                de[a][c] = (e_of(&cp) - e_of(&cm)) / (2.0 * h);
            }
        }
        Ok(de)
    });
    GradScanner::new(energy, grad)
}

fn dist(coords: &[[f64; 3]], a: usize, b: usize) -> f64 {
    let dx = coords[a][0] - coords[b][0];
    let dy = coords[a][1] - coords[b][1];
    let dz = coords[a][2] - coords[b][2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn angle_at(coords: &[[f64; 3]], a: usize, apex: usize, c: usize) -> f64 {
    let u = [
        coords[a][0] - coords[apex][0],
        coords[a][1] - coords[apex][1],
        coords[a][2] - coords[apex][2],
    ];
    let v = [
        coords[c][0] - coords[apex][0],
        coords[c][1] - coords[apex][1],
        coords[c][2] - coords[apex][2],
    ];
    let dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    let nu = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
    let nv = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    (dot / (nu * nv)).clamp(-1.0, 1.0).acos()
}

#[test]
fn geometric_solver_kernel_returns_conv_mol_shape() {
    // kernel mirrors upstream `kernel(method, ..., maxsteps=100) -> (conv, mol)`.
    let mol = perturbed_h2o();
    let scanner = harmonic_scanner(0.5, 0.2);
    let (conv, optimized): (bool, Mole) =
        geometric_solver::kernel(&scanner, &mol, ShimParams::default()).expect("kernel must not error");
    assert!(conv, "the model PES must converge to its minimum");
    // The returned mol carries the optimized geometry (H2O equilibrium).
    let coords = optimized.atom_coords();
    let oh = dist(&coords, 0, 1);
    let ang = angle_at(&coords, 1, 0, 2).to_degrees();
    assert!(
        (oh - EQ_ROH).abs() < 0.02 && (ang - EQ_THETA).abs() < 1.0,
        "kernel's returned mol must be the optimized geometry, got O–H={oh:.4}, ∠={ang:.2}°"
    );
}

#[test]
fn geometric_solver_optimize_returns_mol() {
    // optimize mirrors upstream `optimize(...) -> mol` (== kernel(...).1).
    let mol = perturbed_h2o();
    let scanner = harmonic_scanner(0.5, 0.2);
    let optimized: Mole =
        geometric_solver::optimize(&scanner, &mol, ShimParams::default()).expect("optimize must not error");
    let coords = optimized.atom_coords();
    let oh = dist(&coords, 0, 1);
    assert!(
        (oh - EQ_ROH).abs() < 0.02,
        "optimize must return the optimized Mole, got O–H={oh:.4}"
    );
}

#[test]
fn berny_and_geometric_delegate_to_the_same_one_engine() {
    // The structural single-engine assertion (D-06 / T-07-20): berny is a thin
    // ALIAS over the one native engine, so it must produce the SAME optimized
    // geometry as geometric_solver for the same input.
    let mol = perturbed_h2o();
    let scanner = harmonic_scanner(0.5, 0.2);

    let geo = geometric_solver::optimize(&scanner, &mol, ShimParams::default()).expect("geometric");
    let scanner2 = harmonic_scanner(0.5, 0.2);
    let berny = berny_solver::optimize(&scanner2, &mol, ShimParams::default()).expect("berny");

    let gc = geo.atom_coords();
    let bc = berny.atom_coords();
    assert_eq!(gc.len(), bc.len());
    for (g, b) in gc.iter().zip(bc.iter()) {
        for k in 0..3 {
            assert!(
                (g[k] - b[k]).abs() < 1e-12,
                "berny + geometric must drive the SAME engine to bit-identical geometries (no second optimizer): {} vs {}",
                g[k],
                b[k]
            );
        }
    }

    // And both shims name the ONE native engine (no second berny implementation
    // — the single-engine marker is a single shared constant).
    assert_eq!(
        geometric_solver::engine_name(),
        NATIVE_ENGINE_NAME,
        "geometric_solver must delegate to the one native engine"
    );
    assert_eq!(
        berny_solver::engine_name(),
        NATIVE_ENGINE_NAME,
        "berny_solver must be a thin alias over the SAME one native engine"
    );
    assert_eq!(
        geometric_solver::engine_name(),
        berny_solver::engine_name(),
        "berny + geometric must report the identical single engine"
    );
}

#[test]
fn constraints_kwarg_raises_clear_error_not_silent_noop() {
    // T-07-17 / D-07: a non-None constraints kwarg must raise a CLEAR error,
    // never be silently ignored (which would mis-optimize a constrained input).
    let mol = perturbed_h2o();
    let scanner = harmonic_scanner(0.5, 0.2);
    let params = ShimParams {
        constraints: Some("freeze distance 1 2".to_string()),
        ..ShimParams::default()
    };

    // Both shims must reject constraints with the same clear error.
    for result in [
        geometric_solver::kernel(&scanner, &mol, params.clone()),
        berny_solver::kernel(&scanner, &mol, params.clone()),
    ] {
        match result {
            Err(GeomError::ConstraintsUnsupported) => { /* the required clear error */ }
            Err(other) => panic!("expected ConstraintsUnsupported, got a different error: {other}"),
            Ok(_) => panic!("constraints kwarg was SILENTLY IGNORED — must raise a clear error (T-07-17)"),
        }
    }

    // The error message must be descriptive (cite the deferred feature).
    let msg = format!("{}", GeomError::ConstraintsUnsupported);
    assert!(
        msg.to_lowercase().contains("constraint"),
        "the constraints error must be descriptive, got: {msg}"
    );
}

#[test]
fn maxsteps_out_of_range_rejected_at_shim_boundary() {
    // T-07-18: the shims cap/validate maxsteps (DoS guard). 0 and an absurd
    // value must both be rejected with a clear error.
    let mol = perturbed_h2o();
    let scanner = harmonic_scanner(0.5, 0.2);
    for bad in [0usize, 10_000_001usize] {
        let params = ShimParams {
            maxsteps: bad,
            ..ShimParams::default()
        };
        let r = geometric_solver::kernel(&scanner, &mol, params);
        assert!(
            matches!(r, Err(GeomError::InvalidMaxSteps { .. })),
            "maxsteps={bad} must be rejected at the shim boundary (T-07-18), got {r:?}"
        );
    }
}

#[test]
fn shim_default_maxsteps_is_100() {
    // D-07: the shims default maxsteps to 100 (the upstream default).
    let p = ShimParams::default();
    assert_eq!(p.maxsteps, 100, "the shim default maxsteps must be 100 (upstream parity)");
    assert!(p.constraints.is_none(), "default constraints must be None");
}
