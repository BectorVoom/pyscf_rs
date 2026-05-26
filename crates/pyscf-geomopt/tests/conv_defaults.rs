//! Integration test: the 5 GAU convergence constants + the optimizer-level
//! defaults are LOCKED to the geomeTRIC GAU preset (GEOMOPT-04, Task 2 of
//! plan 07-04). Always-on, no SCF/cintx dependency.

use pyscf_geomopt::{ConvParams, GeometryOptimizer, check_converged};

#[test]
fn gau_convergence_constants_exact() {
    // The 5 GAU energy/gradient/displacement thresholds (GEOMOPT-04, LOCKED).
    let p = ConvParams::gau();
    assert_eq!(p.energy, 1.0e-6, "convergence_energy must be 1.0e-6 Eh");
    assert_eq!(p.grms, 3.0e-4, "convergence_grms must be 3.0e-4 Eh/Bohr");
    assert_eq!(p.gmax, 4.5e-4, "convergence_gmax must be 4.5e-4 Eh/Bohr");
    assert_eq!(p.drms, 1.2e-3, "convergence_drms must be 1.2e-3 Bohr");
    assert_eq!(p.dmax, 1.8e-3, "convergence_dmax must be 1.8e-3 Bohr");
}

#[test]
fn optimizer_level_defaults_locked() {
    // maxsteps == 100, initial trust == 0.1, tmax == 0.3 (LOCKED).
    let opt = GeometryOptimizer::new();
    assert_eq!(opt.maxsteps, 100, "default maxsteps must be 100");
    assert_eq!(opt.conv_params.trust, 0.1, "initial trust must be 0.1 Bohr");
    assert_eq!(opt.conv_params.tmax, 0.3, "tmax must be 0.3 Bohr");
    assert!(
        !opt.has_constraints,
        "constraints default off (raises a clear error if set)"
    );
}

#[test]
fn default_optimizer_uses_gau_preset() {
    let opt = GeometryOptimizer::default();
    let p = ConvParams::gau();
    assert_eq!(opt.conv_params.energy, p.energy);
    assert_eq!(opt.conv_params.grms, p.grms);
    assert_eq!(opt.conv_params.gmax, p.gmax);
    assert_eq!(opt.conv_params.drms, p.drms);
    assert_eq!(opt.conv_params.dmax, p.dmax);
}

#[test]
fn convergence_requires_all_five_criteria() {
    let p = ConvParams::gau();
    // Just inside all thresholds → converged.
    assert!(check_converged(&p, 9e-7, 2.9e-4, 4.4e-4, 1.1e-3, 1.7e-3).converged);
    // Energy just over → not converged.
    assert!(!check_converged(&p, 2e-6, 2.9e-4, 4.4e-4, 1.1e-3, 1.7e-3).converged);
    // Displacement-max just over → not converged.
    assert!(!check_converged(&p, 9e-7, 2.9e-4, 4.4e-4, 1.1e-3, 2.0e-3).converged);
}
