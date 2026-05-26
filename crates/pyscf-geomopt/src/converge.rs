//! The 5-criterion GAU convergence check + the LOCKED optimizer defaults
//! (GEOMOPT-04).
//!
//! **geomeTRIC GAU preset (LOCKED, GEOMOPT-04):** all 5 criteria must hold.
//!
//! ## Displacement unit (Pitfall 6 / RESEARCH A2 — confirmed at port time)
//!
//! The displacement thresholds `convergence_drms = 1.2e-3` and
//! `convergence_dmax = 1.8e-3` are operatively in **Bohr** (atomic units).
//! geomeTRIC's readthedocs documents the displacement criteria in Bohr, while
//! the PySCF `geometric_solver.py` docstring labels the same numeric values
//! Angstrom; the geomeTRIC engine internally overrides `ang2bohr`/`bohr2ang`
//! so the operative numeric thresholds are as tabled and compared in Bohr.
//! This optimizer operates internally in Bohr (the `Mole` invariant —
//! coordinates are stored in Bohr), so the displacement RMS/max are computed
//! in Bohr and compared directly to these constants. **Confirmed unit: Bohr.**

/// Energy-change convergence threshold (Hartree). GAU preset.
pub const CONVERGENCE_ENERGY: f64 = 1.0e-6;
/// RMS-gradient convergence threshold (Hartree/Bohr). GAU preset.
pub const CONVERGENCE_GRMS: f64 = 3.0e-4;
/// Max-gradient convergence threshold (Hartree/Bohr). GAU preset.
pub const CONVERGENCE_GMAX: f64 = 4.5e-4;
/// RMS-displacement convergence threshold (Bohr — see lib.rs Pitfall 6 note).
pub const CONVERGENCE_DRMS: f64 = 1.2e-3;
/// Max-displacement convergence threshold (Bohr — see lib.rs Pitfall 6 note).
pub const CONVERGENCE_DMAX: f64 = 1.8e-3;

/// Default max optimizer steps (LOCKED, capped at the optimize entry).
pub const DEFAULT_MAXSTEPS: usize = 100;
/// Hard upper bound on user-supplied `maxsteps` (T-07-10 DoS guard).
pub const MAX_ALLOWED_MAXSTEPS: usize = 10_000;
/// Initial trust radius (Bohr). LOCKED.
pub const DEFAULT_TRUST: f64 = 0.1;
/// Maximum trust radius (Bohr). LOCKED.
pub const DEFAULT_TMAX: f64 = 0.3;

/// The 5 GAU convergence thresholds + the initial trust radius.
#[derive(Debug, Clone, Copy)]
pub struct ConvParams {
    /// Energy-change threshold (Hartree).
    pub energy: f64,
    /// RMS-gradient threshold (Hartree/Bohr).
    pub grms: f64,
    /// Max-gradient threshold (Hartree/Bohr).
    pub gmax: f64,
    /// RMS-displacement threshold (Bohr).
    pub drms: f64,
    /// Max-displacement threshold (Bohr).
    pub dmax: f64,
    /// Initial trust radius (Bohr).
    pub trust: f64,
    /// Maximum trust radius (Bohr).
    pub tmax: f64,
}

impl ConvParams {
    /// The locked geomeTRIC GAU preset (GEOMOPT-04).
    pub fn gau() -> Self {
        Self {
            energy: CONVERGENCE_ENERGY,
            grms: CONVERGENCE_GRMS,
            gmax: CONVERGENCE_GMAX,
            drms: CONVERGENCE_DRMS,
            dmax: CONVERGENCE_DMAX,
            trust: DEFAULT_TRUST,
            tmax: DEFAULT_TMAX,
        }
    }
}

impl Default for ConvParams {
    fn default() -> Self {
        Self::gau()
    }
}

/// The per-step convergence report (which criteria passed).
#[derive(Debug, Clone, Copy)]
pub struct ConvReport {
    /// All 5 criteria satisfied.
    pub converged: bool,
    /// |ΔE| < energy threshold.
    pub energy_ok: bool,
    /// RMS gradient < grms threshold.
    pub grms_ok: bool,
    /// Max gradient < gmax threshold.
    pub gmax_ok: bool,
    /// RMS displacement < drms threshold.
    pub drms_ok: bool,
    /// Max displacement < dmax threshold.
    pub dmax_ok: bool,
}

/// Check the 5-criterion GAU convergence (GEOMOPT-04). ALL five must hold:
/// `|ΔE| < energy`, `grad_rms < grms`, `grad_max < gmax`,
/// `disp_rms < drms`, `disp_max < dmax`. Energy in Hartree; gradients in
/// Hartree/Bohr; displacements in Bohr (Pitfall 6 — confirmed Bohr).
pub fn check_converged(
    params: &ConvParams,
    e_change: f64,
    grad_rms: f64,
    grad_max: f64,
    disp_rms: f64,
    disp_max: f64,
) -> ConvReport {
    let energy_ok = e_change.abs() < params.energy;
    let grms_ok = grad_rms < params.grms;
    let gmax_ok = grad_max < params.gmax;
    let drms_ok = disp_rms < params.drms;
    let dmax_ok = disp_max < params.dmax;
    ConvReport {
        converged: energy_ok && grms_ok && gmax_ok && drms_ok && dmax_ok,
        energy_ok,
        grms_ok,
        gmax_ok,
        drms_ok,
        dmax_ok,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gau_constants_are_the_locked_preset() {
        assert_eq!(CONVERGENCE_ENERGY, 1.0e-6);
        assert_eq!(CONVERGENCE_GRMS, 3.0e-4);
        assert_eq!(CONVERGENCE_GMAX, 4.5e-4);
        assert_eq!(CONVERGENCE_DRMS, 1.2e-3);
        assert_eq!(CONVERGENCE_DMAX, 1.8e-3);
    }

    #[test]
    fn all_five_criteria_must_hold() {
        let p = ConvParams::gau();
        // All comfortably inside thresholds → converged.
        let r = check_converged(&p, 1e-8, 1e-5, 1e-5, 1e-5, 1e-5);
        assert!(r.converged);
        // One criterion (gmax) violated → NOT converged.
        let r2 = check_converged(&p, 1e-8, 1e-5, 1e-3, 1e-5, 1e-5);
        assert!(!r2.converged);
        assert!(!r2.gmax_ok);
        assert!(r2.energy_ok && r2.grms_ok && r2.drms_ok && r2.dmax_ok);
    }
}
