//! The 5-criterion GAU convergence check + the LOCKED optimizer defaults
//! (GEOMOPT-04).
//!
//! Bodies land in Task 2 of plan 07-04; this Task-1 skeleton ships the locked
//! constants + the [`ConvParams`] type so `lib.rs` compiles. Task 2 fills the
//! [`check_converged`] criteria and the `conv_defaults` unit test.
//!
//! **geomeTRIC GAU preset (LOCKED, GEOMOPT-04):** all 5 criteria must hold.

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

/// Check the 5-criterion GAU convergence (Task-1 skeleton: returns a report
/// with all flags `false`; Task 2 fills the real comparisons).
pub fn check_converged(
    _params: &ConvParams,
    _e_change: f64,
    _grad_rms: f64,
    _grad_max: f64,
    _disp_rms: f64,
    _disp_max: f64,
) -> ConvReport {
    // Task-1 skeleton stub: real criteria land in Task 2.
    ConvReport {
        converged: false,
        energy_ok: false,
        grms_ok: false,
        gmax_ok: false,
        drms_ok: false,
        dmax_ok: false,
    }
}
