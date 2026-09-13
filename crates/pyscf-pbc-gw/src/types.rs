//! Shared G0W0 driver types (`pbc/gw`).
//!
//! Gate C is stated **per route** (19-01 Task 3): analytic continuation (AC)
//! and contour deformation (CD) approximate the same self-energy differently,
//! so [`GwRoute`] travels with every result and route-blind comparisons are
//! refused (see [`crate::error::PbcGwError::RouteBlindComparison`]).

/// Which G0W0 route produced a result. AC and CD must never share a gate
/// number (19-01 Gate C).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GwRoute {
    /// Analytic continuation via Padé (`krgw_ac`, `kugw_ac`).
    AnalyticContinuation,
    /// Contour deformation (`krgw_cd`).
    ContourDeformation,
    /// Slow explicit reference (`kgw_slow`, `gw_slow`, supercell).
    Slow,
}

/// G0W0 driver configuration.
///
/// Ports the shared keyword surface: number of imaginary-frequency points
/// (`nomega`, part of the method — a different grid is a different answer,
/// 19-10), maximum QP iterations, convergence tolerance, and the orbital
/// window the correction is evaluated for.
#[derive(Debug, Clone)]
pub struct GwConfig {
    /// Number of imaginary-frequency grid points (pinned on both sides).
    pub nomega: usize,
    /// Maximum quasiparticle-equation iterations.
    pub max_cycle: usize,
    /// QP energy convergence tolerance (Hartree).
    pub conv_tol: f64,
    /// First orbital index in the correction window.
    pub orlo: usize,
    /// One-past-last orbital index in the correction window.
    pub orhi: usize,
}

impl Default for GwConfig {
    fn default() -> Self {
        Self {
            nomega: 40,
            max_cycle: 50,
            conv_tol: 1e-6,
            orlo: 0,
            orhi: 0,
        }
    }
}

/// Quasiparticle energies for the correction window, with their route.
///
/// `qp_energy[i]` is the G0W0 energy of orbital `orlo + i`. `route` records
/// the approximation so downstream gates cannot mix AC with CD numbers.
#[derive(Debug, Clone)]
pub struct QpResult {
    /// G0W0 quasiparticle energies (Hartree), length `orhi - orlo`.
    pub qp_energy: Vec<f64>,
    /// The route that produced them.
    pub route: GwRoute,
    /// Whether every QP equation in the window met `conv_tol`.
    pub converged: bool,
}
