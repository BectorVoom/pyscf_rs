//! Shared ADC driver types (`pbc/adc`).
//!
//! Gate D sits at upstream's own **4 decimals** (19-01): gating at 1e-8 would
//! be four orders tighter than the reference implementation's own tests.

/// ADC method level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdcLevel {
    /// ADC(2) — strict second order.
    Adc2,
    /// ADC(2)-x — extended, first-order 2p-2h/2h-2p block.
    Adc2x,
    /// ADC(3) — third order (reference values only).
    Adc3,
}

/// ADC driver configuration (IP and EA share it; the manifold differs).
#[derive(Debug, Clone)]
pub struct AdcConfig {
    /// Method level.
    pub level: AdcLevel,
    /// Number of roots per k-point.
    pub nroots: usize,
    /// Davidson residual tolerance.
    pub conv_tol: f64,
    /// Maximum Davidson iterations.
    pub max_cycle: usize,
}

impl Default for AdcConfig {
    fn default() -> Self {
        Self {
            level: AdcLevel::Adc2,
            nroots: 3,
            conv_tol: 1e-9,
            max_cycle: 50,
        }
    }
}

/// ADC roots for one k-point: energies ascending with an explicit count.
///
/// Davidson roots are order-sensitive (Phase 16 measured upstream's own
/// `nroots` spread at 5.11e-7): compare sorted eigenvalues with
/// `nroots` stated, never positionally.
#[derive(Debug, Clone)]
pub struct AdcRoots {
    /// Root energies ascending (Hartree), length `nroots`.
    pub energies: Vec<f64>,
    /// Spectroscopic factors per root, length `nroots`.
    pub spec_factors: Vec<f64>,
    /// Whether the solve met `conv_tol`.
    pub converged: bool,
}
