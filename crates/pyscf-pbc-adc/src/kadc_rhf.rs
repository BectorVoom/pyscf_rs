//! k-point ADC(2) driver base (`pbc/adc/kadc_rhf.py`, 326 l).
//!
//! Ports the `RADC` driver shape (`kernel_gs`: transform → amplitudes →
//! energy) over the 19-14 base. The IP (19-15) and EA (19-16) manifolds build
//! on this driver and on [`crate::amplitudes`] — never on a fork of either.

use crate::amplitudes::{KadcAmplitudes, build_amplitudes};
use crate::error::PbcAdcError;
use crate::kadc_ao2mo::KadcEris;
use crate::types::AdcLevel;

/// ADC driver configuration (method + counts; spectra flow per call).
#[derive(Debug, Clone)]
pub struct KadcDriver {
    /// Method level (only ADC(2) executes; see [`crate::amplitudes`]).
    pub level: AdcLevel,
    /// Number of k-points.
    pub nkpts: usize,
    /// Occupied / virtual counts per k-point.
    pub nocc: usize,
    /// Virtual count per k-point.
    pub nvir: usize,
}

impl KadcDriver {
    /// Ground-state ADC(2): amplitudes + correlation energy
    /// (`kadc_rhf.kernel_gs`, the `MPn` print is the caller's).
    pub fn kernel_gs(
        &self,
        eris: &KadcEris,
        e_occ_k: &[Vec<f64>],
        e_vir_k: &[Vec<f64>],
        kconserv: &[usize],
    ) -> Result<(f64, KadcAmplitudes), PbcAdcError> {
        if eris.nkpts != self.nkpts || eris.nocc != self.nocc || eris.nvir != self.nvir {
            return Err(PbcAdcError::ShapeMismatch {
                expected: self.nkpts,
                got: eris.nkpts,
            });
        }
        let method = match self.level {
            AdcLevel::Adc2 => "adc(2)",
            AdcLevel::Adc2x => "adc(2)-x",
            AdcLevel::Adc3 => "adc(3)",
        };
        build_amplitudes(eris, e_occ_k, e_vir_k, kconserv, method)
    }
}
