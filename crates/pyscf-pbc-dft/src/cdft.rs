//! Constrained DFT — plan 12-07. Port of `pyscf/pbc/dft/cdft.py:36-72`.
//!
//! The whole method is one line of physics: add a CONSTANT diagonal shift to
//! the effective potential, in a chosen (possibly non-AO) working basis, and
//! leave `ecoul`/`exc` untouched so the reported total energy is still the
//! unshifted one.
//!
//! ```text
//! shift = a[:, orbital] ⊗ offset          (cdft.py:57-62)
//! V_eff ← V_eff + diag(shift)
//! ```
//!
//! Upstream's `cdft` monkey-patches `mf.get_veff`; Rust cannot, so this module
//! exposes the shift as a value ([`ShiftHamiltonian`]) plus the two operations
//! that consume it. A driver applies it by calling
//! [`ShiftHamiltonian::apply`] on the `get_veff` output.
//!
//! Upstream emits a `DeprecationWarning` on import and points at
//! `examples/1-advanced/033-constrained_dft.py` for the maintained
//! implementation; this port carries the same status.

use pyscf_algebra::CTensor;
use pyscf_pbc_scf::types::{KDms, KMats};

use crate::error::PbcDftError;
use crate::xc::err;

/// The constant Hamiltonian shift a constrained-DFT run adds to `V_eff`.
#[derive(Debug, Clone)]
pub struct ShiftHamiltonian {
    /// `nao x nao` ROW-MAJOR, real. Upstream's `mf.shift_hamiltonian`.
    pub matrix: Vec<f64>,
    /// AO dimension.
    pub nao: usize,
}

impl ShiftHamiltonian {
    /// `cdft(mf, cell, offset, orbital, basis)` — `cdft.py:36-72`.
    ///
    /// `basis` is the working basis as a COLUMN-MAJOR `nao x nao` matrix
    /// (`basis[ao + col*nao]`); `None` means the AO basis itself. `orbital`
    /// selects the column that is shifted, and `offset` is the shift in
    /// Hartree.
    ///
    /// # Errors
    /// [`PbcDftError`] when `orbital` is out of range or `basis` has the wrong
    /// length.
    pub fn new(
        nao: usize,
        offset: f64,
        orbital: usize,
        basis: Option<&[f64]>,
    ) -> Result<Self, PbcDftError> {
        // cdft.py:48-51 — `a = basis` or the identity.
        // cdft.py:57 — `iaoi = a.T[orbital, :]`, i.e. COLUMN `orbital` of `a`.
        let column: Vec<f64> = match basis {
            None => {
                if orbital >= nao {
                    return Err(err(format!(
                        "cdft: orbital {orbital} is out of range for nao = {nao}"
                    )));
                }
                let mut v = vec![0.0; nao];
                v[orbital] = 1.0;
                v
            }
            Some(a) => {
                if a.len() != nao * nao {
                    return Err(err(format!(
                        "cdft: the working basis has {} entries, expected {}",
                        a.len(),
                        nao * nao
                    )));
                }
                if orbital >= nao {
                    return Err(err(format!(
                        "cdft: orbital {orbital} is out of range for nao = {nao}"
                    )));
                }
                (0..nao).map(|i| a[i + orbital * nao]).collect()
            }
        };
        // cdft.py:61 — `numpy.diag(iaoi) * offset`.
        let mut matrix = vec![0.0_f64; nao * nao];
        for i in 0..nao {
            matrix[i * nao + i] = column[i] * offset;
        }
        Ok(Self { matrix, nao })
    }

    /// Add the shift to every `(channel, k)` block of a `get_veff` result —
    /// `cdft.py:64-70`. The energy components are deliberately NOT touched.
    pub fn apply(&self, veff: &mut KDms) {
        for set in veff.iter_mut() {
            self.apply_channel(set);
        }
    }

    /// Add the shift to one channel's k-resolved matrices.
    pub fn apply_channel(&self, mats: &mut KMats) {
        for m in mats.iter_mut() {
            for i in 0..self.nao * self.nao {
                m.re[i] += self.matrix[i];
            }
        }
    }

    /// The shift as a [`CTensor`], for a caller that wants to inspect it.
    pub fn as_ctensor(&self) -> CTensor {
        CTensor::from_real(&self.matrix)
    }
}
