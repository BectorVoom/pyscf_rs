//! Full X2C one-electron Hamiltonian (`pbc/x2c/x2c1e.py`, 286 l).
//!
//! Upstream's `x2c1e` runs the same Foldy-Wouthuysen decoupling as `sfx2c1e`
//! on the spinor (2-component) blocks: the transform core is shared, and the
//! difference is the block content (spin-orbit coupling in the off-diagonal
//! spin blocks), not the algebra. This module therefore reuses
//! [`crate::sfx2c1e`] (`xmatrix` + `hcore_fw` + `renorm_r`) over the
//! spin-doubled problem: for real symmetric spin blocks the full X2C is the
//! per-spin FW transform assembled block-diagonally, which coincides with the
//! spin-free result doubled (asserted in tests). Explicit complex
//! spin-orbit blocks (spinor `pbc_intor` with `nao_2c`) need the periodic
//! spinor-integral machinery outside this phase's scope and are refused with
//! [`crate::error::PbcX2cError::NotYetImplemented`] rather than silently
//! dropped — dropping SO coupling while claiming full X2C is the
//! plausible-wrong-number shape Gate E guards against.

use crate::error::PbcX2cError;
use crate::sfx2c1e::{LIGHT_SPEED, hcore_fw, xmatrix};

/// Full X2C picture-change Hamiltonian at one k-point (real spin blocks).
///
/// `t/v/w/s` are the row-major `nao × nao` spin-free blocks shared by both
/// spins (closed-shell spinor structure). Returns the row-major `2nao × 2nao`
/// block-diagonal `[h1e, h1e]` Hamiltonian. With explicit spin-orbit blocks,
/// use [`x2c1e_hcore_so`] (refused until the spinor integrals land).
pub fn x2c1e_hcore(t: &[f64], v: &[f64], w: &[f64], s: &[f64], nao: usize) -> Result<Vec<f64>, PbcX2cError> {
    x2c1e_hcore_at_c(t, v, w, s, nao, LIGHT_SPEED)
}

/// [`x2c1e_hcore`] at an explicit speed of light.
pub fn x2c1e_hcore_at_c(
    t: &[f64],
    v: &[f64],
    w: &[f64],
    s: &[f64],
    nao: usize,
    c: f64,
) -> Result<Vec<f64>, PbcX2cError> {
    let x = xmatrix(t, v, w, s, nao, c)?;
    let h1 = hcore_fw(t, v, w, s, &x, nao, c)?;
    let mut out = vec![0.0f64; 4 * nao * nao];
    for i in 0..nao {
        for j in 0..nao {
            out[i * 2 * nao + j] = h1[i * nao + j];
            out[(nao + i) * 2 * nao + nao + j] = h1[i * nao + j];
        }
    }
    Ok(out)
}

/// Full X2C with explicit spin-orbit blocks — refused.
///
/// The spinor off-diagonal blocks need periodic 2-component integrals
/// (`nao_2c`) that no shipped crate provides. Returning the spin-free double
/// here would claim full X2C while dropping SO coupling; refuse instead.
pub fn x2c1e_hcore_so(
    _t: &[f64],
    _v: &[f64],
    _w: &[f64],
    _s: &[f64],
    _nao: usize,
) -> Result<Vec<f64>, PbcX2cError> {
    Err(PbcX2cError::NotYetImplemented {
        module: "x2c/x2c1e spin-orbit blocks (needs periodic spinor integrals)",
    })
}
