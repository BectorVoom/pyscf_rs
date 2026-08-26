//! Shape bookkeeping and the Ewald `G = 0` exchange correction —
//! `pyscf/pbc/df/df_jk.py:1437-1500` (plans 11-07 / 11-08).
//!
//! PBC-MASTER-PLAN plan 14-04 says `_ewald_exxdiv_for_G0` is shared by
//! FFTDF/AFTDF/GDF and must be written ONCE, here, and re-exported — never
//! duplicated per builder. This module is that single home.

use pyscf_algebra::CTensor;
use pyscf_core::PyscfRsError;
use pyscf_pbc_gto::{Cell, PbcIntorOpts, is_zero, madelung, member, pbc_intor};

use crate::zlinalg::{forder_to_c, zaxpy_real, zmm_small};

/// One spin/density channel's k-resolved matrices: `dms[k]` is `nao x nao`
/// row-major. Upstream's `dms[i]` slice of `_format_dms`'s `(nset, nkpts, nao,
/// nao)`.
pub type KMats = Vec<CTensor>;

/// `_format_kpts_band(kpts_band, kpts)` — `df_jk.py:1448-1453`.
///
/// `None` means "evaluate at the sampling k-points themselves", which is what
/// the SCF driver always wants.
pub fn format_kpts_band<'a>(
    kpts_band: Option<&'a [[f64; 3]]>,
    kpts: &'a [[f64; 3]],
) -> &'a [[f64; 3]] {
    kpts_band.unwrap_or(kpts)
}

/// `true` when `kpts_band` addresses exactly the sampling k-points, which is
/// the fast path of [`ewald_exxdiv_for_g0`].
pub fn band_is_kpts(kpts_band: Option<&[[f64; 3]]>, kpts: &[[f64; 3]]) -> bool {
    match kpts_band {
        None => true,
        Some(b) => b.len() == kpts.len() && b.iter().zip(kpts).all(|(x, y)| x == y),
    }
}

/// `_ewald_exxdiv_for_G0(cell, kpts, dms, vk, kpts_band)` — `df_jk.py:1479-1500`.
///
/// Adds `madelung * S D S` to the exchange matrix. This is the probe-charge
/// correction that `get_coulG` deliberately does NOT apply inside the k-loop:
/// upstream's comment says doing it at the end "bypasses any discretization
/// errors that arise from the FFT", and that is worth several digits.
///
/// `vk[i][kband]` is updated in place.
///
/// # Errors
/// Propagates [`pbc_intor`] and [`madelung`] — including `madelung`'s
/// `NotYetImplemented { phase: 12 }` for a 2-D cell.
pub fn ewald_exxdiv_for_g0(
    cell: &Cell,
    kpts: &[[f64; 3]],
    dms: &[KMats],
    vk: &mut [KMats],
    kpts_band: Option<&[[f64; 3]]>,
) -> Result<(), PyscfRsError> {
    let nao = cell.mol.nao_nr;
    let s = pbc_intor(
        cell,
        "int1e_ovlp",
        kpts,
        PbcIntorOpts {
            hermi: 1,
            ..PbcIntorOpts::default()
        },
    )?;
    let mad = madelung(cell, kpts, None)?;

    let shift = |sk: &CTensor, dm: &CTensor| -> CTensor {
        // madelung * S D S, with the two products taken in upstream's order.
        let sd = zmm_small(sk, dm, nao, nao, nao);
        zmm_small(&sd, sk, nao, nao, nao)
    };

    // Phase-10 output is F-order per k-point; Phase 11 works row-major.
    let smats: Vec<CTensor> = s.kmats.iter().map(|m| forder_to_c(m, nao, nao)).collect();

    if band_is_kpts(kpts_band, kpts) {
        // df_jk.py:1491-1494
        for (k, sk) in smats.iter().enumerate() {
            for (i, dmset) in dms.iter().enumerate() {
                let t = shift(sk, &dmset[k]);
                zaxpy_real(&mut vk[i][k], mad, &t);
            }
        }
    } else {
        // df_jk.py:1495-1500 — a band point that coincides with a sampling
        // k-point gets the correction; one that does not gets none.
        let band = kpts_band.unwrap_or(kpts);
        for (k, kpt) in kpts.iter().enumerate() {
            for kp in member(kpt, band) {
                for (i, dmset) in dms.iter().enumerate() {
                    let t = shift(&smats[k], &dmset[k]);
                    zaxpy_real(&mut vk[i][kp], mad, &t);
                }
            }
        }
    }
    Ok(())
}

/// `is_zero(kpts)` over a whole list — upstream's `is_zero(kpts)` on a
/// `(nkpts, 3)` array, which is `True` only when EVERY k-point is gamma.
pub fn all_gamma(kpts: &[[f64; 3]]) -> bool {
    kpts.iter().all(|k| is_zero(k))
}
