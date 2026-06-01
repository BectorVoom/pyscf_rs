//! Molecular cart→sph coefficient matrix — the `Mole.cart2sph_coeff` analog
//! (`pyscf/gto/mole.py`).
//!
//! Used by the cartesian-basis init-guess density projection
//! (`pyscf-scf::init_guess`): a spherical density `D_sph` is mapped to the
//! cartesian molecular basis via `D_cart = C · D_sph · Cᵀ`.
//!
//! `C[cart, sph]` is block-diagonal over shell-contractions; each block is the
//! **transpose** of the libcint `g_trans_cart2sph` matrix exposed by
//! `pyscf_kernels::cart2sph_l_matrix`. This equals PySCF's
//! `Mole.cart2sph_coeff(normalized='sp')` element-for-element (verified vs PySCF
//! 2.12.1: s/p blocks identity, d/f/… match), because `'sp'` normalization is
//! the identity for `l ≤ 1` and the libcint convention already carries the
//! l ≥ 2 angular factors.

use pyscf_core::raw_layout::{ANG_OF, BAS_SLOTS, NCTR_OF};
use pyscf_core::{Mole, PyscfRsError};

#[inline]
fn ncart(l: u32) -> usize {
    ((l + 1) * (l + 2) / 2) as usize
}
#[inline]
fn nsph(l: u32) -> usize {
    (2 * l + 1) as usize
}

/// Build the molecular cart→sph coefficient matrix `C`.
///
/// Returns `(C, nao_sph)` where `C` is row-major `[nao_cart × nao_sph]`
/// (`C[cart * nao_sph + sph]`) and `nao_cart` equals `mol.nao_nr` when the Mole
/// is cartesian. Walks `mol._bas` in shell order (matching the cartesian AO
/// ordering `make_env` produces and the spherical AO ordering of a spherical
/// sibling Mole). `NotYetImplemented{phase:4}` propagates for `l > 6`.
pub fn cart2sph_coeff(mol: &Mole) -> Result<(Vec<f64>, usize), PyscfRsError> {
    let nbas = mol.nbas;

    let mut nao_cart = 0usize;
    let mut nao_sph = 0usize;
    for ib in 0..nbas {
        let l = mol._bas[ib * BAS_SLOTS + ANG_OF] as u32;
        let nctr = mol._bas[ib * BAS_SLOTS + NCTR_OF] as usize;
        nao_cart += ncart(l) * nctr;
        nao_sph += nsph(l) * nctr;
    }

    let mut c = vec![0.0f64; nao_cart * nao_sph];
    let mut coff = 0usize;
    let mut soff = 0usize;
    for ib in 0..nbas {
        let l = mol._bas[ib * BAS_SLOTS + ANG_OF] as u32;
        let nctr = mol._bas[ib * BAS_SLOTS + NCTR_OF] as usize;
        let (nc, ns) = (ncart(l), nsph(l));
        // t[m * nc + cart] = c2s_coeff(l, m, cart). C block = tᵀ.
        let t = pyscf_kernels::cart2sph_l_matrix(l)?;
        for _ictr in 0..nctr {
            for cart in 0..nc {
                for m in 0..ns {
                    c[(coff + cart) * nao_sph + (soff + m)] = t[m * nc + cart];
                }
            }
            coff += nc;
            soff += ns;
        }
    }

    Ok((c, nao_sph))
}
