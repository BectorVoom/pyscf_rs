//! The single-centre AO Fourier transform — `pyscf/gto/ft_ao.py` (plan 13-02),
//! plus the fake nuclear cell it is evaluated on (`aft.py:247-274`).
//!
//! ```text
//! ft_ao[μ, G] = Σ_prim c · (π/α)^{3/2} · e^{−|G|²/4α}
//!               · Σ_{t,u,v} E_t^{i_x,0} E_u^{i_y,0} E_v^{i_z,0}
//!                 · (−iG_x)^t (−iG_y)^u (−iG_z)^v · e^{−iG·A}
//! ```
//!
//! # Deviation from the plan's file placement, recorded deliberately
//!
//! PBC-MASTER-PLAN §8.5 plan 13-02 puts this in `crates/pyscf-gto/src/ft_ao.rs`,
//! mirroring upstream's `pyscf/gto/ft_ao.py`. It lives here because the
//! McMurchie–Davidson recursion it must reuse ([`super::mcmurchie`]) lives here:
//! putting `ft_ao` in `pyscf-gto` would either duplicate the recursion — which
//! plan 13-02's own must-haves forbid ("one recursion, one place to be wrong") —
//! or move the recursion into the molecular crate, which is a bigger change than
//! this phase justifies. Its only consumers are AFTDF's `get_nuc` and (Phase 14)
//! GDF's `weighted_ft_ao`, both periodic.
//!
//! # Why this is host code and not a kernel
//!
//! `ft_aopair` is `O(nao² · nG · nimgs · nprim²)`; this is `O(nao · nG · nprim)`
//! with no lattice sum, and its only caller evaluates it on the fake nuclear
//! cell, where `nao = natm` and every shell is a single `s` primitive. A launch
//! would cost more than the arithmetic.

use pyscf_core::Mole;
use pyscf_core::raw_layout::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, PTR_COEFF, PTR_COORD,
    PTR_ENV_START, PTR_EXP,
};
use pyscf_kernels::{cart_powers, cart2sph_l_matrix, common_fac_sp};
use pyscf_pbc_gto::Cell;

use super::mcmurchie::e_coefficients;
use crate::error::PbcDfError;

/// `ft_ao(mol, Gv)` — the planar `(ngrids, nao)` transform, row-major.
///
/// The `l = 0` case is `c·(π/α)^{3/2}·e^{−G²/4α}·e^{−iG·A}`; higher `l` picks up
/// the Hermite polynomial in `(−iG)`.
///
/// # Errors
/// Propagates the cart→sph table lookup for `l > 6`.
pub fn ft_ao_mol(mol: &Mole, gv: &[[f64; 3]]) -> Result<(Vec<f64>, Vec<f64>), PbcDfError> {
    let ngrids = gv.len();
    let pi = std::f64::consts::PI;

    // Cartesian pass first; contract to spherical afterwards, per shell.
    let mut nao_out = 0usize;
    for ib in 0..mol.nbas {
        let l = mol._bas[ib * BAS_SLOTS + ANG_OF] as u32;
        let nctr = mol._bas[ib * BAS_SLOTS + NCTR_OF] as usize;
        nao_out += nctr
            * if mol.cart {
                ncart(l)
            } else {
                2 * l as usize + 1
            };
    }
    let mut re = vec![0.0f64; ngrids * nao_out];
    let mut im = vec![0.0f64; ngrids * nao_out];

    let mut off = 0usize;
    for ib in 0..mol.nbas {
        let row = ib * BAS_SLOTS;
        let l = mol._bas[row + ANG_OF] as u32;
        let nprim = mol._bas[row + NPRIM_OF] as usize;
        let nctr = mol._bas[row + NCTR_OF] as usize;
        let pe = mol._bas[row + PTR_EXP] as usize;
        let pc = mol._bas[row + PTR_COEFF] as usize;
        let atom = mol._bas[row + ATOM_OF] as usize;
        let pcoord = mol._atm[atom * ATM_SLOTS + PTR_COORD] as usize;
        let a = [mol._env[pcoord], mol._env[pcoord + 1], mol._env[pcoord + 2]];
        let nc = ncart(l);
        let powers = cart_powers(l);
        let cfac = common_fac_sp(l);
        let t = if mol.cart {
            None
        } else {
            Some(cart2sph_l_matrix(l).map_err(PbcDfError::from)?)
        };
        let nout = if mol.cart { nc } else { 2 * l as usize + 1 };

        for ictr in 0..nctr {
            for (g, gvec) in gv.iter().enumerate() {
                let g2 = gvec[0] * gvec[0] + gvec[1] * gvec[1] + gvec[2] * gvec[2];
                let th = -(gvec[0] * a[0] + gvec[1] * a[1] + gvec[2] * a[2]);
                let (sn, cs) = th.sin_cos();
                // Cartesian accumulators for this shell at this G.
                let mut cre = vec![0.0f64; nc];
                let mut cim = vec![0.0f64; nc];
                for p in 0..nprim {
                    let alpha = mol._env[pe + p];
                    let coef = mol._env[pc + ictr * nprim + p];
                    let w = coef * cfac * (pi / alpha).powf(1.5) * (-g2 / (4.0 * alpha)).exp();
                    if w == 0.0 {
                        continue;
                    }
                    // Single centre: P = A, so x_PA = x_PB = 0 and K_AB = 1.
                    let ex = e_coefficients(l, 0, alpha, 0.0, 0.0, 1.0);
                    for (ci, &(ix, iy, iz)) in powers.iter().enumerate() {
                        let (mut pr, mut pim) = (0.0f64, 0.0f64);
                        let mut gxp = 1.0f64;
                        for tt in 0..=ix {
                            let et = ex.get(ix, 0, tt);
                            let mut gyp = 1.0f64;
                            for uu in 0..=iy {
                                let eu = ex.get(iy, 0, uu);
                                let etu = et * eu * gxp * gyp;
                                let mut gzp = 1.0f64;
                                for vv in 0..=iz {
                                    let ww = etu * ex.get(iz, 0, vv) * gzp;
                                    match (tt + uu + vv) % 4 {
                                        0 => pr += ww,
                                        1 => pim -= ww,
                                        2 => pr -= ww,
                                        _ => pim += ww,
                                    }
                                    gzp *= gvec[2];
                                }
                                gyp *= gvec[1];
                            }
                            gxp *= gvec[0];
                        }
                        cre[ci] += w * (pr * cs - pim * sn);
                        cim[ci] += w * (pr * sn + pim * cs);
                    }
                }
                // cart → sph (or straight through for a cartesian Mole).
                for m in 0..nout {
                    let (mut ar, mut ai) = (0.0f64, 0.0f64);
                    match &t {
                        None => {
                            ar = cre[m];
                            ai = cim[m];
                        }
                        Some(tm) => {
                            for c in 0..nc {
                                let w = tm[m * nc + c];
                                if w != 0.0 {
                                    ar += w * cre[c];
                                    ai += w * cim[c];
                                }
                            }
                        }
                    }
                    re[g * nao_out + off + m] = ar;
                    im[g * nao_out + off + m] = ai;
                }
            }
            off += nout;
        }
    }
    Ok((re, im))
}

/// `ft_ao(cell, Gv, kpt)` — `pbc/df/ft_ao.py:93-100`.
///
/// There is NO lattice sum here: gamma evaluates at `Gv`, any other k-point at
/// `Gv + kpt`. Its only caller runs it on the fake nuclear cell, whose
/// `rcut = 0.1`, so images never contribute.
///
/// # Errors
/// As [`ft_ao_mol`].
pub fn ft_ao_kpt(
    mol: &Mole,
    gv: &[[f64; 3]],
    kpt: [f64; 3],
) -> Result<(Vec<f64>, Vec<f64>), PbcDfError> {
    if kpt[0].abs() + kpt[1].abs() + kpt[2].abs() < 1e-9 {
        ft_ao_mol(mol, gv)
    } else {
        let shifted: Vec<[f64; 3]> = gv
            .iter()
            .map(|g| [g[0] + kpt[0], g[1] + kpt[1], g[2] + kpt[2]])
            .collect();
        ft_ao_mol(mol, &shifted)
    }
}

#[inline]
fn ncart(l: u32) -> usize {
    ((l as usize + 1) * (l as usize + 2)) / 2
}

/// `_fake_nuc(cell, with_pseudo)` — `aft.py:247-274`.
///
/// A `Mole` of one steep `s` shell per atom, standing in for the nuclear charge
/// density. With a GTH pseudopotential the width comes from the potential
/// (`eta = 0.5/r_loc²`); without one it is a numerical point charge
/// (`eta = 1e16`).
///
/// # Errors
/// Never — the shape is fixed — but the signature matches its siblings.
pub fn fake_nuc(cell: &Cell, with_pseudo: bool) -> Result<Mole, PbcDfError> {
    let natm = cell.mol.natm;
    let mut m = cell.mol.clone();
    m.nbas = natm;
    m.cart = false;

    let mut atm = cell.mol._atm.clone();
    let mut env = vec![0.0f64; PTR_ENV_START];
    // Atom coordinates first, then (eta, norm) per atom — upstream's layout.
    let coords = cell.mol.atom_coords();
    for r in &coords {
        env.extend_from_slice(r);
    }
    for (ia, a) in atm.chunks_mut(ATM_SLOTS).enumerate() {
        a[PTR_COORD] = (PTR_ENV_START + ia * 3) as i32;
    }

    let half_sph_norm = 0.5 / std::f64::consts::PI.sqrt();
    let mut bas = Vec::with_capacity(natm * BAS_SLOTS);
    let mut ptr = PTR_ENV_START + natm * 3;
    for ia in 0..natm {
        let symb = &cell.mol._atom[ia].0;
        let eta = match (with_pseudo, cell.pseudo.as_ref().and_then(|p| p.get(symb))) {
            (true, Some(gth)) => 0.5 / (gth.rloc * gth.rloc),
            _ => 1e16,
        };
        // `norm = half_sph_norm / gaussian_int(2, eta)`.
        let norm = half_sph_norm / gaussian_int(2, eta);
        env.push(eta);
        env.push(norm);
        // [atom, l=0, nprim=1, nctr=1, kappa=0, ptr_exp, ptr_coeff, 0]
        bas.extend_from_slice(&[ia as i32, 0, 1, 1, 0, ptr as i32, (ptr + 1) as i32, 0]);
        ptr += 2;
    }
    m._atm = atm;
    m._bas = bas;
    m._env = env;
    m.nao_nr = natm;
    m.ao_loc_nr = (0..=natm as i32).collect();
    let _ = CHARGE_OF;
    Ok(m)
}

/// `pyscf.gto.gaussian_int(n, alpha)` = `Γ((n+1)/2) / (2·α^((n+1)/2))`.
///
/// Only `n = 2` is needed here, where `(n+1)/2 = 1.5` and `Γ(1.5) = √π/2`.
fn gaussian_int(n: u32, alpha: f64) -> f64 {
    let n1 = (n as f64 + 1.0) * 0.5;
    gamma_half_integer(n1) / (2.0 * alpha.powf(n1))
}

/// `Γ(x)` for `x` a positive half-integer or integer — all this module needs.
fn gamma_half_integer(x: f64) -> f64 {
    // Γ(1/2) = √π; Γ(x+1) = x·Γ(x). Reduce until `v` is 0.5 or 1.0 — the loop
    // bound is `> 1.0`, NOT `> 1.5`: stopping at 1.5 leaves Γ(1.5) = 1 instead
    // of √π/2, which is exactly the factor `_fake_nuc`'s normalisation is off by.
    let mut v = x;
    let mut acc = 1.0f64;
    while v > 1.0 {
        v -= 1.0;
        acc *= v;
    }
    acc * if (v - 0.5).abs() < 1e-12 {
        std::f64::consts::PI.sqrt()
    } else {
        1.0
    }
}
