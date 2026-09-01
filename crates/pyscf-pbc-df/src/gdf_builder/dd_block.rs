//! `_outcore_dd_block` — `pyscf/pbc/df/rsdf_builder.py:535-698`. Plan 17-10
//! Task 3.
//!
//! Re-routes the smooth-smooth block of `(ij|L)` out of the real-space
//! lattice sum (where it converges slowly — both AO factors are diffuse) and
//! into an FFT, exactly as [`crate::ft_ao::rs_cell::RsCell::smooth_basis_cell`]
//! carves out. D-PBC-23 measured this as a re-route, not a screen: upstream's
//! `exclude_dd_block = true` is the MORE accurate route.
//!
//! ```text
//! Vaux[a, r] = ifft( ft_ao(auxcell, G, -kq) * coulG(-kq) )[r] * exp(-i kq.r)
//! j3c_dd[mu, nu, a] = SUM_r conj(ao_mu(ki, r)) * ao_nu(kj, r) * Vaux[a, r]
//! ```
//! against the PLAIN (uncompensated) auxiliary cell — the model-charge
//! subtraction that makes the COMPACT blocks converge in real space is not
//! needed here because the FFT already converges the smooth-smooth block
//! directly (`kq = kj - ki`, matching every other k-difference convention in
//! this crate, e.g. [`crate::gdf_builder::j3c::weighted_ft_ao`]).
//!
//! # Performance note
//!
//! This is a direct `O(nkptij . nao_d^2 . naux . ngrids)` quadrature — no BLAS,
//! no device kernel. `nao_d` (the smooth-cell AO count) is a small fraction of
//! `nao` on every system this plan gates, so it is fast there; a production
//! workload with a larger smooth block would want the batched `PBC_kzdot_CNN`
//! contraction upstream uses. Left as a follow-on (see `17-10-SUMMARY.md`).

use pyscf_algebra::CTensor;
use pyscf_pbc_gto::Cell;
use pyscf_pbc_gto::coulg::{CoulGArgs, get_coulg};
use pyscf_pbc_gto::eval_gto::eval_ao_kpts;
use pyscf_pbc_gto::gv::{get_gv_weights, get_uniform_grids};
use pyscf_pbc_tools::fft::ifft;

use crate::error::PbcDfError;
use crate::ft_ao::ft_ao_kpt;
use crate::incore::Aosym;
use crate::incore::int3c::KptPair;

fn pair_row(aosym: Aosym, nao: usize, mu: usize, nu: usize) -> Option<usize> {
    match aosym {
        Aosym::S1 => Some(mu * nao + nu),
        Aosym::S2 => {
            if mu >= nu {
                Some(mu * (mu + 1) / 2 + nu)
            } else {
                None
            }
        }
    }
}

/// `_outcore_dd_block` — the smooth-smooth `(ij|L)` block via FFT, against the
/// PLAIN `auxcell` (upstream's `self.auxcell`, not the fused/compensated
/// cell). One [`CTensor`] per `kptij` pair, shape `(nao_pair(smooth), naux)`
/// row-major — the SAME layout [`crate::gdf_builder::j3c::outcore_auxe2`]
/// returns, so the caller can scatter-add it directly.
///
/// Returns an empty `Vec` when `smooth` carries no AOs (nothing to route) —
/// the caller's contract is "no correction", not "zero-filled correction",
/// which is what makes He-fcc's `exclude_dd_block` cost EXACTLY ZERO by
/// construction (D-PBC-23).
///
/// # Errors
/// Propagates the FFT mesh / grid construction, the Coulomb kernel and the AO
/// evaluation.
pub fn fft_dd_block(
    smooth: &Cell,
    auxcell: &Cell,
    aosym: Aosym,
    kptij: &[KptPair],
    omega: Option<f64>,
) -> Result<Vec<CTensor>, PbcDfError> {
    let nao = smooth.mol.nao_nr;
    let naux = auxcell.mol.nao_nr;
    if nao == 0 || naux == 0 {
        return Ok(Vec::new());
    }
    let nao_pair = aosym.nao_pair(nao);

    let mesh = smooth.mesh;
    let gvw = get_gv_weights(smooth, Some(mesh)).map_err(PbcDfError::from)?;
    let gv = &gvw.gv;
    let ngrids = gv.len();
    let coords = get_uniform_grids(smooth, Some(mesh), true).map_err(PbcDfError::from)?;

    // De-duplicate the k-points AO evaluation actually needs.
    let mut kpt_list: Vec<[f64; 3]> = Vec::new();
    let mut kpt_index = |k: [f64; 3]| -> usize {
        const TOL: f64 = 1e-12;
        if let Some(i) = kpt_list
            .iter()
            .position(|x| (x[0] - k[0]).abs() < TOL && (x[1] - k[1]).abs() < TOL && (x[2] - k[2]).abs() < TOL)
        {
            i
        } else {
            kpt_list.push(k);
            kpt_list.len() - 1
        }
    };
    let ki_idx: Vec<usize> = kptij.iter().map(|p| kpt_index(p.ki)).collect();
    let kj_idx: Vec<usize> = kptij.iter().map(|p| kpt_index(p.kj)).collect();

    let ao_out =
        eval_ao_kpts(smooth, "GTOval_sph", &coords, &kpt_list).map_err(PbcDfError::from)?;
    debug_assert_eq!(ao_out.nao, nao);
    debug_assert_eq!(ao_out.ngrids, ngrids);

    let mut out = Vec::with_capacity(kptij.len());
    for (idx, pair) in kptij.iter().enumerate() {
        let kq = [
            pair.kj[0] - pair.ki[0],
            pair.kj[1] - pair.ki[1],
            pair.kj[2] - pair.ki[2],
        ];
        let neg_kq = [-kq[0], -kq[1], -kq[2]];

        // `auxG = ft_ao(auxcell, Gv, -kq).T`, `(ngrids, naux)` row-major —
        // `ft_ao_kpt`'s own planar convention.
        let (mut auxg_re, mut auxg_im) = ft_ao_kpt(&auxcell.mol, gv, neg_kq)?;

        // `coulG(-kq)` — full-range, no `exxdiv` (the model-charge divergence
        // this term exists to bypass does not apply to a plain aux cell).
        let coulg = get_coulg(
            smooth,
            CoulGArgs {
                k: neg_kq,
                exxdiv: None,
                kpts: None,
                mesh: Some(mesh),
                gv: Some(gv),
                wrap_around: true,
                omega,
            },
        )
        .map_err(PbcDfError::from)?;
        for g in 0..ngrids {
            let w = coulg[g];
            for a in 0..naux {
                auxg_re[g * naux + a] *= w;
                auxg_im[g * naux + a] *= w;
            }
        }

        // `Vaux[a, :] = ifft(auxG[:, a])`, one column at a time.
        let mut vaux_re = vec![0.0_f64; naux * ngrids];
        let mut vaux_im = vec![0.0_f64; naux * ngrids];
        for a in 0..naux {
            let col_re: Vec<f64> = (0..ngrids).map(|g| auxg_re[g * naux + a]).collect();
            let col_im: Vec<f64> = (0..ngrids).map(|g| auxg_im[g * naux + a]).collect();
            let ct = ifft(&CTensor { re: col_re, im: col_im }, mesh).map_err(PbcDfError::from)?;
            for g in 0..ngrids {
                vaux_re[a * ngrids + g] = ct.re[g];
                vaux_im[a * ngrids + g] = ct.im[g];
            }
        }

        // `Vaux *= exp(-i coords . kq)`.
        for (g, r) in coords.iter().enumerate() {
            let ph = -(r[0] * kq[0] + r[1] * kq[1] + r[2] * kq[2]);
            let (c, s) = (ph.cos(), ph.sin());
            for a in 0..naux {
                let re = vaux_re[a * ngrids + g];
                let im = vaux_im[a * ngrids + g];
                vaux_re[a * ngrids + g] = re * c - im * s;
                vaux_im[a * ngrids + g] = re * s + im * c;
            }
        }

        let ao_i = &ao_out.kaos[ki_idx[idx]];
        let ao_j = &ao_out.kaos[kj_idx[idx]];

        let mut re = vec![0.0_f64; nao_pair * naux];
        let mut im = vec![0.0_f64; nao_pair * naux];
        for mu in 0..nao {
            for nu in 0..nao {
                let Some(row) = pair_row(aosym, nao, mu, nu) else {
                    continue;
                };
                let (mb, nb) = (mu * ngrids, nu * ngrids);
                for a in 0..naux {
                    let ab = a * ngrids;
                    let mut sr = 0.0_f64;
                    let mut si = 0.0_f64;
                    for g in 0..ngrids {
                        // conj(ao_i[mu,g]) * ao_j[nu,g] * Vaux[a,g]
                        let (pr, pi) = (ao_i.re[mb + g], -ao_i.im[mb + g]);
                        let (qr, qi) = (ao_j.re[nb + g], ao_j.im[nb + g]);
                        let apr = pr * qr - pi * qi;
                        let api = pr * qi + pi * qr;
                        let (vr, vi) = (vaux_re[ab + g], vaux_im[ab + g]);
                        sr += apr * vr - api * vi;
                        si += apr * vi + api * vr;
                    }
                    re[row * naux + a] = sr;
                    im[row * naux + a] = si;
                }
            }
        }
        out.push(CTensor { re, im });
    }
    Ok(out)
}
