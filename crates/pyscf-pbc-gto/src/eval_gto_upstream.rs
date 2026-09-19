//! Upstream-order periodic AO values — `libpbc`'s `PBCGTOval_sph_deriv0`
//! (`pyscf/lib/pbc/grid_ao.c`) statement for statement, on the host.
//!
//! # Why a second evaluator
//!
//! [`crate::eval_ao_kpts`] is the production path (device kernels, W-09/A-04
//! screens, K-08..K-10 accumulation). It agrees with upstream to ~1e-12, which
//! is not bit-exact: it screens images per 128-point bounding box where
//! upstream uses the exact block minimum over 56 points, it has no `non0tab`
//! image cap, and it sums images in a different order. Those three move the He
//! `get_nuc` by 2e-13. This module reproduces upstream's arithmetic instead, for
//! the one consumer that needs upstream's bits (`FFTDF.get_nuc`). Plan 10-04's
//! "no new AO evaluator" rule is deliberately set aside here, for that reason
//! only; nothing on the SCF path calls it.
//!
//! # What is reproduced
//!
//! * `make_mask` → `PBCnr_ao_screen`: per (56-point block, shell), the number
//!   of distance-sorted images that can contribute (`non0tab`).
//! * `PBCeval_sph_iter`: images in windows of `IMGBLK = 40`; an image counts
//!   for a block iff `sqrt(min_g |r_g - (R + L)|^2) < rcut[shell]`; the shell
//!   values come from `GTOcontract_exp0` + `GTOshell_eval_grid_cart`; the
//!   window is folded in with `dgemm_('N','T', ...)`, reproduced by
//!   [`pyscf_algebra::openblas_emu::dgemm_nt`].
//! * `expLk = exp(1j * Ls @ kpts.T)`: a left-to-right 3-term dot, then
//!   `(cos, sin)`.
//!
//! # Scope
//!
//! Shells with `l <= 4` (s, p, d, f, g): `l <= 1` is written straight into
//! `pao` by `GTOshell_eval_grid_cart`'s explicit branches; `l = 2, 3` use the
//! explicit branches plus `CINTc2s_ket_sph1` ([`crate::c2s_tables`]); `l = 4`
//! uses the generic `xpows/ypows/zpows` branch plus the explicit `g` transform.
//! `l >= 5` would need libcint's BLAS `a_ket_cart2spheric` and is refused with
//! `Ok(None)` — a documented boundary, not a silent fallback. When
//! `cell.mol.cart` is true the `PBCeval_cart_iter` branch runs instead (no `c2s`
//! step, Cartesian widths). Value evaluation (`deriv = 0`) only.

use crate::c2s_tables::c2s_ket_sph1;
use crate::cell::Cell;
use crate::eval_gto::{EvalAoKptsOutput, estimate_rcut_for_eval};
use pyscf_algebra::CTensor;
use pyscf_algebra::openblas_emu::dgemm_nt;
use pyscf_core::PyscfRsError;
use pyscf_core::raw_layout::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, NCTR_OF, NPRIM_OF, PTR_COEFF, PTR_COORD, PTR_EXP,
};

/// `grid_ao_drv.h` `BLKSIZE`.
pub const BLKSIZE: usize = 56;
/// `grid_ao.c` `IMGBLK`.
pub const IMGBLK: usize = 40;
/// `grid_ao_drv.h` `EXPCUTOFF`, used when `env[PTR_EXPCUTOFF]` is 0.
const EXPCUTOFF: f64 = 50.0;
/// `env[PTR_EXPCUTOFF]` (`mole.py`: `PTR_EXPCUTOFF = 0`).
const PTR_EXPCUTOFF: usize = 0;
/// `non0tab`'s saturating "every image" marker.
const ALL_IMAGES: usize = 255;

/// `CINTcommon_fac_sp(l)` — libcint's literals, digit for digit.
#[allow(clippy::excessive_precision)]
fn common_fac_sp(l: i32) -> f64 {
    match l {
        0 => 0.282094791773878143,
        1 => 0.488602511902919921,
        _ => 1.0,
    }
}

struct Shell {
    atom: usize,
    l: i32,
    nprim: usize,
    nctr: usize,
    exps: Vec<f64>,
    /// `coeff[k*nprim + j]`, libcint layout.
    coeff: Vec<f64>,
    ao0: usize,
}

fn shells(cell: &Cell, cart: bool) -> Vec<Shell> {
    let bas = &cell.mol._bas;
    let env = &cell.mol._env;
    let nbas = bas.len() / BAS_SLOTS;
    let mut ao0 = 0;
    (0..nbas)
        .map(|ib| {
            let row = &bas[ib * BAS_SLOTS..(ib + 1) * BAS_SLOTS];
            let (l, nprim, nctr) = (row[ANG_OF], row[NPRIM_OF] as usize, row[NCTR_OF] as usize);
            let pe = row[PTR_EXP] as usize;
            let pc = row[PTR_COEFF] as usize;
            let sh = Shell {
                atom: row[ATOM_OF] as usize,
                l,
                nprim,
                nctr,
                exps: env[pe..pe + nprim].to_vec(),
                coeff: env[pc..pc + nprim * nctr].to_vec(),
                ao0,
            };
            ao0 += nctr * ao_stride(l, cart);
            sh
        })
        .collect()
}

/// AO width of one contraction: `2l+1` spherical, `(l+1)(l+2)/2` Cartesian
/// (`PBCeval_sph_iter`'s `deg` vs `PBCeval_cart_iter`'s `deg`).
fn ao_stride(l: i32, cart: bool) -> usize {
    if cart {
        ((l + 1) * (l + 2) / 2) as usize
    } else {
        (2 * l + 1) as usize
    }
}

/// `GTOshell_eval_grid_cart` (`pyscf/lib/gto/deriv1.c`) for `l >= 2`, writing
/// the `dcart = (l+1)(l+2)/2` Cartesian components of every contraction into
/// `out[k*dcart*bgrids + c*bgrids + i]`.
///
/// `l = 2, 3` are the explicit branches with C's left-associative order
/// (`exps*gridx*gridx` is `(exps*gridx)*gridx`); larger `l` is the generic
/// `xpows/ypows/zpows` loop (`xpows[lx] = xpows[lx-1]*gridx`, then
/// `((xp*yp)*zp)*exps`), enumerated `lx` descending, `ly` descending — the
/// same order the explicit branches list.
fn eval_cart(l: i32, ectr: &[f64], g2a: &[[f64; 3]], nctr: usize, bgrids: usize, out: &mut [f64]) {
    let dcart = ((l + 1) * (l + 2) / 2) as usize;
    if l == 2 {
        for k in 0..nctr {
            let e = &ectr[k * bgrids..(k + 1) * bgrids];
            let o = k * dcart * bgrids;
            for (i, d) in g2a.iter().enumerate() {
                let (x, y, z) = (d[0], d[1], d[2]);
                let ex = e[i] * x;
                let ey = e[i] * y;
                let ez = e[i] * z;
                out[o + i] = ex * x;
                out[o + bgrids + i] = ex * y;
                out[o + 2 * bgrids + i] = ex * z;
                out[o + 3 * bgrids + i] = ey * y;
                out[o + 4 * bgrids + i] = ey * z;
                out[o + 5 * bgrids + i] = ez * z;
            }
        }
    } else if l == 3 {
        for k in 0..nctr {
            let e = &ectr[k * bgrids..(k + 1) * bgrids];
            let o = k * dcart * bgrids;
            for (i, d) in g2a.iter().enumerate() {
                let (x, y, z) = (d[0], d[1], d[2]);
                let ex = e[i] * x;
                let ey = e[i] * y;
                let ez = e[i] * z;
                let exx = ex * x;
                let exy = ex * y;
                let exz = ex * z;
                let eyy = ey * y;
                let eyz = ey * z;
                let ezz = ez * z;
                out[o + i] = exx * x;
                out[o + bgrids + i] = exx * y;
                out[o + 2 * bgrids + i] = exx * z;
                out[o + 3 * bgrids + i] = exy * y;
                out[o + 4 * bgrids + i] = exy * z;
                out[o + 5 * bgrids + i] = exz * z;
                out[o + 6 * bgrids + i] = eyy * y;
                out[o + 7 * bgrids + i] = eyy * z;
                out[o + 8 * bgrids + i] = eyz * z;
                out[o + 9 * bgrids + i] = ezz * z;
            }
        }
    } else {
        let lu = l as usize;
        for k in 0..nctr {
            let e = &ectr[k * bgrids..(k + 1) * bgrids];
            let o = k * dcart * bgrids;
            for (i, d) in g2a.iter().enumerate() {
                let mut xp = vec![1.0_f64; lu + 1];
                let mut yp = vec![1.0_f64; lu + 1];
                let mut zp = vec![1.0_f64; lu + 1];
                for lx in 1..=lu {
                    xp[lx] = xp[lx - 1] * d[0];
                    yp[lx] = yp[lx - 1] * d[1];
                    zp[lx] = zp[lx - 1] * d[2];
                }
                let mut c = 0;
                for lx in (0..=lu).rev() {
                    for ly in (0..=lu - lx).rev() {
                        let lz = lu - lx - ly;
                        out[o + c * bgrids + i] = ((xp[lx] * yp[ly]) * zp[lz]) * e[i];
                        c += 1;
                    }
                }
            }
        }
    }
}

fn atom_coord(cell: &Cell, ia: usize) -> [f64; 3] {
    let p = cell.mol._atm[ia * ATM_SLOTS + PTR_COORD] as usize;
    let env = &cell.mol._env;
    [env[p], env[p + 1], env[p + 2]]
}

/// `PBCnr_ao_screen` for one block: how many of the distance-sorted images can
/// contribute to shell `sh` anywhere in the block (`0` = none).
fn non0_images(
    block: &[[f64; 3]],
    ls: &[[f64; 3]],
    ratm: [f64; 3],
    sh: &Shell,
    expcutoff: f64,
) -> usize {
    let logcoeff: Vec<f64> = (0..sh.nprim)
        .map(|j| {
            let mut maxc = 0.0_f64;
            for i in 0..sh.nctr {
                maxc = maxc.max(sh.coeff[i * sh.nprim + j].abs());
            }
            maxc.ln()
        })
        .collect();
    for m in (0..ls.len()).rev() {
        let rl = [ratm[0] + ls[m][0], ratm[1] + ls[m][1], ratm[2] + ls[m][2]];
        for g in block {
            let dr = [g[0] - rl[0], g[1] - rl[1], g[2] - rl[2]];
            let rr = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
            for j in 0..sh.nprim {
                let arr = sh.exps[j] * rr;
                if arr - logcoeff[j] < expcutoff {
                    return ALL_IMAGES.min(m + 1);
                }
            }
        }
    }
    0
}

/// `eval_ao_kpts(cell, coords, kpts)` (deriv 0, spherical or Cartesian) with
/// upstream's rounding.
///
/// The image list is `eval_gto`'s own (`eval_gto.py:136-138`): the EVAL
/// variant at `max(rcut)`, stably sorted by length — the same list `make_mask`
/// builds, so one list serves both the screen and the sum.
///
/// Returns `Ok(None)` when the basis has a shell with `l >= 5` (libcint's BLAS
/// `a_ket_cart2spheric` is not ported) — a documented boundary.
///
/// # Errors
/// Propagates [`estimate_rcut_for_eval`] and the lattice-image construction.
pub fn eval_ao_kpts_upstream(
    cell: &Cell,
    coords: &[[f64; 3]],
    kpts: &[[f64; 3]],
) -> Result<Option<EvalAoKptsOutput>, PyscfRsError> {
    let cart = cell.mol.cart;
    let shells = shells(cell, cart);
    if shells.iter().any(|s| s.l > 4) {
        return Ok(None);
    }
    let nao = cell.mol.nao_nr;
    debug_assert_eq!(
        shells
            .iter()
            .map(|s| s.nctr * ao_stride(s.l, cart))
            .sum::<usize>(),
        nao,
        "shell strides must tile nao_nr (cart={cart})"
    );
    let rcut = estimate_rcut_for_eval(cell, 0)?;
    let rmax = rcut.iter().copied().fold(0.0_f64, f64::max);
    let mut ls = crate::lattice::get_lattice_ls_eval(cell, rmax)?;
    ls.sort_by(|a, b| pyscf_pbc_tools::mat3::norm3(a).total_cmp(&pyscf_pbc_tools::mat3::norm3(b)));
    let ls = &ls[..];
    let env = &cell.mol._env;
    let expcutoff = if env[PTR_EXPCUTOFF] == 0.0 {
        EXPCUTOFF
    } else {
        env[PTR_EXPCUTOFF]
    };
    let ngrids = coords.len();
    let nkpts = kpts.len();
    let nkpts2 = 2 * nkpts;
    let nimgs = ls.len();

    // expLk[img][k] = exp(1j * (L . k)), stored as the interleaved (re, im)
    // doubles `dgemm_` reads: expk[img*nkpts2 + 2k (+1)].
    let mut expk = vec![0.0; nimgs * nkpts2];
    for (img, l) in ls.iter().enumerate() {
        for (k, kp) in kpts.iter().enumerate() {
            let theta = l[0] * kp[0] + l[1] * kp[1] + l[2] * kp[2];
            expk[img * nkpts2 + 2 * k] = theta.cos();
            expk[img * nkpts2 + 2 * k + 1] = theta.sin();
        }
    }

    let mut out_re = vec![vec![0.0; nao * ngrids]; nkpts];
    let mut out_im = vec![vec![0.0; nao * ngrids]; nkpts];
    let atom_r: Vec<[f64; 3]> = (0..cell.mol.natm).map(|ia| atom_coord(cell, ia)).collect();

    for (ib, block) in coords.chunks(BLKSIZE).enumerate() {
        let ip = ib * BLKSIZE;
        let bgrids = block.len();
        for (ish, sh) in shells.iter().enumerate() {
            let ratm = atom_r[sh.atom];
            let n0 = non0_images(block, ls, ratm, sh, expcutoff);
            let bas_nimgs = if n0 == ALL_IMAGES {
                nimgs
            } else {
                n0.min(nimgs)
            };
            let deg = ao_stride(sh.l, cart);
            let dcart = ((sh.l + 1) * (sh.l + 2) / 2) as usize;
            let nfunc = sh.nctr * deg;
            let dimc = nfunc * bgrids;
            let fac = common_fac_sp(sh.l);

            // grid2atm and its block minimum distance, per image.
            let mut grid2atm = vec![[0.0_f64; 3]; bas_nimgs * bgrids];
            let mut min_dist = vec![0.0_f64; bas_nimgs];
            for m in 0..bas_nimgs {
                let rl = [ratm[0] + ls[m][0], ratm[1] + ls[m][1], ratm[2] + ls[m][2]];
                let mut dist_min = 1e9_f64;
                for (ig, g) in block.iter().enumerate() {
                    let d = [g[0] - rl[0], g[1] - rl[1], g[2] - rl[2]];
                    grid2atm[m * bgrids + ig] = d;
                    let dist = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                    dist_min = dist.min(dist_min);
                }
                min_dist[m] = dist_min.sqrt();
            }

            let mut aobufk = vec![0.0_f64; nkpts2 * dimc];
            let mut aobuf: Vec<f64> = Vec::with_capacity(IMGBLK * dimc);
            let mut phase: Vec<f64> = Vec::with_capacity(IMGBLK * nkpts2);
            let mut i0 = 0;
            while i0 < bas_nimgs {
                let count_max = IMGBLK.min(bas_nimgs - i0);
                aobuf.clear();
                phase.clear();
                for il in i0..i0 + count_max {
                    // `min_grid2atm[iL] < rcut[bas_id]` (distances are finite).
                    if min_dist[il] >= rcut[ish] {
                        continue;
                    }
                    let g2a = &grid2atm[il * bgrids..(il + 1) * bgrids];
                    // GTOcontract_exp0: ectr[k][i] = sum_j exp(-a_j rr) fac c_kj.
                    let mut ectr = vec![0.0_f64; sh.nctr * bgrids];
                    for j in 0..sh.nprim {
                        for (i, d) in g2a.iter().enumerate() {
                            let rr = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                            let arr = sh.exps[j] * rr;
                            let eprim = (-arr).exp() * fac;
                            for k in 0..sh.nctr {
                                ectr[k * bgrids + i] += eprim * sh.coeff[k * sh.nprim + j];
                            }
                        }
                    }
                    // `GTOshell_eval_grid_cart`: `l <= 1` writes straight into
                    // `pao`; `l >= 2` writes Cartesians into a scratch buffer
                    // and `CINTc2s_ket_sph1` transforms each `(comp,
                    // contraction)` (`pao += deg*bgrids`, `pcart +=
                    // dcart*bgrids`). `PBCeval_cart_iter` skips the transform.
                    let base = aobuf.len();
                    aobuf.resize(base + dimc, 0.0);
                    let pao = &mut aobuf[base..];
                    if sh.l <= 1 {
                        for k in 0..sh.nctr {
                            let e = &ectr[k * bgrids..(k + 1) * bgrids];
                            if sh.l == 0 {
                                pao[k * bgrids..(k + 1) * bgrids].copy_from_slice(e);
                            } else {
                                let o = k * 3 * bgrids;
                                for (i, d) in g2a.iter().enumerate() {
                                    pao[o + i] = d[0] * e[i];
                                    pao[o + bgrids + i] = d[1] * e[i];
                                    pao[o + 2 * bgrids + i] = d[2] * e[i];
                                }
                            }
                        }
                    } else if cart {
                        eval_cart(sh.l, &ectr, g2a, sh.nctr, bgrids, pao);
                    } else {
                        let mut cart_gto = vec![0.0_f64; sh.nctr * dcart * bgrids];
                        eval_cart(sh.l, &ectr, g2a, sh.nctr, bgrids, &mut cart_gto);
                        for k in 0..sh.nctr {
                            let cblock = &cart_gto[k * dcart * bgrids..(k + 1) * dcart * bgrids];
                            let sblock = &mut pao[k * deg * bgrids..(k + 1) * deg * bgrids];
                            c2s_ket_sph1(sh.l, sblock, cblock, bgrids, bgrids);
                        }
                    }
                    phase.extend_from_slice(&expk[il * nkpts2..(il + 1) * nkpts2]);
                }
                let count = aobuf.len() / dimc.max(1);
                if count > 0 {
                    // dgemm_(N, T, dimc, nkpts2, count, 1, aobuf, dimc,
                    //        pexpLk, nkpts2, 1, aobufk, dimc)
                    dgemm_nt(dimc, nkpts2, count, &aobuf, &phase, &mut aobufk);
                }
                i0 += IMGBLK;
            }

            // _copy: aobufk[(2k)*dimc + f*bgrids + i] -> out[k][ao0+f][ip+i]
            for k in 0..nkpts {
                let (re, im) = (&aobufk[2 * k * dimc..], &aobufk[(2 * k + 1) * dimc..]);
                for f in 0..nfunc {
                    let row = (sh.ao0 + f) * ngrids + ip;
                    out_re[k][row..row + bgrids].copy_from_slice(&re[f * bgrids..(f + 1) * bgrids]);
                    out_im[k][row..row + bgrids].copy_from_slice(&im[f * bgrids..(f + 1) * bgrids]);
                }
            }
        }
    }

    // eval_gto.py:169-173 — a gamma k-point keeps only the real part.
    let gamma: Vec<bool> = kpts
        .iter()
        .map(|k| k[0].abs() + k[1].abs() + k[2].abs() < 1e-9)
        .collect();
    let kaos = out_re
        .into_iter()
        .zip(out_im)
        .zip(&gamma)
        .map(|((re, im), &g)| {
            let im = if g { vec![0.0; im.len()] } else { im };
            CTensor::from_planes(re, im)
        })
        .collect();
    Ok(Some(EvalAoKptsOutput {
        kaos,
        ngrids,
        nao,
        comp: 1,
        gamma,
    }))
}
