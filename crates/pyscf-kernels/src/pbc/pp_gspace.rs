//! K-PP1..3 — the reciprocal-space non-local pseudopotential
//! (`pyscf/pbc/df/fft.py:114-176`) evaluated in G-blocks on the device.
//!
//! K-PP3 (this task) accumulates `B[row, q] += Σ_{g < nb} S[row, g] · A[q, g]`
//! (complex). One lane owns one `(row, q)` element and walks `g` upward,
//! starting from the value already in `B` — so splitting `g` into blocks
//! changes no bit (the C1 gate).
//!
//! All buffers are planar (`re` / `im` separate, never interleaved), row-major.
//!
//! Generic over the device float (`F: Float`, AGENTS.md §3 / RULE 5).

use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use cubecl::server::Handle;
use pyscf_algebra::dispatch_backend;
use pyscf_algebra::launch::{launch_1d, upload};
use pyscf_algebra::{AlgebraClient, AlgebraError};

use crate::scalar::DeviceScalar;

/// K-PP3 — `B[row, q] += Σ_{g < nb} S[row, g] · A[q, g]`. One lane owns one
/// `(row, q)` element and walks `g` upward, starting from the value already
/// in `B`. One accumulator per element and a fixed `g` order make the result
/// independent of how `g` was split into blocks, bit for bit.
///
/// Generic over the device float (`F: Float`, AGENTS.md §3).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn pp_fold_kernel<F: Float>(
    s_re: &Array<F>,
    s_im: &Array<F>,
    a_re: &Array<F>,
    a_im: &Array<F>,
    b_re: &mut Array<F>,
    b_im: &mut Array<F>,
    nrow: usize,
    nao: usize,
    nbs: usize,
    nb: usize,
) {
    let i = ABSOLUTE_POS;
    if i < nrow * nao {
        let row = i / nao;
        let q = i % nao;
        let sb = row * nbs;
        let ab = q * nbs;
        let mut sr = b_re[i];
        let mut si = b_im[i];
        for g in 0..nb {
            let xr = s_re[sb + g];
            let xi = s_im[sb + g];
            let yr = a_re[ab + g];
            let yi = a_im[ab + g];
            sr += xr * yr - xi * yi;
            si += xr * yi + xi * yr;
        }
        b_re[i] = sr;
        b_im[i] = si;
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_pp_fold_on_handles<R: Runtime, F: DeviceScalar>(
    client: &ComputeClient<R>,
    s_re: &Handle,
    s_im: &Handle,
    a_re: &Handle,
    a_im: &Handle,
    b_re: &Handle,
    b_im: &Handle,
    nrow: usize,
    nao: usize,
    nbs: usize,
    nb: usize,
) {
    let (count, dim) = launch_1d(client, nrow * nao, nb);
    unsafe {
        pp_fold_kernel::launch_unchecked::<F, R>(
            client,
            count,
            dim,
            // SAFETY: the six element counts are the host slices' lengths
            // (validated in `pp_fold`); every lane guards `i < nrow*nao` and
            // reads `g < nb <= nbs` inside its row/AO stride.
            ArrayArg::from_raw_parts(s_re.clone(), nrow * nbs),
            ArrayArg::from_raw_parts(s_im.clone(), nrow * nbs),
            ArrayArg::from_raw_parts(a_re.clone(), nao * nbs),
            ArrayArg::from_raw_parts(a_im.clone(), nao * nbs),
            ArrayArg::from_raw_parts(b_re.clone(), nrow * nao),
            ArrayArg::from_raw_parts(b_im.clone(), nrow * nao),
            nrow,
            nao,
            nbs,
            nb,
        );
    }
}

fn launch_pp_fold<R: Runtime, F: DeviceScalar>(
    client: &ComputeClient<R>,
    s_re: &[F],
    s_im: &[F],
    a_re: &[F],
    a_im: &[F],
    b_re: &[F],
    b_im: &[F],
    nrow: usize,
    nao: usize,
    nbs: usize,
    nb: usize,
) -> (Vec<F>, Vec<F>) {
    let s_re_h = upload::<R, F>(client, s_re);
    let s_im_h = upload::<R, F>(client, s_im);
    let a_re_h = upload::<R, F>(client, a_re);
    let a_im_h = upload::<R, F>(client, a_im);
    let b_re_h = upload::<R, F>(client, b_re);
    let b_im_h = upload::<R, F>(client, b_im);
    launch_pp_fold_on_handles::<R, F>(
        client, &s_re_h, &s_im_h, &a_re_h, &a_im_h, &b_re_h, &b_im_h, nrow, nao, nbs, nb,
    );
    let bytes = client.read(vec![b_re_h, b_im_h]);
    (
        bytemuck::cast_slice::<u8, F>(&bytes[0]).to_vec(),
        bytemuck::cast_slice::<u8, F>(&bytes[1]).to_vec(),
    )
}

/// K-PP3 on host slices: returns the updated `(b_re, b_im)`.
///
/// # Errors
/// [`AlgebraError::ShapeMismatch`] when a slice length does not match
/// `nrow`, `nao`, `nbs`, or when `nb > nbs`.
#[allow(clippy::too_many_arguments)]
pub fn pp_fold(
    client: &AlgebraClient,
    s_re: &[f64],
    s_im: &[f64],
    a_re: &[f64],
    a_im: &[f64],
    b_re: &[f64],
    b_im: &[f64],
    nrow: usize,
    nao: usize,
    nbs: usize,
    nb: usize,
) -> Result<(Vec<f64>, Vec<f64>), AlgebraError> {
    if nb > nbs {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("nb {nb} <= nbs {nbs}"),
            actual: format!("nb {nb} > nbs {nbs}"),
        });
    }
    let expect = |name: &str, got: usize, want: usize| {
        if got != want {
            return Err(AlgebraError::ShapeMismatch {
                expected: format!("{name} len {want}"),
                actual: format!("{got}"),
            });
        }
        Ok(())
    };
    expect("s_re", s_re.len(), nrow * nbs)?;
    expect("s_im", s_im.len(), nrow * nbs)?;
    expect("a_re", a_re.len(), nao * nbs)?;
    expect("a_im", a_im.len(), nao * nbs)?;
    expect("b_re", b_re.len(), nrow * nao)?;
    expect("b_im", b_im.len(), nrow * nao)?;
    let out = dispatch_backend!(
        client,
        c,
        Rt,
        launch_pp_fold::<Rt, f64>(c, s_re, s_im, a_re, a_im, b_re, b_im, nrow, nao, nbs, nb)
    );
    Ok(out)
}

/// Host tables for K-PP2. Built by `pyscf_pbc_df::pp_gspace`.
#[derive(Debug, Clone, Default)]
pub struct PpProjTables {
    pub row_r: Vec<f64>,
    pub row_par: Vec<f64>,
    pub row_term0: Vec<u32>,
    pub row_nterm: Vec<u32>,
    pub term_c: Vec<f64>,
    pub term_pow: Vec<u32>,
}

/// K-PP2 — one lane per `(row, g)`; index `row * nbs + g`.
///
/// With row `p`, block point `g` (global point `g0 + g`), mesh G-vector `Gv`,
/// `Gk = Gv + kpt`, `q2 = |Gk|²:
///
/// ```text
/// poly = Σ_{t in terms(p)} term_c[t] · Gk_x^ix · Gk_y^iy · Gk_z^iz
/// val  = poly · coef(p) · exp(−alpha(p)·q2) · (c0 + c1·x2 + c2·x2²)
/// S    = val · (cos θ + i·sin θ),   θ = +(Gv · R_atom(p))
/// ```
///
/// Concrete `f64`: it calls `cube_math::double` for bit-exact `exp`/`sincos`
/// (the documented exception — see `ft_aopair.rs:27-37`).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn pp_proj_kernel(
    gv: &Array<f64>,
    kpt: &Array<f64>,
    row_r: &Array<f64>,
    row_par: &Array<f64>,
    row_term0: &Array<u32>,
    row_nterm: &Array<u32>,
    term_c: &Array<f64>,
    term_pow: &Array<u32>,
    s_re: &mut Array<f64>,
    s_im: &mut Array<f64>,
    nrow: usize,
    nbs: usize,
    g0: usize,
    nb: usize,
) {
    let i = ABSOLUTE_POS;
    if i < nrow * nbs {
        let p = i / nbs;
        let g = i % nbs;
        if g < nb {
            let o = (g0 + g) * 3;
            let gx = gv[o];
            let gy = gv[o + 1];
            let gz = gv[o + 2];
            let qx = gx + kpt[0];
            let qy = gy + kpt[1];
            let qz = gz + kpt[2];
            let q2 = qx * qx + qy * qy + qz * qz;

            let alpha = row_par[p * 5];
            let coef = row_par[p * 5 + 1];
            let x2 = q2 * 2.0 * alpha;
            let ql = row_par[p * 5 + 2] + row_par[p * 5 + 3] * x2 + row_par[p * 5 + 4] * x2 * x2;

            let t0 = row_term0[p] as usize;
            let nt = row_nterm[p] as usize;
            let mut poly = 0.0;
            for t in t0..(t0 + nt) {
                let ix = term_pow[t * 3] as usize;
                let iy = term_pow[t * 3 + 1] as usize;
                let iz = term_pow[t * 3 + 2] as usize;
                let mut w = term_c[t];
                if ix == 1 {
                    w *= qx;
                }
                if ix == 2 {
                    w *= qx * qx;
                }
                if ix == 3 {
                    w *= qx * qx * qx;
                }
                if iy == 1 {
                    w *= qy;
                }
                if iy == 2 {
                    w *= qy * qy;
                }
                if iy == 3 {
                    w *= qy * qy * qy;
                }
                if iz == 1 {
                    w *= qz;
                }
                if iz == 2 {
                    w *= qz * qz;
                }
                if iz == 3 {
                    w *= qz * qz * qz;
                }
                poly += w;
            }

            let rad =
                coef * cube_math::double::exp::exp(0.0 - alpha * q2, cube_math::MathConfig::EXACT);
            let val = poly * rad * ql;
            let theta = gx * row_r[p * 3] + gy * row_r[p * 3 + 1] + gz * row_r[p * 3 + 2];
            let (sn, cs) = cube_math::double::trig::sincos(theta, cube_math::MathConfig::EXACT);
            s_re[i] = val * cs;
            s_im[i] = val * sn;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_pp_proj_on_handles<R: Runtime>(
    client: &ComputeClient<R>,
    gv: &Handle,
    kpt: &Handle,
    tables: &[Handle; 6],
    s_re: &Handle,
    s_im: &Handle,
    t: &PpProjTables,
    ngrids: usize,
    nrow: usize,
    nbs: usize,
    g0: usize,
    nb: usize,
) {
    let (count, dim) = launch_1d(client, nrow * nbs, 60);
    unsafe {
        pp_proj_kernel::launch_unchecked::<R>(
            client,
            count,
            dim,
            // SAFETY: every element count below is the uploaded host Vec's
            // length (validated in `pp_proj_block`); lanes guard
            // `i < nrow*nbs` and `g < nb`, and every term range was checked
            // against `term_c.len()` with powers ≤ 3.
            ArrayArg::from_raw_parts(gv.clone(), ngrids * 3),
            ArrayArg::from_raw_parts(kpt.clone(), 3),
            ArrayArg::from_raw_parts(tables[0].clone(), nrow * 3),
            ArrayArg::from_raw_parts(tables[1].clone(), nrow * 5),
            ArrayArg::from_raw_parts(tables[2].clone(), nrow),
            ArrayArg::from_raw_parts(tables[3].clone(), nrow),
            ArrayArg::from_raw_parts(tables[4].clone(), t.term_c.len()),
            ArrayArg::from_raw_parts(tables[5].clone(), t.term_c.len() * 3),
            ArrayArg::from_raw_parts(s_re.clone(), nrow * nbs),
            ArrayArg::from_raw_parts(s_im.clone(), nrow * nbs),
            nrow,
            nbs,
            g0,
            nb,
        );
    }
}

/// `pyscf_algebra::launch::upload` is bounded by `DeviceScalar`, which the
/// index tables' `u32` is not. Same one-copy staging as `ft_aopair.rs:262-266`.
fn upload_u32<R: Runtime>(client: &ComputeClient<R>, data: &[u32]) -> Handle {
    client.create_from_slice(bytemuck::cast_slice(data))
}

fn launch_pp_proj<R: Runtime>(
    client: &ComputeClient<R>,
    t: &PpProjTables,
    gv: &[f64],
    kpt: [f64; 3],
    g0: usize,
    nb: usize,
    nrow: usize,
    nbs: usize,
) -> (Vec<f64>, Vec<f64>) {
    let gv_h = upload::<R, f64>(client, gv);
    let kpt_h = upload::<R, f64>(client, &[kpt[0], kpt[1], kpt[2]]);
    let tables = [
        upload::<R, f64>(client, &t.row_r),
        upload::<R, f64>(client, &t.row_par),
        upload_u32::<R>(client, &t.row_term0),
        upload_u32::<R>(client, &t.row_nterm),
        upload::<R, f64>(client, &t.term_c),
        upload_u32::<R>(client, &t.term_pow),
    ];
    let s_re_h = client.empty(nrow * nbs * core::mem::size_of::<f64>());
    let s_im_h = client.empty(nrow * nbs * core::mem::size_of::<f64>());
    launch_pp_proj_on_handles::<R>(
        client,
        &gv_h,
        &kpt_h,
        &tables,
        &s_re_h,
        &s_im_h,
        t,
        gv.len() / 3,
        nrow,
        nbs,
        g0,
        nb,
    );
    let bytes = client.read(vec![s_re_h, s_im_h]);
    (
        bytemuck::cast_slice::<u8, f64>(&bytes[0]).to_vec(),
        bytemuck::cast_slice::<u8, f64>(&bytes[1]).to_vec(),
    )
}

/// K-PP2 on host slices for one block: `(s_re, s_im)`, each `(nrow, nb)`
/// row-major (the test passes `nbs = nb`).
///
/// # Errors
/// [`AlgebraError::ShapeMismatch`] when a table or slice length is
/// inconsistent (`gv` not a multiple of 3, `g0 + nb` past the grid, table
/// lengths, term ranges, powers above 3).
pub fn pp_proj_block(
    client: &AlgebraClient,
    t: &PpProjTables,
    gv: &[f64],
    kpt: [f64; 3],
    g0: usize,
    nb: usize,
) -> Result<(Vec<f64>, Vec<f64>), AlgebraError> {
    if gv.len() % 3 != 0 {
        return Err(AlgebraError::ShapeMismatch {
            expected: "gv len a multiple of 3".to_string(),
            actual: format!("{} elements", gv.len()),
        });
    }
    if t.row_r.len() % 3 != 0 {
        return Err(AlgebraError::ShapeMismatch {
            expected: "row_r len a multiple of 3".to_string(),
            actual: format!("{} elements", t.row_r.len()),
        });
    }
    let nrow = t.row_r.len() / 3;
    let ngrids = gv.len() / 3;
    let expect = |name: &str, got: usize, want: usize| {
        if got != want {
            return Err(AlgebraError::ShapeMismatch {
                expected: format!("{name} len {want}"),
                actual: format!("{got}"),
            });
        }
        Ok(())
    };
    expect("row_par", t.row_par.len(), 5 * nrow)?;
    expect("row_term0", t.row_term0.len(), nrow)?;
    expect("row_nterm", t.row_nterm.len(), nrow)?;
    expect("term_pow", t.term_pow.len(), 3 * t.term_c.len())?;
    if !matches!(g0.checked_add(nb), Some(end) if end <= ngrids) {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("g0 {g0} + nb {nb} <= ngrids {ngrids}"),
            actual: format!("g0 {g0} + nb {nb}"),
        });
    }
    for p in 0..nrow {
        let end = t.row_term0[p] as usize + t.row_nterm[p] as usize;
        if end > t.term_c.len() {
            return Err(AlgebraError::ShapeMismatch {
                expected: format!("row {p} term range within {} entries", t.term_c.len()),
                actual: format!("range ends at {end}"),
            });
        }
    }
    for (ti, w) in t.term_pow.iter().enumerate() {
        if *w > 3 {
            return Err(AlgebraError::ShapeMismatch {
                expected: "term powers <= 3".to_string(),
                actual: format!("term_pow[{ti}] = {w}"),
            });
        }
    }
    if nrow == 0 || nb == 0 {
        return Ok((vec![0.0; nrow * nb], vec![0.0; nrow * nb]));
    }
    // The test passes `nbs = nb`: one block, no ragged tail.
    let nbs = nb;
    let out = dispatch_backend!(
        client,
        c,
        Rt,
        launch_pp_proj::<Rt>(c, t, gv, kpt, g0, nb, nrow, nbs)
    );
    Ok(out)
}

/// Host tables for K-PP1. Built by `pyscf_pbc_df::pp_gspace`.
#[derive(Debug, Clone, Default)]
pub struct PpFtAoTables {
    pub nao: usize,
    pub ao_r: Vec<f64>,
    pub ao_l: Vec<u32>,
    pub ao_prim0: Vec<u32>,
    pub ao_nprim: Vec<u32>,
    pub ao_term0: Vec<u32>,
    pub ao_nterm: Vec<u32>,
    pub prim: Vec<f64>,
    pub prim_eoff: Vec<u32>,
    pub etab: Vec<f64>,
    pub term_c: Vec<f64>,
    pub term_pow: Vec<u32>,
}

/// K-PP1 — one lane per `(q, g)`; index `q * nbs + g` (AO-major).
///
/// Lane `(q, g)` with `q` a SPHERICAL AO index writes
/// `A[q, g] = ft_ao(cell, Gv, kpt)[g, q] / sqrt(vol)` for the block
/// `Gv[g0..g0+nb]`. With `Gk = Gv[g0+g] + kpt`, `g2 = |Gk|²`, AO centre `R`,
/// angular momentum `l`, `nt = l + 1`:
///
/// ```text
/// acc = Σ_primitives Σ_terms term_c · weight · exp(−g2/4α) · poly · e^{−iGk·R}
/// ```
///
/// where `poly` is the Hermite polynomial in `−iGk` from the primitive's `E`
/// table and `weight` (host-side) already holds `1/sqrt(vol)`.
///
/// Concrete `f64`: it calls `cube_math::double` (see `ft_aopair.rs:27-37`).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn pp_ftao_kernel(
    gv: &Array<f64>,
    kpt: &Array<f64>,
    ao_r: &Array<f64>,
    ao_l: &Array<u32>,
    ao_prim0: &Array<u32>,
    ao_nprim: &Array<u32>,
    ao_term0: &Array<u32>,
    ao_nterm: &Array<u32>,
    prim: &Array<f64>,
    prim_eoff: &Array<u32>,
    etab: &Array<f64>,
    term_c: &Array<f64>,
    term_pow: &Array<u32>,
    a_re: &mut Array<f64>,
    a_im: &mut Array<f64>,
    nao: usize,
    nbs: usize,
    g0: usize,
    nb: usize,
) {
    let i = ABSOLUTE_POS;
    if i < nao * nbs {
        let q = i / nbs;
        let g = i % nbs;
        if g < nb {
            let o = (g0 + g) * 3;
            let gx = gv[o] + kpt[0];
            let gy = gv[o + 1] + kpt[1];
            let gz = gv[o + 2] + kpt[2];
            let g2 = gx * gx + gy * gy + gz * gz;

            let nt = ao_l[q] as usize + 1;
            let r0 = ao_prim0[q] as usize;
            let nr = ao_nprim[q] as usize;
            let t0 = ao_term0[q] as usize;
            let ntm = ao_nterm[q] as usize;

            let mut acc_re = 0.0;
            let mut acc_im = 0.0;
            for r in r0..(r0 + nr) {
                let alpha = prim[r * 2];
                let w = prim[r * 2 + 1]
                    * cube_math::double::exp::exp(
                        0.0 - g2 / (4.0 * alpha),
                        cube_math::MathConfig::EXACT,
                    );
                let eb = prim_eoff[r] as usize;
                for t in t0..(t0 + ntm) {
                    let ix = term_pow[t * 3] as usize;
                    let iy = term_pow[t * 3 + 1] as usize;
                    let iz = term_pow[t * 3 + 2] as usize;
                    let bx = eb + ix * nt;
                    let by = eb + iy * nt;
                    let bz = eb + iz * nt;
                    let mut poly_re = 0.0;
                    let mut poly_im = 0.0;
                    let mut gxp = 1.0;
                    for tt in 0..(ix + 1) {
                        let et = etab[bx + tt];
                        let mut gyp = 1.0;
                        for uu in 0..(iy + 1) {
                            let etu = et * etab[by + uu] * gxp * gyp;
                            let mut gzp = 1.0;
                            for vv in 0..(iz + 1) {
                                let ww = etu * etab[bz + vv] * gzp;
                                let n = (tt + uu + vv) % 4;
                                if n == 0 {
                                    poly_re += ww;
                                } else if n == 1 {
                                    poly_im -= ww;
                                } else if n == 2 {
                                    poly_re -= ww;
                                } else {
                                    poly_im += ww;
                                }
                                gzp *= gz;
                            }
                            gyp *= gy;
                        }
                        gxp *= gx;
                    }
                    let c = term_c[t] * w;
                    acc_re += c * poly_re;
                    acc_im += c * poly_im;
                }
            }

            let theta = 0.0 - (gx * ao_r[q * 3] + gy * ao_r[q * 3 + 1] + gz * ao_r[q * 3 + 2]);
            let (sn, cs) = cube_math::double::trig::sincos(theta, cube_math::MathConfig::EXACT);
            a_re[i] = acc_re * cs - acc_im * sn;
            a_im[i] = acc_re * sn + acc_im * cs;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_pp_ftao_on_handles<R: Runtime>(
    client: &ComputeClient<R>,
    gv: &Handle,
    kpt: &Handle,
    tables: &[Handle; 11],
    a_re: &Handle,
    a_im: &Handle,
    t: &PpFtAoTables,
    ngrids: usize,
    nbs: usize,
    g0: usize,
    nb: usize,
) {
    let nao = t.nao;
    let (count, dim) = launch_1d(client, nao * nbs, 400);
    unsafe {
        pp_ftao_kernel::launch_unchecked::<R>(
            client,
            count,
            dim,
            // SAFETY: every element count below is the uploaded host Vec's
            // length (validated in `pp_ftao_block`); lanes guard
            // `i < nao*nbs` and `g < nb`, and every primitive/term range was
            // checked with powers ≤ 3.
            ArrayArg::from_raw_parts(gv.clone(), ngrids * 3),
            ArrayArg::from_raw_parts(kpt.clone(), 3),
            ArrayArg::from_raw_parts(tables[0].clone(), nao * 3),
            ArrayArg::from_raw_parts(tables[1].clone(), nao),
            ArrayArg::from_raw_parts(tables[2].clone(), nao),
            ArrayArg::from_raw_parts(tables[3].clone(), nao),
            ArrayArg::from_raw_parts(tables[4].clone(), nao),
            ArrayArg::from_raw_parts(tables[5].clone(), nao),
            ArrayArg::from_raw_parts(tables[6].clone(), t.prim.len()),
            ArrayArg::from_raw_parts(tables[7].clone(), t.prim_eoff.len()),
            ArrayArg::from_raw_parts(tables[8].clone(), t.etab.len()),
            ArrayArg::from_raw_parts(tables[9].clone(), t.term_c.len()),
            ArrayArg::from_raw_parts(tables[10].clone(), t.term_c.len() * 3),
            ArrayArg::from_raw_parts(a_re.clone(), nao * nbs),
            ArrayArg::from_raw_parts(a_im.clone(), nao * nbs),
            nao,
            nbs,
            g0,
            nb,
        );
    }
}

fn launch_pp_ftao<R: Runtime>(
    client: &ComputeClient<R>,
    t: &PpFtAoTables,
    gv: &[f64],
    kpt: [f64; 3],
    g0: usize,
    nb: usize,
    nbs: usize,
) -> (Vec<f64>, Vec<f64>) {
    let nao = t.nao;
    let gv_h = upload::<R, f64>(client, gv);
    let kpt_h = upload::<R, f64>(client, &[kpt[0], kpt[1], kpt[2]]);
    let tables = [
        upload::<R, f64>(client, &t.ao_r),
        upload_u32::<R>(client, &t.ao_l),
        upload_u32::<R>(client, &t.ao_prim0),
        upload_u32::<R>(client, &t.ao_nprim),
        upload_u32::<R>(client, &t.ao_term0),
        upload_u32::<R>(client, &t.ao_nterm),
        upload::<R, f64>(client, &t.prim),
        upload_u32::<R>(client, &t.prim_eoff),
        upload::<R, f64>(client, &t.etab),
        upload::<R, f64>(client, &t.term_c),
        upload_u32::<R>(client, &t.term_pow),
    ];
    let a_re_h = client.empty(nao * nbs * core::mem::size_of::<f64>());
    let a_im_h = client.empty(nao * nbs * core::mem::size_of::<f64>());
    launch_pp_ftao_on_handles::<R>(
        client,
        &gv_h,
        &kpt_h,
        &tables,
        &a_re_h,
        &a_im_h,
        t,
        gv.len() / 3,
        nbs,
        g0,
        nb,
    );
    let bytes = client.read(vec![a_re_h, a_im_h]);
    (
        bytemuck::cast_slice::<u8, f64>(&bytes[0]).to_vec(),
        bytemuck::cast_slice::<u8, f64>(&bytes[1]).to_vec(),
    )
}

/// K-PP1 on host slices for one block: `(a_re, a_im)`, each `(nao, nb)`
/// AO-major (the caller passes `nbs = nb` for a single block).
///
/// # Errors
/// [`AlgebraError::ShapeMismatch`] when a table or slice length is
/// inconsistent (`gv` not a multiple of 3, `g0 + nb` past the grid, table
/// lengths, primitive/term ranges, powers above 3).
pub fn pp_ftao_block(
    client: &AlgebraClient,
    t: &PpFtAoTables,
    gv: &[f64],
    kpt: [f64; 3],
    g0: usize,
    nb: usize,
) -> Result<(Vec<f64>, Vec<f64>), AlgebraError> {
    if gv.len() % 3 != 0 {
        return Err(AlgebraError::ShapeMismatch {
            expected: "gv len a multiple of 3".to_string(),
            actual: format!("{} elements", gv.len()),
        });
    }
    let nao = t.nao;
    let ngrids = gv.len() / 3;
    if t.ao_r.len() % 3 != 0 || t.ao_r.len() / 3 != nao {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("ao_r len {}", 3 * nao),
            actual: format!("{} elements", t.ao_r.len()),
        });
    }
    let expect = |name: &str, got: usize, want: usize| {
        if got != want {
            return Err(AlgebraError::ShapeMismatch {
                expected: format!("{name} len {want}"),
                actual: format!("{got}"),
            });
        }
        Ok(())
    };
    for (name, v) in [
        ("ao_l", &t.ao_l),
        ("ao_prim0", &t.ao_prim0),
        ("ao_nprim", &t.ao_nprim),
        ("ao_term0", &t.ao_term0),
        ("ao_nterm", &t.ao_nterm),
    ] {
        expect(name, v.len(), nao)?;
    }
    if t.prim.len() % 2 != 0 {
        return Err(AlgebraError::ShapeMismatch {
            expected: "prim len even".to_string(),
            actual: format!("{} elements", t.prim.len()),
        });
    }
    let nprim_total = t.prim.len() / 2;
    expect("prim_eoff", t.prim_eoff.len(), nprim_total)?;
    expect("term_pow", t.term_pow.len(), 3 * t.term_c.len())?;
    if !matches!(g0.checked_add(nb), Some(end) if end <= ngrids) {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("g0 {g0} + nb {nb} <= ngrids {ngrids}"),
            actual: format!("g0 {g0} + nb {nb}"),
        });
    }
    for q in 0..nao {
        let r_end = t.ao_prim0[q] as usize + t.ao_nprim[q] as usize;
        if r_end > nprim_total {
            return Err(AlgebraError::ShapeMismatch {
                expected: format!("AO {q} primitive range within {nprim_total}"),
                actual: format!("range ends at {r_end}"),
            });
        }
        let t_end = t.ao_term0[q] as usize + t.ao_nterm[q] as usize;
        if t_end > t.term_c.len() {
            return Err(AlgebraError::ShapeMismatch {
                expected: format!("AO {q} term range within {}", t.term_c.len()),
                actual: format!("range ends at {t_end}"),
            });
        }
    }
    for (ti, w) in t.term_pow.iter().enumerate() {
        if *w > 3 {
            return Err(AlgebraError::ShapeMismatch {
                expected: "term powers <= 3".to_string(),
                actual: format!("term_pow[{ti}] = {w}"),
            });
        }
    }
    if nao == 0 || nb == 0 {
        return Ok((vec![0.0; nao * nb], vec![0.0; nao * nb]));
    }
    let nbs = nb;
    let out = dispatch_backend!(
        client,
        c,
        Rt,
        launch_pp_ftao::<Rt>(c, t, gv, kpt, g0, nb, nbs)
    );
    Ok(out)
}

/// Everything one `pp_gspace_project` call uploads.
#[derive(Debug, Clone, Default)]
pub struct PpGspaceTables {
    /// `(ngrids, 3)` mesh G-vectors, flattened (NOT shifted by k).
    pub gv: Vec<f64>,
    pub proj: PpProjTables,
    pub ftao: PpFtAoTables,
}

fn validate_gspace_tables(t: &PpGspaceTables) -> Result<(usize, usize, usize), AlgebraError> {
    let expect = |name: &str, got: usize, want: usize| {
        if got != want {
            return Err(AlgebraError::ShapeMismatch {
                expected: format!("{name} len {want}"),
                actual: format!("{got}"),
            });
        }
        Ok(())
    };
    if t.gv.len() % 3 != 0 {
        return Err(AlgebraError::ShapeMismatch {
            expected: "gv len a multiple of 3".to_string(),
            actual: format!("{} elements", t.gv.len()),
        });
    }
    let ngrids = t.gv.len() / 3;
    if t.proj.row_r.len() % 3 != 0 {
        return Err(AlgebraError::ShapeMismatch {
            expected: "proj.row_r len a multiple of 3".to_string(),
            actual: format!("{} elements", t.proj.row_r.len()),
        });
    }
    let nrow = t.proj.row_r.len() / 3;
    let nao = t.ftao.nao;
    expect("proj.row_par", t.proj.row_par.len(), 5 * nrow)?;
    expect("proj.row_term0", t.proj.row_term0.len(), nrow)?;
    expect("proj.row_nterm", t.proj.row_nterm.len(), nrow)?;
    expect(
        "proj.term_pow",
        t.proj.term_pow.len(),
        3 * t.proj.term_c.len(),
    )?;
    for p in 0..nrow {
        let end = t.proj.row_term0[p] as usize + t.proj.row_nterm[p] as usize;
        if end > t.proj.term_c.len() {
            return Err(AlgebraError::ShapeMismatch {
                expected: format!("proj row {p} term range within {}", t.proj.term_c.len()),
                actual: format!("range ends at {end}"),
            });
        }
    }
    for (ti, w) in t.proj.term_pow.iter().enumerate() {
        if *w > 3 {
            return Err(AlgebraError::ShapeMismatch {
                expected: "proj term powers <= 3".to_string(),
                actual: format!("term_pow[{ti}] = {w}"),
            });
        }
    }
    if t.ftao.ao_r.len() != 3 * nao {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("ftao.ao_r len {}", 3 * nao),
            actual: format!("{} elements", t.ftao.ao_r.len()),
        });
    }
    for (name, len) in [
        ("ftao.ao_l", t.ftao.ao_l.len()),
        ("ftao.ao_prim0", t.ftao.ao_prim0.len()),
        ("ftao.ao_nprim", t.ftao.ao_nprim.len()),
        ("ftao.ao_term0", t.ftao.ao_term0.len()),
        ("ftao.ao_nterm", t.ftao.ao_nterm.len()),
    ] {
        expect(name, len, nao)?;
    }
    if t.ftao.prim.len() % 2 != 0 {
        return Err(AlgebraError::ShapeMismatch {
            expected: "ftao.prim len even".to_string(),
            actual: format!("{} elements", t.ftao.prim.len()),
        });
    }
    let nprim_total = t.ftao.prim.len() / 2;
    expect("ftao.prim_eoff", t.ftao.prim_eoff.len(), nprim_total)?;
    expect(
        "ftao.term_pow",
        t.ftao.term_pow.len(),
        3 * t.ftao.term_c.len(),
    )?;
    for q in 0..nao {
        let r_end = t.ftao.ao_prim0[q] as usize + t.ftao.ao_nprim[q] as usize;
        if r_end > nprim_total {
            return Err(AlgebraError::ShapeMismatch {
                expected: format!("ftao AO {q} primitive range within {nprim_total}"),
                actual: format!("range ends at {r_end}"),
            });
        }
        let t_end = t.ftao.ao_term0[q] as usize + t.ftao.ao_nterm[q] as usize;
        if t_end > t.ftao.term_c.len() {
            return Err(AlgebraError::ShapeMismatch {
                expected: format!("ftao AO {q} term range within {}", t.ftao.term_c.len()),
                actual: format!("range ends at {t_end}"),
            });
        }
    }
    for (ti, w) in t.ftao.term_pow.iter().enumerate() {
        if *w > 3 {
            return Err(AlgebraError::ShapeMismatch {
                expected: "ftao term powers <= 3".to_string(),
                actual: format!("term_pow[{ti}] = {w}"),
            });
        }
    }
    Ok((ngrids, nrow, nao))
}

fn run_gspace_project<R: Runtime>(
    client: &ComputeClient<R>,
    t: &PpGspaceTables,
    kpts: &[[f64; 3]],
    block: usize,
) -> Result<Vec<(Vec<f64>, Vec<f64>)>, AlgebraError> {
    let (ngrids, nrow, nao) = validate_gspace_tables(t)?;
    if ngrids == 0 || nrow == 0 {
        return Ok(kpts.iter().map(|_| (Vec::new(), Vec::new())).collect());
    }
    // Upload once: gv and every table, before the k loop.
    let gv_h = upload::<R, f64>(client, &t.gv);
    let proj_tables = [
        upload::<R, f64>(client, &t.proj.row_r),
        upload::<R, f64>(client, &t.proj.row_par),
        upload_u32::<R>(client, &t.proj.row_term0),
        upload_u32::<R>(client, &t.proj.row_nterm),
        upload::<R, f64>(client, &t.proj.term_c),
        upload_u32::<R>(client, &t.proj.term_pow),
    ];
    let ftao = &t.ftao;
    let ftao_tables = [
        upload::<R, f64>(client, &ftao.ao_r),
        upload_u32::<R>(client, &ftao.ao_l),
        upload_u32::<R>(client, &ftao.ao_prim0),
        upload_u32::<R>(client, &ftao.ao_nprim),
        upload_u32::<R>(client, &ftao.ao_term0),
        upload_u32::<R>(client, &ftao.ao_nterm),
        upload::<R, f64>(client, &ftao.prim),
        upload_u32::<R>(client, &ftao.prim_eoff),
        upload::<R, f64>(client, &ftao.etab),
        upload::<R, f64>(client, &ftao.term_c),
        upload_u32::<R>(client, &ftao.term_pow),
    ];
    // Allocate the block buffers once and reuse them for every k-point.
    let nbs = block.clamp(1, ngrids);
    let a_re_h = client.empty(nao * nbs * core::mem::size_of::<f64>());
    let a_im_h = client.empty(nao * nbs * core::mem::size_of::<f64>());
    let s_re_h = client.empty(nrow * nbs * core::mem::size_of::<f64>());
    let s_im_h = client.empty(nrow * nbs * core::mem::size_of::<f64>());
    let mut out = Vec::with_capacity(kpts.len());
    for kpt in kpts {
        let kpt_h = upload::<R, f64>(client, &[kpt[0], kpt[1], kpt[2]]);
        // `B` starts at zero: `client.empty` would leave allocator garbage.
        let b_re_h = upload::<R, f64>(client, &vec![0.0f64; nrow * nao]);
        let b_im_h = upload::<R, f64>(client, &vec![0.0f64; nrow * nao]);
        let mut g0 = 0;
        while g0 < ngrids {
            let nb = (ngrids - g0).min(nbs);
            launch_pp_ftao_on_handles::<R>(
                client,
                &gv_h,
                &kpt_h,
                &ftao_tables,
                &a_re_h,
                &a_im_h,
                ftao,
                ngrids,
                nbs,
                g0,
                nb,
            );
            launch_pp_proj_on_handles::<R>(
                client,
                &gv_h,
                &kpt_h,
                &proj_tables,
                &s_re_h,
                &s_im_h,
                &t.proj,
                ngrids,
                nrow,
                nbs,
                g0,
                nb,
            );
            launch_pp_fold_on_handles::<R, f64>(
                client, &s_re_h, &s_im_h, &a_re_h, &a_im_h, &b_re_h, &b_im_h, nrow, nao, nbs, nb,
            );
            g0 += nb;
        }
        // The only read-back: one call per k-point.
        let bytes = client.read(vec![b_re_h, b_im_h]);
        out.push((
            bytemuck::cast_slice::<u8, f64>(&bytes[0]).to_vec(),
            bytemuck::cast_slice::<u8, f64>(&bytes[1]).to_vec(),
        ));
    }
    Ok(out)
}

/// `B[row, q]` for every k-point: planar, row-major `(nrow, nao)`.
/// `block` = G-points per block (clamped to `1..=ngrids`).
///
/// # Errors
/// [`AlgebraError::ShapeMismatch`] when the tables are inconsistent (same
/// checks as [`pp_proj_block`] and [`pp_ftao_block`]).
pub fn pp_gspace_project(
    client: &AlgebraClient,
    t: &PpGspaceTables,
    kpts: &[[f64; 3]],
    block: usize,
) -> Result<Vec<(Vec<f64>, Vec<f64>)>, AlgebraError> {
    dispatch_backend!(client, c, Rt, run_gspace_project::<Rt>(c, t, kpts, block))
}
