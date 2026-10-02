//! The non-local GTH pseudopotential in reciprocal space —
//! `pyscf/pbc/df/fft.py:114-176`.
//!
//! Upstream evaluates the non-local part on the FFT mesh: for every atom with
//! a pseudopotential, every channel `l` with `nproj > 0`, every projector
//! `i < nproj` and every `m < 2l+1` it builds one row
//! `S[row, g] = conj(SI[a, g]) · pYlm[i, m, g]`, folds it against the
//! analytic AO transform `aokG`, and sandwiches with the raw coupling matrix.
//! This module is the host (plain Rust) reference every K-PP kernel is tested
//! against; T10 wires it into `Fftdf::get_pp` as the default route.

use pyscf_kernels::{cart_powers, cart2sph_l_matrix, common_fac_sp};
use pyscf_pbc_gto::Cell;
use rayon::prelude::*;

use crate::error::PbcDfError;

/// One projector function — one row of `S[row, g]`.
#[derive(Debug, Clone)]
pub struct ProjRow {
    pub atom: usize,
    pub l: usize,
    pub i: usize,
    pub m: usize,
    pub rl: f64,
}

/// One `(atom, l)` channel. Its rows are `row0 + i·(2l+1) + m`.
#[derive(Debug, Clone)]
pub struct ProjChannel {
    pub l: usize,
    pub nproj: usize,
    pub row0: usize,
    /// RAW `nproj × nproj` coupling matrix, row-major (`GthProjector::h`).
    pub h: Vec<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct ProjTables {
    pub rows: Vec<ProjRow>,
    pub channels: Vec<ProjChannel>,
}

/// `[c0, c1, c2]` with `qli = c0 + c1·x² + c2·x⁴` (`pp.py:150-185`).
/// The stored coefficients already include the prefactor.
pub fn qli_coeffs(l: usize, i: usize) -> Option<[f64; 3]> {
    let (p, c0, c1, c2) = match (l, i) {
        (0, 0) => (4.0 * 2.0f64.sqrt(), 1.0, 0.0, 0.0),
        (0, 1) => (8.0 * (2.0 / 15.0f64).sqrt(), 3.0, -1.0, 0.0),
        (0, 2) => (16.0 / 3.0 * (2.0 / 105.0f64).sqrt(), 15.0, -10.0, 1.0),
        (1, 0) => (8.0 * (1.0 / 3.0f64).sqrt(), 1.0, 0.0, 0.0),
        (1, 1) => (16.0 * (1.0 / 105.0f64).sqrt(), 5.0, -1.0, 0.0),
        (1, 2) => (32.0 / 3.0 * (1.0 / 1155.0f64).sqrt(), 35.0, -14.0, 1.0),
        (2, 0) => (8.0 * (2.0 / 15.0f64).sqrt(), 1.0, 0.0, 0.0),
        (2, 1) => (16.0 / 3.0 * (2.0 / 105.0f64).sqrt(), 7.0, -1.0, 0.0),
        (2, 2) => (32.0 / 3.0 * (2.0 / 15015.0f64).sqrt(), 63.0, -18.0, 1.0),
        (3, 0) => (16.0 * (1.0 / 105.0f64).sqrt(), 1.0, 0.0, 0.0),
        (3, 1) => (32.0 / 3.0 * (1.0 / 1155.0f64).sqrt(), 9.0, -1.0, 0.0),
        (3, 2) => (64.0 / 45.0 * (1.0 / 1001.0f64).sqrt(), 99.0, -22.0, 1.0),
        _ => return None,
    };
    Some([p * c0, p * c1, p * c2])
}

/// Rows in the order atom, l, i, m (`fft.py:133-150`).
pub fn proj_tables(cell: &Cell) -> Result<ProjTables, PbcDfError> {
    let mut rows = Vec::new();
    let mut channels = Vec::new();
    for ia in 0..cell.mol.natm {
        let sym = &cell.mol._atom[ia].0;
        let Some(pseudo) = cell.pseudo.as_ref().and_then(|p| p.get(sym)) else {
            continue;
        };
        for (l, proj) in pseudo.projectors.iter().enumerate() {
            if proj.nproj == 0 {
                continue;
            }
            let row0 = rows.len();
            for i in 0..proj.nproj {
                if qli_coeffs(l, i).is_none() {
                    return Err(PbcDfError::Backend(format!(
                        "pp_gspace: no qli for l={l} i={i}"
                    )));
                }
                for m in 0..2 * l + 1 {
                    rows.push(ProjRow {
                        atom: ia,
                        l,
                        i,
                        m,
                        rl: proj.r,
                    });
                }
            }
            channels.push(ProjChannel {
                l,
                nproj: proj.nproj,
                row0,
                h: proj.h.clone(),
            });
        }
    }
    Ok(ProjTables { rows, channels })
}

/// `S[row, g]`, planar, row-major `(nrow, gv.len())`: index `row * n + g`.
///
/// With `Gk = gv[g] + kpt`, `alpha = ½·rl²`, `x² = |Gk|²·rl²` (`fft.py:128-150`):
/// the real-solid-harmonic Gaussian of the row times `qli(x², l, i)` times
/// the phase `exp(+i·Gv[g]·R_a)` — `conj(SI)`, which uses `Gv`, NOT `Gk`.
pub fn proj_values_host(
    cell: &Cell,
    t: &ProjTables,
    gv: &[[f64; 3]],
    kpt: [f64; 3],
) -> Result<(Vec<f64>, Vec<f64>), PbcDfError> {
    let n = gv.len();
    let coords = cell.mol.atom_coords();
    let pi = std::f64::consts::PI;
    let rows: Result<Vec<(Vec<f64>, Vec<f64>)>, PbcDfError> = t
        .rows
        .par_iter()
        .map(|row| {
            let l = row.l as u32;
            let tm = cart2sph_l_matrix(l).map_err(PbcDfError::from)?;
            let powers = cart_powers(l);
            let nc = powers.len();
            let r = coords[row.atom];
            let alpha = 0.5 * row.rl * row.rl;
            let coef = row.rl.powf(l as f64 + 1.5) * pi.powf(1.25) * common_fac_sp(l);
            let [c0, c1, c2] = qli_coeffs(row.l, row.i).ok_or_else(|| {
                PbcDfError::Backend(format!("pp_gspace: no qli for l={} i={}", row.l, row.i))
            })?;
            let mut re = vec![0.0f64; n];
            let mut im = vec![0.0f64; n];
            for (g, gvec) in gv.iter().enumerate() {
                let qx = gvec[0] + kpt[0];
                let qy = gvec[1] + kpt[1];
                let qz = gvec[2] + kpt[2];
                let q2 = qx * qx + qy * qy + qz * qz;
                let x2 = q2 * row.rl * row.rl;
                let ql = c0 + c1 * x2 + c2 * x2 * x2;
                let mut poly = 0.0f64;
                for (c, &(ix, iy, iz)) in powers.iter().enumerate() {
                    let w = tm[row.m * nc + c];
                    if w != 0.0 {
                        poly += w * qx.powi(ix as i32) * qy.powi(iy as i32) * qz.powi(iz as i32);
                    }
                }
                let val = poly * coef * (-alpha * q2).exp() * ql;
                let th = gvec[0] * r[0] + gvec[1] * r[1] + gvec[2] * r[2];
                let (sn, cs) = th.sin_cos();
                re[g] = val * cs;
                im[g] = val * sn;
            }
            Ok((re, im))
        })
        .collect();
    let rows = rows?;
    let mut out_re = vec![0.0f64; t.rows.len() * n];
    let mut out_im = vec![0.0f64; t.rows.len() * n];
    for (row, (re, im)) in rows.into_iter().enumerate() {
        out_re[row * n..(row + 1) * n].copy_from_slice(&re);
        out_im[row * n..(row + 1) * n].copy_from_slice(&im);
    }
    Ok((out_re, out_im))
}

/// Which evaluation of the non-local pseudopotential FFTDF uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PpNonlocal {
    /// Upstream's: reciprocal space on the FFT mesh (`fft.py:114-176`).
    Reciprocal,
    /// Analytic, mesh-independent (`pyscf_pbc_gto::pseudo::get_pp_nl`).
    RealSpace,
}

/// `PYSCF_PBC_FFTDF_PP_NL=realspace` selects [`PpNonlocal::RealSpace`];
/// anything else (or unset) is upstream's [`PpNonlocal::Reciprocal`].
/// Read on every call — not cached — so one process can test both.
pub fn pp_nonlocal_route() -> PpNonlocal {
    match std::env::var("PYSCF_PBC_FFTDF_PP_NL") {
        Ok(v) if v.trim().eq_ignore_ascii_case("realspace") => PpNonlocal::RealSpace,
        _ => PpNonlocal::Reciprocal,
    }
}

/// G-points per block for a memory budget in MB (`A` is `nb × nao` and `S`
/// is `nrow × nb`, 16 bytes per complex value). At least 1024 points.
pub fn block_points(budget_mb: f64, nao: usize, nrow: usize, ngrids: usize) -> usize {
    let per_point = 16.0 * (nao + nrow) as f64;
    let nb = (budget_mb * 1024.0 * 1024.0 / per_point) as usize;
    nb.max(1024).min(ngrids.max(1))
}

/// `PYSCF_PBC_PP_NL_BUDGET_MB`, default 1024. Read on every call.
pub fn budget_mb() -> f64 {
    std::env::var("PYSCF_PBC_PP_NL_BUDGET_MB")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|v| *v > 0.0)
        .unwrap_or(1024.0)
}

/// `B[row, q]` on the host, G walked in blocks of `block` points.
/// Planar, row-major `(nrow, nao)`.
pub fn project_host(
    cell: &Cell,
    t: &ProjTables,
    gv: &[[f64; 3]],
    kpt: [f64; 3],
    block: usize,
) -> Result<(Vec<f64>, Vec<f64>), PbcDfError> {
    let nao = cell.mol.nao_nr;
    let nrow = t.rows.len();
    let ngrids = gv.len();
    let mut b_re = vec![0.0f64; nrow * nao];
    let mut b_im = vec![0.0f64; nrow * nao];
    if nrow == 0 || nao == 0 {
        return Ok((b_re, b_im));
    }
    let mut g0 = 0;
    while g0 < ngrids {
        let g1 = (g0 + block.max(1)).min(ngrids);
        let nb = g1 - g0;
        let (a_re, a_im) = crate::ft_ao::ft_ao_kpt(&cell.mol, &gv[g0..g1], kpt)?;
        let (s_re, s_im) = proj_values_host(cell, t, &gv[g0..g1], kpt)?;
        // One accumulator per `(row, q)` element, carried across the blocks,
        // and `g` ascending: the sum does not depend on how the grid was
        // split, bit for bit (the property K-PP3 has on the device).
        b_re.par_chunks_mut(nao)
            .zip(b_im.par_chunks_mut(nao))
            .enumerate()
            .for_each(|(row, (br, bi))| {
                for g in 0..nb {
                    let (sr, si) = (s_re[row * nb + g], s_im[row * nb + g]);
                    let (ar, ai) = (&a_re[g * nao..(g + 1) * nao], &a_im[g * nao..(g + 1) * nao]);
                    for q in 0..nao {
                        br[q] += sr * ar[q] - si * ai[q];
                        bi[q] += sr * ai[q] + si * ar[q];
                    }
                }
            });
        g0 = g1;
    }
    // `A = ft_ao / sqrt(vol)` (`fft.py:126`), applied once to the finished sum.
    let inv_sqrt_vol = 1.0 / cell.vol().sqrt();
    for v in b_re.iter_mut().chain(b_im.iter_mut()) {
        *v *= inv_sqrt_vol;
    }
    Ok((b_re, b_im))
}

/// `V[p, q]` from `B` — the sum over channels in the formula above.
/// Row-major `(nao, nao)`.
pub fn sandwich(
    t: &ProjTables,
    b_re: &[f64],
    b_im: &[f64],
    nao: usize,
    vol: f64,
) -> pyscf_algebra::CTensor {
    let mut out = pyscf_algebra::CTensor::zeros(nao * nao);
    for ch in &t.channels {
        let nl2 = 2 * ch.l + 1;
        for i in 0..ch.nproj {
            for j in 0..ch.nproj {
                let h = ch.h[i * ch.nproj + j];
                if h == 0.0 {
                    continue;
                }
                for m in 0..nl2 {
                    let ri = ch.row0 + i * nl2 + m;
                    let rj = ch.row0 + j * nl2 + m;
                    for p in 0..nao {
                        let (br, bi) = (b_re[ri * nao + p], b_im[ri * nao + p]);
                        // conj(B[ri, p]) * h
                        let (cr, ci) = (br * h, -bi * h);
                        for q in 0..nao {
                            let (dr, di) = (b_re[rj * nao + q], b_im[rj * nao + q]);
                            out.re[p * nao + q] += cr * dr - ci * di;
                            out.im[p * nao + q] += cr * di + ci * dr;
                        }
                    }
                }
            }
        }
    }
    let inv_vol = 1.0 / vol;
    for v in out.re.iter_mut().chain(out.im.iter_mut()) {
        *v *= inv_vol;
    }
    out
}

/// The non-local pseudopotential at every k-point, reciprocal-space route.
pub fn get_pp_nl_gspace(
    cell: &Cell,
    mesh: [usize; 3],
    kpts: &[[f64; 3]],
) -> Result<Vec<pyscf_algebra::CTensor>, PbcDfError> {
    let gv = pyscf_pbc_gto::get_gv(cell, Some(mesh))?;
    let t = proj_tables(cell)?;
    let nao = cell.mol.nao_nr;
    let block = block_points(budget_mb(), nao, t.rows.len(), gv.len());
    let vol = cell.vol();
    let ngrids = gv.len();
    let nrow = t.rows.len();
    // The device route is the default; `PYSCF_PBC_PP_NL_HOST=1` — or a cell
    // without K-PP1 tables — takes the host route (the reference).
    let force_host = std::env::var("PYSCF_PBC_PP_NL_HOST").is_ok_and(|v| v == "1");
    let device_blocks = if force_host {
        None
    } else {
        project_device(cell, &t, &gv, kpts, block)?
    };
    let (b_all, device) = match device_blocks {
        Some(b) => (b, "device"),
        None => {
            let mut out = Vec::with_capacity(kpts.len());
            for kpt in kpts {
                out.push(project_host(cell, &t, &gv, *kpt, block)?);
            }
            (out, "host")
        }
    };
    tracing::info!(
        ngrids,
        nao,
        nrow,
        block,
        device,
        "pp_gspace: non-local pseudopotential"
    );
    let mut out = Vec::with_capacity(kpts.len());
    for (b_re, b_im) in &b_all {
        out.push(sandwich(&t, b_re, b_im, nao, vol));
    }
    Ok(out)
}

/// K-PP1 tables for `cell`. `None` when a shell has `l > 3` or the Mole is
/// cartesian — the caller then uses the host route.
pub fn ftao_tables(
    cell: &Cell,
) -> Result<Option<pyscf_kernels::pbc::pp_gspace::PpFtAoTables>, PbcDfError> {
    use pyscf_core::raw_layout::{
        ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, NCTR_OF, NPRIM_OF, PTR_COEFF, PTR_COORD, PTR_EXP,
    };

    use crate::ft_ao::mcmurchie::e_coefficients;

    let mol = &cell.mol;
    if mol.cart {
        return Ok(None);
    }
    let pi = std::f64::consts::PI;
    let inv_sqrt_vol = 1.0 / cell.vol().sqrt();
    let mut t = pyscf_kernels::pbc::pp_gspace::PpFtAoTables::default();
    for ib in 0..mol.nbas {
        let row = ib * BAS_SLOTS;
        let l = mol._bas[row + ANG_OF] as u32;
        if l > 3 {
            return Ok(None);
        }
        let nprim = mol._bas[row + NPRIM_OF] as usize;
        let nctr = mol._bas[row + NCTR_OF] as usize;
        let pe = mol._bas[row + PTR_EXP] as usize;
        let pc = mol._bas[row + PTR_COEFF] as usize;
        let atom = mol._bas[row + ATOM_OF] as usize;
        let pcoord = mol._atm[atom * ATM_SLOTS + PTR_COORD] as usize;
        let a = [mol._env[pcoord], mol._env[pcoord + 1], mol._env[pcoord + 2]];
        let powers = cart_powers(l);
        let nc = powers.len();
        let cfac = common_fac_sp(l);
        let tmat = cart2sph_l_matrix(l).map_err(PbcDfError::from)?;
        let nout = 2 * l as usize + 1;
        // The `E` tables depend only on the exponents, so they are built once
        // per shell; the weights depend on the contraction coefficients, so
        // the primitive records repeat per contraction.
        let mut eoffs = Vec::with_capacity(nprim);
        for p in 0..nprim {
            let alpha = mol._env[pe + p];
            eoffs.push(t.etab.len() as u32);
            t.etab
                .extend(e_coefficients(l, 0, alpha, 0.0, 0.0, 1.0).data);
        }
        // One AO per (contraction, spherical component) — the AO order of
        // `ft_ao_mol`'s output.
        for ictr in 0..nctr {
            let prim0 = (t.prim.len() / 2) as u32;
            for p in 0..nprim {
                let alpha = mol._env[pe + p];
                let coef = mol._env[pc + ictr * nprim + p];
                t.prim.push(alpha);
                // The weight without the per-G `exp(−g2/4α)`; keep every
                // primitive, including one whose weight is 0.0.
                t.prim
                    .push(coef * cfac * (pi / alpha).powf(1.5) * inv_sqrt_vol);
                t.prim_eoff.push(eoffs[p]);
            }
            for m in 0..nout {
                t.ao_r.extend_from_slice(&a);
                t.ao_l.push(l);
                t.ao_prim0.push(prim0);
                t.ao_nprim.push(nprim as u32);
                t.ao_term0.push(t.term_c.len() as u32);
                for c in 0..nc {
                    let w = tmat[m * nc + c];
                    if w != 0.0 {
                        t.term_c.push(w);
                        let (ix, iy, iz) = powers[c];
                        t.term_pow.push(ix);
                        t.term_pow.push(iy);
                        t.term_pow.push(iz);
                    }
                }
                let term0 = t.ao_term0[t.ao_term0.len() - 1] as usize;
                t.ao_nterm.push((t.term_c.len() - term0) as u32);
            }
        }
    }
    t.nao = t.ao_l.len();
    Ok(Some(t))
}

/// Host tables for K-PP2, built from [`ProjTables`].
pub fn proj_kernel_tables(
    cell: &Cell,
    t: &ProjTables,
) -> Result<pyscf_kernels::pbc::pp_gspace::PpProjTables, PbcDfError> {
    let coords = cell.mol.atom_coords();
    let pi = std::f64::consts::PI;
    let mut out = pyscf_kernels::pbc::pp_gspace::PpProjTables::default();
    for row in &t.rows {
        let r = coords[row.atom];
        out.row_r.extend_from_slice(&r);
        let alpha = 0.5 * row.rl * row.rl;
        let coef = row.rl.powf(row.l as f64 + 1.5) * pi.powf(1.25) * common_fac_sp(row.l as u32);
        let [c0, c1, c2] = qli_coeffs(row.l, row.i).ok_or_else(|| {
            PbcDfError::Backend(format!("pp_gspace: no qli for l={} i={}", row.l, row.i))
        })?;
        out.row_par.extend_from_slice(&[alpha, coef, c0, c1, c2]);
        let tm = cart2sph_l_matrix(row.l as u32).map_err(PbcDfError::from)?;
        let powers = cart_powers(row.l as u32);
        let nc = powers.len();
        out.row_term0.push(out.term_c.len() as u32);
        let mut nterm = 0u32;
        for c in 0..nc {
            let w = tm[row.m * nc + c];
            if w != 0.0 {
                out.term_c.push(w);
                let (ix, iy, iz) = powers[c];
                out.term_pow.push(ix);
                out.term_pow.push(iy);
                out.term_pow.push(iz);
                nterm += 1;
            }
        }
        out.row_nterm.push(nterm);
    }
    Ok(out)
}

/// One device call per `get_pp`: K-PP1 → K-PP2 → K-PP3 block by block,
/// returning only the small `B` matrix per k-point. `None` when the cell has
/// no K-PP1 tables (cartesian Mole or `l > 3`) — the caller uses the host
/// route.
pub fn project_device(
    cell: &Cell,
    t: &ProjTables,
    gv: &[[f64; 3]],
    kpts: &[[f64; 3]],
    block: usize,
) -> Result<Option<Vec<(Vec<f64>, Vec<f64>)>>, PbcDfError> {
    let Some(ftao) = ftao_tables(cell)? else {
        return Ok(None);
    };
    let mut flat_gv = Vec::with_capacity(gv.len() * 3);
    for g in gv {
        flat_gv.extend_from_slice(g);
    }
    let tables = pyscf_kernels::pbc::pp_gspace::PpGspaceTables {
        gv: flat_gv,
        proj: proj_kernel_tables(cell, t)?,
        ftao,
    };
    let client = pyscf_algebra::select_backend()
        .map_err(|e| PbcDfError::Backend(format!("pp_gspace: backend selection: {e}")))?
        .client;
    let b = pyscf_kernels::pbc::pp_gspace::pp_gspace_project(&client, &tables, kpts, block)
        .map_err(|e| PbcDfError::Backend(format!("pp_gspace device projection: {e}")))?;
    Ok(Some(b))
}
