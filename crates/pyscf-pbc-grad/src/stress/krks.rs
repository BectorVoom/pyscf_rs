//! Plan 18-13 Task 2 — `pyscf/pbc/grad/krks_stress.py` (404 l), the k index.
//!
//! # The eight shared symbols (imported, never reimplemented)
//!
//! Upstream imports the same eight from `rks_stress` (`krks_stress.py:74-83`);
//! this module imports them from [`super::rks`]:
//!
//! | upstream (`rks_stress.py`) | here |
//! |---|---|
//! | `strain_tensor_dispalcement` | [`rks::strain_tensor_displacement`](super::rks::strain_tensor_displacement) |
//! | `_finite_diff_cells` | [`rks::finite_diff_cells`](super::rks::finite_diff_cells) |
//! | `_get_weight_strain_derivatives` | [`rks::weight_strain_derivatives`](super::rks::weight_strain_derivatives) |
//! | `_get_coulG_strain_derivatives` | [`rks::coulg_strain_derivatives`](super::rks::coulg_strain_derivatives) |
//! | `_eval_ao_strain_derivatives` | [`rks::eval_ao_strain_derivatives`](super::rks::eval_ao_strain_derivatives) |
//! | `_get_vpplocG_strain_derivatives` | [`rks::vpplocg_strain_derivatives`](super::rks::vpplocg_strain_derivatives) |
//! | `_get_pp_nonloc_strain_derivatives` | k-point form below (the base symbol is gamma-only) |
//! | `ewald` | [`rks::ewald_strain`](super::rks::ewald_strain) |
//!
//! # `get_ovlp` / `get_kin` are the k-point closed forms, not FD
//!
//! [`krks_ovlp_strain`] / [`krks_kin_strain`] (`krks_stress.py:84-115`) ship
//! 18-12 Task 1's closed form
//! ([`rks::ip_strain_closed_form`](super::rks::ip_strain_closed_form)) at the
//! caller's k-points. The reason the gamma closed form carries over unchanged
//! to k > 0 is D-PBC-31 clause 3: `:88` fixes the k-points in FRACTIONAL
//! coordinates and `:93-94` re-derives the Cartesian ones per displaced cell,
//! so `k·L` is strain-invariant and the Bloch phase does not participate in
//! the derivative. The strain derivative acts on the integrals only, exactly
//! as at Γ.
//!
//! The corollary trap: straining the cell while carrying the original
//! CARTESIAN k-points differentiates a different quantity — correct at Γ,
//! wrong at every other k-point. [`finite_diff_cells`](super::rks::finite_diff_cells)
//! returns the transformed k-points (`kpts_plus`/`kpts_minus`); the FD oracle
//! in `tests/krks_stress.rs` uses them, and carries the untransformed ones
//! once, locally, to show the gate catches it.
//!
//! # `get_vxc`: all k-points per block
//!
//! `_eval_ao_strain_derivatives(cell, coords, kpts, deriv)` at `:173` returns
//! `nkpts` strain tables and `block_loop(..., kpts=kpts)` at `:171` returns
//! `nkpts` ordinary AO tables beside them. `rho0`/`rho1` accumulate with `+=`
//! across k (`:183-197`), stay k-independent, and are averaged (`1/nkpts`,
//! `2/nkpts` at `:227-229`) — the fused-contraction ruling survives, sized
//! through [`rks::strain_block_size_nset`](super::rks::strain_block_size_nset)
//! with `bytes_per_elem = 16` (complex).
//!
//! # Reported units
//!
//! Ha/Bohr³ through [`rks::to_stress`](super::rks::to_stress), matching 18-12.
//!
//! # Layout conventions (fixed here)
//!
//! * Integral matrices (`pbc_intor` products, [`get_ovlp`](pyscf_pbc_gto::get_ovlp),
//!   [`get_pp_nl`](pyscf_pbc_gto::pseudo::get_pp_nl), [`IpStrainK`](super::rks::IpStrainK)
//!   planes) are F-order: `M[i,j]` at `i + j*nao`.
//! * Density matrices (SCF `dm`, this module's inputs) are row-major
//!   [`CTensor`]: `D[i,j]` at `(re/im)[i*nao + j]`.
//! * Local orbitals from `kspu::make_minao_lo` are COLUMN-MAJOR `nao × nlo`.
//! * [`LocalOrbStrain`] planes are `((x*3+y)*nao + i)*nlo + j` per k-point.

use pyscf_algebra::{CTensor, oracle_sum};
use pyscf_core::{CoreError, PyscfRsError};
use pyscf_pbc_gto::Cell;

use super::rks::{
    IpStrainK, VxcStrainOpts, coulg_strain_derivatives, eval_ao_strain_derivatives, ewald_strain,
    finite_diff_cells, ip_strain_closed_form, strain_ao_block_comps, strain_block_size_nset,
    strain_block_size_nset_from_env, to_stress, vpplocg_strain_derivatives,
    weight_strain_derivatives,
};
use crate::error::PbcGradError;

fn invalid(msg: String) -> PyscfRsError {
    PyscfRsError::Core(CoreError::InvalidMolecule(msg))
}

fn lift_dft<T>(r: Result<T, pyscf_pbc_dft::PbcDftError>) -> Result<T, PyscfRsError> {
    r.map_err(|e| match e {
        pyscf_pbc_dft::PbcDftError::Core(x) => x,
        other => PyscfRsError::Core(CoreError::InvalidMolecule(other.to_string())),
    })
}

fn lift_tools<T>(r: Result<T, pyscf_pbc_tools::PbcToolsError>) -> Result<T, PyscfRsError> {
    r.map_err(|e| match e {
        pyscf_pbc_tools::PbcToolsError::Core(x) => x,
        #[allow(unreachable_patterns)]
        other => PyscfRsError::Core(CoreError::InvalidMolecule(other.to_string())),
    })
}

/// Sum of complex terms, reduced through [`oracle_sum`] per channel
/// (ALG-06: the only summation primitive on this path).
fn csum(terms: &[(f64, f64)]) -> (f64, f64) {
    let re: Vec<f64> = terms.iter().map(|t| t.0).collect();
    let im: Vec<f64> = terms.iter().map(|t| t.1).collect();
    (oracle_sum(&re), oracle_sum(&im))
}

/// `Re Tr(T·D)` for an F-order integral matrix `t` and a row-major density
/// `d`: `Σ_ij Re(T[i,j]·D[j,i])` — upstream's `einsum('ij,ji->', t, d).real`
/// (`krks_stress.py:312,317`).
pub fn re_trace_kdot(t_re: &[f64], t_im: &[f64], d: &CTensor, nao: usize) -> f64 {
    let mut terms = Vec::with_capacity(nao * nao);
    for i in 0..nao {
        for j in 0..nao {
            let (tr, ti) = (t_re[i + j * nao], t_im[i + j * nao]);
            let (dr, di) = (d.re[j * nao + i], d.im[j * nao + i]);
            terms.push(tr * dr - ti * di);
        }
    }
    oracle_sum(&terms)
}

/// Require Hermitian k-point densities (row-major [`CTensor`], one per k).
/// Upstream assumes Hermitian dm throughout (`krks_stress.py:221`); this is a
/// guard, not a symmetriser.
fn require_hermitian_dms(dm_kpts: &[CTensor], nao: usize, who: &str) -> Result<(), PyscfRsError> {
    for (k, d) in dm_kpts.iter().enumerate() {
        if d.re.len() != nao * nao || d.im.len() != nao * nao {
            return Err(PbcGradError::ShapeMismatch {
                expected: nao * nao,
                got: d.re.len().min(d.im.len()),
            }
            .into());
        }
        for i in 0..nao {
            for j in 0..nao {
                let (a_re, a_im) = (d.re[i * nao + j], d.im[i * nao + j]);
                let (b_re, b_im) = (d.re[j * nao + i], d.im[j * nao + i]);
                if a_re != b_re || a_im != -b_im {
                    return Err(invalid(format!(
                        "{who}: k-point {k} density is not Hermitian at ({i},{j})"
                    )));
                }
                if !a_re.is_finite() || !a_im.is_finite() {
                    return Err(invalid(format!(
                        "{who}: non-finite k-point {k} density at ({i},{j})"
                    )));
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// get_ovlp / get_kin — krks_stress.py:84-115, closed form (clause 2)
// ---------------------------------------------------------------------------

/// `get_ovlp` (`krks_stress.py:84-99`): the k-point overlap strain derivative.
///
/// Ships 18-12 Task 1's closed form at the caller's ABSOLUTE Cartesian
/// k-points (fixed fractional coordinates — clause 3), NOT the `:92-97`
/// finite difference. Returns one [`IpStrainK`] per k-point, arrangement A;
/// the FD form is the test oracle only (`tests/krks_stress.rs`).
pub fn krks_ovlp_strain(cell: &Cell, kpts: &[[f64; 3]]) -> Result<Vec<IpStrainK>, PyscfRsError> {
    if kpts.is_empty() {
        return Err(invalid(
            "krks_ovlp_strain: k-points must be explicit (pass [[0,0,0]] for the gamma-equivalent point)"
                .into(),
        ));
    }
    ip_strain_closed_form(cell, kpts, "int1e_ipovlp")
}

/// `get_kin` (`krks_stress.py:100-115`): the k-point kinetic strain
/// derivative. Same closed-form discipline as [`krks_ovlp_strain`].
pub fn krks_kin_strain(cell: &Cell, kpts: &[[f64; 3]]) -> Result<Vec<IpStrainK>, PyscfRsError> {
    if kpts.is_empty() {
        return Err(invalid(
            "krks_kin_strain: k-points must be explicit (pass [[0,0,0]] for the gamma-equivalent point)"
                .into(),
        ));
    }
    ip_strain_closed_form(cell, kpts, "int1e_ipkin")
}

// ---------------------------------------------------------------------------
// get_vxc — krks_stress.py:116-263 (LDA fused per clause 7; GGA/MGGA refused)
// ---------------------------------------------------------------------------

/// `get_vxc` (`krks_stress.py:116-263`): strain derivatives of Coulomb and XC
/// over k-point-sampled densities, as `dE/dε` (NOT divided by `vol` —
/// [`krks_stress_kernel`] reports through [`to_stress`]).
///
/// `dm_kpts[k]` is the row-major complex density at `kpts[k]` (upstream's
/// `(nkpts, nao, nao)`). LDA only (`deriv = 0`, `:138-140`); GGA/MGGA are
/// refused by name (the `deriv2` / `vtau` boundary of 18-12 §12).
///
/// Per-block/per-k contraction (`:175-184`): `c0 = dm.T·ao[0]`,
/// `rho0 += partial_dot(ao[0], c0).real`,
/// `rho1 += einsum('xyig,ig->xyg', ao1, c0.conj()).real` with the grid
/// response folded into `ao1` (`:181`). The XC functional is pointwise in the
/// k-averaged `rho0`, so the first contraction is evaluated per block and
/// accumulated into the 3×3 output through materialised per-block partials
/// reduced by [`oracle_sum`] (D-PBC-17). Full-grid survivors: `rho0`
/// (k-averaged) and `rho1[:,:,0]` (9 planes, SCALED by `2/nkpts`).
pub fn krks_get_vxc(
    cell: &Cell,
    dm_kpts: &[CTensor],
    kpts: &[[f64; 3]],
    xc_code: &str,
    opts: VxcStrainOpts,
) -> Result<[[f64; 3]; 3], PyscfRsError> {
    use pyscf_pbc_dft::xc::{RhoEff, XcType, eval_xc_eff_rks};

    if cell.low_dim_ft_type == pyscf_pbc_gto::LowDimFtType::InfVacuum {
        return Err(invalid(
            "krks_get_vxc: low_dim_ft_type = inf_vacuum is not supported (krks_stress.py:125)"
                .into(),
        ));
    }
    if cell.dimension == 1 {
        return Err(invalid(
            "krks_get_vxc: dimension = 1 is not supported (krks_stress.py:126)".into(),
        ));
    }
    let nao = cell.mol.nao_nr;
    let nkpts = kpts.len();
    if nkpts == 0 || dm_kpts.len() != nkpts {
        return Err(PbcGradError::ShapeMismatch {
            expected: nkpts,
            got: dm_kpts.len(),
        }
        .into());
    }
    require_hermitian_dms(dm_kpts, nao, "krks_get_vxc")?;
    match lift_dft(XcType::of(xc_code)) {
        Ok(XcType::Lda) => {}
        Ok(XcType::Gga) => {
            return Err(PbcGradError::NotYetImplemented {
                phase: 18,
                what: "krks_get_vxc GGA: the grid response of the gradient-density rows \
                       needs second-derivative AOs (GTOval_sph_deriv2), which the \
                       molecular kernel defers (Phase-4 scope)",
            }
            .into());
        }
        Err(e) if e.to_string().contains("meta-GGA") => {
            return Err(PbcGradError::NotYetImplemented {
                phase: 18,
                what: "krks_get_vxc meta-GGA: XcType refuses MGGA (no tau from the periodic \
                       AO evaluator) and the XC surface exposes no vtau; needs the \
                       deriv2 kernel (Phase 4) plus a tau potential",
            }
            .into());
        }
        Err(e) => return Err(e),
    }

    let grids = cell.uniform_grids(None)?;
    let mesh = grids.mesh;
    let ngrids = grids.coords.len();
    if ngrids == 0 || nao == 0 {
        return Err(PbcGradError::ShapeMismatch {
            expected: 1,
            got: 0,
        }
        .into());
    }
    let vol = cell.vol();
    let (weight_0, _) = weight_strain_derivatives(vol, ngrids)?;

    let (ao_comps, strain_comps) = strain_ao_block_comps(0).ok_or_else(|| {
        invalid("krks_get_vxc: strain block components for deriv = 0 are unavailable".into())
    })?;
    // 18-13 Task 2: the k-point block holds nkpts complex AO tables beside
    // nkpts strain tables — size against their sum in complex bytes.
    let blk = match opts.max_memory_mb {
        Some(mb) => strain_block_size_nset(ngrids, nao, ao_comps, strain_comps, 16, mb, nkpts, 1),
        None => {
            strain_block_size_nset_from_env(ngrids, nao, ao_comps, strain_comps, 16, nkpts, 1)
        }
    };
    if blk == 0 {
        return Err(PbcGradError::ShapeMismatch {
            expected: 1,
            got: 0,
        }
        .into());
    }

    let mut rho0_full = vec![0.0f64; ngrids];
    let mut rho1_full = vec![0.0f64; 9 * ngrids];
    let mut acc: [Vec<f64>; 9] = Default::default();

    let mut p0 = 0usize;
    while p0 < ngrids {
        let p1 = (p0 + blk).min(ngrids);
        let chunk = &grids.coords[p0..p1];
        let blen = p1 - p0;
        let ao = pyscf_pbc_gto::eval_ao_kpts(cell, "GTOval_sph_deriv1", chunk, kpts)?;
        if ao.comp != 4 || ao.ngrids != blen || ao.nao != nao || ao.kaos.len() != nkpts {
            return Err(PbcGradError::ShapeMismatch {
                expected: 4 * blen * nao,
                got: ao.kaos.first().map_or(0, |k| k.len()),
            }
            .into());
        }
        let strain = eval_ao_strain_derivatives(cell, chunk, kpts, 0)?;
        if strain.nkpts() != nkpts {
            return Err(PbcGradError::ShapeMismatch {
                expected: nkpts,
                got: strain.nkpts(),
            }
            .into());
        }
        // rho accumulated with += across k (`:183-184`), k-independent.
        let mut rho0_blk = vec![0.0f64; blen];
        let mut rho1_blk = vec![0.0f64; 9 * blen];
        for (k, (dm, ao_k)) in dm_kpts.iter().zip(ao.kaos.iter()).enumerate() {
            let at_re = |c: usize, g: usize, mu: usize| ao_k.re[c * blen * nao + g + mu * blen];
            let at_im = |c: usize, g: usize, mu: usize| ao_k.im[c * blen * nao + g + mu * blen];
            // c0 = dm.T·ao[0] (`:182`): transpose, NOT conjugate.
            let mut c0_re = vec![0.0f64; nao * blen];
            let mut c0_im = vec![0.0f64; nao * blen];
            for mu in 0..nao {
                for g in 0..blen {
                    let mut terms = Vec::with_capacity(nao);
                    for nu in 0..nao {
                        let (dr, di) = (dm.re[nu * nao + mu], dm.im[nu * nao + mu]);
                        let (ar, ai) = (at_re(0, g, nu), at_im(0, g, nu));
                        terms.push((dr * ar - di * ai, dr * ai + di * ar));
                    }
                    let (sr, si) = csum(&terms);
                    c0_re[mu * blen + g] = sr;
                    c0_im[mu * blen + g] = si;
                }
            }
            for g in 0..blen {
                // partial_dot(ao[0], c0).real (`:183`): conj(ao)·c0.
                let mut t0 = Vec::with_capacity(nao);
                for mu in 0..nao {
                    let (ar, ai) = (at_re(0, g, mu), at_im(0, g, mu));
                    let (cr, ci) = (c0_re[mu * blen + g], c0_im[mu * blen + g]);
                    t0.push(ar * cr + ai * ci);
                }
                rho0_blk[g] += oracle_sum(&t0);
                for x in 0..3 {
                    for y in 0..3 {
                        // einsum('xyig,ig->xyg', ao1, c0.conj()).real (`:184`).
                        let mut terms = Vec::with_capacity(nao);
                        for mu in 0..nao {
                            let (sr, si) = strain.get(k, x, y, 0, g, mu);
                            let grid_re = at_re(1 + x, g, mu) * chunk[g][y];
                            let grid_im = at_im(1 + x, g, mu) * chunk[g][y];
                            let (ar, ai) = (sr + grid_re, si + grid_im);
                            let (cr, ci) = (c0_re[mu * blen + g], c0_im[mu * blen + g]);
                            terms.push(ar * cr + ai * ci);
                        }
                        rho1_blk[(x * 3 + y) * blen + g] += oracle_sum(&terms);
                    }
                }
            }
        }
        // `:227-229` — rho0 *= 1/nkpts, rho1 *= 2/nkpts — folded into the
        // block: survivors carry the scaling, and the per-block partial with
        // it is pointwise-identical to the full-grid contraction sliced.
        let inv_nk = 1.0 / nkpts as f64;
        for (g, v) in rho0_blk.iter().enumerate() {
            rho0_full[p0 + g] += v * inv_nk;
        }
        let rho_blk = RhoEff {
            nvar: 1,
            ngrids: blen,
            data: rho0_blk.iter().map(|v| v * inv_nk).collect(),
        };
        let vxc_blk = lift_dft(eval_xc_eff_rks(xc_code, &rho_blk))?;
        let vxc = vxc_blk.row(0, 0);
        for x in 0..3 {
            for y in 0..3 {
                let mut partial = Vec::with_capacity(blen);
                for g in 0..blen {
                    let scaled = 2.0 * inv_nk * rho1_blk[(x * 3 + y) * blen + g];
                    rho1_full[(x * 3 + y) * ngrids + p0 + g] += scaled;
                    partial.push(scaled * vxc[g]);
                }
                acc[x * 3 + y].push(oracle_sum(&partial));
            }
        }
        p0 = p1;
    }

    // `:232-233` — the fused contraction times weight_0, plus the
    // scalar-on-the-diagonal exc term (clause 6).
    let mut out = [[0.0f64; 3]; 3];
    for x in 0..3 {
        for y in 0..3 {
            out[x][y] = oracle_sum(&acc[x * 3 + y]) * weight_0;
        }
    }
    let rho_full = RhoEff {
        nvar: 1,
        ngrids,
        data: rho0_full.clone(),
    };
    let xc_full = lift_dft(eval_xc_eff_rks(xc_code, &rho_full))?;
    {
        let mut terms = Vec::with_capacity(ngrids);
        for g in 0..ngrids {
            terms.push(rho0_full[g] * xc_full.exc[g]);
        }
        let scalar = oracle_sum(&terms) * weight_0;
        for x in 0..3 {
            out[x][x] = oracle_sum(&[out[x][x], scalar]);
        }
    }

    // `:235-237` — G-vectors, full-grid Coulomb kernel, `rhoG = fft(rho0)`.
    let gv = pyscf_pbc_gto::get_gv(cell, Some(mesh))?;
    let coulg0 = pyscf_pbc_gto::get_coulg(
        cell,
        pyscf_pbc_gto::CoulGArgs {
            mesh: Some(mesh),
            gv: Some(&gv),
            ..pyscf_pbc_gto::CoulGArgs::new()
        },
    )?;
    let rho_g = lift_tools(pyscf_pbc_tools::fft(
        &CTensor::from_planes(rho0_full.clone(), vec![0.0; ngrids]),
        mesh,
    ))?;

    if opts.with_j {
        // `:238-243`, identical in form to 18-12 §12.2 on the k-averaged
        // density (`rho1[:,:,0]` is already the averaged, scaled survivor).
        let cg1 = coulg_strain_derivatives(&gv, &coulg0)?;
        let mut v_r = vec![0.0f64; ngrids];
        {
            let mut re = vec![0.0f64; ngrids];
            let mut im = vec![0.0f64; ngrids];
            for g in 0..ngrids {
                re[g] = rho_g.re[g] * coulg0[g];
                im[g] = rho_g.im[g] * coulg0[g];
            }
            let back = lift_tools(pyscf_pbc_tools::ifft(&CTensor::from_planes(re, im), mesh))?;
            v_r.copy_from_slice(&back.re);
        }
        let mut exc_j = [[0.0f64; 3]; 3];
        for x in 0..3 {
            for y in 0..3 {
                let mut t1 = Vec::with_capacity(ngrids);
                for g in 0..ngrids {
                    t1.push(rho1_full[(x * 3 + y) * ngrids + g] * v_r[g]);
                }
                let plane = cg1.plane(x, y);
                let mut t3 = Vec::with_capacity(ngrids);
                for g in 0..ngrids {
                    let norm2 = oracle_sum(&[rho_g.re[g] * rho_g.re[g], rho_g.im[g] * rho_g.im[g]]);
                    t3.push(plane[g] * norm2);
                }
                exc_j[x][y] = oracle_sum(&[
                    oracle_sum(&t1) * weight_0 * 2.0,
                    oracle_sum(&t3) * (weight_0 / ngrids as f64),
                ]);
            }
        }
        let mut tj = Vec::with_capacity(ngrids);
        for g in 0..ngrids {
            tj.push(rho0_full[g] * v_r[g]);
        }
        let j_diag = oracle_sum(&tj) * weight_0;
        for x in 0..3 {
            for y in 0..3 {
                let half = 0.5 * oracle_sum(&[exc_j[x][y], if x == y { j_diag } else { 0.0 }]);
                out[x][y] = oracle_sum(&[out[x][y], half]);
            }
        }
    }

    if opts.with_nuc {
        // `:245-262`, identical in form to 18-12 §12.3 on the k-averaged
        // density; the non-local PP term is the k-point FD below.
        let has_pp = (0..cell.mol.natm).any(|ia| cell.atom_pseudo(ia).is_some());
        if has_pp {
            let vpp = vpplocg_strain_derivatives(cell, mesh)?;
            let v0 = CTensor::from_planes(vpp.v0_re.clone(), vpp.v0_im.clone());
            let vpploc_r = lift_tools(pyscf_pbc_tools::ifft(&v0, mesh))?.re;
            let pp_nl = pp_nonloc_strain_kpts(cell, dm_kpts, kpts)?;
            for x in 0..3 {
                for y in 0..3 {
                    let mut t1 = Vec::with_capacity(ngrids);
                    for g in 0..ngrids {
                        t1.push(rho1_full[(x * 3 + y) * ngrids + g] * vpploc_r[g]);
                    }
                    let mut t2 = Vec::with_capacity(ngrids);
                    for g in 0..ngrids {
                        t2.push(
                            vpp.v1_re[x * 3 + y][g] * rho_g.re[g]
                                + vpp.v1_im[x * 3 + y][g] * rho_g.im[g],
                        );
                    }
                    let ene = oracle_sum(&[
                        oracle_sum(&t1),
                        oracle_sum(&t2) / ngrids as f64,
                        pp_nl[x][y],
                    ]);
                    out[x][y] = oracle_sum(&[out[x][y], ene]);
                }
            }
        } else {
            let si = pyscf_pbc_gto::get_si(cell, Some(&gv), None, None)?;
            let charges = cell.atom_charges();
            let natm = charges.len();
            let mut zg_re = vec![0.0f64; ngrids];
            let mut zg_im = vec![0.0f64; ngrids];
            for g in 0..ngrids {
                let mut tr = Vec::with_capacity(natm);
                let mut ti = Vec::with_capacity(natm);
                for ia in 0..natm {
                    let q = -(charges[ia] as f64);
                    tr.push(q * si.re[ia * ngrids + g]);
                    ti.push(q * si.im[ia * ngrids + g]);
                }
                zg_re[g] = oracle_sum(&tr);
                zg_im[g] = oracle_sum(&ti);
            }
            let mut vr = vec![0.0f64; ngrids];
            {
                let mut re = vec![0.0f64; ngrids];
                let mut im = vec![0.0f64; ngrids];
                for g in 0..ngrids {
                    re[g] = zg_re[g] * coulg0[g];
                    im[g] = zg_im[g] * coulg0[g];
                }
                vr.copy_from_slice(
                    &lift_tools(pyscf_pbc_tools::ifft(&CTensor::from_planes(re, im), mesh))?.re,
                );
            }
            let cg1 = coulg_strain_derivatives(&gv, &coulg0)?;
            for x in 0..3 {
                for y in 0..3 {
                    let mut t1 = Vec::with_capacity(ngrids);
                    for g in 0..ngrids {
                        t1.push(rho1_full[(x * 3 + y) * ngrids + g] * vr[g]);
                    }
                    let plane = cg1.plane(x, y);
                    let mut t2 = Vec::with_capacity(ngrids);
                    for g in 0..ngrids {
                        let a = rho_g.re[g] * plane[g];
                        let b = rho_g.im[g] * plane[g];
                        t2.push(a * zg_re[g] + b * zg_im[g]);
                    }
                    let ene = oracle_sum(&[oracle_sum(&t1), oracle_sum(&t2) / ngrids as f64]);
                    out[x][y] = oracle_sum(&[out[x][y], ene]);
                }
            }
        }
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// k-point PP non-local strain — the k-point form of
// rks_stress._get_pp_nonloc_strain_derivatives(cell, mesh, dm_kpts, kpts)
// ---------------------------------------------------------------------------

/// Non-local pseudopotential energy over k-point densities:
///
/// ```text
/// E_nl = (1/nkpts) Σ_k Re Tr(dm_k · V_nl,k)
/// ```
///
/// `dm_k` is row-major, `V_nl,k` ([`get_pp_nl`](pyscf_pbc_gto::pseudo::get_pp_nl))
/// F-order. The `1/nkpts` is upstream's k-average (`eval_pp_nonloc` returns
/// `vppnl / nkpts`, up to the plane-wave-convention volume that the
/// matrix-trace energy never carries — 18-12 §8); cells with no projectors
/// yield exactly `0` (the `get_pp_nl` empty-blocks branch).
pub fn pp_nl_energy_kpts(
    cell: &Cell,
    dm_kpts: &[CTensor],
    kpts: &[[f64; 3]],
) -> Result<f64, PyscfRsError> {
    let nao = cell.mol.nao_nr;
    let nkpts = kpts.len();
    if nkpts == 0 || dm_kpts.len() != nkpts {
        return Err(PbcGradError::ShapeMismatch {
            expected: nkpts,
            got: dm_kpts.len(),
        }
        .into());
    }
    let vnl = pyscf_pbc_gto::pseudo::get_pp_nl(cell, kpts)?;
    let mut per_k = Vec::with_capacity(nkpts);
    for (dm, v) in dm_kpts.iter().zip(vnl.iter()) {
        if dm.re.len() != nao * nao || v.re.len() != nao * nao {
            return Err(PbcGradError::ShapeMismatch {
                expected: nao * nao,
                got: dm.re.len().min(v.re.len()),
            }
            .into());
        }
        let mut terms = Vec::with_capacity(nao * nao);
        for i in 0..nao {
            for j in 0..nao {
                // Re(D[i,j]·V[j,i]): dm row-major, V F-order.
                let (dr, di) = (dm.re[i * nao + j], dm.im[i * nao + j]);
                let (vr, vi) = (v.re[j + i * nao], v.im[j + i * nao]);
                terms.push(dr * vr - di * vi);
            }
        }
        per_k.push(oracle_sum(&terms));
    }
    Ok(oracle_sum(&per_k) / nkpts as f64)
}

/// The k-point `_get_pp_nonloc_strain_derivatives` (`krks_stress.py:251`,
/// via `rks_stress.py:309-385`).
///
/// Upstream finite-differences its `eval_pp_nonloc`; this port
/// finite-differences [`pp_nl_energy_kpts`] (the `get_pp_nl` contraction —
/// the same scalar up to the plane-wave-convention volume) at half step
/// `max(1e-5, sqrt(precision·0.1))` (`rks_stress.py:377`), with k-points at
/// fixed fractional coordinates ([`finite_diff_cells`], clause 3).
pub fn pp_nonloc_strain_kpts(
    cell: &Cell,
    dm_kpts: &[CTensor],
    kpts: &[[f64; 3]],
) -> Result<[[f64; 3]; 3], PyscfRsError> {
    let disp = (cell.precision * 0.1).sqrt().max(1e-5);
    if !disp.is_finite() {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "pp_nonloc k-point strain: non-finite displacement from precision {}",
            cell.precision
        ))));
    }
    let full_disp = 2.0 * disp;
    let mut out = [[0.0; 3]; 3];
    for x in 0..3 {
        for y in 0..3 {
            let pair = finite_diff_cells(cell, kpts, x, y, full_disp)?;
            let e_plus = pp_nl_energy_kpts(&pair.plus, dm_kpts, &pair.kpts_plus)?;
            let e_minus = pp_nl_energy_kpts(&pair.minus, dm_kpts, &pair.kpts_minus)?;
            out[x][y] = oracle_sum(&[e_plus, -e_minus]) / full_disp;
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// kernel — krks_stress.py:265-334; the reported units are Ha/Bohr^3
// ---------------------------------------------------------------------------

/// `kernel` (`krks_stress.py:265-334`): the asymmetric 3×3 stress tensor as a
/// PRESSURE in **Ha/Bohr³** (`σ = dE/dε / vol**, `:330`).
///
/// Upstream takes the gradient object (SCF + `make_rdm1e` inside); this port
/// takes the converged inputs directly — `dm_kpts` / `dme_kpts` (row-major
/// [`CTensor`] per k-point) — so the strained-cell SCF loop of the Gate-D
/// oracle stays in the test, never in the production path:
///
/// ```text
/// σ = ewald_strain + (1/nkpts) Σ_k [ReTr(kin_k·dm_k) − ReTr(ovlp_k·dme_k)]
///     + get_vxc(with_j, with_nuc)
/// σ /= vol
/// ```
///
/// (`:310-319`: per-k traces summed, `.real`, divided by `nkpts`).
/// `get_ovlp`/`get_kin` are the clause-2 CLOSED FORMS (not the `:304-319`
/// finite differences).
///
/// # Errors
/// * Hybrid functionals: upstream raises `NotImplementedError` (`:289-290`);
///   this port refuses by name.
/// * GGA/MGGA: refused by [`krks_get_vxc`].
pub fn krks_stress_kernel(
    cell: &Cell,
    dm_kpts: &[CTensor],
    dme_kpts: &[CTensor],
    kpts: &[[f64; 3]],
    xc_code: &str,
) -> Result<[[f64; 3]; 3], PyscfRsError> {
    use pyscf_pbc_dft::xc::is_hybrid_xc;

    let nao = cell.mol.nao_nr;
    let nkpts = kpts.len();
    if nkpts == 0 || dm_kpts.len() != nkpts || dme_kpts.len() != nkpts {
        return Err(PbcGradError::ShapeMismatch {
            expected: nkpts,
            got: dm_kpts.len().min(dme_kpts.len()),
        }
        .into());
    }
    require_hermitian_dms(dm_kpts, nao, "krks_stress_kernel")?;
    for d in dme_kpts {
        if d.re.len() != nao * nao
            || d.im.len() != nao * nao
            || d.re.iter().chain(&d.im).any(|v| !v.is_finite())
        {
            return Err(PbcGradError::ShapeMismatch {
                expected: nao * nao,
                got: d.re.len().min(d.im.len()),
            }
            .into());
        }
    }
    if lift_dft(is_hybrid_xc(xc_code))? {
        return Err(PbcGradError::NotYetImplemented {
            phase: 18,
            what: "stress tensor for hybrid DFT (krks_stress.py:289-290 raises NotImplementedError)",
        }
        .into());
    }

    let mut sigma = ewald_strain(cell)?;
    let kin = krks_kin_strain(cell, kpts)?;
    let ovlp = krks_ovlp_strain(cell, kpts)?;
    for x in 0..3 {
        for y in 0..3 {
            // One materialised partial per k-point, reduced through
            // oracle_sum, then the 1/nkpts average (`:314`, `:319`).
            let mut partials = Vec::with_capacity(nkpts);
            for (k, (dm, dme)) in dm_kpts.iter().zip(dme_kpts.iter()).enumerate() {
                let e_kin = re_trace_kdot(&kin[k].re[x * 3 + y], &kin[k].im[x * 3 + y], dm, nao);
                let e_ovlp =
                    re_trace_kdot(&ovlp[k].re[x * 3 + y], &ovlp[k].im[x * 3 + y], dme, nao);
                partials.push(oracle_sum(&[e_kin, -e_ovlp]));
            }
            sigma[x][y] = oracle_sum(&[sigma[x][y], oracle_sum(&partials) / nkpts as f64]);
        }
    }
    let vxc = krks_get_vxc(
        cell,
        dm_kpts,
        kpts,
        xc_code,
        VxcStrainOpts {
            with_j: true,
            with_nuc: true,
            max_memory_mb: None,
        },
    )?;
    for x in 0..3 {
        for y in 0..3 {
            sigma[x][y] = oracle_sum(&[sigma[x][y], vxc[x][y]]);
        }
    }
    let vol = cell.vol();
    to_stress(sigma, vol)
}

// ---------------------------------------------------------------------------
// Task 4 — DFT+U strain terms (krks_stress.py:336-403)
// ---------------------------------------------------------------------------

/// One k-point's first-order local orbitals: the `(3, 3, nao, nlo)` strain
/// derivative of the Löwdin MINAO projector set, complex planes.
///
/// Per-k layout `((x*3+y)*nao + i)*nlo + j` (row-major in `(i, j)`).
#[derive(Debug, Clone)]
pub struct LocalOrbStrain {
    /// AO count.
    pub nao: usize,
    /// Local-orbital count.
    pub nlo: usize,
    /// Real planes, one per k-point.
    pub re: Vec<Vec<f64>>,
    /// Imaginary planes, one per k-point.
    pub im: Vec<Vec<f64>>,
}

impl LocalOrbStrain {
    /// Number of k-points.
    pub fn nkpts(&self) -> usize {
        self.re.len()
    }

    /// `(re, im)` of strain `(x, y)`, AO `i`, local orbital `j`, at k-point `k`.
    pub fn get(&self, k: usize, x: usize, y: usize, i: usize, j: usize) -> (f64, f64) {
        let p = ((x * 3 + y) * self.nao + i) * self.nlo + j;
        (self.re[k][p], self.im[k][p])
    }
}

/// `_get_first_order_local_orbitals` (`krks_stress.py:336-361`): the STRAIN
/// derivatives of the local orbitals — 18-08 Task 1 built the
/// NUCLEAR-COORDINATE ones; same Löwdin orthogonalisation, a different
/// perturbation, and the names must not be confusable.
///
/// Upstream's own form is a central difference of `_make_minao_lo` at half
/// step `1e-5` (`:351-360`) — the FD here IS the port, not an oracle
/// substitution — with k-points at fixed fractional coordinates (clause 3)
/// and the MINAO reference rebuilt on each displaced cell (the reference
/// atoms strain with the cell, exactly as upstream's strained `pcell1`/`pcell2`
/// at `:355`).
///
/// `_set_U` / `_make_minao_lo` / `reference_mol` are called from
/// `pyscf_pbc_dft::kspu`, never reimplemented.
pub fn first_order_local_orbitals(
    cell: &Cell,
    minao_ref: &str,
    kpts: &[[f64; 3]],
) -> Result<LocalOrbStrain, PyscfRsError> {
    use pyscf_pbc_dft::kspu::{make_minao_lo, reference_cell};

    let nkpts = kpts.len();
    if nkpts == 0 {
        return Err(PbcGradError::ShapeMismatch {
            expected: 1,
            got: 0,
        }
        .into());
    }
    let nao = cell.mol.nao_nr;
    let pcell = lift_dft(reference_cell(cell, minao_ref))?;
    let c0 = lift_dft(make_minao_lo(cell, &pcell, kpts))?;
    let nlo = c0
        .first()
        .map(|c| {
            if nao == 0 {
                0
            } else {
                c.re.len() / nao
            }
        })
        .unwrap_or(0);
    if nlo == 0 {
        return Err(PbcGradError::ShapeMismatch {
            expected: 1,
            got: 0,
        }
        .into());
    }

    // Upstream's half step (`krks_stress.py:351`); `finite_diff_cells` takes
    // the FULL separation.
    const FULL_DISP: f64 = 2e-5;
    let mut re = vec![vec![0.0; 9 * nao * nlo]; nkpts];
    let mut im = vec![vec![0.0; 9 * nao * nlo]; nkpts];
    for x in 0..3 {
        for y in 0..3 {
            let pair = finite_diff_cells(cell, kpts, x, y, FULL_DISP)?;
            let pcell_plus = lift_dft(reference_cell(&pair.plus, minao_ref))?;
            let pcell_minus = lift_dft(reference_cell(&pair.minus, minao_ref))?;
            let c_plus = lift_dft(make_minao_lo(&pair.plus, &pcell_plus, &pair.kpts_plus))?;
            let c_minus = lift_dft(make_minao_lo(&pair.minus, &pcell_minus, &pair.kpts_minus))?;
            for ((cp, cm), (r, m)) in c_plus
                .iter()
                .zip(c_minus.iter())
                .zip(re.iter_mut().zip(im.iter_mut()))
            {
                if cp.re.len() != nao * nlo || cm.re.len() != nao * nlo {
                    return Err(PbcGradError::ShapeMismatch {
                        expected: nao * nlo,
                        got: cp.re.len().min(cm.re.len()),
                    }
                    .into());
                }
                // `make_minao_lo` returns COLUMN-MAJOR `nao × nlo`
                // (`C[i,j]` at `i + j*nao`); the strain table is row-major
                // in `(i, j)`.
                for i in 0..nao {
                    for j in 0..nlo {
                        let p = ((x * 3 + y) * nao + i) * nlo + j;
                        let q = i + j * nao;
                        r[p] = oracle_sum(&[cp.re[q], -cm.re[q]]) / FULL_DISP;
                        m[p] = oracle_sum(&[cp.im[q], -cm.im[q]]) / FULL_DISP;
                    }
                }
            }
        }
    }
    Ok(LocalOrbStrain { nao, nlo, re, im })
}

/// `_hubbard_U_deriv1` (`krks_stress.py:363-403`): the DFT+U strain term for
/// a closed-shell k-point density.
///
/// Asserts, like upstream (`:364-366`): no linear-response perturbation
/// (`alpha`), no caller-supplied local orbitals (always the Löwdin MINAO
/// set), and a named MINAO reference. `dm_kpts[k]` is the row-major complex
/// density; the `1/nkpts` weighting is `:391`.
///
/// The E_U oracle for the gate (`test_krks_stress.py:373 `test_hubbard_U`)
/// lives in `tests/krks_stress.rs`: a central difference of
/// `kspu::add_vhubbard`'s `E_U` over strained cells at fixed fractional
/// k-points — an independent code path (energy vs analytic derivative), not
/// the same algebra re-run.
pub fn hubbard_u_deriv1(
    cell: &Cell,
    dm_kpts: &[CTensor],
    kpts: &[[f64; 3]],
    cfg: &pyscf_pbc_dft::kspu::HubbardU,
) -> Result<[[f64; 3]; 3], PyscfRsError> {
    use pyscf_pbc_dft::kspu::{make_minao_lo, reference_cell, set_u};

    let nao = cell.mol.nao_nr;
    let nkpts = kpts.len();
    if nkpts == 0 || dm_kpts.len() != nkpts {
        return Err(PbcGradError::ShapeMismatch {
            expected: nkpts,
            got: dm_kpts.len(),
        }
        .into());
    }
    require_hermitian_dms(dm_kpts, nao, "hubbard_u_deriv1")?;
    if !cfg.alpha.is_empty() {
        return Err(invalid(
            "hubbard_u_deriv1: linear-response alpha is not supported (krks_stress.py:364 asserts mf.alpha is None)".into(),
        ));
    }
    if cfg.c_ao_lo.is_some() {
        return Err(invalid(
            "hubbard_u_deriv1: caller-supplied C_ao_lo is not supported (krks_stress.py:365 asserts mf.C_ao_lo is None)".into(),
        ));
    }
    if cfg.minao_ref.is_empty() {
        return Err(invalid(
            "hubbard_u_deriv1: a named minao_ref is required (krks_stress.py:366)".into(),
        ));
    }

    let pcell = lift_dft(reference_cell(cell, &cfg.minao_ref))?;
    let resolved = lift_dft(set_u(&pcell, cfg))?;
    if resolved.indices.is_empty() {
        return Ok([[0.0; 3]; 3]);
    }
    let stack: Vec<usize> = resolved.indices.iter().flatten().copied().collect();
    let nu = stack.len();
    let c_lo = lift_dft(make_minao_lo(cell, &pcell, kpts))?;
    let nlo = c_lo[0].re.len() / nao;
    let c1 = first_order_local_orbitals(cell, &cfg.minao_ref, kpts)?;
    if c1.nlo != nlo {
        return Err(PbcGradError::ShapeMismatch {
            expected: nlo,
            got: c1.nlo,
        }
        .into());
    }
    let ovlp0 = pyscf_pbc_gto::get_ovlp(cell, kpts)?;
    let ovlp1 = krks_ovlp_strain(cell, kpts)?;

    // Column-major access to C0's stacked LO columns: C0u[i,a].
    let c0u = |ck: &CTensor, i: usize, a: usize| -> (f64, f64) {
        let q = i + stack[a] * nao;
        (ck.re[q], ck.im[q])
    };
    // F-order overlap: S[p,q] at p + q*nao.
    let s0 = |sk: &CTensor, p: usize, q: usize| -> (f64, f64) {
        (sk.re[p + q * nao], sk.im[p + q * nao])
    };

    // One materialised sigma partial per k-point, reduced at the end.
    let mut sigma_k: Vec<[[f64; 3]; 3]> = Vec::with_capacity(nkpts);
    for (k, dm) in dm_kpts.iter().enumerate() {
        // C_inv = C0^H · S0 (`:387`), (nu × nao).
        let mut cinv_re = vec![0.0; nu * nao];
        let mut cinv_im = vec![0.0; nu * nao];
        for a in 0..nu {
            for b in 0..nao {
                let mut terms = Vec::with_capacity(nao);
                for i in 0..nao {
                    let (cr, ci) = c0u(&c_lo[k], i, a);
                    let (sr, si) = s0(&ovlp0[k], i, b);
                    // conj(C)·S.
                    terms.push((cr * sr + ci * si, cr * si - ci * sr));
                }
                let (sr, si) = csum(&terms);
                cinv_re[a * nao + b] = sr;
                cinv_im[a * nao + b] = si;
            }
        }
        // T = C_inv · dm (`:388` inner factor), (nu × nao).
        let mut t_re = vec![0.0; nu * nao];
        let mut t_im = vec![0.0; nu * nao];
        for a in 0..nu {
            for j in 0..nao {
                let mut terms = Vec::with_capacity(nao);
                for b in 0..nao {
                    let (cr, ci) = (cinv_re[a * nao + b], cinv_im[a * nao + b]);
                    let (dr, di) = (dm.re[b * nao + j], dm.im[b * nao + j]);
                    terms.push((cr * dr - ci * di, cr * di + ci * dr));
                }
                let (sr, si) = csum(&terms);
                t_re[a * nao + j] = sr;
                t_im[a * nao + j] = si;
            }
        }
        // dm_deriv0 = T · C_inv^H (`:388`), (nu × nu).
        let mut p0_re = vec![0.0; nu * nu];
        let mut p0_im = vec![0.0; nu * nu];
        for a in 0..nu {
            for b in 0..nu {
                let mut terms = Vec::with_capacity(nao);
                for j in 0..nao {
                    let (tr, ti) = (t_re[a * nao + j], t_im[a * nao + j]);
                    let (cr, ci) = (cinv_re[b * nao + j], cinv_im[b * nao + j]);
                    // T·conj(C_inv).
                    terms.push((tr * cr + ti * ci, ti * cr - tr * ci));
                }
                let (sr, si) = csum(&terms);
                p0_re[a * nu + b] = sr;
                p0_im[a * nu + b] = si;
            }
        }
        let mut sig = [[0.0; 3]; 3];
        for x in 0..3 {
            for y in 0..3 {
                // SC1 = S0·C1 + S1·C0 (`:393-394`), (nao × nu).
                // dm_deriv1 = T·SC1 (`:395`), (nu × nu).
                let mut p1_re = vec![0.0; nu * nu];
                let mut p1_im = vec![0.0; nu * nu];
                for a in 0..nu {
                    for b in 0..nu {
                        let mut terms = Vec::with_capacity(nao);
                        for p in 0..nao {
                            // SC1[p,b].
                            let mut sc = Vec::with_capacity(2 * nao);
                            for q in 0..nao {
                                let (sr, si) = s0(&ovlp0[k], p, q);
                                let (cr, ci) = c1.get(k, x, y, q, stack[b]);
                                sc.push((sr * cr - si * ci, sr * ci + si * cr));
                                let (tr, ti) = ovlp1[k].get(x, y, p, q, nao);
                                let (ur, ui) = c0u(&c_lo[k], q, b);
                                sc.push((tr * ur - ti * ui, tr * ui + ti * ur));
                            }
                            let (scr, sci) = csum(&sc);
                            let (tr, ti) = (t_re[a * nao + p], t_im[a * nao + p]);
                            terms.push((tr * scr - ti * sci, tr * sci + ti * scr));
                        }
                        let (sr, si) = csum(&terms);
                        p1_re[a * nu + b] = sr;
                        p1_im[a * nu + b] = si;
                    }
                }
                // `:397-403` over the site blocks.
                let mut off = 0usize;
                let mut acc = Vec::new();
                for (idx, val) in resolved.indices.iter().zip(resolved.u_val.iter()) {
                    let n = idx.len();
                    let mut t_diag = Vec::with_capacity(n);
                    let mut t_prod = Vec::with_capacity(n * n);
                    for ai in 0..n {
                        let p = off + ai;
                        t_diag.push(p1_re[p * nu + p]);
                        for bi in 0..n {
                            let q = off + bi;
                            // Re(P1[p,q]·P0[q,p]).
                            let (ar, ai_) = (p1_re[p * nu + q], p1_im[p * nu + q]);
                            let (br, bi_) = (p0_re[q * nu + p], p0_im[q * nu + p]);
                            t_prod.push(ar * br - ai_ * bi_);
                        }
                    }
                    let w = val * 0.5 / nkpts as f64;
                    acc.push(w * oracle_sum(&[oracle_sum(&t_diag) * 2.0, -oracle_sum(&t_prod) * 2.0]));
                    off += n;
                }
                sig[x][y] = oracle_sum(&acc);
            }
        }
        sigma_k.push(sig);
    }
    let mut sigma = [[0.0; 3]; 3];
    for x in 0..3 {
        for y in 0..3 {
            let terms: Vec<f64> = sigma_k.iter().map(|s| s[x][y]).collect();
            sigma[x][y] = oracle_sum(&terms);
        }
    }
    Ok(sigma)
}
