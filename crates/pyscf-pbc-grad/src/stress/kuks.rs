//! Plan 18-13 Task 3 — `pyscf/pbc/grad/kuks_stress.py` (308 l), both indices.
//!
//! # Imports with the upstream addresses
//!
//! Note the import line (`kuks_stress.py:24`):
//!
//! ```python
//! from pyscf.pbc.grad.krks_stress import get_ovlp, _get_first_order_local_orbitals
//! ```
//!
//! — `get_ovlp` comes from **KRKS** (the k-point one), not from `rks_stress`,
//! and so does the first-order local-orbital helper. This module does the
//! same: [`krks_ovlp_strain`](super::krks::krks_ovlp_strain),
//! [`first_order_local_orbitals`](super::krks::first_order_local_orbitals) and
//! the k-point PP helper
//! [`pp_nonloc_strain_kpts`](super::krks::pp_nonloc_strain_kpts) come from
//! [`super::krks`]. Getting that import wrong substitutes a gamma overlap
//! into a k-point stress — correct at `nkpts = 1`, wrong everywhere else —
//! and the `nkpts = 1` k-point-vs-gamma agreement gate in
//! `tests/kuks_stress.rs` is the cheapest test that catches it.
//!
//! The other six shared symbols come from [`super::rks`] (via
//! `kuks_stress.py:26-34`):
//! [`finite_diff_cells`](super::rks::finite_diff_cells) (re-exported for the
//! FD oracles; the production path builds no strained cells itself),
//! [`weight_strain_derivatives`](super::rks::weight_strain_derivatives),
//! [`coulg_strain_derivatives`](super::rks::coulg_strain_derivatives),
//! [`eval_ao_strain_derivatives`](super::rks::eval_ao_strain_derivatives),
//! [`vpplocg_strain_derivatives`](super::rks::vpplocg_strain_derivatives) and
//! [`ewald_strain`](super::rks::ewald_strain).
//!
//! # What differs from KRKS
//!
//! Only `get_vxc` (`:36`) and `kernel` (`:192`): `rho0`/`rho1` grow the spin
//! axis (`(2, nvar, ngrids)`, `(3,3,2,nvar,ngrids)` at `:80-81`) and
//! `eval_xc_eff` is called with `spin = 1` (`:155`).
//!
//! # Reported units
//!
//! Ha/Bohr³ through [`rks::to_stress`](super::rks::to_stress), matching 18-12.
//!
//! # GGA/MGGA
//!
//! Refused by name (missing `deriv2` AO kernel / `vtau`), as in KRKS.

use pyscf_algebra::{CTensor, oracle_sum};
use pyscf_core::{CoreError, PyscfRsError};
use pyscf_pbc_gto::Cell;

use super::krks::{
    first_order_local_orbitals, krks_kin_strain, krks_ovlp_strain, pp_nonloc_strain_kpts,
    re_trace_kdot,
};
use super::rks::{
    VxcStrainOpts, coulg_strain_derivatives, eval_ao_strain_derivatives, ewald_strain,
    strain_ao_block_comps, strain_block_size_nset, strain_block_size_nset_from_env, to_stress,
    vpplocg_strain_derivatives, weight_strain_derivatives,
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

/// Complex `c0 +=` helper used inside the fused loop.
fn csum(terms: &[(f64, f64)]) -> (f64, f64) {
    let re: Vec<f64> = terms.iter().map(|t| t.0).collect();
    let im: Vec<f64> = terms.iter().map(|t| t.1).collect();
    (oracle_sum(&re), oracle_sum(&im))
}

/// Require Hermitian spin×k densities in flat spin-major order
/// (`dm[(s*nkpts + k)]`, upstream's `(2, nkpts, nao, nao)`).
fn require_spin_k_dms(
    dm: &[CTensor],
    nkpts: usize,
    nao: usize,
    who: &str,
) -> Result<(), PyscfRsError> {
    if dm.len() != 2 * nkpts {
        return Err(PbcGradError::ShapeMismatch {
            expected: 2 * nkpts,
            got: dm.len(),
        }
        .into());
    }
    for (b, d) in dm.iter().enumerate() {
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
                        "{who}: spin-k block {b} is not Hermitian at ({i},{j})"
                    )));
                }
                if !a_re.is_finite() || !a_im.is_finite() {
                    return Err(invalid(format!(
                        "{who}: non-finite spin-k block {b} at ({i},{j})"
                    )));
                }
            }
        }
    }
    Ok(())
}

/// `get_vxc` (`kuks_stress.py:36-190`): strain derivatives of Coulomb and XC
/// over spin- and k-resolved densities, as `dE/dε` (NOT divided by `vol` —
/// [`kuks_stress_kernel`] reports through [`to_stress`]).
///
/// `dm_spin` is flat spin-major (`dm_spin[s*nkpts + k]`, row-major complex,
/// upstream's `(2, nkpts, nao, nao)`). LDA only; GGA/MGGA refused by name.
///
/// Per-block/per-k/per-spin contraction (`:94-105`): `c0 = dm[s].T·ao[0]`,
/// `rho0[s] += partial_dot`, `rho1[:,:,s] += einsum(ao1, c0.conj())` with the
/// grid response folded into `ao1` (`:101`). Averaging `:151-153`
/// (`rho0 *= 1/nkpts`, `rho1 *= 2/nkpts`) is folded into the block; the XC
/// potential is pointwise (`spin = 1`, `:155`), so per-block eval == full-grid
/// eval sliced. Full-grid survivors: `rho0` per spin and `rho1[:,:,:,0]` per
/// spin (SCALED).
pub fn kuks_get_vxc(
    cell: &Cell,
    dm_spin: &[CTensor],
    kpts: &[[f64; 3]],
    xc_code: &str,
    opts: VxcStrainOpts,
) -> Result<[[f64; 3]; 3], PyscfRsError> {
    use pyscf_pbc_dft::xc::{RhoEff, XcType, eval_xc_eff_uks};

    if cell.low_dim_ft_type == pyscf_pbc_gto::LowDimFtType::InfVacuum {
        return Err(invalid(
            "kuks_get_vxc: low_dim_ft_type = inf_vacuum is not supported (kuks_stress.py:45)"
                .into(),
        ));
    }
    if cell.dimension == 1 {
        return Err(invalid(
            "kuks_get_vxc: dimension = 1 is not supported (kuks_stress.py:46)".into(),
        ));
    }
    let nao = cell.mol.nao_nr;
    let nkpts = kpts.len();
    if nkpts == 0 {
        return Err(PbcGradError::ShapeMismatch {
            expected: 1,
            got: 0,
        }
        .into());
    }
    require_spin_k_dms(dm_spin, nkpts, nao, "kuks_get_vxc")?;
    match lift_dft(XcType::of(xc_code)) {
        Ok(XcType::Lda) => {}
        Ok(XcType::Gga) => {
            return Err(PbcGradError::NotYetImplemented {
                phase: 18,
                what: "kuks_get_vxc GGA: the grid response of the gradient-density rows \
                       needs second-derivative AOs (GTOval_sph_deriv2), which the \
                       molecular kernel defers (Phase-4 scope)",
            }
            .into());
        }
        Err(e) if e.to_string().contains("meta-GGA") => {
            return Err(PbcGradError::NotYetImplemented {
                phase: 18,
                what: "kuks_get_vxc meta-GGA: XcType refuses MGGA (no tau from the periodic \
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
        invalid("kuks_get_vxc: strain block components for deriv = 0 are unavailable".into())
    })?;
    let blk = match opts.max_memory_mb {
        Some(mb) => strain_block_size_nset(ngrids, nao, ao_comps, strain_comps, 16, mb, nkpts, 2),
        None => {
            strain_block_size_nset_from_env(ngrids, nao, ao_comps, strain_comps, 16, nkpts, 2)
        }
    };
    if blk == 0 {
        return Err(PbcGradError::ShapeMismatch {
            expected: 1,
            got: 0,
        }
        .into());
    }

    let mut rho0_full = vec![0.0f64; 2 * ngrids];
    let mut rho1_full = vec![0.0f64; 2 * 9 * ngrids];
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
        let mut rho0_blk = vec![0.0f64; 2 * blen];
        let mut rho1_blk = vec![0.0f64; 2 * 9 * blen];
        for k in 0..nkpts {
            let ao_k = &ao.kaos[k];
            let at_re = |c: usize, g: usize, mu: usize| ao_k.re[c * blen * nao + g + mu * blen];
            let at_im = |c: usize, g: usize, mu: usize| ao_k.im[c * blen * nao + g + mu * blen];
            for s in 0..2 {
                let dm = &dm_spin[s * nkpts + k];
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
                    let mut t0 = Vec::with_capacity(nao);
                    for mu in 0..nao {
                        let (ar, ai) = (at_re(0, g, mu), at_im(0, g, mu));
                        let (cr, ci) = (c0_re[mu * blen + g], c0_im[mu * blen + g]);
                        t0.push(ar * cr + ai * ci);
                    }
                    rho0_blk[s * blen + g] += oracle_sum(&t0);
                    for x in 0..3 {
                        for y in 0..3 {
                            let mut terms = Vec::with_capacity(nao);
                            for mu in 0..nao {
                                let (sr, si) = strain.get(k, x, y, 0, g, mu);
                                let grid_re = at_re(1 + x, g, mu) * chunk[g][y];
                                let grid_im = at_im(1 + x, g, mu) * chunk[g][y];
                                let (ar, ai) = (sr + grid_re, si + grid_im);
                                let (cr, ci) = (c0_re[mu * blen + g], c0_im[mu * blen + g]);
                                terms.push(ar * cr + ai * ci);
                            }
                            rho1_blk[(s * 9 + x * 3 + y) * blen + g] += oracle_sum(&terms);
                        }
                    }
                }
            }
        }
        let inv_nk = 1.0 / nkpts as f64;
        for s in 0..2 {
            for (g, v) in rho0_blk[s * blen..(s + 1) * blen].iter().enumerate() {
                rho0_full[s * ngrids + p0 + g] += v * inv_nk;
            }
        }
        let rho_a = RhoEff {
            nvar: 1,
            ngrids: blen,
            data: rho0_blk[..blen].iter().map(|v| v * inv_nk).collect(),
        };
        let rho_b = RhoEff {
            nvar: 1,
            ngrids: blen,
            data: rho0_blk[blen..2 * blen].iter().map(|v| v * inv_nk).collect(),
        };
        let vxc_blk = lift_dft(eval_xc_eff_uks(xc_code, &rho_a, &rho_b))?;
        for s in 0..2 {
            let vxc = vxc_blk.row(s, 0);
            for x in 0..3 {
                for y in 0..3 {
                    let mut partial = Vec::with_capacity(blen);
                    for g in 0..blen {
                        let scaled = 2.0 * inv_nk * rho1_blk[(s * 9 + x * 3 + y) * blen + g];
                        rho1_full[(s * 9 + x * 3 + y) * ngrids + p0 + g] += scaled;
                        partial.push(scaled * vxc[g]);
                    }
                    acc[x * 3 + y].push(oracle_sum(&partial));
                }
            }
        }
        p0 = p1;
    }

    // `:156-159` — the fused contraction times weight_0, plus the
    // scalar-on-the-diagonal exc term over the spin-summed rho0.
    let mut out = [[0.0f64; 3]; 3];
    for x in 0..3 {
        for y in 0..3 {
            out[x][y] = oracle_sum(&acc[x * 3 + y]) * weight_0;
        }
    }
    let rho_a_full = RhoEff {
        nvar: 1,
        ngrids,
        data: rho0_full[..ngrids].to_vec(),
    };
    let rho_b_full = RhoEff {
        nvar: 1,
        ngrids,
        data: rho0_full[ngrids..2 * ngrids].to_vec(),
    };
    let xc_full = lift_dft(eval_xc_eff_uks(xc_code, &rho_a_full, &rho_b_full))?;
    {
        let mut terms = Vec::with_capacity(ngrids);
        for g in 0..ngrids {
            terms.push(oracle_sum(&[rho0_full[g], rho0_full[ngrids + g]]) * xc_full.exc[g]);
        }
        let scalar = oracle_sum(&terms) * weight_0;
        for x in 0..3 {
            out[x][x] = oracle_sum(&[out[x][x], scalar]);
        }
    }

    // `:161-163` — spin-summed survivors for the Coulomb terms.
    let mut rho_tot = vec![0.0f64; ngrids];
    for g in 0..ngrids {
        rho_tot[g] = oracle_sum(&[rho0_full[g], rho0_full[ngrids + g]]);
    }
    let mut rho1_tot = vec![0.0f64; 9 * ngrids];
    for c in 0..9 {
        for g in 0..ngrids {
            rho1_tot[c * ngrids + g] =
                oracle_sum(&[rho1_full[c * ngrids + g], rho1_full[(9 + c) * ngrids + g]]);
        }
    }
    // Spin-summed k-point densities for the PP trace (`dm_kpts.sum(axis=0)`,
    // `:177-178`).
    let mut dm_sum: Vec<CTensor> = Vec::with_capacity(nkpts);
    for k in 0..nkpts {
        let mut re = vec![0.0; nao * nao];
        let mut im = vec![0.0; nao * nao];
        for p in 0..nao * nao {
            re[p] = oracle_sum(&[dm_spin[k].re[p], dm_spin[nkpts + k].re[p]]);
            im[p] = oracle_sum(&[dm_spin[k].im[p], dm_spin[nkpts + k].im[p]]);
        }
        dm_sum.push(CTensor::from_planes(re, im));
    }

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
        &CTensor::from_planes(rho_tot.clone(), vec![0.0; ngrids]),
        mesh,
    ))?;

    if opts.with_j {
        // `:164-169`, identical in form to KRKS on the summed density.
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
                    t1.push(rho1_tot[(x * 3 + y) * ngrids + g] * v_r[g]);
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
            tj.push(rho_tot[g] * v_r[g]);
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
        // `:171-189` on the summed density and the spin-summed `dm`.
        let has_pp = (0..cell.mol.natm).any(|ia| cell.atom_pseudo(ia).is_some());
        if has_pp {
            let vpp = vpplocg_strain_derivatives(cell, mesh)?;
            let v0 = CTensor::from_planes(vpp.v0_re.clone(), vpp.v0_im.clone());
            let vpploc_r = lift_tools(pyscf_pbc_tools::ifft(&v0, mesh))?.re;
            let pp_nl = pp_nonloc_strain_kpts(cell, &dm_sum, kpts)?;
            for x in 0..3 {
                for y in 0..3 {
                    let mut t1 = Vec::with_capacity(ngrids);
                    for g in 0..ngrids {
                        t1.push(rho1_tot[(x * 3 + y) * ngrids + g] * vpploc_r[g]);
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
                        t1.push(rho1_tot[(x * 3 + y) * ngrids + g] * vr[g]);
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

/// `kernel` (`kuks_stress.py:192-260`): the asymmetric 3×3 stress tensor as a
/// PRESSURE in **Ha/Bohr³** (`σ = dE/dε / vol**, `:257`).
///
/// `dm_spin` is flat spin-major (`dm_spin[s*nkpts + k]`,
/// `mf.make_rdm1()` at `:249`); `dme_sum` is the SPIN-SUMMED
/// energy-weighted density per k-point (`mf_grad.make_rdm1e().sum(axis=0)` at
/// `:225`). The strained-cell SCF loop of the Gate-D oracle stays in the
/// test, never in the production path:
///
/// ```text
/// σ = ewald_strain + (1/nkpts) Σ_k [ReTr(kin_k·dm0_k) − ReTr(ovlp_k·dme_k)]
///     + get_vxc(with_j, with_nuc)
/// σ /= vol
/// ```
///
/// with `dm0 = dm_spin.sum(axis=0)` (`:224`). `get_ovlp`/`get_kin` are the
/// clause-2 CLOSED FORMS imported from KRKS (not gamma overlaps).
///
/// # Errors
/// * Hybrid functionals: upstream raises `NotImplementedError` (`:216-217`);
///   this port refuses by name.
/// * GGA/MGGA: refused by [`kuks_get_vxc`].
pub fn kuks_stress_kernel(
    cell: &Cell,
    dm_spin: &[CTensor],
    dme_sum: &[CTensor],
    kpts: &[[f64; 3]],
    xc_code: &str,
) -> Result<[[f64; 3]; 3], PyscfRsError> {
    use pyscf_pbc_dft::xc::is_hybrid_xc;

    let nao = cell.mol.nao_nr;
    let nkpts = kpts.len();
    if nkpts == 0 || dme_sum.len() != nkpts {
        return Err(PbcGradError::ShapeMismatch {
            expected: nkpts,
            got: dme_sum.len(),
        }
        .into());
    }
    require_spin_k_dms(dm_spin, nkpts, nao, "kuks_stress_kernel")?;
    for d in dme_sum {
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
            what: "stress tensor for hybrid DFT (kuks_stress.py:216-217 raises NotImplementedError)",
        }
        .into());
    }

    // Spin-summed densities for the hcore traces (`:224`).
    let mut dm0: Vec<CTensor> = Vec::with_capacity(nkpts);
    for k in 0..nkpts {
        let mut re = vec![0.0; nao * nao];
        let mut im = vec![0.0; nao * nao];
        for p in 0..nao * nao {
            re[p] = oracle_sum(&[dm_spin[k].re[p], dm_spin[nkpts + k].re[p]]);
            im[p] = oracle_sum(&[dm_spin[k].im[p], dm_spin[nkpts + k].im[p]]);
        }
        dm0.push(CTensor::from_planes(re, im));
    }

    let mut sigma = ewald_strain(cell)?;
    let kin = krks_kin_strain(cell, kpts)?;
    let ovlp = krks_ovlp_strain(cell, kpts)?;
    for x in 0..3 {
        for y in 0..3 {
            let mut partials = Vec::with_capacity(nkpts);
            for (k, (dm, dme)) in dm0.iter().zip(dme_sum.iter()).enumerate() {
                let e_kin = re_trace_kdot(&kin[k].re[x * 3 + y], &kin[k].im[x * 3 + y], dm, nao);
                let e_ovlp =
                    re_trace_kdot(&ovlp[k].re[x * 3 + y], &ovlp[k].im[x * 3 + y], dme, nao);
                partials.push(oracle_sum(&[e_kin, -e_ovlp]));
            }
            sigma[x][y] = oracle_sum(&[sigma[x][y], oracle_sum(&partials) / nkpts as f64]);
        }
    }
    let vxc = kuks_get_vxc(
        cell,
        dm_spin,
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

/// `_hubbard_U_deriv1` (`kuks_stress.py:263-307`): the DFT+U strain term for
/// a spin-resolved k-point density.
///
/// Same spin-independent projectors as the KRKS form
/// ([`first_order_local_orbitals`], [`krks_ovlp_strain`] — both from KRKS),
/// applied per spin channel (`:288-307`): `dm_deriv0`/`dm_deriv1` per spin,
/// the diagonal term doubled and the product term QUADRUPLED (`:305-307`
/// carry `* 2` and `* 4`; the KRKS form carries `* 2` and `* 2`).
///
/// The E_U oracle for the gate (`test_kuks_stress.py:194 `test_hubbard_U`)
/// lives in `tests/kuks_stress_hubbard.rs`: a central difference of the
/// two-channel `kspu::add_vhubbard` `E_U` (`kukspu.py:96`, restored by
/// 20-13-FIX) over strained cells at fixed fractional k-points.
pub fn hubbard_u_deriv1_uks(
    cell: &Cell,
    dm_spin: &[CTensor],
    kpts: &[[f64; 3]],
    cfg: &pyscf_pbc_dft::kspu::HubbardU,
) -> Result<[[f64; 3]; 3], PyscfRsError> {
    use pyscf_pbc_dft::kspu::{make_minao_lo, reference_cell, set_u};

    let nao = cell.mol.nao_nr;
    let nkpts = kpts.len();
    if nkpts == 0 {
        return Err(PbcGradError::ShapeMismatch {
            expected: 1,
            got: 0,
        }
        .into());
    }
    require_spin_k_dms(dm_spin, nkpts, nao, "hubbard_u_deriv1_uks")?;
    if !cfg.alpha.is_empty() {
        return Err(invalid(
            "hubbard_u_deriv1_uks: linear-response alpha is not supported (kuks_stress.py:264 asserts mf.alpha is None)".into(),
        ));
    }
    if cfg.c_ao_lo.is_some() {
        return Err(invalid(
            "hubbard_u_deriv1_uks: caller-supplied C_ao_lo is not supported (kuks_stress.py:265 asserts mf.C_ao_lo is None)".into(),
        ));
    }
    if cfg.minao_ref.is_empty() {
        return Err(invalid(
            "hubbard_u_deriv1_uks: a named minao_ref is required (kuks_stress.py:266)".into(),
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

    let c0u = |ck: &CTensor, i: usize, a: usize| -> (f64, f64) {
        let q = i + stack[a] * nao;
        (ck.re[q], ck.im[q])
    };
    let s0 = |sk: &CTensor, p: usize, q: usize| -> (f64, f64) {
        (sk.re[p + q * nao], sk.im[p + q * nao])
    };

    let mut sigma_k: Vec<[[f64; 3]; 3]> = Vec::with_capacity(nkpts);
    for k in 0..nkpts {
        // C_inv = C0^H · S0 (spin-independent), (nu × nao).
        let mut cinv_re = vec![0.0; nu * nao];
        let mut cinv_im = vec![0.0; nu * nao];
        for a in 0..nu {
            for b in 0..nao {
                let mut terms = Vec::with_capacity(nao);
                for i in 0..nao {
                    let (cr, ci) = c0u(&c_lo[k], i, a);
                    let (sr, si) = s0(&ovlp0[k], i, b);
                    terms.push((cr * sr + ci * si, cr * si - ci * sr));
                }
                let (sr, si) = csum(&terms);
                cinv_re[a * nao + b] = sr;
                cinv_im[a * nao + b] = si;
            }
        }
        // Per-spin T = C_inv · dm and dm_deriv0 = T · C_inv^H
        // (`:288-291` loops `dm_s in dm`).
        let mut t_spin = Vec::with_capacity(2);
        let mut p0_spin = Vec::with_capacity(2);
        for s in 0..2 {
            let dm = &dm_spin[s * nkpts + k];
            let mut tr = vec![0.0; nu * nao];
            let mut ti = vec![0.0; nu * nao];
            for a in 0..nu {
                for j in 0..nao {
                    let mut terms = Vec::with_capacity(nao);
                    for b in 0..nao {
                        let (cr, ci) = (cinv_re[a * nao + b], cinv_im[a * nao + b]);
                        let (dr, di) = (dm.re[b * nao + j], dm.im[b * nao + j]);
                        terms.push((cr * dr - ci * di, cr * di + ci * dr));
                    }
                    let (sr, si) = csum(&terms);
                    tr[a * nao + j] = sr;
                    ti[a * nao + j] = si;
                }
            }
            let mut pr = vec![0.0; nu * nu];
            let mut pi = vec![0.0; nu * nu];
            for a in 0..nu {
                for b in 0..nu {
                    let mut terms = Vec::with_capacity(nao);
                    for j in 0..nao {
                        let (xr, xi) = (tr[a * nao + j], ti[a * nao + j]);
                        let (cr, ci) = (cinv_re[b * nao + j], cinv_im[b * nao + j]);
                        terms.push((xr * cr + xi * ci, xi * cr - xr * ci));
                    }
                    let (sr, si) = csum(&terms);
                    pr[a * nu + b] = sr;
                    pi[a * nu + b] = si;
                }
            }
            t_spin.push((tr, ti));
            p0_spin.push((pr, pi));
        }
        let mut sig = [[0.0; 3]; 3];
        for x in 0..3 {
            for y in 0..3 {
                let mut acc = Vec::new();
                for s in 0..2 {
                    let (tr, ti) = &t_spin[s];
                    let (p0r, p0i) = &p0_spin[s];
                    // dm_deriv1 = T·SC1 per spin (`:299`).
                    let mut p1_re = vec![0.0; nu * nu];
                    let mut p1_im = vec![0.0; nu * nu];
                    for a in 0..nu {
                        for b in 0..nu {
                            let mut terms = Vec::with_capacity(nao);
                            for p in 0..nao {
                                let mut sc = Vec::with_capacity(2 * nao);
                                for q in 0..nao {
                                    let (sr, si) = s0(&ovlp0[k], p, q);
                                    let (cr, ci) = c1.get(k, x, y, q, stack[b]);
                                    sc.push((sr * cr - si * ci, sr * ci + si * cr));
                                    let (ur, ui) = ovlp1[k].get(x, y, p, q, nao);
                                    let (vr, vi) = c0u(&c_lo[k], q, b);
                                    sc.push((ur * vr - ui * vi, ur * vi + ui * vr));
                                }
                                let (scr, sci) = csum(&sc);
                                let (xr, xi) = (tr[a * nao + p], ti[a * nao + p]);
                                terms.push((xr * scr - xi * sci, xr * sci + xi * scr));
                            }
                            let (sr, si) = csum(&terms);
                            p1_re[a * nu + b] = sr;
                            p1_im[a * nu + b] = si;
                        }
                    }
                    let mut off = 0usize;
                    for (idx, val) in resolved.indices.iter().zip(resolved.u_val.iter()) {
                        let n = idx.len();
                        let mut t_diag = Vec::with_capacity(n);
                        let mut t_prod = Vec::with_capacity(n * n);
                        for ai in 0..n {
                            let p = off + ai;
                            t_diag.push(p1_re[p * nu + p]);
                            for bi in 0..n {
                                let q = off + bi;
                                let (ar, ai_) = (p1_re[p * nu + q], p1_im[p * nu + q]);
                                let (br, bi_) = (p0r[q * nu + p], p0i[q * nu + p]);
                                t_prod.push(ar * br - ai_ * bi_);
                            }
                        }
                        // `:305-307`: `* 2` on the diagonal, `* 4` on the product.
                        let w = val * 0.5 / nkpts as f64;
                        acc.push(
                            w * oracle_sum(&[
                                oracle_sum(&t_diag) * 2.0,
                                -oracle_sum(&t_prod) * 4.0,
                            ]),
                        );
                        off += n;
                    }
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
