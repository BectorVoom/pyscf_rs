//! Plan 18-13 Task 1 — `pyscf/pbc/grad/uks_stress.py` (246 l), the spin index
//! at gamma.
//!
//! # The eight shared symbols (imported, never reimplemented)
//!
//! Upstream imports all eight from `rks_stress` (`uks_stress.py:24-33`); this
//! module imports them from [`super::rks`]:
//!
//! | upstream (`rks_stress.py`) | here |
//! |---|
//! | `strain_tensor_dispalcement` | [`rks::strain_tensor_displacement`](super::rks::strain_tensor_displacement) |
//! | `_finite_diff_cells` | [`rks::finite_diff_cells`](super::rks::finite_diff_cells) |
//! | `_get_weight_strain_derivatives` | [`rks::weight_strain_derivatives`](super::rks::weight_strain_derivatives) |
//! | `_get_coulG_strain_derivatives` | [`rks::coulg_strain_derivatives`](super::rks::coulg_strain_derivatives) |
//! | `_eval_ao_strain_derivatives` | [`rks::eval_ao_strain_derivatives`](super::rks::eval_ao_strain_derivatives) |
//! | `_get_vpplocG_strain_derivatives` | [`rks::vpplocg_strain_derivatives`](super::rks::vpplocg_strain_derivatives) |
//! | `_get_pp_nonloc_strain_derivatives` | [`rks::pp_nonloc_strain_derivatives`](super::rks::pp_nonloc_strain_derivatives) |
//! | `ewald` | [`rks::ewald_strain`](super::rks::ewald_strain) |
//!
//! # What differs from 18-12
//!
//! Only `get_vxc` (`uks_stress.py:34`) and `kernel` (`:184`): `rho0`/`rho1`
//! grow a leading spin axis (`(2, nvar, ngrids)`, `(3,3,2,nvar,ngrids)` at
//! `:76-77`) and `eval_xc_eff` is called with `spin = 1` (`:148`).
//! Everything 18-12 ruled applies unchanged: the fused XC contraction, the
//! [`oracle_sum`](pyscf_algebra::oracle_sum) accumulator, `rho0` and
//! `rho1[:,:,:,0]` full-grid, the block budget counting both AO arrays —
//! sized here through
//! [`strain_block_size_nset`](super::rks::strain_block_size_nset) with
//! `nset = 2` (the spin axis doubles `rho0`/`rho1` and leaves the AO arrays
//! alone).
//!
//! # Reported units
//!
//! Ha/Bohr³ through [`rks::to_stress`](super::rks::to_stress), matching 18-12.
//!
//! # GGA/MGGA
//!
//! Refused by name (missing `deriv2` AO kernel / `vtau`), exactly as in
//! [`rks::get_vxc`](super::rks::get_vxc) — never a silent spin-summed fallback.

use pyscf_algebra::{CTensor, oracle_sum};
use pyscf_core::{CoreError, PyscfRsError};
use pyscf_pbc_gto::Cell;

use super::rks::{
    VxcStrainOpts, coulg_strain_derivatives, eval_ao_strain_derivatives, ewald_strain,
    pp_nonloc_strain_derivatives, strain_ao_block_comps, strain_block_size_nset,
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

/// Require two symmetric real densities in spin-major flat order
/// (`dm[s*nao*nao + i*nao + j]`, upstream's `(2, nao, nao)` C-order).
fn require_spin_dms(dm: &[f64], nao: usize) -> Result<(), PyscfRsError> {
    if dm.len() != 2 * nao * nao {
        return Err(PbcGradError::ShapeMismatch {
            expected: 2 * nao * nao,
            got: dm.len(),
        }
        .into());
    }
    for s in 0..2 {
        for i in 0..nao {
            for j in 0..nao {
                let a = dm[s * nao * nao + i * nao + j];
                let b = dm[s * nao * nao + j * nao + i];
                if a != b {
                    return Err(invalid(format!(
                        "uks_get_vxc: spin-{s} density is not symmetric at ({i},{j}) \
                         (upstream assumes hermitian dm, uks_stress.py:139)"
                    )));
                }
                if !a.is_finite() {
                    return Err(invalid(format!(
                        "uks_get_vxc: non-finite spin-{s} density at ({i},{j})"
                    )));
                }
            }
        }
    }
    Ok(())
}

/// `get_vxc` (`uks_stress.py:34-182`): strain derivatives of Coulomb and XC at
/// the gamma point for a spin-polarized density, as `dE/dε` (NOT divided by
/// `vol` — [`uks_stress_kernel`] reports through [`to_stress`]).
///
/// `dm` is spin-major flat (`dm[s*nao*nao + i*nao + j]`). LDA only; GGA/MGGA
/// are refused by name (the `deriv2` / `vtau` boundary of 18-12 §12).
///
/// The fused LDA contraction of 18-12 §12 carries over with one extra axis:
/// per block and per spin the `rho1·vxc` partial is materialised and reduced
/// through [`oracle_sum`] at the end (D-PBC-17 discipline); `rho1 *= 2`
/// (`:146`) is folded into the block. Full-grid survivors: `rho0` per spin
/// (`(2, ngrids)`) and `rho1[:,:,:,0]` per spin (`(2, 9, ngrids)`, SCALED).
pub fn uks_get_vxc(
    cell: &Cell,
    dm: &[f64],
    xc_code: &str,
    opts: VxcStrainOpts,
) -> Result<[[f64; 3]; 3], PyscfRsError> {
    use pyscf_pbc_dft::xc::{RhoEff, XcType, eval_xc_eff_uks};

    if cell.low_dim_ft_type == pyscf_pbc_gto::LowDimFtType::InfVacuum {
        return Err(invalid(
            "uks_get_vxc: low_dim_ft_type = inf_vacuum is not supported (uks_stress.py:43)".into(),
        ));
    }
    if cell.dimension == 1 {
        return Err(invalid(
            "uks_get_vxc: dimension = 1 is not supported (uks_stress.py:44)".into(),
        ));
    }
    let nao = cell.mol.nao_nr;
    require_spin_dms(dm, nao)?;
    match lift_dft(XcType::of(xc_code)) {
        Ok(XcType::Lda) => {}
        Ok(XcType::Gga) => {
            return Err(PbcGradError::NotYetImplemented {
                phase: 18,
                what: "uks_get_vxc GGA: the grid response of the gradient-density rows \
                       needs second-derivative AOs (GTOval_sph_deriv2), which the \
                       molecular kernel defers (Phase-4 scope)",
            }
            .into());
        }
        Err(e) if e.to_string().contains("meta-GGA") => {
            return Err(PbcGradError::NotYetImplemented {
                phase: 18,
                what: "uks_get_vxc meta-GGA: XcType refuses MGGA (no tau from the periodic \
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
        invalid("uks_get_vxc: strain block components for deriv = 0 are unavailable".into())
    })?;
    // The spin axis doubles rho0/rho1 and leaves the AO arrays alone: size
    // through the nset-parameterised budget with nset = 2 (18-13 Task 1).
    let blk = match opts.max_memory_mb {
        Some(mb) => strain_block_size_nset(ngrids, nao, ao_comps, strain_comps, 8, mb, 1, 2),
        None => strain_block_size_nset_from_env(ngrids, nao, ao_comps, strain_comps, 8, 1, 2),
    };
    if blk == 0 {
        return Err(PbcGradError::ShapeMismatch {
            expected: 1,
            got: 0,
        }
        .into());
    }

    // Full-grid survivors: rho0 per spin and rho1[:,:,0] per spin (SCALED).
    let mut rho0_full = vec![0.0f64; 2 * ngrids];
    let mut rho1_full = vec![0.0f64; 2 * 9 * ngrids];
    // One materialised partial per block per spin per (x,y).
    let mut acc: [Vec<f64>; 9] = Default::default();

    let gamma: [[f64; 3]; 0] = [];
    let mut p0 = 0usize;
    while p0 < ngrids {
        let p1 = (p0 + blk).min(ngrids);
        let chunk = &grids.coords[p0..p1];
        let blen = p1 - p0;
        let ao = pyscf_pbc_gto::eval_ao_kpts(cell, "GTOval_sph_deriv1", chunk, &gamma)?;
        if ao.comp != 4 || ao.ngrids != blen || ao.nao != nao || ao.kaos.len() != 1 {
            return Err(PbcGradError::ShapeMismatch {
                expected: 4 * blen * nao,
                got: ao.kaos.first().map_or(0, |k| k.len()),
            }
            .into());
        }
        let ao_re = &ao.kaos[0].re;
        let strain = eval_ao_strain_derivatives(cell, chunk, &gamma, 0)?;
        let at_ao = |c: usize, g: usize, mu: usize| ao_re[c * blen * nao + g + mu * blen];
        // Per-spin block densities (upstream `:96-99` loops `s in range(2)`).
        let mut rho0_blk = vec![0.0f64; 2 * blen];
        let mut rho1_blk = vec![0.0f64; 2 * 9 * blen];
        for s in 0..2 {
            let off = s * nao * nao;
            let mut c0 = vec![0.0f64; nao * blen];
            for mu in 0..nao {
                for g in 0..blen {
                    let mut terms = Vec::with_capacity(nao);
                    for nu in 0..nao {
                        terms.push(dm[off + nu * nao + mu] * at_ao(0, g, nu));
                    }
                    c0[mu * blen + g] = oracle_sum(&terms);
                }
            }
            for g in 0..blen {
                let mut t0 = Vec::with_capacity(nao);
                for mu in 0..nao {
                    t0.push(at_ao(0, g, mu) * c0[mu * blen + g]);
                }
                rho0_blk[s * blen + g] = oracle_sum(&t0);
                for x in 0..3 {
                    for y in 0..3 {
                        let mut terms = Vec::with_capacity(nao);
                        for mu in 0..nao {
                            let b = strain.get(0, x, y, 0, g, mu).0;
                            let grid = at_ao(1 + x, g, mu) * chunk[g][y];
                            terms.push((b + grid) * c0[mu * blen + g]);
                        }
                        rho1_blk[(s * 9 + x * 3 + y) * blen + g] = oracle_sum(&terms);
                    }
                }
            }
        }
        for s in 0..2 {
            for (g, v) in rho0_blk[s * blen..(s + 1) * blen].iter().enumerate() {
                rho0_full[s * ngrids + p0 + g] = *v;
            }
        }
        // spin = 1 (`uks_stress.py:148`): the XC potential is pointwise, so
        // per-block eval == full-grid eval sliced, bit for bit.
        let rho_a = RhoEff {
            nvar: 1,
            ngrids: blen,
            data: rho0_blk[..blen].to_vec(),
        };
        let rho_b = RhoEff {
            nvar: 1,
            ngrids: blen,
            data: rho0_blk[blen..2 * blen].to_vec(),
        };
        let vxc_blk = lift_dft(eval_xc_eff_uks(xc_code, &rho_a, &rho_b))?;
        for s in 0..2 {
            let vxc = vxc_blk.row(s, 0);
            for x in 0..3 {
                for y in 0..3 {
                    let mut partial = Vec::with_capacity(blen);
                    for g in 0..blen {
                        let scaled = 2.0 * rho1_blk[(s * 9 + x * 3 + y) * blen + g];
                        rho1_full[(s * 9 + x * 3 + y) * ngrids + p0 + g] = scaled;
                        partial.push(scaled * vxc[g]);
                    }
                    acc[x * 3 + y].push(oracle_sum(&partial));
                }
            }
        }
        p0 = p1;
    }

    // `:149` — the fused contraction times weight_0, plus the
    // scalar-on-the-diagonal exc term over the SPIN-SUMMED rho0 (`:150-152`).
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

    // `:154-156` — G-vectors, full-grid Coulomb kernel, `rhoG = fft(rho0)`
    // over the spin-summed density.
    let mut rho_tot = vec![0.0f64; ngrids];
    for g in 0..ngrids {
        rho_tot[g] = oracle_sum(&[rho0_full[g], rho0_full[ngrids + g]]);
    }
    let mut rho1_tot = vec![0.0f64; 9 * ngrids];
    for c in 0..9 {
        for g in 0..ngrids {
            rho1_tot[c * ngrids + g] = oracle_sum(&[rho1_full[c * ngrids + g], rho1_full[(9 + c) * ngrids + g]]);
        }
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

    // Spin-summed density for the Coulomb / PP traces (`dm.sum(axis=0)`,
    // `:170`; upstream's `get_j` oracle contracts `dm` summed the same way).
    let mut dm_tot = vec![0.0f64; nao * nao];
    for p in 0..nao * nao {
        dm_tot[p] = oracle_sum(&[dm[p], dm[nao * nao + p]]);
    }

    if opts.with_j {
        // `:157-162`, identical in form to 18-12 §12.2 on the summed density.
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
        // `:164-181`, identical in form to 18-12 §12.3 on the summed density
        // and the summed `dm` (`:170` passes `dm.sum(axis=0)`).
        let has_pp = (0..cell.mol.natm).any(|ia| cell.atom_pseudo(ia).is_some());
        if has_pp {
            let vpp = vpplocg_strain_derivatives(cell, mesh)?;
            let v0 = CTensor::from_planes(vpp.v0_re.clone(), vpp.v0_im.clone());
            let vpploc_r = lift_tools(pyscf_pbc_tools::ifft(&v0, mesh))?.re;
            let pp_nl = pp_nonloc_strain_derivatives(cell, &dm_tot)?;
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

/// `kernel` (`uks_stress.py:184-245`): the asymmetric 3×3 stress tensor as a
/// PRESSURE in **Ha/Bohr³** (`σ = dE/dε / vol**, `:242`).
///
/// Upstream takes the gradient object (SCF + `make_rdm1e` inside); this port
/// takes the converged inputs directly — `dm_spin` (spin-major flat,
/// `2·nao²`) and `dme0` (the SPIN-SUMMED energy-weighted density,
/// `mf_grad.make_rdm1e().sum(axis=0)`, `:219`) — so the strained-cell SCF
/// loop of the Gate-D oracle stays in the test, never in the production path:
///
/// ```text
/// σ = ewald_strain + Tr(kin·dm0) − Tr(ovlp·dme0) + uks_get_vxc(with_j, with_nuc)
/// σ /= vol
/// ```
///
/// with `dm0 = dm_spin.sum(axis=0)` (`:218`). `get_ovlp`/`get_kin` are the
/// clause-2 CLOSED FORMS (not the `:222-235` finite differences).
///
/// # Errors
/// * Hybrid functionals: upstream raises `NotImplementedError` (`:209`);
///   this port refuses by name.
/// * DFT+U: upstream raises `NotImplementedError` (`:210-211`); refused here.
/// * GGA/MGGA: refused by [`uks_get_vxc`].
pub fn uks_stress_kernel(
    cell: &Cell,
    dm_spin: &[f64],
    dme0: &[f64],
    xc_code: &str,
) -> Result<[[f64; 3]; 3], PyscfRsError> {
    use pyscf_pbc_dft::xc::is_hybrid_xc;
    use super::rks::{kin_strain_gamma, ovlp_strain_gamma};

    let nao = cell.mol.nao_nr;
    require_spin_dms(dm_spin, nao)?;
    if dme0.len() != nao * nao {
        return Err(PbcGradError::ShapeMismatch {
            expected: nao * nao,
            got: dme0.len(),
        }
        .into());
    }
    if dme0.iter().any(|v| !v.is_finite()) {
        return Err(invalid(
            "uks_stress_kernel: non-finite energy-weighted density".into(),
        ));
    }
    if lift_dft(is_hybrid_xc(xc_code))? {
        return Err(PbcGradError::NotYetImplemented {
            phase: 18,
            what: "stress tensor for hybrid DFT (uks_stress.py:208-209 raises NotImplementedError)",
        }
        .into());
    }

    // Spin-summed densities for the hcore traces (`:218-219`).
    let mut dm0 = vec![0.0f64; nao * nao];
    for p in 0..nao * nao {
        dm0[p] = oracle_sum(&[dm_spin[p], dm_spin[nao * nao + p]]);
    }
    for i in 0..nao {
        for j in 0..nao {
            if dm0[i * nao + j] != dm0[j * nao + i] {
                return Err(invalid(format!(
                    "uks_stress_kernel: spin-summed density is not symmetric at ({i},{j})"
                )));
            }
        }
    }

    let trace = |mat: &[f64], dm: &[f64]| -> f64 {
        let mut terms = Vec::with_capacity(nao * nao);
        for i in 0..nao {
            for j in 0..nao {
                terms.push(mat[i + j * nao] * dm[j * nao + i]);
            }
        }
        oracle_sum(&terms)
    };

    let mut sigma = ewald_strain(cell)?;
    // `:222-235` via the clause-2 closed forms (arrangement A, the same
    // production path as the closed-shell kernel).
    let kin = kin_strain_gamma(cell)?;
    let ovlp = ovlp_strain_gamma(cell)?;
    for x in 0..3 {
        for y in 0..3 {
            let e_kin = trace(&kin.re[x * 3 + y], &dm0);
            let e_ovlp = trace(&ovlp.re[x * 3 + y], dme0);
            sigma[x][y] = oracle_sum(&[sigma[x][y], e_kin, -e_ovlp]);
        }
    }
    let vxc = uks_get_vxc(
        cell,
        dm_spin,
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
