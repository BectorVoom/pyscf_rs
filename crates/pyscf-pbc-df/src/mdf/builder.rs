//! `_CCMDFBuilder` — the compensated-charge mixed-density-fitting builder
//! (`pyscf/pbc/df/mdf.py:354-460`), plan 14-06.
//!
//! # What MDF changes, and it is only four things
//!
//! `_CCMDFBuilder` subclasses `_CCGDFBuilder` and overrides `get_2c2e`,
//! `weighted_ft_ao`, `gen_j3c_loader`, `add_ft_j3c` and `solve_cderi` — and the
//! last three are one idea between them. This module supplies the two
//! genuinely different quantities; the pipeline they feed is 14-02's, selected
//! by [`crate::gdf_builder::j3c::Scheme::Mixed`].
//!
//! The idea: GDF fits `(mu nu|P)` in a Gaussian auxiliary basis and uses plane
//! waves only to carry the compensating charge's long-range tail. MDF makes the
//! plane waves part of the BASIS. Its fitting set is `{G}` together with the
//! Gaussians orthogonalised against them, `|g> - |G><G|g>`, so
//!
//! ```text
//! j2c = <g|g> - <g|G><G|g>          (mdf.py:371-374, the comment is upstream's)
//! j3c = (mu nu|g) - (mu nu|G)<G|g>
//! ```
//!
//! and the plane-wave block `(mu nu|G) coulG (G|rs)` is added back by
//! `aft_jk` / `aft_ao2mo` at contraction time. That is why MDF converges to
//! FFTDF as its mesh rises and GDF does not: the residual the Gaussians cannot
//! fit is carried exactly, on a grid.
//!
//! # `j2c_eig_always = True`, and upstream says why
//!
//! `mdf.py:362-365`: "For MDF, large difference may be found in results between
//! the CD/ED treatments. In some systems, small integral errors can lead to a
//! difference in the total energy/orbital energy around 4th decimal place.
//! Abandon CD treatment for better numerical stability." Subtracting the
//! plane-wave projection can push the metric indefinite, and Cholesky on an
//! indefinite matrix does not merely lose accuracy — it fails or returns
//! nonsense. [`crate::mdf::Mdf`] therefore forces the eigen route and does not
//! offer the choice.
//!
//! # The mesh is MDF's own, not `j2c_mesh`
//!
//! `_CCGDFBuilder.get_2c2e` evaluates its plane-wave part at
//! `precision = auxcell.precision²` (`gdf_builder.py:150-158`) because the
//! compensating-charge metric is more sensitive than the tensor. MDF's metric
//! is evaluated on **`self.mesh`** — the same grid the residual is carried on —
//! because here the plane waves are the basis, and metric and tensor must be
//! projected against the SAME set or the fit is inconsistent.

use pyscf_algebra::CTensor;
use pyscf_pbc_gto::{Cell, CoulGArgs, get_coulg};

use crate::error::PbcDfError;
use crate::ft_ao::single::ft_ao_kpt;
use crate::gdf_builder::fuse::FusedCell;
use crate::incore::fill_2c2e;

/// `(GauxR, GauxI, rows)` — the shape [`crate::gdf_builder::j3c`]'s
/// `add_ft_j3c` consumes.
pub type WeightedFtAo = (Vec<f64>, Vec<f64>, Vec<usize>);

/// `MDF.weighted_coulG(kpt, exx=False, mesh)` — `mdf.py:143-172`.
///
/// `aft.weighted_coulG` plus one screening step that exists **only inside
/// MDF**, and upstream's comment (`mdf.py:143-155`) is the whole
/// justification: PySCF 2.10 and earlier dropped the plane waves at
/// `±Gmax ± 0.5` to keep the basis symmetric under `G -> -G`, because an
/// asymmetric plane-wave set puts an imaginary part into the Coulomb energy.
/// The screen was removed from `tools.pbc.get_coulG` (it broke supercell /
/// k-point consistency) and re-applied here, where MDF's coarse mesh makes the
/// asymmetry matter.
///
/// It fires only at a half-integer scaled k-point — which on a 2x2x2
/// Monkhorst-Pack mesh is EVERY k-point difference, so it is not an edge case
/// on the systems this phase gates.
///
/// # Errors
/// Propagates the G-vector build and `get_coulG`.
pub fn weighted_coulg(
    cell: &Cell,
    kpt: [f64; 3],
    mesh: [usize; 3],
) -> Result<Vec<f64>, PbcDfError> {
    let mut w = crate::gdf_builder::j2c::weighted_coulg(cell, kpt, mesh)?;
    screen_pw_edges(cell, kpt, mesh, &mut w);
    Ok(w)
}

/// The edge screen of [`weighted_coulg`], applied in place to an
/// already-weighted `coulG`.
///
/// Split out so [`crate::aftdf::Aftdf`] can apply it to the kernel `aft_jk` and
/// `aft_ao2mo` build — the MDF half of `mdf_jk` runs through those, and
/// upstream reaches the same code by `mydf` being an `MDF` whose
/// `weighted_coulG` is this one (`mdf.py:143`).
pub fn screen_pw_edges(cell: &Cell, kpt: [f64; 3], mesh: [usize; 3], w: &mut [f64]) {
    if kpt.iter().all(|v| v.abs() < 1e-9) {
        return;
    }
    let scaled = cell.get_scaled_kpts(std::slice::from_ref(&kpt));
    let s = scaled[0];
    // `Gv` is C-order over `(gx, gy, gz)` with `gz` fastest
    // (`cell.py:568`), so axis `n`'s index is a plain strided selection.
    let stride = [mesh[1] * mesh[2], mesh[2], 1];
    for n in 0..cell.dimension as usize {
        let kill = if (s[n] + 0.5).abs() < 1e-12 {
            // k = -0.5: `-Gmax - 0.5` lies on the edge.
            Some(mesh[n] / 2 + 1)
        } else if (s[n] - 0.5).abs() < 1e-12 && mesh[n] % 2 == 1 {
            // k = +0.5: `Gmax + 0.5` lies on the edge, odd mesh only.
            Some(mesh[n] / 2)
        } else {
            None
        };
        let Some(j) = kill else { continue };
        if j >= mesh[n] {
            continue;
        }
        for (g, v) in w.iter_mut().enumerate() {
            if (g / stride[n]) % mesh[n] == j {
                *v = 0.0;
            }
        }
    }
}

/// `_CCMDFBuilder.get_2c2e(uniq_kpts)` — `mdf.py:369-400`.
///
/// Returns one `(naux, naux)` row-major metric per entry of `uniq_kpts`.
///
/// Three differences from [`crate::gdf_builder::j2c::get_2c2e`], all of them
/// load-bearing:
/// 1. `fuse` is applied FIRST, so the plane-wave projection is subtracted from
///    the already-compensated `naux x naux` metric rather than from the
///    `nauxc x nauxc` one;
/// 2. the projection uses EVERY fused auxiliary function, not just the model
///    charges;
/// 3. the mesh is MDF's own (see the module docs).
///
/// # Errors
/// Propagates the lattice sum, the G-vector build and `get_coulG`.
pub fn get_2c2e(
    cell: &Cell,
    fused: &FusedCell,
    uniq_kpts: &[[f64; 3]],
    mesh: [usize; 3],
) -> Result<Vec<CTensor>, PbcDfError> {
    let nauxc = fused.nauxc();
    let naux = fused.naux();

    // `j2c = fused_cell.pbc_intor('int2c2e', hermi=0, kpts=uniq_kpts)`.
    let raw: Vec<CTensor> = fill_2c2e(&fused.fused, 0, uniq_kpts, None)?
        .into_iter()
        .map(|m| crate::zlinalg::forder_to_c(&m, nauxc, nauxc))
        .collect();

    let gv = pyscf_pbc_gto::gv::get_gv(&fused.fused.cell, Some(mesh))?;
    let ngrids = gv.len();
    let mut out = Vec::with_capacity(uniq_kpts.len());

    for (k, kpt) in uniq_kpts.iter().enumerate() {
        // `j2c_k = self.fuse(self.fuse(j2c[k]), axis=1)` — BOTH indices, first.
        let (r, i) = fused.fuse_rows_complex(&raw[k].re, &raw[k].im, nauxc);
        let mut m = CTensor {
            re: fused.fuse_cols(&r, naux),
            im: fused.fuse_cols(&i, naux),
        };
        // `(j2c_k + j2c_k.conj().T) * .5`
        for p in 0..naux {
            for q in p..naux {
                let (a, b) = (p * naux + q, q * naux + p);
                let re = (m.re[a] + m.re[b]) * 0.5;
                let im = (m.im[a] - m.im[b]) * 0.5;
                m.re[a] = re;
                m.re[b] = re;
                m.im[a] = im;
                m.im[b] = -im;
            }
        }

        let coulg = weighted_coulg(cell, *kpt, mesh)?;
        // `auxG = self.fuse(ft_ao(fused_cell, Gv, kpt).T)` — `(ngrids, naux)`
        // after fusing the auxiliary axis.
        let (agr_full, agi_full) = ft_ao_kpt(&fused.fused.cell.mol, &gv, *kpt)?;
        let agr = fused.fuse_cols(&agr_full, ngrids);
        let agi = fused.fuse_cols(&agi_full, ngrids);

        let gamma = kpt.iter().all(|v| v.abs() < 1e-9);
        for (g, &w) in coulg.iter().enumerate().take(ngrids) {
            if w == 0.0 {
                continue;
            }
            let base = g * naux;
            for p in 0..naux {
                // conj(auxG[p]) * coulG
                let (ar, ai) = (agr[base + p] * w, -agi[base + p] * w);
                if ar == 0.0 && ai == 0.0 {
                    continue;
                }
                let row = p * naux;
                for q in 0..naux {
                    let (br, bi) = (agr[base + q], agi[base + q]);
                    m.re[row + q] -= ar * br - ai * bi;
                    // `if is_zero(kpt): j2c_k -= (...).real` — upstream keeps
                    // only the real part at the gamma difference.
                    if !gamma {
                        m.im[row + q] -= ar * bi + ai * br;
                    }
                }
            }
        }
        out.push(m);
    }
    Ok(out)
}

/// `_CCMDFBuilder.weighted_ft_ao(kpt)` — `mdf.py:410-418`.
///
/// Returns `(GauxR, GauxI, rows)` in the shape
/// [`crate::gdf_builder::j3c`]'s `add_ft_j3c` consumes: `(ngrids, naux)`
/// row-major planes, and the buffer rows they accumulate into — here simply
/// `0..naux`, because MDF's working buffer has no separate model-charge rows.
///
/// # Errors
/// Propagates the G-vector build and `get_coulG`.
pub fn weighted_ft_ao(
    cell: &Cell,
    fused: &FusedCell,
    kpt: [f64; 3],
    mesh: [usize; 3],
) -> Result<WeightedFtAo, PbcDfError> {
    let naux = fused.naux();
    let gv = pyscf_pbc_gto::gv::get_gv(&fused.fused.cell, Some(mesh))?;
    let ngrids = gv.len();
    let (agr_full, agi_full) = ft_ao_kpt(&fused.fused.cell.mol, &gv, kpt)?;
    let mut re = fused.fuse_cols(&agr_full, ngrids);
    let mut im = fused.fuse_cols(&agi_full, ngrids);
    let coulg = weighted_coulg(cell, kpt, mesh)?;
    for (g, &w) in coulg.iter().enumerate().take(ngrids) {
        for p in 0..naux {
            re[g * naux + p] *= w;
            im[g * naux + p] *= w;
        }
    }
    Ok((re, im, (0..naux).collect()))
}

/// The Coulomb kernel MDF's plane-wave half uses, exposed so
/// [`crate::mdf::mdf_jk`] can hand the same one to `aft_jk`.
///
/// # Errors
/// Propagates `get_coulG`.
pub fn plain_coulg(
    cell: &Cell,
    kpt: [f64; 3],
    mesh: [usize; 3],
    gv: &[[f64; 3]],
) -> Result<Vec<f64>, PbcDfError> {
    Ok(get_coulg(
        cell,
        CoulGArgs {
            k: kpt,
            mesh: Some(mesh),
            gv: Some(gv),
            ..Default::default()
        },
    )?)
}
