//! The 2-centre metric `j2c` and its decomposition —
//! `pyscf/pbc/df/gdf_builder.py:139-196` and
//! `pyscf/pbc/df/rsdf_builder.py:215-247` (plan 14-02, Task 5).
//!
//! # What the metric is
//!
//! `j2c[P,Q] = (P|Q)` over the COMPENSATED auxiliary basis: the real-space
//! lattice sum of `int2c2e` over the fused cell, minus the model-charge blocks
//! evaluated in reciprocal space, symmetrised, and finally `fuse`d on both
//! indices. `cderi` is then `j2c^{-1/2} j3c`, so an error here is an error in
//! every fitted integral.
//!
//! # Three things upstream does that are easy to miss
//!
//! 1. **`hermi = 0`.** The `int2c2e` lattice sum is not Hermitian unless the
//!    image list is symmetric, and upstream says so in a comment
//!    (`gdf_builder.py:143-148`). The matrix is symmetrised at the END, after
//!    the plane-wave part is folded in.
//! 2. **The mesh is NOT `self.mesh`.** It is recomputed at
//!    `precision = auxcell.precision²` — the metric is more sensitive than the
//!    3-centre tensor, and `self.mesh` is not fine enough
//!    (`gdf_builder.py:150-158`).
//! 3. **`fuse` is applied TWICE**, once per index.
//!
//! # The `ft_ao` path needs no `modrho_scale`
//!
//! [`crate::ft_ao::single::ft_ao_kpt`] reads its contraction coefficients from
//! `mol._env`, which [`crate::incore::auxcell::apply_modrho`] has already
//! rewritten — so the analytic Fourier transform is monopole-normalised for
//! free. The cintx path reads `_basis` instead, which is unit-NORM, and *does*
//! need the scale. Mixing the two conventions is the defect this note exists to
//! prevent.

use pyscf_algebra::CTensor;
use pyscf_pbc_gto::{Cell, CoulGArgs, get_coulg};

use crate::error::PbcDfError;
use crate::ft_ao::single::ft_ao_kpt;
use crate::gdf_builder::eta::estimate_ke_cutoff_for_eta;
use crate::gdf_builder::fuse::FusedCell;
use crate::incore::fill_2c2e;

/// How [`decompose_j2c`] factorised the metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum J2cTag {
    /// Cholesky — `cderi` comes from a triangular solve.
    Cd,
    /// Eigenvalue decomposition — `cderi` is a dense multiply.
    Eig,
}

/// The factorised metric `gdf_builder::solve_cderi` consumes.
#[derive(Debug, Clone)]
pub struct CdJ2c {
    /// `L` (lower Cholesky factor) for [`J2cTag::Cd`], or `v/sqrt(w)` for
    /// [`J2cTag::Eig`], row-major `(rank, naux)`.
    pub j2c: CTensor,
    /// The rank the decomposition kept — `naux` for Cholesky.
    pub rank: usize,
    /// The NEGATIVE branch of the eigen route, for 2-D truncated Coulomb.
    /// Kept as a field so the shape is right; the path itself is refused.
    pub j2c_negative: Option<CTensor>,
    /// Which route produced it.
    pub tag: J2cTag,
}

/// `linear_dep_threshold` — `rsdf_builder.py:LINEAR_DEP_THR`.
pub const LINEAR_DEP_THRESHOLD: f64 = 1e-10;

/// `weighted_coulG(kpt, exx=False, mesh)` — `aft.py:236-245`, on the ORBITAL
/// cell. `get_2c2e` and `weighted_ft_ao` both use it with `exx = False`.
///
/// # Errors
/// Propagates `get_coulG` and the G-vector build.
pub fn weighted_coulg(cell: &Cell, kpt: [f64; 3], mesh: [usize; 3]) -> Result<Vec<f64>, PbcDfError> {
    let gv = pyscf_pbc_gto::gv::get_gv(cell, Some(mesh))?;
    let gw = pyscf_pbc_gto::gv::get_gv_weights(cell, Some(mesh))?;
    let mut coulg = get_coulg(
        cell,
        CoulGArgs {
            k: kpt,
            exxdiv: None,
            mesh: Some(mesh),
            gv: Some(&gv),
            ..Default::default()
        },
    )?;
    for (g, v) in coulg.iter_mut().enumerate() {
        *v *= gw.weight(g);
    }
    // (the enumerate index IS the G index here; `gw.weight` is a method, not a
    // slice, so there is nothing to zip against)
    Ok(coulg)
}

/// The mesh `get_2c2e` evaluates its plane-wave part on — `gdf_builder.py:150-158`.
///
/// # Errors
/// Propagates `cutoff_to_mesh`.
pub fn j2c_mesh(fused: &FusedCell) -> Result<[usize; 3], PbcDfError> {
    let auxcell = &fused.auxcell.cell;
    let precision = auxcell.precision * auxcell.precision;
    let ke = estimate_ke_cutoff_for_eta(auxcell, fused.eta, Some(precision));
    Ok(auxcell.cutoff_to_mesh(ke)?)
}

/// `get_2c2e(uniq_kpts)` — `gdf_builder.py:139-196`, the `dimension == 3` path.
///
/// Returns one `(naux, naux)` row-major complex metric per entry of `uniq_kpts`.
///
/// # Errors
/// Propagates the lattice sum, the G-vector build and `get_coulG`.
pub fn get_2c2e(
    cell: &Cell,
    fused: &FusedCell,
    uniq_kpts: &[[f64; 3]],
) -> Result<Vec<CTensor>, PbcDfError> {
    let nauxc = fused.nauxc();
    let naux = fused.naux();

    // Pass 1 — the real-space lattice sum over the FUSED cell, `hermi = 0`.
    // `pbc_intor` returns F-order; transpose to row-major so the plane-wave
    // pass below can index `[p * nauxc + q]` throughout.
    let mut j2c: Vec<CTensor> = fill_2c2e(&fused.fused, 0, uniq_kpts)?
        .into_iter()
        .map(|m| crate::zlinalg::forder_to_c(&m, nauxc, nauxc))
        .collect();

    // Pass 2 — subtract the model-charge blocks in reciprocal space.
    let mesh = j2c_mesh(fused)?;
    let gv = pyscf_pbc_gto::gv::get_gv(&fused.fused.cell, Some(mesh))?;
    let ngrids = gv.len();
    tracing::debug!(
        "get_2c2e: 2c2e integrals precision {:e}, mesh {mesh:?} ({ngrids} PWs)",
        cell.precision * cell.precision
    );

    let chg: Vec<usize> = (0..nauxc)
        .filter(|q| !fused.aux_ao.contains(q))
        .collect();

    for (k, kpt) in uniq_kpts.iter().enumerate() {
        let coulg = weighted_coulg(cell, *kpt, mesh)?;
        // `auxG = ft_ao(fused_cell, Gv, kpt).T` — `[g * nauxc + q]`.
        let (agr, agi) = ft_ao_kpt(&fused.fused.cell.mol, &gv, *kpt)?;

        // `j2c_p[c][q] = SUM_g conj(auxG[c][g]) coulG[g] auxG[q][g]`.
        let nchg = chg.len();
        let mut pr = vec![0.0_f64; nchg * nauxc];
        let mut pi = vec![0.0_f64; nchg * nauxc];
        for g in 0..ngrids {
            let w = coulg[g];
            if w == 0.0 {
                continue;
            }
            let base = g * nauxc;
            for (ci, &c) in chg.iter().enumerate() {
                let (ar, ai) = (agr[base + c] * w, agi[base + c] * w);
                let row = ci * nauxc;
                for q in 0..nauxc {
                    let (br, bi) = (agr[base + q], agi[base + q]);
                    // conj(a) * b
                    pr[row + q] += ar * br + ai * bi;
                    pi[row + q] += ar * bi - ai * br;
                }
            }
        }

        let m = &mut j2c[k];
        for (ci, &c) in chg.iter().enumerate() {
            let row = ci * nauxc;
            for q in 0..nauxc {
                // `j2c[naux:] -= j2c_p`
                m.re[c * nauxc + q] -= pr[row + q];
                m.im[c * nauxc + q] -= pi[row + q];
            }
            // `j2c[:naux, naux:] -= j2c_p[:, :naux].conj().T`
            for a in 0..naux {
                let fa = fused.aux_ao[a];
                m.re[fa * nauxc + c] -= pr[row + fa];
                m.im[fa * nauxc + c] += pi[row + fa];
            }
        }

        // `j2c[k] = (j2c[k] + j2c[k].conj().T) * .5`
        for p in 0..nauxc {
            for q in p..nauxc {
                let (a, b) = (p * nauxc + q, q * nauxc + p);
                let re = (m.re[a] + m.re[b]) * 0.5;
                let im = (m.im[a] - m.im[b]) * 0.5;
                m.re[a] = re;
                m.re[b] = re;
                m.im[a] = im;
                m.im[b] = -im;
            }
        }
    }

    // `j2c[k] = self.fuse(self.fuse(j2c[k]), axis=1)`
    Ok(j2c
        .iter()
        .map(|m| {
            let (r, i) = fused.fuse_rows_complex(&m.re, &m.im, nauxc);
            CTensor {
                re: fused.fuse_cols(&r, naux),
                im: fused.fuse_cols(&i, naux),
            }
        })
        .collect())
}

/// `decompose_j2c(j2c)` — `rsdf_builder.py:215-247`.
///
/// Cholesky first; on failure, the eigenvalue route with everything below
/// [`LINEAR_DEP_THRESHOLD`] dropped.
///
/// **Do not pre-check the spectrum.** Diamond's metric has `eig_min = 3.17e-11`
/// — BELOW the threshold — and upstream still returns `"CD"`, because Cholesky
/// is attempted first and succeeds. A port that inspects the eigenvalues and
/// pre-emptively takes the eigen branch disagrees with upstream on the flagship
/// system (`measurements/params.py`).
///
/// # Errors
/// [`PbcDfError::Core`] when the eigen decomposition itself fails.
pub fn decompose_j2c(j2c: &CTensor, n: usize, j2c_eig_always: bool) -> Result<CdJ2c, PbcDfError> {
    if !j2c_eig_always
        && let Some(l) = cholesky_lower(j2c, n)
    {
        return Ok(CdJ2c {
            j2c: l,
            rank: n,
            j2c_negative: None,
            tag: J2cTag::Cd,
        });
    }
    eigenvalue_decomposed_metric(j2c, n)
}

/// `cholesky_decomposed_metric(j2c)` — `rsdf_builder.py:222-229`.
///
/// Returns the LOWER factor `L` with `L Lᴴ = j2c`, or `None` when the matrix is
/// not positive definite — which is upstream's `scipy.linalg.LinAlgError`, the
/// signal that sends it to the eigen route.
fn cholesky_lower(j2c: &CTensor, n: usize) -> Option<CTensor> {
    pyscf_algebra::zcholesky(j2c, n).ok()
}

/// `eigenvalue_decomposed_metric(j2c)` — `rsdf_builder.py:231-246`.
///
/// `v[w > thr] / sqrt(w)`, and the negative branch split off for the 2-D
/// truncated-Coulomb case (kept as a field; the path itself is refused
/// downstream per 14-CONTEXT's non-goals).
///
/// # Errors
/// [`PbcDfError::Core`] when the Hermitian eigensolver fails.
pub fn eigenvalue_decomposed_metric(j2c: &CTensor, n: usize) -> Result<CdJ2c, PbcDfError> {
    // A standard Hermitian eigenproblem is the generalised one with `S = I`;
    // `pyscf-algebra` exposes only the generalised form (D-PBC-04).
    let mut eye = CTensor {
        re: vec![0.0; n * n],
        im: vec![0.0; n * n],
    };
    for i in 0..n {
        eye.re[i * n + i] = 1.0;
    }
    let (values, vectors) = pyscf_algebra::zeigh_gen(j2c, &eye, n).map_err(|e| {
        PbcDfError::Core(pyscf_core::PyscfRsError::Core(
            pyscf_core::CoreError::InvalidMolecule(format!(
                "decompose_j2c: the Hermitian eigensolver failed on a {n}x{n} metric: {e}"
            )),
        ))
    })?;
    let mut rows_pos: Vec<usize> = Vec::new();
    let mut rows_neg: Vec<usize> = Vec::new();
    for (i, w) in values.iter().enumerate() {
        if *w > LINEAR_DEP_THRESHOLD {
            rows_pos.push(i);
        } else if *w < -LINEAR_DEP_THRESHOLD {
            rows_neg.push(i);
        }
    }
    // Upstream keeps `v[:, mask].conj().T / sqrt(w[mask])`, i.e. one ROW per
    // surviving eigenvector.
    //
    // **`zeigh_gen` returns `C` COLUMN-MAJOR (F-order)** — its module docs say
    // so — so element `(q, i)` lives at `q + i*n`, not at `q*n + i`. Reading it
    // row-major transposes the eigenvector matrix, which is still orthogonal
    // and still has the right eigenvalues, so nothing crashes and nothing looks
    // obviously wrong; the fitted tensor is simply built in the wrong basis.
    // **No gate had ever exercised this branch**: `j2ctag` is `CD` on every
    // system in `measurements/params.py`, so `decompose_j2c` always took
    // Cholesky. MDF (`j2c_eig_always = True`, `mdf.py:365`) is the first
    // consumer of the eigen route, and it is what caught this — at 6.3e6 Ha.
    let take = |rows: &[usize]| -> CTensor {
        let mut re = vec![0.0_f64; rows.len() * n];
        let mut im = vec![0.0_f64; rows.len() * n];
        for (r, &i) in rows.iter().enumerate() {
            let s = 1.0 / values[i].abs().sqrt();
            for q in 0..n {
                // `v[:, mask].conj().T / sqrt(w)` — column i of `vectors`
                // becomes row r of the output, conjugated.
                re[r * n + q] = vectors.re[q + i * n] * s;
                im[r * n + q] = -vectors.im[q + i * n] * s;
            }
        }
        CTensor { re, im }
    };
    Ok(CdJ2c {
        rank: rows_pos.len(),
        j2c: take(&rows_pos),
        j2c_negative: if rows_neg.is_empty() {
            None
        } else {
            Some(take(&rows_neg))
        },
        tag: J2cTag::Eig,
    })
}
