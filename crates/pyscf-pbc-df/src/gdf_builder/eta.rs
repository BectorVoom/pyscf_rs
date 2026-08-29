//! The compensating-charge width `eta` and its companions —
//! `pyscf/pbc/df/gdf_builder.py:888-1062` (plan 14-02, Task 3).
//!
//! # What `eta` is
//!
//! GDF's 3-centre integrals are made short-ranged by subtracting a smooth
//! Gaussian of the same charge from every auxiliary function. `eta` is that
//! Gaussian's exponent, and it is the scheme's one real tuning knob:
//!
//! * too LARGE and the model charge is compact, so the real-space part stays
//!   long-ranged and the lattice sum is expensive;
//! * too SMALL and the model charge is diffuse, so the reciprocal-space part
//!   needs a fine mesh to resolve it.
//!
//! Upstream picks it from the plane-wave cutoff it is willing to pay for
//! ([`guess_eta`]) and bounds it from below by [`ETA_MIN`].
//!
//! # Measured targets (`measurements/params.py`)
//!
//! | cell | `eta` | `mesh` | `ke_cutoff` |
//! |---|---|---|---|
//! | diamond 2x2x2 | 0.46488312492994555 | `[11,11,11]` | 21.721883440437864 |
//! | diamond gamma | 0.6839707371739572 | `[13,13,13]` | 31.27951215423053 |
//! | He-fcc 2x2x2 | 0.37482108075015924 | `[9,9,9]` | 19.65348325887675 |
//!
//! The gamma/2x2x2 split is `guess_eta`'s `ke_cutoff = 30 * nkpts^(-1/3)`: more
//! k-points buy a coarser mesh, which forces a smaller `eta`.
//!
//! # `estimate_rcut` here is NOT `incore::estimate_rcut`
//!
//! `gdf_builder.estimate_rcut` (`:932`) and `incore.estimate_rcut` (`:440`)
//! share a name, a shape and nothing else: this one takes `cs` from
//! `_extract_pgto_params(cell, 'min')` (the libcint contraction coefficient),
//! the other from `gto_norm`; the `fac` prefactors differ; and the fixed-point
//! exponent is `l3 - 1` here against `l3 - 2` there. Plan 14-01 measured what
//! confusing them costs: 15.815 where upstream says 17.266.

use std::f64::consts::PI;

use pyscf_core::raw_layout::{ANG_OF, BAS_SLOTS};
use pyscf_pbc_gto::Cell;

use crate::error::PbcDfError;
use crate::incore::gaussian_int;

/// `ETA_MIN` — `gdf_builder.py:45`. The floor on the model-charge exponent.
pub const ETA_MIN: f64 = 0.1;

/// `gto_norm(l, alpha) = 1 / sqrt(gaussian_int(2l + 2, 2 alpha))`.
fn gto_norm(l: i32, alpha: f64) -> f64 {
    1.0 / gaussian_int(2 * l + 2, 2.0 * alpha).sqrt()
}

fn all_exps(cell: &Cell) -> Vec<f64> {
    (0..cell.mol.nbas)
        .flat_map(|i| pyscf_pbc_gto::cutoff::bas_exp(cell, i))
        .collect()
}

/// `estimate_ke_cutoff_for_eta(cell, eta, precision)` — `gdf_builder.py:1044-1062`.
///
/// The plane-wave cutoff needed to resolve a model charge of width `eta` to
/// `precision` in the AFT Coulomb integrals. Two fixed-point sweeps from 20 Ha.
pub fn estimate_ke_cutoff_for_eta(cell: &Cell, eta: f64, precision: Option<f64>) -> f64 {
    let precision = precision.unwrap_or(cell.precision);
    let ai = all_exps(cell).into_iter().fold(f64::NEG_INFINITY, f64::max);
    let aij = ai * 2.0;
    let ci = gto_norm(0, ai);
    let ck = gto_norm(0, eta);
    let theta = 1.0 / (1.0 / aij + 1.0 / eta);
    let norm_ang = (4.0 * PI).powf(-1.5);
    let mut fac = 32.0 * PI.powi(5) * ci * ci * ck * norm_ang * (2.0 * aij) / (aij * eta).powf(1.5);
    fac /= precision;

    let mut ecut = 20.0_f64;
    ecut = (fac * (ecut * 2.0).powf(-0.5)).ln() * 2.0 * theta;
    ecut = (fac * (ecut * 2.0).powf(-0.5)).ln() * 2.0 * theta;
    ecut
}

/// `estimate_eta_for_ke_cutoff(cell, ke_cutoff, precision)` —
/// `gdf_builder.py:1023-1042`. The inverse of [`estimate_ke_cutoff_for_eta`]:
/// the LARGEST `eta` a given mesh can carry, capped at 4.
pub fn estimate_eta_for_ke_cutoff(cell: &Cell, ke_cutoff: f64, precision: Option<f64>) -> f64 {
    let precision = precision.unwrap_or(cell.precision);
    let ai = all_exps(cell).into_iter().fold(f64::NEG_INFINITY, f64::max);
    let aij = ai * 2.0;
    let ci = gto_norm(0, ai);
    let norm_ang = (4.0 * PI).powf(-1.5);
    let c1 = ci * ci * norm_ang;
    let fac = 64.0 * PI.powi(5) * c1 * (aij * ke_cutoff * 2.0).powf(-0.5) / precision;

    let eta0 = 4.0_f64;
    let eta = 1.0 / ((fac * eta0.powf(-1.5)).ln() * 2.0 / ke_cutoff - 1.0 / aij);
    if eta < 0.0 { 4.0 } else { eta.min(4.0) }
}

/// `estimate_eta_min(cell, precision)` — `gdf_builder.py:1009-1021`.
///
/// The smallest `eta` whose model charge is still negligible at the boundary of
/// the image sphere: `4 pi rmax^2 exp(-eta/2 rmax^2) < precision`.
///
/// # Errors
/// Propagates `cell.try_rcut`.
pub fn estimate_eta_min(cell: &Cell, precision: Option<f64>) -> Result<f64, PbcDfError> {
    let precision = precision.unwrap_or(cell.precision);
    let lmax = (0..cell.mol.nbas)
        .map(|i| cell.mol._bas[i * BAS_SLOTS + ANG_OF].max(0))
        .max()
        .unwrap_or(0)
        .min(4);
    let rcut = cell.try_rcut()?;
    let eta = (4.0 * PI * rcut.powi(lmax + 2) / precision).ln() / (rcut * rcut);
    Ok(eta.max(ETA_MIN))
}

/// The `(eta, mesh, ke_cutoff)` triple [`guess_eta`] settles on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EtaChoice {
    /// The compensating-charge exponent.
    pub eta: f64,
    /// The plane-wave mesh the reciprocal-space half runs on.
    pub mesh: [usize; 3],
    /// The kinetic-energy cutoff that mesh corresponds to.
    pub ke_cutoff: f64,
}

/// `_guess_eta(cell, kpts, mesh)` — `gdf_builder.py:888-916`.
///
/// **`cell` here is the AUXCELL**, not the orbital cell — `_CCGDFBuilder.build`
/// calls `_guess_eta(auxcell, kpts, self.mesh)`. Passing the orbital cell gives
/// a plausible but wrong `eta`.
///
/// `mesh = None` picks `ke_cutoff = max(30 * nkpts^(-1/3), ke_min)`; an explicit
/// mesh is honoured, with a warning when it is too coarse for `cell.precision`.
///
/// # Errors
/// [`PbcDfError::Core`] when the lattice is singular, and propagates
/// `cutoff_to_mesh`.
pub fn guess_eta(
    cell: &Cell,
    kpts: &[[f64; 3]],
    mesh: Option<[usize; 3]>,
) -> Result<EtaChoice, PbcDfError> {
    let a = cell.lattice_vectors();
    if cell.dimension == 0 {
        let mesh = match mesh {
            Some(m) => m,
            None => cell.try_mesh()?,
        };
        let ke_cutoff = pyscf_pbc_tools::mesh::mesh_to_cutoff(&a, mesh)?
            .into_iter()
            .fold(f64::INFINITY, f64::min);
        let eta = estimate_eta_for_ke_cutoff(cell, ke_cutoff, Some(cell.precision));
        return Ok(EtaChoice { eta, mesh, ke_cutoff });
    }

    // `eta_min = ETA_MIN` — upstream leaves the `estimate_eta_min` call
    // commented out at :898 and uses the constant.
    let ke_min = estimate_ke_cutoff_for_eta(cell, ETA_MIN, Some(cell.precision));

    let mesh = match mesh {
        None => {
            let nkpts = kpts.len().max(1) as f64;
            let ke_cutoff = (30.0 * nkpts.powf(-1.0 / 3.0)).max(ke_min);
            cell.cutoff_to_mesh(ke_cutoff)?
        }
        Some(m) => {
            let mesh_min = cell.cutoff_to_mesh(ke_min)?;
            let dim = cell.dimension as usize;
            if (0..dim).any(|i| m[i] < mesh_min[i]) {
                tracing::warn!(
                    "guess_eta: mesh {m:?} is not enough to converge to the required \
                     integral precision {:e}; recommended mesh is {mesh_min:?}",
                    cell.precision
                );
            }
            m
        }
    };

    let dim = (cell.dimension as usize).max(1);
    let ke_cutoff = pyscf_pbc_tools::mesh::mesh_to_cutoff(&a, mesh)?[..dim]
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let eta = estimate_eta_for_ke_cutoff(cell, ke_cutoff, Some(cell.precision));
    Ok(EtaChoice { eta, mesh, ke_cutoff })
}

/// `estimate_rcut(rs_cell, auxcell, precision, exclude_dd_block=False)` —
/// `gdf_builder.py:932-1007`, the `exclude_dd_block = False` half.
///
/// The 3-centre image radius the COMPENSATED builder uses. See the module docs
/// for why this is not `incore::estimate_rcut`. The `exclude_dd_block = True`
/// half is D-PBC-23 and lives in Phase 17.
///
/// Measured targets: **16.729034885581783** (diamond), **10.750308556151602**
/// (He-fcc), both against the FUSED auxiliary cell.
pub fn estimate_rcut(cell: &Cell, fused: &Cell, precision: Option<f64>) -> f64 {
    // `precision = rs_cell.precision * 1e-1` — upstream nudges it because the
    // measured errors come out slightly above `cell.precision`.
    let precision = precision.unwrap_or(cell.precision * 1e-1);
    if cell.mol.nbas == 0 || fused.mol.nbas == 0 {
        return 0.0;
    }

    let (cell_exps, cs) = pyscf_pbc_gto::extract_pgto_params(cell, pyscf_pbc_gto::PgtoOp::Min);
    let ls: Vec<i32> = (0..cell.mol.nbas)
        .map(|i| cell.mol._bas[i * BAS_SLOTS + ANG_OF].max(0))
        .collect();
    let aux_exps: Vec<f64> = (0..fused.mol.nbas)
        .map(|i| {
            pyscf_pbc_gto::cutoff::bas_exp(fused, i)
                .into_iter()
                .fold(f64::INFINITY, f64::min)
        })
        .collect();

    let ai_idx = argmin(&cell_exps);
    let ak_idx = argmin(&aux_exps);
    let ai = cell_exps[ai_idx];
    let ak = aux_exps[ak_idx];
    let li = f64::from(ls[ai_idx]);
    let lk_i = fused.mol._bas[ak_idx * BAS_SLOTS + ANG_OF].max(0);
    let ci = cs[ai_idx];
    let ck = 1.0 / (4.0 * PI) / gaussian_int(lk_i + 2, ak);
    let lk = f64::from(lk_i);

    let vol = cell.vol();
    let r_start = cell.try_rcut().unwrap_or(20.0);

    let mut rcut = 0.0_f64;
    for (&aj, (&cj, &lj_i)) in cell_exps.iter().zip(cs.iter().zip(ls.iter())) {
        let lj = f64::from(lj_i);
        let aij = ai + aj;
        let lij = li + lj;
        let l3 = lij + lk;
        let theta = 1.0 / (1.0 / aij + 1.0 / ak);
        let norm_ang = ((2.0 * li + 1.0) * (2.0 * lj + 1.0)).sqrt() / (4.0 * PI);
        let c1 = ci * cj * ck * norm_ang;
        let sfac = aij * aj / (aij * aj + ai * theta);
        let fl = 2.0_f64;
        // fac = 2**li * pi**2.5 * c1 * theta**(l3-.5)
        let mut fac = 2.0_f64.powf(li) * PI.powf(2.5) * c1 * theta.powf(l3 - 0.5);
        fac *= 2.0 * PI / vol / theta;
        fac /= aij.powf(li + 1.5) * ak.powf(lk + 1.5) * aj.powf(lj);
        fac *= fl / precision;

        // NOTE the `(sfac*r0)**(l3 - 1)` exponent — `incore` uses `l3 - 2`.
        let step =
            |r0: f64| ((fac * r0 * (sfac * r0).powf(l3 - 1.0) + 1.0).ln() / (sfac * theta)).sqrt();
        let r0 = step(step(r_start));
        if r0.is_finite() && r0 > rcut {
            rcut = r0;
        }
    }
    rcut
}

fn argmin(v: &[f64]) -> usize {
    let mut idx = 0usize;
    for (k, x) in v.iter().enumerate().skip(1) {
        if *x < v[idx] {
            idx = k;
        }
    }
    idx
}
