//! Plan 14-07, sub-task 7a — the ω machinery of `rsdf_builder.py`.
//!
//! # What range separation is, and why `ω` is the whole scheme
//!
//! GDF makes the 3-centre lattice sum converge by neutralising the auxiliary
//! functions with a compensating charge. Range-separated DF splits the KERNEL
//! instead:
//!
//! ```text
//! 1/r = erfc(w r)/r  +  erf(w r)/r
//!       ~~~~~~~~~~~     ~~~~~~~~~~
//!       short range     long range
//!       real space      few plane waves
//! ```
//!
//! The short-range half decays like a Gaussian, so its real-space lattice sum
//! terminates quickly; the long-range half is smooth, so a coarse grid resolves
//! it. Upstream defaults to this route (`GDF._prefer_ccdf = False`), and it is
//! why `df.GDF()` is the fastest builder in `measurements/builders.out`
//! (6.4 s against FFTDF's 30.0 s on diamond 2x2x2).
//!
//! **A wrong `ω` does not fail loudly. It produces a plausible 1e-6.** That is
//! why plan 14-07 makes 7a a separate, fully-tested sub-task, and why
//! `measurements/omega.py` recorded every number in this module before a line
//! of it was written.
//!
//! # Everything here is a pure function
//!
//! Nothing in this module needs a short-range integral, so all of it ships —
//! unlike `_RSGDFBuilder` itself, which does not. See [`crate::rsdf_builder`]'s
//! module docs for that.
//!
//! # `cs` is `gto_norm`, not the `_env` coefficient
//!
//! [`estimate_ke_cutoff_for_omega`] overwrites the extracted contraction
//! coefficients with `gto.gto_norm(l, alpha)` (`rsdf_builder.py:1600`), where
//! [`estimate_rcut`] and [`estimate_ft_rcut`] keep the libcint ones. Phase 13
//! shipped a defect from exactly this confusion (`21.186` against `20.420` Bohr
//! — `13-VERIFICATION.md` defect 2), so the two are kept visibly apart here.

use pyscf_pbc_gto::Cell;
use pyscf_pbc_gto::cutoff::{PgtoOp, bas_angular, extract_pgto_params};

use crate::error::PbcDfError;

/// `OMEGA_MIN` — `rsdf_builder.py:52`.
pub const OMEGA_MIN: f64 = 0.08;

/// `RCUT_THRESHOLD` — `rsdf_builder.py:56`, the `_RangeSeparatedCell` split
/// radius. Recorded for completeness; this port has no `_RangeSeparatedCell`
/// (D-PBC-21/23).
pub const RCUT_THRESHOLD: f64 = 1.0;

/// `gto.gto_norm(l, alpha)` — `pyscf/gto/mole.py:120-155`.
///
/// `pyscf_gto::make_env::gto_norm` is `pub(crate)`, so it is restated here, as
/// `incore::auxcell::gaussian_int` restates its own dependency for the same
/// reason.
fn gto_norm(l: i32, alpha: f64) -> f64 {
    // 1 / sqrt(gaussian_int(2l + 2, 2 alpha))
    let n = 2 * l + 2;
    let h = (f64::from(n) + 1.0) * 0.5;
    let gi = 0.5 * libm::tgamma(h) / (2.0 * alpha).powf(h);
    1.0 / gi.sqrt()
}

/// `aft._estimate_ke_cutoff(alpha, l, c, precision, omega)` — `aft.py:276-288`.
///
/// **Not** `cell._estimate_ke_cutoff`
/// ([`pyscf_pbc_gto::cutoff::estimate_ke_cutoff_pgto`]): that one is derived
/// from the NUCLEAR-ATTRACTION integral and this one from the 4-centre Coulomb
/// repulsion. They differ in `norm_ang` (squared here), in the power of
/// `2*alpha`, and in the iteration's multiplier. Substituting one for the other
/// is silent and wrong.
pub fn estimate_ke_cutoff_pgto_4c(alpha: f64, l: i32, c: f64, precision: f64, omega: f64) -> f64 {
    let lf = f64::from(l);
    let norm_ang = ((2.0 * lf + 1.0) / (4.0 * std::f64::consts::PI)).powi(2);
    let fac = 8.0 * std::f64::consts::PI.powi(5) * c.powi(4) * norm_ang
        / (2.0 * alpha).powf(4.0 * lf + 2.0)
        / precision;
    let exponent = 2.0 * lf - 0.5;
    let mut ecut = 20.0_f64;
    if omega <= 0.0 {
        ecut = (fac * (ecut * 0.5).powf(exponent) + 1.0).ln() * 2.0 * alpha;
        ecut = (fac * (ecut * 0.5).powf(exponent) + 1.0).ln() * 2.0 * alpha;
    } else {
        let theta = 1.0 / (1.0 / (2.0 * alpha) + 1.0 / (2.0 * omega * omega));
        ecut = (fac * (ecut * 0.5).powf(exponent) + 1.0).ln() * theta;
        ecut = (fac * (ecut * 0.5).powf(exponent) + 1.0).ln() * theta;
    }
    ecut
}

/// `estimate_omega_min(cell, precision)` — `rsdf_builder.py:1580-1594`.
///
/// The smallest `ω` for which the attenuated Coulomb potential of a point
/// charge is already below `precision` at `cell.rcut`, using
/// `erfc(z) < exp(-z²)/(z sqrt(pi))`.
pub fn estimate_omega_min(cell: &Cell, precision: Option<f64>) -> f64 {
    let precision = precision.unwrap_or(cell.precision);
    let rcut = cell.rcut;
    let omega = OMEGA_MIN;
    let v = -(precision * rcut * rcut * omega).ln();
    (v.sqrt() / rcut).max(OMEGA_MIN)
}

/// `estimate_ke_cutoff_for_omega(cell, omega, precision)` —
/// `rsdf_builder.py:1595-1605`.
pub fn estimate_ke_cutoff_for_omega(cell: &Cell, omega: f64, precision: Option<f64>) -> f64 {
    let precision = precision.unwrap_or(cell.precision);
    let (exps, _) = extract_pgto_params(cell, PgtoOp::Max);
    exps.iter()
        .enumerate()
        .map(|(i, &e)| {
            let l = bas_angular(cell, i);
            // `cs = gto.gto_norm(ls, exps)` — the extracted coefficients are
            // DISCARDED here. See the module docs.
            estimate_ke_cutoff_pgto_4c(e, l, gto_norm(l, e), precision, omega)
        })
        .fold(f64::NEG_INFINITY, f64::max)
}

/// `estimate_omega_for_ke_cutoff(cell, ke_cutoff, precision)` —
/// `rsdf_builder.py:1606-1620`.
///
/// The `precision *= 1e-2` penalty is upstream's and is load-bearing: it says
/// so in a comment ("estimation based on ∫dk 4π/k² exp(-k²/4ω) sometimes is not
/// enough to converge the 2-electron integrals").
pub fn estimate_omega_for_ke_cutoff(cell: &Cell, ke_cutoff: f64, precision: Option<f64>) -> f64 {
    let precision = precision.unwrap_or(cell.precision) * 1e-2;
    let lmax = (0..cell.mol.nbas)
        .map(|i| bas_angular(cell, i))
        .max()
        .unwrap_or(0);
    let kmax = (ke_cutoff * 2.0).sqrt();
    let log_rest =
        (precision / (16.0 * std::f64::consts::PI.powi(2) * kmax.powi(lmax))).ln();
    (-0.5 * ke_cutoff / log_rest).sqrt()
}

/// `_round_off_to_odd_mesh(mesh)` — `rsdf_builder.py:1405-1416`.
///
/// **This looks trivial and is not.** An EVEN axis breaks the conjugation
/// symmetry between `k` and `-k` (`np.fft.fftfreq` has no `-Gmax` partner for
/// `+Gmax`), and `_make_j3c` uses that symmetry to absorb the auxiliary
/// basis's linear dependency. It also changes `Gv`, and therefore every
/// downstream number.
pub fn round_off_to_odd_mesh(mesh: [usize; 3]) -> [usize; 3] {
    [
        (mesh[0] / 2) * 2 + 1,
        (mesh[1] / 2) * 2 + 1,
        (mesh[2] / 2) * 2 + 1,
    ]
}

/// `_estimate_meshz(cell, precision)` — `rsdf_builder.py:1367-1377`.
///
/// The `z` mesh a 2-D cell with truncated Coulomb needs. Only the `z` axis is
/// returned, and it is floored at `cell.mesh[2]`.
///
/// # Errors
/// Propagates `cutoff_to_mesh`.
pub fn estimate_meshz(cell: &Cell, precision: Option<f64>) -> Result<usize, PbcDfError> {
    let precision = precision.unwrap_or(cell.precision);
    let (exps, _) = extract_pgto_params(cell, PgtoOp::Max);
    let e = exps.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let ke_cut = -precision.ln() * 2.0 * e;
    let meshz = cell.cutoff_to_mesh(ke_cut)?[2];
    let own = cell.try_mesh().map(|m| m[2]).unwrap_or(0);
    Ok(meshz.max(own))
}

/// `estimate_rs_2c2e_rcut(auxcell, omega, precision)` —
/// `rsdf_builder.py:1622-1631`.
pub fn estimate_rs_2c2e_rcut(auxcell: &Cell, omega: f64, precision: Option<f64>) -> f64 {
    let precision = precision.unwrap_or(auxcell.precision);
    let aux_exp = (0..auxcell.mol.nbas)
        .flat_map(|i| pyscf_pbc_gto::cutoff::bas_exp(auxcell, i))
        .fold(f64::INFINITY, f64::min);
    let theta = if omega == 0.0 {
        aux_exp / 2.0
    } else {
        1.0 / (2.0 / aux_exp + omega.powi(-2))
    };
    let fac = 2.0 * std::f64::consts::PI.powf(3.5) / auxcell.vol()
        * aux_exp.powi(-3)
        * theta.powf(-1.5);
    ((fac / auxcell.rcut / precision + 1.0).ln() / theta).sqrt()
}

/// `estimate_rcut(rs_cell, rs_auxcell, omega, precision)` —
/// `rsdf_builder.py:1418-1504`, the `exclude_dd_block = False` branch.
///
/// One radius per ORBITAL shell, for the short-range 3-centre integral. The
/// auxiliary side collapses to its single most diffuse function
/// (`aux_exps.argmin()`), which is why 14-02 could aggregate its own screens
/// per auxiliary ATOM and still be more conservative than upstream.
///
/// **`precision` defaults to `cell.precision * 1e-1`**, not `cell.precision`;
/// upstream's comment says the observed errors run slightly larger than the
/// nominal target.
///
/// This port has no `_RangeSeparatedCell`, so it is called with the PLAIN cell.
/// `measurements/omega.out` records both, and the maxima agree exactly — the
/// split only refines the SMALLER radii.
pub fn estimate_rcut(
    cell: &Cell,
    auxcell: &Cell,
    omega: f64,
    precision: Option<f64>,
) -> Vec<f64> {
    let precision = precision.unwrap_or(cell.precision * 1e-1);
    if cell.mol.nbas == 0 || auxcell.mol.nbas == 0 {
        return vec![0.0];
    }
    if omega == 0.0 {
        // No SR integrals in int3c2e at omega = 0; upstream asserts
        // `dimension == 0` here.
        return vec![0.0];
    }
    let (cell_exps, cs) = extract_pgto_params(cell, PgtoOp::Min);
    let ls: Vec<i32> = (0..cell.mol.nbas).map(|i| bas_angular(cell, i)).collect();

    // `aux_exps = [e.min() for e in rs_auxcell.bas_exps()]`, then argmin.
    let aux_exps: Vec<f64> = (0..auxcell.mol.nbas)
        .map(|i| {
            pyscf_pbc_gto::cutoff::bas_exp(auxcell, i)
                .into_iter()
                .fold(f64::INFINITY, f64::min)
        })
        .collect();
    let aux_min_idx = argmin(&aux_exps);
    let ak = aux_exps[aux_min_idx];
    let lk = bas_angular(auxcell, aux_min_idx);

    let ai_idx = argmin(&cell_exps);
    let ai = cell_exps[ai_idx];
    let li = ls[ai_idx];
    let ci = cs[ai_idx];

    // `ck` normalises the auxiliary basis so `\int chi_k dr = 1`.
    let ck = 1.0 / (4.0 * std::f64::consts::PI)
        / crate::incore::auxcell::gaussian_int(lk + 2, ak);

    let r_init = cell.rcut;
    (0..cell.mol.nbas)
        .map(|j| {
            let aj = cell_exps[j];
            let lj = ls[j];
            let cj = cs[j];
            let aij = ai + aj;
            let lij = li + lj;
            let l3 = f64::from(lij + lk);
            let theta = 1.0 / (omega.powi(-2) + 1.0 / aij + 1.0 / ak);
            let norm_ang = ((2.0 * f64::from(li) + 1.0) * (2.0 * f64::from(lj) + 1.0)).sqrt()
                / (4.0 * std::f64::consts::PI);
            let c1 = ci * cj * ck * norm_ang;
            let sfac = aij * aj / (aij * aj + ai * theta);
            let fl = 2.0;
            let mut fac = 2.0_f64.powi(li)
                * std::f64::consts::PI.powf(2.5)
                * c1
                * theta.powf(l3 - 0.5);
            fac *= 2.0 * std::f64::consts::PI / cell.vol() / theta;
            fac /= aij.powf(f64::from(li) + 1.5) * ak.powf(f64::from(lk) + 1.5) * aj.powi(lj);
            fac *= fl / precision;

            let mut r0 = r_init;
            for _ in 0..2 {
                r0 = ((fac * r0 * (sfac * r0).powf(l3 - 1.0) + 1.0).ln() / (sfac * theta)).sqrt();
            }
            r0
        })
        .collect()
}

/// `estimate_ft_rcut(rs_cell, precision)` — `rsdf_builder.py:1506-1578`, the
/// `exclude_dd_block = False` branch.
///
/// The Schwarz-based radius for the ANALYTIC FT of an AO pair. **`precision`
/// defaults to `cell.precision * 1e-2`** — upstream tightens it by two orders
/// specifically to improve the Hermitian symmetry of MO integrals for post-HF,
/// and Phase 13 measured that asymmetry falling from 5.133e-11 to 2.665e-15
/// when the radius converged.
///
/// The two `r0` iterations are NOT the same expression: the first uses
/// `fl = 2 pi r0 / theta + 1`, the second `fl = 2 pi / vol * r0 / theta`.
/// Copying either one twice changes the answer.
pub fn estimate_ft_rcut(cell: &Cell, precision: Option<f64>) -> Vec<f64> {
    let precision = precision.unwrap_or(cell.precision * 1e-2);
    let (exps, cs) = extract_pgto_params(cell, PgtoOp::Min);
    let ls: Vec<i32> = (0..cell.mol.nbas).map(|i| bas_angular(cell, i)).collect();
    if exps.is_empty() {
        return vec![0.0];
    }
    let ai_idx = argmin(&exps);
    let ai = exps[ai_idx];
    let li = ls[ai_idx];
    let ci = cs[ai_idx];
    let r_init = cell.rcut;

    (0..cell.mol.nbas)
        .map(|j| {
            let aj = exps[j];
            let lj = ls[j];
            let cj = cs[j];
            let aij = ai + aj;
            let lij = li + lj;
            let norm_ang = ((2.0 * f64::from(li) + 1.0) * (2.0 * f64::from(lj) + 1.0)).sqrt()
                / (4.0 * std::f64::consts::PI);
            let c1 = ci * cj * norm_ang;
            let theta = ai * aj / aij;
            let aij1 = aij.powf(-0.5);
            let fac = std::f64::consts::PI.powf(1.5)
                * c1
                * aij1.powi(lij + 3)
                * (2.0 * aij / std::f64::consts::PI).powf(0.25)
                * aij.powi(lij)
                / precision;

            let mut r0 = r_init;
            let dri = aj * aij1 * r0 + 1.0;
            let drj = ai * aij1 * r0 + 1.0;
            let fl = 2.0 * std::f64::consts::PI * r0 / theta + 1.0;
            r0 = ((fac * dri.powi(li) * drj.powi(lj) * fl + 1.0).ln() / theta).sqrt();

            let dri = aj * aij1 * r0 + 1.0;
            let drj = ai * aij1 * r0 + 1.0;
            let fl = 2.0 * std::f64::consts::PI / cell.vol() * r0 / theta;
            ((fac * dri.powi(li) * drj.powi(lj) * fl + 1.0).ln() / theta).sqrt()
        })
        .collect()
}

/// `_guess_omega(cell, kpts, mesh)` — `rsdf_builder.py:1330-1365`, the
/// `dimension > 0` branch.
///
/// Returns `(omega, mesh, ke_cutoff)`. With `mesh = None` the cutoff starts at
/// the empirical `20 * (nao/25 * nkpts)^(-1/3)`, is floored at the cutoff
/// `OMEGA_MIN` needs, and — the part that is easy to drop — is CAPPED by the
/// cutoff the largest usable `omega` implies, because upstream found numerical
/// trouble in the Rys polynomials for SR integrals with `nroots > 3`. The cap
/// only applies when the cell has an `l > 0` shell.
///
/// # Errors
/// Propagates `cutoff_to_mesh` / `mesh_to_cutoff`.
pub fn guess_omega(
    cell: &Cell,
    kpts: &[[f64; 3]],
    mesh: Option<[usize; 3]>,
) -> Result<(f64, [usize; 3], f64), PbcDfError> {
    let a = cell.a;
    if cell.dimension == 0 {
        let m = match mesh {
            Some(m) => m,
            None => cell.try_mesh()?,
        };
        let ke = pyscf_pbc_tools::mesh::mesh_to_cutoff(&a, m)?
            .into_iter()
            .fold(f64::INFINITY, f64::min);
        return Ok((0.0, m, ke));
    }

    let ke_min = estimate_ke_cutoff_for_omega(cell, OMEGA_MIN, None);

    let mesh = match mesh {
        Some(m) => m,
        None => {
            let nkpts = kpts.len().max(1) as f64;
            let nao = cell.mol.nao_nr as f64;
            let mut ke_cutoff = 20.0 * (nao / 25.0 * nkpts).powf(-1.0 / 3.0);
            ke_cutoff = ke_cutoff.max(ke_min);
            // `exps = [e for l, e in zip(ls, bas_exps()) if l != 0]`
            let mut omega_max = f64::INFINITY;
            for i in 0..cell.mol.nbas {
                if bas_angular(cell, i) != 0 {
                    for e in pyscf_pbc_gto::cutoff::bas_exp(cell, i) {
                        omega_max = omega_max.min(e);
                    }
                }
            }
            if omega_max.is_finite() {
                let omega_max = omega_max.sqrt() * 2.0;
                let ke_max = estimate_ke_cutoff_for_omega(cell, omega_max, None);
                ke_cutoff = ke_cutoff.min(ke_max);
            }
            cell.cutoff_to_mesh(ke_cutoff)?
        }
    };

    let ke_cutoff = pyscf_pbc_tools::mesh::mesh_to_cutoff(&a, mesh)?
        .into_iter()
        .take(cell.dimension as usize)
        .fold(f64::INFINITY, f64::min);
    let omega = estimate_omega_for_ke_cutoff(cell, ke_cutoff, None);
    Ok((omega, mesh, ke_cutoff))
}

/// `weighted_coulG_SR(kpt, exx, mesh)` — `rsdf_builder.py:202-203`.
///
/// `weighted_coulG(kpt, False, mesh, -omega)`. **The sign is the convention
/// this port already uses** ([`crate::traits::JkOpts::omega`]: `> 0`
/// long-range, `< 0` short-range, `None` full), so no second convention is
/// introduced.
///
/// # Errors
/// Propagates `get_coulG`.
pub fn weighted_coulg_sr(
    df: &crate::aftdf::Aftdf,
    kpt: [f64; 3],
    mesh: [usize; 3],
    omega: f64,
) -> Result<Vec<f64>, PbcDfError> {
    df.weighted_coulg(kpt, None, mesh, Some(-omega))
}

/// `weighted_coulG_LR(kpt, exx, mesh)` — `rsdf_builder.py:195-200`.
///
/// Upstream's comment explains why this is a DIFFERENCE rather than a direct
/// `+omega` evaluation: "The long range part Coulomb kernel has to be computed
/// as the difference between coulG(cell.omega) - coulG(df.omega). It allows
/// this module to handle the SR- and regular integrals in the same framework."
/// Evaluating `get_coulG(+omega)` instead would be wrong whenever the CELL
/// itself carries an omega (an RSH functional), because then the "full" kernel
/// is not `1/r`.
///
/// # Errors
/// Propagates `get_coulG`.
pub fn weighted_coulg_lr(
    df: &crate::aftdf::Aftdf,
    kpt: [f64; 3],
    exx: Option<pyscf_pbc_gto::ExxDiv>,
    mesh: [usize; 3],
    omega: f64,
) -> Result<Vec<f64>, PbcDfError> {
    let full = df.weighted_coulg(kpt, exx, mesh, None)?;
    let sr = weighted_coulg_sr(df, kpt, mesh, omega)?;
    Ok(full.iter().zip(sr.iter()).map(|(f, s)| f - s).collect())
}

/// `_gaussian_int(cell)` — `rsdf_builder.py:1401-1403`.
///
/// `\int g(r) dr` for every AO, i.e. `ft_ao(cell, G = 0).real`. It is what
/// `rsdf.get_aux_chg` uses to find the CHARGED auxiliary functions, and it must
/// agree with 14-01's monopole convention.
///
/// # Errors
/// Propagates the single-centre FT.
pub fn gaussian_int(cell: &Cell) -> Result<Vec<f64>, PbcDfError> {
    let (re, _im) = crate::ft_ao::single::ft_ao_kpt(&cell.mol, &[[0.0; 3]], [0.0; 3])?;
    Ok(re)
}

fn argmin(v: &[f64]) -> usize {
    let mut best = 0usize;
    for (i, x) in v.iter().enumerate() {
        if *x < v[best] {
            best = i;
        }
    }
    let _ = &best;
    best
}
