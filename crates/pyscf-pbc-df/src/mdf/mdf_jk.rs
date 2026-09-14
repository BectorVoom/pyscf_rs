//! `mdf_jk` — J and K as the SUM of the two halves
//! (`pyscf/pbc/df/mdf_jk.py:68-96`), plan 14-06.
//!
//! ```text
//! vj = df_jk.get_j_kpts(cderi)  +  aft_jk.get_j_kpts(plane waves)
//! vk = df_jk.get_k_kpts(cderi, exxdiv = None)  +  aft_jk.get_k_kpts(pw, exxdiv)
//! ```
//!
//! **`exxdiv` belongs to exactly one of the two summands.** `mdf_jk.py:61-62`
//! passes `None` to the `cderi` half and the caller's `exxdiv` to the
//! plane-wave half, and the reason is structural rather than stylistic: AFTDF
//! folds the `G + k = 0` correction into `coulG`, so it has to be applied where
//! `coulG` is (Phase 13); GDF applies `_ewald_exxdiv_for_G0` to the assembled
//! matrix (14-04). Applying it in both would double it; applying it in the
//! `cderi` half instead would apply it to a fitted quantity. Phase 13 measured
//! this term at ~96 % of the MATRIX difference between builders, so getting it
//! wrong is loud in `vk` and quiet in the energy.

use crate::df_jk::KMats;
use crate::error::PbcDfError;
use crate::mdf::Mdf;
use crate::traits::{JkOpts, JkResult};

fn add_into(dst: &mut [KMats], src: &[KMats]) {
    for (a, b) in dst.iter_mut().zip(src.iter()) {
        for (x, y) in a.iter_mut().zip(b.iter()) {
            for i in 0..x.re.len() {
                x.re[i] += y.re[i];
                x.im[i] += y.im[i];
            }
        }
    }
}

/// `mdf_jk.get_j_kpts` — `mdf_jk.py:68-71`.
///
/// # Errors
/// Propagates both halves.
pub fn get_j_kpts(df: &Mdf, dms: &[KMats], kpts: &[[f64; 3]]) -> Result<Vec<KMats>, PbcDfError> {
    let mut vj = crate::gdf::jk::get_j_kpts(df.gdf()?, dms, kpts)?;
    let pw = crate::aft_jk::get_j_kpts(df.aftdf()?, dms, kpts, None)?;
    add_into(&mut vj, &pw);
    Ok(vj)
}

/// `mdf_jk.get_j_kpts` at `kpts_band` — `mdf_jk.py:68-71` composed from the
/// two halves' own band-k-point routes (both closed by plan 17-10 Task 4).
///
/// # Errors
/// Propagates both halves.
pub fn get_j_kpts_band(
    df: &Mdf,
    dms: &[KMats],
    kpts: &[[f64; 3]],
    kpts_band: &[[f64; 3]],
) -> Result<Vec<KMats>, PbcDfError> {
    let (band_gdf, union) = df.band_gdf(kpts, kpts_band)?;
    let mut vj = crate::gdf::jk::get_j_kpts_band(&band_gdf, &union, dms, kpts, kpts_band)?;
    let pw = crate::aft_jk::get_j_kpts_band(df.aftdf()?, dms, kpts, kpts_band, None)?;
    add_into(&mut vj, &pw);
    Ok(vj)
}

/// `mdf_jk.get_k_kpts` — `mdf_jk.py:74-83`.
///
/// # Errors
/// Propagates both halves; refuses an `exxdiv` upstream also refuses.
pub fn get_k_kpts(
    df: &Mdf,
    dms: &[KMats],
    kpts: &[[f64; 3]],
    exxdiv: Option<pyscf_pbc_gto::ExxDiv>,
) -> Result<Vec<KMats>, PbcDfError> {
    check_exxdiv(exxdiv)?;
    // `None` to the cderi half — see the module docs.
    let mut vk = crate::gdf::jk::get_k_kpts(df.gdf()?, dms, kpts, None)?;
    let pw = crate::aft_jk::get_k_kpts(df.aftdf()?, dms, kpts, exxdiv, None)?;
    add_into(&mut vk, &pw);
    Ok(vk)
}

/// `mdf_jk.get_k_kpts` at `kpts_band` — same composition as
/// [`get_k_kpts`], routed through both halves' band-k-point variants.
///
/// # Errors
/// Propagates both halves; refuses an `exxdiv` upstream also refuses.
pub fn get_k_kpts_band(
    df: &Mdf,
    dms: &[KMats],
    kpts: &[[f64; 3]],
    kpts_band: &[[f64; 3]],
    exxdiv: Option<pyscf_pbc_gto::ExxDiv>,
) -> Result<Vec<KMats>, PbcDfError> {
    check_exxdiv(exxdiv)?;
    let (band_gdf, union) = df.band_gdf(kpts, kpts_band)?;
    // `None` to the cderi half — see the module docs.
    let mut vk = crate::gdf::jk::get_k_kpts_band(&band_gdf, &union, dms, kpts, kpts_band, None)?;
    let pw = crate::aft_jk::get_k_kpts_band(df.aftdf()?, dms, kpts, kpts_band, exxdiv, None)?;
    add_into(&mut vk, &pw);
    Ok(vk)
}

pub(crate) fn check_exxdiv(exxdiv: Option<pyscf_pbc_gto::ExxDiv>) -> Result<(), PbcDfError> {
    if !matches!(exxdiv, None | Some(pyscf_pbc_gto::ExxDiv::Ewald)) {
        return Err(PbcDfError::Core(pyscf_core::PyscfRsError::Core(
            pyscf_core::CoreError::InvalidMolecule(format!(
                "MDF does not support exxdiv {exxdiv:?}; it must be Ewald or None \
                 (mdf_jk.py:76-79 raises the same way)"
            )),
        )));
    }
    Ok(())
}

/// `MDF.get_jk(..., omega)` — upstream's RSH branch, `mdf.py:181-199`,
/// mirrored branch for branch:
///
/// * `omega > 0` and (`dimension >= 2` **or** `low_dim_ft_type !=
///   'inf_vacuum'`) — **AFTDF** at the omega-derived mesh
///   ([`crate::gdf::jk::rsh_aftdf`]). The condition is `or` here where
///   `GDF.get_jk` has `and` (`df.py:470-471`); that is upstream's text and it
///   is kept as written. Upstream's own caveat: "changing to AFT integrator may
///   cause small difference to the MDF integrator".
/// * otherwise `mydf = self` and [`Mdf::range_coulomb`], which admits only
///   `omega < 0` on a periodic cell (so `omega == 0` is an error, as upstream's
///   `assert omega < 0` makes it). The short-range copy runs `_RSMDFBuilder`
///   on a cell carrying `cell.omega = omega`, and its plane-wave half reads the
///   same attenuated `coulG` through `MDF.weighted_coulG`.
///
/// # Errors
/// * `omega >= 0` where upstream's `range_coulomb` asserts `omega < 0`;
/// * [`pyscf_core::PyscfRsError::NotYetImplemented`] where upstream would run
///   `_CCMDFBuilder` on the attenuated cell (`prefer_ccdf = true` — this port's
///   `Mdf` default — or a long-range request on a 0-D cell, `mdf.py:126`);
/// * propagates the AFTDF or range-separated MDF build.
pub fn get_jk_rsh(
    df: &Mdf,
    dms: &[KMats],
    kpts: &[[f64; 3]],
    opts: JkOpts<'_>,
    omega: f64,
) -> Result<JkResult, PbcDfError> {
    let inner = JkOpts {
        omega: None,
        ..opts
    };
    let cell = &df.cell;
    if omega > 0.0
        && (cell.dimension >= 2 || cell.low_dim_ft_type != pyscf_pbc_gto::LowDimFtType::InfVacuum)
    {
        let aft = crate::gdf::jk::rsh_aftdf(cell, &df.kpts, omega)?;
        return crate::aft_jk::get_jk(&aft, dms, kpts, inner);
    }
    if cell.dimension != 0 && omega >= 0.0 {
        // `df.py:520-521` asserts before anything is built.
        df.range_coulomb(omega)?;
    }
    if omega > 0.0 {
        return Err(PbcDfError::Core(
            pyscf_core::PyscfRsError::NotYetImplemented {
                phase: 20,
                what: "MDF.get_jk(omega > 0) on a 0-D inf_vacuum cell — upstream \
                       builds the long-range tensor with _CCMDFBuilder on \
                       cell.omega > 0 (mdf.py:126-128), which this port has not ported",
            },
        ));
    }
    if df.prefer_ccdf {
        return Err(PbcDfError::Core(
            pyscf_core::PyscfRsError::NotYetImplemented {
                phase: 20,
                what: "MDF.get_jk(omega < 0) with prefer_ccdf = true — upstream \
                       builds the short-range tensor with _CCMDFBuilder on \
                       cell.omega < 0 (mdf.py:126-128); only the _RSMDFBuilder \
                       route (prefer_ccdf = false, upstream's default) is ported \
                       for an attenuated cell",
            },
        ));
    }
    let rsh = df.range_coulomb(omega)?;
    get_jk(&rsh, dms, kpts, inner)
}

/// `MDF.get_jk` — the [`crate::traits::PeriodicDf`] entry point
/// (`mdf.py:180-215`).
///
/// # Errors
/// Propagates both halves, and [`get_jk_rsh`] for a set `omega`. `kpts_band`
/// was closed by plan 17-10 Task 4 — both halves now rebuild what they need
/// (GDF's `_cderi` over the k-point union; AFTDF needs no rebuild at all,
/// since its FT loop takes an arbitrary k-point list directly).
pub fn get_jk(
    df: &Mdf,
    dms: &[KMats],
    kpts: &[[f64; 3]],
    opts: JkOpts<'_>,
) -> Result<JkResult, PbcDfError> {
    // `mdf.py:181` — `if omega is not None`. Unlike `GDF.get_jk`, a zero
    // omega is NOT the plain request here; see [`get_jk_rsh`].
    if let Some(omega) = opts.omega {
        return get_jk_rsh(df, dms, kpts, opts, omega);
    }
    if opts.kpts_band.is_some() && !crate::df_jk::band_is_kpts(opts.kpts_band, kpts) {
        let kpts_band = opts.kpts_band.expect("checked Some above");
        return Ok(JkResult {
            vj: if opts.with_j {
                Some(get_j_kpts_band(df, dms, kpts, kpts_band)?)
            } else {
                None
            },
            vk: if opts.with_k {
                Some(get_k_kpts_band(df, dms, kpts, kpts_band, opts.exxdiv)?)
            } else {
                None
            },
        });
    }
    Ok(JkResult {
        vj: if opts.with_j {
            Some(get_j_kpts(df, dms, kpts)?)
        } else {
            None
        },
        vk: if opts.with_k {
            Some(get_k_kpts(df, dms, kpts, opts.exxdiv)?)
        } else {
            None
        },
    })
}
