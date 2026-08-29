//! `mdf_ao2mo` — MO integrals as the SUM of the two halves
//! (`pyscf/pbc/df/mdf_ao2mo.py:31-90`), plan 14-06.
//!
//! Same shape as [`crate::mdf::mdf_jk`]: `aft_ao2mo` plus `df_ao2mo`, both
//! already shipped (13-06 + 14-05). Nothing new is contracted here.

use pyscf_algebra::CTensor;

use crate::df_ao2mo::{Eri, Eri7d, MoCoeff, MoKpts, PairDims};
use crate::error::PbcDfError;
use crate::mdf::Mdf;

fn add_ct(dst: &mut CTensor, src: &CTensor) {
    for i in 0..dst.re.len() {
        dst.re[i] += src.re[i];
        dst.im[i] += src.im[i];
    }
}

fn kvec(df: &Mdf, kidx: [usize; 4]) -> Result<[[f64; 3]; 4], PbcDfError> {
    let n = df.kpts.len();
    let mut out = [[0.0; 3]; 4];
    for (o, &i) in out.iter_mut().zip(kidx.iter()) {
        if i >= n {
            return Err(PbcDfError::Core(pyscf_core::PyscfRsError::Core(
                pyscf_core::CoreError::InvalidMolecule(format!(
                    "mdf_ao2mo: k-point index {i} is out of range for {n} k-points"
                )),
            )));
        }
        *o = df.kpts[i];
    }
    Ok(out)
}

/// `mdf_ao2mo.get_eri(mydf, kpts, compact)` — `mdf_ao2mo.py:31-39`.
///
/// Returned in the `s1` layout on both axes: the two summands do not agree on
/// a packing (`aft_ao2mo.get_eri` has no `s4` branch to match `df_ao2mo`'s),
/// and unpacking is exact, so the sum is formed once, in one shape.
///
/// # Errors
/// Propagates both halves.
pub fn get_eri(df: &Mdf, kidx: [usize; 4]) -> Result<Eri, PbcDfError> {
    let k = kvec(df, kidx)?;
    let mut e = crate::df_ao2mo::get_eri(df.gdf()?, kidx, false)?;
    let pw = crate::pbc_ao2mo::aft_get_eri(df.aftdf()?, k)?;
    add_ct(&mut e.data, &pw);
    Ok(e)
}

/// `mdf_ao2mo.general(mydf, mo_coeffs, kpts, compact)` — `mdf_ao2mo.py:42-52`.
///
/// # Errors
/// Propagates both halves.
pub fn general(df: &Mdf, mos: [&MoCoeff; 4], kidx: [usize; 4]) -> Result<Eri, PbcDfError> {
    let k = kvec(df, kidx)?;
    let mut e = crate::df_ao2mo::general(df.gdf()?, mos, kidx, false)?;
    let pw = crate::pbc_ao2mo::aft_general(df.aftdf()?, mos, k)?;
    add_ct(&mut e.data, &pw.data);
    debug_assert_eq!(e.row, PairDims::plain(mos[0].nmo, mos[1].nmo));
    Ok(e)
}

/// `mdf_ao2mo.ao2mo_7d(mydf, mo_coeff_kpts, kpts, factor)` —
/// `mdf_ao2mo.py:54-90`. The index contract is [`crate::df_ao2mo`]'s.
///
/// # Errors
/// Propagates both halves.
pub fn ao2mo_7d(df: &Mdf, mos: MoKpts<'_>, factor: f64) -> Result<Eri7d, PbcDfError> {
    let mut e = crate::df_ao2mo::ao2mo_7d(df.gdf()?, mos, factor)?;
    let pw = crate::pbc_ao2mo::aft_ao2mo_7d(df.aftdf()?, mos, factor)?;
    add_ct(&mut e.data, &pw.data);
    Ok(e)
}
