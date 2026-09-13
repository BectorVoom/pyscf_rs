//! k-point unrestricted Kohn-Sham TDA/TDDFT (`pbc/tdscf/kuks.py`, 41 l).
//!
//! Thin subclass over the 19-08 coupled k-driver: spin-resolved HF parts at
//! `hyb` plus spin-resolved complex XC-kernel blocks in the coupled
//! `(nk·(da+db))²` layout. Same seam discipline as [`crate::rks`].

use pyscf_algebra::{CTensor, zeigh_gen};

use crate::error::PbcTdscfError;
use crate::rks::{check_hybrid_kernel, hyb_fraction};
use crate::types::{TdaConfig, TdaResult};
use crate::uhf::UhfDims;

/// k-point UKS-TDA driver for one shift.
///
/// `fxc` is the FULL `(nk·(da+db))²` XC-kernel matrix (all k-couplings and
/// spin sectors, as the grid contraction produces it) added directly onto
/// the HF part — no placement logic that could silently drop k-couplings.
#[allow(clippy::too_many_arguments)]
pub fn kernel_kuks_tda(
    eri7_aa_re: &[Vec<f64>],
    eri7_aa_im: &[Vec<f64>],
    eri7_ab_re: &[Vec<f64>],
    eri7_ab_im: &[Vec<f64>],
    eri7_bb_re: &[Vec<f64>],
    eri7_bb_im: &[Vec<f64>],
    fxc_re: &[f64],
    fxc_im: &[f64],
    e_occ_a: &[Vec<f64>],
    e_vir_a: &[Vec<f64>],
    e_occ_b: &[Vec<f64>],
    e_vir_b: &[Vec<f64>],
    kconserv: &[usize],
    nkpts: usize,
    dims: UhfDims,
    nmo_a: usize,
    nmo_b: usize,
    xc: &str,
    kshift: usize,
    cfg: &TdaConfig,
) -> Result<TdaResult, PbcTdscfError> {
    let hyb = hyb_fraction(xc)?;
    check_hybrid_kernel(xc, hyb)?;
    let _ = crate::require_rks_response();
    let (mut a, _b) = crate::kuhf::build_kuab(
        eri7_aa_re, eri7_aa_im, eri7_ab_re, eri7_ab_im, eri7_bb_re, eri7_bb_im,
        e_occ_a, e_vir_a, e_occ_b, e_vir_b, kconserv, nkpts, dims, nmo_a, nmo_b,
        hyb, 1.0 / nkpts as f64,
    )?;
    let (da, db) = (dims.da(), dims.db());
    let nkd = nkpts * (da + db);
    if fxc_re.len() != nkd * nkd || fxc_im.len() != nkd * nkd {
        return Err(PbcTdscfError::ShapeMismatch { expected: nkd * nkd, got: fxc_re.len().min(fxc_im.len()) });
    }
    for i in 0..nkd * nkd {
        a.re[i] += fxc_re[i];
        a.im[i] += fxc_im[i];
    }
    if cfg.nroots > nkd {
        return Err(PbcTdscfError::TooManyRoots { nroots: cfg.nroots, dim: nkd });
    }
    let mut asym = 0.0f64;
    for i in 0..nkd {
        for j in 0..nkd {
            asym = asym.max((a.re[i * nkd + j] - a.re[j * nkd + i]).abs());
            asym = asym.max((a.im[i * nkd + j] + a.im[j * nkd + i]).abs());
        }
    }
    let scale: f64 = a.re.iter().chain(a.im.iter()).map(|x| x.abs()).fold(0.0, f64::max);
    if asym > 1e-8 * scale.max(1.0) {
        return Err(PbcTdscfError::ShapeMismatch { expected: 0, got: (asym * 1e12) as usize });
    }
    let ident = CTensor {
        re: {
            let mut s = vec![0.0f64; nkd * nkd];
            for i in 0..nkd {
                s[i * nkd + i] = 1.0;
            }
            s
        },
        im: vec![0.0f64; nkd * nkd],
    };
    let (evals, _) = zeigh_gen(&a, &ident, nkd).map_err(|e| {
        PbcTdscfError::Core(pyscf_core::PyscfRsError::Core(
            pyscf_core::CoreError::InvalidMolecule(format!("kuks-tda zeigh failed: {e}")),
        ))
    })?;
    Ok(TdaResult {
        energies: evals[..cfg.nroots].to_vec(),
        oscillator: vec![0.0f64; cfg.nroots],
        kshift,
        converged: true,
    })
}
