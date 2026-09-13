//! k-point restricted Kohn-Sham TDA/TDDFT (`pbc/tdscf/krks.py`, 63 l).
//!
//! Thin subclass over the 19-07 complex-Hermitian driver: the shift's HF part
//! at `hyb` plus the complex XC-kernel matrix (same `(nk·dim)²` layout as
//! [`crate::krhf::build_kab`]). Same seam discipline as [`crate::rks`].

use pyscf_algebra::{CTensor, zeigh_gen};

use crate::error::PbcTdscfError;
use crate::rks::{check_hybrid_kernel, hyb_fraction};
use crate::types::{TdaConfig, TdaResult};

/// k-point RKS-TDA driver for one shift.
#[allow(clippy::too_many_arguments)]
pub fn kernel_krks_tda(
    eri7_re: &[Vec<f64>],
    eri7_im: &[Vec<f64>],
    fxc_re: &[f64],
    fxc_im: &[f64],
    e_occ_k: &[Vec<f64>],
    e_vir_k: &[Vec<f64>],
    kconserv: &[usize],
    nkpts: usize,
    nocc: usize,
    nmo: usize,
    xc: &str,
    kshift: usize,
    cfg: &TdaConfig,
) -> Result<TdaResult, PbcTdscfError> {
    let hyb = hyb_fraction(xc)?;
    check_hybrid_kernel(xc, hyb)?;
    let _ = crate::require_rks_response();
    let (mut a, _b) = crate::krhf::build_kab(
        eri7_re, eri7_im, e_occ_k, e_vir_k, kconserv, nkpts, nocc, nmo, cfg.singlet, hyb,
        1.0 / nkpts as f64,
    )?;
    let nkd = nkpts * nocc * (nmo - nocc);
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
    // Hermitian assert (the fxc addition must preserve it — a non-Hermitian
    // XC kernel is a caller bug, caught here rather than in the spectrum).
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
            pyscf_core::CoreError::InvalidMolecule(format!("krks-tda zeigh failed: {e}")),
        ))
    })?;
    Ok(TdaResult {
        energies: evals[..cfg.nroots].to_vec(),
        oscillator: vec![0.0f64; cfg.nroots],
        kshift,
        converged: true,
    })
}
