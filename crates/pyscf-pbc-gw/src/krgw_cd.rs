//! k-point restricted G0W0 by contour deformation (`pbc/gw/krgw_cd.py`, 704 l).
//!
//! Where AC (19-10, [`crate::krgw_ac`]) continues from the imaginary axis by
//! a Padé fit, CD deforms the integration contour and evaluates on the real
//! axis directly. **It has no fit** — that absence is what makes the AC–CD
//! split a meaningful diagnostic rather than noise (19-11 must-have).
//!
//! Port structure (function names are the upstream ones):
//!
//! * the frequency grid is SHARED with AC: [`crate::sigma::imag_grid`] with
//!   `omega0 = 0.5` is `_get_scaled_legendre_roots` (`krgw_cd.py:562`, same
//!   `x0 = 0.5` map — verified element-wise vs `leggauss` in 19-10).
//! * [`sigma_imag_quad`] ports `get_sigmaI_diag` (`:300`): the imaginary-axis
//!   integral `σ -= Σ_k einsum(g0, W_k)/π`, with `emo = ω − i·η·sign − e_m`
//!   and `g0 = wts·emo/(emo² + freqs²)`. `W` (the DF-built screened
//!   interaction) is INJECTED, not built here — the `sr_loop` dielectric
//!   build (`get_WmnI_diag`, `:186`) needs a live GDF object and is the
//!   documented seam, not a silent omission.
//! * [`sigma_residue`] ports the pole-selection half of `get_sigmaR_diag`
//!   (`:339`): poles strictly between `ef` and `ω` are enclosed (`fm = ±1`),
//!   each contributing its injected screened vertex. The `Lpq` dielectric
//!   rebuild per enclosed pole is the same documented seam as above.
//! * [`kernel_krgw_cd`] ports `kernel` (`:46`): per-orbital Newton solve
//!   (`tol = 1e-6`, `maxiter = 50`, `scipy.optimize.newton`). Upstream
//!   REFUSES linearized CD (`:116-121`, "linearization with CD leads to wrong
//!   quasiparticle energy") — this port refuses identically, never silently
//!   linearizing.
//! * Finite-size (`fc`) head/wing corrections (`Del_00`/`Del_P0`, the `q → 0`
//!   machinery) ride as an optional injected per-(k, window-orbital)
//!   correction; `None` means no correction, stated at the call site.
//!
//! Every reduction routes through [`pyscf_algebra::oracle_sum`] (planar
//! `re`/`im` for complex). Host loops only (D-PBC-29 clause 2) — no `#[cube]`
//! (CubeCL manual read 2026-09-13: computation engines here are host
//! orchestration over injected DF data, no new device kernels needed).

use num_complex::Complex64;

use crate::error::PbcGwError;
use crate::krgw_ac::qp_newton;
use crate::types::{GwConfig, GwRoute, QpResult};

/// Contour-deformation configuration: the contour parameters are PART OF THE
/// METHOD and are pinned on both sides (19-11 must-have).
#[derive(Debug, Clone)]
pub struct CdConfig {
    /// Imaginary-axis grid size (`nw`, default 100 — pinned with AC).
    pub nomega: usize,
    /// Broadening `η` (upstream `gw_gw_GW_eta`, default `1e-3` — pinned).
    pub eta: f64,
    /// QP Newton tolerance (upstream `tol = 1e-6` — pinned).
    pub conv_tol: f64,
    /// QP Newton iterations (upstream `maxiter = 50` — pinned).
    pub max_cycle: usize,
}

impl Default for CdConfig {
    fn default() -> Self {
        Self {
            nomega: 100,
            eta: 1e-3,
            conv_tol: 1e-6,
            max_cycle: 50,
        }
    }
}

/// A screened vertex for one enclosed pole (the injected half of
/// `get_sigmaR_diag`: the `Lpq`-built residue evaluated at the pole).
#[derive(Debug, Clone, Copy)]
pub struct CdPole {
    /// Pole energy (mean-field energy of the enclosed state).
    pub energy: f64,
    /// Screened vertex `W` at the pole (already contracted to the diagonal).
    pub vertex: Complex64,
}

/// Imaginary-axis integral half of the CD self-energy (`get_sigmaI_diag`).
///
/// `mf_energy_k[k][m]` are the mean-field energies; `w_k[k][m][n]` the
/// injected screened interaction on (mo, freqs); `sign_k[k][m] =
/// sign(ef − e_m)`; `freqs`/`wts` the pinned scaled-Legendre grid. Returns the
/// complex `σ^I(ω)` accumulated through `oracle_sum` (planar parts).
pub fn sigma_imag_quad(
    omega: f64,
    mf_energy_k: &[Vec<f64>],
    w_k: &[Vec<Vec<Complex64>>],
    sign_k: &[Vec<f64>],
    freqs: &[f64],
    wts: &[f64],
    eta: f64,
) -> Result<Complex64, PbcGwError> {
    let nk = mf_energy_k.len();
    if w_k.len() != nk || sign_k.len() != nk {
        return Err(PbcGwError::ShapeMismatch {
            expected: nk,
            got: w_k.len().min(sign_k.len()),
        });
    }
    if freqs.len() != wts.len() || freqs.is_empty() {
        return Err(PbcGwError::ShapeMismatch {
            expected: freqs.len(),
            got: wts.len(),
        });
    }
    let nw = freqs.len();
    let mut re_terms: Vec<f64> = Vec::new();
    let mut im_terms: Vec<f64> = Vec::new();
    for k in 0..nk {
        let nmo = mf_energy_k[k].len();
        if w_k[k].len() != nmo || sign_k[k].len() != nmo {
            return Err(PbcGwError::ShapeMismatch {
                expected: nmo,
                got: w_k[k].len(),
            });
        }
        for m in 0..nmo {
            if w_k[k][m].len() != nw {
                return Err(PbcGwError::ShapeMismatch {
                    expected: nw,
                    got: w_k[k][m].len(),
                });
            }
            // emo = ω − i·η·sign − e_m (complex scalar per state).
            let emo = Complex64::new(omega - mf_energy_k[k][m], -eta * sign_k[k][m]);
            for n in 0..nw {
                // g0 = wts·emo/(emo² + freqs²); σ -= g0·W/π.
                let denom = emo * emo + Complex64::new(freqs[n] * freqs[n], 0.0);
                if denom.norm() == 0.0 {
                    return Err(PbcGwError::PadeFailure {
                        reason: format!(
                            "vanishing CD quadrature denominator at k={k}, m={m}, n={n}"
                        ),
                    });
                }
                let g0 = emo * Complex64::new(wts[n], 0.0) / denom;
                let term = g0 * w_k[k][m][n] / Complex64::new(std::f64::consts::PI, 0.0);
                re_terms.push(-term.re);
                im_terms.push(-term.im);
            }
        }
    }
    Ok(Complex64::new(
        pyscf_algebra::oracle_sum(&re_terms),
        pyscf_algebra::oracle_sum(&im_terms),
    ))
}

/// Residue half of the CD self-energy (the pole-selection of
/// `get_sigmaR_diag`).
///
/// Enclosed poles: `(ef, ω)` when `ω > ef` (`fm = +1`), `(ω, ef)` otherwise
/// (`fm = −1`) — upstream `:349-355` verbatim, endpoints EXCLUDED (strict
/// inequalities). Returns `fm · Σ_poles vertex`, ordered through
/// `oracle_sum`.
pub fn sigma_residue(omega: f64, ef: f64, poles: &[CdPole]) -> Complex64 {
    let fm = if omega > ef { 1.0 } else { -1.0 };
    let (lo, hi) = if omega > ef { (ef, omega) } else { (omega, ef) };
    let mut re_terms: Vec<f64> = Vec::new();
    let mut im_terms: Vec<f64> = Vec::new();
    for pole in poles {
        if pole.energy > lo && pole.energy < hi {
            re_terms.push(fm * pole.vertex.re);
            im_terms.push(fm * pole.vertex.im);
        }
    }
    Complex64::new(
        pyscf_algebra::oracle_sum(&re_terms),
        pyscf_algebra::oracle_sum(&im_terms),
    )
}

/// Full CD self-energy on the real axis: `σ(ω) = σ^I(ω) + σ^R(ω)`
/// (`get_sigma_diag`, `:140`), plus the optional finite-size correction.
#[allow(clippy::too_many_arguments)]
pub fn sigma_cd_real(
    omega: f64,
    mf_energy_k: &[Vec<f64>],
    w_k: &[Vec<Vec<Complex64>>],
    sign_k: &[Vec<f64>],
    poles: &[CdPole],
    freqs: &[f64],
    wts: &[f64],
    ef: f64,
    eta: f64,
    fc_corr: Option<Complex64>,
) -> Result<Complex64, PbcGwError> {
    let sigma_i = sigma_imag_quad(omega, mf_energy_k, w_k, sign_k, freqs, wts, eta)?;
    let sigma_r = sigma_residue(omega, ef, poles);
    Ok(sigma_i + sigma_r + fc_corr.unwrap_or(Complex64::new(0.0, 0.0)))
}

/// Restricted G0W0-CD driver over the orbital window `cfg.orlo..cfg.orhi`
/// (ports `kernel`, `:46-137`).
///
/// `mf_energy[k][q]` are absolute mean-field energies; `vk_diag`/`vmf_diag`
/// the MO-basis exchange / mean-field-exchange diagonals; `w_k[k][m][n]` the
/// injected screened interaction (full-mo × grid); `poles_k[k][oi]` the
/// enclosed-pole vertices per window orbital. Solves the QP equation per
/// (k, p) by Newton and returns [`GwRoute::ContourDeformation`] — never an AC
/// number (19-01 Gate C, per route).
///
/// # Errors
///
/// * `PbcGwError::NotYetImplemented` when `linearized` is set — upstream
///   raises `NotImplementedError` there too (`:116-121`).
#[allow(clippy::too_many_arguments)]
pub fn kernel_krgw_cd(
    mf_energy: &[Vec<f64>],
    vk_diag: &[Vec<f64>],
    vmf_diag: &[Vec<f64>],
    w_k: &[Vec<Vec<Complex64>>],
    sign_k: &[Vec<f64>],
    poles_k: &[Vec<Vec<CdPole>>],
    freqs: &[f64],
    wts: &[f64],
    ef: f64,
    orbs: std::ops::Range<usize>,
    cfg: &GwConfig,
    cd: &CdConfig,
    linearized: bool,
    fc_corr: Option<&[Vec<Complex64>]>,
) -> Result<QpResult, PbcGwError> {
    if linearized {
        return Err(PbcGwError::NotYetImplemented {
            module: "gw/krgw_cd linearized (upstream raises NotImplementedError: linearization with CD leads to wrong QP energy)",
        });
    }
    let nk = mf_energy.len();
    if vk_diag.len() != nk || vmf_diag.len() != nk {
        return Err(PbcGwError::ShapeMismatch {
            expected: nk,
            got: vk_diag.len(),
        });
    }
    if w_k.len() != nk || sign_k.len() != nk || poles_k.len() != nk {
        return Err(PbcGwError::ShapeMismatch {
            expected: nk,
            got: w_k.len(),
        });
    }
    for pk in poles_k {
        if pk.len() != orbs.len() {
            return Err(PbcGwError::ShapeMismatch {
                expected: orbs.len(),
                got: pk.len(),
            });
        }
    }
    if freqs.len() != cd.nomega || wts.len() != cd.nomega {
        return Err(PbcGwError::ShapeMismatch {
            expected: cd.nomega,
            got: freqs.len().min(wts.len()),
        });
    }
    let mut qp = Vec::with_capacity(nk * orbs.len());
    for k in 0..nk {
        for (oi, p) in orbs.clone().enumerate() {
            if p >= mf_energy[k].len() {
                return Err(PbcGwError::ShapeMismatch {
                    expected: p + 1,
                    got: mf_energy[k].len(),
                });
            }
            let fc = fc_corr.map(|fc| {
                if k < fc.len() && oi < fc[k].len() {
                    fc[k][oi]
                } else {
                    Complex64::new(0.0, 0.0)
                }
            });
            // Real-axis evaluation directly — NO Padé fit (the CD difference).
            let sigma_r = |w: f64| {
                sigma_cd_real(
                    w,
                    mf_energy,
                    w_k,
                    sign_k,
                    &poles_k[k][oi],
                    freqs,
                    wts,
                    ef,
                    cd.eta,
                    fc,
                )
                .map(|z| z.re)
                .unwrap_or(f64::NAN)
            };
            let ep = mf_energy[k][p];
            let e = qp_newton(
                ep,
                sigma_r,
                vk_diag[k][p],
                vmf_diag[k][p],
                cd.conv_tol,
                cd.max_cycle,
            )?;
            if !e.is_finite() {
                return Err(PbcGwError::QpNotConverged {
                    cycles: cd.max_cycle,
                });
            }
            qp.push(e);
        }
    }
    let _ = cfg;
    Ok(QpResult {
        qp_energy: qp,
        route: GwRoute::ContourDeformation,
        converged: true,
    })
}

/// Per-quasiparticle `|E_AC − E_CD|` split (19-01 Task 3 / 19-11 Task 2).
///
/// Both slices must carry their routes; equal routes are REFUSED
/// ([`PbcGwError::RouteBlindComparison`]) — a route-blind number measures the
/// approximation split, not the port.
pub fn ac_cd_split(ac: &QpResult, cd: &QpResult) -> Result<Vec<f64>, PbcGwError> {
    if ac.route == cd.route {
        return Err(PbcGwError::RouteBlindComparison);
    }
    if ac.qp_energy.len() != cd.qp_energy.len() {
        return Err(PbcGwError::ShapeMismatch {
            expected: ac.qp_energy.len(),
            got: cd.qp_energy.len(),
        });
    }
    Ok(ac
        .qp_energy
        .iter()
        .zip(cd.qp_energy.iter())
        .map(|(a, c)| (a - c).abs())
        .collect())
}
