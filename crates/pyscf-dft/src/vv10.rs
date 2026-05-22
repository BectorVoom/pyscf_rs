//! DFT-06: VV10 non-local correlation (NLC) — the `_vv10nlc` double-loop
//! port + `nr_nlc_vxc` orchestration over a coarser `nlcgrids`.
//!
//! Source: `pyscf/dft/numint.py` (Apache-2.0):
//!   - `_vv10nlc`    (numint.py:471-555) — the per-outer-grid-point double-
//!     loop over the inner (vv) grid. We port the **documented pure-Python
//!     reference** (the commented block at numint.py:526-538), NOT the C
//!     `libdft.VXC_vv10nlc` at numint.py:539 (Pitfall 4: that symbol is part
//!     of PySCF's own `libdft` C extension and lives in NO sibling crate —
//!     cintx/libxc_rs/xcfun_rs all lack it; the pure-Python double-loop in
//!     the same file IS the byte-exact reference).
//!   - `nr_nlc_vxc`  (numint.py:1347-1416) — builds ρ+∇ρ on the (coarser)
//!     `nlcgrids`, calls `_vv10nlc` with outer == inner == the nlcgrids
//!     (numint.py:1394), accumulates `excsum = Σ den·exc`, and back-contracts
//!     the GGA Vxc into an `nao × nao` matrix.
//!
//! ## Constants (RESEARCH §4 + A1)
//!   `Kvv = Bvv·1.5·π·(9π)^(−1/6)`; `Beta = ((3/Bvv²)^0.75)/32`.
//!   The bare `'VV10'` default is `Bvv = 5.9`, `Cvv = 0.0093` (A1). The
//!   *authoritative* per-functional source is `libxc_rs::NlcCoefficients`
//!   (`nlc_coeff()`), queried only under `--features libxc`; the default
//!   (xcfun) build hardcodes the bare-VV10 pair (A1 — "hardcode only the
//!   bare 'VV10' default"). libxc_rs is NEVER compiled on the default path.
//!
//! ## Bit-exactness (T-04-07b)
//! Every inner-grid reduction (F, U, W) goes through
//! `pyscf_algebra::oracle_sum` (FMA-free ordered reduction, Pitfall 3/9), so
//! the NLC energy/potential is reproducible bit-for-bit under release-oracle.

use pyscf_algebra::{oracle_dot, oracle_sum};
use pyscf_core::{Density, Mole, PyscfRsError};
use pyscf_grids::Grids;

use crate::error::DftError;
use crate::numint::{NumInt, XcType};

/// VV10 non-local-correlation coefficients `(Bvv, Cvv)`.
///
/// Mirrors upstream `nlc_pars` (the `(Bvv, Cvv)` tuple `_vv10nlc` consumes).
/// The bare `'VV10'` default is `(5.9, 0.0093)` (A1). Per-functional values
/// (e.g. wB97X-V, B97M-V) come from `libxc_rs::NlcCoefficients::nlc_coeff()`
/// under `--features libxc`; the default build uses only the bare default.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NlcCoeffs {
    /// `Bvv` — the b parameter (default 5.9).
    pub bvv: f64,
    /// `Cvv` — the C parameter (default 0.0093).
    pub cvv: f64,
}

impl NlcCoeffs {
    /// The bare `'VV10'` default `(Bvv = 5.9, Cvv = 0.0093)` (A1).
    pub const VV10_DEFAULT: NlcCoeffs = NlcCoeffs { bvv: 5.9, cvv: 0.0093 };

    /// Resolve the NLC coefficients for an `nlc` code. For the bare `'VV10'`
    /// (case-insensitive) returns the A1 default. Per-functional resolution
    /// via `libxc_rs::NlcCoefficients` is `--features libxc`-gated (the
    /// authoritative source); the default build supports only `'VV10'`.
    pub fn for_nlc_code(nlc: &str) -> Result<NlcCoeffs, DftError> {
        let key = nlc.trim().to_ascii_uppercase();
        match key.as_str() {
            "VV10" | "" => Ok(NlcCoeffs::VV10_DEFAULT),
            // Per-functional coeffs are the libxc nlc_coeff() table (A1).
            // Without the libxc backend we only know the bare default — surface
            // a clear error rather than silently substituting the default.
            other => Err(DftError::UnknownFunctional {
                xc: format!(
                    "per-functional VV10 coefficients require the libxc backend \
                     (libxc_rs::NlcCoefficients::nlc_coeff); only the bare 'VV10' default \
                     (Bvv=5.9, Cvv=0.0093) is available on the xcfun-default build (A1)"
                ),
                token: other.to_string(),
            }),
        }
    }
}

/// Output of [`vv10nlc`]: per-grid-point NLC energy density (`exc`, later
/// multiplied by ρ) and the GGA-shaped potential `(vrho, vsigma)`.
///
/// Mirrors upstream `_vv10nlc -> (exc, vxc[2,:])` where `vxc[0]` is `vrho`
/// and `vxc[1]` is `vsigma` (numint.py:552-555).
#[derive(Debug, Clone)]
pub struct Vv10Output {
    /// `exc[i] = Beta + 0.5·F[i]` (energy density; multiplied by ρ later).
    pub exc: Vec<f64>,
    /// `vrho[i] = Beta + F[i] + 1.5·(U[i]·dKdR[i] + W[i]·dW0dR[i])`.
    pub vrho: Vec<f64>,
    /// `vsigma[i] = 1.5·W[i]·dW0dG[i]`.
    pub vsigma: Vec<f64>,
}

/// Density threshold below which an outer/inner grid point is skipped
/// (upstream `thresh = 1e-8`, numint.py:472). Skipped outer points get
/// `exc = vrho = vsigma = 0`; skipped inner points drop out of the sums.
const VV10_THRESH: f64 = 1e-8;

/// `_vv10nlc` — the VV10 non-local-correlation kernel (numint.py:471-555).
///
/// Ports the **commented pure-Python double-loop** (numint.py:526-538), NOT
/// the C `VXC_vv10nlc` (Pitfall 4). For each outer grid point `i` (above the
/// density threshold), it double-loops over the inner (vv) grid accumulating
/// `F = −1.5·Σ T`, `U = Σ T·(1/g + 1/gt)`, `W = Σ T·(1/g + 1/gt)·R2`, then
/// assembles `exc`/`vrho`/`vsigma` per numint.py:552-554.
///
/// Inputs (all on the SAME nlcgrids — outer == inner, numint.py:1394):
///   - `rho` = `[ngrids]` density at the outer points;
///   - `grad` = `(Gx, Gy, Gz)` density gradient components at the outer points;
///   - `coords` = `[ngrids]` outer grid coordinates;
///   - `vvrho`, `vvgrad`, `vvweight`, `vvcoords` = the inner (vv) grid
///     density / gradient / weights / coordinates (== the outer set here).
///
/// All inner reductions use `oracle_sum` (T-04-07b, bit-exact).
#[allow(clippy::too_many_arguments)]
pub fn vv10nlc(
    rho: &[f64],
    grad: (&[f64], &[f64], &[f64]),
    coords: &[[f64; 3]],
    vvrho: &[f64],
    vvgrad: (&[f64], &[f64], &[f64]),
    vvweight: &[f64],
    vvcoords: &[[f64; 3]],
    pars: NlcCoeffs,
) -> Vv10Output {
    let ngrids = rho.len();
    let (gx, gy, gz) = grad;
    let (gxp, gyp, gzp) = vvgrad;

    // Constants and parameters (numint.py:498-503).
    let pi = std::f64::consts::PI;
    let pi43 = 4.0 * pi / 3.0;
    let bvv = pars.bvv;
    let cvv = pars.cvv;
    let kvv = bvv * 1.5 * pi * (9.0 * pi).powf(-1.0 / 6.0);
    let beta = (3.0 / (bvv * bvv)).powf(0.75) / 32.0;

    // ── Pre-compute inner (vv) grid quantities (numint.py:505-509) ─────────
    // Only inner points above threshold contribute (numint.py:487-496).
    let mut inner_idx: Vec<usize> = Vec::with_capacity(ngrids);
    for (j, &rp) in vvrho.iter().enumerate() {
        if rp >= VV10_THRESH {
            inner_idx.push(j);
        }
    }
    let ninner = inner_idx.len();
    // Per-inner-point: RpW, W0p, Kp, coords.
    let mut rpw = vec![0.0_f64; ninner];
    let mut w0p = vec![0.0_f64; ninner];
    let mut kp = vec![0.0_f64; ninner];
    let mut vvc = vec![[0.0_f64; 3]; ninner];
    for (n, &j) in inner_idx.iter().enumerate() {
        let rp = vvrho[j];
        let gp = gxp[j] * gxp[j] + gyp[j] * gyp[j] + gzp[j] * gzp[j];
        // W0p = sqrt(Cvv·(Gp/Rp²)² + Pi43·Rp) (numint.py:506-508).
        let mut w0p_v = gp / (rp * rp);
        w0p_v = cvv * w0p_v * w0p_v;
        w0p[n] = (w0p_v + pi43 * rp).sqrt();
        kp[n] = kvv * rp.powf(1.0 / 6.0); // numint.py:509
        rpw[n] = rp * vvweight[j]; // numint.py:492
        vvc[n] = vvcoords[j];
    }

    // ── Outer grid: per-point double-loop (numint.py:511-538) ──────────────
    let mut exc = vec![0.0_f64; ngrids];
    let mut vrho = vec![0.0_f64; ngrids];
    let mut vsigma = vec![0.0_f64; ngrids];

    // Scratch buffers for the inner reductions (reused across outer points).
    let mut t_buf = vec![0.0_f64; ninner];
    let mut tu_buf = vec![0.0_f64; ninner];
    let mut tw_buf = vec![0.0_f64; ninner];

    for i in 0..ngrids {
        let r = rho[i];
        if r < VV10_THRESH {
            // Below-threshold outer point: exc/vxc stay 0 (numint.py:478-479).
            continue;
        }
        let g = gx[i] * gx[i] + gy[i] * gy[i] + gz[i] * gz[i];

        // Outer-grid W0/K and their derivatives (numint.py:511-518).
        let mut w0tmp = g / (r * r);
        w0tmp = cvv * w0tmp * w0tmp;
        let w0_i = (w0tmp + pi43 * r).sqrt();
        let dw0dr = (0.5 * pi43 * r - 2.0 * w0tmp) / w0_i;
        let dw0dg = w0tmp * r / (g * w0_i);
        let k_i = kvv * r.powf(1.0 / 6.0);
        let dkdr = (1.0 / 6.0) * k_i;

        let ci = coords[i];
        // Inner double-loop (numint.py:526-537, the pure-Python reference).
        for n in 0..ninner {
            let dx = vvc[n][0] - ci[0];
            let dy = vvc[n][1] - ci[1];
            let dz = vvc[n][2] - ci[2];
            let r2 = dx * dx + dy * dy + dz * dz;
            let gp = r2 * w0p[n] + kp[n]; // numint.py:530
            let g_in = r2 * w0_i + k_i; // numint.py:531
            let gt = g_in + gp; // numint.py:532
            let t = rpw[n] / (g_in * gp * gt); // numint.py:533
            let inv = 1.0 / g_in + 1.0 / gt; // numint.py:535 factor
            t_buf[n] = t;
            tu_buf[n] = t * inv;
            tw_buf[n] = t * inv * r2;
        }
        // F = −1.5·Σ T; U = Σ T·(1/g+1/gt); W = Σ T·(1/g+1/gt)·R2.
        // oracle_sum for bit-exact inner reductions (T-04-07b).
        let f = -1.5 * oracle_sum(&t_buf);
        let u = oracle_sum(&tu_buf);
        let w = oracle_sum(&tw_buf);

        // Assemble exc/vrho/vsigma (numint.py:552-554).
        exc[i] = beta + 0.5 * f;
        vrho[i] = beta + f + 1.5 * (u * dkdr + w * dw0dr);
        vsigma[i] = 1.5 * w * dw0dg;
    }

    Vv10Output { exc, vrho, vsigma }
}

/// `nr_nlc_vxc` — VV10 NLC energy + potential matrix over a coarser
/// `nlcgrids` (numint.py:1347-1416).
///
/// Builds ρ + ∇ρ on the (already-built) `nlcgrids`, calls [`vv10nlc`] with
/// outer == inner == the nlcgrids (numint.py:1394), then:
///   - `excsum = Σ_i (ρ[i]·w[i])·exc[i]`  (numint.py:1398-1400, via oracle_dot);
///   - back-contracts the GGA Vxc into an `nao × nao` matrix (numint.py:1407-1415):
///     `Vmat[μν] += Σ_g w[g]·(0.5·vrho[g]·φ_μ·φ_ν + vsigma[g]·∇ρ·∇φ_μ·φ_ν)`,
///     symmetrized `Vmat += Vmatᵀ`.
///
/// Returns `(nelec, excsum, vmat)` mirroring upstream `nr_nlc_vxc`.
/// `nlc` is the NLC code (`"VV10"` for the bare default).
pub fn nr_nlc_vxc(
    ni: &NumInt,
    mol: &Mole,
    nlcgrids: &Grids,
    nlc: &str,
    dm: &Density,
) -> Result<NlcResult, PyscfRsError> {
    let _ = ni; // signature parity with upstream (the NumInt is the eval host).
    let pars = NlcCoeffs::for_nlc_code(nlc).map_err(PyscfRsError::from)?;

    let coords = nlcgrids.coords.as_ref().ok_or_else(|| {
        dft_err("nr_nlc_vxc: nlcgrids not built (coords missing) — call nlcgrids.build(&mol)")
    })?;
    let weights = nlcgrids.weights.as_ref().ok_or_else(|| {
        dft_err("nr_nlc_vxc: nlcgrids not built (weights missing) — call nlcgrids.build(&mol)")
    })?;
    let ngrids = coords.len();
    let nao = mol.nao_nr;

    // VV10 always needs ρ+∇ρ (GGA-shaped); ao_deriv=1 (numint.py:1383).
    let ao = pyscf_gto::eval_gto(mol, "GTOval_sph_deriv1", coords)?.values;
    let (rho, grad) = NumInt::eval_rho(&ao, dm, ngrids, XcType::Gga)?;
    let (dx, dy, dz) = grad
        .as_ref()
        .ok_or_else(|| dft_err("nr_nlc_vxc: GGA ∇ρ missing"))?;

    // Outer == inner == the nlcgrids (numint.py:1394). VV10 is self-coupled.
    let vv = vv10nlc(
        &rho,
        (dx, dy, dz),
        coords,
        &rho,
        (dx, dy, dz),
        weights,
        coords,
        pars,
    );

    // den = ρ·w; nelec = Σ den; excsum = Σ den·exc (numint.py:1398-1400).
    let den: Vec<f64> = (0..ngrids).map(|g| rho[g] * weights[g]).collect();
    let nelec = oracle_sum(&den);
    let excsum = oracle_dot(&den, &vv.exc);

    // GGA Vxc back-contraction (numint.py:1407-1415). The AO buffer is
    // F-order [4, ngrids, nao]: ao[comp*ngrids*nao + (g + mu*ngrids)].
    let ao_at = |comp: usize, g: usize, mu: usize| -> f64 {
        ao[comp * ngrids * nao + (g + mu * ngrids)]
    };
    let mut vmat = vec![0.0_f64; nao * nao];
    let mut row_terms = vec![0.0_f64; ngrids];
    for mu in 0..nao {
        for nu in 0..nao {
            for g in 0..ngrids {
                let w = weights[g];
                let phi_mu = ao_at(0, g, mu);
                let phi_nu = ao_at(0, g, nu);
                // LDA-shaped term: 0.5·w·vrho·φ_μ·φ_ν (numint.py:1411 wv[0]*=.5).
                let mut t = 0.5 * w * vv.vrho[g] * phi_mu * phi_nu;
                // GGA gradient term: 2·w·vsigma·(∇ρ·∇φ_μ)·φ_ν, symmetrized
                // (the same back-contraction shape as the main numint grid
                // loop; the 0.5 folds with the vmat += V + Vᵀ symmetrization).
                let gphi_mu = dx[g] * ao_at(1, g, mu)
                    + dy[g] * ao_at(2, g, mu)
                    + dz[g] * ao_at(3, g, mu);
                t += 0.5 * 2.0 * w * vv.vsigma[g] * gphi_mu * phi_nu;
                row_terms[g] = t;
            }
            let contrib = oracle_sum(&row_terms);
            // Symmetrize: vmat += V + Vᵀ (numint.py:1415).
            vmat[mu * nao + nu] += contrib;
            vmat[nu * nao + mu] += contrib;
        }
    }

    Ok(NlcResult {
        nelec,
        excsum,
        vmat: Density { nao, data: vmat },
    })
}

/// Result of [`nr_nlc_vxc`]: the NLC electron count, NLC energy, and NLC
/// potential matrix. Mirrors upstream `nr_nlc_vxc -> (nelec, excsum, vmat)`.
#[derive(Debug, Clone)]
pub struct NlcResult {
    /// `∫ ρ` over the nlcgrids (diagnostic; the NLC nelec).
    pub nelec: f64,
    /// `Exc_nlc = Σ den·exc` — the NLC energy contribution (adds to Exc).
    pub excsum: f64,
    /// `Vxc_nlc` — the NLC potential matrix (adds to Vxc).
    pub vmat: Density,
}

fn dft_err(msg: &str) -> PyscfRsError {
    PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(msg.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bare 'VV10' default coefficients are Bvv=5.9, Cvv=0.0093 (A1).
    #[test]
    fn vv10_default_coeffs() {
        let c = NlcCoeffs::for_nlc_code("VV10").unwrap();
        assert_eq!(c.bvv, 5.9);
        assert_eq!(c.cvv, 0.0093);
        // Case-insensitive.
        assert_eq!(NlcCoeffs::for_nlc_code("vv10").unwrap(), c);
    }

    /// A per-functional nlc code (without the libxc backend) is rejected with
    /// a clear message rather than silently using the bare default (A1).
    #[test]
    fn per_functional_nlc_requires_libxc() {
        let res = NlcCoeffs::for_nlc_code("WB97X_V");
        assert!(res.is_err(), "per-functional nlc needs libxc nlc_coeff (A1)");
    }

    /// Independent longhand verification of the VV10 inner double-loop on a
    /// tiny 2-point fixture: the F/U/W accumulators + the exc/vrho/vsigma
    /// assembly recomputed by hand match `vv10nlc`. This is the byte-exact
    /// reference (the pure-Python double-loop, NOT VXC_vv10nlc — Pitfall 4).
    #[test]
    fn vv10nlc_matches_independent_longhand() {
        // 2 grid points (outer == inner). Both above threshold.
        let coords = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let rho = vec![0.5_f64, 0.3];
        let gx = vec![0.1_f64, 0.2];
        let gy = vec![0.0_f64, 0.0];
        let gz = vec![0.0_f64, 0.0];
        let w = vec![0.7_f64, 0.9];
        let pars = NlcCoeffs::VV10_DEFAULT;

        let out = vv10nlc(
            &rho,
            (&gx, &gy, &gz),
            &coords,
            &rho,
            (&gx, &gy, &gz),
            &w,
            &coords,
            pars,
        );

        // Independent longhand recomputation (mirrors numint.py:498-554).
        let pi = std::f64::consts::PI;
        let pi43 = 4.0 * pi / 3.0;
        let (bvv, cvv) = (pars.bvv, pars.cvv);
        let kvv = bvv * 1.5 * pi * (9.0 * pi).powf(-1.0 / 6.0);
        let beta = (3.0 / (bvv * bvv)).powf(0.75) / 32.0;

        let n = rho.len();
        // Inner pre-compute.
        let mut w0p = vec![0.0; n];
        let mut kp = vec![0.0; n];
        let mut rpw = vec![0.0; n];
        for j in 0..n {
            let gp = gx[j] * gx[j] + gy[j] * gy[j] + gz[j] * gz[j];
            let mut v = gp / (rho[j] * rho[j]);
            v = cvv * v * v;
            w0p[j] = (v + pi43 * rho[j]).sqrt();
            kp[j] = kvv * rho[j].powf(1.0 / 6.0);
            rpw[j] = rho[j] * w[j];
        }
        for i in 0..n {
            let r = rho[i];
            let g = gx[i] * gx[i] + gy[i] * gy[i] + gz[i] * gz[i];
            let mut w0tmp = g / (r * r);
            w0tmp = cvv * w0tmp * w0tmp;
            let w0_i = (w0tmp + pi43 * r).sqrt();
            let dw0dr = (0.5 * pi43 * r - 2.0 * w0tmp) / w0_i;
            let dw0dg = w0tmp * r / (g * w0_i);
            let k_i = kvv * r.powf(1.0 / 6.0);
            let dkdr = (1.0 / 6.0) * k_i;
            let (mut f, mut u, mut ww) = (0.0_f64, 0.0_f64, 0.0_f64);
            for j in 0..n {
                let dx = coords[j][0] - coords[i][0];
                let dy = coords[j][1] - coords[i][1];
                let dz = coords[j][2] - coords[i][2];
                let r2 = dx * dx + dy * dy + dz * dz;
                let gp = r2 * w0p[j] + kp[j];
                let g_in = r2 * w0_i + k_i;
                let gt = g_in + gp;
                let t = rpw[j] / (g_in * gp * gt);
                let inv = 1.0 / g_in + 1.0 / gt;
                f += t;
                u += t * inv;
                ww += t * inv * r2;
            }
            f *= -1.5;
            let exc_i = beta + 0.5 * f;
            let vrho_i = beta + f + 1.5 * (u * dkdr + ww * dw0dr);
            let vsigma_i = 1.5 * ww * dw0dg;
            assert!(
                (out.exc[i] - exc_i).abs() < 1e-12,
                "exc[{i}]: got {}, want {}",
                out.exc[i],
                exc_i
            );
            assert!(
                (out.vrho[i] - vrho_i).abs() < 1e-12,
                "vrho[{i}]: got {}, want {}",
                out.vrho[i],
                vrho_i
            );
            assert!(
                (out.vsigma[i] - vsigma_i).abs() < 1e-12,
                "vsigma[{i}]: got {}, want {}",
                out.vsigma[i],
                vsigma_i
            );
        }
    }

    /// Below-threshold outer points produce zero exc/vrho/vsigma (numint.py:478).
    #[test]
    fn below_threshold_points_are_zero() {
        let coords = vec![[0.0, 0.0, 0.0]];
        let rho = vec![1e-12_f64]; // below VV10_THRESH = 1e-8
        let z = vec![0.0_f64];
        let w = vec![1.0_f64];
        let out = vv10nlc(
            &rho,
            (&z, &z, &z),
            &coords,
            &rho,
            (&z, &z, &z),
            &w,
            &coords,
            NlcCoeffs::VV10_DEFAULT,
        );
        assert_eq!(out.exc[0], 0.0);
        assert_eq!(out.vrho[0], 0.0);
        assert_eq!(out.vsigma[0], 0.0);
    }
}
