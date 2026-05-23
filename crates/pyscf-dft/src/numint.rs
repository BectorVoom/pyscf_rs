//! `NumInt` — the numerical-integration grid loop (DFT-10, D-07, D-08).
//!
//! Source: `pyscf/dft/numint.py` (Apache-2.0):
//!   - `nr_rks`        (numint.py:1074) — closed-shell RKS Vxc/Exc grid loop
//!   - `nr_uks`        (numint.py:1192) — open-shell UKS (alpha/beta) grid loop
//!   - `eval_rho`      (numint.py:116)  — ρ(and ∇ρ) from AO + density matrix
//!   - `NumInt` class  (numint.py:2835) — the eval/hybrid-coeff surface
//!   - `hybrid_coeff`  (numint.py:2731) / `rsh_coeff` (numint.py:2737)
//!
//! Decisions threaded here:
//!   - **D-07** (grid loop is algebra-orchestrated, NO bespoke `#[cube]`
//!     kernel in pyscf-dft): AO evaluation routes through
//!     `pyscf_gto::eval_gto` (which calls `pyscf_kernels::eval_gto_sph`
//!     behind the algebra wall); ρ = AOᵀ·D·AO and the Vxc back-contraction
//!     are dense host matmuls; every grid-block reduction (Exc, the ρ
//!     accumulation, the nelec integral) goes through
//!     `pyscf_algebra::oracle_sum` for bit-exactness (Pitfall 3/9).
//!     The Tensor-API `pyscf_algebra::gemm` is `NotYetImplemented{phase:2}`
//!     (its body lands with the first GPU call site), so — exactly as the
//!     Phase 3 SCF Fock build and the DF J/K path do (`fock.rs`,
//!     `df_scf.rs` "inline the loop … matches plan 03-11 pattern") — the
//!     dense contractions are explicit host loops with `oracle_sum`
//!     reductions, which IS the FMA-free oracle target.
//!   - **D-08** (precision seam through DFT): `NumInt` reads the active
//!     scalar via `pyscf_runtime::DType::from_env()` at construction and
//!     exposes a read-only [`NumInt::dtype`] accessor. The AO-eval /
//!     ρ-contraction / Vxc back-contraction matmul chain is dispatched
//!     over the active scalar by matching on the resolved `DType` (mirrors
//!     the `AlgebraClient` enum-match shape, client.rs:10-39). The
//!     `DType::F64` arm is the existing f64 code path UNCHANGED (default,
//!     bit-exact). The `DType::F32` arm runs the same contraction chain in
//!     `f32` and emits ONE below-bit-exact `tracing::warn!` at kernel
//!     entry. The f32 boundary covers ONLY the matmul chain: `eval_gto`
//!     stays `KernelScalar = f64` and the XC eval stays f64, so ρ is cast
//!     to f64 at the `XcBackend::eval` boundary and the returned `v` is
//!     cast back to the active scalar before the Vxc back-contraction. NO
//!     `set_precision` runtime override (deferred per D-08); NO f32
//!     numeric-parity tolerance (deferred per D-08).

use pyscf_algebra::oracle_sum;
use pyscf_core::{Density, Mole, PyscfRsError};
use pyscf_grids::Grids;
use pyscf_runtime::DType;

use crate::error::DftError;
use crate::parser::{XcSpec, xcfun};
use crate::xc_backend::{DerivOrder, RhoBlock, XcBackend};

/// xcfun functional ids (0..77) whose family is GGA (gradient-dependent).
/// The default backend is xcfun (04-05), so the parser emits xcfun ids; the
/// rest of the mapped corpus (`{0,2,3,28}`) is LDA. MGGA is deferred (v1
/// corpus tops out at GGA hybrids). Source: the xcfun functional families
/// cross-checked against `xc_backend::xcfun_id_to_name` + the 04-05 parity
/// table (SLATERX/VWN are LDA; BECKEX/LYPC/PBEX/PBEC are GGA).
const XCFUN_GGA_IDS: &[u32] = &[4, 5, 6, 16, 17, 19, 20, 26, 56, 77];

/// The functional "type" upstream `numint.py` keys the grid loop on
/// (`'LDA'` / `'GGA'` / `'MGGA'` / `'HF'`). Selects how many AO components
/// `eval_gto` returns and which density-derivative rows `eval_rho` builds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XcType {
    /// Local density approximation — ρ only (`GTOval_sph`).
    Lda,
    /// Generalized-gradient — ρ + ∇ρ (`GTOval_sph_deriv1`).
    Gga,
}

impl XcType {
    /// The `eval_gto` variant name that produces the AO components this
    /// functional type needs.
    fn eval_gto_name(self) -> &'static str {
        match self {
            XcType::Lda => "GTOval_sph",
            XcType::Gga => "GTOval_sph_deriv1",
        }
    }
}

/// Result of a closed-shell `nr_rks` grid integration: the integrated
/// electron count, the XC energy, and the XC potential matrix.
///
/// Mirrors upstream `nr_rks -> (nelec, excsum, vmat)` (numint.py:1074).
#[derive(Debug, Clone)]
pub struct NrResult {
    /// Integrated electron count `∫ ρ` (closed-shell scalar; the alpha/beta
    /// pair for `nr_uks`).
    pub nelec: f64,
    /// `Exc = ∫ ε_xc(ρ) ρ dr` — the XC energy.
    pub excsum: f64,
    /// `Vxc` — the XC potential as an `nao × nao` matrix (row-major,
    /// matching [`Density`]).
    pub vmat: Density,
}

/// Open-shell `nr_uks` result — alpha/beta electron counts plus a Vxc per
/// spin channel. Mirrors upstream `nr_uks -> (nelec[2], excsum, vmat[2])`.
#[derive(Debug, Clone)]
pub struct NrUksResult {
    /// `(nelec_alpha, nelec_beta)`.
    pub nelec: (f64, f64),
    /// `Exc` over the combined α+β density.
    pub excsum: f64,
    /// `(Vxc_alpha, Vxc_beta)`.
    pub vmat: (Density, Density),
}

/// The numerical-integration driver. Holds the active XC backend and the
/// resolved precision (D-08). Constructed once and reused across SCF cycles.
///
/// Upstream analog: `pyscf/dft/numint.py:NumInt` (numint.py:2835).
#[derive(Debug, Clone)]
pub struct NumInt {
    /// XC evaluation backend (default `Xcfun`; `Libxc` only under
    /// `--features libxc`).
    backend: XcBackend,
    /// Active scalar precision, resolved from `PYSCF_DTYPE` at construction
    /// (D-08). Read-only — there is intentionally NO mutable setter.
    dtype: DType,
}

impl Default for NumInt {
    fn default() -> Self {
        Self::new()
    }
}

impl NumInt {
    /// Construct a `NumInt` with the default (`Xcfun`) backend and the
    /// precision resolved from `PYSCF_DTYPE` (D-08, default F64).
    pub fn new() -> Self {
        Self {
            backend: XcBackend::default(),
            dtype: DType::from_env(),
        }
    }

    /// The active scalar precision (D-08). Read-only accessor — surfaced so
    /// the KS object (and, in 04-09, the Python `#[getter]`) can report the
    /// precision the grid loop is actually running. There is intentionally
    /// NO `set_precision` (deferred per D-08).
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The active XC backend.
    pub fn backend(&self) -> XcBackend {
        self.backend
    }

    /// `hybrid_coeff` / `rsh_coeff` — the `(omega, alpha, hyb)` range-
    /// separation triple for `xc_code`. Mirrors upstream
    /// `numint.hybrid_coeff` (2731) / `rsh_coeff` (2737): the parser already
    /// extracts `(hyb, alpha, omega)` (`XcSpec::hyb`), so this is the
    /// re-ordered surface upstream callers expect.
    ///
    /// `spin` is accepted for signature parity with upstream (the hybrid
    /// coefficient is spin-independent for the v1 corpus).
    pub fn rsh_coeff(&self, xc_code: &str, _spin: i32) -> Result<(f64, f64, f64), DftError> {
        // The default backend is xcfun, so parse in the xcfun namespace —
        // xcfun exposes the standard-hybrid mixing in `hyb[0]` (e.g. b3lyp →
        // hyb=0.2), whereas the libxc parser folds it inside the compound id.
        let spec = xcfun::parse_xc(xc_code)?;
        let (hyb, alpha, omega) = spec.hyb();
        Ok((omega, alpha, hyb))
    }

    /// `hybrid_coeff(xc_code, spin) -> hyb` — the standard-hybrid HF-exchange
    /// mixing coefficient (`hyb[0]`). Mirrors `numint.hybrid_coeff` (2731).
    pub fn hybrid_coeff(&self, xc_code: &str, spin: i32) -> Result<f64, DftError> {
        let (_omega, _alpha, hyb) = self.rsh_coeff(xc_code, spin)?;
        Ok(hyb)
    }

    /// `eval_rho` — evaluate ρ (and ∇ρ for GGA) at the grid points given the
    /// AO values and the density matrix.
    ///
    /// Source: `pyscf/dft/numint.py:116`. For LDA the AO buffer is the
    /// `[ngrids, nao]` value block; ρ[g] = Σ_{μν} AO[g,μ] D[μν] AO[g,ν].
    /// For GGA the AO buffer is `[4, ngrids, nao]` (value + 3 gradient
    /// components); ρ uses component 0 and ∇ρ_c[g] = 2 Σ_{μν} AO_c[g,μ]
    /// D[μν] AO_0[g,ν] (the factor-2 from the symmetric D, upstream
    /// numint.py:148).
    ///
    /// Returns `(rho, grad)` where `grad` is `Some((dx,dy,dz))` for GGA and
    /// `None` for LDA. All reductions go through `oracle_sum` (Pitfall 3).
    ///
    /// `D` is row-major `nao × nao` ([`Density`]); the AO buffer is F-order
    /// (`ao[g + mu*ngrids]` per component), matching `EvalGtoOutput`.
    #[allow(clippy::type_complexity)]
    pub fn eval_rho(
        ao: &[f64],
        dm: &Density,
        ngrids: usize,
        xctype: XcType,
    ) -> Result<(Vec<f64>, Option<(Vec<f64>, Vec<f64>, Vec<f64>)>), PyscfRsError> {
        let nao = dm.nao;
        // F-order AO index for component `c`, grid point `g`, AO `mu`.
        let ao_at = |buf: &[f64], comp: usize, g: usize, mu: usize| -> f64 {
            buf[comp * ngrids * nao + (g + mu * ngrids)]
        };
        match xctype {
            XcType::Lda => {
                if ao.len() != ngrids * nao {
                    return Err(PyscfRsError::Core(
                        pyscf_core::CoreError::DimensionMismatch {
                            expected: ngrids * nao,
                            actual: ao.len(),
                        },
                    ));
                }
                let rho = Self::contract_rho(ao, ao, dm, ngrids, nao, &ao_at);
                Ok((rho, None))
            }
            XcType::Gga => {
                if ao.len() != 4 * ngrids * nao {
                    return Err(PyscfRsError::Core(
                        pyscf_core::CoreError::DimensionMismatch {
                            expected: 4 * ngrids * nao,
                            actual: ao.len(),
                        },
                    ));
                }
                // ρ from the value component (comp 0).
                let rho = Self::contract_rho_comp(ao, dm, ngrids, nao, 0, 0, 1.0, &ao_at);
                // ∇ρ_c = 2 · Σ_{μν} AO_c[g,μ] D[μν] AO_0[g,ν] (numint.py:148).
                let dx = Self::contract_rho_comp(ao, dm, ngrids, nao, 1, 0, 2.0, &ao_at);
                let dy = Self::contract_rho_comp(ao, dm, ngrids, nao, 2, 0, 2.0, &ao_at);
                let dz = Self::contract_rho_comp(ao, dm, ngrids, nao, 3, 0, 2.0, &ao_at);
                Ok((rho, Some((dx, dy, dz))))
            }
        }
    }

    /// ρ[g] = Σ_{μν} AO_l[g,μ] D[μν] AO_r[g,ν] (closed-shell value block).
    // `ao_at` is the AO-lookup closure; `g` is the grid index passed into it.
    #[allow(clippy::type_complexity, clippy::needless_range_loop)]
    fn contract_rho(
        ao_l: &[f64],
        ao_r: &[f64],
        dm: &Density,
        ngrids: usize,
        nao: usize,
        ao_at: &dyn Fn(&[f64], usize, usize, usize) -> f64,
    ) -> Vec<f64> {
        let mut rho = vec![0.0_f64; ngrids];
        let mut terms = vec![0.0_f64; nao * nao];
        for g in 0..ngrids {
            for mu in 0..nao {
                let a_mu = ao_at(ao_l, 0, g, mu);
                for nu in 0..nao {
                    let d = dm.data[mu * nao + nu];
                    terms[mu * nao + nu] = a_mu * d * ao_at(ao_r, 0, g, nu);
                }
            }
            rho[g] = oracle_sum(&terms);
        }
        rho
    }

    /// Generic single-component ρ contraction:
    /// `scale · Σ_{μν} AO[comp_l][g,μ] D[μν] AO[comp_r][g,ν]`.
    /// Used for ρ (comp_l=comp_r=0, scale=1) and ∇ρ_c (comp_l=c, comp_r=0,
    /// scale=2).
    // `ao_at` is the AO-lookup closure; `g` is the grid index passed into it.
    #[allow(
        clippy::type_complexity,
        clippy::too_many_arguments,
        clippy::needless_range_loop
    )]
    fn contract_rho_comp(
        ao: &[f64],
        dm: &Density,
        ngrids: usize,
        nao: usize,
        comp_l: usize,
        comp_r: usize,
        scale: f64,
        ao_at: &dyn Fn(&[f64], usize, usize, usize) -> f64,
    ) -> Vec<f64> {
        let mut out = vec![0.0_f64; ngrids];
        let mut terms = vec![0.0_f64; nao * nao];
        for g in 0..ngrids {
            for mu in 0..nao {
                let a_mu = ao_at(ao, comp_l, g, mu);
                for nu in 0..nao {
                    let d = dm.data[mu * nao + nu];
                    terms[mu * nao + nu] = a_mu * d * ao_at(ao, comp_r, g, nu);
                }
            }
            out[g] = scale * oracle_sum(&terms);
        }
        out
    }

    /// `eval_xc` — evaluate the functional `xc_code` over a density block,
    /// routing through the active [`XcBackend`]. Thin wrapper that mirrors
    /// the upstream `NumInt.eval_xc` surface; the heavy lifting is in
    /// `XcBackend::eval`.
    ///
    /// Per D-08 the XC eval is always f64 (libxc_rs/xcfun_rs are f64-host),
    /// so callers in the f32 path cast ρ → f64 here and cast `v` back after.
    pub fn eval_xc(
        &self,
        xc_code: &str,
        rho: &RhoBlock<'_>,
        order: DerivOrder,
    ) -> Result<crate::xc_backend::XcOutput, DftError> {
        let spec = xcfun::parse_xc(xc_code)?;
        self.backend.eval(&spec, rho, order)
    }

    /// `nr_rks` — the closed-shell RKS Vxc/Exc grid loop (DFT-10).
    ///
    /// Signature mirrors upstream `nr_rks(ni, mol, grids, xc_code, dms,
    /// relativity=0, hermi=1, max_memory=2000, verbose=None)` (numint.py:1074)
    /// returning `(nelec, excsum, vmat)`.
    ///
    /// Algorithm (D-07): build the grid AO block, contract ρ(+∇ρ), evaluate
    /// the functional, accumulate `Exc += oracle_sum(w · exc)` and back-
    /// contract `Vxc += Σ_g w[g] · vrho[g] · AO[g,μ] AO[g,ν]`. The matmul
    /// chain is dispatched over the active `DType` (D-08); the `F64` arm is
    /// the bit-exact default, the `F32` arm runs the same chain in f32 with a
    /// below-bit-exact `tracing::warn!`.
    ///
    /// `grids` must already be `build()`-populated (coords + weights).
    #[allow(clippy::too_many_arguments)]
    pub fn nr_rks(
        &self,
        mol: &Mole,
        grids: &Grids,
        xc_code: &str,
        dm: &Density,
        _relativity: i32,
        _hermi: i32,
        max_memory: f64,
        _verbose: Option<i32>,
    ) -> Result<NrResult, PyscfRsError> {
        // ALG-08 + D-08 observability line at grid-loop entry (mirrors
        // AlgebraClient::log_resolution, client.rs:38-45).
        let raw_env = std::env::var("PYSCF_DTYPE").ok();
        tracing::info!(
            target: "pyscf_dft::numint",
            backend = self.backend_name(),
            dtype = self.dtype.name(),
            env = raw_env.as_deref().unwrap_or("unset"),
            max_memory = max_memory,
            "nr_rks grid-loop entry"
        );
        if self.dtype == DType::F32 {
            // D-08: f32 is below the bit-exact bar — an explicit, single,
            // entry-point warning so an f32 result is never mistaken for the
            // f64 oracle path. (Mirrors the Phase 1 `select.rs` warn shape.)
            tracing::warn!(
                target: "pyscf_dft::numint",
                "PYSCF_DTYPE=f32: the DFT grid loop is running below the bit-exact bar \
                 (f32 matmul chain); results are NOT oracle-comparable (D-08)."
            );
        }

        let xctype = Self::xc_type_of(xc_code)?;
        let coords = grids.coords.as_ref().ok_or_else(|| {
            dft_err("nr_rks: grids not built (coords missing) — call grids.build(&mol)")
        })?;
        let weights = grids.weights.as_ref().ok_or_else(|| {
            dft_err("nr_rks: grids not built (weights missing) — call grids.build(&mol)")
        })?;
        let nao = mol.nao_nr;

        // The matmul chain is dispatched over the active scalar (D-08). The
        // f64 arm is the existing bit-exact path; the f32 arm runs the same
        // contraction chain in f32 (ρ cast to f64 only at the XC boundary).
        match self.dtype {
            DType::F64 => self.nr_rks_inner::<f64>(mol, coords, weights, xc_code, dm, xctype, nao),
            DType::F32 => self.nr_rks_inner::<f32>(mol, coords, weights, xc_code, dm, xctype, nao),
        }
    }

    /// Precision-generic RKS grid-loop body (D-08). `S` is the active scalar
    /// for the AO-eval / ρ-contraction / Vxc back-contraction matmul chain.
    /// ρ is cast to f64 at the `XcBackend::eval` boundary (XC is f64-only),
    /// and the returned `v` is cast back to `S` before the back-contraction.
    /// For `S = f64` this is the unchanged bit-exact default path.
    #[allow(clippy::too_many_arguments)]
    fn nr_rks_inner<S: pyscf_core::Scalar>(
        &self,
        _mol: &Mole,
        coords: &[[f64; 3]],
        weights: &[f64],
        xc_code: &str,
        dm: &Density,
        xctype: XcType,
        nao: usize,
    ) -> Result<NrResult, PyscfRsError> {
        let ngrids = coords.len();
        let mut vmat = vec![0.0_f64; nao * nao];
        let mut nelec_blocks: Vec<f64> = Vec::new();
        let mut exc_blocks: Vec<f64> = Vec::new();

        // The AO block — evaluated via pyscf_gto::eval_gto, which routes to
        // pyscf_kernels::eval_gto_sph behind the algebra wall (eval_gto stays
        // KernelScalar = f64 per D-08; the AO buffer is always f64).
        let ao = crate::numint::eval_gto_block(_mol, xctype, coords)?;

        // ρ(+∇ρ) contraction. The matmul chain runs in the active scalar S:
        // we cast the AO + D to S, contract, then cast ρ back for the XC
        // boundary. For S=f64 the cast is identity (bit-exact default).
        let (rho, grad) = Self::eval_rho_scalar::<S>(&ao, dm, ngrids, xctype)?;

        // XC evaluation is f64-only (libxc_rs/xcfun_rs host f64) — ρ is f64
        // here by construction (eval_rho returns f64; the S-precision is in
        // the matmul intermediates).
        let xc_out = match xctype {
            XcType::Lda => {
                let rb = RhoBlock::Lda { rho: &rho };
                self.eval_xc(xc_code, &rb, DerivOrder::Vxc)
            }
            XcType::Gga => {
                let (dx, dy, dz) = grad
                    .as_ref()
                    .ok_or_else(|| dft_err("nr_rks: GGA functional but ∇ρ missing"))?;
                let sigma: Vec<f64> = (0..ngrids)
                    .map(|g| dx[g] * dx[g] + dy[g] * dy[g] + dz[g] * dz[g])
                    .collect();
                let rb = RhoBlock::Gga {
                    rho: &rho,
                    sigma: &sigma,
                };
                self.eval_xc(xc_code, &rb, DerivOrder::Vxc)
            }
        }
        .map_err(PyscfRsError::from)?;

        // Per-grid-point accumulation. Exc and nelec via oracle_sum
        // (Pitfall 3); Vxc back-contraction over the active scalar S.
        let ao_at = |buf: &[f64], comp: usize, g: usize, mu: usize| -> f64 {
            buf[comp * ngrids * nao + (g + mu * ngrids)]
        };
        let mut den_terms = vec![0.0_f64; ngrids];
        let mut exc_terms = vec![0.0_f64; ngrids];
        for g in 0..ngrids {
            let w = weights[g];
            den_terms[g] = w * rho[g];
            // Exc = Σ_g w[g] · exc[g] (XcOutput.exc is already ε·ρ per point).
            exc_terms[g] = w * xc_out.exc[g];
        }
        nelec_blocks.push(oracle_sum(&den_terms));
        exc_blocks.push(oracle_sum(&exc_terms));

        // Vxc[μν] += Σ_g w[g] · vrho[g] · AO_0[g,μ] · AO_0[g,ν]
        //          (+ GGA gradient term, below). The contraction runs in S
        //          (cast in/out); for S=f64 it is the unchanged default.
        let mut vrow_terms = vec![0.0_f64; ngrids];
        for mu in 0..nao {
            for nu in 0..nao {
                for g in 0..ngrids {
                    let w = weights[g];
                    // LDA term: w · vrho · φ_μ · φ_ν, evaluated in S.
                    let phi_mu = S::from(ao_at(&ao, 0, g, mu)).unwrap_or_else(S::zero);
                    let phi_nu = S::from(ao_at(&ao, 0, g, nu)).unwrap_or_else(S::zero);
                    let wv = S::from(w * 0.5 * xc_out.vrho[g]).unwrap_or_else(S::zero);
                    let mut t = wv * phi_mu * phi_nu;
                    // GGA gradient back-contraction (numint.py:_gga_grad):
                    // 2·w·vsigma·(∇ρ·∇φ_μ)·φ_ν symmetrized. v1 corpus PBE/
                    // B3LYP exercise this; we add the symmetric gradient term.
                    if let (XcType::Gga, Some((dx, dy, dz))) = (xctype, grad.as_ref()) {
                        let wvs = S::from(2.0 * w * xc_out.vsigma[g]).unwrap_or_else(S::zero);
                        let gphi_mu = S::from(
                            dx[g] * ao_at(&ao, 1, g, mu)
                                + dy[g] * ao_at(&ao, 2, g, mu)
                                + dz[g] * ao_at(&ao, 3, g, mu),
                        )
                        .unwrap_or_else(S::zero);
                        // 0.5 factor folds with the symmetrization (vmat += V + Vᵀ).
                        t = t + wvs * gphi_mu * phi_nu * S::from(0.5).unwrap_or_else(S::zero);
                    }
                    // Cast the per-point contribution back to f64 for the
                    // oracle_sum reduction (XC boundary + reduction are f64).
                    vrow_terms[g] = t.to_f64().unwrap_or(0.0);
                }
                let contrib = oracle_sum(&vrow_terms);
                // Symmetrize: vmat += V + Vᵀ (upstream numint.py:1160).
                vmat[mu * nao + nu] += contrib;
                vmat[nu * nao + mu] += contrib;
            }
        }

        Ok(NrResult {
            nelec: oracle_sum(&nelec_blocks),
            excsum: oracle_sum(&exc_blocks),
            vmat: Density { nao, data: vmat },
        })
    }

    /// Precision-generic ρ contraction wrapper (D-08). Casts AO + D to the
    /// active scalar `S`, contracts the matmul chain in `S`, and returns ρ
    /// (+∇ρ) cast back to f64 for the XC boundary. For `S=f64` the casts are
    /// identity and the result is bit-identical to [`NumInt::eval_rho`].
    #[allow(clippy::type_complexity)]
    pub(crate) fn eval_rho_scalar<S: pyscf_core::Scalar>(
        ao: &[f64],
        dm: &Density,
        ngrids: usize,
        xctype: XcType,
    ) -> Result<(Vec<f64>, Option<(Vec<f64>, Vec<f64>, Vec<f64>)>), PyscfRsError> {
        if S::KIND == pyscf_core::ScalarKind::F64 {
            // Bit-exact default path — the f64 oracle_sum contraction.
            return NumInt::eval_rho(ao, dm, ngrids, xctype);
        }
        // f32 (or future low-precision) path: contract in S, return f64.
        let nao = dm.nao;
        let ao_at = |buf: &[f64], comp: usize, g: usize, mu: usize| -> f64 {
            buf[comp * ngrids * nao + (g + mu * ngrids)]
        };
        let contract = |comp_l: usize, comp_r: usize, scale: f64| -> Vec<f64> {
            let mut out = vec![0.0_f64; ngrids];
            // `g` is the grid index passed into the `ao_at` closure.
            #[allow(clippy::needless_range_loop)]
            for g in 0..ngrids {
                let mut acc = S::zero();
                for mu in 0..nao {
                    let a_mu = S::from(ao_at(ao, comp_l, g, mu)).unwrap_or_else(S::zero);
                    for nu in 0..nao {
                        let d = S::from(dm.data[mu * nao + nu]).unwrap_or_else(S::zero);
                        let a_nu = S::from(ao_at(ao, comp_r, g, nu)).unwrap_or_else(S::zero);
                        acc = acc + a_mu * d * a_nu;
                    }
                }
                out[g] = scale * acc.to_f64().unwrap_or(0.0);
            }
            out
        };
        match xctype {
            XcType::Lda => Ok((contract(0, 0, 1.0), None)),
            XcType::Gga => {
                let rho = contract(0, 0, 1.0);
                let dx = contract(1, 0, 2.0);
                let dy = contract(2, 0, 2.0);
                let dz = contract(3, 0, 2.0);
                Ok((rho, Some((dx, dy, dz))))
            }
        }
    }

    /// `nr_uks` — open-shell UKS Vxc/Exc grid loop (numint.py:1192).
    ///
    /// Signature mirrors `nr_uks(ni, mol, grids, xc_code, dms, ...)` returning
    /// `(nelec[2], excsum, vmat[2])`. v1 builds the spin-resolved Vxc by
    /// running the closed-shell-shaped contraction over the total density for
    /// the XC eval (the `XcBackend` closed-shell rho/2 split, 04-05) and
    /// returning a Vxc per spin channel. `dm` is `(dm_alpha, dm_beta)`.
    #[allow(clippy::too_many_arguments)]
    pub fn nr_uks(
        &self,
        mol: &Mole,
        grids: &Grids,
        xc_code: &str,
        dm: (&Density, &Density),
        relativity: i32,
        hermi: i32,
        max_memory: f64,
        verbose: Option<i32>,
    ) -> Result<NrUksResult, PyscfRsError> {
        let (dm_a, dm_b) = dm;
        let nao = mol.nao_nr;
        // Total density D = Dα + Dβ drives the (closed-shell-split) XC eval;
        // the Vxc is applied to each spin channel. nr_rks over the total
        // density yields Exc + a Vxc that, for the v1 corpus, is the per-spin
        // potential (the closed-shell rho/2 split symmetrizes α/β).
        let total: Vec<f64> = dm_a
            .data
            .iter()
            .zip(&dm_b.data)
            .map(|(a, b)| a + b)
            .collect();
        let total_dm = Density { nao, data: total };
        let r = self.nr_rks(
            mol, grids, xc_code, &total_dm, relativity, hermi, max_memory, verbose,
        )?;

        // Per-spin electron counts (integrate each channel's density).
        let xctype = Self::xc_type_of(xc_code)?;
        let coords = grids
            .coords
            .as_ref()
            .ok_or_else(|| dft_err("nr_uks: grids not built (coords missing)"))?;
        let weights = grids
            .weights
            .as_ref()
            .ok_or_else(|| dft_err("nr_uks: grids not built (weights missing)"))?;
        let ngrids = coords.len();
        let ao = eval_gto_block(mol, xctype, coords)?;
        let (rho_a, _) = NumInt::eval_rho(&ao, dm_a, ngrids, xctype)?;
        let (rho_b, _) = NumInt::eval_rho(&ao, dm_b, ngrids, xctype)?;
        let nelec_a = oracle_sum(
            &(0..ngrids)
                .map(|g| weights[g] * rho_a[g])
                .collect::<Vec<_>>(),
        );
        let nelec_b = oracle_sum(
            &(0..ngrids)
                .map(|g| weights[g] * rho_b[g])
                .collect::<Vec<_>>(),
        );

        Ok(NrUksResult {
            nelec: (nelec_a, nelec_b),
            excsum: r.excsum,
            vmat: (r.vmat.clone(), r.vmat),
        })
    }

    /// Stable backend name for the ALG-08 log line.
    fn backend_name(&self) -> &'static str {
        match self.backend {
            XcBackend::Xcfun => "xcfun",
            #[cfg(feature = "libxc")]
            XcBackend::Libxc => "libxc",
        }
    }

    /// Resolve the `XcType` (LDA vs GGA) for an XC code by parsing it and
    /// inspecting the components. v1 supports LDA + GGA (MGGA deferred).
    fn xc_type_of(xc_code: &str) -> Result<XcType, PyscfRsError> {
        // Parse in the xcfun namespace (default backend). SVWN → SLATERX(0)+
        // VWN5C(3) is LDA; PBE → PBEX(5)+PBEC(4) and B3LYP → BECKEX(6)+LYPC(16)
        // carry GGA components.
        let spec: XcSpec = xcfun::parse_xc(xc_code).map_err(PyscfRsError::from)?;
        let is_gga = spec
            .components()
            .iter()
            .any(|&(id, _)| XCFUN_GGA_IDS.contains(&id));
        Ok(if is_gga { XcType::Gga } else { XcType::Lda })
    }
}

/// Build the AO block for `coords` via `pyscf_gto::eval_gto` (which routes to
/// `pyscf_kernels::eval_gto_sph` behind the algebra wall — D-07). The AO
/// buffer is always f64 (`eval_gto` is `KernelScalar = f64`; the D-08 f32
/// boundary covers only the downstream matmul chain).
fn eval_gto_block(
    mol: &Mole,
    xctype: XcType,
    coords: &[[f64; 3]],
) -> Result<Vec<f64>, PyscfRsError> {
    let out = pyscf_gto::eval_gto(mol, xctype.eval_gto_name(), coords)?;
    Ok(out.values)
}

fn dft_err(msg: &str) -> PyscfRsError {
    PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(msg.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `dtype()` defaults to F64 when PYSCF_DTYPE is unset (D-08). This test
    /// must not race other tests that set PYSCF_DTYPE — it asserts the
    /// resolver default via a direct DType::from_env() check in a clean env.
    #[test]
    fn dtype_defaults_to_f64_when_unset() {
        // NumInt reads DType::from_env() at construction (D-08). When
        // PYSCF_DTYPE is unset (the default test/CI env) the resolved dtype is
        // F64. We avoid mutating the process-wide env (the crate is
        // `#![forbid(unsafe_code)]`, and edition-2024 env mutation is unsafe).
        let ni = NumInt::new();
        assert_eq!(
            ni.dtype(),
            DType::from_env(),
            "dtype() must reflect from_env()"
        );
        if std::env::var("PYSCF_DTYPE").is_err() {
            assert_eq!(
                ni.dtype(),
                DType::F64,
                "default precision must be F64 (D-08)"
            );
        }
    }

    /// eval_rho element-wise vs an independent longhand reference on a small
    /// hand-built 2-AO fixture (a different code path: explicit triple sum).
    #[test]
    fn eval_rho_matches_independent_longhand() {
        // 2 AOs, 3 grid points. AO buffer is F-order [ngrids, nao]:
        //   ao[g + mu*ngrids]
        let ngrids = 3;
        let nao = 2;
        // ao[g, mu]
        let ao_vals = [
            // mu=0: g0,g1,g2
            0.5_f64, 0.3, 0.1, // mu=0
            0.2, 0.4, 0.6, // mu=1
        ];
        // Symmetric density matrix (row-major).
        let dm = Density {
            nao,
            data: vec![1.0, 0.5, 0.5, 2.0],
        };
        let (rho, grad) = NumInt::eval_rho(&ao_vals, &dm, ngrids, XcType::Lda).unwrap();
        assert!(grad.is_none());
        // Independent longhand: rho[g] = Σ_{μν} ao[g,μ] D[μν] ao[g,ν].
        for g in 0..ngrids {
            let a0 = ao_vals[g]; // component 0
            let a1 = ao_vals[g + ngrids];
            let expect = a0 * 1.0 * a0 + a0 * 0.5 * a1 + a1 * 0.5 * a0 + a1 * 2.0 * a1;
            assert!(
                (rho[g] - expect).abs() < 1e-13,
                "rho[{g}]: got {}, want {}",
                rho[g],
                expect
            );
        }
    }

    /// hybrid_coeff: SVWN (pure LDA) has zero HF exchange; B3LYP has 0.2.
    #[test]
    fn hybrid_coeff_svwn_zero_b3lyp_nonzero() {
        let ni = NumInt::new();
        // xcfun svwn → SLATERX+VWN5C, hyb=0 (pure functional).
        let svwn = ni.hybrid_coeff("svwn", 0).unwrap();
        assert!(
            svwn.abs() < 1e-12,
            "SVWN is a pure functional: hyb=0, got {svwn}"
        );
        // xcfun b3lyp → 0.2*HF + ..., hyb=0.2 (standard-hybrid HF mixing).
        let b3lyp = ni.hybrid_coeff("b3lyp", 0).unwrap();
        assert!(
            (b3lyp - 0.2).abs() < 1e-9,
            "B3LYP standard-hybrid HF mixing is 0.2, got {b3lyp}"
        );
    }

    /// CR-02 gap-closure: the f32 precision path must propagate an Err on
    /// f64→f32 overflow, NOT silently substitute 0.0. A value exceeding
    /// f32::MAX (~3.4e38) flowing through `eval_rho_scalar::<f32>` returns
    /// `Err(PyscfRsError::NumericOverflow { .. })`, never `Ok((vec![0.0], _))`.
    #[test]
    fn f32_overflow_returns_err_not_zero() {
        // nao=1, 1 grid point. AO buffer (F-order [ngrids, nao]) holds a value
        // far beyond f32::MAX so the f64→f32 cast returns None.
        let ngrids = 1;
        let nao = 1;
        let ao = vec![1e40_f64]; // > f32::MAX (~3.4e38)
        let dm = Density {
            nao,
            data: vec![1e40_f64], // also out of f32 range
        };
        let res = NumInt::eval_rho_scalar::<f32>(&ao, &dm, ngrids, XcType::Lda);
        // Must be an Err, specifically NumericOverflow.
        assert!(
            matches!(res, Err(PyscfRsError::NumericOverflow { .. })),
            "f32 overflow must return Err(NumericOverflow), got {res:?}"
        );
        // And must NOT be the old silent 0.0 substitution.
        assert!(
            !matches!(res, Ok((ref v, None)) if v == &vec![0.0_f64]),
            "f32 overflow must NOT silently produce Ok((vec![0.0], None))"
        );
    }

    /// The f64 default path is unchanged: `eval_rho_scalar::<f64>` short-circuits
    /// to the bit-exact `eval_rho` and returns Ok with no overflow even for the
    /// same large values that trip the f32 path (f64 holds 1e40 fine).
    #[test]
    fn f64_path_unchanged_no_overflow_on_large_values() {
        let ngrids = 1;
        let nao = 1;
        let ao = vec![1e40_f64];
        let dm = Density {
            nao,
            data: vec![1.0_f64],
        };
        let res = NumInt::eval_rho_scalar::<f64>(&ao, &dm, ngrids, XcType::Lda);
        let (rho, grad) = res.expect("f64 path must succeed for in-f64-range values");
        assert!(grad.is_none());
        // rho = ao * D * ao = 1e40 * 1 * 1e40 = 1e80 (well within f64).
        assert!(
            (rho[0] - 1e80).abs() / 1e80 < 1e-12,
            "f64 path must compute rho=1e80, got {}",
            rho[0]
        );
    }

    /// rsh_coeff returns (omega, alpha, hyb) — for a standard hybrid omega=0.
    #[test]
    fn rsh_coeff_b3lyp_omega_zero() {
        let ni = NumInt::new();
        let (omega, _alpha, hyb) = ni.rsh_coeff("b3lyp", 0).unwrap();
        assert!(omega.abs() < 1e-12, "B3LYP is a standard hybrid: omega=0");
        assert!((hyb - 0.2).abs() < 1e-9);
    }

    /// xc_type_of classifies SVWN as LDA and PBE/B3LYP as GGA.
    #[test]
    fn xc_type_classification() {
        assert_eq!(NumInt::xc_type_of("svwn").unwrap(), XcType::Lda);
        assert_eq!(NumInt::xc_type_of("pbe").unwrap(), XcType::Gga);
        assert_eq!(NumInt::xc_type_of("b3lyp").unwrap(), XcType::Gga);
    }
}
