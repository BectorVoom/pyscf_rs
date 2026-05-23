//! `XcBackend` — the cfg-gated XC evaluation seam (D-03, DFT-03).
//!
//! Structural model: `pyscf-algebra::client::AlgebraClient` (client.rs:10-39)
//! — an enum-of-clients with a `#[cfg(feature = …)]` variant and a
//! `match self { … }` dispatch. Here the default-compiled variant is `Xcfun`
//! (routes to the always-building `xcfun_rs`); the `Libxc` variant and ALL
//! libxc eval code live behind `#[cfg(feature = "libxc")]`, so a default build
//! never names a `libxc_rs` symbol (Pitfall 5 — `libxc_rs` is 266 kernel
//! crates, ~6h compile).
//!
//! License: Apache-2.0. Requirement: DFT-03 (bit-identical XC eval).
//!
//! Family dispatch: the parsed [`XcSpec`] is evaluated over a block of grid
//! points whose density-derivative input is sized by family —
//! LDA(rho) / GGA(rho,sigma) / MGGA(rho,sigma,lapl,tau) — mirroring upstream
//! `eval_xc`'s `rho` row count.
//!
//! # WGPU f64 honesty (DFT-11) and the D-08 f32 escape hatch
//!
//! Requirement DFT-11 closes the "WGPU-f64-hole" honesty gap (Pitfall 6 / the
//! Pitfall-3 silent-precision-degradation re-validation) at the XC-eval seam.
//! Two paths must NEVER be confused:
//!
//! 1. **Default f64 + a wgpu adapter without `shader-f64`** → the pipeline
//!    falls back to **CPU-f64** and emits a single [`tracing::warn!`], it
//!    NEVER silently downgrades to f32 (that would lie to the chemistry user).
//!    The XC-eval-portion capability decision is DELEGATED to `xcfun_rs`'s
//!    existing probe — [`xcfun_gpu::auto_backend`] selects the runtime and
//!    [`xcfun_gpu::error_routing::must_fall_back_to_cpu`] routes ERF-bearing
//!    (range-separated) functionals on `Wgpu`/`Metal` to CPU. We do NOT
//!    re-implement a `shader-f64` probe here (see [`xc_eval_substrate`]).
//!    `libxc_rs` is CPU-host-only by construction, so its XC-eval path is
//!    ALWAYS CPU regardless of the selected GPU runtime.
//!
//! 2. **User-set `PYSCF_DTYPE=f32`** → the SEPARATE, EXPLICIT opt-in escape
//!    hatch (D-08). When a user deliberately sets `PYSCF_DTYPE=f32`, the
//!    04-06 `NumInt` f32 matmul chain runs and the below-bit-exact
//!    `tracing::warn!` fires at kernel entry. This is HONEST (user-requested)
//!    precision, NOT a silent degradation — it is the only way to actually run
//!    DFT *on the GPU* of a `shader-f64`-less wgpu adapter (the f64 path on
//!    that adapter is hard-errored in `pyscf-algebra::select` per D-09; f32
//!    lets the user opt past that). The explicit-f32 case is therefore NEVER
//!    treated as a "silent degradation to block" — it is the documented
//!    escape hatch (see `docs/env-vars.md` and [`pipeline_fallback`]).
//!
//! The pipeline-level wgpu→CPU fallback reuses Phase 1's `PYSCF_BACKEND`
//! resolver model (`pyscf_runtime::BackendKind` + the unrecognised-backend
//! "fall back to CPU + warn" pattern, mirrored from `pyscf-algebra::select`).

use crate::error::DftError;
use crate::parser::XcSpec;

// DFT-11 delegation targets — the EXISTING xcfun_rs shader-f64/ERF probe, NOT
// a re-implementation. `Backend` is the runtime discriminator; `auto_backend`
// is the priority-chain selector (env override → ROCm → CUDA → Metal-f64 →
// Wgpu-f64 → CPU); `must_fall_back_to_cpu` routes ERF-bearing functionals on
// Wgpu/Metal to the CPU substrate. `Dependency` (re-exported by xcfun-rs from
// xcfun-core) is the functional capability bitset these consume.
use xcfun_gpu::Backend as XcfunBackend;
use xcfun_gpu::auto_backend;
use xcfun_gpu::error_routing::must_fall_back_to_cpu;
use xcfun_rs::Dependency;

/// Functional family — selects the per-point input layout and the backend's
/// variable encoding (xcfun `Vars`, libxc `Family`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    /// Local density approximation: input is `rho` only.
    Lda,
    /// Generalized-gradient approximation: input is `(rho, sigma)`.
    Gga,
    /// Meta-GGA: input is `(rho, sigma, lapl, tau)`.
    Mgga,
}

impl Family {
    /// Number of per-point input scalars (unpolarized) for this family.
    /// LDA = 1 (rho), GGA = 2 (rho, sigma), MGGA = 4 (rho, sigma, lapl, tau).
    pub fn input_width(self) -> usize {
        match self {
            Family::Lda => 1,
            Family::Gga => 2,
            Family::Mgga => 4,
        }
    }
}

/// One grid block's density-derivative input, family-sized (unpolarized).
///
/// The `*_block` slices are per-point columns (length = number of grid points).
/// LDA reads only `rho`; GGA additionally reads `sigma`; MGGA additionally
/// reads `lapl` + `tau`. This is the family-dispatch sizing the backend uses to
/// build the flat per-point input buffer.
#[derive(Debug, Clone)]
pub enum RhoBlock<'a> {
    /// `rho[ip]`.
    Lda { rho: &'a [f64] },
    /// `rho[ip]`, `sigma[ip] = |∇rho|²`.
    Gga { rho: &'a [f64], sigma: &'a [f64] },
    /// `rho`, `sigma`, `lapl = ∇²rho`, `tau = kinetic energy density`.
    Mgga {
        rho: &'a [f64],
        sigma: &'a [f64],
        lapl: &'a [f64],
        tau: &'a [f64],
    },
}

impl RhoBlock<'_> {
    /// The family implied by which density-derivative rows are present.
    pub fn family(&self) -> Family {
        match self {
            RhoBlock::Lda { .. } => Family::Lda,
            RhoBlock::Gga { .. } => Family::Gga,
            RhoBlock::Mgga { .. } => Family::Mgga,
        }
    }

    /// Number of grid points in the block.
    pub fn n_points(&self) -> usize {
        match self {
            RhoBlock::Lda { rho } => rho.len(),
            RhoBlock::Gga { rho, .. } => rho.len(),
            RhoBlock::Mgga { rho, .. } => rho.len(),
        }
    }
}

/// Derivative order requested from the functional (mirrors libxc
/// `DerivativeOrder` / the xcfun `order` argument): 0 = energy density only,
/// 1 = + first derivatives (the XC potential), etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DerivOrder {
    /// Exc only.
    Exc = 0,
    /// Exc + Vxc (first derivatives).
    Vxc = 1,
    /// + second derivatives (fxc).
    Fxc = 2,
}

impl DerivOrder {
    fn as_u32(self) -> u32 {
        self as u32
    }
}

/// XC evaluation output over a grid block (unpolarized).
///
/// `exc[ip]` is the energy density per unit volume at point `ip`
/// (xcfun `output[0]` for `Mode::PartialDerivatives`); `vrho[ip]` is the
/// first derivative ∂f/∂rho (the LDA/GGA potential). Higher-order derivatives
/// are returned in the family-specific extra fields when `order >= Vxc`.
#[derive(Debug, Clone, Default)]
pub struct XcOutput {
    /// Per-point XC energy density (`f = rho · e_xc`).
    pub exc: Vec<f64>,
    /// Per-point ∂f/∂rho (present when `order >= Vxc`).
    pub vrho: Vec<f64>,
    /// Per-point ∂f/∂sigma (GGA/MGGA, `order >= Vxc`).
    pub vsigma: Vec<f64>,
}

/// Spin-polarized (open-shell, UKS) XC evaluation output over a grid block
/// (CR-01).
///
/// Unlike [`XcOutput`] (which is the closed-shell `rho/2`-split surface), this
/// carries the GENUINE per-spin first derivatives: `vrho_a[ip] = ∂f/∂rho_a`
/// and `vrho_b[ip] = ∂f/∂rho_b` are computed from independent `rho_a`/`rho_b`
/// inputs, so they DIFFER for an asymmetric (`rho_a != rho_b`) density. The
/// `vsigma_*` fields are the GGA gradient-variable derivatives
/// (`∂f/∂sigma_aa`, `∂f/∂sigma_ab`, `∂f/∂sigma_bb`); they are all-zero for LDA.
#[derive(Debug, Clone, Default)]
pub struct UksXcOutput {
    /// Per-point XC energy density `f` (xcfun `output[0]`).
    pub exc: Vec<f64>,
    /// Per-point ∂f/∂rho_a — the alpha-spin LDA/GGA potential.
    pub vrho_a: Vec<f64>,
    /// Per-point ∂f/∂rho_b — the beta-spin LDA/GGA potential.
    pub vrho_b: Vec<f64>,
    /// Per-point ∂f/∂sigma_aa (GGA; zero for LDA).
    pub vsigma_aa: Vec<f64>,
    /// Per-point ∂f/∂sigma_ab (GGA cross-term; zero for LDA).
    pub vsigma_ab: Vec<f64>,
    /// Per-point ∂f/∂sigma_bb (GGA; zero for LDA).
    pub vsigma_bb: Vec<f64>,
}

/// The cfg-gated XC backend seam (D-03). `Xcfun` is the default-compiled
/// backend; `Libxc` is only present under `--features libxc`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum XcBackend {
    /// Native-Rust xcfun (`xcfun_rs`) — the DEFAULT backend, always compiled.
    #[default]
    Xcfun,
    /// Native-Rust libxc (`libxc_rs`) — ONLY compiled under `--features libxc`
    /// (CI-only; the default build never references libxc_rs symbols).
    #[cfg(feature = "libxc")]
    Libxc,
}

impl XcBackend {
    /// Evaluate the functional `spec` over `rho` at derivative `order`.
    ///
    /// Dispatch mirrors `AlgebraClient` (client.rs match-on-self):
    /// - `Xcfun` → the xcfun_rs path (below).
    /// - `Libxc` (gated) → the libxc_rs path (in the `#[cfg(feature =
    ///   "libxc")]` module).
    /// - When the crate is built WITHOUT `--features libxc`, there is no
    ///   `Libxc` variant to match, so the dispatch is exhaustive over `Xcfun`
    ///   alone and never names a libxc_rs symbol.
    ///
    /// # Errors
    /// [`DftError::BackendEval`] on a backend-side failure (unknown functional
    /// name, buffer mismatch); [`DftError::LibxcFeatureNotEnabled`] is reserved
    /// for callers that explicitly request the libxc backend in a default
    /// build (see [`XcBackend::require_libxc`]).
    pub fn eval(
        &self,
        spec: &XcSpec,
        rho: &RhoBlock<'_>,
        order: DerivOrder,
    ) -> Result<XcOutput, DftError> {
        match self {
            XcBackend::Xcfun => xcfun_eval(spec, rho, order),
            #[cfg(feature = "libxc")]
            XcBackend::Libxc => libxc_impl::libxc_eval(spec, rho, order),
        }
    }

    /// Spin-polarized (open-shell, UKS) evaluation (CR-01).
    ///
    /// Evaluates the functional `spec` over INDEPENDENT alpha/beta densities
    /// `rho_a`/`rho_b` (and, for GGA, the gradient variables `sigma_aa`,
    /// `sigma_ab`, `sigma_bb`) and returns the per-spin first derivatives in a
    /// [`UksXcOutput`]. This is the genuine open-shell path: `vrho_a` and
    /// `vrho_b` differ whenever `rho_a != rho_b`, so the caller can build two
    /// distinct `Vxc` matrices (NOT one cloned twice).
    ///
    /// Family is inferred from `sigma_aa`: `None` → LDA (`A_B`), `Some` → GGA
    /// (`A_B_GAA_GAB_GBB`). It is cross-checked against the functional's
    /// declared family; a mismatch is a clean [`DftError::BackendEval`].
    ///
    /// # Errors
    /// [`DftError::BackendEval`] on a backend-side failure (unknown name,
    /// family mismatch, buffer mismatch). The `libxc` backend arm returns
    /// `BackendEval("libxc UKS eval not yet implemented")` — UKS libxc support
    /// is deferred (CR-01 scopes the xcfun default path only).
    #[allow(clippy::too_many_arguments)]
    pub fn eval_uks(
        &self,
        spec: &XcSpec,
        rho_a: &[f64],
        rho_b: &[f64],
        sigma_aa: Option<&[f64]>,
        sigma_ab: Option<&[f64]>,
        sigma_bb: Option<&[f64]>,
        order: DerivOrder,
    ) -> Result<UksXcOutput, DftError> {
        match self {
            XcBackend::Xcfun => {
                xcfun_eval_uks(spec, rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb, order)
            }
            #[cfg(feature = "libxc")]
            XcBackend::Libxc => Err(DftError::BackendEval(
                "libxc UKS eval not yet implemented".into(),
            )),
        }
    }

    /// Helper for callers that want the libxc backend: returns
    /// `Err(LibxcFeatureNotEnabled)` in a default build instead of silently
    /// falling back. The `#[cfg(not(feature = "libxc"))]` arm is the
    /// "feature off" branch the interface contract names.
    pub fn require_libxc() -> Result<XcBackend, DftError> {
        #[cfg(feature = "libxc")]
        {
            Ok(XcBackend::Libxc)
        }
        #[cfg(not(feature = "libxc"))]
        {
            Err(DftError::LibxcFeatureNotEnabled)
        }
    }
}

/// Where the XC-eval portion actually runs after the DFT-11 honesty decision.
///
/// This is the *substrate* the per-block XC evaluation lands on, distinct from
/// the precision (`DType`) the surrounding matmul chain uses. The default-f64
/// honesty contract is: a `shader-f64`-less wgpu/metal adapter NEVER silently
/// computes f32 XC energies — it falls back to [`Cpu`](XcEvalSubstrate::Cpu).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XcEvalSubstrate {
    /// The XC eval runs on the CPU substrate (always f64-correct). This is the
    /// honest fallback for a `shader-f64`-less GPU adapter under default f64,
    /// and is ALWAYS the substrate for the CPU-host-only `libxc_rs` path.
    Cpu,
    /// The XC eval runs on the selected GPU runtime (probe reported f64
    /// capability and no ERF→CPU routing was required).
    Gpu,
}

/// DFT-11 — decide the XC-eval substrate by **delegating** to `xcfun_rs`'s
/// existing probe. This re-implements NOTHING: it calls
/// [`xcfun_gpu::auto_backend`] (the priority-chain selector that already gates
/// `Wgpu` on the `shader-f64` Vulkan extension and `Metal` on hardware f64)
/// and [`xcfun_gpu::error_routing::must_fall_back_to_cpu`] (ERF-bearing
/// range-separated functionals on `Wgpu`/`Metal` → CPU).
///
/// `deps` is the functional's capability bitset (from
/// `xcfun_rs::Functional::dependencies`-derived flags, e.g. `Dependency::ERF`
/// for range-separated functionals). Returns [`XcEvalSubstrate::Cpu`] whenever
/// the resolved backend is the CPU substrate OR an ERF-fallback is required;
/// otherwise [`XcEvalSubstrate::Gpu`].
///
/// On the common case / CI default (no `shader-f64` device, no
/// `XCFUN_FORCE_BACKEND`), `auto_backend()` returns `Backend::Cpu`, so this
/// returns `Cpu` — the honest default-f64 fallback. NEVER returns a path that
/// silently computes f32 for an f64 request.
pub fn xc_eval_substrate(deps: Dependency) -> XcEvalSubstrate {
    let backend = auto_backend();
    // ERF-bearing functionals on wgpu/metal route to CPU (range-separation
    // needs f64; WGSL has no f64 type) — delegated, not re-implemented.
    if must_fall_back_to_cpu(deps, backend) {
        return XcEvalSubstrate::Cpu;
    }
    match backend {
        XcfunBackend::Cpu => XcEvalSubstrate::Cpu,
        // Wgpu/Metal are only ever returned by auto_backend() when the
        // shader-f64 / hardware-f64 probe PASSED (the gate is inside
        // auto_backend), so an f64 XC eval there is honest.
        XcfunBackend::Wgpu | XcfunBackend::Metal | XcfunBackend::Cuda | XcfunBackend::Rocm => {
            XcEvalSubstrate::Gpu
        }
    }
}

/// The pipeline-level wgpu→CPU honesty fallback (DFT-11), reusing Phase 1's
/// `PYSCF_BACKEND` resolver model. Reads `PYSCF_BACKEND`; when the user
/// explicitly requested `wgpu`/`metal` but [`xc_eval_substrate`] reports the
/// XC eval cannot run there at f64 (the `shader-f64`-less adapter case) AND the
/// active precision is **f64** (`f64_requested == true`), emits a single clear
/// [`tracing::warn!`] (mirroring `pyscf-algebra::select`'s uncompiled-backend
/// warning) and returns [`XcEvalSubstrate::Cpu`] — the eval then runs on CPU
/// producing correct f64 numbers, NEVER a silent f32 downgrade.
///
/// `f64_requested` is `true` when the active `PYSCF_DTYPE` is f64 (the default).
/// When the user has explicitly set `PYSCF_DTYPE=f32` (`f64_requested == false`)
/// this is the D-08 escape hatch: the f32 run is the user's honest, warned
/// opt-in (handled by the 04-06 `NumInt` f32 matmul chain + its below-bit-exact
/// warn), NOT a silent degradation — so NO additional fallback warning fires
/// here and the (already-warned) f32 path proceeds.
///
/// Returns the resolved substrate for the XC eval. `deps` is the functional
/// capability bitset (forwarded to [`xc_eval_substrate`]).
pub fn pipeline_fallback(deps: Dependency, f64_requested: bool) -> XcEvalSubstrate {
    let requested = std::env::var("PYSCF_BACKEND").ok();
    let normalised = requested.as_deref().unwrap_or("auto").to_ascii_lowercase();
    let explicit_gpu = matches!(normalised.as_str(), "wgpu" | "metal");

    let substrate = xc_eval_substrate(deps);

    if substrate == XcEvalSubstrate::Cpu && explicit_gpu && f64_requested {
        // Default-f64 honesty path: the user asked for a GPU adapter but it
        // lacks shader-f64 for the requested f64 precision. Fall back to CPU
        // with a warning — NEVER silently compute f32 (DFT-11 / Pitfall 6).
        // This mirrors Phase 1 `pyscf-algebra::select`'s uncompiled/unavailable
        // backend warn-and-fall-back-to-CPU model (PYSCF_BACKEND resolver).
        tracing::warn!(
            target: "pyscf_dft::xc_backend",
            backend = %normalised,
            "PYSCF_BACKEND={normalised} requested but the adapter lacks shader-f64 for the \
             requested f64 precision; the XC eval falls back to CPU-f64 (correct numbers) \
             rather than silently degrading to f32. Set PYSCF_DTYPE=f32 to opt explicitly \
             into the (below-bit-exact) f32 GPU path (D-08), or PYSCF_BACKEND=cpu/auto."
        );
    }
    // When f64_requested == false (PYSCF_DTYPE=f32) the f32 opt-in is the
    // honest, already-warned escape hatch (D-08) — do NOT emit a fallback warn
    // and do NOT block it; the 04-06 NumInt f32 chain owns that warning.
    substrate
}

/// Map an xcfun functional id (0..77, as emitted by `parser::xcfun`) to the
/// canonical xcfun name accepted by `xcfun_rs::Functional::set`. Only the
/// parity-corpus subset is mapped; an unmapped id is a clean `BackendEval`
/// error (never a panic).
fn xcfun_id_to_name(id: u32) -> Option<&'static str> {
    Some(match id {
        0 => "SLATERX",
        2 => "VWN3C",
        3 => "VWN5C",
        4 => "PBEC",
        5 => "PBEX",
        6 => "BECKEX",
        16 => "LYPC",
        17 => "OPTX",
        19 => "REVPBEX",
        20 => "RPBEX",
        26 => "PW91X",
        28 => "PW92C",
        56 => "P86C",
        77 => "PW91C",
        _ => return None,
    })
}

/// The xcfun_rs (default) evaluation path. Builds a `Functional` from the
/// parsed spec, picks the family + `Vars`/`Mode`, and runs `eval_vec` over the
/// block. Family is detected from the spec via `is_gga`/`is_metagga` AND the
/// supplied `RhoBlock` (they must agree).
fn xcfun_eval(spec: &XcSpec, rho: &RhoBlock<'_>, order: DerivOrder) -> Result<XcOutput, DftError> {
    use xcfun_rs::{Functional, Mode, Vars};

    let mut func = Functional::new();
    for &(id, fac) in spec.components() {
        let name = xcfun_id_to_name(id).ok_or_else(|| {
            DftError::BackendEval(format!("xcfun: no name mapping for functional id {id}"))
        })?;
        func.set(name, fac)
            .map_err(|e| DftError::BackendEval(format!("xcfun set('{name}',{fac}): {e:?}")))?;
    }

    // Family dispatch: pick Vars by the functional's declared dependencies,
    // cross-checked against the RhoBlock the caller supplied.
    let detected = if func.is_metagga() {
        Family::Mgga
    } else if func.is_gga() {
        Family::Gga
    } else {
        Family::Lda
    };
    if detected != rho.family() {
        return Err(DftError::BackendEval(format!(
            "xcfun: functional family {:?} does not match supplied RhoBlock family {:?}",
            detected,
            rho.family()
        )));
    }

    // Spin-resolved Vars per family. The CPU kernel-launch path supports the
    // spin-resolved (alpha/beta) layouts ONLY — the total-density `N`/`A`
    // variants are not in `xcfun_kernels::dispatch` (verified empirically:
    // `Vars::N`/`Vars::A` return `NotConfigured`). For a closed-shell
    // (spin-unpolarized) block we split the density symmetrically:
    //   rho_a = rho_b = rho/2,  sigma_{aa,ab,bb} = sigma/4,  tau_{a,b} = tau/2
    // which reproduces the total-density XC energy (verified: SLATERX at
    // rho_a=rho_b=0.5 gives the exact total-rho Slater value).
    //   LDA  -> A_B                          (a, b)
    //   GGA  -> A_B_GAA_GAB_GBB              (a, b, gaa, gab, gbb)
    //   MGGA -> A_B_GAA_GAB_GBB_TAUA_TAUB    (a, b, gaa, gab, gbb, taua, taub)
    let vars = match detected {
        Family::Lda => Vars::A_B,
        Family::Gga => Vars::A_B_GAA_GAB_GBB,
        Family::Mgga => Vars::A_B_GAA_GAB_GBB_TAUA_TAUB,
    };

    let np = rho.n_points();
    let mut out = XcOutput {
        exc: vec![0.0; np],
        vrho: vec![0.0; np],
        vsigma: vec![0.0; np],
    };
    if np == 0 {
        return Ok(out);
    }

    func.eval_setup(vars, Mode::PartialDerivatives, order.as_u32())
        .map_err(|e| DftError::BackendEval(format!("xcfun eval_setup: {e:?}")))?;

    let inlen = func.input_length();
    let outlen = func
        .output_length()
        .map_err(|e| DftError::BackendEval(format!("xcfun output_length: {e:?}")))?;

    // Build the pitched per-point input buffer with the closed-shell spin
    // split (eval_vec layout: point p reads density[p*inlen .. p*inlen+inlen]).
    let mut density = vec![0.0_f64; np * inlen];
    for ip in 0..np {
        let base = ip * inlen;
        match rho {
            RhoBlock::Lda { rho } => {
                let half = rho[ip] * 0.5;
                density[base] = half; // a
                density[base + 1] = half; // b
            }
            RhoBlock::Gga { rho, sigma } => {
                let half = rho[ip] * 0.5;
                let quarter = sigma[ip] * 0.25;
                density[base] = half; // a
                density[base + 1] = half; // b
                density[base + 2] = quarter; // gaa
                density[base + 3] = quarter; // gab
                density[base + 4] = quarter; // gbb
            }
            RhoBlock::Mgga {
                rho, sigma, tau, ..
            } => {
                let half = rho[ip] * 0.5;
                let quarter = sigma[ip] * 0.25;
                let tau_half = tau[ip] * 0.5;
                density[base] = half; // a
                density[base + 1] = half; // b
                density[base + 2] = quarter; // gaa
                density[base + 3] = quarter; // gab
                density[base + 4] = quarter; // gbb
                density[base + 5] = tau_half; // taua
                density[base + 6] = tau_half; // taub
            }
        }
    }
    let mut out_buf = vec![0.0_f64; np * outlen];

    func.eval_vec(&density, inlen, &mut out_buf, outlen, np)
        .map_err(|e| DftError::BackendEval(format!("xcfun eval_vec: {e:?}")))?;

    // Output layout (PartialDerivatives): out[0]=energy density,
    // out[1..inlen+1]=∂/∂x_i (potential). For the spin-resolved layout the
    // first two derivatives are vrho_a, vrho_b (equal for closed-shell), and
    // the gradient derivatives follow. We surface vrho_a as the (closed-shell)
    // density potential and the first gradient derivative as vsigma.
    for ip in 0..np {
        let base = ip * outlen;
        out.exc[ip] = out_buf[base];
        if order >= DerivOrder::Vxc && outlen > 1 {
            out.vrho[ip] = out_buf[base + 1]; // ∂f/∂rho_a
            if detected != Family::Lda && outlen > 3 {
                // For A_B_GAA_GAB_GBB the order-1 block is
                // [f, da, db, dgaa, dgab, dgbb]; index 3 = ∂f/∂gaa.
                out.vsigma[ip] = out_buf[base + 3];
            }
        }
    }
    Ok(out)
}

/// The xcfun_rs spin-polarized (open-shell, UKS) evaluation path (CR-01).
///
/// Mirrors [`xcfun_eval`] but builds the per-point input from the GENUINE
/// `rho_a[ip]`/`rho_b[ip]` (NOT a `rho/2` symmetric split), so the returned
/// `vrho_a`/`vrho_b` differ for an asymmetric density. Family is inferred from
/// `sigma_aa` (`None` → LDA, `Some` → GGA) and cross-checked against the
/// functional's declared family.
///
/// xcfun spin-resolved layout:
///   - LDA (`A_B`):              input `[rho_a, rho_b]`,
///     order-1 output `[f, ∂f/∂rho_a, ∂f/∂rho_b]`.
///   - GGA (`A_B_GAA_GAB_GBB`):  input `[rho_a, rho_b, gaa, gab, gbb]`,
///     order-1 output `[f, ∂f/∂rho_a, ∂f/∂rho_b, ∂f/∂gaa, ∂f/∂gab, ∂f/∂gbb]`.
#[allow(clippy::too_many_arguments)]
fn xcfun_eval_uks(
    spec: &XcSpec,
    rho_a: &[f64],
    rho_b: &[f64],
    sigma_aa: Option<&[f64]>,
    sigma_ab: Option<&[f64]>,
    sigma_bb: Option<&[f64]>,
    order: DerivOrder,
) -> Result<UksXcOutput, DftError> {
    use xcfun_rs::{Functional, Mode, Vars};

    let np = rho_a.len();
    if rho_b.len() != np {
        return Err(DftError::BackendEval(format!(
            "xcfun_eval_uks: rho_a len {np} != rho_b len {}",
            rho_b.len()
        )));
    }

    let mut func = Functional::new();
    for &(id, fac) in spec.components() {
        let name = xcfun_id_to_name(id).ok_or_else(|| {
            DftError::BackendEval(format!("xcfun: no name mapping for functional id {id}"))
        })?;
        func.set(name, fac)
            .map_err(|e| DftError::BackendEval(format!("xcfun set('{name}',{fac}): {e:?}")))?;
    }

    // Family detection from the functional's declared dependencies, cross-
    // checked against the supplied gradient-variable inputs (sigma_aa present
    // ⇒ caller intends GGA; absent ⇒ LDA). MGGA UKS is deferred (CR-01).
    let detected = if func.is_metagga() {
        Family::Mgga
    } else if func.is_gga() {
        Family::Gga
    } else {
        Family::Lda
    };
    let supplied = if sigma_aa.is_some() {
        Family::Gga
    } else {
        Family::Lda
    };
    if detected == Family::Mgga {
        return Err(DftError::BackendEval(
            "xcfun_eval_uks: MGGA open-shell eval not yet implemented (CR-01 covers LDA + GGA)"
                .into(),
        ));
    }
    if detected != supplied {
        return Err(DftError::BackendEval(format!(
            "xcfun_eval_uks: functional family {detected:?} does not match supplied input \
             family {supplied:?} (sigma_aa {} present)",
            if sigma_aa.is_some() { "is" } else { "is NOT" }
        )));
    }

    // Spin-resolved Vars per family — A_B (LDA) / A_B_GAA_GAB_GBB (GGA).
    let vars = match detected {
        Family::Lda => Vars::A_B,
        Family::Gga => Vars::A_B_GAA_GAB_GBB,
        Family::Mgga => unreachable!("MGGA rejected above"),
    };

    let mut out = UksXcOutput {
        exc: vec![0.0; np],
        vrho_a: vec![0.0; np],
        vrho_b: vec![0.0; np],
        vsigma_aa: vec![0.0; np],
        vsigma_ab: vec![0.0; np],
        vsigma_bb: vec![0.0; np],
    };
    if np == 0 {
        return Ok(out);
    }

    func.eval_setup(vars, Mode::PartialDerivatives, order.as_u32())
        .map_err(|e| DftError::BackendEval(format!("xcfun eval_setup: {e:?}")))?;

    let inlen = func.input_length();
    let outlen = func
        .output_length()
        .map_err(|e| DftError::BackendEval(format!("xcfun output_length: {e:?}")))?;

    // For GGA the caller must supply all three sigma channels.
    let (saa, sab, sbb) = if detected == Family::Gga {
        let saa = sigma_aa.ok_or_else(|| {
            DftError::BackendEval("xcfun_eval_uks: GGA requires sigma_aa".into())
        })?;
        let sab = sigma_ab.ok_or_else(|| {
            DftError::BackendEval("xcfun_eval_uks: GGA requires sigma_ab".into())
        })?;
        let sbb = sigma_bb.ok_or_else(|| {
            DftError::BackendEval("xcfun_eval_uks: GGA requires sigma_bb".into())
        })?;
        if saa.len() != np || sab.len() != np || sbb.len() != np {
            return Err(DftError::BackendEval(
                "xcfun_eval_uks: sigma_* length must equal rho length".into(),
            ));
        }
        (Some(saa), Some(sab), Some(sbb))
    } else {
        (None, None, None)
    };

    // Build the per-point input from the GENUINE rho_a/rho_b (NOT a rho/2
    // split). Point p reads density[p*inlen .. p*inlen+inlen].
    let mut density = vec![0.0_f64; np * inlen];
    for ip in 0..np {
        let base = ip * inlen;
        density[base] = rho_a[ip]; // a
        density[base + 1] = rho_b[ip]; // b
        if detected == Family::Gga {
            density[base + 2] = saa.unwrap()[ip]; // gaa
            density[base + 3] = sab.unwrap()[ip]; // gab
            density[base + 4] = sbb.unwrap()[ip]; // gbb
        }
    }
    let mut out_buf = vec![0.0_f64; np * outlen];

    func.eval_vec(&density, inlen, &mut out_buf, outlen, np)
        .map_err(|e| DftError::BackendEval(format!("xcfun eval_vec: {e:?}")))?;

    // Output layout (PartialDerivatives, spin-resolved):
    //   LDA: [f, ∂f/∂rho_a, ∂f/∂rho_b]                         (indices 0..2)
    //   GGA: [f, ∂f/∂rho_a, ∂f/∂rho_b, ∂f/∂gaa, ∂f/∂gab, ∂f/∂gbb] (0..5)
    for ip in 0..np {
        let base = ip * outlen;
        out.exc[ip] = out_buf[base];
        if order >= DerivOrder::Vxc && outlen > 2 {
            out.vrho_a[ip] = out_buf[base + 1];
            out.vrho_b[ip] = out_buf[base + 2];
            if detected == Family::Gga && outlen > 5 {
                out.vsigma_aa[ip] = out_buf[base + 3];
                out.vsigma_ab[ip] = out_buf[base + 4];
                out.vsigma_bb[ip] = out_buf[base + 5];
            }
        }
    }
    Ok(out)
}

/// libxc_rs (gated) evaluation path. Entirely inside `#[cfg(feature =
/// "libxc")]` so a default build never references `libxc_rs`.
#[cfg(feature = "libxc")]
mod libxc_impl {
    use super::{DerivOrder, DftError, Family, RhoBlock, XcOutput, XcSpec};
    use libxc_rs::{
        BatchEvaluator, DerivativeOrder, Functional, FunctionalId, GgaInput, GgaOutput, LdaInput,
        LdaOutput, MggaInput, MggaOutput, Spin,
    };

    fn deriv_order(order: DerivOrder) -> DerivativeOrder {
        match order {
            DerivOrder::Exc => DerivativeOrder::Exc,
            DerivOrder::Vxc => DerivativeOrder::Vxc,
            DerivOrder::Fxc => DerivativeOrder::Fxc,
        }
    }

    /// Evaluate the spec via libxc_rs, summing each component's contribution
    /// weighted by its factor. Family/derivative dispatch picks
    /// LdaInput/GgaInput/MggaInput by the libxc functional's `family()`.
    pub(super) fn libxc_eval(
        spec: &XcSpec,
        rho: &RhoBlock<'_>,
        order: DerivOrder,
    ) -> Result<XcOutput, DftError> {
        let np = rho.n_points();
        let spin = Spin::Unpolarized;
        let dord = deriv_order(order);

        let mut out = XcOutput {
            exc: vec![0.0; np],
            vrho: vec![0.0; np],
            vsigma: vec![0.0; np],
        };
        if np == 0 {
            return Ok(out);
        }

        let mut batch = BatchEvaluator::new(spin, np);

        for &(id, fac) in spec.components() {
            let fid = FunctionalId::from_raw(u16::try_from(id).map_err(|_| {
                DftError::BackendEval(format!("libxc: functional id {id} out of range"))
            })?)
            .map_err(|e| DftError::BackendEval(format!("libxc lookup id {id}: {e:?}")))?;
            let func = Functional::new(fid, spin)
                .map_err(|e| DftError::BackendEval(format!("libxc Functional::new: {e:?}")))?;

            // Family dispatch via the functional's declared family.
            match fid.family() {
                libxc_rs::Family::Lda => {
                    let RhoBlock::Lda { rho } = rho else {
                        return Err(family_mismatch(Family::Lda, rho.family()));
                    };
                    let input = LdaInput::new(rho, np, spin)
                        .map_err(|e| DftError::BackendEval(format!("libxc LdaInput: {e:?}")))?;
                    let mut zk = vec![0.0; np];
                    let mut vrho = vec![0.0; np];
                    let mut o = LdaOutput::new(
                        Some(&mut zk),
                        if order >= DerivOrder::Vxc {
                            Some(&mut vrho)
                        } else {
                            None
                        },
                        None,
                        None,
                        None,
                        np,
                        spin,
                    )
                    .map_err(|e| DftError::BackendEval(format!("libxc LdaOutput: {e:?}")))?;
                    batch
                        .evaluate(&func, &input, dord, &mut o)
                        .map_err(|e| DftError::BackendEval(format!("libxc eval LDA: {e:?}")))?;
                    for ip in 0..np {
                        out.exc[ip] += fac * zk[ip] * rho[ip];
                        out.vrho[ip] += fac * vrho[ip];
                    }
                }
                libxc_rs::Family::Gga => {
                    let RhoBlock::Gga { rho, sigma } = rho else {
                        return Err(family_mismatch(Family::Gga, rho.family()));
                    };
                    let input = GgaInput::new(rho, sigma, np, spin)
                        .map_err(|e| DftError::BackendEval(format!("libxc GgaInput: {e:?}")))?;
                    let mut zk = vec![0.0; np];
                    let mut vrho = vec![0.0; np];
                    let mut vsigma = vec![0.0; np];
                    let want = order >= DerivOrder::Vxc;
                    let mut o = GgaOutput::new(
                        Some(&mut zk),
                        if want { Some(&mut vrho) } else { None },
                        if want { Some(&mut vsigma) } else { None },
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        np,
                        spin,
                    )
                    .map_err(|e| DftError::BackendEval(format!("libxc GgaOutput: {e:?}")))?;
                    batch
                        .evaluate(&func, &input, dord, &mut o)
                        .map_err(|e| DftError::BackendEval(format!("libxc eval GGA: {e:?}")))?;
                    for ip in 0..np {
                        out.exc[ip] += fac * zk[ip] * rho[ip];
                        out.vrho[ip] += fac * vrho[ip];
                        out.vsigma[ip] += fac * vsigma[ip];
                    }
                }
                libxc_rs::Family::Mgga => {
                    let RhoBlock::Mgga {
                        rho,
                        sigma,
                        lapl,
                        tau,
                    } = rho
                    else {
                        return Err(family_mismatch(Family::Mgga, rho.family()));
                    };
                    let input = MggaInput::new(rho, sigma, lapl, tau, np, spin)
                        .map_err(|e| DftError::BackendEval(format!("libxc MggaInput: {e:?}")))?;
                    let mut zk = vec![0.0; np];
                    let mut vrho = vec![0.0; np];
                    let mut vsigma = vec![0.0; np];
                    let want = order >= DerivOrder::Vxc;
                    let mut o = MggaOutput {
                        zk: Some(&mut zk),
                        vrho: if want { Some(&mut vrho) } else { None },
                        vsigma: if want { Some(&mut vsigma) } else { None },
                        ..Default::default()
                    };
                    o.validate(np, spin)
                        .map_err(|e| DftError::BackendEval(format!("libxc MggaOutput: {e:?}")))?;
                    batch
                        .evaluate(&func, &input, dord, &mut o)
                        .map_err(|e| DftError::BackendEval(format!("libxc eval MGGA: {e:?}")))?;
                    for ip in 0..np {
                        out.exc[ip] += fac * zk[ip] * rho[ip];
                        out.vrho[ip] += fac * vrho[ip];
                        out.vsigma[ip] += fac * vsigma[ip];
                    }
                }
            }
        }
        Ok(out)
    }

    fn family_mismatch(want: Family, got: Family) -> DftError {
        DftError::BackendEval(format!(
            "libxc: functional family {want:?} does not match supplied RhoBlock family {got:?}"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::libxc;
    use crate::parser::xcfun;

    /// Slater LDA exchange has a closed analytic form (unpolarized):
    ///   f(rho)  = -(3/4)(3/π)^(1/3) · rho^(4/3)        (energy density)
    ///   vrho    = -(3/π)^(1/3) · rho^(1/3)             (∂f/∂rho)
    /// This is an independent oracle for the xcfun SLATERX path — no live
    /// xcfun process required (the analytic value IS the reference).
    fn slater_ref(rho: f64) -> (f64, f64) {
        let c = (3.0_f64 / std::f64::consts::PI).cbrt();
        let f = -0.75 * c * rho.powf(4.0 / 3.0);
        let v = -c * rho.cbrt();
        (f, v)
    }

    #[test]
    fn xcfun_lda_slater_matches_analytic() {
        // 'slater,' -> xcfun SLATERX(0), factor 1.
        let spec = xcfun::parse_xc("slater,").unwrap();
        let rho_vals = [0.1_f64, 0.5, 1.0, 2.5];
        let rb = RhoBlock::Lda { rho: &rho_vals };
        let out = XcBackend::Xcfun.eval(&spec, &rb, DerivOrder::Vxc).unwrap();

        for (ip, &r) in rho_vals.iter().enumerate() {
            let (f, v) = slater_ref(r);
            assert!(
                (out.exc[ip] - f).abs() < 1e-10,
                "exc[{ip}] rho={r}: got {}, want {f}",
                out.exc[ip]
            );
            assert!(
                (out.vrho[ip] - v).abs() < 1e-10,
                "vrho[{ip}] rho={r}: got {}, want {v}",
                out.vrho[ip]
            );
        }
    }

    #[test]
    fn xcfun_gga_pbe_x_finite_and_consistent() {
        // PBE exchange (GGA): assert finite, negative energy density and a
        // sane sigma-derivative over a small block. (PBEX reduces to Slater at
        // sigma=0, which we cross-check.)
        let spec = xcfun::parse_xc("pbe,").unwrap(); // PBEX only, X part
        let rho_vals = [0.1_f64, 0.5, 1.0];
        let sigma_vals = [0.0_f64, 0.0, 0.0]; // sigma=0 -> reduces to Slater-like
        let rb = RhoBlock::Gga {
            rho: &rho_vals,
            sigma: &sigma_vals,
        };
        let out = XcBackend::Xcfun.eval(&spec, &rb, DerivOrder::Vxc).unwrap();

        for (ip, &r) in rho_vals.iter().enumerate() {
            assert!(out.exc[ip].is_finite(), "exc[{ip}] finite");
            assert!(out.exc[ip] < 0.0, "PBE-X energy density is negative");
            // At sigma=0 the PBE exchange enhancement factor is 1, so the
            // energy density equals Slater exchange exactly.
            let (f, _) = slater_ref(r);
            assert!(
                (out.exc[ip] - f).abs() < 1e-9,
                "PBE-X at sigma=0 should equal Slater: exc[{ip}]={}, slater={f}",
                out.exc[ip]
            );
            assert!(out.vrho[ip].is_finite());
            assert!(out.vsigma[ip].is_finite());
        }
    }

    #[test]
    fn family_mismatch_is_clean_error() {
        // A GGA functional fed an LDA block must error, not panic.
        let spec = xcfun::parse_xc("pbe,").unwrap();
        let rho_vals = [0.5_f64];
        let rb = RhoBlock::Lda { rho: &rho_vals };
        assert!(matches!(
            XcBackend::Xcfun.eval(&spec, &rb, DerivOrder::Vxc),
            Err(DftError::BackendEval(_))
        ));
    }

    #[test]
    fn require_libxc_default_build_returns_feature_not_enabled() {
        // In a default build (no --features libxc) this MUST be the
        // feature-not-enabled error, never a libxc_rs symbol reference.
        #[cfg(not(feature = "libxc"))]
        assert!(matches!(
            XcBackend::require_libxc(),
            Err(DftError::LibxcFeatureNotEnabled)
        ));
        #[cfg(feature = "libxc")]
        assert!(matches!(XcBackend::require_libxc(), Ok(XcBackend::Libxc)));
    }

    #[test]
    fn libxc_spec_routes_to_libxc_ids() {
        // The libxc parser produces libxc ids (B88=106). The backend's libxc
        // path (gated) consumes them; here we just assert the parser side
        // (the default build's contract: libxc spec ids != xcfun spec ids).
        let spec = libxc::parse_xc("blyp").unwrap();
        assert_eq!(spec.components(), &[(106, 1.0), (131, 1.0)]);
    }
}
