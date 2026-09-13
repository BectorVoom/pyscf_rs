//! Periodic response-function seam (`pbc/scf/_response_functions.py`, 47 l).
//!
//! Ports the `_get_jk`/`_get_j`/`_get_k` k-shift routing and the
//! `gen_response` presence contract (19-CONTEXT §1.6):
//!
//! * `pbc/dft/rks.py:268` sets `gen_response = NotImplemented` on the PBC
//!   `KohnShamDFT` **base**; only the concrete `RKS` rebinds it (`:411`, from
//!   the module-level definition at `:146`). A binding that supplies
//!   `gen_response` universally would diverge from upstream silently, so the
//!   base type here carries `HAS_GEN_RESPONSE = false` and only the concrete
//!   [`RksGenResponse`] carries `true`.
//! * `_get_jk` at `kshift == 0` routes to the direct per-k build; at nonzero
//!   shift it requires a GDF/RSDF-fitting backend (`_get_jk_kshift`) and
//!   refuses `omega != 0` — both refusals are [`ResponseError`] variants,
//!   never silent fallbacks.

use pyscf_core::PyscfRsError;

/// Which J/K backend backs a response build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseJkBackend {
    /// Direct per-k J/K (`mf.get_jk` at `kshift == 0`).
    Direct,
    /// Density-fitted J/K supporting the k-shift route (`GDF`/`RSDF`).
    Fitted,
}

/// Resolve the `_get_jk` dispatch of `_response_functions.py:30-44`.
///
/// * `kshift == 0` → [`ResponseJkBackend::Direct`] (either backend serves it).
/// * `kshift != 0`, `omega` unset/zero, fitted backend → `Fitted` (the
///   `_get_jk_kshift` path).
/// * `kshift != 0` with `omega != 0` → refused (upstream raises
///   `NotImplementedError`).
/// * `kshift != 0` on a non-fitting backend → refused (upstream logs an error
///   and raises `NotImplementedError`: "Non-zero kshift is only supported by
///   GDF/RSDF").
pub fn resolve_jk_route(
    kshift: usize,
    omega: Option<f64>,
    backend: ResponseJkBackend,
) -> Result<ResponseJkBackend, ResponseError> {
    if kshift == 0 {
        return Ok(ResponseJkBackend::Direct);
    }
    if let Some(w) = omega {
        if w != 0.0 {
            return Err(ResponseError::KshiftWithRangeSeparation);
        }
    }
    match backend {
        ResponseJkBackend::Fitted => Ok(ResponseJkBackend::Fitted),
        ResponseJkBackend::Direct => Err(ResponseError::KshiftNeedsFittedJk),
    }
}

/// Response-seam errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResponseError {
    /// Nonzero k-shift with range separation (`omega != 0`).
    KshiftWithRangeSeparation,
    /// Nonzero k-shift on a non-density-fitted J/K backend.
    KshiftNeedsFittedJk,
}

impl std::fmt::Display for ResponseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResponseError::KshiftWithRangeSeparation => {
                write!(f, "non-zero kshift with range separation (omega != 0) is not implemented")
            }
            ResponseError::KshiftNeedsFittedJk => {
                write!(f, "non-zero kshift is only supported by GDF/RSDF")
            }
        }
    }
}

impl From<ResponseError> for PyscfRsError {
    fn from(e: ResponseError) -> Self {
        PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(e.to_string()))
    }
}

/// PBC Kohn-Sham DFT base: `gen_response` is **absent** here
/// (`pbc/dft/rks.py:268` sets it `NotImplemented`). The constant (not a
/// method) is what makes the absence testable: a test asserts
/// `PbcKohnShamBase::HAS_GEN_RESPONSE == false`.
#[derive(Debug, Clone, Copy)]
pub struct PbcKohnShamBase;

impl PbcKohnShamBase {
    /// Upstream's `:268` — the base class does not provide `gen_response`.
    pub const HAS_GEN_RESPONSE: bool = false;
}

/// Concrete PBC RKS response binding (`pbc/dft/rks.py:411` rebinds the
/// module-level `:146` definition). Only this type provides the response.
#[derive(Debug, Clone, Copy)]
pub struct RksGenResponse;

impl RksGenResponse {
    /// The concrete class rebinds `gen_response`.
    pub const HAS_GEN_RESPONSE: bool = true;
}
