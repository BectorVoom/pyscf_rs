//! DFT error type. `DftError` is the public-facing error returned by the XC
//! string parsers (`parser::libxc`, `parser::xcfun`) and the `XcBackend`
//! evaluation dispatch.
//!
//! Provenance: mirrors the `pyscf-core::error::PyscfRsError` `#[from]` variant
//! convention (BasisLoad/EcpLoad at pyscf-core/src/error.rs:30-42). Phase 4
//! (DFT-02 parser parity, DFT-03 bit-exact eval). License: Apache-2.0
//! (matching upstream PySCF — `pyscf/dft/libxc.py`, `pyscf/dft/xcfun.py`).

use thiserror::Error;

/// Errors raised by the DFT XC-string parsers and the XcBackend seam.
///
/// Trust boundary T-04-05a/b: the XC string is *untrusted user input* crossing
/// `dft.RKS(mol, xc=<arbitrary string>)`. The parser is a pure string→spec
/// mapping — it never evaluates arbitrary code, and every malformed input must
/// surface here as a clean `Err`, never a panic or unbounded recursion.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum DftError {
    /// An unknown functional token that resolves against neither the inline
    /// `XC_CODES`/`XC_ALIAS` tables nor a raw integer ID. Upstream raises
    /// `KeyError`/`NotImplementedError`; we return a clean `Err`.
    #[error("unknown XC functional token '{token}' (XC string: '{xc}')")]
    UnknownFunctional { xc: String, token: String },

    /// A token (factor, RSH triple, SR_HF/LR_HF omega, raw integer ID) that
    /// could not be parsed as a number. Upstream would raise `ValueError`.
    #[error("malformed XC token '{token}' in XC string '{xc}': {reason}")]
    MalformedToken {
        xc: String,
        token: String,
        reason: String,
    },

    /// An XC string with more than one comma (upstream `split(',')` unpacks
    /// into exactly two parts and raises `ValueError: too many values`).
    #[error("malformed XC string '{xc}': expected at most one ',' separating X and C parts")]
    TooManyCommas { xc: String },

    /// Two range-separated terms requested incompatible omega values
    /// (upstream `assign_omega`: "Different values of omega found for RSH
    /// functionals").
    #[error("conflicting RSH omega values in XC string '{xc}': {existing} vs {requested}")]
    ConflictingOmega {
        xc: String,
        existing: f64,
        requested: f64,
    },

    /// The bounded-depth guard on compound-functional expansion
    /// (`0.5*b3lyp`-style unit scaling) tripped — protects against adversarial
    /// or cyclic alias chains (threat T-04-05b, DoS via unbounded recursion).
    #[error(
        "XC compound-expansion depth limit ({limit}) exceeded for '{xc}' — possible cyclic alias"
    )]
    ExpansionDepthExceeded { xc: String, limit: usize },

    /// The `libxc` backend was requested but the crate was not compiled with
    /// `--features libxc`. The default build never references `libxc_rs`
    /// symbols (Pitfall 5); this variant is what the
    /// `#[cfg(not(feature = "libxc"))]` arm returns.
    #[error(
        "libxc backend not enabled — rebuild with `--features libxc` \
         (default builds use the xcfun_rs backend)"
    )]
    LibxcFeatureNotEnabled,

    /// A backend (xcfun_rs or, gated, libxc_rs) rejected an evaluation request
    /// — unknown functional name for the backend, buffer-size mismatch, etc.
    #[error("XC backend evaluation error: {0}")]
    BackendEval(String),
}

/// Bridge `DftError` into the workspace-wide `PyscfRsError` so the DFT grid
/// loop + KS hooks can propagate XC failures through the `?` operator into the
/// SCF kernel's `Result<_, PyscfRsError>` return type (DFT-01/DFT-08).
///
/// `pyscf-core` cannot carry a `#[from] DftError` variant (that would force a
/// `pyscf-core → pyscf-dft` dependency cycle — pyscf-dft already depends on
/// pyscf-core), so this `From` impl lives HERE (the orphan rule permits it:
/// `DftError` is local) and maps onto the existing
/// `CoreError::BasisParse` string-carrying variant, preserving the full
/// `DftError` Display message. Mirrors the cross-crate error-bridging
/// convention the pyscf-core error doc-comment anticipates ("Phases 2-7 add
/// method-specific variants via `#[from]` conversions").
impl From<DftError> for pyscf_core::PyscfRsError {
    fn from(e: DftError) -> Self {
        pyscf_core::PyscfRsError::Core(pyscf_core::CoreError::BasisParse(format!(
            "DFT/XC error: {e}"
        )))
    }
}
