//! GTO-11: Zero-copy re-export of `cintx_core::BasisSet`.
//!
//! `cintx_core::BasisSet` is internally Arc-backed:
//!
//! ```text
//! pub struct BasisSet {
//!     atoms: Arc<[Atom]>,
//!     shells: Arc<[Arc<Shell>]>,
//!     meta: BasisMeta,
//! }
//! ```
//!
//! so re-export carries the Arc — clones bump the refcount, no data copy.
//!
//! Per D-03, slot constants live in `cintx_compat::raw` and are re-exported
//! through `raw_layout` so callers don't need to add `cintx-compat` to every
//! Cargo.toml.
//!
//! Plan 02-04 (this commit) deletes the temporary `raw_atm_layout` mirror
//! that plan 02-02 ships in this file. Downstream code (`mole.rs`,
//! `pyscf-gto::make_env`) now imports from `pyscf_core::raw_layout::*` which
//! is `cintx_compat::raw::*` verbatim — drift between cintx and pyscf-rs is
//! impossible by construction (T-02-04-01 mitigation).

pub use cintx_core::{Atom, BasisMeta, BasisSet, NuclearModel as CintxNuclearModel, Shell};

/// Re-export of `cintx_compat::raw` slot constants (D-03 mandates reuse).
///
/// Use as:
///
/// ```ignore
/// use pyscf_core::raw_layout::{ATM_SLOTS, CHARGE_OF, PTR_ENV_START, BAS_SLOTS};
/// ```
///
/// The constants describe the libcint `_atm` / `_bas` / `_env` flat-array
/// layout. There is intentionally NO local definition — every value is a
/// `pub use` of the corresponding cintx_compat constant so any libcint
/// layout change in cintx-compat is picked up automatically.
pub mod raw_layout {
    pub use cintx_compat::raw::{
        ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, FRAC_CHARGE_NUC, GAUSSIAN_NUC,
        KAPPA_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC, PTR_COEFF, PTR_COORD,
        PTR_ENV_START, PTR_EXP, PTR_FRAC_CHARGE, PTR_ZETA,
    };
}
