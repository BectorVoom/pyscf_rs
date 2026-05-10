//! Basis-set type. Phase 1 declares the shape; Phase 2 (GTO-11) wires
//! the zero-copy re-export from `cintx_core::BasisSet`.
//!
//! Phase 1 cannot do the cintx wiring because pyscf-core MUST have zero
//! compute deps (FOUND-02). Phase 2's gto crate provides the bridge.
//!
//! Phase 2 plan 02-02 status: this file still ships the Phase-1
//! placeholder `BasisSet` struct + a `raw_atm_layout` slot-constants
//! module. The constants mirror `cintx_compat::raw` (ATM_SLOTS, BAS_SLOTS,
//! PTR_ENV_START, ...) so `mole.rs` can implement `atom_charges()` etc.
//! without pulling cintx-compat into pyscf-core's dep graph (FOUND-02 zero
//! compute deps). Plan 02-04 deletes this local mirror and replaces with
//! `pub use cintx_compat::raw::*;` once pyscf-core gets the dep.

/// Basis-set placeholder. Phase 2 plan 02-04 replaces this with a re-export
/// of `cintx_core::BasisSet` (GTO-11) — pyscf-rs does not maintain a
/// parallel basis structure.
#[derive(Debug, Default, Clone)]
pub struct BasisSet {
    /// Basis-set name (e.g., "cc-pvdz"). Phase 2 plan 02-04 fills the rest.
    pub name: String,
}

/// Local mirror of `cintx_compat::raw` slot constants for the Phase 2
/// plan 02-02 Mole implementation. These constants describe the libcint
/// `_atm` / `_bas` / `_env` flat-array layout per D-03.
///
/// **TEMPORARY**: plan 02-04 deletes this module and replaces references
/// with `cintx_compat::raw::*` once pyscf-core takes a dep on cintx-compat.
/// The constants here MUST match `cintx/crates/cintx-compat/src/raw.rs:15-41`
/// exactly. Verified against that file as of 2026-05-10.
pub mod raw_atm_layout {
    /// _atm slot count per atom row (libcint `cint_bas.h`).
    pub const ATM_SLOTS: usize = 6;
    /// _atm slot index — atomic number / nuclear charge.
    pub const CHARGE_OF: usize = 0;
    /// _atm slot index — pointer into _env where the atom's `[x,y,z]` lives.
    pub const PTR_COORD: usize = 1;
    /// _atm slot index — nuclear model (Point=1 / Gaussian=2 / FracCharge=3).
    pub const NUC_MOD_OF: usize = 2;
    /// _atm slot index — pointer into _env for the Gaussian-finite zeta value.
    pub const PTR_ZETA: usize = 3;

    /// _bas slot count per shell row.
    pub const BAS_SLOTS: usize = 8;
    /// _bas slot index — owning atom index.
    pub const ATOM_OF: usize = 0;
    /// _bas slot index — angular momentum l.
    pub const ANG_OF: usize = 1;
    /// _bas slot index — number of primitive Gaussians.
    pub const NPRIM_OF: usize = 2;
    /// _bas slot index — number of contracted functions.
    pub const NCTR_OF: usize = 3;
    /// _bas slot index — kappa (relativistic). 0 = non-relativistic / spheric.
    pub const KAPPA_OF: usize = 4;
    /// _bas slot index — pointer into _env for the primitive exponents.
    pub const PTR_EXP: usize = 5;
    /// _bas slot index — pointer into _env for the contraction coefficients.
    pub const PTR_COEFF: usize = 6;

    /// First _env index available for user data; libcint reserves [0, 20).
    pub const PTR_ENV_START: usize = 20;
}
