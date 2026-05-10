//! Basis-set type. Phase 1 declares the shape; Phase 2 (GTO-11) wires
//! the zero-copy re-export from `cintx_core::BasisSet`.
//!
//! Phase 1 cannot do the cintx wiring because pyscf-core MUST have zero
//! compute deps (FOUND-02). Phase 2's gto crate provides the bridge.

/// Basis-set placeholder. Phase 2 replaces this with a re-export of
/// `cintx_core::BasisSet` (GTO-11) — pyscf-rs does not maintain a
/// parallel basis structure.
#[derive(Debug, Default, Clone)]
pub struct BasisSet {
    /// Basis-set name (e.g., "cc-pvdz"). Phase 2 fills the rest.
    pub name: String,
}
