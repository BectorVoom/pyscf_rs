//! `DiisStorable` trait — generic over the iterate type.
//!
//! Source: D-09 — single trait, multiple impls. For SCF the iterate is a Fock
//! matrix (`pyscf-scf::FockSubspace` impls this in plan 03-04). For Phase 6
//! CCSD the iterate is an `(T1, T2)` amplitude tuple
//! (`pyscf-ccsd::AmpsSubspace` will impl this later).

/// Marker for any object that the CDIIS ring buffer can store + linearly
/// combine. Implementors are expected to route `dot` through
/// `pyscf_algebra::oracle_dot` (Pitfall 9 — bit-identical cross-platform
/// reductions).
pub trait DiisStorable {
    /// Flat read-only view of the iterate's storage.
    fn as_flat(&self) -> &[f64];
    /// In-place replacement of the iterate's storage from a flat slice.
    /// Caller guarantees `slice.len() == self.len()`.
    // Intentionally `&mut self` (in-place loader reusing the iterate's
    // allocation), so the `from_*`-takes-no-self convention does not apply.
    #[allow(clippy::wrong_self_convention)]
    fn from_flat(&mut self, slice: &[f64]);
    /// Inner product with another iterate. MUST be implemented via
    /// `pyscf_algebra::oracle_dot` for Pitfall 9 mitigation.
    fn dot(&self, other: &Self) -> f64;
    /// Number of stored `f64` elements.
    fn len(&self) -> usize;
    /// Convenience: `true` when `len() == 0`.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
