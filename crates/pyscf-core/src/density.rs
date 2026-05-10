//! Density matrix type. Phase 1 declares the shape; Phase 3 (SCF) wires
//! AO-basis density-matrix construction.

/// AO-basis density matrix. For RHF/RKS this is a single `nao × nao`
/// matrix; for UHF/UKS it's a pair (alpha, beta); for GHF it's a
/// 2*nao × 2*nao spinor density. Phase 3 implements.
#[derive(Debug, Default, Clone)]
pub struct Density {
    pub nao: usize,
    /// Row-major AO density matrix flattened. Phase 3 wires the actual
    /// data via the AlgebraClient buffer (Plan 04 introduces opaque
    /// `Tensor`).
    pub data: Vec<f64>,
}
