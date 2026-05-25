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

impl Density {
    /// Wrap a flat `nao × nao` value buffer as a `Density`.
    ///
    /// Added by gap-closure plan 02-10 so the cintx-backed `EcpEngine`
    /// (`pyscf_gto::ecp_engine_cintx::CintxEcpEngine`) can hand the
    /// stitched `int1e_ecp` matrix back through the `EcpEngine` trait
    /// without re-deriving `nao` at the call site. `nao` is the AO-axis
    /// dimension; `data.len()` must equal `nao * nao` for the square
    /// 1e-ECP matrix the dispatcher expects.
    pub fn from_flat(nao: usize, data: Vec<f64>) -> Self {
        Density { nao, data }
    }
}
