//! Coupled-cluster amplitude tensors. Phase 1 declares; Phase 6 (CCSD)
//! wires t1/t2 storage with the tensor-arena pattern (CCSD-11).

/// CCSD amplitude container. Phase 6 implements with the tensor-arena
/// pattern from `pyscf-runtime::WorkspacePool` so `Wabef` (and friends)
/// don't allocate-and-drop per iteration (CCSD-11, Pitfall 20).
#[derive(Debug, Default, Clone)]
pub struct Amplitudes {
    pub nocc: usize,
    pub nvir: usize,
    /// t1 amplitudes `[nocc, nvir]` flattened. Phase 6 wires.
    pub t1: Vec<f64>,
    /// t2 amplitudes `[nocc, nocc, nvir, nvir]` flattened. Phase 6
    /// wires (likely as opaque Tensor for spillability).
    pub t2: Vec<f64>,
}
