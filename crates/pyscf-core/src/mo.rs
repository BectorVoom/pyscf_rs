//! Molecular-orbital coefficient matrix. Phase 1 declares; Phase 3
//! (SCF-13 canonicalize_signs + eigenvector storage) wires.

/// MO coefficient matrix `C[ao_i, mo_j]`. Phase 3 fills the data via
/// AlgebraClient::eigh. Sign canonicalization (SCF-13) is applied on
/// construction.
#[derive(Debug, Default, Clone)]
pub struct MOCoefficients {
    pub nao: usize,
    pub nmo: usize,
    /// Column-major (Fortran order) coefficient matrix flattened.
    /// Column-major matches PySCF/LAPACK convention (Pitfall 8 — F-order
    /// layout preserved on output).
    pub data: Vec<f64>,
    /// MO energies, one per MO column.
    pub energies: Vec<f64>,
    /// Occupation numbers (0, 1, or 2 for restricted; 0 or 1 for
    /// unrestricted alpha/beta).
    pub occupations: Vec<f64>,
}
