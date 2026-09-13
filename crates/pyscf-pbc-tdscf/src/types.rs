//! Shared TDA/TDHF driver types (`pbc/tdscf`).
//!
//! Every method module (`rhf`, `krhf`, `uhf`, `kuhf`, `rks`, `uks`, `krks`,
//! `kuks`) consumes these two structs and returns [`TdaResult`]. The response
//! matrix itself is method-built (HF vs KS kernels differ); the
//! diagonalization, root sorting and oscillator-strength contraction live in
//! [`crate::davidson`].

/// Tamm-Dancoff (or full TDHF) driver configuration.
///
/// Ports the common keyword surface of `pbc/tdscf/{rhf,krhf,uhf,kuhf}.py`:
/// `nroots`, `conv_tol`, `max_cycle`, `singlet` (RHF only) and `tria` (TDA vs
/// full TDHF). `kshift` selects the momentum-transfer block for k-point
/// drivers; gamma drivers fix it to zero.
#[derive(Debug, Clone)]
pub struct TdaConfig {
    /// Number of roots to return (sorted ascending within the k-shift block).
    pub nroots: usize,
    /// Davidson residual-norm convergence tolerance.
    pub conv_tol: f64,
    /// Maximum Davidson iterations.
    pub max_cycle: usize,
    /// Restricted drivers: `true` = singlet manifold, `false` = triplet.
    pub singlet: bool,
    /// `true` = Tamm-Dancoff approximation (Hermitian `A` only);
    /// `false` = full TDHF (`[A B; -B -A]`, non-symmetric).
    pub tda: bool,
    /// Momentum-transfer index for k-point drivers (gamma drivers: 0).
    pub kshift: usize,
}

impl Default for TdaConfig {
    fn default() -> Self {
        Self {
            nroots: 3,
            conv_tol: 1e-9,
            max_cycle: 50,
            singlet: true,
            tda: true,
            kshift: 0,
        }
    }
}

/// Converged excitation spectrum for one k-shift block.
///
/// `energies` are sorted ascending (Gate A compares sorted eigenvalues with an
/// explicit root count — never positionally across shifts). `oscillator` holds
/// the length-gauge oscillator strength per root when transition dipoles are
/// available, else zeros. `kshift` records which momentum-transfer block the
/// roots belong to (19-07: never globally sort across shifts).
#[derive(Debug, Clone)]
pub struct TdaResult {
    /// Excitation energies, ascending, length `nroots` (Hartree).
    pub energies: Vec<f64>,
    /// Oscillator strength per root (length-gauge), length `nroots`.
    pub oscillator: Vec<f64>,
    /// Momentum-transfer block index these roots belong to.
    pub kshift: usize,
    /// Whether the Davidson/dense solve met `conv_tol`.
    pub converged: bool,
}
