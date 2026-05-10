//! Method-dispatch traits. Phase 1 declares signatures; Phase 3 (Scf),
//! Phase 4 (KohnSham), Phase 5 (PostScf for MP2), Phase 6 (PostScf for
//! CCSD), Phase 7 (Gradient) implement.

use crate::density::Density;
use crate::energy::Energy;
use crate::error::PyscfRsError;
use crate::mole::Mole;

/// Top-level method trait. Every pyscf-rs method (HF, DFT, MP2, CCSD,
/// gradients) implements `Method::kernel` returning the total energy.
pub trait Method {
    /// Run the method on its bound `Mole`. Returns total energy.
    fn kernel(&mut self) -> Result<Energy, PyscfRsError>;

    /// Borrow the molecule the method is bound to.
    fn mol(&self) -> &Mole;
}

/// Self-consistent-field methods (RHF/UHF/GHF/RKS/UKS). Phase 3 wires
/// the RHF/UHF/GHF impls; Phase 4 wires RKS/UKS via the KohnSham
/// sub-trait.
pub trait Scf: Method {
    /// Spin-restricted (Density) or spin-unrestricted (a pair) — concrete
    /// impl picks via this associated type.
    type DensityT;

    /// Last-converged density. Phase 3 wires.
    fn density(&self) -> Option<&Self::DensityT>;
}

/// Kohn-Sham specialisation. Phase 4 (DFT-08) wires DFT-specific hooks
/// (`get_veff`, `define_xc_`).
pub trait KohnSham: Scf {
    /// XC functional string (e.g., `"b3lyp"`, `"pbe,pbe"`). Phase 4
    /// (DFT-02) parses this against the upstream alias table.
    fn xc(&self) -> &str;
}

/// Post-SCF methods (MP2, CCSD). Bind a converged SCF reference and
/// produce a correlation energy.
pub trait PostScf: Method {
    /// The Scf reference this post-SCF method correlates against.
    type Reference: Scf;

    /// Borrow the SCF reference.
    fn reference(&self) -> &Self::Reference;

    /// Correlation energy contribution (so `total = reference + correlation`).
    fn e_correlation(&self) -> Result<Energy, PyscfRsError>;
}

/// Analytical gradient. Phase 7 (GRAD-01..07) wires per-method impls.
pub trait Gradient {
    /// Gradient `dE/dR` in Hartree/Bohr, indexed by atom (rows) × xyz
    /// (cols).
    fn gradient(&self) -> Result<Vec<[f64; 3]>, PyscfRsError>;
}

/// Integral engine — the cintx wrapping seam (Phase 2 GTO-06 wires).
/// Phase 1 declares the trait so other crates can program against it
/// without depending on cintx directly.
pub trait IntegralEngine {
    /// Compute the named integral (e.g., `"int1e_ovlp_sph"`,
    /// `"int2e"`). Returns the array layout upstream PySCF returns
    /// (F-order where applicable — Pitfall 8).
    fn intor(
        &self,
        mol: &Mole,
        name: &str,
    ) -> Result<Density, PyscfRsError>;
}
