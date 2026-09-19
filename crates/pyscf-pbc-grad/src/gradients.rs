//! The common periodic-gradient seam.
//!
//! Bodies land in the method-specific Phase-18 plans.  Keeping the complete
//! surface here prevents each driver from inventing a subtly different scanner
//! or optimizer contract while still making an unavailable calculation a loud,
//! typed refusal.

use crate::error::PbcGradError;
use pyscf_algebra::CTensor;
use pyscf_core::PyscfRsError;
use pyscf_pbc_gto::Cell;
use pyscf_pbc_scf::types::{KDms, KMats};

/// `(natm, 3)` Cartesian gradient in Ha/Bohr.
pub type Gradient = Vec<[f64; 3]>;
/// Bra derivative of k-resolved AO matrices, in `(Cartesian, k)` order.
/// Each matrix is column-major, matching periodic integrals. Atom-resolved
/// hcore_generator calls select an atom separately; overlap has no atom axis.
pub type GradMatrices = [KMats; 3];
/// Energy scanner used by periodic finite-difference gates.
pub type EnergyScanner = Box<dyn Fn(&Cell) -> Result<f64, PyscfRsError> + Send + Sync>;

/// Shared API mirrored from `pyscf.pbc.grad.krhf.GradientsBase`.
pub trait Gradients {
    /// The cell held by the underlying mean-field calculation.
    fn cell(&self) -> &Cell;

    /// Cartesian k-points; gamma is the default for single-point methods.
    fn kpts(&self) -> &[[f64; 3]] {
        &[[0.0; 3]]
    }

    fn grad_elec(&self) -> Result<Gradient, PyscfRsError> {
        Err(PbcGradError::NotYetImplemented {
            phase: 18,
            what: "grad_elec",
        }
        .into())
    }
    fn get_hcore(&self) -> Result<GradMatrices, PyscfRsError> {
        Err(PbcGradError::NotYetImplemented {
            phase: 18,
            what: "get_hcore",
        }
        .into())
    }
    fn hcore_generator(&self, atom: usize) -> Result<[KMats; 3], PyscfRsError> {
        let _ = atom;
        Err(PbcGradError::NotYetImplemented {
            phase: 18,
            what: "hcore_generator",
        }
        .into())
    }
    /// `get_ovlp` — `krhf.py:114-115`: `-pbc_intor('int1e_ipovlp')`.
    ///
    /// The minus lives HERE, not in `pbc_intor` (18-03 Task 2): the integral
    /// carries the ket derivative and the gradient needs the bra one.
    fn get_ovlp(&self) -> Result<GradMatrices, PyscfRsError> {
        let integral =
            pyscf_pbc_gto::pbc_intor(self.cell(), "int1e_ipovlp", self.kpts(), Default::default())?;
        let size = integral.ni * integral.nj;
        Ok(std::array::from_fn(|c| {
            integral
                .kmats
                .iter()
                .map(|matrix| CTensor {
                    re: matrix.re[c * size..(c + 1) * size]
                        .iter()
                        .map(|v| -v)
                        .collect(),
                    im: matrix.im[c * size..(c + 1) * size]
                        .iter()
                        .map(|v| -v)
                        .collect(),
                })
                .collect()
        }))
    }
    fn get_jk(&self, _dm: &KDms) -> Result<(GradMatrices, GradMatrices), PyscfRsError> {
        Err(PbcGradError::NotYetImplemented {
            phase: 18,
            what: "get_jk",
        }
        .into())
    }
    fn get_j(&self, dm: &KDms) -> Result<GradMatrices, PyscfRsError> {
        self.get_jk(dm).map(|(j, _)| j)
    }
    fn get_k(&self, dm: &KDms) -> Result<GradMatrices, PyscfRsError> {
        self.get_jk(dm).map(|(_, k)| k)
    }
    fn grad_nuc(&self) -> Result<Gradient, PyscfRsError> {
        pyscf_pbc_gto::ewald_nuc_grad(self.cell(), None, None)
    }
    fn make_rdm1e(&self) -> Result<KDms, PyscfRsError> {
        Err(PbcGradError::NotYetImplemented {
            phase: 18,
            what: "make_rdm1e",
        }
        .into())
    }
    fn extra_force(&self, atom: usize, envs: &[CTensor]) -> Result<[f64; 3], PyscfRsError> {
        let _ = (atom, envs);
        Ok([0.0; 3])
    }
    fn as_scanner(&self) -> Result<EnergyScanner, PyscfRsError> {
        Err(PbcGradError::NotYetImplemented {
            phase: 18,
            what: "as_scanner",
        }
        .into())
    }
    /// `optimizer(solver='ase')` — `krhf.py:290-298`.
    ///
    /// Upstream accepts ONLY `'ase'` here; `'geometric'` (and anything else)
    /// raises `RuntimeError('Optimization solver … not supported')`. That
    /// refusal is mirrored exactly — it is upstream's behaviour (18-CONTEXT
    /// §1.7), not a port gap: the geomeTRIC path is reachable only through
    /// `pbc.geomopt.optimize`, which 18-14 ports. The `'ase'` branch itself
    /// belongs to Phase 20's `tools/pyscf_ase` and is a named refusal until
    /// then.
    fn optimizer(&self, solver: &str) -> Result<(), PyscfRsError> {
        if solver != "ase" {
            return Err(PbcGradError::UnsupportedOptimizer {
                solver: solver.into(),
            }
            .into());
        }
        Err(PbcGradError::NotYetImplemented {
            phase: 18,
            what: "ASE optimizer",
        }
        .into())
    }
}
