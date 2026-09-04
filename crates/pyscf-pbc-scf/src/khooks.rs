//! The periodic overrideable-hook surface — D-PBC-13, plan 11-09.
//!
//! Same shape as the molecular `pyscf_scf::OverrideHooks`: the driver
//! ([`crate::kscf::kernel`]) is generic over this trait and calls nothing else,
//! so `KRHF`, `KUHF`, `KROHF`, `KGHF` (and, in Phase 12, `KRKS`/`KUKS`) are
//! implementations rather than forks of the loop — and so the Phase-3 PyO3
//! bridge can put a Python subclass behind exactly these methods.
//!
//! Unlike the molecular trait, the cell, the k-points and the density-fitting
//! object are carried by the IMPLEMENTOR rather than threaded through every
//! call: a periodic hook needs all three and there is no case where the driver
//! wants to hand a method a different cell than the one it was built with.

use pyscf_algebra::CTensor;
use pyscf_core::PyscfRsError;
use pyscf_pbc_gto::Cell;

use crate::types::{KDms, KInitGuess, KMats};

/// The eleven hooks the periodic SCF driver dispatches through.
pub trait KOverrideHooks {
    /// The cell.
    fn cell(&self) -> &Cell;
    /// The sampling k-points.
    fn kpts(&self) -> &[[f64; 3]];
    /// Density channels: 1 for RHF/GHF, 2 for UHF/ROHF.
    fn nset(&self) -> usize;

    /// Fock channels handed to [`KOverrideHooks::eig`], and therefore the
    /// number of `(block, k)` MO sets the driver carries.
    ///
    /// This EQUALS [`KOverrideHooks::nset`] for every method except ROHF, where
    /// two density channels collapse into ONE Roothaan effective Fock
    /// (`krohf.py:85-120`) and so into one set of orbitals.
    fn nfock(&self) -> usize {
        self.nset()
    }
    /// AO dimension of one block. `nao` for everything except GHF, where it is
    /// `2 * nao`.
    fn nao(&self) -> usize {
        self.cell().mol.nao_nr
    }

    /// `S^k`, ROW-MAJOR.
    ///
    /// # Errors
    /// Implementation-specific.
    fn get_ovlp(&self) -> Result<KMats, PyscfRsError>;

    /// `H^k = T^k + V^k`, ROW-MAJOR.
    ///
    /// # Errors
    /// Implementation-specific.
    fn get_hcore(&self) -> Result<KMats, PyscfRsError>;

    /// The initial density matrices.
    ///
    /// # Errors
    /// Implementation-specific.
    fn get_init_guess(&self, mode: &KInitGuess, s1e: &KMats) -> Result<KDms, PyscfRsError>;

    /// `V_HF[s][k]` for the given densities.
    ///
    /// # Errors
    /// Implementation-specific.
    fn get_veff(&self, dms: &KDms) -> Result<KDms, PyscfRsError>;

    /// The Fock matrices to diagonalise — `khf.py:133-137` (`h1e + vhf`) or,
    /// for ROHF, the Roothaan effective Fock built from the two spin Focks.
    ///
    /// Returns [`KOverrideHooks::nfock`] channels.
    ///
    /// # Errors
    /// Implementation-specific.
    fn get_fock(&self, h1e: &KMats, vhf: &KDms, dms: &KDms) -> Result<KDms, PyscfRsError> {
        let _ = dms;
        Ok(crate::kscf::bare_fock(h1e, vhf))
    }

    /// The density matrices the DIIS error vector `FDS - SDF` is built against.
    ///
    /// The default is the density channels themselves. ROHF overrides it with
    /// the spin-summed density, because its DIIS iterate is the single Roothaan
    /// Fock (`krohf.py:74-79`).
    fn diis_dms(&self, dms: &KDms) -> KDms {
        dms.clone()
    }

    /// Solve `F^k C^k = S^k C^k eps^k` for every `(set, k)`.
    ///
    /// Returns `(mo_energy, mo_coeff)` indexed `set * nkpts + k`; `mo_coeff` is
    /// COLUMN-MAJOR.
    ///
    /// # Errors
    /// Implementation-specific.
    fn eig(&self, fock: &KDms, s1e: &KMats) -> Result<(Vec<Vec<f64>>, Vec<CTensor>), PyscfRsError>;

    /// Assign occupations with ONE Fermi level per spin channel across the
    /// whole Brillouin zone. Returns `(mo_occ, fermi_per_channel)`.
    ///
    /// # Errors
    /// Implementation-specific.
    fn get_occ(&self, mo_energy: &[Vec<f64>]) -> Result<(Vec<Vec<f64>>, Vec<f64>), PyscfRsError>;

    /// Build the density matrices from occupied orbitals.
    ///
    /// # Errors
    /// Implementation-specific.
    fn make_rdm1(&self, mo_coeff: &[CTensor], mo_occ: &[Vec<f64>]) -> Result<KDms, PyscfRsError>;

    /// `(e_elec, e_coul)`.
    ///
    /// # Errors
    /// Implementation-specific.
    fn energy_elec(&self, dms: &KDms, h1e: &KMats, vhf: &KDms) -> Result<(f64, f64), PyscfRsError>;

    /// The Ewald nuclear repulsion of the cell.
    ///
    /// # Errors
    /// Propagates `cell.ewald()`.
    fn energy_nuc(&self) -> Result<f64, PyscfRsError> {
        self.cell().energy_nuc()
    }

    /// The SCF convergence gradient — `khf.py:227-236`.
    ///
    /// The default is the occupied-virtual block of the MO-basis Fock matrix,
    /// concatenated over `(set, k)`. Smearing overrides it with the strict
    /// lower triangle of the FULL MO-basis Fock matrix
    /// (`pbc/scf/smearing.py:25-31`), because with fractional occupations the
    /// occupied-virtual split no longer separates the stationary conditions.
    fn get_grad(
        &self,
        mo_coeff: &[CTensor],
        mo_occ: &[Vec<f64>],
        h1e: &KMats,
        vhf: &KDms,
    ) -> Vec<f64> {
        let nao = self.nao();
        let nkpts = self.kpts().len();
        let fock = crate::kscf::bare_fock(h1e, vhf);
        let mut g = Vec::new();
        for (s, set) in fock.iter().enumerate() {
            for (k, f) in set.iter().enumerate() {
                let i = s * nkpts + k;
                g.extend_from_slice(&crate::kocc::get_grad(&mo_coeff[i], &mo_occ[i], f, nao));
            }
        }
        g
    }

    /// Optional post-`get_occ` hook used by smearing to report the entropy
    /// term `-sigma * S`. `None` when smearing is off.
    fn free_energy(&self) -> Option<f64> {
        None
    }
}
