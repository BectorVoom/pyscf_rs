//! RKS — Restricted Kohn-Sham. Source: `pyscf/dft/rks.py` (the RKS class +
//! the KS SCF driver layered on `scf/hf.py`). DFT-01.
//!
//! `RKS` mirrors `pyscf_scf::RHF` (rhf.rs:23-178) PLUS the DFT attributes
//! (`xc`, `grids`, `nlc`, `nlcgrids`, `_numint`), and `RKS::kernel` reuses
//! the Phase 3 generic `kernel<H>` (rhf.rs:138-148) verbatim — passing a KS
//! hooks impl whose effective potential is `J + Vxc − hyb·K`.
//!
//! D-08: the active scalar precision is surfaced read-only by delegating to
//! `_numint.dtype()` (so 04-09 can expose it through a Python `#[getter]`).
//! There is NO precision setter (deferred per D-08).
//!
//! ### JK availability note (Phase 2 rollup gap)
//! The KS `get_veff` needs `(J, K)`. The direct `int2e_sph` arity-4 path is
//! `NotYetImplemented{phase:2}` (Phase 2 verification-rollup gap, mirrored in
//! `pyscf-scf::fock::default_get_jk`), and the DF `int3c2e_sph` arity-3 path
//! is likewise pending the cintx-ops base symbol (`pyscf-df` notes this). So
//! the bit-exact RKS *energy* oracle (`rks_uks_bitexact`) is the CI-only
//! `--features python` arm — exactly the convention 04-04/04-05 established
//! for the grid + parser oracles. `RKS::kernel` itself is the real driver:
//! it reuses `pyscf_scf::kernel<H>` and routes the KS get_veff; when working
//! 2-electron integrals land it converges with no further change.

use pyscf_core::{MOCoefficients, Mole, PyscfRsError};
use pyscf_grids::Grids;
use pyscf_runtime::DType;
use pyscf_scf::ScfResult;

use crate::numint::NumInt;

/// Restricted Kohn-Sham object. Extends the RHF 30-attribute floor (SCF-14)
/// with the DFT-specific attributes (`xc`, `grids`, `nlc`, `nlcgrids`,
/// `_numint`).
pub struct RKS {
    // ── Inherited SCF attributes (RHF floor, rhf.rs:23-56) ──────────────
    pub mol: Mole,
    pub mo_coeff: Option<MOCoefficients>,
    pub mo_energy: Option<Vec<f64>>,
    pub mo_occ: Option<Vec<f64>>,
    pub e_tot: f64,
    pub e_elec: f64,
    pub converged: bool,
    pub cycles: u32,
    pub verbose: u8,
    pub chkfile: Option<std::path::PathBuf>,
    pub max_memory: f64,
    pub direct_scf: bool,
    pub direct_scf_tol: f64,
    pub init_guess: String,
    pub level_shift: f64,
    pub damp: f64,
    pub diis: bool,
    pub diis_space: u32,
    pub diis_start_cycle: u32,
    pub diis_damp: f64,
    pub diis_file: Option<std::path::PathBuf>,
    pub max_cycle: u32,
    pub conv_tol: f64,
    pub conv_tol_grad: Option<f64>,
    pub with_df: Option<Box<dyn std::any::Any + Send + Sync>>,
    pub disp: Option<String>,
    pub do_disp: bool,
    pub nelec: Option<(u32, u32)>,
    pub scf_summary: std::collections::HashMap<String, f64>,

    // ── DFT-specific attributes (rks.py) ───────────────────────────────
    /// The XC functional string (e.g. `"svwn"`, `"pbe"`, `"b3lyp"`).
    pub xc: String,
    /// The DFT integration grid (Becke; pyscf-grids).
    pub grids: Grids,
    /// The non-local-correlation functional string (VV10; DFT-06 — 04-07
    /// fills the eval, this is the attribute floor slot).
    pub nlc: String,
    /// The (coarser) NLC integration grid (VV10 `nlcgrids`; 04-07).
    pub nlcgrids: Grids,
    /// The numerical-integration driver (the grid loop owner). Carries the
    /// active D-08 precision (`_numint.dtype()`).
    pub _numint: NumInt,
}

impl std::fmt::Debug for RKS {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RKS")
            .field("mol", &self.mol)
            .field("xc", &self.xc)
            .field("e_tot", &self.e_tot)
            .field("converged", &self.converged)
            .field("cycles", &self.cycles)
            .field("max_memory", &self.max_memory)
            .field("init_guess", &self.init_guess)
            .field("max_cycle", &self.max_cycle)
            .field("conv_tol", &self.conv_tol)
            .field("nlc", &self.nlc)
            .field("dtype", &self._numint.dtype().name())
            // Opaque slots elided (grids/_numint/with_df): manual Debug, like RHF.
            .field("grids", &"<Grids>")
            .field("nlcgrids", &"<Grids>")
            .field("_numint", &"<NumInt>")
            .field("with_df", &self.with_df.as_ref().map(|_| "<dyn Any>"))
            .finish()
    }
}

impl RKS {
    /// Build an RKS object for `mol` with functional `xc` (mirrors
    /// `dft.RKS(mol, xc=...)`). Defaults match the RHF floor + the rks.py
    /// CLASS defaults (level-3 Becke grid).
    pub fn new(mol: Mole, xc: &str) -> Self {
        let grids = Grids::new();
        let nlcgrids = Grids::new();
        Self {
            mol,
            mo_coeff: None,
            mo_energy: None,
            mo_occ: None,
            e_tot: 0.0,
            e_elec: 0.0,
            converged: false,
            cycles: 0,
            verbose: 3,
            chkfile: None,
            max_memory: 4000.0,
            direct_scf: true,
            direct_scf_tol: 1e-13,
            init_guess: "minao".to_string(),
            level_shift: 0.0,
            damp: 0.0,
            diis: true,
            diis_space: 8,
            diis_start_cycle: 1,
            diis_damp: 0.0,
            diis_file: None,
            max_cycle: 50,
            conv_tol: 1e-9,
            conv_tol_grad: None,
            with_df: None,
            disp: None,
            do_disp: false,
            nelec: None,
            scf_summary: Default::default(),
            xc: xc.to_string(),
            grids,
            nlc: String::new(),
            nlcgrids,
            _numint: NumInt::new(),
        }
    }

    /// The active scalar precision (D-08), delegating to `_numint.dtype()`.
    /// Read-only — there is intentionally NO `set_precision` (deferred per
    /// D-08). 04-09 exposes this through a Python `#[getter]`.
    pub fn dtype(&self) -> DType {
        self._numint.dtype()
    }

    /// Drive the KS SCF via the Phase 3 generic `kernel<H>`, reusing the
    /// cycle loop verbatim (rhf.rs:138-148 pattern). The KS effective
    /// potential `J + Vxc − hyb·K` is supplied through the `KsHooks` impl
    /// whose `get_veff` adds Vxc and scales K by `hyb`.
    ///
    /// Builds the integration grid first (`grids.build(&mol)`), then runs the
    /// kernel. See the module-level JK-availability note: convergence depends
    /// on working 2-electron integrals (Phase 2 rollup gap); the driver
    /// itself is complete.
    pub fn kernel(&mut self) -> Result<ScfResult, PyscfRsError> {
        // Build the Becke grid for this geometry (coords + weights) once.
        let _ = self.grids.build(&self.mol);
        let cfg = self.to_kernel_config();

        // Disjoint immutable borrows for the KS hooks; the kernel runs to
        // completion before we write the result back to mutable fields.
        let result = {
            let hooks = crate::hooks::KsHooks::new(&self._numint, &self.mol, &self.grids, &self.xc);
            pyscf_scf::kernel(&self.mol, &hooks, cfg)?
        };
        self.mo_coeff = Some(result.mo_coeff.clone());
        self.mo_energy = Some(result.mo_energy.clone());
        self.mo_occ = Some(result.mo_occ.clone());
        self.e_tot = result.e_tot.0;
        self.converged = result.converged;
        self.cycles = result.cycles;
        Ok(result)
    }

    /// Convert the RKS struct state into a `KernelConfig` (mirrors
    /// `RHF::to_kernel_config`, rhf.rs:153-177).
    pub fn to_kernel_config(&self) -> pyscf_scf::KernelConfig {
        pyscf_scf::KernelConfig {
            conv_tol: self.conv_tol,
            conv_tol_grad: self.conv_tol_grad,
            max_cycle: self.max_cycle,
            diis: self.diis,
            diis_space: self.diis_space,
            diis_start_cycle: self.diis_start_cycle,
            diis_damp: self.diis_damp,
            level_shift: self.level_shift,
            damp: self.damp,
            direct_scf: self.direct_scf,
            direct_scf_tol: self.direct_scf_tol,
            init_guess: match self.init_guess.as_str() {
                "minao" => pyscf_scf::InitGuessMode::Minao,
                "atom" => pyscf_scf::InitGuessMode::Atom,
                "1e" => pyscf_scf::InitGuessMode::OneElectron,
                "huckel" => pyscf_scf::InitGuessMode::Huckel,
                _ => pyscf_scf::InitGuessMode::Minao,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyscf_core::Unit;
    use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};

    fn h2o() -> Mole {
        M(MoleBuildArgs {
            atom: AtomInput::String("O 0 0 0; H 0 0 0.96; H 0 0.93 -0.24".into()),
            basis: BasisInput::Name("sto-3g".into()),
            unit: Unit::Ang,
            ..Default::default()
        })
        .expect("build H2O")
    }

    /// The RKS attribute floor includes the DFT-specific fields.
    #[test]
    fn attribute_floor_has_dft_fields() {
        let ks = RKS::new(h2o(), "b3lyp");
        assert_eq!(ks.xc, "b3lyp");
        assert_eq!(ks.nlc, "");
        // grids / nlcgrids / _numint exist (compile-time field access).
        let _g = &ks.grids;
        let _ng = &ks.nlcgrids;
        let _ni = &ks._numint;
    }

    /// dtype() delegates to _numint.dtype(); when PYSCF_DTYPE is unset it is
    /// F64 (D-08 default). Asserted only when the env is genuinely unset so
    /// the test never mutates the (process-wide) env under `forbid(unsafe)`.
    #[test]
    fn dtype_delegates_to_numint() {
        let ks = RKS::new(h2o(), "svwn");
        // dtype() must equal whatever DType::from_env() resolves (delegation).
        assert_eq!(ks.dtype(), DType::from_env());
        // In the default CI/test environment PYSCF_DTYPE is unset → F64.
        if std::env::var("PYSCF_DTYPE").is_err() {
            assert_eq!(
                ks.dtype(),
                DType::F64,
                "default precision must be F64 (D-08)"
            );
        }
        // (No set_precision method exists — enforced by the absence of any
        // mutable-precision method on RKS; the source assertion lives in the
        // rks_uks_bitexact test.)
    }
}
