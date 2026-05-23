//! UKS — Unrestricted Kohn-Sham. Source: `pyscf/dft/uks.py` (the UKS class +
//! the open-shell KS SCF driver layered on `scf/uhf.py`). DFT-01.
//!
//! `UKS` mirrors `pyscf_scf::UHF` (uhf.rs) PLUS the same DFT attributes as
//! `RKS` (`xc`, `grids`, `nlc`, `nlcgrids`, `_numint`). `UKS::kernel` reuses
//! the Phase 3 generic `kernel<H>` (the same generic cycle the restricted
//! path uses) with the KS hooks — the open-shell alpha/beta split lives in
//! `NumInt::nr_uks` (numint.py:1192) which the KS get_veff dispatches to for
//! spin-resolved Vxc.
//!
//! D-08: the active precision is surfaced read-only via `_numint.dtype()`
//! (no setter — deferred per D-08).
//!
//! See `crate::rks` for the JK-availability note (the bit-exact UKS energy
//! oracle is the CI-only `--features python` arm; the driver itself is real).

use pyscf_core::{MOCoefficients, Mole, PyscfRsError};
use pyscf_grids::Grids;
use pyscf_runtime::DType;
use pyscf_scf::ScfResult;

use crate::numint::NumInt;

/// Unrestricted Kohn-Sham object. Open-shell analog of [`crate::rks::RKS`];
/// `nelec` is the meaningful `(alpha, beta)` pair (uhf.rs:46).
pub struct UKS {
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
    /// `(n_alpha, n_beta)` — meaningful for the open-shell path.
    pub nelec: Option<(u32, u32)>,
    pub scf_summary: std::collections::HashMap<String, f64>,

    // ── DFT-specific attributes (uks.py) ───────────────────────────────
    pub xc: String,
    pub grids: Grids,
    pub nlc: String,
    pub nlcgrids: Grids,
    pub _numint: NumInt,
}

impl std::fmt::Debug for UKS {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UKS")
            .field("mol", &self.mol)
            .field("xc", &self.xc)
            .field("e_tot", &self.e_tot)
            .field("converged", &self.converged)
            .field("cycles", &self.cycles)
            .field("nelec", &self.nelec)
            .field("nlc", &self.nlc)
            .field("dtype", &self._numint.dtype().name())
            .field("grids", &"<Grids>")
            .field("nlcgrids", &"<Grids>")
            .field("_numint", &"<NumInt>")
            .finish()
    }
}

impl UKS {
    /// Build a UKS object for `mol` with functional `xc` (mirrors
    /// `dft.UKS(mol, xc=...)`).
    pub fn new(mol: Mole, xc: &str) -> Self {
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
            grids: Grids::new(),
            nlc: String::new(),
            nlcgrids: Grids::new(),
            _numint: NumInt::new(),
        }
    }

    /// The active scalar precision (D-08), delegating to `_numint.dtype()`.
    /// Read-only — no `set_precision` (deferred per D-08).
    pub fn dtype(&self) -> DType {
        self._numint.dtype()
    }

    /// Drive the open-shell KS SCF via the Phase 3 generic `kernel<H>`,
    /// reusing the cycle loop with the KS hooks (the open-shell Vxc lives in
    /// `NumInt::nr_uks`). Builds the grid first.
    pub fn kernel(&mut self) -> Result<ScfResult, PyscfRsError> {
        let _ = self.grids.build(&self.mol);
        let cfg = self.to_kernel_config();
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

    /// Convert UKS state into a `KernelConfig` (mirrors RKS::to_kernel_config).
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

    /// An open-shell fixture (doublet — one unpaired electron).
    fn open_shell() -> Mole {
        M(MoleBuildArgs {
            atom: AtomInput::String("H 0 0 0; H 0 0 1.4; H 0 0 2.8".into()),
            basis: BasisInput::Name("sto-3g".into()),
            spin: 1,
            unit: Unit::Bohr,
            ..Default::default()
        })
        .expect("build open-shell H3")
    }

    #[test]
    fn attribute_floor_has_dft_fields() {
        let ks = UKS::new(open_shell(), "pbe");
        assert_eq!(ks.xc, "pbe");
        let _g = &ks.grids;
        let _ni = &ks._numint;
    }

    #[test]
    fn dtype_delegates_to_numint() {
        let ks = UKS::new(open_shell(), "svwn");
        assert_eq!(ks.dtype(), DType::from_env());
        if std::env::var("PYSCF_DTYPE").is_err() {
            assert_eq!(ks.dtype(), DType::F64);
        }
    }
}
