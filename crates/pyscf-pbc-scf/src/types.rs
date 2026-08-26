//! Shared periodic-SCF types — plan 11-09.
//!
//! # Layout conventions (fixed here, used everywhere below)
//!
//! * every `nao x nao` matrix is a ROW-MAJOR [`CTensor`] — the layout
//!   `pyscf_algebra::zeigh_gen` and `zgemm_dense` take. Phase 10's products are
//!   F-order and are converted on ingest by
//!   `pyscf_pbc_df::zlinalg::forder_to_c`;
//! * MO coefficients are COLUMN-MAJOR (`c[ao + mo * nao]`) — the layout
//!   `zeigh_gen` RETURNS, and upstream's `mo_coeff[:, i]` convention;
//! * a spin/density channel is a "set"; `nset` is 1 for RHF/GHF and 2 for
//!   UHF/ROHF. Index a `(set, k)` pair with [`KScfResult::idx`].

use pyscf_algebra::CTensor;

/// k-resolved matrices for ONE channel: `mats[k]` is `nao x nao` row-major.
pub type KMats = Vec<CTensor>;

/// `nset` channels of k-resolved matrices — upstream's
/// `(nset, nkpts, nao, nao)`.
pub type KDms = Vec<KMats>;

/// Which initial guess the driver starts from — the periodic subset of
/// `khf.py:345-386`.
#[derive(Debug, Clone, Default)]
pub enum KInitGuess {
    /// `'minao'` — the molecular MINAO guess cast to every k-point
    /// (`_cast_mol_init_guess`, `khf.py:345-362`). Upstream's default.
    #[default]
    Minao,
    /// `'atom'` — superposition of atomic densities, likewise cast.
    Atom,
    /// `'1e'` — diagonalise `hcore`.
    OneElectron,
    /// `'chkfile'` — read `scf/mo_coeff` + `scf/mo_occ` back from an HDF5
    /// checkpoint written by this crate or by upstream.
    Chkfile(std::path::PathBuf),
    /// A caller-supplied density matrix, already k-resolved.
    UserDm(KDms),
}

/// Driver settings. Defaults follow `khf.KSCF.__init__` and
/// `scf.hf.SCF.__init__`.
#[derive(Debug, Clone)]
pub struct KScfConfig {
    /// Energy convergence threshold. Upstream's periodic default is
    /// `max(cell.precision * 10, 1e-8)`; [`KScfConfig::for_cell`] applies it.
    pub conv_tol: f64,
    /// Orbital-gradient threshold. `None` means `sqrt(conv_tol)`, upstream's
    /// rule.
    pub conv_tol_grad: Option<f64>,
    /// Maximum SCF cycles.
    pub max_cycle: u32,
    /// Enable Pulay (C-)DIIS.
    pub diis: bool,
    /// DIIS subspace size.
    pub diis_space: usize,
    /// First cycle at which DIIS extrapolates.
    pub diis_start_cycle: u32,
    /// Fock damping applied before DIIS starts (`mf.damp`).
    pub damp: f64,
    /// Level shift applied to the virtual block (`mf.level_shift`).
    pub level_shift: f64,
    /// The initial guess.
    pub init_guess: KInitGuess,
    /// Write the converged result to this HDF5 checkpoint.
    pub chkfile: Option<std::path::PathBuf>,
    /// Emit per-cycle `tracing::info!` lines.
    pub verbose: bool,
}

impl Default for KScfConfig {
    fn default() -> Self {
        Self {
            conv_tol: 1e-8,
            conv_tol_grad: None,
            max_cycle: 50,
            diis: true,
            diis_space: 8,
            diis_start_cycle: 1,
            damp: 0.0,
            level_shift: 0.0,
            init_guess: KInitGuess::default(),
            chkfile: None,
            verbose: false,
        }
    }
}

impl KScfConfig {
    /// Upstream's periodic default `conv_tol = max(cell.precision*10, 1e-8)`
    /// (`khf.py:483`).
    pub fn for_cell(cell: &pyscf_pbc_gto::Cell) -> Self {
        Self {
            conv_tol: (cell.precision * 10.0).max(1e-8),
            ..Default::default()
        }
    }

    /// The gradient threshold actually used: `conv_tol_grad` or `sqrt(conv_tol)`.
    pub fn grad_tol(&self) -> f64 {
        self.conv_tol_grad.unwrap_or_else(|| self.conv_tol.sqrt())
    }
}

/// What the driver returns.
#[derive(Debug, Clone)]
pub struct KScfResult {
    /// Total energy (electronic + Ewald nuclear repulsion).
    pub e_tot: f64,
    /// Electronic energy.
    pub e_elec: f64,
    /// Two-electron (Coulomb + exchange) part of the electronic energy.
    pub e_coul: f64,
    /// Nuclear repulsion (`cell.ewald()`).
    pub e_nuc: f64,
    /// `mo_energy[idx(set, k)]`, ascending within each block.
    pub mo_energy: Vec<Vec<f64>>,
    /// `mo_coeff[idx(set, k)]`, COLUMN-MAJOR `nao x nmo`.
    pub mo_coeff: Vec<CTensor>,
    /// `mo_occ[idx(set, k)]`.
    pub mo_occ: Vec<Vec<f64>>,
    /// The converged density matrices.
    pub dm: KDms,
    /// Whether the convergence criteria were met.
    pub converged: bool,
    /// Cycles actually run.
    pub cycles: u32,
    /// Channel count (1 for RHF/GHF, 2 for UHF/ROHF).
    pub nset: usize,
    /// k-point count.
    pub nkpts: usize,
    /// Fermi level, from the last `get_occ`.
    pub fermi: Vec<f64>,
    /// Smearing entropy contribution `-sigma * S`, when smearing is on.
    pub e_free: Option<f64>,
    /// `(e_tot + e_free) / 2` — the zero-temperature extrapolation upstream
    /// reports alongside a smeared energy.
    pub e_zero: Option<f64>,
}

impl KScfResult {
    /// Flat index of the `(set, k)` block.
    pub fn idx(&self, set: usize, k: usize) -> usize {
        set * self.nkpts + k
    }
}
