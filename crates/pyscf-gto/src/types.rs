//! User-facing input types for `pyscf_gto::M(...)`.
//!
//! Mirrors the upstream PySCF kwargs surface (`pyscf/gto/mole.py:106-118`).
//! Phase 3 PyO3 binding wraps these with `#[pyfunction]` + `From<PyAny>`
//! impls; Phase 2 plan 02-02 ships the Rust-native typed kwargs path.

use pyscf_core::{ParsedBasis, ParsedEcp, Unit};
use std::collections::HashMap;
use std::path::PathBuf;

/// Atom-input form. Phase 2 ships forms 1-4; form 5 (callable) is Phase 3.
///
/// Source: `pyscf/gto/mole.py:320-415` `format_atom`.
#[derive(Debug, Clone)]
pub enum AtomInput {
    /// Form 1 — `"H 0 0 0; O 0 0 1; H 0 0 2"` (semicolon or newline separated).
    String(String),
    /// Form 2 — `[("H", [0,0,0]), ("O", [0,0,1]), ...]`.
    Tuples(Vec<(String, [f64; 3])>),
    /// Form 3 (alias for 2) — `[("H", vec![0,0,0]), ...]` with `Vec<f64>`
    /// coords. Useful when the caller has runtime-sized coordinate slices.
    TupleVec(Vec<(String, Vec<f64>)>),
    /// Form 4 — file path to `.xyz` / `.raw` atom listing.
    FilePath(PathBuf),
    /// Form 5 — Python callable. Deferred to Phase 3 BIND-02. Constructing
    /// this variant in Phase 2 returns `NotYetImplemented{phase:3}` at
    /// `format_atom` time.
    Callable,
}

impl Default for AtomInput {
    fn default() -> Self {
        Self::String(String::new())
    }
}

/// Basis-input form. Plan 02-03 fills the body; plan 02-02 only declares
/// the enum so `MoleBuildArgs` can carry it.
#[derive(Debug, Clone)]
pub enum BasisInput {
    /// `"cc-pvdz"`, `"sto-3g"`, `"6-31g**"`, `"def2-svp"` — ALIAS lookup.
    Name(String),
    /// `{"H": "sto-3g", "O": "cc-pvdz"}` — per-element basis.
    PerElement(HashMap<String, BasisInput>),
    /// Raw NWChem / Gaussian-94 text passed to `parse_nwchem`.
    NwchemText(String),
    /// Raw CP2K text (detected by "GTH" prefix) passed to `parse_cp2k`.
    Cp2kText(String),
    /// Already-parsed shell list (programmatic API for Phase 3 PyO3).
    Parsed(ParsedBasis),
}

impl Default for BasisInput {
    fn default() -> Self {
        Self::Name(String::new())
    }
}

/// ECP-input form. Plan 02-07 fills the body.
#[derive(Debug, Clone, Default)]
pub enum EcpInput {
    #[default]
    None,
    Name(String),
    PerElement(HashMap<String, EcpInput>),
    NwchemEcpText(String),
    Parsed(ParsedEcp),
}

/// Typed-kwargs analog of upstream `Mole.build(**kwargs)`.
/// Phase 3 PyO3 wraps with `#[pyfunction]` + Python kwargs.
///
/// Source: `pyscf/gto/mole.py:106-118` `M` factory.
#[derive(Debug, Clone)]
pub struct MoleBuildArgs {
    pub atom: AtomInput,
    pub basis: BasisInput,
    pub ecp: EcpInput,
    pub charge: i32,
    pub spin: i32,
    pub cart: bool,
    pub unit: Unit,
    pub verbose: u8,
    /// Memory ceiling in MB; default matches upstream `Mole.max_memory = 4000`.
    pub max_memory: f64,
    pub output: Option<PathBuf>,
    /// Origin shift for `format_atom` (default `[0, 0, 0]`).
    pub origin: [f64; 3],
    /// Rotation axes for `format_atom` (default identity).
    pub axes: [[f64; 3]; 3],
}

impl Default for MoleBuildArgs {
    fn default() -> Self {
        Self {
            atom: AtomInput::default(),
            basis: BasisInput::default(),
            ecp: EcpInput::default(),
            charge: 0,
            spin: 0,
            cart: false,
            unit: Unit::default(),
            verbose: 0,
            max_memory: 4000.0,
            output: None,
            origin: [0.0, 0.0, 0.0],
            axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }
}

impl MoleBuildArgs {
    /// Fluent constructor for the common path: just atom + basis.
    pub fn new(atom: AtomInput, basis: BasisInput) -> Self {
        Self {
            atom,
            basis,
            ..Default::default()
        }
    }
}
