//! User-facing input types for `pyscf_gto::M(...)`.
//!
//! Mirrors the upstream PySCF kwargs surface (`pyscf/gto/mole.py:106-118`).
//! Phase 3 PyO3 binding wraps these with `#[pyfunction]` + `From<PyAny>`
//! impls; Phase 2 plan 02-02 ships the Rust-native typed kwargs path.

use pyscf_core::{ParsedBasis, ParsedEcp, PyscfRsError, Unit};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

/// A callable atom source: a closure that, when invoked, produces the atom
/// specification to parse.
///
/// Mirrors upstream PySCF's callable `atom=` form (`pyscf/gto/mole.py`), where
/// `atom` may be any object whose call returns the atom list/string — used for
/// geometry that is generated at build time (geomopt / `as_scanner`). The
/// closure returns another [`AtomInput`] (typically `String`/`Tuples`), which
/// [`crate::format_atom::format_atom`] resolves recursively — **one level
/// only**: a callable that returns another callable is rejected, both to bound
/// the recursion and to keep the "callable produces a concrete spec" contract.
///
/// `Arc` (not `Box`) keeps [`AtomInput`] `Clone` — `MoleBuildArgs` and the
/// built `Mole` clone freely; `Send + Sync` lets them cross threads.
pub type AtomCallable = Arc<dyn Fn() -> Result<AtomInput, PyscfRsError> + Send + Sync>;

/// Atom-input form. All five upstream forms are supported; form 5 (callable)
/// resolves entirely in Rust via a closure (see [`AtomCallable`]).
///
/// Source: `pyscf/gto/mole.py:320-415` `format_atom`.
#[derive(Clone)]
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
    /// Form 5 — callable atom source (see [`AtomCallable`]). The closure is
    /// invoked at `format_atom` time and its produced spec resolved (one
    /// level). Build with [`AtomInput::callable`].
    Callable(AtomCallable),
}

// Manual `Debug`: the `Callable` closure is not `Debug`, so the enum cannot
// derive it. Every other variant forwards to its inner value's `Debug`.
impl std::fmt::Debug for AtomInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::String(s) => f.debug_tuple("String").field(s).finish(),
            Self::Tuples(t) => f.debug_tuple("Tuples").field(t).finish(),
            Self::TupleVec(t) => f.debug_tuple("TupleVec").field(t).finish(),
            Self::FilePath(p) => f.debug_tuple("FilePath").field(p).finish(),
            Self::Callable(_) => f.write_str("Callable(<closure>)"),
        }
    }
}

impl AtomInput {
    /// Build an [`AtomInput::Callable`] from any closure that produces an atom
    /// spec. The closure is invoked once, at `format_atom` time.
    ///
    /// ```
    /// use pyscf_gto::AtomInput;
    /// let atom = AtomInput::callable(|| Ok(AtomInput::String("H 0 0 0; H 0 0 1.4".into())));
    /// ```
    pub fn callable<F>(f: F) -> Self
    where
        F: Fn() -> Result<AtomInput, PyscfRsError> + Send + Sync + 'static,
    {
        Self::Callable(Arc::new(f))
    }
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
