//! Molecule type. Phase 2 plan 02-02 (GTO-08) implements the ≥30 attribute
//! floor; plan 02-04 (GTO-04 + GTO-11) wires the cintx flat-array
//! projection (`_atm`/`_bas`/`_env`/`ao_loc_nr`/`nao_nr`) and the
//! zero-copy `Arc<BasisSet>` storage.
//!
//! Source-of-truth: `pyscf/gto/mole.py:2189+` `MoleBase` class declaration
//! (every `self.X = Y` field initialiser maps to a `pub X: ...` field
//! here) and the `format_atom` / `make_env` outputs at `mole.py:320-1105`.
//!
//! Slot layout for `_atm`, `_bas`, `_env` is consumed via
//! `crate::raw_layout::*` which is a `pub use cintx_compat::raw::*` per
//! D-03 — there is no local mirror, so drift between cintx-compat and
//! pyscf-rs is impossible by construction (T-02-04-01 mitigation).

use crate::error::{CoreError, PyscfRsError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use crate::basis_set::BasisSet;

/// Inversion-of-control builder hook (F-11).
///
/// `pyscf-core` is zero-compute-deps (FOUND-02): it cannot parse basis input
/// itself and cannot depend on `pyscf-gto` (that would be circular). To still
/// let `Mole::build()` work as the upstream-faithful entry point, the higher
/// `pyscf-gto` layer *registers* its builder here at runtime. The hook type is
/// `fn(&mut Mole)` — a purely core-owned signature, so no `pyscf-gto` type ever
/// leaks into `pyscf-core` and the crate dependency direction (`gto → core`)
/// is unchanged.
pub type MoleBuilderHook = fn(&mut Mole) -> Result<(), PyscfRsError>;

static MOLE_BUILDER: OnceLock<MoleBuilderHook> = OnceLock::new();

/// Register the process-global [`Mole`] builder hook (F-11).
///
/// Called by `pyscf-gto` (from `pyscf_gto::M` / `build_from` and the PyO3
/// module init) to arm `Mole::build()`. Idempotent: only the first
/// registration wins; subsequent calls are silently ignored (there is exactly
/// one builder impl in-tree, so a re-registration is never a conflict).
pub fn register_mole_builder(hook: MoleBuilderHook) {
    // Ignore the `Err` returned when already set — re-registration is a no-op.
    let _ = MOLE_BUILDER.set(hook);
}

/// `true` once a builder hook has been registered (test/diagnostic helper).
pub fn mole_builder_is_registered() -> bool {
    MOLE_BUILDER.get().is_some()
}

/// Length unit for atom coordinates.
///
/// Internal storage is always Bohr (atomic units); `format_atom` converts
/// from the user-supplied unit into Bohr at parse time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Unit {
    /// Angstrom — upstream PySCF default. Conversion: 1 Å = 1.8897261339213 Bohr.
    #[default]
    Ang,
    /// Bohr / Atomic Units — passes through unchanged.
    Bohr,
    /// Synonym for `Bohr` (`unit='AU'` upstream).
    AU,
}

impl Unit {
    /// Conversion factor to Bohr (the internal unit).
    ///
    /// 1.8897261339213 = CODATA 2014 Bohr-radius reciprocal in Å. Matches
    /// upstream `pyscf/data/nist.py BOHR` (verbatim).
    pub fn length_in_au(self) -> f64 {
        match self {
            Self::Ang => 1.8897261339213,
            Self::Bohr | Self::AU => 1.0,
        }
    }

    /// Parse from upstream string forms (`"Ang"`, `"Angstrom"`, `"ang"`,
    /// `"B"`, `"Bohr"`, `"AU"`, `"au"`). Case-insensitive.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "ang" | "angstrom" | "a" => Some(Self::Ang),
            "b" | "bohr" => Some(Self::Bohr),
            "au" => Some(Self::AU),
            _ => None,
        }
    }
}

/// Nuclear model — point / Gaussian-finite / ECP-fractional.
///
/// Mirrors libcint `NUC_MOD_OF` integer values per `cintx_compat::raw`:
/// `POINT_NUC=1`, `GAUSSIAN_NUC=2`, `FRAC_CHARGE_NUC=3`. Note libcint uses
/// 1/2/3 (NOT 0/1/2) — pyscf-rs follows cintx-compat exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NuclearModel {
    #[default]
    Point,
    Gaussian,
    FracCharge,
}

/// Parsed (symbol, [x, y, z]) pair — internal `_atom` representation.
/// Coordinates are always in Bohr regardless of input `Unit`.
pub type ParsedAtom = (String, [f64; 3]);

/// Parsed basis-set entry (per element). Plan 02-03 fills the body; plan
/// 02-02 (this plan) ships the placeholder shape so `Mole._basis` has a
/// real type.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ParsedBasis {
    /// Per-shell `ShellSpec` (`(l, exponents, coeffs)`).
    pub shells: Vec<ShellSpec>,
}

/// One shell specification — angular momentum + primitives + contraction
/// matrix.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShellSpec {
    /// Angular momentum l (0=s, 1=p, 2=d, ...).
    pub l: u8,
    /// Primitive Gaussian exponents.
    pub exponents: Vec<f64>,
    /// Contraction-coefficient matrix `coeffs[ctr_idx][prim_idx]`
    /// (F-order on flatten).
    pub coeffs: Vec<Vec<f64>>,
}

/// Parsed ECP entry (per element). Plan 02-07 fills the body.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ParsedEcp {
    /// Number of core electrons removed by the ECP.
    pub n_core: u32,
    /// Per-channel ECP shells. Plan 02-07 expands the field.
    pub channels: Vec<EcpShell>,
}

/// One ECP channel shell. Plan 02-07 expands.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EcpShell {
    /// l = -1 marks the local (UL) channel; 0..=5 marks projector channels.
    pub l: i8,
    pub n_powers: Vec<i32>,
    pub exponents: Vec<f64>,
    pub coeffs: Vec<f64>,
}

/// Parsed Goedecker–Teter–Hutter (GTH) pseudopotential entry (per element).
///
/// Source: `pyscf/gto/basis/parse_cp2k_pp.py`. GTH pseudopotentials are the
/// PBC-pseudopotential construct (`Cell.pseudo`), NOT molecular ECPs: they
/// carry a Gaussian-erf local part (`rloc` + a short polynomial in `r²`) plus
/// per-angular-momentum non-local Gaussian projectors coupled by a symmetric
/// `h` matrix. Kept DISTINCT from [`ParsedEcp`] — which models the NWChem
/// `r^n e^{-ζr²}` ECP form consumed by the molecular ECP integral path — since
/// the two have incompatible radial structure and feed different engines.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GthPseudo {
    /// Valence electron count per angular momentum (`s, p, d, …`); the sum is
    /// the effective core charge `Zion` the pseudopotential represents.
    pub nelec: Vec<u32>,
    /// Local part: Gaussian radius `r_loc`.
    pub rloc: f64,
    /// Local part: polynomial coefficients `C_1 … C_nexp`.
    pub local_coeffs: Vec<f64>,
    /// Non-local projector channels, one per angular momentum `l = 0, 1, …`
    /// in file order.
    pub projectors: Vec<GthProjector>,
}

/// One GTH non-local projector channel (a fixed angular momentum `l`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GthProjector {
    /// Projector Gaussian radius `r_l`.
    pub r: f64,
    /// Number of projectors for this `l` (the `h` matrix dimension).
    pub nproj: usize,
    /// Full **symmetric** `nproj × nproj` coupling matrix, row-major
    /// (`h[i * nproj + j]`). The file stores only the upper triangle; the
    /// lower triangle is mirrored on parse.
    pub h: Vec<f64>,
}

/// The ≥30-attribute Mole floor.
///
/// EVERY attribute in RESEARCH "Standard Stack" §"Mole Attribute Floor"
/// is represented either as a public field or a public method (the lazy
/// items — `enuc()`, `mass_list()`, `atom_charges()`, `atom_coords()`).
///
/// Source: `pyscf/gto/mole.py:2189+` `MoleBase` class.
#[derive(Debug, Clone, Default)]
pub struct Mole {
    // === User input (preserved verbatim — items 1, 2, 31) ===
    /// Raw user-supplied `atom=` argument (stringified for echo / dumps).
    pub atom: String,
    /// Raw user-supplied `basis=` argument (stringified for echo / dumps).
    pub basis: String,
    /// Raw user-supplied `ecp=` argument (item 31).
    pub ecp: String,

    // === Scalar molecular state (items 3, 4, 5, 12, 13, 14, 15, 16, 24, 27, 28, 29) ===
    /// Total molecular charge (electrons subtracted).
    pub charge: i32,
    /// Multiplicity − 1 (singlet = 0).
    pub spin: i32,
    /// Number of electrons (computed: `sum(atom_charges) - charge`).
    pub nelectron: usize,
    /// `false` = spherical AOs (default); `true` = Cartesian AOs.
    pub cart: bool,
    /// Verbosity 0..=9. Phase 3 wires to `tracing-subscriber`.
    pub verbose: u8,
    /// Memory ceiling in MB. Phase 6 CCSD-08 honours this.
    pub max_memory: f64,
    /// Length unit at input time (atom coordinates already converted to Bohr).
    pub unit: Unit,
    /// Optional log-file path. Phase 3 wires to a `tracing` writer.
    pub output: Option<PathBuf>,
    /// `true` once `Mole::build()` succeeds (plan 02-04 sets this).
    pub _built: bool,
    /// Symmetry detection. v1 is C1 only — always `false`.
    pub symmetry: bool,
    /// Symmetry group name. v1 is `"C1"` always.
    pub groupname: String,
    /// Top group name. v1 is `"C1"` always.
    pub topgroup: String,

    // === Derived from atom-input (items 6, 21) ===
    /// Atom count. Item 6.
    pub natm: usize,
    /// `(symbol, [x, y, z]_in_Bohr)` pairs — internal _atom representation.
    /// Item 21. Populated by `format_atom` (plan 02-02 wires).
    pub _atom: Vec<ParsedAtom>,

    // === Derived from basis (items 7, 8, 9, 10, 22) — plan 02-04 populates ===
    /// Shell count.
    pub nbas: usize,
    /// Number of spherical AOs. Plan 02-04 populates.
    pub nao_nr: usize,
    /// Number of spinor AOs. Phase 2 stub OK; rare.
    pub nao_2c: usize,
    /// `ao_loc_nr` — cumulative AO offsets per shell. Plan 02-04 populates.
    pub ao_loc_nr: Vec<i32>,
    /// Per-element parsed basis. Plan 02-03 populates.
    pub _basis: HashMap<String, ParsedBasis>,

    // === ECP (items 20, 23) — plan 02-07 populates ===
    /// `_ecpbas` — per-shell ECP rows, `n_ecp_shells * BAS_SLOTS (=8)` i32 elements.
    pub _ecpbas: Vec<i32>,
    /// Per-element parsed ECP. Plan 02-07 populates.
    pub _ecp: HashMap<String, ParsedEcp>,

    // === Flat-array projection (items 17, 18, 19) — plan 02-04 populates per D-03 ===
    /// Per-atom rows; `ATM_SLOTS=6` i32 entries each. Layout follows
    /// `crate::raw_layout::*` (re-export of `cintx_compat::raw`).
    pub _atm: Vec<i32>,
    /// Per-shell rows; `BAS_SLOTS=8` i32 entries each. Layout follows
    /// `crate::raw_layout::*` (re-export of `cintx_compat::raw`).
    pub _bas: Vec<i32>,
    /// Flat double array. First `PTR_ENV_START=20` slots are libcint reserved.
    pub _env: Vec<f64>,

    // === Per-atom nuclear model overrides (items 25, 26) ===
    /// Per-atom-symbol nuclear model override (default `Point`).
    pub nucmod: HashMap<String, NuclearModel>,
    /// Optional finite-nucleus zeta override per atom symbol.
    pub nucprop: HashMap<String, f64>,

    // === Cintx zero-copy basis (item driving GTO-11; plan 02-04 populates) ===
    /// `Arc<BasisSet>` so `Mole.clone()` is cheap and `basis_set()`
    /// returns a zero-copy borrow. Plan 02-04 wires `cintx_core::BasisSet`
    /// here once the projection lands; plan 02-02 leaves this `None`.
    pub basis_set: Option<Arc<BasisSet>>,

    // === Lazy (items 30, 33) — methods, not fields ===
    // enuc: f64                    — see method `enuc()` below
    // mass: Vec<f64>               — see method `mass_list()` below

    // === Out-of-scope (item 32) ===
    /// Pseudopotentials (PBC). Always `None` in v1.
    pub pseudo: Option<()>,
}

impl Mole {
    /// Build (basis load + flat-array projection) — the upstream-faithful
    /// `mol.build()` entry point.
    ///
    /// `pyscf-core` is zero-compute-deps (FOUND-02) and cannot parse basis
    /// input itself, so the real work is delegated to a builder hook the
    /// `pyscf-gto` layer registers at runtime via [`register_mole_builder`]
    /// (F-11 inversion-of-control). Dispatch contract:
    ///
    /// - **hook registered** → invoke it (always, so the upstream
    ///   `mol.copy(); mol.basis = aux; mol.build()` rebuild pattern works) and
    ///   return `Ok(self)`. Any gto front-door use (`pyscf_gto::M` /
    ///   `build_from`, or the PyO3 module init) arms the hook, so a Mole cloned
    ///   from a built one can always be rebuilt directly.
    /// - **hook not registered, already `_built`** → `Ok(self)` (idempotent:
    ///   the gto front-door already populated this Mole).
    /// - **hook not registered, not built** → `NotYetImplemented` with an
    ///   actionable message. Reachable only on a pure cold start that never
    ///   touched `pyscf-gto`; calling `pyscf_gto::register_mole_builder()` (or
    ///   any `pyscf_gto::M`) once enables this path.
    pub fn build(&mut self) -> Result<&mut Self, PyscfRsError> {
        match MOLE_BUILDER.get() {
            Some(hook) => {
                hook(self)?;
                Ok(self)
            }
            None if self._built => Ok(self),
            None => Err(PyscfRsError::NotYetImplemented {
                phase: 2,
                what: "Mole::build(): no builder registered (pyscf-core is zero-compute-deps, FOUND-02). \
                       Build via pyscf_gto::M(MoleBuildArgs { atom, basis, .. }), or call \
                       pyscf_gto::register_mole_builder() once to arm Mole::build().",
            }),
        }
    }

    /// GTO-11 zero-copy accessor — returns the `Arc<BasisSet>` if built.
    /// Cloning the returned `Arc` bumps the refcount; the underlying
    /// `cintx_core::BasisSet` is never deep-copied.
    pub fn basis_set(&self) -> Option<&Arc<BasisSet>> {
        self.basis_set.as_ref()
    }

    /// GTO-11 — returns a clone of the inner `Arc<BasisSet>` (cheap, refcount bump).
    /// Errors if `mol.build()` hasn't run.
    pub fn cintx_basis(&self) -> Result<Arc<BasisSet>, PyscfRsError> {
        self.basis_set.as_ref().cloned().ok_or_else(|| {
            PyscfRsError::Core(CoreError::InvalidMolecule(
                "Mole not built — call pyscf_gto::M(args) or mol.build() first".into(),
            ))
        })
    }

    // ---------- Method-floor obligations (RESEARCH "Mole Attribute Floor" closing paragraph) ----------

    /// Per-atom nuclear charges, derived from `_atm[i * ATM_SLOTS + CHARGE_OF]`.
    ///
    /// Plan 02-04 populates `_atm`; until `_built` is set, this falls back to
    /// a symbol-table lookup. The signature is stable across Phase 2.
    pub fn atom_charges(&self) -> Vec<i32> {
        use crate::raw_layout::{ATM_SLOTS, CHARGE_OF};
        if self._atm.len() < self.natm * ATM_SLOTS {
            // Plan 02-02 only populates _atom (not _atm); fall back to a
            // symbol-table lookup so `atom_charges()` works for plan 02-02
            // tests. Plan 02-04 populates _atm and the slow path drops out.
            return self
                ._atom
                .iter()
                .map(|(s, _)| crate::elements::charge_for_symbol(s).unwrap_or(0))
                .collect();
        }
        (0..self.natm)
            .map(|i| self._atm[i * ATM_SLOTS + CHARGE_OF])
            .collect()
    }

    /// Per-atom coordinates in Bohr.
    pub fn atom_coords(&self) -> Vec<[f64; 3]> {
        self._atom.iter().map(|(_, xyz)| *xyz).collect()
    }

    /// Single-atom coordinate accessor (in Bohr).
    pub fn atom_coord(&self, i: usize) -> [f64; 3] {
        self._atom[i].1
    }

    /// Atomic mass list (item 33 lazy).
    ///
    /// Plan 02-02 returns a stub `vec![0.0; natm]`; Phase 4 DFT may wire
    /// real ATOMIC_MASS values from `pyscf/data/elements.py` if VV10 needs
    /// them.
    pub fn mass_list(&self) -> Vec<f64> {
        vec![0.0; self.natm]
    }

    /// Classical Coulomb (nuclear repulsion) energy (item 30 lazy).
    ///
    /// Source: `pyscf/gto/mole.py:1522` `classical_coulomb_energy`.
    /// Coordinates are already in Bohr (Mole invariant), so no unit
    /// conversion is needed here.
    pub fn enuc(&self) -> f64 {
        let coords = self.atom_coords();
        let charges = self.atom_charges();
        let mut e = 0.0;
        for i in 0..self.natm {
            for j in (i + 1)..self.natm {
                let dx = coords[i][0] - coords[j][0];
                let dy = coords[i][1] - coords[j][1];
                let dz = coords[i][2] - coords[j][2];
                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                if r > 0.0 {
                    e += (charges[i] as f64) * (charges[j] as f64) / r;
                }
            }
        }
        e
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_from_str_ang_variants() {
        assert_eq!(Unit::parse("Ang"), Some(Unit::Ang));
        assert_eq!(Unit::parse("ang"), Some(Unit::Ang));
        assert_eq!(Unit::parse("Angstrom"), Some(Unit::Ang));
        assert_eq!(Unit::parse("A"), Some(Unit::Ang));
    }

    #[test]
    fn unit_from_str_bohr_variants() {
        assert_eq!(Unit::parse("Bohr"), Some(Unit::Bohr));
        assert_eq!(Unit::parse("bohr"), Some(Unit::Bohr));
        assert_eq!(Unit::parse("B"), Some(Unit::Bohr));
        assert_eq!(Unit::parse("AU"), Some(Unit::AU));
        assert_eq!(Unit::parse("au"), Some(Unit::AU));
    }

    #[test]
    fn unit_from_str_unknown() {
        assert_eq!(Unit::parse("nm"), None);
        assert_eq!(Unit::parse(""), None);
    }

    #[test]
    fn unit_length_in_au_factors() {
        assert_eq!(Unit::Ang.length_in_au(), 1.8897261339213);
        assert_eq!(Unit::Bohr.length_in_au(), 1.0);
        assert_eq!(Unit::AU.length_in_au(), 1.0);
    }

    #[test]
    fn default_mole_enuc_is_zero() {
        let mol = Mole::default();
        assert_eq!(mol.enuc(), 0.0);
    }

    // F-11 — IoC builder hook. In the pyscf-core test binary `pyscf-gto` is
    // never linked, so no builder is ever registered: this exercises the
    // unregistered (cold-start / FOUND-02) arm of `Mole::build()`.

    #[test]
    fn build_unregistered_unbuilt_is_not_yet_implemented() {
        assert!(
            !mole_builder_is_registered(),
            "no gto hook should be registered in the pyscf-core test binary"
        );
        let mut mol = Mole::default();
        match mol.build() {
            Err(PyscfRsError::NotYetImplemented { phase, what }) => {
                assert_eq!(phase, 2);
                assert!(
                    what.contains("pyscf_gto::M") && what.contains("register_mole_builder"),
                    "error message should point at the gto front-door + register helper, got: {what}"
                );
            }
            other => panic!("expected NotYetImplemented, got {other:?}"),
        }
    }

    #[test]
    fn build_unregistered_already_built_is_idempotent_ok() {
        // A Mole the gto front-door already populated carries `_built = true`;
        // calling `build()` again with no hook registered is a no-op Ok.
        let mut mol = Mole {
            _built: true,
            ..Default::default()
        };
        assert!(mol.build().is_ok());
        assert!(mol._built);
    }

    #[test]
    fn default_mole_mass_list_empty() {
        let mol = Mole::default();
        assert!(mol.mass_list().is_empty());
    }

    #[test]
    fn enuc_two_atom_classical() {
        let mol = Mole {
            natm: 2,
            _atom: vec![
                ("H".to_string(), [0.0, 0.0, 0.0]),
                ("H".to_string(), [0.0, 0.0, 1.4]),
            ],
            ..Default::default()
        };
        // V_NN = Z_H * Z_H / r = 1 * 1 / 1.4
        let expected = 1.0 / 1.4;
        let got = mol.enuc();
        assert!(
            (got - expected).abs() < 1e-12,
            "got {got}, expected {expected}"
        );
    }
}
