//! `pyscf._native.pbc.gto` — the periodic `Cell` (plan 20-09).
//!
//! `PyCell` wraps [`pyscf_pbc_gto::Cell`], the first argument of every other
//! periodic binding. Four facts shape it:
//!
//! 1. **Upstream's `Cell` is mutable-then-`build()`.** Scripts set `cell.atom`,
//!    `cell.a`, `cell.basis`, `cell.pseudo` and call `cell.build()`. `PyCell`
//!    keeps those inputs (`CellInput`) and a built [`Cell`]. Changing a build
//!    input DROPS the built state (accessors then raise until `build()` runs
//!    again) — except `mesh` and `rcut`, which upstream treats as plain
//!    attributes and which are applied to the built cell in place.
//! 2. **`Cell` `Deref`s to `Mole`** (`cell.rs:148`). The molecular surface is NOT
//!    re-implemented here: `cell.mol` is a `pyscf._native.gto.Mole` over the
//!    cell's molecular half, and `__getattr__` forwards the names in
//!    [`MOLE_SURFACE`] to it — the Python image of the Rust `Deref`. Names that
//!    mean something DIFFERENT on a periodic cell (`intor`, `eval_gto`,
//!    `energy_nuc`, `dumps`, `atom_charges`, `intor_spinor`) are either bound
//!    periodically below or deliberately not forwarded.
//! 3. **`Cell::get_hcore` refuses unconditionally** (`hcore.rs:181`). It is NOT
//!    bound; `hasattr(cell, "get_hcore")` is `False`. hcore belongs to the DF
//!    object: `pyscf._native.pbc.df.FFTDF(cell).get_hcore(kpts)` (plan 20-10).
//! 4. **Refusals raise.** Every `NotYetImplemented` the Rust surface returns
//!    reaches Python as `PyscfRsRuntimeError` with `kind == "NotYetImplemented"`,
//!    never a silent wrong answer.
//!
//! Geometry note for callers: `unit="Ang"` is CODATA-2014 here and CODATA-2010
//! upstream, differing in the 8th digit of every lattice vector — build
//! comparison fixtures in Bohr.

use std::collections::HashMap;
use std::sync::OnceLock;

use numpy::ndarray::Array3;
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::{
    PyAttributeError, PyNotImplementedError, PyRuntimeError, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_gto::{
    ALattice, BravaisLattice, Cell, CellBuildArgs, CoulGArgs, KPath, LowDimFtType,
};

use crate::errors::pyscf_to_py;
use crate::gto::PyMole;
use crate::pbc::convert::{
    comp_block_to_py, extract_kpts, extract_kpts_opt, extract_mat3, extract_usize3,
    kpts_to_pyarray, mat3_to_pyarray,
};

/// Dotted module name every class in this file reports.
pub const MODULE: &str = "pyscf._native.pbc.gto";

/// The molecular names `cell.<name>` forwards to `cell.mol.<name>` — the
/// `Deref` surface. Kept to names whose molecular meaning IS the periodic one.
pub const MOLE_SURFACE: [&str; 6] = [
    "nao_nr",
    "nao_2c",
    "natm",
    "nbas",
    "nelectron",
    "atom_symbol",
];

/// Register `Cell`, `KPath`, `M` and the free functions on the child module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCell>()?;
    m.add_class::<PyKPath>()?;
    m.add_function(wrap_pyfunction!(make_cell, m)?)?;
    m.add_function(wrap_pyfunction!(make_kpts, m)?)?;
    m.add_function(wrap_pyfunction!(get_kconserv, m)?)?;
    m.add_function(wrap_pyfunction!(band_path, m)?)?;
    m.add_function(wrap_pyfunction!(band_path_from_segments, m)?)?;
    m.add_function(wrap_pyfunction!(detect_lattice, m)?)?;
    m.add_function(wrap_pyfunction!(super_cell, m)?)?;
    m.add_function(wrap_pyfunction!(cell_plus_imgs, m)?)?;
    m.add_function(wrap_pyfunction!(get_coulg, m)?)?;
    m.add_function(wrap_pyfunction!(dumps, m)?)?;
    m.add_function(wrap_pyfunction!(loads, m)?)?;
    m.add_function(wrap_pyfunction!(pack, m)?)?;
    m.add_function(wrap_pyfunction!(unpack, m)?)?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Build inputs
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum AtomSpec {
    Text(String),
    /// Canonical Bohr labels of an already-built cell (`from_cell`) — re-fed
    /// verbatim, no symbol normalisation.
    Tuples(Vec<(String, [f64; 3])>),
    /// A user list (`[['He', (x, y, z)], ('C', x, y, z), 'O 0 0 1', …]`,
    /// `mole.py:393-401`), flattened to `(label, [x, y, z])`. Fed as
    /// `AtomInput::TupleVec`, whose labels go through the same `atom_symbol`
    /// normalisation as the string form (`format_atom.rs:51-57` vs `:129`).
    List(Vec<(String, Vec<f64>)>),
}

/// `cell.pseudo` as given: a name, or an upstream per-element dict
/// (`mole.py:2575-2591`, keys matched EXACTLY against the `_atom` labels,
/// `elements.py:1146` `_symbol`; a `'default'` key covers every atom,
/// `mole.py:3954-3964`).
#[derive(Debug, Clone)]
enum PseudoSpec {
    Name(String),
    PerElement(Vec<(String, String)>),
}

#[derive(Debug, Clone)]
struct CellInput {
    atom: AtomSpec,
    /// `BasisInput::PerElement(Parsed)` for a cell rebuilt from a built one
    /// (supercells, `loads`) — re-fed bit-identically, as `super_cell` does.
    basis: BasisInput,
    pseudo: Option<PseudoSpec>,
    a: Option<ALattice>,
    unit: String,
    mesh: Option<[usize; 3]>,
    ke_cutoff: Option<f64>,
    rcut: Option<f64>,
    precision: f64,
    dimension: u8,
    low_dim_ft_type: LowDimFtType,
    fractional: bool,
    exp_to_discard: Option<f64>,
    charge: i32,
    spin: i32,
    cart: bool,
    use_particle_mesh_ewald: bool,
    space_group_symmetry: bool,
    symmorphic: bool,
    use_loose_rcut: bool,
    verbose: i64,
}

impl Default for CellInput {
    fn default() -> Self {
        Self {
            atom: AtomSpec::Text(String::new()),
            basis: BasisInput::Name("sto-3g".into()),
            pseudo: None,
            a: None,
            unit: "angstrom".into(),
            mesh: None,
            ke_cutoff: None,
            rcut: None,
            precision: pyscf_pbc_gto::DEFAULT_PRECISION,
            dimension: 3,
            low_dim_ft_type: LowDimFtType::None,
            fractional: false,
            exp_to_discard: None,
            charge: 0,
            spin: 0,
            cart: false,
            use_particle_mesh_ewald: false,
            space_group_symmetry: false,
            symmorphic: false,
            use_loose_rcut: false,
            verbose: 3,
        }
    }
}

impl CellInput {
    fn to_args(&self) -> PyResult<CellBuildArgs> {
        let unit = Unit::parse(&self.unit)
            .ok_or_else(|| PyValueError::new_err(format!("unsupported unit {:?}", self.unit)))?;
        let atom = match &self.atom {
            AtomSpec::Text(s) => AtomInput::String(s.clone()),
            AtomSpec::Tuples(t) => AtomInput::Tuples(t.clone()),
            AtomSpec::List(t) => AtomInput::TupleVec(t.clone()),
        };
        let basis = self.basis.clone();
        let a = self.a.clone().ok_or_else(|| {
            PyValueError::new_err("cell.a (the lattice vectors) must be set before build()")
        })?;
        Ok(CellBuildArgs {
            mole: MoleBuildArgs {
                atom,
                basis,
                charge: self.charge,
                spin: self.spin,
                cart: self.cart,
                unit,
                ..Default::default()
            },
            a,
            mesh: self.mesh,
            ke_cutoff: self.ke_cutoff,
            rcut: self.rcut,
            precision: self.precision,
            dimension: self.dimension,
            low_dim_ft_type: self.low_dim_ft_type,
            fractional: self.fractional,
            exp_to_discard: self.exp_to_discard,
            use_particle_mesh_ewald: self.use_particle_mesh_ewald,
            space_group_symmetry: self.space_group_symmetry,
            symmorphic: self.symmorphic,
            use_loose_rcut: self.use_loose_rcut,
            pseudo: self.pseudo_name()?,
        })
    }

    /// The single pseudopotential name handed to `CellBuildArgs::pseudo`.
    ///
    /// `CellBuildArgs` carries ONE name (`types.rs` `pseudo: Option<String>`),
    /// resolved for every element of the cell (`pseudo/mod.rs` `resolve_pseudo`).
    /// A per-element dict is that same cell exactly when every entry names the
    /// same pseudopotential and every atom the name form would give a
    /// pseudopotential is a key (checked after the build, [`check_pseudo_dict`]).
    /// A dict naming DIFFERENT pseudopotentials per element has no single-name
    /// equivalent and is refused.
    fn pseudo_name(&self) -> PyResult<Option<String>> {
        match &self.pseudo {
            None => Ok(None),
            Some(PseudoSpec::Name(n)) => Ok(Some(n.clone())),
            Some(PseudoSpec::PerElement(v)) => {
                let Some((_, first)) = v.first() else {
                    return Ok(None);
                };
                if v.iter().any(|(_, n)| !n.eq_ignore_ascii_case(first)) {
                    return Err(PyNotImplementedError::new_err(format!(
                        "cell.pseudo: a per-element dict naming different pseudopotentials \
                         ({:?}) is not bound — pyscf_pbc_gto::CellBuildArgs::pseudo carries a \
                         single name (plan 20-19 B)",
                        v.iter().map(|(_, n)| n.as_str()).collect::<Vec<_>>()
                    )));
                }
                Ok(Some(first.clone()))
            }
        }
    }

    /// Inputs that rebuild `cell` bit-identically: Cartesian Bohr atoms, the
    /// parsed basis, the Bohr lattice, and the resolved mesh/rcut pinned.
    fn from_cell(cell: &Cell) -> Self {
        let basis = if cell.mol._basis.is_empty() {
            BasisInput::Name(cell.mol.basis.clone())
        } else {
            BasisInput::PerElement(
                cell.mol
                    ._basis
                    .iter()
                    .map(|(k, pb)| (k.clone(), BasisInput::Parsed(pb.clone())))
                    .collect(),
            )
        };
        Self {
            atom: AtomSpec::Tuples(cell.mol._atom.clone()),
            basis,
            pseudo: cell.pseudo_name.clone().map(PseudoSpec::Name),
            a: Some(ALattice::Matrix(cell.a)),
            unit: "bohr".into(),
            mesh: Some(cell.mesh),
            ke_cutoff: cell.ke_cutoff,
            rcut: Some(cell.rcut),
            precision: cell.precision,
            dimension: cell.dimension,
            low_dim_ft_type: cell.low_dim_ft_type,
            fractional: false,
            exp_to_discard: cell.exp_to_discard,
            charge: cell.mol.charge,
            spin: cell.mol.spin,
            cart: cell.mol.cart,
            use_particle_mesh_ewald: cell.use_particle_mesh_ewald,
            space_group_symmetry: cell.space_group_symmetry,
            symmorphic: cell.symmorphic,
            use_loose_rcut: cell.use_loose_rcut,
            verbose: cell.mol.verbose as i64,
        }
    }
}

/// An atom label as upstream `_atom_symbol` accepts it (`elements.py:1192-1199`):
/// a string, or a nuclear charge (int, or an all-digit string) mapped through
/// `ELEMENTS`. Letter case / suffixes are left to `pyscf-gto`'s `atom_symbol`.
fn atom_label(sym: &Bound<'_, PyAny>) -> PyResult<String> {
    fn by_charge(z: i64) -> PyResult<String> {
        usize::try_from(z)
            .ok()
            .and_then(|z| pyscf_core::elements::ELEMENTS.get(z))
            .map(|s| s.to_string())
            .ok_or_else(|| PyValueError::new_err(format!("no element with nuclear charge {z}")))
    }
    if let Ok(s) = sym.extract::<String>() {
        let t = s.trim();
        if !t.is_empty() && t.bytes().all(|b| b.is_ascii_digit()) {
            return by_charge(
                t.parse()
                    .map_err(|_| PyValueError::new_err(format!("bad nuclear charge {t:?}")))?,
            );
        }
        return Ok(s);
    }
    if !sym.is_instance_of::<PyFloat>()
        && let Ok(z) = sym.extract::<i64>()
    {
        return by_charge(z);
    }
    Err(PyTypeError::new_err(
        "an atom symbol must be a string or a nuclear charge",
    ))
}

/// `cell.atom` — a string, or upstream's list form (`mole.py:393-401`):
/// every entry is either a string `'He 0 0 0'` (commas allowed), a
/// `[symbol, x, y, z]` sequence (entry[1] a Python int/float), or a
/// `[symbol, (x, y, z)]` sequence — lists and tuples alike.
fn extract_atom(v: &Bound<'_, PyAny>) -> PyResult<AtomSpec> {
    if let Ok(s) = v.extract::<String>() {
        return Ok(AtomSpec::Text(s));
    }
    let bad = || {
        PyTypeError::new_err(
            "cell.atom must be a string or a list of 'Sym x y z' / [symbol, (x, y, z)] / \
             [symbol, x, y, z] entries",
        )
    };
    let mut out = Vec::new();
    for it in v.try_iter().map_err(|_| bad())? {
        let it = it?;
        if let Ok(line) = it.extract::<String>() {
            // mole.py:395-397 — str2atm(atom.replace(',', ' ')), '#' lines skipped
            if line.trim_start().starts_with('#') {
                continue;
            }
            let line = line.replace(',', " ");
            let tok: Vec<&str> = line.split_whitespace().collect();
            if tok.len() < 4 {
                return Err(PyValueError::new_err(format!(
                    "Coordinates error in {line:?}"
                )));
            }
            let mut xyz = Vec::with_capacity(3);
            for t in &tok[1..4] {
                xyz.push(t.parse::<f64>().map_err(|_| {
                    PyValueError::new_err(format!("Failed to parse geometry {line:?}"))
                })?);
            }
            out.push((atom_label(&PyString::new(it.py(), tok[0]).into_any())?, xyz));
            continue;
        }
        let n = it.len().map_err(|_| bad())?;
        if n < 2 {
            return Err(bad());
        }
        let label = atom_label(&it.get_item(0)?)?;
        let second = it.get_item(1)?;
        // mole.py:399 — isinstance(atom[1], (int, float)) → atom[1:4]
        let xyz: Vec<f64> =
            if second.is_instance_of::<PyInt>() || second.is_instance_of::<PyFloat>() {
                (1..n.min(4))
                    .map(|i| it.get_item(i)?.extract::<f64>())
                    .collect::<PyResult<_>>()?
            } else {
                second
                    .try_iter()
                    .map_err(|_| bad())?
                    .map(|x| x?.extract::<f64>())
                    .collect::<PyResult<_>>()?
            };
        if xyz.len() != 3 {
            return Err(PyValueError::new_err(format!(
                "atom {label:?}: coordinates must have 3 entries, got {}",
                xyz.len()
            )));
        }
        out.push((label, xyz));
    }
    Ok(AtomSpec::List(out))
}

/// A 1-D float sequence (list, tuple or ndarray row).
fn f64_seq(x: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    x.try_iter()?.map(|v| v?.extract::<f64>()).collect()
}

/// `isinstance(x, (numpy.integer, int))` — an integer that is not a float.
fn is_py_int(x: &Bound<'_, PyAny>) -> bool {
    !x.is_instance_of::<PyFloat>() && !x.is_instance_of::<PyString>() && x.extract::<i64>().is_ok()
}

/// A basis given as a string (`gto/basis/__init__.py:658-691`): a name, or —
/// when it spans lines — basis-set text, CP2K when it mentions `GTH`,
/// NWChem otherwise.
fn basis_from_str(s: String) -> BasisInput {
    if s.contains('\n') {
        if s.contains("GTH") {
            BasisInput::Cp2kText(s)
        } else {
            BasisInput::NwchemText(s)
        }
    } else {
        BasisInput::Name(s)
    }
}

/// One element's basis in upstream's internal list format
/// (`mole.py:418-503`): `[[l, (e, c1, …), …], [l, kappa, (e, c1, …), …], …]`,
/// or a list of such lists (concatenated). Mirrors what upstream does to it
/// before `make_env`: empty shells dropped and shells stably sorted by `l`
/// (`format_basis`, `mole.py:457-464`), primitive rows sorted descending
/// (`make_bas_env`, `mole.py:995-1000`). Normalisation (`gto_norm` +
/// `_nomalize_contracted_ao`) is `pyscf-gto`'s `make_env.rs`, shared with
/// every other basis form.
fn shells_from_list(what: &str, raw: &Bound<'_, PyAny>) -> PyResult<pyscf_core::ParsedBasis> {
    let items: Vec<Bound<'_, PyAny>> = raw.try_iter()?.collect::<PyResult<_>>()?;
    // mole.py:490-492 — a str member or a non-int head means a list of bases.
    let internal = !items.iter().any(|x| x.is_instance_of::<PyString>())
        && items
            .first()
            .map(|b| b.get_item(0).map(|h| is_py_int(&h)).unwrap_or(false))
            .unwrap_or(true);
    let mut shells: Vec<pyscf_core::ShellSpec> = Vec::new();
    if !internal {
        for sub in &items {
            if sub.is_instance_of::<PyString>() {
                return Err(PyNotImplementedError::new_err(format!(
                    "{what}: a basis list mixing basis NAMES with shells is not bound (plan 20-19 B)"
                )));
            }
            shells.extend(shells_from_list(what, sub)?.shells);
        }
    } else {
        for b in &items {
            let n = b.len()?;
            if n == 0 {
                continue; // `[b for b in _basis if b]`
            }
            let l: i64 = b.get_item(0)?.extract()?;
            let l = u8::try_from(l)
                .map_err(|_| PyValueError::new_err(format!("{what}: bad angular momentum {l}")))?;
            let mut start = 1;
            if n > 1 && is_py_int(&b.get_item(1)?) {
                let kappa: i64 = b.get_item(1)?.extract()?;
                if kappa != 0 {
                    return Err(PyNotImplementedError::new_err(format!(
                        "{what}: kappa = {kappa} shells are not bound (the non-relativistic \
                         _bas has KAPPA_OF = 0, make_env.rs)"
                    )));
                }
                start = 2;
            }
            let mut rows: Vec<Vec<f64>> = (start..n)
                .map(|i| f64_seq(&b.get_item(i)?))
                .collect::<PyResult<_>>()?;
            if rows.is_empty() {
                return Err(PyValueError::new_err(format!(
                    "{what}: shell with l={l} has no primitives"
                )));
            }
            let width = rows[0].len();
            if width < 2 || rows.iter().any(|r| r.len() != width) {
                return Err(PyValueError::new_err(format!(
                    "{what}: every primitive of a shell must be (exponent, c1, c2, …) of one length"
                )));
            }
            // `sorted(b[1:], reverse=True)` — lexicographic on the rows, stable.
            rows.sort_by(|x, y| {
                y.iter()
                    .zip(x)
                    .map(|(a, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .find(|o| o.is_ne())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let exponents = rows.iter().map(|r| r[0]).collect();
            let coeffs = (1..width)
                .map(|c| rows.iter().map(|r| r[c]).collect())
                .collect();
            shells.push(pyscf_core::ShellSpec {
                l,
                exponents,
                coeffs,
            });
        }
    }
    shells.sort_by_key(|s| s.l);
    Ok(pyscf_core::ParsedBasis { shells })
}

/// One element's basis value: a string or an internal-format shell list.
fn basis_value(what: &str, v: &Bound<'_, PyAny>) -> PyResult<BasisInput> {
    if let Ok(s) = v.extract::<String>() {
        return Ok(basis_from_str(s));
    }
    if v.try_iter().is_ok() {
        return Ok(BasisInput::Parsed(shells_from_list(what, v)?));
    }
    Err(PyTypeError::new_err(format!(
        "{what} must be a basis name, basis text or a shell list"
    )))
}

/// `cell.basis` — a name (or basis text), a shell list applied to every atom
/// (`mole.py:3955-3957`), or a `{element: name | shell list}` dict (a
/// `'default'` key covers the rest, resolved by `format_basis.rs`).
fn extract_basis(v: &Bound<'_, PyAny>) -> PyResult<BasisInput> {
    if let Ok(d) = v.cast::<PyDict>() {
        let mut out = HashMap::new();
        for (k, val) in d.iter() {
            let k = atom_label(&k)?;
            let entry = basis_value(&format!("cell.basis[{k:?}]"), &val)?;
            out.insert(k, entry);
        }
        return Ok(BasisInput::PerElement(out));
    }
    basis_value("cell.basis", v)
}

/// `cell.pseudo` — `None`, a name, or a `{element: name}` dict.
fn extract_pseudo(v: &Bound<'_, PyAny>) -> PyResult<Option<PseudoSpec>> {
    if v.is_none() {
        return Ok(None);
    }
    if let Ok(s) = v.extract::<String>() {
        return Ok(Some(PseudoSpec::Name(s)));
    }
    if let Ok(d) = v.cast::<PyDict>() {
        let mut out = Vec::new();
        for (k, val) in d.iter() {
            let k = atom_label(&k)?;
            let n: String = val.extract().map_err(|_| {
                PyNotImplementedError::new_err(format!(
                    "cell.pseudo[{k:?}]: only pseudopotential NAMES are bound; parsed GTH \
                     parameter lists are not (plan 20-19 B)"
                ))
            })?;
            out.push((k, n));
        }
        return Ok(Some(PseudoSpec::PerElement(out)));
    }
    Err(PyTypeError::new_err(
        "cell.pseudo must be None, a pseudopotential name or a {element: name} dict",
    ))
}

/// After a build with the collapsed name ([`CellInput::pseudo_name`]), check a
/// per-element dict really describes that cell: upstream gives a
/// pseudopotential only to atoms whose label is a key (`mole.py:2586-2591`)
/// or to every atom under `'default'`.
fn check_pseudo_dict(spec: &Option<PseudoSpec>, cell: &Cell) -> PyResult<()> {
    let Some(PseudoSpec::PerElement(v)) = spec else {
        return Ok(());
    };
    if v.iter().any(|(k, _)| k == "default") {
        return Ok(());
    }
    for (label, _) in &cell.mol._atom {
        let keyed = v.iter().any(|(k, _)| k == label);
        let has_pp = cell.pseudo.as_ref().and_then(|p| p.get(label)).is_some();
        if has_pp && !keyed {
            return Err(PyNotImplementedError::new_err(format!(
                "cell.pseudo: atom {label:?} is not a key of the pseudo dict, so upstream keeps it \
                 all-electron while other elements carry a pseudopotential; that mixed cell \
                 is not bound (pyscf_pbc_gto::CellBuildArgs::pseudo is one name for every \
                 element, plan 20-19 B)"
            )));
        }
        if keyed && !has_pp {
            return Err(PyRuntimeError::new_err(format!(
                "cell.pseudo: pseudopotential {:?} has no entry for atom {label:?}",
                v.iter()
                    .find(|(k, _)| k == label)
                    .map(|(_, n)| n.as_str())
                    .unwrap_or("")
            )));
        }
    }
    Ok(())
}

fn extract_lattice(v: &Bound<'_, PyAny>) -> PyResult<ALattice> {
    if let Ok(s) = v.extract::<String>() {
        return Ok(ALattice::Str(s));
    }
    Ok(ALattice::Matrix(extract_mat3(v, "cell.a")?))
}

fn opt_f64(v: &Bound<'_, PyAny>) -> PyResult<Option<f64>> {
    if v.is_none() {
        Ok(None)
    } else {
        v.extract().map(Some)
    }
}

fn parse_low_dim(v: &Bound<'_, PyAny>) -> PyResult<LowDimFtType> {
    if v.is_none() {
        return Ok(LowDimFtType::None);
    }
    let s: String = v.extract()?;
    match s.to_ascii_lowercase().as_str() {
        "inf_vacuum" => Ok(LowDimFtType::InfVacuum),
        "" | "none" => Ok(LowDimFtType::None),
        other => Err(PyNotImplementedError::new_err(format!(
            "low_dim_ft_type = {other:?} is not bound (only None and 'inf_vacuum')"
        ))),
    }
}

fn not_built() -> PyErr {
    PyValueError::new_err(
        "Cell is not built: call cell.build() (a build input changed since the last build)",
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// PyCell
// ─────────────────────────────────────────────────────────────────────────────

/// `pyscf.pbc.gto.Cell` — a periodic cell.
///
/// There is deliberately NO `get_hcore` method: the periodic core Hamiltonian
/// depends on how the nuclear term is integrated, so it lives on the DF object —
/// `pyscf.pbc.df.FFTDF(cell, kpts).get_hcore(kpts)`.
#[pyclass(
    subclass,
    name = "Cell",
    module = "pyscf._native.pbc.gto",
    skip_from_py_object
)]
pub struct PyCell {
    input: CellInput,
    built: Option<Cell>,
    mol_view: OnceLock<Py<PyMole>>,
    /// The Python objects last assigned to `atom` / `basis` / `pseudo`,
    /// returned as-is by the getters (upstream stores the attribute verbatim).
    user_inputs: UserInputs,
    /// `cell.output` — the log file path (`mole.py:2536-2549`).
    output: Option<String>,
    /// `cell.stdout` — `sys.stdout` until `build()` opens `output`.
    stdout: Option<Py<PyAny>>,
}

#[derive(Default)]
struct UserInputs {
    atom: Option<Py<PyAny>>,
    basis: Option<Py<PyAny>>,
    pseudo: Option<Py<PyAny>>,
}

impl UserInputs {
    fn clone_ref(&self, py: Python<'_>) -> Self {
        let c = |o: &Option<Py<PyAny>>| o.as_ref().map(|x| x.clone_ref(py));
        Self {
            atom: c(&self.atom),
            basis: c(&self.basis),
            pseudo: c(&self.pseudo),
        }
    }
}

impl PyCell {
    /// Wrap an already-built Rust cell (supercells, `loads`, the DF `cell`).
    pub fn from_cell(cell: Cell) -> Self {
        Self {
            input: CellInput::from_cell(&cell),
            built: Some(cell),
            mol_view: OnceLock::new(),
            user_inputs: UserInputs::default(),
            output: None,
            stdout: None,
        }
    }

    /// The built Rust cell, or a `ValueError` naming `build()`.
    pub fn inner(&self) -> PyResult<&Cell> {
        self.built.as_ref().ok_or_else(not_built)
    }

    fn invalidate(&mut self) {
        self.built = None;
        self.mol_view = OnceLock::new();
    }

    fn do_build(&mut self) -> PyResult<()> {
        let mut cell = Cell::build(self.input.to_args()?).map_err(pyscf_to_py)?;
        check_pseudo_dict(&self.input.pseudo, &cell)?;
        // cell.py:1770-1772 — lattice symmetry (and the enlarged auto mesh), 20-14.
        crate::pbc::symm::ensure_lattice_symmetry(&mut cell)
            .map_err(crate::pbc::symm::pbc_symm_to_py)?;
        self.built = Some(cell);
        self.mol_view = OnceLock::new();
        Ok(())
    }

    /// Apply one upstream attribute. Unknown names are a `TypeError`.
    fn apply(&mut self, key: &str, v: &Bound<'_, PyAny>) -> PyResult<()> {
        match key {
            "mesh" => {
                let mesh = if v.is_none() {
                    None
                } else {
                    Some(extract_usize3(v, "cell.mesh")?)
                };
                self.input.mesh = mesh;
                match (mesh, self.built.as_mut()) {
                    (Some(m), Some(c)) => {
                        c.mesh = m;
                        c._mesh_from_build = false;
                    }
                    (None, Some(_)) => self.invalidate(),
                    _ => {}
                }
                return Ok(());
            }
            "rcut" => {
                let rcut = opt_f64(v)?;
                self.input.rcut = rcut;
                match (rcut, self.built.as_mut()) {
                    (Some(r), Some(c)) => {
                        c.rcut = r;
                        c._rcut_from_build = false;
                    }
                    (None, Some(_)) => self.invalidate(),
                    _ => {}
                }
                return Ok(());
            }
            "verbose" => {
                self.input.verbose = v.extract()?;
                return Ok(());
            }
            // mole.py:2517 — a plain attribute; build() opens it
            "output" => {
                self.output = if v.is_none() {
                    None
                } else {
                    Some(v.extract()?)
                };
                return Ok(());
            }
            "stdout" => {
                self.stdout = if v.is_none() {
                    None
                } else {
                    Some(v.clone().unbind())
                };
                return Ok(());
            }
            // upstream build() flags with no analogue here
            "dump_input" | "parse_arg" | "max_memory" => return Ok(()),
            _ => {}
        }
        match key {
            "atom" => {
                self.input.atom = extract_atom(v)?;
                self.user_inputs.atom = Some(v.clone().unbind());
            }
            "basis" => {
                self.input.basis = extract_basis(v)?;
                self.user_inputs.basis = Some(v.clone().unbind());
            }
            "pseudo" => {
                self.input.pseudo = extract_pseudo(v)?;
                self.user_inputs.pseudo = Some(v.clone().unbind());
            }
            "a" => self.input.a = Some(extract_lattice(v)?),
            "unit" => {
                let s: String = v.extract()?;
                Unit::parse(&s)
                    .ok_or_else(|| PyValueError::new_err(format!("unsupported unit {s:?}")))?;
                self.input.unit = s;
            }
            "ke_cutoff" => self.input.ke_cutoff = opt_f64(v)?,
            "precision" => self.input.precision = v.extract()?,
            "dimension" => self.input.dimension = v.extract()?,
            "low_dim_ft_type" => self.input.low_dim_ft_type = parse_low_dim(v)?,
            "fractional" => self.input.fractional = v.extract()?,
            "exp_to_discard" => self.input.exp_to_discard = opt_f64(v)?,
            "charge" => self.input.charge = v.extract()?,
            "spin" => self.input.spin = v.extract()?,
            "cart" => self.input.cart = v.extract()?,
            "use_particle_mesh_ewald" => self.input.use_particle_mesh_ewald = v.extract()?,
            "space_group_symmetry" => self.input.space_group_symmetry = v.extract()?,
            "symmorphic" => self.input.symmorphic = v.extract()?,
            "use_loose_rcut" => self.input.use_loose_rcut = v.extract()?,
            other => {
                return Err(PyTypeError::new_err(format!(
                    "Cell: unsupported keyword/attribute {other:?}"
                )));
            }
        }
        self.invalidate();
        Ok(())
    }

    fn stdout_obj(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.stdout {
            Some(s) => Ok(s.clone_ref(py)),
            None => Ok(py.import("sys")?.getattr("stdout")?.unbind()),
        }
    }

    fn mol_view(&self, py: Python<'_>) -> PyResult<Py<PyMole>> {
        let cell = self.inner()?;
        if let Some(m) = self.mol_view.get() {
            return Ok(m.clone_ref(py));
        }
        let m = Py::new(py, PyMole::from_mole(cell.mol.clone()))?;
        let _ = self.mol_view.set(m.clone_ref(py));
        Ok(m)
    }
}

/// `mole.py:2535-2549` — before anything is parsed, open `cell.output` as
/// `cell.stdout` unless that file is already the stream, announcing it on
/// `sys.stdout` when `verbose > QUIET`.
fn open_output(slf: &Bound<'_, PyCell>) -> PyResult<()> {
    let py = slf.py();
    let Some(output) = slf.borrow().output.clone() else {
        return Ok(());
    };
    let stdout = PyCell::stdout_obj(&slf.borrow(), py)?;
    let name = stdout.bind(py).getattr("name").ok();
    if let Some(name) = name
        && name.extract::<String>().is_ok_and(|n| n == output)
    {
        return Ok(());
    }
    let builtins = py.import("builtins")?;
    if slf.borrow().input.verbose > 0 {
        let os_path = py.import("os.path")?;
        let msg = if os_path.call_method1("isfile", (&output,))?.is_truthy()? {
            format!("overwrite output file: {output}")
        } else {
            format!("output file: {output}")
        };
        builtins.call_method1("print", (msg,))?;
    }
    let path = if output == "/dev/null" {
        py.import("os")?.getattr("devnull")?.extract::<String>()?
    } else {
        output
    };
    let kw = PyDict::new(py);
    kw.set_item("encoding", "utf-8")?;
    let f = builtins.call_method("open", (path, "w"), Some(&kw))?;
    slf.borrow_mut().stdout = Some(f.unbind());
    Ok(())
}

fn resolve_kpts_arg(
    kpts: Option<&Bound<'_, PyAny>>,
    kpt: Option<&Bound<'_, PyAny>>,
) -> PyResult<(Vec<[f64; 3]>, bool)> {
    if let Some((k, _)) = extract_kpts_opt(kpt)? {
        return Ok((k, true));
    }
    Ok(extract_kpts_opt(kpts)?.unwrap_or_else(|| (vec![[0.0; 3]], true)))
}

/// Shared body of `pbc_intor` / `intor`.
fn cell_pbc_intor(
    py: Python<'_>,
    cell: &Cell,
    intor: &str,
    comp: Option<usize>,
    hermi: i32,
    kpts: Option<&Bound<'_, PyAny>>,
    kpt: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let (klist, single) = resolve_kpts_arg(kpts, kpt)?;
    let out = cell
        .pbc_intor(intor, &klist, comp, hermi)
        .map_err(pyscf_to_py)?;
    // upstream `if gamma_point(kpts_lst): out = out.real`
    let real = out.gamma.iter().all(|g| *g);
    if single {
        return comp_block_to_py(py, &out.kmats[0], out.ni, out.nj, out.comp, real);
    }
    let items = out
        .kmats
        .iter()
        .map(|t| comp_block_to_py(py, t, out.ni, out.nj, out.comp, real))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyList::new(py, items)?.into_any().unbind())
}

/// Shared body of `pbc_eval_gto` / `eval_gto`.
fn cell_eval_gto(
    py: Python<'_>,
    cell: &Cell,
    eval_name: &str,
    coords: &Bound<'_, PyAny>,
    kpts: Option<&Bound<'_, PyAny>>,
    kpt: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let (coords, _) = extract_kpts(coords)?;
    let (klist, single) = resolve_kpts_arg(kpts, kpt)?;
    let out = cell
        .pbc_eval_gto(eval_name, &coords, &klist)
        .map_err(pyscf_to_py)?;
    if single {
        return comp_block_to_py(
            py,
            &out.kaos[0],
            out.ngrids,
            out.nao,
            out.comp,
            out.gamma[0],
        );
    }
    let items = out
        .kaos
        .iter()
        .zip(&out.gamma)
        .map(|(t, g)| comp_block_to_py(py, t, out.ngrids, out.nao, out.comp, *g))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyList::new(py, items)?.into_any().unbind())
}

#[pymethods]
impl PyCell {
    #[new]
    #[pyo3(signature = (**kwargs))]
    fn new(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut c = PyCell {
            input: CellInput::default(),
            built: None,
            mol_view: OnceLock::new(),
            user_inputs: UserInputs::default(),
            output: None,
            stdout: None,
        };
        if let Some(kw) = kwargs {
            for (k, v) in kw.iter() {
                let k: String = k.extract()?;
                c.apply(&k, &v)?;
            }
        }
        Ok(c)
    }

    /// `cell.build(**kwargs)` — apply the keywords (through the attribute
    /// setters, so a Python subclass's properties see them) and build. Returns
    /// the cell, as upstream does.
    #[pyo3(signature = (*_args, **kwargs))]
    fn build<'py>(
        slf: &Bound<'py, Self>,
        _args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        if let Some(kw) = kwargs {
            for (k, v) in kw.iter() {
                let k: String = k.extract()?;
                slf.borrow_mut().apply(&k, &v)?;
            }
        }
        open_output(slf)?;
        slf.borrow_mut().do_build()?;
        Ok(slf.clone())
    }

    /// Whether the cell currently holds a build matching its inputs.
    #[getter]
    fn _built(&self) -> bool {
        self.built.is_some()
    }

    /// The cell's molecular half as a `pyscf._native.gto.Mole` (the `Deref`).
    #[getter]
    fn mol(&self, py: Python<'_>) -> PyResult<Py<PyMole>> {
        self.mol_view(py)
    }

    /// Forward the [`MOLE_SURFACE`] names to `cell.mol` — the Python image of
    /// `impl Deref for Cell { type Target = Mole }`.
    ///
    /// 20-18: any other public name goes through upstream's method-style
    /// constructor lookup (`cell.KRKS(xc=, kpts=)`, `cell.KRHF()`, …;
    /// `pyscf/pbc/gto/cell.py:1407-1511`), ported in Python as
    /// `pyscf.pbc.gto._cell_methods.cell_method`. It returns `NotImplemented`
    /// when the name is not a method, and the `AttributeError` below is raised.
    fn __getattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if MOLE_SURFACE.contains(&name) {
            let mol = slf.borrow().mol_view(py)?;
            return Ok(mol.bind(py).getattr(name)?.unbind());
        }
        if !name.starts_with('_') && name != "get_hcore" {
            // cell.py:1411-1416 skips private names; the borrow is released
            // before Python runs, so the constructors may borrow the cell.
            // Without the overlay (`pyscf.pbc.gto._cell_methods` not importable,
            // e.g. `_native` used under the vendored tree) the name is simply
            // not an attribute; errors raised by the lookup itself propagate.
            if let Ok(lookup) = py
                .import("pyscf.pbc.gto._cell_methods")
                .and_then(|m| m.getattr("cell_method"))
            {
                let v = lookup.call1((slf, name))?;
                if !v.is(py.NotImplemented()) {
                    return Ok(v.unbind());
                }
            }
        }
        if name == "get_hcore" {
            return Err(PyAttributeError::new_err(
                "Cell has no get_hcore: the periodic core Hamiltonian depends on the \
                 nuclear-integral route, so it lives on the DF object — \
                 pyscf.pbc.df.FFTDF(cell, kpts).get_hcore(kpts) (Cell::get_hcore refuses, hcore.rs:181)",
            ));
        }
        Err(PyAttributeError::new_err(format!(
            "'Cell' object has no attribute {name:?}"
        )))
    }

    fn __repr__(&self) -> String {
        match &self.built {
            Some(c) => format!(
                "<pyscf._native.pbc.gto.Cell natm={} nao={} dimension={} mesh={:?} built>",
                c.mol.natm, c.mol.nao_nr, c.dimension, c.mesh
            ),
            None => "<pyscf._native.pbc.gto.Cell (not built)>".to_string(),
        }
    }

    // ── build inputs (getters + setters) ────────────────────────────────────

    #[getter]
    fn atom(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if let Some(o) = &self.user_inputs.atom {
            return Ok(o.clone_ref(py));
        }
        Ok(match &self.input.atom {
            AtomSpec::Text(s) => s.clone().into_pyobject(py)?.into_any().unbind(),
            AtomSpec::Tuples(t) => PyList::new(py, t.iter().map(|(s, x)| (s.clone(), x.to_vec())))?
                .into_any()
                .unbind(),
            AtomSpec::List(t) => PyList::new(py, t.iter().map(|(s, x)| (s.clone(), x.clone())))?
                .into_any()
                .unbind(),
        })
    }
    #[setter]
    fn set_atom(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
        self.apply("atom", v)
    }

    #[getter]
    fn basis(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if let Some(o) = &self.user_inputs.basis {
            return Ok(o.clone_ref(py));
        }
        Ok(match &self.input.basis {
            BasisInput::Name(s) => s.clone().into_pyobject(py)?.into_any().unbind(),
            _ => {
                let text = self
                    .built
                    .as_ref()
                    .map(|c| c.mol.basis.clone())
                    .unwrap_or_default();
                text.into_pyobject(py)?.into_any().unbind()
            }
        })
    }
    #[setter]
    fn set_basis(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
        self.apply("basis", v)
    }

    #[getter]
    fn pseudo(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if let Some(o) = &self.user_inputs.pseudo {
            return Ok(o.clone_ref(py));
        }
        Ok(match &self.input.pseudo {
            None => py.None(),
            Some(PseudoSpec::Name(n)) => n.clone().into_pyobject(py)?.into_any().unbind(),
            Some(PseudoSpec::PerElement(v)) => {
                let d = PyDict::new(py);
                for (k, n) in v {
                    d.set_item(k, n)?;
                }
                d.into_any().unbind()
            }
        })
    }
    #[setter]
    fn set_pseudo(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
        self.apply("pseudo", v)
    }

    /// The lattice as the user gave it (string or 3x3 list), in `cell.unit`.
    #[getter]
    fn a(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(match &self.input.a {
            None => py.None(),
            Some(ALattice::Str(s)) => s.clone().into_pyobject(py)?.into_any().unbind(),
            Some(other) => {
                let m = other.to_matrix().map_err(pyscf_to_py)?;
                mat3_to_pyarray(py, &m)?.into_any().unbind()
            }
        })
    }
    #[setter]
    fn set_a(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
        self.apply("a", v)
    }

    #[getter]
    fn unit(&self) -> String {
        self.input.unit.clone()
    }
    #[setter]
    fn set_unit(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
        self.apply("unit", v)
    }

    /// The FFT mesh — the resolved one (`Cell::try_mesh`) once built.
    #[getter]
    fn mesh(&self) -> PyResult<Option<Vec<usize>>> {
        match &self.built {
            Some(c) => Ok(Some(c.try_mesh().map_err(pyscf_to_py)?.to_vec())),
            None => Ok(self.input.mesh.map(|m| m.to_vec())),
        }
    }
    #[setter]
    fn set_mesh(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
        self.apply("mesh", v)
    }

    /// The lattice-sum radius — the resolved one (`Cell::try_rcut`) once built.
    #[getter]
    fn rcut(&self) -> PyResult<Option<f64>> {
        match &self.built {
            Some(c) => Ok(Some(c.try_rcut().map_err(pyscf_to_py)?)),
            None => Ok(self.input.rcut),
        }
    }
    #[setter]
    fn set_rcut(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
        self.apply("rcut", v)
    }

    #[getter]
    fn ke_cutoff(&self) -> Option<f64> {
        self.input.ke_cutoff
    }
    #[setter]
    fn set_ke_cutoff(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
        self.apply("ke_cutoff", v)
    }

    #[getter]
    fn precision(&self) -> f64 {
        self.input.precision
    }
    #[setter]
    fn set_precision(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
        self.apply("precision", v)
    }

    #[getter]
    fn dimension(&self) -> u8 {
        self.input.dimension
    }
    #[setter]
    fn set_dimension(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
        self.apply("dimension", v)
    }

    #[getter]
    fn low_dim_ft_type(&self) -> Option<&'static str> {
        match self.input.low_dim_ft_type {
            LowDimFtType::None => None,
            LowDimFtType::InfVacuum => Some("inf_vacuum"),
        }
    }
    #[setter]
    fn set_low_dim_ft_type(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
        self.apply("low_dim_ft_type", v)
    }

    #[getter]
    fn fractional(&self) -> bool {
        self.input.fractional
    }
    #[setter]
    fn set_fractional(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
        self.apply("fractional", v)
    }

    #[getter]
    fn exp_to_discard(&self) -> Option<f64> {
        self.input.exp_to_discard
    }
    #[setter]
    fn set_exp_to_discard(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
        self.apply("exp_to_discard", v)
    }

    #[getter]
    fn charge(&self) -> i32 {
        self.input.charge
    }
    #[setter]
    fn set_charge(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
        self.apply("charge", v)
    }

    #[getter]
    fn spin(&self) -> i32 {
        self.input.spin
    }
    #[setter]
    fn set_spin(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
        self.apply("spin", v)
    }

    #[getter]
    fn cart(&self) -> bool {
        self.input.cart
    }
    #[setter]
    fn set_cart(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
        self.apply("cart", v)
    }

    #[getter]
    fn use_particle_mesh_ewald(&self) -> bool {
        self.input.use_particle_mesh_ewald
    }
    #[setter]
    fn set_use_particle_mesh_ewald(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
        self.apply("use_particle_mesh_ewald", v)
    }

    #[getter]
    fn space_group_symmetry(&self) -> bool {
        self.input.space_group_symmetry
    }
    #[setter]
    fn set_space_group_symmetry(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
        self.apply("space_group_symmetry", v)
    }

    #[getter]
    fn symmorphic(&self) -> bool {
        self.input.symmorphic
    }
    #[setter]
    fn set_symmorphic(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
        self.apply("symmorphic", v)
    }

    #[getter]
    fn use_loose_rcut(&self) -> bool {
        self.input.use_loose_rcut
    }
    #[setter]
    fn set_use_loose_rcut(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
        self.apply("use_loose_rcut", v)
    }

    /// `cell.output` — the log file `build()` opens as `cell.stdout`
    /// (`mole.py:2536-2549`); `None` logs to `sys.stdout`.
    #[getter]
    fn output(&self) -> Option<String> {
        self.output.clone()
    }
    #[setter]
    fn set_output(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
        self.apply("output", v)
    }

    /// `cell.stdout` — `sys.stdout` unless `build()` opened `cell.output`.
    #[getter]
    fn stdout(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.stdout_obj(py)
    }
    #[setter]
    fn set_stdout(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
        self.apply("stdout", v)
    }

    #[getter]
    fn verbose(&self) -> i64 {
        self.input.verbose
    }
    #[setter]
    fn set_verbose(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
        self.apply("verbose", v)
    }

    // ── lattice / k-point accessors (built) ─────────────────────────────────

    /// Lattice vectors in Bohr, one per row (`cell.rs:226`).
    fn lattice_vectors<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        mat3_to_pyarray(py, &self.inner()?.lattice_vectors())
    }

    /// Cell volume in Bohr^3 (`cell.rs:231`).
    #[getter]
    fn vol(&self) -> PyResult<f64> {
        Ok(self.inner()?.vol())
    }

    /// `norm_to * inv(a.T)`, one reciprocal vector per row (`cell.rs:257/296`).
    #[pyo3(signature = (norm_to = 2.0 * std::f64::consts::PI))]
    fn reciprocal_vectors<'py>(
        &self,
        py: Python<'py>,
        norm_to: f64,
    ) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        let b = self
            .inner()?
            .reciprocal_vectors(norm_to)
            .map_err(pyscf_to_py)?;
        mat3_to_pyarray(py, &b)
    }

    /// Absolute k-points (1/Bohr) from scaled ones (`cell.rs:302`).
    fn get_abs_kpts(&self, py: Python<'_>, scaled_kpts: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let (k, single) = extract_kpts(scaled_kpts)?;
        let out = self.inner()?.get_abs_kpts(&k).map_err(pyscf_to_py)?;
        kpts_out(py, &out, single)
    }

    /// Scaled k-points from absolute ones (`cell.rs:309`).
    fn get_scaled_kpts(&self, py: Python<'_>, abs_kpts: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let (k, single) = extract_kpts(abs_kpts)?;
        let out = self.inner()?.get_scaled_kpts(&k);
        kpts_out(py, &out, single)
    }

    /// Monkhorst-Pack mesh (`kpts_mesh.rs:47`), absolute k-points `(nkpts, 3)`.
    /// With `space_group_symmetry` or `time_reversal_symmetry` the mesh is handed
    /// to `pyscf._native.pbc.symm.make_kpts` and a `KPoints` is returned, as
    /// upstream's `cell.make_kpts` does (`cell.py:874-883`, plan 20-14).
    #[pyo3(signature = (nks, wrap_around = false, with_gamma_point = true, scaled_center = None,
                        space_group_symmetry = false, time_reversal_symmetry = false))]
    #[allow(clippy::too_many_arguments)]
    fn make_kpts<'py>(
        slf: &Bound<'py, Self>,
        nks: &Bound<'py, PyAny>,
        wrap_around: bool,
        with_gamma_point: bool,
        scaled_center: Option<&Bound<'py, PyAny>>,
        space_group_symmetry: bool,
        time_reversal_symmetry: bool,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let (k, cell_symm) = {
            let this = slf.borrow();
            let cell = this.inner()?;
            let k = make_kpts_impl(py, cell, nks, wrap_around, with_gamma_point, scaled_center)?;
            (k, cell.space_group_symmetry)
        };
        if !(space_group_symmetry || time_reversal_symmetry) {
            return Ok(k.into_any().unbind());
        }
        if space_group_symmetry && !cell_symm {
            return Err(PyRuntimeError::new_err(
                "Using k-point symmetry now requires cell to be built with space group symmetry \
                 info:\ncell.space_group_symmetry = True\ncell.symmorphic = False\ncell.build()",
            ));
        }
        let (kv, _) = extract_kpts(k.as_any())?;
        Ok(crate::pbc::symm::kpoints_for_cell(
            py,
            slf.as_any(),
            &kv,
            space_group_symmetry,
            time_reversal_symmetry,
        )?
        .into_any())
    }

    /// Electrons over `nkpts` k-points — valence counts for GTH atoms
    /// (`cell.rs:347`).
    #[pyo3(signature = (nkpts = 1))]
    fn tot_electrons(&self, nkpts: usize) -> PyResult<usize> {
        Ok(self.inner()?.tot_electrons(nkpts))
    }

    /// Effective nuclear charges, `int32` — `Zion` for pseudopotential atoms
    /// (`cell.rs:800`). Inherent, so it shadows the molecular `atom_charges`.
    fn atom_charges<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i32>>> {
        Ok(PyArray1::from_vec(py, self.inner()?.atom_charges()))
    }

    /// The GTH pseudopotential of atom `ia` as a dict, or `None` when that atom
    /// is all-electron (`cell.rs:807`). A pyscf-rs addition (no upstream name).
    fn atom_pseudo(&self, py: Python<'_>, ia: usize) -> PyResult<Option<Py<PyDict>>> {
        let cell = self.inner()?;
        if ia >= cell.mol.natm {
            return Err(PyValueError::new_err(format!(
                "atom index {ia} out of range"
            )));
        }
        let Some(pp) = cell.atom_pseudo(ia) else {
            return Ok(None);
        };
        let d = PyDict::new(py);
        d.set_item("nelec", pp.nelec.clone())?;
        d.set_item("rloc", pp.rloc)?;
        d.set_item("local_coeffs", pp.local_coeffs.clone())?;
        let projs = PyList::empty(py);
        for p in &pp.projectors {
            let pd = PyDict::new(py);
            pd.set_item("r", p.r)?;
            pd.set_item("nproj", p.nproj)?;
            pd.set_item("h", p.h.clone())?;
            projs.append(pd)?;
        }
        d.set_item("projectors", projs)?;
        Ok(Some(d.unbind()))
    }

    /// Ewald nuclear repulsion (`ewald.rs:377`).
    fn energy_nuc(&self) -> PyResult<f64> {
        self.inner()?.energy_nuc().map_err(pyscf_to_py)
    }

    /// `cell.ewald(ew_eta, ew_cut)` (`ewald.rs:209`).
    #[pyo3(signature = (ew_eta = None, ew_cut = None))]
    fn ewald(&self, ew_eta: Option<f64>, ew_cut: Option<f64>) -> PyResult<f64> {
        pyscf_pbc_gto::ewald(self.inner()?, ew_eta, ew_cut).map_err(pyscf_to_py)
    }

    /// Lattice images per axis (`cell.rs:456`).
    fn nimgs(&self) -> PyResult<Vec<usize>> {
        Ok(self.inner()?.nimgs().map_err(pyscf_to_py)?.to_vec())
    }

    // ── evaluation surface ─────────────────────────────────────────────────

    /// `cell.pbc_intor(intor, comp, hermi, kpts, kpt)` (`pbc_intor.rs:740`).
    ///
    /// `kpts=None`/`kpt=` → one array (real when gamma); `kpts` of shape
    /// `(nkpts, 3)` → a LIST of per-k arrays (the 20-07 `KMats` form; complex
    /// unless every k-point is gamma). Shapes `(nao, nao)` or `(comp, nao, nao)`.
    /// Families outside `SUPPORTED_INTORS` and spinor names raise.
    #[pyo3(signature = (intor, comp = None, hermi = 0, kpts = None, kpt = None))]
    fn pbc_intor(
        &self,
        py: Python<'_>,
        intor: &str,
        comp: Option<usize>,
        hermi: i32,
        kpts: Option<&Bound<'_, PyAny>>,
        kpt: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        cell_pbc_intor(py, self.inner()?, intor, comp, hermi, kpts, kpt)
    }

    /// Upstream's `Cell.intor` IS `pbc_intor`.
    #[pyo3(signature = (intor, comp = None, hermi = 0, kpts = None, kpt = None))]
    fn intor(
        &self,
        py: Python<'_>,
        intor: &str,
        comp: Option<usize>,
        hermi: i32,
        kpts: Option<&Bound<'_, PyAny>>,
        kpt: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        cell_pbc_intor(py, self.inner()?, intor, comp, hermi, kpts, kpt)
    }

    /// `cell.pbc_eval_gto(eval_name, coords, kpts, kpt)` (`eval_gto.rs:303`):
    /// `(ngrids, nao)` / `(comp, ngrids, nao)` per k-point; a list for a
    /// `(nkpts, 3)` `kpts`. Complex per k unless that k-point is gamma.
    #[pyo3(signature = (eval_name, coords, kpts = None, kpt = None))]
    fn pbc_eval_gto(
        &self,
        py: Python<'_>,
        eval_name: &str,
        coords: &Bound<'_, PyAny>,
        kpts: Option<&Bound<'_, PyAny>>,
        kpt: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        cell_eval_gto(py, self.inner()?, eval_name, coords, kpts, kpt)
    }

    /// Upstream's `Cell.eval_gto` IS `pbc_eval_gto`.
    #[pyo3(signature = (eval_name, coords, kpts = None, kpt = None))]
    fn eval_gto(
        &self,
        py: Python<'_>,
        eval_name: &str,
        coords: &Bound<'_, PyAny>,
        kpts: Option<&Bound<'_, PyAny>>,
        kpt: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        cell_eval_gto(py, self.inner()?, eval_name, coords, kpts, kpt)
    }

    // ── serialisation ──────────────────────────────────────────────────────

    /// JSON of the packed cell (`dumps_loads.rs:159`).
    fn dumps(&self) -> PyResult<String> {
        pyscf_pbc_gto::dumps(self.inner()?).map_err(pyscf_to_py)
    }

    /// An independent copy (built state included).
    fn copy(&self, py: Python<'_>) -> Self {
        PyCell {
            input: self.input.clone(),
            built: self.built.clone(),
            mol_view: OnceLock::new(),
            user_inputs: self.user_inputs.clone_ref(py),
            output: self.output.clone(),
            stdout: self.stdout.as_ref().map(|s| s.clone_ref(py)),
        }
    }

    /// Pickling: `loads(cell.dumps())`.
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, (String,))> {
        let f = PyModule::import(py, MODULE)?.getattr("loads")?;
        Ok((f, (self.dumps()?,)))
    }
}

fn kpts_out(py: Python<'_>, k: &[[f64; 3]], single: bool) -> PyResult<Py<PyAny>> {
    if single {
        return Ok(PyArray1::from_vec(py, k[0].to_vec()).into_any().unbind());
    }
    Ok(kpts_to_pyarray(py, k)?.into_any().unbind())
}

fn make_kpts_impl<'py>(
    py: Python<'py>,
    cell: &Cell,
    nks: &Bound<'py, PyAny>,
    wrap_around: bool,
    with_gamma_point: bool,
    scaled_center: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    let nks = extract_usize3(nks, "nks")?;
    let center = match extract_kpts_opt(scaled_center)? {
        None => None,
        Some((c, _)) => Some(c[0]),
    };
    let k = pyscf_pbc_gto::make_kpts(cell, nks, wrap_around, with_gamma_point, center)
        .map_err(pyscf_to_py)?;
    kpts_to_pyarray(py, &k)
}

// ─────────────────────────────────────────────────────────────────────────────
// KPath
// ─────────────────────────────────────────────────────────────────────────────

/// A sampled band path (`kpath.rs:140`). A pyscf-rs addition: upstream has no
/// `band_path`, so nothing here shadows an upstream name.
#[pyclass(
    name = "KPath",
    module = "pyscf._native.pbc.gto",
    frozen,
    skip_from_py_object
)]
pub struct PyKPath {
    inner: KPath,
}

#[pymethods]
impl PyKPath {
    /// Absolute k-points (1/Bohr) — what `get_bands` takes.
    #[getter]
    fn kpts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        kpts_to_pyarray(py, &self.inner.abs)
    }
    /// Fractional coordinates.
    #[getter]
    fn scaled_kpts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        kpts_to_pyarray(py, &self.inner.scaled)
    }
    /// Cumulative path distance (the band-plot x axis).
    #[getter]
    fn x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.x.clone())
    }
    #[getter]
    fn tick_x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.tick_x.clone())
    }
    #[getter]
    fn tick_labels(&self) -> Vec<String> {
        self.inner.tick_labels.clone()
    }
    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

fn parse_bravais(s: &str) -> PyResult<BravaisLattice> {
    match s.to_ascii_lowercase().as_str() {
        "cubic" | "sc" => Ok(BravaisLattice::Cubic),
        "fcc" => Ok(BravaisLattice::Fcc),
        "bcc" => Ok(BravaisLattice::Bcc),
        "hexagonal" | "hex" => Ok(BravaisLattice::Hexagonal),
        "tetragonal" | "tet" => Ok(BravaisLattice::Tetragonal),
        other => Err(PyValueError::new_err(format!(
            "unknown Bravais lattice {other:?} (cubic, fcc, bcc, hexagonal, tetragonal)"
        ))),
    }
}

fn bravais_name(b: BravaisLattice) -> &'static str {
    match b {
        BravaisLattice::Cubic => "cubic",
        BravaisLattice::Fcc => "fcc",
        BravaisLattice::Bcc => "bcc",
        BravaisLattice::Hexagonal => "hexagonal",
        BravaisLattice::Tetragonal => "tetragonal",
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Free functions
// ─────────────────────────────────────────────────────────────────────────────

fn cell_ref<'a>(cell: &'a PyRef<'_, PyCell>) -> PyResult<&'a Cell> {
    cell.inner()
}

/// `pyscf.pbc.gto.M(**kwargs)` — construct and build (`cell.rs:749`).
#[pyfunction(name = "M", signature = (**kwargs))]
fn make_cell(py: Python<'_>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Py<PyCell>> {
    let c = Py::new(py, PyCell::new(kwargs)?)?;
    let b = c.bind(py);
    open_output(b)?;
    b.borrow_mut().do_build()?;
    Ok(c)
}

/// `make_kpts(cell, nks, wrap_around, with_gamma_point, scaled_center)`.
#[pyfunction(signature = (cell, nks, wrap_around = false, with_gamma_point = true, scaled_center = None))]
fn make_kpts<'py>(
    py: Python<'py>,
    cell: PyRef<'py, PyCell>,
    nks: &Bound<'py, PyAny>,
    wrap_around: bool,
    with_gamma_point: bool,
    scaled_center: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    make_kpts_impl(
        py,
        cell_ref(&cell)?,
        nks,
        wrap_around,
        with_gamma_point,
        scaled_center,
    )
}

/// `get_kconserv(cell, kpts)` → `(nkpts, nkpts, nkpts)` int32 (`kpts_mesh.rs:127`).
#[pyfunction]
fn get_kconserv<'py>(
    py: Python<'py>,
    cell: PyRef<'py, PyCell>,
    kpts: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, numpy::PyArray3<i32>>> {
    let (k, _) = extract_kpts(kpts)?;
    let kc = pyscf_pbc_gto::get_kconserv(cell_ref(&cell)?, &k);
    let n = kc.nkpts;
    let arr = Array3::from_shape_vec((n, n, n), kc.data)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

/// `band_path(cell, lattice=None, npoints=100)` (`kpath.rs:175`). `lattice=None`
/// detects it and raises when the cell matches no standard form.
#[pyfunction(signature = (cell, lattice = None, npoints = 100))]
fn band_path(cell: PyRef<'_, PyCell>, lattice: Option<&str>, npoints: usize) -> PyResult<PyKPath> {
    let c = cell_ref(&cell)?;
    let lat = match lattice {
        Some(s) => parse_bravais(s)?,
        None => pyscf_pbc_gto::detect_lattice(c).ok_or_else(|| {
            PyValueError::new_err("band_path: lattice not recognised; pass lattice= explicitly")
        })?,
    };
    let inner = pyscf_pbc_gto::band_path(c, lat, npoints).map_err(pyscf_to_py)?;
    Ok(PyKPath { inner })
}

/// `band_path_from_segments(cell, [[(label, (kx, ky, kz)), ...], ...], npoints)`
/// (`kpath.rs:211`).
#[pyfunction(signature = (cell, segments, npoints = 100))]
fn band_path_from_segments(
    cell: PyRef<'_, PyCell>,
    segments: Vec<Vec<(String, [f64; 3])>>,
    npoints: usize,
) -> PyResult<PyKPath> {
    let inner = pyscf_pbc_gto::band_path_from_segments(cell_ref(&cell)?, &segments, npoints)
        .map_err(pyscf_to_py)?;
    Ok(PyKPath { inner })
}

/// `detect_lattice(cell)` → `'fcc'`, … or `None` (`kpath.rs:316`).
#[pyfunction]
fn detect_lattice(cell: PyRef<'_, PyCell>) -> PyResult<Option<&'static str>> {
    Ok(pyscf_pbc_gto::detect_lattice(cell_ref(&cell)?).map(bravais_name))
}

/// `super_cell(cell, ncopy, wrap_around=False)` (`supercell.rs:53`). Raises for
/// a cell with `space_group_symmetry` (`supercell.rs:101`).
#[pyfunction(signature = (cell, ncopy, wrap_around = false))]
fn super_cell(
    cell: PyRef<'_, PyCell>,
    ncopy: &Bound<'_, PyAny>,
    wrap_around: bool,
) -> PyResult<PyCell> {
    let n = extract_usize3(ncopy, "ncopy")?;
    let sc = pyscf_pbc_gto::super_cell(cell_ref(&cell)?, n, wrap_around).map_err(pyscf_to_py)?;
    Ok(PyCell::from_cell(sc))
}

/// `cell_plus_imgs(cell, nimgs)` (`supercell.rs:77`).
#[pyfunction]
fn cell_plus_imgs(cell: PyRef<'_, PyCell>, nimgs: &Bound<'_, PyAny>) -> PyResult<PyCell> {
    let n = extract_usize3(nimgs, "nimgs")?;
    let sc = pyscf_pbc_gto::cell_plus_imgs(cell_ref(&cell)?, n).map_err(pyscf_to_py)?;
    Ok(PyCell::from_cell(sc))
}

/// `get_coulG(cell, k=None, exx=None, mesh=None, Gv=None, wrap_around=True,
/// omega=None, kpts=None)` (`coulg.rs:79`). `exx='vcut_sph'`/`'vcut_ws'` on a
/// `dimension < 3` cell raises (`exxdiv_vcut.rs:54,235`). Bound here because
/// every argument is a cell quantity; 20-14's `pbc.tools` re-exports it.
#[pyfunction(name = "get_coulG", signature = (cell, k = None, exx = None, mesh = None, Gv = None,
                                              wrap_around = true, omega = None, kpts = None))]
#[allow(non_snake_case, clippy::too_many_arguments)]
fn get_coulg<'py>(
    py: Python<'py>,
    cell: PyRef<'py, PyCell>,
    k: Option<&Bound<'py, PyAny>>,
    exx: Option<&Bound<'py, PyAny>>,
    mesh: Option<&Bound<'py, PyAny>>,
    Gv: Option<&Bound<'py, PyAny>>,
    wrap_around: bool,
    omega: Option<f64>,
    kpts: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let c = cell_ref(&cell)?;
    let k = extract_kpts_opt(k)?.map_or([0.0; 3], |(v, _)| v[0]);
    let exxdiv = crate::pbc::tools::extract_exxdiv(exx)?;
    let mesh = match mesh {
        Some(m) if !m.is_none() => Some(extract_usize3(m, "mesh")?),
        _ => None,
    };
    let gv = extract_kpts_opt(Gv)?.map(|(g, _)| g);
    let kpts = extract_kpts_opt(kpts)?.map(|(g, _)| g);
    let args = CoulGArgs {
        k,
        exxdiv,
        kpts: kpts.as_deref(),
        mesh,
        gv: gv.as_deref(),
        wrap_around,
        omega,
    };
    let v = pyscf_pbc_gto::get_coulg(c, args).map_err(pyscf_to_py)?;
    Ok(PyArray1::from_vec(py, v))
}

/// `dumps(cell)` — JSON (`dumps_loads.rs:159`).
#[pyfunction]
fn dumps(cell: PyRef<'_, PyCell>) -> PyResult<String> {
    pyscf_pbc_gto::dumps(cell_ref(&cell)?).map_err(pyscf_to_py)
}

/// `loads(json)` → a built `Cell` (`dumps_loads.rs:170`).
#[pyfunction]
fn loads(json: &str) -> PyResult<PyCell> {
    Ok(PyCell::from_cell(
        pyscf_pbc_gto::loads(json).map_err(pyscf_to_py)?,
    ))
}

/// `pack(cell)` → a dict of the packed state (`dumps_loads.rs:72`).
#[pyfunction]
fn pack<'py>(py: Python<'py>, cell: PyRef<'py, PyCell>) -> PyResult<Bound<'py, PyAny>> {
    let s = pyscf_pbc_gto::dumps(cell_ref(&cell)?).map_err(pyscf_to_py)?;
    PyModule::import(py, "json")?.call_method1("loads", (s,))
}

/// `unpack(dict)` → a built `Cell` (`dumps_loads.rs:105`).
#[pyfunction]
fn unpack(py: Python<'_>, packed: &Bound<'_, PyAny>) -> PyResult<PyCell> {
    let s: String = PyModule::import(py, "json")?
        .call_method1("dumps", (packed,))?
        .extract()?;
    Ok(PyCell::from_cell(
        pyscf_pbc_gto::loads(&s).map_err(pyscf_to_py)?,
    ))
}
