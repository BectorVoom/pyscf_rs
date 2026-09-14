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
use pyo3::exceptions::{PyAttributeError, PyNotImplementedError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_gto::{
    ALattice, BravaisLattice, Cell, CellBuildArgs, CoulGArgs, ExxDiv, KPath, LowDimFtType,
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
    Tuples(Vec<(String, [f64; 3])>),
}

#[derive(Debug, Clone)]
enum BasisSpec {
    Name(String),
    PerElement(Vec<(String, String)>),
    /// The parsed per-element basis of an already-built cell (supercells,
    /// `loads`) — re-fed bit-identically, exactly as `super_cell` does.
    Parsed(HashMap<String, pyscf_core::ParsedBasis>),
}

#[derive(Debug, Clone)]
struct CellInput {
    atom: AtomSpec,
    basis: BasisSpec,
    pseudo: Option<String>,
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
            basis: BasisSpec::Name("sto-3g".into()),
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
        };
        let basis = match &self.basis {
            BasisSpec::Name(n) => BasisInput::Name(n.clone()),
            BasisSpec::PerElement(v) => BasisInput::PerElement(
                v.iter()
                    .map(|(k, n)| (k.clone(), BasisInput::Name(n.clone())))
                    .collect(),
            ),
            BasisSpec::Parsed(p) => BasisInput::PerElement(
                p.iter()
                    .map(|(k, pb)| (k.clone(), BasisInput::Parsed(pb.clone())))
                    .collect(),
            ),
        };
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
            pseudo: self.pseudo.clone(),
        })
    }

    /// Inputs that rebuild `cell` bit-identically: Cartesian Bohr atoms, the
    /// parsed basis, the Bohr lattice, and the resolved mesh/rcut pinned.
    fn from_cell(cell: &Cell) -> Self {
        let basis = if cell.mol._basis.is_empty() {
            BasisSpec::Name(cell.mol.basis.clone())
        } else {
            BasisSpec::Parsed(cell.mol._basis.clone())
        };
        Self {
            atom: AtomSpec::Tuples(cell.mol._atom.clone()),
            basis,
            pseudo: cell.pseudo_name.clone(),
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

fn extract_atom(v: &Bound<'_, PyAny>) -> PyResult<AtomSpec> {
    if let Ok(s) = v.extract::<String>() {
        return Ok(AtomSpec::Text(s));
    }
    let items: Vec<Bound<'_, PyAny>> = v.extract().map_err(|_| {
        PyTypeError::new_err("cell.atom must be a string or a list of (symbol, (x, y, z))")
    })?;
    let mut out = Vec::with_capacity(items.len());
    for it in items {
        let (sym, xyz): (String, [f64; 3]) = it
            .extract()
            .or_else(|_| {
                let t: (String, Vec<f64>) = it.extract()?;
                if t.1.len() != 3 {
                    return Err(PyValueError::new_err(
                        "atom coordinates must have 3 entries",
                    ));
                }
                Ok((t.0, [t.1[0], t.1[1], t.1[2]]))
            })
            .map_err(|_| {
                PyTypeError::new_err("each cell.atom entry must be (symbol, (x, y, z))")
            })?;
        out.push((sym, xyz));
    }
    Ok(AtomSpec::Tuples(out))
}

fn extract_basis(v: &Bound<'_, PyAny>) -> PyResult<BasisSpec> {
    if let Ok(s) = v.extract::<String>() {
        return Ok(BasisSpec::Name(s));
    }
    if let Ok(d) = v.cast::<PyDict>() {
        let mut out = Vec::new();
        for (k, val) in d.iter() {
            let k: String = k.extract()?;
            let n: String = val.extract().map_err(|_| {
                PyNotImplementedError::new_err(format!(
                    "cell.basis[{k:?}]: only basis NAMES are bound (plan 20-09); explicit shell lists are not"
                ))
            })?;
            out.push((k, n));
        }
        return Ok(BasisSpec::PerElement(out));
    }
    Err(PyTypeError::new_err(
        "cell.basis must be a name or a {element: name} dict",
    ))
}

fn extract_pseudo(v: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    if v.is_none() {
        return Ok(None);
    }
    v.extract::<String>().map(Some).map_err(|_| {
        PyNotImplementedError::new_err(
            "cell.pseudo: only a single pseudopotential NAME (e.g. 'gth-pade') is bound (plan 20-09)",
        )
    })
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
}

impl PyCell {
    /// Wrap an already-built Rust cell (supercells, `loads`, the DF `cell`).
    pub fn from_cell(cell: Cell) -> Self {
        Self {
            input: CellInput::from_cell(&cell),
            built: Some(cell),
            mol_view: OnceLock::new(),
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
        let cell = Cell::build(self.input.to_args()?).map_err(pyscf_to_py)?;
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
            // upstream build() flags with no analogue here
            "dump_input" | "parse_arg" | "output" | "max_memory" | "stdout" => return Ok(()),
            _ => {}
        }
        match key {
            "atom" => self.input.atom = extract_atom(v)?,
            "basis" => self.input.basis = extract_basis(v)?,
            "pseudo" => self.input.pseudo = extract_pseudo(v)?,
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
    fn __getattr__(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        if MOLE_SURFACE.contains(&name) {
            let mol = self.mol_view(py)?;
            return Ok(mol.bind(py).getattr(name)?.unbind());
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
        Ok(match &self.input.atom {
            AtomSpec::Text(s) => s.clone().into_pyobject(py)?.into_any().unbind(),
            AtomSpec::Tuples(t) => PyList::new(py, t.iter().map(|(s, x)| (s.clone(), x.to_vec())))?
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
        Ok(match &self.input.basis {
            BasisSpec::Name(s) => s.clone().into_pyobject(py)?.into_any().unbind(),
            BasisSpec::PerElement(v) => {
                let d = PyDict::new(py);
                for (k, n) in v {
                    d.set_item(k, n)?;
                }
                d.into_any().unbind()
            }
            BasisSpec::Parsed(_) => {
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
    fn pseudo(&self) -> Option<String> {
        self.input.pseudo.clone()
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
    #[pyo3(signature = (nks, wrap_around = false, with_gamma_point = true, scaled_center = None,
                        space_group_symmetry = false, time_reversal_symmetry = false))]
    fn make_kpts<'py>(
        &self,
        py: Python<'py>,
        nks: &Bound<'py, PyAny>,
        wrap_around: bool,
        with_gamma_point: bool,
        scaled_center: Option<&Bound<'py, PyAny>>,
        space_group_symmetry: bool,
        time_reversal_symmetry: bool,
    ) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        make_kpts_impl(
            py,
            self.inner()?,
            nks,
            wrap_around,
            with_gamma_point,
            scaled_center,
            space_group_symmetry || time_reversal_symmetry,
        )
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
    fn copy(&self) -> Self {
        PyCell {
            input: self.input.clone(),
            built: self.built.clone(),
            mol_view: OnceLock::new(),
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
    symmetry: bool,
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    if symmetry {
        return Err(PyNotImplementedError::new_err(
            "make_kpts(space_group_symmetry/time_reversal_symmetry=True) returns a KPoints \
             object, which lives in pyscf.pbc.symm (pyscf_pbc_symm::kpts::make_kpts, plan 20-14); \
             build the plain mesh here and pass it there",
        ));
    }
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
    let mut c = PyCell::new(kwargs)?;
    c.do_build()?;
    Py::new(py, c)
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
        false,
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
    let exxdiv = match exx {
        None => None,
        Some(v) if v.is_none() => None,
        Some(v) => match v.extract::<String>() {
            Ok(s) => ExxDiv::parse(&s),
            Err(_) => {
                if v.extract::<bool>().unwrap_or(false) {
                    Some(ExxDiv::Ewald)
                } else {
                    None
                }
            }
        },
    };
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
