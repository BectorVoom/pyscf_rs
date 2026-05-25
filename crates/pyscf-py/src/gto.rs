//! `gto` submodule - minimal Python-facing Mole wrapper.

use crate::errors::pyscf_to_py;
use pyo3::prelude::*;
use pyscf_core::{Mole, Unit};
use pyscf_gto::{AtomInput, BasisInput, EcpInput, MoleBuildArgs};

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMole>()?;
    m.add_function(wrap_pyfunction!(make_mole, m)?)?;
    Ok(())
}

#[pyclass(name = "Mole", module = "pyscf._native.gto", skip_from_py_object)]
#[derive(Clone)]
pub struct PyMole {
    inner: Mole,
    atom_text: String,
    basis_text: String,
    unit_text: String,
}

#[pymethods]
impl PyMole {
    #[getter]
    fn atom(&self) -> String {
        self.atom_text.clone()
    }

    #[getter]
    fn basis(&self) -> String {
        self.basis_text.clone()
    }

    #[getter]
    fn charge(&self) -> i32 {
        self.inner.charge
    }

    #[getter]
    fn spin(&self) -> i32 {
        self.inner.spin
    }

    #[getter]
    fn cart(&self) -> bool {
        self.inner.cart
    }

    #[getter]
    fn unit(&self) -> String {
        self.unit_text.clone()
    }

    fn nao_nr(&self) -> usize {
        self.inner.nao_nr
    }

    fn dumps(&self) -> PyResult<String> {
        pyscf_gto::dumps(&self.inner).map_err(pyscf_to_py)
    }

    fn copy(&self) -> Self {
        self.clone()
    }
}

#[pyfunction(name = "M", signature = (atom="", basis="", charge=0, spin=0, cart=false, unit="Ang"))]
fn make_mole(
    atom: &str,
    basis: &str,
    charge: i32,
    spin: i32,
    cart: bool,
    unit: &str,
) -> PyResult<PyMole> {
    let parsed_unit = Unit::parse(unit).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("unsupported unit {unit:?}"))
    })?;
    let inner = pyscf_gto::M(MoleBuildArgs {
        atom: AtomInput::String(atom.to_string()),
        basis: BasisInput::Name(basis.to_string()),
        ecp: EcpInput::None,
        charge,
        spin,
        cart,
        unit: parsed_unit,
        ..Default::default()
    })
    .map_err(pyscf_to_py)?;
    Ok(PyMole {
        inner,
        atom_text: atom.to_string(),
        basis_text: basis.to_string(),
        unit_text: unit.to_string(),
    })
}
