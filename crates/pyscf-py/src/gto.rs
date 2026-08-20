//! `gto` submodule - minimal Python-facing Mole wrapper.

use crate::errors::pyscf_to_py;
use numpy::ndarray::{ArrayD, IxDyn, ShapeBuilder};
use numpy::{Complex64, IntoPyArray, PyArrayDyn};
use pyo3::exceptions::PyValueError;
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

impl PyMole {
    /// Build a `PyMole` from a native `Mole` (used by the geomopt bridge to
    /// return the optimized molecule to Python). The text accessors derive from
    /// the `Mole`'s own `atom`/`basis`/`unit` fields. The internal geometry is
    /// stored in Bohr; the `atom` text mirrors whatever the `Mole` was built
    /// with (the geomopt bridge re-seeds the `_atom` geometry via `set_geom_`).
    pub(crate) fn from_mole(inner: Mole) -> Self {
        let atom_text = inner.atom.clone();
        let basis_text = inner.basis.clone();
        let unit_text = match inner.unit {
            Unit::Ang => "Ang",
            Unit::Bohr => "Bohr",
            Unit::AU => "AU",
        }
        .to_string();
        PyMole {
            inner,
            atom_text,
            basis_text,
            unit_text,
        }
    }
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

    /// 2-component (spinor) AO count, `n2c = Σ 2·(2l+1)·nctr` (F-03 T1).
    fn nao_2c(&self) -> usize {
        self.inner.nao_2c
    }

    /// F-03 T5 — complex spinor integrals as a numpy `complex128` array.
    ///
    /// Wraps [`pyscf_gto::intor_spinor`]: `name` is a spinor operator (e.g.
    /// `"int1e_ovlp_spinor"`, `"int1e_kin_spinor"`, `"int1e_nuc_spinor"`, or
    /// `"int2e_spinor"`; a bare/`_sph`/`_cart` name is normalised to `_spinor`).
    /// Returns an F-order array shaped `[n2c, n2c]` (1e) or `[n2c; 4]` (2e).
    ///
    /// The `re`/`im` F-order buffers are interleaved into `Complex64` and handed
    /// to numpy with the matching column-major (`order='F'`) layout, so the
    /// result indexes identically to upstream `mol.intor("…_spinor")`.
    fn intor_spinor<'py>(
        &self,
        py: Python<'py>,
        name: &str,
    ) -> PyResult<Bound<'py, PyArrayDyn<Complex64>>> {
        let out = pyscf_gto::intor_spinor(&self.inner, name).map_err(pyscf_to_py)?;
        let data: Vec<Complex64> = out
            .re
            .iter()
            .zip(out.im.iter())
            .map(|(&re, &im)| Complex64::new(re, im))
            .collect();
        // F-order: `IxDyn(&shape).f()` reads the flat column-major buffer with
        // the given shape, so element (i,j,…) = data[i + j·n + …] — exactly the
        // F-order convention of `IntorOutputComplex`.
        let arr = ArrayD::from_shape_vec(IxDyn(&out.shape).f(), data).map_err(|e| {
            PyValueError::new_err(format!(
                "intor_spinor('{name}'): cannot shape {} elements as {:?} (F-order): {e}",
                out.re.len(),
                out.shape,
            ))
        })?;
        Ok(arr.into_pyarray(py))
    }

    fn eval_gto<'py>(
        &self,
        py: Python<'py>,
        eval_name: &str,
        coords: numpy::PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let coords_slice = coords.as_slice()?;
        let coords_vec: Vec<[f64; 3]> = coords_slice
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();
        let out = pyscf_gto::eval_gto(&self.inner, eval_name, &coords_vec).map_err(pyscf_to_py)?;
        let n_vals = out.values.len();
        let shape = out.shape;
        let arr = ArrayD::from_shape_vec(IxDyn(&shape).f(), out.values).map_err(|e| {
            PyValueError::new_err(format!(
                "eval_gto('{eval_name}'): cannot shape {} elements as {:?} (F-order): {e}",
                n_vals,
                shape,
            ))
        })?;
        Ok(arr.into_pyarray(py))
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
