//! `pyscf._native.pbc.ao2mo` — the backward-compatible `pbc/ao2mo/eris.py`
//! wrappers (plan 20-15), over `pyscf-pbc-ao2mo/src/eris.rs`.
//!
//! | Python | Rust | returns |
//! |---|---|---|
//! | `general(cell, mo_coeffs, kpts=None, compact=False)` | `general` | `(nmo1*nmo2, nmo3*nmo4)` complex |
//! | `get_mo_eri(cell, mo_coeffs, kpts=None)` | `get_mo_eri` | as `general` |
//! | `get_mo_pairs_G(cell, mo_coeffs, kpts=None, q=None)` | `get_mo_pairs_g` | `(ngrids, nmoi*nmoj)` complex |
//! | `get_mo_pairs_invG(cell, mo_coeffs, kpts=None, q=None)` | `get_mo_pairs_invg` | `(ngrids, nmoi*nmoj)` complex |
//! | `assemble_eri(cell, orb_pair_invG1, orb_pair_G2, q=None)` | `assemble_eri` | `(npair1, npair2)` complex |
//! | `get_ao_pairs_G(cell, kpts=None)` | `get_ao_pairs_g` | `(ngrids, nao*nao)` complex |
//! | `get_ao_eri(cell, kpts=None)` | `get_ao_eri` | `(nao*nao, nao*nao)` complex |
//!
//! Every wrapper builds a fresh `FFTDF` over `cell`, as upstream's do. Results
//! are always complex and never `s2`-packed (upstream packs at Γ when real).
//!
//! # Pair normalisation (20-15 SUMMARY D6)
//!
//! `pyscf_pbc_ao2mo::get_mo_pairs_g` / `get_mo_pairs_invg` / `get_ao_pairs_g`
//! return `FFT[ρ] · Ω/N` (`pbc_ao2mo::fft_ao_pairs_g`'s `scale`), while
//! upstream's `eris.get_mo_pairs_G` and `FFTDF.get_ao_pairs` return the bare
//! `tools.fft` and `assemble_eri` applies `Ω/N²` itself (`eris.py:219`). The
//! Rust `assemble_eri` also applies `Ω/N²`, so feeding it the Rust pairs is
//! off by `Ω/N`. The bindings multiply the pair arrays by `N/Ω` so the Python
//! surface has upstream's convention and `assemble_eri(get_mo_pairs_G(..),
//! get_mo_pairs_invG(..))` reproduces `get_mo_eri`.

use numpy::{Complex64, PyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyscf_algebra::CTensor;
use pyscf_pbc_ao2mo::{PairG, PbcAo2moError};
use pyscf_pbc_df::MoCoeff;

use crate::bridge::extract_cell_from_pyany;
use crate::errors::{PyscfRsRuntimeError, pyscf_to_py};
use crate::pbc::convert::extract_kpts;
use crate::pbc::df::pbc_df_to_py;
use crate::pbc::mp::{complex_from_py, complex_to_py};

/// Register the seven wrappers on `pyscf._native.pbc.ao2mo`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(general, m)?)?;
    m.add_function(wrap_pyfunction!(get_mo_eri, m)?)?;
    m.add_function(wrap_pyfunction!(get_mo_pairs_g, m)?)?;
    m.add_function(wrap_pyfunction!(get_mo_pairs_invg, m)?)?;
    m.add_function(wrap_pyfunction!(assemble_eri, m)?)?;
    m.add_function(wrap_pyfunction!(get_ao_pairs_g, m)?)?;
    m.add_function(wrap_pyfunction!(get_ao_eri, m)?)?;
    Ok(())
}

fn ao2mo_to_py(err: PbcAo2moError) -> PyErr {
    match err {
        PbcAo2moError::Core(e) => pyscf_to_py(e),
        PbcAo2moError::Df(e) => pbc_df_to_py(e),
        other => {
            PyscfRsRuntimeError::new_err((other.to_string(), "PbcAo2mo", Vec::<String>::new()))
        }
    }
}

fn mo_list(obj: &Bound<'_, PyAny>, n: usize) -> PyResult<Vec<MoCoeff>> {
    let items = obj.try_iter()?.collect::<PyResult<Vec<_>>>()?;
    if items.len() != n {
        return Err(PyValueError::new_err(format!(
            "mo_coeffs must hold {n} (nao, nmo) arrays, got {}",
            items.len()
        )));
    }
    items
        .iter()
        .map(|x| {
            let (c, shape) = complex_from_py(x)?;
            if shape.len() != 2 {
                return Err(PyValueError::new_err(format!(
                    "each MO coefficient block must be (nao, nmo), got {shape:?}"
                )));
            }
            Ok(MoCoeff::new(shape[0], shape[1], c))
        })
        .collect()
}

fn kpts_n<const N: usize>(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<[[f64; 3]; N]>> {
    let Some(obj) = obj.filter(|x| !x.is_none()) else {
        return Ok(None);
    };
    let (k, _) = extract_kpts(obj)?;
    <[[f64; 3]; N]>::try_from(k.as_slice())
        .map(Some)
        .map_err(|_| PyValueError::new_err(format!("kpts must hold {N} k-points")))
}

fn vec3(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<[f64; 3]>> {
    obj.filter(|x| !x.is_none())
        .map(|x| x.extract::<[f64; 3]>())
        .transpose()
}

/// Undo the Rust pair convention's `Ω/N` factor (see the module doc, D6).
fn upstream_pair_scale(data: &mut CTensor, ngrids: usize, vol: f64) {
    let f = ngrids as f64 / vol;
    data.re.iter_mut().for_each(|x| *x *= f);
    data.im.iter_mut().for_each(|x| *x *= f);
}

fn pair_to_py<'py>(
    py: Python<'py>,
    mut p: PairG,
    vol: f64,
) -> PyResult<Bound<'py, PyArrayDyn<Complex64>>> {
    upstream_pair_scale(&mut p.data, p.ngrids, vol);
    complex_to_py(py, &p.data, &[p.ngrids, p.npair])
}

fn pair_from_py(obj: &Bound<'_, PyAny>) -> PyResult<PairG> {
    let (data, shape) = complex_from_py(obj)?;
    if shape.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "an orbital-pair array must be (ngrids, npair), got {shape:?}"
        )));
    }
    Ok(PairG {
        ngrids: shape[0],
        npair: shape[1],
        data,
    })
}

/// `general(cell, mo_coeffs, kpts=None, compact=False)` (`eris.py:34`).
#[pyfunction]
#[pyo3(signature = (cell, mo_coeffs, kpts = None, compact = false))]
fn general<'py>(
    py: Python<'py>,
    cell: &Bound<'py, PyAny>,
    mo_coeffs: &Bound<'py, PyAny>,
    kpts: Option<&Bound<'py, PyAny>>,
    compact: bool,
) -> PyResult<Bound<'py, PyArrayDyn<Complex64>>> {
    let c = extract_cell_from_pyany(py, cell)?;
    let mos = mo_list(mo_coeffs, 4)?;
    let k = kpts_n::<4>(kpts)?;
    let eri = py
        .detach(|| pyscf_pbc_ao2mo::general(&c, [&mos[0], &mos[1], &mos[2], &mos[3]], k, compact))
        .map_err(ao2mo_to_py)?;
    complex_to_py(py, &eri.data, &[eri.row.len(), eri.col.len()])
}

/// `get_mo_eri(cell, mo_coeffs, kpts=None)` (`eris.py:41`).
#[pyfunction]
#[pyo3(signature = (cell, mo_coeffs, kpts = None))]
fn get_mo_eri<'py>(
    py: Python<'py>,
    cell: &Bound<'py, PyAny>,
    mo_coeffs: &Bound<'py, PyAny>,
    kpts: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyArrayDyn<Complex64>>> {
    let c = extract_cell_from_pyany(py, cell)?;
    let mos = mo_list(mo_coeffs, 4)?;
    let k = kpts_n::<4>(kpts)?;
    let eri = py
        .detach(|| pyscf_pbc_ao2mo::get_mo_eri(&c, [&mos[0], &mos[1], &mos[2], &mos[3]], k))
        .map_err(ao2mo_to_py)?;
    complex_to_py(py, &eri.data, &[eri.row.len(), eri.col.len()])
}

/// `get_mo_pairs_G(cell, mo_coeffs, kpts=None, q=None)` (`eris.py:59`).
#[pyfunction]
#[pyo3(name = "get_mo_pairs_G", signature = (cell, mo_coeffs, kpts = None, q = None))]
fn get_mo_pairs_g<'py>(
    py: Python<'py>,
    cell: &Bound<'py, PyAny>,
    mo_coeffs: &Bound<'py, PyAny>,
    kpts: Option<&Bound<'py, PyAny>>,
    q: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyArrayDyn<Complex64>>> {
    let c = extract_cell_from_pyany(py, cell)?;
    let mos = mo_list(mo_coeffs, 2)?;
    let (k, q) = (kpts_n::<2>(kpts)?, vec3(q)?);
    let p = py
        .detach(|| pyscf_pbc_ao2mo::get_mo_pairs_g(&c, [&mos[0], &mos[1]], k, q))
        .map_err(ao2mo_to_py)?;
    pair_to_py(py, p, c.vol())
}

/// `get_mo_pairs_invG(cell, mo_coeffs, kpts=None, q=None)` (`eris.py:112`).
#[pyfunction]
#[pyo3(name = "get_mo_pairs_invG", signature = (cell, mo_coeffs, kpts = None, q = None))]
fn get_mo_pairs_invg<'py>(
    py: Python<'py>,
    cell: &Bound<'py, PyAny>,
    mo_coeffs: &Bound<'py, PyAny>,
    kpts: Option<&Bound<'py, PyAny>>,
    q: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyArrayDyn<Complex64>>> {
    let c = extract_cell_from_pyany(py, cell)?;
    let mos = mo_list(mo_coeffs, 2)?;
    let (k, q) = (kpts_n::<2>(kpts)?, vec3(q)?);
    let p = py
        .detach(|| pyscf_pbc_ao2mo::get_mo_pairs_invg(&c, [&mos[0], &mos[1]], k, q))
        .map_err(ao2mo_to_py)?;
    pair_to_py(py, p, c.vol())
}

/// `assemble_eri(cell, orb_pair_invG1, orb_pair_G2, q=None)` (`eris.py:165`).
#[pyfunction]
#[pyo3(signature = (cell, orb_pair_invG1, orb_pair_G2, q = None, verbose = None))]
#[allow(non_snake_case)]
fn assemble_eri<'py>(
    py: Python<'py>,
    cell: &Bound<'py, PyAny>,
    orb_pair_invG1: &Bound<'py, PyAny>,
    orb_pair_G2: &Bound<'py, PyAny>,
    q: Option<&Bound<'py, PyAny>>,
    verbose: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyArrayDyn<Complex64>>> {
    let _ = verbose;
    let c = extract_cell_from_pyany(py, cell)?;
    let (left, right) = (pair_from_py(orb_pair_invG1)?, pair_from_py(orb_pair_G2)?);
    let q = vec3(q)?;
    let out: CTensor = py
        .detach(|| pyscf_pbc_ao2mo::assemble_eri(&c, &left, &right, q))
        .map_err(ao2mo_to_py)?;
    complex_to_py(py, &out, &[left.npair, right.npair])
}

/// `get_ao_pairs_G(cell, kpts=None)` (`eris.py:231`) — the forward pairs only
/// (upstream's body returns `FFTDF.get_ao_pairs`, one array; its docstring's
/// `(G, invG)` pair is stale).
#[pyfunction]
#[pyo3(name = "get_ao_pairs_G", signature = (cell, kpts = None))]
fn get_ao_pairs_g<'py>(
    py: Python<'py>,
    cell: &Bound<'py, PyAny>,
    kpts: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyArrayDyn<Complex64>>> {
    let c = extract_cell_from_pyany(py, cell)?;
    let k = kpts_n::<2>(kpts)?;
    let nao = c.mol.nao_nr;
    let (re, im) = py
        .detach(|| pyscf_pbc_ao2mo::get_ao_pairs_g(&c, k))
        .map_err(ao2mo_to_py)?;
    let npair = nao * nao;
    let ngrids = re.len().checked_div(npair).unwrap_or(0);
    let mut data = CTensor { re, im };
    upstream_pair_scale(&mut data, ngrids, c.vol());
    complex_to_py(py, &data, &[ngrids, npair])
}

/// `get_ao_eri(cell, kpts=None)` (`eris.py:244`).
#[pyfunction]
#[pyo3(signature = (cell, kpts = None))]
fn get_ao_eri<'py>(
    py: Python<'py>,
    cell: &Bound<'py, PyAny>,
    kpts: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyArrayDyn<Complex64>>> {
    let c = extract_cell_from_pyany(py, cell)?;
    let k = kpts_n::<4>(kpts)?;
    let nao = c.mol.nao_nr;
    let out = py
        .detach(|| pyscf_pbc_ao2mo::get_ao_eri(&c, k))
        .map_err(ao2mo_to_py)?;
    complex_to_py(py, &out, &[nao * nao, nao * nao])
}
