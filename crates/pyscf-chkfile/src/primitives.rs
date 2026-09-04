//! HDF5 chkfile primitives — h5py-compatible reads/writes.
//!
//! Source: pyscf/lib/chkfile.py:28-191 (primitive layer + save_mol) and
//! pyscf/scf/chkfile.py:25-42 (SCF schema). Every encoding choice
//! (VarLenUnicode mol JSON, F-order mo_coeff, 1D/scalar datasets) matches
//! upstream byte-for-byte; the cross-language ORACLE-08 round-trip (plan
//! 03-08) is the empirical seal.
//!
//! Pitfall 8 mitigation: `write_dataset_f_order` writes the transpose of the
//! input view so the on-disk byte layout corresponds to F-order column-
//! major. h5py reading the same dataset with column-major interpretation
//! recovers the original LAPACK layout.
use crate::error::ChkfileError;
use hdf5_metno as hdf5;
use hdf5_metno::types::VarLenUnicode;
use ndarray::{Array1, Array2, ArrayView2};
use std::path::Path;
use std::str::FromStr;

/// Open or create a chkfile for write. Mirrors `pyscf/lib/chkfile.py:dump`
/// behaviour: if file exists, append; otherwise create.
pub fn open_for_write<P: AsRef<Path>>(path: P) -> Result<hdf5::File, ChkfileError> {
    let p = path.as_ref();
    if p.exists() {
        // For an empty file (NamedTempFile creates an empty file on init),
        // `append` fails. Detect and switch to `create`.
        let is_empty = std::fs::metadata(p).map(|m| m.len() == 0).unwrap_or(false);
        if is_empty {
            hdf5::File::create(p).map_err(ChkfileError::from)
        } else {
            hdf5::File::append(p).map_err(ChkfileError::from)
        }
    } else {
        hdf5::File::create(p).map_err(ChkfileError::from)
    }
}

/// Open chkfile for read-only access.
pub fn open_for_read<P: AsRef<Path>>(path: P) -> Result<hdf5::File, ChkfileError> {
    hdf5::File::open(path).map_err(ChkfileError::from)
}

/// Read a sub-group (e.g. group `"scf"` inside a chkfile).
pub fn read_group(file: &hdf5::File, name: &str) -> Result<hdf5::Group, ChkfileError> {
    file.group(name).map_err(ChkfileError::from)
}

/// Write `mol.dumps()` JSON string under `/mol` as VL Unicode.
/// Source: pyscf/lib/chkfile.py:179-191 `save_mol`.
pub fn write_mol(file: &hdf5::File, mol_json: &str) -> Result<(), ChkfileError> {
    let vl = VarLenUnicode::from_str(mol_json).map_err(|_| ChkfileError::InvalidUtf8)?;
    if file.link_exists("mol") {
        file.unlink("mol")?;
    }
    file.new_dataset::<VarLenUnicode>()
        .create("mol")?
        .write_scalar(&vl)?;
    Ok(())
}

/// Read `/mol` VL Unicode string.
pub fn read_mol(file: &hdf5::File) -> Result<String, ChkfileError> {
    let vl: VarLenUnicode = file.dataset("mol")?.read_scalar()?;
    Ok(vl.as_str().to_string())
}

/// Write a scalar f64 (e.g. `e_tot`).
pub fn write_scalar_f64(group: &hdf5::Group, key: &str, value: f64) -> Result<(), ChkfileError> {
    if group.link_exists(key) {
        group.unlink(key)?;
    }
    group
        .new_dataset::<f64>()
        .create(key)?
        .write_scalar(&value)?;
    Ok(())
}

/// Read a scalar f64.
pub fn read_scalar_f64(group: &hdf5::Group, key: &str) -> Result<f64, ChkfileError> {
    Ok(group.dataset(key)?.read_scalar::<f64>()?)
}

/// Write a 1D f64 array (e.g. `mo_energy`, `mo_occ`).
pub fn write_dataset_1d(group: &hdf5::Group, key: &str, data: &[f64]) -> Result<(), ChkfileError> {
    if group.link_exists(key) {
        group.unlink(key)?;
    }
    let arr = Array1::from_vec(data.to_vec());
    group
        .new_dataset::<f64>()
        .shape([arr.len()])
        .create(key)?
        .write(&arr)?;
    Ok(())
}

/// Read a 1D f64 array.
pub fn read_dataset_1d(group: &hdf5::Group, key: &str) -> Result<Vec<f64>, ChkfileError> {
    let arr: Array1<f64> = group.dataset(key)?.read_1d()?;
    Ok(arr.to_vec())
}

/// Write a 2D f64 dataset in C-order (default). For density matrices etc.
pub fn write_dataset_c_order(
    group: &hdf5::Group,
    key: &str,
    data: &Array2<f64>,
) -> Result<(), ChkfileError> {
    if group.link_exists(key) {
        group.unlink(key)?;
    }
    // Ensure C-contiguous (hdf5-metno `write` requires standard layout).
    let c_owned = data.as_standard_layout().to_owned();
    group
        .new_dataset::<f64>()
        .shape(c_owned.shape())
        .create(key)?
        .write(&c_owned)?;
    Ok(())
}

/// Write a 2D f64 dataset preserving F-order (column-major) on disk.
/// For `mo_coeff` per pyscf/scf/chkfile.py:28-42 — upstream writes
/// `mf.mo_coeff` which is F-order LAPACK output. Pitfall 8 mitigation.
///
/// Strategy: hdf5-metno's `write` requires C-contiguous memory; we write
/// the TRANSPOSE so the on-disk byte layout is column-major relative to the
/// original (nao, nmo) shape. Reading via h5py with explicit F-order yields
/// the original matrix; reading via `read_dataset_2d` (which interprets as
/// C-order) returns the transpose. This is the symmetric round-trip
/// convention asserted by ORACLE-08 (plan 03-08).
pub fn write_dataset_f_order(
    group: &hdf5::Group,
    key: &str,
    data: ArrayView2<f64>,
) -> Result<(), ChkfileError> {
    if group.link_exists(key) {
        group.unlink(key)?;
    }
    // Transposed C-contiguous copy: data.t() has reversed strides; .to_owned()
    // materialises a standard-layout array of shape (ncols, nrows).
    let transposed = data.t().as_standard_layout().to_owned();
    group
        .new_dataset::<f64>()
        .shape(transposed.shape())
        .create(key)?
        .write(&transposed)?;
    Ok(())
}

/// Read a 2D f64 dataset (caller handles F-order vs C-order interpretation).
pub fn read_dataset_2d(group: &hdf5::Group, key: &str) -> Result<Array2<f64>, ChkfileError> {
    Ok(group.dataset(key)?.read_2d()?)
}

// ---------------------------------------------------------------------------
// Complex datasets — the periodic schema (v2.0, plan 11-11).
// ---------------------------------------------------------------------------

/// h5py's on-disk representation of `numpy.complex128`: an HDF5 COMPOUND type
/// of two doubles named `r` and `i`.
///
/// Writing this type means `h5py.File(...)['scf/mo_coeff'][:]` reads back a
/// NumPy complex array with no conversion step, which is what an upstream
/// `pbc.scf` chkfile contains (`khf.KSCF.dump_chk`). D-05 keeps the
/// `hdf5-metno` dependency — and therefore the `H5Type` derive — inside this
/// crate; `pyscf-pbc-scf` consumes the two helpers below.
#[derive(hdf5_metno::H5Type, Clone, Copy, Debug, PartialEq, Default)]
#[repr(C)]
pub struct H5Complex {
    /// Real part.
    pub r: f64,
    /// Imaginary part.
    pub i: f64,
}

/// Write a rank-3 complex dataset in C-order.
///
/// # Errors
/// Propagates every HDF5 operation, and returns
/// [`ChkfileError::ShapeMismatch`] when `data.len() != shape.product()`.
pub fn write_dataset_3d_complex(
    group: &hdf5::Group,
    key: &str,
    shape: [usize; 3],
    data: &[H5Complex],
) -> Result<(), ChkfileError> {
    let n = shape[0] * shape[1] * shape[2];
    if data.len() != n {
        return Err(ChkfileError::ShapeMismatch {
            key: key.into(),
            expected: shape.to_vec(),
            actual: vec![data.len()],
        });
    }
    if group.link_exists(key) {
        group.unlink(key)?;
    }
    let arr =
        ndarray::ArrayView3::from_shape(shape, data).map_err(|_| ChkfileError::ShapeMismatch {
            key: key.into(),
            expected: shape.to_vec(),
            actual: vec![data.len()],
        })?;
    group
        .new_dataset::<H5Complex>()
        .shape(shape)
        .create(key)?
        .write(&arr)?;
    Ok(())
}

/// Read a rank-3 complex dataset written by [`write_dataset_3d_complex`] (or
/// by h5py as `complex128`).
///
/// # Errors
/// Propagates every HDF5 operation.
pub fn read_dataset_3d_complex(
    group: &hdf5::Group,
    key: &str,
) -> Result<([usize; 3], Vec<H5Complex>), ChkfileError> {
    let arr: ndarray::Array3<H5Complex> = group.dataset(key)?.read()?;
    let s = arr.shape();
    let shape = [s[0], s[1], s[2]];
    Ok((shape, arr.into_iter().collect()))
}
