//! Periodic SCF checkpoint files — plan 11-11, port of
//! `pyscf/pbc/scf/chkfile.py` and `khf.KSCF.dump_chk` (`khf.py:702-720`).
//!
//! # Schema
//!
//! Upstream's periodic chkfile is the molecular one plus the k-point list:
//!
//! ```text
//! /mol              cell JSON (from `cell.dumps()`)
//! /scf/e_tot        scalar f64
//! /scf/kpts         (nkpts, 3) f64          <- khf.py:713
//! /scf/mo_energy    (nsets*nkpts, nmo) f64
//! /scf/mo_occ       (nsets*nkpts, nmo) f64
//! /scf/mo_coeff     (nsets*nkpts, nmo, nao) COMPLEX
//! ```
//!
//! `mo_coeff` is genuinely complex, so it is written as an HDF5 COMPOUND type
//! with fields `r` and `i` — which is exactly h5py's own `complex128`
//! representation, so `h5py.File(...)['scf/mo_coeff'][:]` reads it back as a
//! NumPy complex array without any conversion. The `(nmo, nao)` ordering of the
//! last two axes is upstream's F-order `mo_coeff` written through the same
//! transpose convention `pyscf_chkfile::write_dataset_f_order` uses for the
//! molecular case (Pitfall 8).

use ndarray::Array2;
use pyscf_algebra::CTensor;
use pyscf_chkfile::{ChkfileError, H5Complex, primitives};

use crate::types::KScfResult;

/// Write a periodic SCF result to `path`.
///
/// `cell_json` is `cell.dumps()`; pass an empty string to omit `/mol`.
///
/// # Errors
/// Propagates every HDF5 operation.
pub fn dump_kscf_to_file(
    path: &std::path::Path,
    result: &KScfResult,
    kpts: &[[f64; 3]],
    nao: usize,
    cell_json: &str,
) -> Result<(), ChkfileError> {
    let file = primitives::open_for_write(path)?;
    if !cell_json.is_empty() {
        primitives::write_mol(&file, cell_json)?;
    }
    let scf = if file.link_exists("scf") {
        file.group("scf")?
    } else {
        file.create_group("scf")?
    };
    primitives::write_scalar_f64(&scf, "e_tot", result.e_tot)?;
    primitives::write_scalar_f64(&scf, "e_nuc", result.e_nuc)?;

    // scf/kpts — khf.py:713.
    let mut kp = Array2::<f64>::zeros((kpts.len(), 3));
    for (k, v) in kpts.iter().enumerate() {
        for j in 0..3 {
            kp[(k, j)] = v[j];
        }
    }
    primitives::write_dataset_c_order(&scf, "kpts", &kp)?;

    let nblocks = result.mo_energy.len();
    let nmo = result.mo_energy.first().map_or(0, Vec::len);
    let mut e = Array2::<f64>::zeros((nblocks, nmo));
    let mut o = Array2::<f64>::zeros((nblocks, nmo));
    for b in 0..nblocks {
        for i in 0..nmo {
            // `zeigh_gen` pads dropped linear dependencies with +inf, which
            // HDF5 stores faithfully but NumPy comparisons choke on; upstream
            // never produces them because LAPACK's `eigh` keeps every root.
            let v = result.mo_energy[b][i];
            e[(b, i)] = if v.is_finite() { v } else { 0.0 };
            o[(b, i)] = result.mo_occ[b][i];
        }
    }
    primitives::write_dataset_c_order(&scf, "mo_energy", &e)?;
    primitives::write_dataset_c_order(&scf, "mo_occ", &o)?;

    // mo_coeff: (nblocks, nmo, nao) — the transpose of the COLUMN-MAJOR
    // `nao x nmo` blocks, so a reader that interprets the last two axes in
    // F-order recovers `C[ao, mo]`.
    let mut c = vec![H5Complex::default(); nblocks * nmo * nao];
    for (b, m) in result.mo_coeff.iter().enumerate() {
        for mo in 0..nmo {
            for ao in 0..nao {
                let p = ao + mo * nao;
                c[(b * nmo + mo) * nao + ao] = H5Complex {
                    r: m.re[p],
                    i: m.im[p],
                };
            }
        }
    }
    primitives::write_dataset_3d_complex(&scf, "mo_coeff", [nblocks, nmo, nao], &c)?;
    Ok(())
}

/// What [`load_kscf_from_file`] recovers.
#[derive(Debug, Clone)]
pub struct KScfCheckpoint {
    /// Total energy.
    pub e_tot: f64,
    /// The k-points the calculation used.
    pub kpts: Vec<[f64; 3]>,
    /// `mo_energy[block]`.
    pub mo_energy: Vec<Vec<f64>>,
    /// `mo_occ[block]`.
    pub mo_occ: Vec<Vec<f64>>,
    /// `mo_coeff[block]`, COLUMN-MAJOR `nao x nmo`.
    pub mo_coeff: Vec<CTensor>,
    /// AO count.
    pub nao: usize,
}

/// Read a periodic SCF checkpoint back.
///
/// # Errors
/// Propagates every HDF5 operation.
pub fn load_kscf_from_file(
    path: &std::path::Path,
) -> Result<KScfCheckpoint, ChkfileError> {
    let file = primitives::open_for_read(path)?;
    let scf = primitives::read_group(&file, "scf")?;
    let e_tot = primitives::read_scalar_f64(&scf, "e_tot")?;
    let kp: Array2<f64> = primitives::read_dataset_2d(&scf, "kpts")?;
    let kpts = (0..kp.shape()[0])
        .map(|k| [kp[(k, 0)], kp[(k, 1)], kp[(k, 2)]])
        .collect();
    let e: Array2<f64> = primitives::read_dataset_2d(&scf, "mo_energy")?;
    let o: Array2<f64> = primitives::read_dataset_2d(&scf, "mo_occ")?;
    let (shape, c) = primitives::read_dataset_3d_complex(&scf, "mo_coeff")?;
    let (nblocks, nmo, nao) = (shape[0], shape[1], shape[2]);

    let mut mo_coeff = Vec::with_capacity(nblocks);
    for b in 0..nblocks {
        let mut re = vec![0.0_f64; nao * nmo];
        let mut im = vec![0.0_f64; nao * nmo];
        for mo in 0..nmo {
            for ao in 0..nao {
                let v = c[(b * nmo + mo) * nao + ao];
                re[ao + mo * nao] = v.r;
                im[ao + mo * nao] = v.i;
            }
        }
        mo_coeff.push(CTensor::from_planes(re, im));
    }
    Ok(KScfCheckpoint {
        e_tot,
        kpts,
        mo_energy: (0..nblocks)
            .map(|b| (0..e.shape()[1]).map(|i| e[(b, i)]).collect())
            .collect(),
        mo_occ: (0..nblocks)
            .map(|b| (0..o.shape()[1]).map(|i| o[(b, i)]).collect())
            .collect(),
        mo_coeff,
        nao,
    })
}
