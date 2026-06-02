//! pyscf-scf::chkfile — per-method chkfile schema for SCF.
//!
//! Source: pyscf/scf/chkfile.py:25-42 `dump_scf` / `load_scf`. Schema:
//!   /mol            (VL Unicode JSON from `mol.dumps()` — written at root)
//!   /scf/e_tot      (scalar f64)
//!   /scf/mo_energy  (1D f64, length nao)
//!   /scf/mo_occ     (1D f64, length nao)
//!   /scf/mo_coeff   (2D f64, F-order on disk, logical shape [nao, nmo])
//!
//! Per D-06: per-method module owns its own schema. Phase 4-7 add their
//! own `Checkpointable` impls (`KsResult`, `CcsdResult`, `OptimState`).
//!
//! Pitfall 8: `mo_coeff` is F-order in memory (`MOCoefficients.data`) and
//! F-order on disk via `pyscf_chkfile::write_dataset_f_order`.
use crate::ScfResult;
use ndarray::{Array2, ArrayView2, ShapeBuilder};
use pyscf_chkfile::hdf5;
use pyscf_chkfile::{Checkpointable, ChkfileError, primitives};
use pyscf_core::{Energy, MOCoefficients};

impl Checkpointable for ScfResult {
    fn dump(&self, scf_group: &hdf5::Group) -> Result<(), ChkfileError> {
        primitives::write_scalar_f64(scf_group, "e_tot", self.e_tot.0)?;
        primitives::write_dataset_1d(scf_group, "mo_energy", &self.mo_energy)?;
        primitives::write_dataset_1d(scf_group, "mo_occ", &self.mo_occ)?;
        // mo_coeff: build an ArrayView2 over the F-order buffer.
        // MOCoefficients.data is F-order: data[i + j * nao] = C[i, j].
        // We construct a [nao, nmo] view with explicit F-strides so the
        // write_dataset_f_order transposition produces a correctly-shaped
        // [nmo, nao] C-contig payload on disk.
        let nao = self.mo_coeff.nao;
        let nmo = self.mo_coeff.nmo;
        if self.mo_coeff.data.len() != nao * nmo {
            return Err(ChkfileError::ShapeMismatch {
                key: "mo_coeff".into(),
                expected: vec![nao, nmo],
                actual: vec![self.mo_coeff.data.len()],
            });
        }
        // F-order strides: (1, nao) for shape [nao, nmo].
        let view: ArrayView2<f64> =
            ArrayView2::from_shape((nao, nmo).strides((1, nao)), &self.mo_coeff.data).map_err(
                |_| ChkfileError::ShapeMismatch {
                    key: "mo_coeff".into(),
                    expected: vec![nao, nmo],
                    actual: vec![self.mo_coeff.data.len()],
                },
            )?;
        primitives::write_dataset_f_order(scf_group, "mo_coeff", view)?;
        Ok(())
    }

    fn load(scf_group: &hdf5::Group) -> Result<Self, ChkfileError> {
        let e_tot = Energy(primitives::read_scalar_f64(scf_group, "e_tot")?);
        let mo_energy = primitives::read_dataset_1d(scf_group, "mo_energy")?;
        let mo_occ = primitives::read_dataset_1d(scf_group, "mo_occ")?;
        // On disk we stored the transpose: shape was [nmo, nao] C-contig.
        // Reading via read_2d gives an Array2 of shape [nmo, nao] in C-order.
        // To reconstruct the F-order MOCoefficients.data (flat indexed as
        // data[i + j * nao] = C[i, j]), we read back the [nmo, nao] matrix M
        // and store data[i + j*nao] = M[j, i] (M is C-order so M[j, i] is the
        // (j, i) element of the transpose-of-original, which is the original).
        let mat_on_disk: Array2<f64> = primitives::read_dataset_2d(scf_group, "mo_coeff")?;
        // mat_on_disk shape: [disk_rows, disk_cols] where disk_rows = nmo,
        // disk_cols = nao (because we wrote the transpose).
        let disk_rows = mat_on_disk.shape()[0];
        let disk_cols = mat_on_disk.shape()[1];
        let nmo = disk_rows;
        let nao = disk_cols;
        let mut data = Vec::with_capacity(nao * nmo);
        for j in 0..nmo {
            for i in 0..nao {
                // F-order layout: data[i + j*nao] = C[i, j].
                // C[i, j] in original orientation = mat_on_disk[j, i].
                data.push(mat_on_disk[(j, i)]);
            }
        }
        Ok(ScfResult {
            e_tot,
            mo_coeff: MOCoefficients {
                nao,
                nmo,
                data,
                energies: mo_energy.clone(),
                occupations: mo_occ.clone(),
            },
            mo_energy,
            mo_occ,
            // chkfile is only written on converged SCF per upstream convention
            // (pyscf/scf/hf.py:2056). Loading treats prior result as converged.
            converged: true,
            // upstream schema does not store iteration count; preserve 0.
            cycles: 0,
        })
    }
}

/// Top-level helper: write entire ScfResult + mol JSON to a chkfile at `path`.
/// Mirrors upstream `pyscf.lib.chkfile.save` + `pyscf.scf.chkfile.dump_scf`.
pub fn dump_scf_to_file(
    path: &std::path::Path,
    mol_json: &str,
    result: &ScfResult,
) -> Result<(), ChkfileError> {
    let f = primitives::open_for_write(path)?;
    primitives::write_mol(&f, mol_json)?;
    // Recreate /scf group if it exists (idempotent dump).
    if f.link_exists("scf") {
        f.unlink("scf")?;
    }
    let scf = f.create_group("scf")?;
    result.dump(&scf)?;
    f.flush()?;
    Ok(())
}

/// Read an ScfResult back from a chkfile.
pub fn load_scf_from_file(path: &std::path::Path) -> Result<ScfResult, ChkfileError> {
    let f = primitives::open_for_read(path)?;
    let scf = primitives::read_group(&f, "scf")?;
    ScfResult::load(&scf)
}

/// Read just the serialized `mol` JSON (written by `dump_scf_to_file` via
/// `mol.dumps()`) back from a chkfile. Used by the cross-basis init-guess
/// projection (`init_guess_by_chkfile`) to reconstruct the *prior*
/// molecule/basis so the cross-overlap `<current|prior>` can be formed.
pub fn load_mol_json_from_file(path: &std::path::Path) -> Result<String, ChkfileError> {
    let f = primitives::open_for_read(path)?;
    primitives::read_mol(&f)
}
