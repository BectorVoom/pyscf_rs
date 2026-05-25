//! pyscf-dft::chkfile — per-method chkfile schema for KS-DFT (D-06).
//!
//! Source: `pyscf/dft/rks.py` (the KS result attributes) + `pyscf/scf/chkfile.py`
//! :25-42 (`dump_scf`/`load_scf` — the SCF base schema KS extends) + the EXACT
//! Phase 3 analog `pyscf-scf::chkfile` (chkfile.rs:21-123, `impl Checkpointable
//! for ScfResult`). Plan 04-08 Task 2 ships `impl Checkpointable for KsResult`.
//!
//! Schema (the `/scf` group, upstream-compatible, PLUS the DFT metadata):
//!   /mol            (VL Unicode JSON from `mol.dumps()` — written at root)
//!   /scf/e_tot      (scalar f64)
//!   /scf/mo_energy  (1D f64, length nao)
//!   /scf/mo_occ     (1D f64, length nao)
//!   /scf/mo_coeff   (2D f64, F-order on disk, logical shape [nao, nmo])
//!   /scf/xc         (VL Unicode — the XC functional string, DFT metadata)
//!   /scf/grids_level  (scalar f64 — the Becke grid density level, DFT metadata)
//!   /scf/grids_scheme (VL Unicode — the prune scheme name, DFT metadata)
//!
//! Per D-06: each per-method module owns its own schema. The KS schema reuses
//! the upstream `/scf` group verbatim (so a PySCF `mf.from_chk()` on a DF/KS
//! chkfile recovers the SCF block) and ADDS the `xc` + `grids` metadata KS
//! needs to reconstruct the functional + integration grid on load.
//!
//! Pitfall 8: `mo_coeff` is F-order in memory (`MOCoefficients.data`) and
//! F-order on disk via `pyscf_chkfile::primitives::write_dataset_f_order`.
//!
//! ## D-05 sole-owner discipline
//! This module uses `pyscf_chkfile::primitives::*` (and the re-exported
//! `pyscf_chkfile::hdf5` alias for the VL-Unicode `xc`/`grids_scheme` strings)
//! — pyscf-dft adds **no** `hdf5-metno` dependency of its own (the algebra-wall
//! analog: pyscf-chkfile is the sole owner of the hdf5-metno crate).
//!
//! ## Tampering boundary (T-04-08)
//! `KsResult::load` deserializes a (possibly untrusted) checkpoint file: every
//! dataset is read via the Phase 3 primitives, which validate dataset
//! shapes/dims before allocation and return `ChkfileError` (never panic) on a
//! malformed file — reusing the `ScfResult::load` validation discipline.

use ndarray::{Array2, ArrayView2, ShapeBuilder};
use std::str::FromStr;

use pyscf_chkfile::hdf5;
use pyscf_chkfile::hdf5::types::VarLenUnicode;
use pyscf_chkfile::{Checkpointable, ChkfileError, primitives};
use pyscf_core::{Energy, MOCoefficients};
use pyscf_scf::ScfResult;

/// The grid metadata persisted alongside a KS result — the (already-built)
/// Becke integration grid's control knobs (density level + prune scheme name)
/// so a loader can reconstruct an equivalent `Grids` (`pyscf-grids`).
#[derive(Debug, Clone, PartialEq)]
pub struct GridsMeta {
    /// The Becke grid density level (`Grids::level`, 0..=9; class default 3).
    pub level: usize,
    /// The prune-scheme name (`Grids::prune` — `"nwchem"`/`"sg1"`/`"treutler"`/
    /// `"none"`). Stored as a string so the on-disk schema is stable across
    /// enum-variant additions.
    pub scheme: String,
}

impl Default for GridsMeta {
    fn default() -> Self {
        // The upstream `Grids` class defaults (gen_grid.py:490-499 / pyscf-grids
        // `Grids::default`): level 3, nwchem prune.
        Self {
            level: 3,
            scheme: "nwchem".to_string(),
        }
    }
}

/// A Kohn-Sham SCF result + the DFT-specific metadata the chkfile persists.
///
/// Wraps the Phase 3 `ScfResult` (e_tot / mo_coeff / mo_energy / mo_occ /
/// converged / cycles) — so the on-disk `/scf` group is byte-identical to the
/// SCF schema (upstream `mf.from_chk` compatibility) — and ADDS the KS
/// `xc` functional string + the `grids` metadata (DFT-07 chkfile / D-06).
#[derive(Debug, Clone)]
pub struct KsResult {
    /// The underlying SCF result (the `/scf` group payload).
    pub scf: ScfResult,
    /// The XC functional string (e.g. `"svwn"`, `"pbe"`, `"b3lyp"`).
    pub xc: String,
    /// The integration-grid metadata (level + prune scheme).
    pub grids: GridsMeta,
}

impl KsResult {
    /// Build a `KsResult` from an `ScfResult` + the KS functional + grid meta.
    pub fn new(scf: ScfResult, xc: impl Into<String>, grids: GridsMeta) -> Self {
        Self {
            scf,
            xc: xc.into(),
            grids,
        }
    }
}

/// Write a VL-Unicode string under `group/key` (the `xc` + `grids_scheme`
/// metadata). Group-level analog of `primitives::write_mol` (which is
/// file-root only) — uses the re-exported `pyscf_chkfile::hdf5` alias so
/// pyscf-dft adds no hdf5-metno dep (D-05). Idempotent (unlinks first).
fn write_string(group: &hdf5::Group, key: &str, value: &str) -> Result<(), ChkfileError> {
    let vl = VarLenUnicode::from_str(value).map_err(|_| ChkfileError::InvalidUtf8)?;
    if group.link_exists(key) {
        group.unlink(key)?;
    }
    group
        .new_dataset::<VarLenUnicode>()
        .create(key)?
        .write_scalar(&vl)?;
    Ok(())
}

/// Read a VL-Unicode string from `group/key`. Returns `ChkfileError` on a
/// missing/malformed dataset (never panics — T-04-08).
fn read_string(group: &hdf5::Group, key: &str) -> Result<String, ChkfileError> {
    let vl: VarLenUnicode = group.dataset(key)?.read_scalar()?;
    Ok(vl.as_str().to_string())
}

impl Checkpointable for KsResult {
    fn dump(&self, scf_group: &hdf5::Group) -> Result<(), ChkfileError> {
        // ── The upstream /scf block (byte-identical to ScfResult::dump) ────
        primitives::write_scalar_f64(scf_group, "e_tot", self.scf.e_tot.0)?;
        primitives::write_dataset_1d(scf_group, "mo_energy", &self.scf.mo_energy)?;
        primitives::write_dataset_1d(scf_group, "mo_occ", &self.scf.mo_occ)?;
        // mo_coeff: F-order on disk (Pitfall 8). MOCoefficients.data is F-order:
        // data[i + j*nao] = C[i, j]; build a [nao, nmo] view with F-strides so
        // write_dataset_f_order's transposition yields a [nmo, nao] C-contig
        // payload — identical to ScfResult::dump (chkfile.rs:26-50).
        let nao = self.scf.mo_coeff.nao;
        let nmo = self.scf.mo_coeff.nmo;
        if self.scf.mo_coeff.data.len() != nao * nmo {
            return Err(ChkfileError::ShapeMismatch {
                key: "mo_coeff".into(),
                expected: vec![nao, nmo],
                actual: vec![self.scf.mo_coeff.data.len()],
            });
        }
        let view: ArrayView2<f64> =
            ArrayView2::from_shape((nao, nmo).strides((1, nao)), &self.scf.mo_coeff.data).map_err(
                |_| ChkfileError::ShapeMismatch {
                    key: "mo_coeff".into(),
                    expected: vec![nao, nmo],
                    actual: vec![self.scf.mo_coeff.data.len()],
                },
            )?;
        primitives::write_dataset_f_order(scf_group, "mo_coeff", view)?;

        // ── The DFT metadata (xc + grids — DFT-07 chkfile) ─────────────────
        write_string(scf_group, "xc", &self.xc)?;
        primitives::write_scalar_f64(scf_group, "grids_level", self.grids.level as f64)?;
        write_string(scf_group, "grids_scheme", &self.grids.scheme)?;
        Ok(())
    }

    fn load(scf_group: &hdf5::Group) -> Result<Self, ChkfileError> {
        // ── The upstream /scf block (identical to ScfResult::load) ─────────
        let e_tot = Energy(primitives::read_scalar_f64(scf_group, "e_tot")?);
        let mo_energy = primitives::read_dataset_1d(scf_group, "mo_energy")?;
        let mo_occ = primitives::read_dataset_1d(scf_group, "mo_occ")?;
        // On disk mo_coeff was stored as the transpose: [nmo, nao] C-contig.
        // read_dataset_2d (which validates the dataset shape before allocation,
        // T-04-08) gives the [nmo, nao] matrix; reconstruct the F-order data
        // (data[i + j*nao] = C[i,j] = mat_on_disk[j, i]). Identical to
        // ScfResult::load (chkfile.rs:54-95).
        let mat_on_disk: Array2<f64> = primitives::read_dataset_2d(scf_group, "mo_coeff")?;
        let nmo = mat_on_disk.shape()[0];
        let nao = mat_on_disk.shape()[1];
        let mut data = Vec::with_capacity(nao * nmo);
        for j in 0..nmo {
            for i in 0..nao {
                data.push(mat_on_disk[(j, i)]);
            }
        }
        let scf = ScfResult {
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
            // chkfile is only written on converged SCF (pyscf/scf/hf.py:2056);
            // loading treats the prior result as converged. Upstream schema
            // does not store the iteration count → preserve 0.
            converged: true,
            cycles: 0,
        };

        // ── The DFT metadata (xc + grids) ──────────────────────────────────
        let xc = read_string(scf_group, "xc")?;
        let level = primitives::read_scalar_f64(scf_group, "grids_level")? as usize;
        let scheme = read_string(scf_group, "grids_scheme")?;

        Ok(KsResult {
            scf,
            xc,
            grids: GridsMeta { level, scheme },
        })
    }
}

/// Top-level helper: write an entire `KsResult` + mol JSON to a chkfile at
/// `path`. Mirrors `pyscf-scf::chkfile::dump_scf_to_file` (chkfile.rs:101-116)
/// — the `/scf` group + the `/mol` root JSON, plus the KS xc/grids metadata
/// inside `/scf`.
pub fn dump_ks_to_file(
    path: &std::path::Path,
    mol_json: &str,
    result: &KsResult,
) -> Result<(), ChkfileError> {
    let f = primitives::open_for_write(path)?;
    primitives::write_mol(&f, mol_json)?;
    if f.link_exists("scf") {
        f.unlink("scf")?;
    }
    let scf = f.create_group("scf")?;
    result.dump(&scf)?;
    f.flush()?;
    Ok(())
}

/// Read a `KsResult` back from a chkfile. Mirrors
/// `pyscf-scf::chkfile::load_scf_from_file` (chkfile.rs:119-123).
pub fn load_ks_from_file(path: &std::path::Path) -> Result<KsResult, ChkfileError> {
    let f = primitives::open_for_read(path)?;
    let scf = primitives::read_group(&f, "scf")?;
    KsResult::load(&scf)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_ks_result() -> KsResult {
        let nao = 4;
        let nmo = 4;
        // F-order flat data: data[i + j*nao] for (i, j).
        let mut data = vec![0.0_f64; nao * nmo];
        for j in 0..nmo {
            for i in 0..nao {
                data[i + j * nao] = ((i * 3 + j + 1) as f64) * 0.1;
            }
        }
        let scf = ScfResult {
            e_tot: Energy(-76.319_876_f64),
            mo_coeff: MOCoefficients {
                nao,
                nmo,
                data: data.clone(),
                energies: vec![-18.74, -0.92, -0.47, -0.39],
                occupations: vec![2.0, 2.0, 2.0, 2.0],
            },
            mo_energy: vec![-18.74, -0.92, -0.47, -0.39],
            mo_occ: vec![2.0, 2.0, 2.0, 2.0],
            converged: true,
            cycles: 9,
        };
        KsResult::new(
            scf,
            "b3lyp",
            GridsMeta {
                level: 3,
                scheme: "nwchem".to_string(),
            },
        )
    }

    /// KsResult dump → load round-trips identically (e_tot, mo_*, xc, grids).
    #[test]
    fn ks_result_round_trips() {
        let original = sample_ks_result();
        let tmp = tempfile::NamedTempFile::new().expect("tmp");
        let path = tmp.path();
        dump_ks_to_file(path, r#"{"atom":"O 0 0 0","basis":"cc-pvdz"}"#, &original).expect("dump");
        let loaded = load_ks_from_file(path).expect("load");

        assert!((loaded.scf.e_tot.0 - original.scf.e_tot.0).abs() < 1e-12);
        assert_eq!(loaded.xc, original.xc);
        assert_eq!(loaded.grids, original.grids);
        assert_eq!(loaded.scf.mo_coeff.nao, original.scf.mo_coeff.nao);
        assert_eq!(loaded.scf.mo_coeff.nmo, original.scf.mo_coeff.nmo);
        for (k, (a, b)) in loaded
            .scf
            .mo_coeff
            .data
            .iter()
            .zip(&original.scf.mo_coeff.data)
            .enumerate()
        {
            assert!((a - b).abs() < 1e-12, "mo_coeff[{k}] mismatch: {a} vs {b}");
        }
        for (a, b) in loaded.scf.mo_energy.iter().zip(&original.scf.mo_energy) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    /// The schema includes the upstream /scf keys PLUS xc + grids metadata.
    #[test]
    fn schema_has_scf_keys_and_dft_metadata() {
        let tmp = tempfile::NamedTempFile::new().expect("tmp");
        let path = tmp.path();
        dump_ks_to_file(path, "{}", &sample_ks_result()).expect("dump");
        let f = primitives::open_for_read(path).expect("reopen");
        assert!(f.link_exists("mol"), "/mol missing");
        let scf = f.group("scf").expect("scf group");
        // Upstream SCF keys.
        assert!(scf.link_exists("e_tot"), "/scf/e_tot missing");
        assert!(scf.link_exists("mo_energy"), "/scf/mo_energy missing");
        assert!(scf.link_exists("mo_occ"), "/scf/mo_occ missing");
        assert!(scf.link_exists("mo_coeff"), "/scf/mo_coeff missing");
        // DFT metadata keys (DFT-07 chkfile).
        assert!(scf.link_exists("xc"), "/scf/xc missing");
        assert!(scf.link_exists("grids_level"), "/scf/grids_level missing");
        assert!(scf.link_exists("grids_scheme"), "/scf/grids_scheme missing");
    }

    /// `KsResult::load` returns a `ChkfileError` (never panics) on a chkfile
    /// missing the DFT metadata — the T-04-08 tampering-boundary discipline.
    #[test]
    fn load_missing_metadata_errors_not_panics() {
        let tmp = tempfile::NamedTempFile::new().expect("tmp");
        let path = tmp.path();
        // Write only the SCF block (no xc/grids) by dumping an ScfResult.
        {
            let f = primitives::open_for_write(path).expect("create");
            primitives::write_mol(&f, "{}").expect("mol");
            let g = f.create_group("scf").expect("group");
            sample_ks_result().scf.dump(&g).expect("scf dump");
            f.flush().expect("flush");
        }
        // Loading as a KsResult must fail cleanly (missing /scf/xc), not panic.
        let res = load_ks_from_file(path);
        assert!(res.is_err(), "load of metadata-less chkfile must error");
    }
}
