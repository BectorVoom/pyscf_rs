//! GEOMOPT-05: HDF5 optimizer-state checkpoint (dump/load + resume).
//!
//! Persists the *live* optimizer state — enough to resume a partially-converged
//! optimization — to an HDF5 group: the current Cartesian geometry (Bohr), the
//! trust radius, the BFGS approximate Hessian (+ its update counter), the
//! accumulated internal-coordinate / internal-gradient history from the
//! previous step, and the step counter.
//!
//! ## Sole-owner HDF5 discipline (D-07 / GEOMOPT-05)
//!
//! This module does NOT add its own `hdf5-metno` dependency. It routes through
//! the [`pyscf_chkfile::hdf5`] re-exported alias (`pyscf-chkfile` is the *sole*
//! owner of `hdf5-metno`, D-05) and the `pyscf_chkfile::primitives` helpers, so
//! the encoding choices (scalar / 1D-f64 datasets) match the rest of the
//! checkpoint surface byte-for-byte. The dump/load round-trip shape mirrors
//! `crates/pyscf-scf/src/chkfile.rs` (`dump`/`load` over an `hdf5::Group`).
//!
//! ## Schema (`/opt_state` group)
//!
//! ```text
//!   /opt_state/schema_version  (scalar f64; T-07-19 version guard)
//!   /opt_state/nint            (scalar f64, the internal-coordinate count)
//!   /opt_state/natm            (scalar f64, the atom count)
//!   /opt_state/n_updates       (scalar f64, the BFGS update counter)
//!   /opt_state/step            (scalar f64, the step counter)
//!   /opt_state/trust           (scalar f64, the trust radius, Bohr)
//!   /opt_state/e_tot           (scalar f64, the last total energy)
//!   /opt_state/coords          (1D f64, length 3*natm — row-major [x,y,z]…)
//!   /opt_state/hessian         (1D f64, length nint*nint — row-major)
//!   /opt_state/prev_q          (1D f64, length nint — present iff has_prev)
//!   /opt_state/prev_g_int      (1D f64, length nint — present iff has_prev)
//!   /opt_state/prev_e          (scalar f64 — present iff has_prev)
//!   /opt_state/has_prev        (scalar f64, 1.0 iff the prev-step history exists)
//! ```
//!
//! ## Corrupt-checkpoint guard (T-07-19)
//!
//! A loaded state whose shapes are mutually inconsistent (e.g. a `hessian`
//! whose length ≠ `nint·nint`, or `coords` whose length ≠ `3·natm`) is rejected
//! with a clear [`GeomError`] on `dump`/`load` — never silently resumed from
//! garbage.

use crate::error::GeomError;
use pyscf_chkfile::hdf5;
use pyscf_chkfile::primitives;

/// The on-disk schema version (bumped if the layout changes; T-07-19 guard).
pub const CHECKPOINT_SCHEMA_VERSION: f64 = 1.0;

/// The live optimizer state required to resume a partially-converged run.
///
/// This is the exact set of mutable quantities the `optimize` outer loop
/// carries across iterations (see `crate::optimize`): the geometry, the trust
/// radius, the BFGS Hessian (+ counter), the previous-step internal-coordinate
/// / internal-gradient / energy history (for the BFGS update + trust-radius
/// quality factor), the step counter, and the last total energy.
#[derive(Debug, Clone)]
pub struct OptimizerState {
    /// The current Cartesian geometry `(natm, 3)` in Bohr.
    pub coords: Vec<[f64; 3]>,
    /// The current trust radius (Bohr).
    pub trust: f64,
    /// The BFGS approximate Hessian, row-major `(nint, nint)`.
    pub hessian: Vec<f64>,
    /// The internal-coordinate dimension (`nint = hessian.len().sqrt()`).
    pub nint: usize,
    /// The number of BFGS updates applied so far (capped at `MAX_BFGS_UPDATES`).
    pub n_updates: usize,
    /// The number of optimizer steps already taken.
    pub step: usize,
    /// The previous step's internal-coordinate values `q` (for the BFGS update).
    pub prev_q: Option<Vec<f64>>,
    /// The previous step's internal gradient `g_int` (for the BFGS update).
    pub prev_g_int: Option<Vec<f64>>,
    /// The previous step's total energy (for the trust-radius quality factor).
    pub prev_e: Option<f64>,
    /// The last total energy (Hartree).
    pub e_tot: f64,
}

impl OptimizerState {
    /// Validate the internal shape invariants (T-07-19): the Hessian must be
    /// `nint·nint`, every `prev_*` vector (when present) must be length `nint`,
    /// and the prev-step trio must be all-present or all-absent.
    fn validate(&self) -> Result<(), GeomError> {
        let natm = self.coords.len();
        if self.hessian.len() != self.nint * self.nint {
            return Err(GeomError::CheckpointCorrupt {
                what: format!(
                    "hessian length {} ≠ nint·nint = {}",
                    self.hessian.len(),
                    self.nint * self.nint
                ),
            });
        }
        let has_prev = self.prev_q.is_some();
        if self.prev_g_int.is_some() != has_prev || self.prev_e.is_some() != has_prev {
            return Err(GeomError::CheckpointCorrupt {
                what: "prev-step history (prev_q / prev_g_int / prev_e) must be all-present or all-absent".into(),
            });
        }
        if let Some(q) = &self.prev_q
            && q.len() != self.nint
        {
            return Err(GeomError::CheckpointCorrupt {
                what: format!("prev_q length {} ≠ nint {}", q.len(), self.nint),
            });
        }
        if let Some(g) = &self.prev_g_int
            && g.len() != self.nint
        {
            return Err(GeomError::CheckpointCorrupt {
                what: format!("prev_g_int length {} ≠ nint {}", g.len(), self.nint),
            });
        }
        let _ = natm; // coords length is recomputed on load; no fixed expectation here
        Ok(())
    }

    /// Dump the optimizer state to an HDF5 group via the `pyscf-chkfile` alias.
    ///
    /// Mirrors `pyscf_scf::chkfile::ScfResult::dump` (a `&hdf5::Group` consumer
    /// using the `pyscf_chkfile::primitives` scalar/1D helpers).
    ///
    /// # Errors
    /// - [`GeomError::CheckpointCorrupt`] if the state's shapes are inconsistent
    ///   (validated BEFORE any write so a malformed state is never persisted).
    /// - [`GeomError::Chkfile`] if an underlying HDF5 write fails.
    pub fn dump(&self, group: &hdf5::Group) -> Result<(), GeomError> {
        self.validate()?;

        primitives::write_scalar_f64(group, "schema_version", CHECKPOINT_SCHEMA_VERSION)?;
        primitives::write_scalar_f64(group, "nint", self.nint as f64)?;
        primitives::write_scalar_f64(group, "natm", self.coords.len() as f64)?;
        primitives::write_scalar_f64(group, "n_updates", self.n_updates as f64)?;
        primitives::write_scalar_f64(group, "step", self.step as f64)?;
        primitives::write_scalar_f64(group, "trust", self.trust)?;
        primitives::write_scalar_f64(group, "e_tot", self.e_tot)?;

        // Flatten the (natm, 3) geometry to a 1D row-major [x,y,z]… buffer.
        let mut coords_flat = Vec::with_capacity(3 * self.coords.len());
        for c in &self.coords {
            coords_flat.push(c[0]);
            coords_flat.push(c[1]);
            coords_flat.push(c[2]);
        }
        primitives::write_dataset_1d(group, "coords", &coords_flat)?;
        primitives::write_dataset_1d(group, "hessian", &self.hessian)?;

        let has_prev = self.prev_q.is_some();
        primitives::write_scalar_f64(group, "has_prev", if has_prev { 1.0 } else { 0.0 })?;
        if has_prev {
            // validate() guarantees the trio is all-present + length nint.
            let q = self.prev_q.as_ref().expect("prev_q present (validated)");
            let g = self.prev_g_int.as_ref().expect("prev_g_int present (validated)");
            let e = self.prev_e.expect("prev_e present (validated)");
            primitives::write_dataset_1d(group, "prev_q", q)?;
            primitives::write_dataset_1d(group, "prev_g_int", g)?;
            primitives::write_scalar_f64(group, "prev_e", e)?;
        }
        Ok(())
    }

    /// Load the optimizer state back from an HDF5 group, validating the shape
    /// invariants on the way in (T-07-19: a corrupt/inconsistent checkpoint
    /// fails cleanly rather than resuming from garbage).
    ///
    /// # Errors
    /// - [`GeomError::CheckpointCorrupt`] if the schema version is unknown or
    ///   the loaded shapes are mutually inconsistent.
    /// - [`GeomError::Chkfile`] if an underlying HDF5 read fails.
    pub fn load(group: &hdf5::Group) -> Result<Self, GeomError> {
        let version = primitives::read_scalar_f64(group, "schema_version")?;
        if version != CHECKPOINT_SCHEMA_VERSION {
            return Err(GeomError::CheckpointCorrupt {
                what: format!(
                    "unknown checkpoint schema_version {version} (expected {CHECKPOINT_SCHEMA_VERSION})"
                ),
            });
        }
        let nint = primitives::read_scalar_f64(group, "nint")? as usize;
        let natm = primitives::read_scalar_f64(group, "natm")? as usize;
        let n_updates = primitives::read_scalar_f64(group, "n_updates")? as usize;
        let step = primitives::read_scalar_f64(group, "step")? as usize;
        let trust = primitives::read_scalar_f64(group, "trust")?;
        let e_tot = primitives::read_scalar_f64(group, "e_tot")?;

        let coords_flat = primitives::read_dataset_1d(group, "coords")?;
        if coords_flat.len() != 3 * natm {
            return Err(GeomError::CheckpointCorrupt {
                what: format!("coords length {} ≠ 3·natm = {}", coords_flat.len(), 3 * natm),
            });
        }
        let coords: Vec<[f64; 3]> = coords_flat
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();

        let hessian = primitives::read_dataset_1d(group, "hessian")?;

        let has_prev = primitives::read_scalar_f64(group, "has_prev")? != 0.0;
        let (prev_q, prev_g_int, prev_e) = if has_prev {
            (
                Some(primitives::read_dataset_1d(group, "prev_q")?),
                Some(primitives::read_dataset_1d(group, "prev_g_int")?),
                Some(primitives::read_scalar_f64(group, "prev_e")?),
            )
        } else {
            (None, None, None)
        };

        let state = OptimizerState {
            coords,
            trust,
            hessian,
            nint,
            n_updates,
            step,
            prev_q,
            prev_g_int,
            prev_e,
            e_tot,
        };
        // Re-validate the assembled state (catches hessian / prev_* mismatches).
        state.validate()?;
        Ok(state)
    }
}
