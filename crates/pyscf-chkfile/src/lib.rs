//! pyscf-chkfile: HDF5 chkfile primitives + Checkpointable trait.
//!
//! Source: D-06 — Phase 3 introduces this crate as the sole owner of the
//! `hdf5-metno` dependency, mirroring the algebra-wall discipline. The crate
//! exposes (Phase 3 plan 06) a `Checkpointable` trait that per-method modules
//! impl on their result type, plus HDF5 primitives that wrap the upstream
//! schema at `pyscf/lib/chkfile.py:28-191` + `pyscf/scf/chkfile.py:25-42`.
//!
//! Phase 3 status: empty skeleton (Plan 03-01). Plan 03-06 fills the bodies.
#![forbid(unsafe_code)]
#![warn(clippy::unwrap_used)]
