//! pyscf-chkfile: HDF5 chkfile primitives + Checkpointable trait.
//!
//! Source: D-05 (hdf5-metno crate sole owner) + D-06 (per-method schema
//! modules). Schema matches upstream:
//!   - pyscf/lib/chkfile.py:28-191 (primitive layer + save_mol/load_mol)
//!   - pyscf/scf/chkfile.py:25-42  (dump_scf/load_scf — SCF schema)
//!
//! Pitfall 11 mitigation: ORACLE-08 round-trip (plan 03-08) is the empirical
//! seal. This crate ships the primitives; plan 03-08 validates compatibility
//! against real h5py-written upstream chkfiles.
//!
//! Pitfall 8 mitigation: mo_coeff is F-order both in memory
//! (`MOCoefficients.data`) and on disk (`write_dataset_f_order`).
//!
//! Plan 03-06 fills primitives + Checkpointable trait; per-method
//! `Checkpointable` impls live in the consumer crates (pyscf-scf::chkfile,
//! pyscf-dft::chkfile in Phase 4, etc.).
#![forbid(unsafe_code)]
#![warn(clippy::unwrap_used)]

pub mod checkpointable;
pub mod error;
pub mod primitives;

pub use checkpointable::Checkpointable;
pub use error::ChkfileError;
pub use primitives::{
    open_for_read, open_for_write, read_dataset_1d, read_dataset_2d, read_group, read_mol,
    read_scalar_f64, write_dataset_1d, write_dataset_c_order, write_dataset_f_order, write_mol,
    write_scalar_f64,
};

/// Re-export the hdf5-metno crate alias so downstream per-method modules
/// (e.g. pyscf-scf::chkfile) can name `pyscf_chkfile::hdf5::Group` without
/// adding their own hdf5-metno dep. Preserves the "sole owner" discipline
/// (D-05 / algebra-wall analog).
pub use hdf5_metno as hdf5;
