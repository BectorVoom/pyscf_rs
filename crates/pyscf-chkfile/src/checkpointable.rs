//! `Checkpointable` — per-method dump/load surface (D-06).
//!
//! Source: D-06 — each method-result type implements this trait against an
//! HDF5 Group. pyscf-scf::ScfResult impls in plan 03-06; future phases:
//! pyscf-dft::KsResult (Phase 4), pyscf-ccsd::CcsdResult (Phase 6),
//! pyscf-geomopt::OptimState (Phase 7).
//!
//! The trait surface is intentionally minimal: dump + load. Schema choice
//! lives in the per-method module (e.g. pyscf-scf::chkfile mirrors
//! pyscf/scf/chkfile.py:25-42 byte-for-byte).
use crate::error::ChkfileError;
use hdf5_metno as hdf5;

pub trait Checkpointable: Sized {
    /// Write `self` under the given HDF5 Group (e.g. group `"/scf"`).
    fn dump(&self, group: &hdf5::Group) -> Result<(), ChkfileError>;

    /// Load `Self` from the given HDF5 Group.
    fn load(group: &hdf5::Group) -> Result<Self, ChkfileError>;
}
