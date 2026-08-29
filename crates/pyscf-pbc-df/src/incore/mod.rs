//! `pyscf/pbc/df/incore.py` — the auxiliary cell and the 3-centre integral over
//! a DOUBLE lattice sum (plan 14-01).
//!
//! # What this module is, and what it deliberately is not
//!
//! Upstream's `incore.py` reaches its 3-centre integrals through
//! `Int3cBuilder.gen_int3c_kernel`, which builds a Born–von-Kármán supermole
//! (`ft_ao.ExtendedMole`) and hands it to a C driver. **D-PBC-23 declines that
//! machinery** — see `.planning/phases/14-gdf-mdf-rsdf-rsjk/14-CONTEXT.md` — and
//! this module ports the mathematical content instead, the same call D-PBC-21
//! made for `ft_aopair` in Phase 13.
//!
//! The precedent to copy is already in the workspace:
//! `pyscf_pbc_gto::pseudo::vloc_part2` is a gamma-point 3-centre double lattice
//! sum with the auxiliary centre pinned to the origin cell, verified against
//! upstream. [`int3c`] generalises it to k-points, general auxiliary angular
//! momenta, and `aosym`.

pub mod auxcell;
pub mod int3c;

pub use auxcell::{AuxCell, HALF_SPH_NORM, gaussian_int, make_auxcell, make_modrho_basis};
pub use int3c::{Aosym, KptPair, aux_e2, aux_e2_intor, estimate_rcut, fill_2c2e};
