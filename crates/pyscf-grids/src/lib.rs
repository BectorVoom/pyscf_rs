//! pyscf-grids: Becke molecular integration grids.
//!
//! Source: D-05 / D-06 — Phase 4 introduces this crate. Ports upstream
//! `pyscf/dft/gen_grid.py` (Becke partitioning + grid assembly),
//! `pyscf/dft/radi.py` (radial quadrature schemes + Bragg/Treutler/Mura
//! grids), `pyscf/dft/LebedevGrid.py` (angular Lebedev points/weights), and
//! the atomic radii tables from `pyscf/data/radii.py`.
//!
//! The grid points + weights must reproduce upstream byte-for-byte across
//! `grid.level` 0..9 (DFT-04 / DFT-09 verification). This crate stays behind
//! the algebra wall: it depends only on `pyscf-core` + `pyscf-algebra` and
//! never references a `cubecl-*` runtime crate directly (ALG-06 / Pitfall 2).
//!
//! Plan 04-04 fills the module bodies shipped as empty skeletons by plan 04-01.
#![forbid(unsafe_code)]
#![warn(clippy::unwrap_used)]

pub mod radial;
pub mod radii;
pub mod lebedev;
pub mod prune;
pub mod partition;
pub mod levels;
