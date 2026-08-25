//! pyscf-pbc-tools: fft/ifft, get_coulG, madelung, lattice_Ls, super_cell, k2gamma
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::unwrap_used)]

pub mod error;
pub use error::*;

// Phase 9 plan 09-04 — closed-form 3x3 lattice algebra (shared with
// `pyscf-pbc-gto`, which re-exports it) and the mesh <-> cutoff conversions.
pub mod mat3;
pub mod mesh;

// Phase 9 plan 09-06 — lattice sums (`get_lattice_Ls`, `check_lattice_sum_range`,
// `get_monkhorst_pack_size`, `round_to_cell0`) and the supercell geometry
// (`super_cell`, `cell_plus_imgs`). The `Cell`-taking wrappers live in
// `pyscf_pbc_gto::{lattice, supercell}` — see the module docs for why.
pub mod lattice;
pub mod supercell;

pub use lattice::{
    check_lattice_sum_range, get_lattice_ls, max_atom_pair_distance,
    monkhorst_pack_size_from_scaled, qr_row2, round_to_cell0, round_to_cell0_default,
};
pub use mat3::{cross3, det3, dot3, inv3, norm3, transpose3};
pub use mesh::{
    cutoff_to_gs, cutoff_to_mesh, gs_to_cutoff, mesh_to_cutoff, qr_heights, qr_r22_abs,
    qr_r22_abs_closed_form,
};
pub use supercell::{
    cell_plus_imgs_translations, image_atom_coords, scale_lattice, super_cell_translations,
};
