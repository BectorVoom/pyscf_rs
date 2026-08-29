//! pyscf-pbc-gto: Cell, Gv, SI, ewald, GTH pseudo, pbc_intor, eval_gto periodic, neighborlist
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::unwrap_used)]

pub mod error;
pub use error::*;

// Phase 9 plan 09-03 — the `Cell` type (D-PBC-01: OWNS a `Mole`, `Deref`s to it).
pub mod cell;
pub mod dumps_loads;

// Phase 9 plan 09-04 — cutoff / rcut / mesh estimators.
pub mod coulg;
pub mod exxdiv_vcut;
pub mod cutoff;

// Phase 9 plan 09-05 — G-vectors (K-01), structure factors (K-02), uniform grids.
pub mod grids;
pub mod gv;

// Phase 9 plan 09-06 — lattice sums and supercells (`tools/pbc.py:587-786`,
// `:836-840`). The geometry cores live in `pyscf_pbc_tools::{lattice, supercell}`.
pub mod lattice;
pub mod supercell;

// Phase 9 plan 09-07 — k-point meshes (`cell.py:827-884`) plus the `Cell`-taking
// wrappers over `pyscf_pbc_lib::kpts_helper`.
pub mod kpts_mesh;

// Phase 9 plan 09-08 — Ewald summation (`cell.py:650-824`) and the
// particle-mesh Ewald B-splines (`gto/ewald_methods.py:30-176`). K-05 / K-06
// live in `pyscf_kernels::pbc::ewald`.
pub mod ewald;
pub mod ewald_pme;

// Phase 10 plan 10-02 — shell-pair neighbor list + lattice-sum screening
// (D-PBC-08). Consumed by `pbc_intor`.
pub mod neighborlist;

// Phase 10 plan 10-03 — `pbc_intor` / `intor_cross`, the lattice-sum driver
// (D-PBC-07). THE core of Phase 10.
pub mod pbc_intor;

// Phase 10 plan 10-04 — periodic AO evaluation (`eval_ao_kpts`, K-08).
pub mod eval_gto;

// Phase 10 plan 10-07 — `get_ovlp` / `get_hcore` assembly.
pub mod hcore;

pub mod pseudo;
pub mod types;

// PBC-MASTER-PLAN §9.2 — the five shared reference systems. Feature-gated so
// they are not compiled into a production build; downstream PBC crates get them
// with `pyscf-pbc-gto = { path = "...", features = ["test-systems"] }` in their
// `[dev-dependencies]`. Do NOT redefine these per crate.
#[cfg(feature = "test-systems")]
pub mod test_systems;

pub use cell::{Cell, M, det3, estimate_mesh, estimate_rcut, inv3, transpose3};
pub use coulg::{CoulGArgs, ExxDiv, get_coulg, get_coulg_at_gv, madelung};
pub use cutoff::{
    INTEGRAL_PRECISION, PgtoOp, RCUT_EPS, RCUT_MAX_CYCLE, bas_rcut, error_for_ke_cutoff,
    estimate_ke_cutoff, estimate_ke_cutoff_pgto, estimate_rcut_pgto, extract_pgto_params,
    get_bounding_sphere, get_nimgs, mesh_inf_vacuum, pgf_rcut, pgf_rcut_c, rcut_by_shells,
    rcut_by_shells_with_pgf,
};
pub use dumps_loads::{CellPack, dumps, loads, pack, unpack};
pub use eval_gto::{
    EvalAoKptsOutput, estimate_rcut_for_eval, eval_ao_kpts, eval_ao_kpts_with_images,
};
pub use ewald::{
    EWALD_G0_SENTINEL, EWALD_R_MIN, ewald, ewald_g_space, ewald_real_space, ewald_self,
    get_ewald_params,
};
pub use ewald_pme::{
    Bspline, EWALD_DIRECT_R_MIN, INTERPOLATION_ORDER, bspline, bspline_grad, bspline_value,
    get_ewald_direct, particle_mesh_ewald, pme_charge_mesh,
};
pub use grids::UniformGrids;
pub use gv::{
    GvWeights, fftfreq, fftfreq_scaled, get_gv, get_gv_weights, get_si, get_uniform_grids,
};
pub use hcore::{HcoreParts, get_hcore, get_hcore_parts, get_ovlp, get_t};
pub use kpts_mesh::{
    KCONSERV_TOL, KIdx, KPT_DIFF_TOL, Kconserv, Kconserv3, UniqueKpts, WITH_GAMMA, WRAP_AROUND,
    gamma_point, get_kconserv, get_kconserv3, intersection, is_gamma_point, is_zero, make_kpts,
    make_kpts_default, make_kpts_with_symmetry, member, round_to_fbz, unique,
};
pub use lattice::{
    check_lattice_sum_range, get_lattice_ls, get_lattice_ls_default, get_monkhorst_pack_size,
    get_monkhorst_pack_size_default, lattice_sum_dimension,
};
pub use neighborlist::{
    NeighborList, NeighborPair, build_neighbor_list, build_neighbor_list_for_shlpairs,
};
pub use pbc_intor::{
    GAMMA_IMAG_WARN_TOL, KPT_GAMMA_TOL, PBC_INTOR_SHELL_WARN_LIMIT, PbcIntorOpts, PbcIntorOutput,
    SUPPORTED_INTORS, intor_cross, intor_cross_with_images, is_gamma, lattice_images, pbc_intor,
};
pub use pseudo::{PseudoData, resolve_pseudo};
pub use supercell::{cell_plus_imgs, super_cell};
pub use types::{ALattice, CellBuildArgs, DEFAULT_PRECISION, LowDimFtType};
