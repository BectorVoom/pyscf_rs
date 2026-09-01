//! Multigrid grid-level task list — plan 17-11 Task 1.
//!
//! Ports `pyscf/pbc/dft/multigrid/multigrid.py`'s
//! `_primitive_gto_cutoff` (`:1753-1781`) and `multi_grids_tasks_for_ke_cut`
//! (`:1624-1751`), the DEFAULT task-splitting route
//! (`TASKS_TYPE = getattr(__config__, 'pbc_dft_multigrid_tasks_type', 'ke_cut')`,
//! `multigrid.py:64`). `multi_grids_tasks_for_rcut` is the `'rcut'` alternative
//! upstream ships but does not default to; it is NOT ported — nothing in this
//! plan's Gate E measurements exercises it (`measurements/gate_multigrid.py`
//! never sets `pbc_dft_multigrid_tasks_type`), and porting a second ladder
//! that no gate observes would be untested code by construction.
//!
//! # The one deliberate simplification, stated up front
//!
//! Upstream splits shells by PRIMITIVE: `_primitive_gto_cutoff` returns an
//! `rcut`/`ke_cutoff` array over each shell's primitives, and
//! `multi_grids_tasks_for_ke_cut` can assign different primitives of the SAME
//! contracted shell to different grid levels (`make_cell_dense_exp`'s
//! `idx = where((ke0 < ke) & (ke <= ke1))` slices the primitive axis, then
//! rebuilds a smaller-`nprim` shell via `cell_dense._env[...] = cs1...`).
//!
//! This port keeps that primitive-level granularity — see [`Pshell`] — but
//! reaches it by fully DECONTRACTING every shell into one [`Pshell`] per
//! primitive up front, rather than upstream's `decontract_basis(to_cart=True,
//! aggregate=True)` + `h_coeff`/`t_coeff`/`t_cell` machinery
//! (`multigrid.py:614-624`, `_get_j_pass2:906-911`). The math is the same
//! sandwich transform (`E` here plays `h_coeff`/`t_coeff`'s role, built once
//! in [`build_pshells`]); this port just never materialises an intermediate
//! `pyscf_pbc_gto::Cell` for the decontracted/aggregated basis, because
//! `PshellGridTable` (`pyscf_kernels::multigrid_collocate`) only needs flat
//! `(centre, alpha, powers)` records, not a libcint-drivable `Cell`. Recorded
//! here rather than left implicit, per the plan's "judgment call, stated out
//! loud" convention (D-PBC-21/23/27 precedent).

use pyscf_core::raw_layout::{ANG_OF, ATOM_OF, BAS_SLOTS, NCTR_OF, NPRIM_OF, PTR_COEFF, PTR_EXP};
use pyscf_kernels::{cart2sph_l_matrix, cart_powers, common_fac_sp};
use pyscf_pbc_gto::Cell;
use pyscf_pbc_gto::cutoff::{estimate_ke_cutoff_pgto, omega};

use crate::error::PbcDftError;

/// `EXTRA_PREC` — `multigrid.py:53`. Folded into the `rcut`/`ke_cutoff`
/// precision the same way upstream folds it, at the one call site that reads
/// it directly (`multi_grids_tasks_for_ke_cut` does not; `eval_mat`'s
/// `log_prec` does — kept here as a documented constant for that future use).
pub const EXTRA_PREC: f64 = 1e-2;

/// `RMAX_FACTOR_ORTH` / `RMAX_FACTOR_NONORTH` / `KE_RATIO` — `multigrid.py:57-63`.
const KE_RATIO: f64 = 1.3;
/// `INIT_MESH_ORTH` — `multigrid.py:60`.
const INIT_MESH_ORTH: [usize; 3] = [12, 12, 12];
/// `INIT_MESH_NONORTH` — `multigrid.py:61`.
const INIT_MESH_NONORTH: [usize; 3] = [32, 32, 32];

/// One decontracted primitive Gaussian — the atomic unit grid-level
/// assignment operates on (see the module doc for why this differs from
/// upstream's per-shell-with-primitive-slicing representation).
#[derive(Debug, Clone)]
pub struct Pshell {
    /// Original (contracted) shell index, for diagnostics only.
    pub orig_bas: usize,
    /// Angular momentum.
    pub l: i32,
    /// Real-space centre, Bohr.
    pub center: [f64; 3],
    /// Primitive exponent.
    pub alpha: f64,
    /// `_primitive_gto_cutoff`'s per-primitive lattice-sum cutoff radius.
    pub rcut: f64,
    /// `_primitive_gto_cutoff`'s per-primitive planewave `ke_cutoff`.
    pub ke_cutoff: f64,
    /// First Cartesian AO index this pshell owns in the decontracted
    /// Cartesian AO space (`ncart(l)` contiguous slots starting here).
    pub cart_ao0: usize,
    /// `ctr_coeff * common_fac_sp(l)` — the SAME scalar `build_pshells` bakes
    /// into every row of `E` for this pshell (`multigrid_collocate.rs`'s
    /// `pshell_coef`, computed once here so plan 17-12's pair-fusion code
    /// (`crate::multigrid::pair`) does not have to re-derive it from
    /// `mol._env` a second time). Added by plan 17-12 Task 1 — purely
    /// additive, v1's own use of `Pshell` (`build_pshells`'s `E`-matrix
    /// construction) is unchanged.
    pub coef: f64,
}

impl Pshell {
    /// Number of Cartesian components this pshell contributes.
    pub fn ncart(&self) -> usize {
        ncart(self.l as u32)
    }
}

/// The decontracted representation of a `Cell`'s basis: every primitive as
/// its own [`Pshell`], plus the dense expansion matrix `E` mapping the
/// CONTRACTED (spherical or cartesian, `cell.mol`-native) AO index to the
/// decontracted CARTESIAN pshell-AO index.
///
/// `E` is `nao_p x nao` row-major (`E[p * nao + ao]`), so a contracted
/// density matrix `dm` (`nao x nao`) sandwiches as `dm_p = E . dm . E^T`
/// (`nao_p x nao_p`) and a decontracted potential matrix `v_p` sandwiches
/// back as `v = E^T . v_p . E`.
#[derive(Debug, Clone)]
pub struct Decontracted {
    pub pshells: Vec<Pshell>,
    /// Total decontracted Cartesian AO count (`sum ncart(pshell.l)`).
    pub nao_p: usize,
    /// Contracted AO count (`cell.mol.nao_nr`).
    pub nao: usize,
    /// `nao_p x nao` row-major.
    pub expand: Vec<f64>,
}

/// `_primitive_gto_cutoff(cell)` — `multigrid.py:1753-1781`, evaluated at ONE
/// primitive `(alpha, l, c)` rather than vectorised over a shell's primitive
/// axis (the Rust caller loops).
///
/// `precision = cell.precision / max(vol, 1)` is the caller's job (passed in
/// as `precision`, already divided) so this stays a pure per-primitive
/// formula, matching upstream's per-element vectorised body exactly.
pub fn primitive_rcut(alpha: f64, l: i32, c: f64, precision: f64, cell_rcut: f64) -> f64 {
    let norm_ang = ((2 * l) as f64 + 1.0) / (4.0 * std::f64::consts::PI);
    let fac = 2.0 * std::f64::consts::PI * c * c * norm_ang / alpha / precision;
    let mut r = cell_rcut;
    // Upstream repeats this exact fixed-point update line twice
    // (`multigrid.py:1774-1775`) — not a typo, a deliberate 2-sweep refine.
    r = ((fac * r.powf(l as f64 + 1.0) + 1.0).ln() / alpha).sqrt();
    r = ((fac * r.powf(l as f64 + 1.0) + 1.0).ln() / alpha).sqrt();
    r
}

/// Whether `a`'s off-diagonal entries are all (numerically) zero —
/// `abs(a - diag(a.diagonal())).max() < 1e-12`, the orthogonality test
/// `multigrid.py` runs before choosing `INIT_MESH_ORTH`/`_NONORTH`.
#[allow(clippy::needless_range_loop)]
fn is_orthogonal(a: &[[f64; 3]; 3]) -> bool {
    for i in 0..3 {
        for j in 0..3 {
            if i != j && a[i][j].abs() >= 1e-12 {
                return false;
            }
        }
    }
    true
}

#[inline]
fn ncart(l: u32) -> usize {
    ((l as usize + 1) * (l as usize + 2)) / 2
}

/// Decontract `cell`'s basis into [`Pshell`]s and build the `E` expansion
/// matrix. `cell.precision`/`cell.vol()`/`cell.omega` feed
/// `_primitive_gto_cutoff`; `cell.mol.cart` decides whether `E` also carries
/// a `cart2sph` factor.
///
/// # Errors
/// [`PbcDftError`] if a shell's `NCTR_OF != 1` after `Cell::build` (this
/// port's contracted-shell handling assumes upstream's own convention: GTH
/// basis sets, the only ones this milestone's reference systems use, are
/// always single-contraction per shell — see the fixture check in
/// `crates/pyscf-pbc-dft/tests/multigrid.rs`) or if `cart2sph_l_matrix`
/// fails for an `l` this port does not carry a table for.
pub fn build_pshells(cell: &Cell) -> Result<Decontracted, PbcDftError> {
    let mol = &cell.mol;
    let vol = cell.vol();
    let precision = cell.precision / vol.max(1.0);
    let om = omega(cell);
    let cell_rcut = cell.rcut;
    let coords = mol.atom_coords();

    let nao = mol.nao_nr;
    let mut pshells = Vec::new();
    let mut nao_p = 0usize;
    // Dense E, built incrementally: E[p_row * nao + ao_col].
    let mut expand_rows: Vec<Vec<f64>> = Vec::new();

    for ib in 0..mol.nbas {
        let row = ib * BAS_SLOTS;
        let l = mol._bas[row + ANG_OF];
        let nprim = mol._bas[row + NPRIM_OF] as usize;
        let nctr = mol._bas[row + NCTR_OF] as usize;
        if nctr != 1 {
            return Err(PbcDftError::from(pyscf_core::PyscfRsError::Core(
                pyscf_core::CoreError::InvalidMolecule(format!(
                    "multigrid: shell {ib} has nctr={nctr} != 1; this port only \
                     handles single-contraction shells (every GTH basis set this \
                     milestone gates satisfies this)"
                )),
            )));
        }
        let atom = mol._bas[row + ATOM_OF] as usize;
        let center = coords[atom];
        let pe = mol._bas[row + PTR_EXP] as usize;
        let pc = mol._bas[row + PTR_COEFF] as usize;
        let ao0 = mol.ao_loc_nr[ib] as usize;
        let nc = ncart(l as u32);
        let cfac = common_fac_sp(l as u32);
        let t = if mol.cart {
            None
        } else {
            Some(cart2sph_l_matrix(l as u32)?)
        };
        let nout = if mol.cart { nc } else { 2 * l as usize + 1 };

        for p in 0..nprim {
            let alpha = mol._env[pe + p];
            let raw_coeff = mol._env[pc + p]; // nctr == 1, so pc + 0*nprim + p
            let rcut = primitive_rcut(alpha, l, raw_coeff.abs(), precision, cell_rcut);
            let ke_cutoff = estimate_ke_cutoff_pgto(alpha, l, raw_coeff.abs(), precision, om);

            let cart_ao0 = nao_p;
            let scale = raw_coeff * cfac;
            pshells.push(Pshell {
                orig_bas: ib,
                l,
                center,
                alpha,
                rcut,
                ke_cutoff,
                cart_ao0,
                coef: scale,
            });
            nao_p += nc;
            for c in 0..nc {
                let mut erow = vec![0.0f64; nao];
                match &t {
                    None => erow[ao0 + c] = scale,
                    Some(tm) => {
                        for m in 0..nout {
                            let w = tm[m * nc + c];
                            if w != 0.0 {
                                erow[ao0 + m] = scale * w;
                            }
                        }
                    }
                }
                expand_rows.push(erow);
            }
        }
    }

    let mut expand = vec![0.0f64; nao_p * nao];
    for (p, erow) in expand_rows.into_iter().enumerate() {
        expand[p * nao..p * nao + nao].copy_from_slice(&erow);
    }

    Ok(Decontracted {
        pshells,
        nao_p,
        nao,
        expand,
    })
}

/// One grid level: its own uniform mesh, and the two pshell partitions
/// `multi_grids_tasks_for_ke_cut` needs (`grids_dense`/`grids_sparse`,
/// `multigrid.py:1690-1730`).
#[derive(Debug, Clone)]
pub struct GridLevel {
    pub mesh: [usize; 3],
    /// Pshells natively assigned to this level (`shls_dense`).
    pub dense: Vec<usize>,
    /// Pshells from EARLIER (coarser) levels (`shls_sparse`) — the
    /// dense-sparse cross term is evaluated at THIS level's (finer) mesh.
    pub sparse: Vec<usize>,
    /// `max(rcut)` over `dense` — diagnostic / image-list sizing.
    pub rcut: f64,
}

/// `multi_grids_tasks_for_ke_cut(cell, fft_mesh)` — `multigrid.py:1624-1751`,
/// operating on [`Pshell`]s (see the module doc).
///
/// # Errors
/// [`PbcDftError`] from [`build_pshells`], or if `cutoff_to_mesh`/
/// `mesh_to_cutoff` fail (singular lattice).
pub fn multi_grids_tasks_for_ke_cut(
    cell: &Cell,
    decon: &Decontracted,
    fft_mesh: [usize; 3],
) -> Result<Vec<GridLevel>, PbcDftError> {
    let a = cell.lattice_vectors();
    let init_mesh = if is_orthogonal(&a) {
        INIT_MESH_ORTH
    } else {
        INIT_MESH_NONORTH
    };
    let ke_min_axes = pyscf_pbc_tools::mesh::mesh_to_cutoff(&a, init_mesh)?;
    let ke_cutoff_min = ke_min_axes.into_iter().fold(f64::INFINITY, f64::min);
    let ke_cutoff_max = decon
        .pshells
        .iter()
        .map(|p| p.ke_cutoff)
        .fold(f64::NEG_INFINITY, f64::max);

    let mut ke_delim = vec![0.0f64, ke_cutoff_min];
    let mut ke1 = ke_cutoff_min;
    while ke1 < ke_cutoff_max {
        ke1 *= KE_RATIO;
        ke_delim.push(ke1);
    }

    let mut levels = Vec::new();
    for w in ke_delim.windows(2) {
        let (ke0, ke1) = (w[0], w[1]);
        let mut dense: Vec<usize> = (0..decon.pshells.len())
            .filter(|&i| {
                let ke = decon.pshells[i].ke_cutoff;
                ke0 < ke && ke <= ke1
            })
            .collect();
        if dense.is_empty() {
            continue;
        }
        let mut mesh = pyscf_pbc_tools::mesh::cutoff_to_mesh(&a, ke1)?;
        let reached_target = (0..3).all(|d| mesh[d] >= fft_mesh[d]);
        if reached_target {
            // Swallow every remaining pshell — this becomes the last level.
            dense = (0..decon.pshells.len())
                .filter(|&i| decon.pshells[i].ke_cutoff > ke0)
                .collect();
        }
        for d in 0..3 {
            mesh[d] = mesh[d].min(fft_mesh[d]);
        }
        let sparse: Vec<usize> = (0..decon.pshells.len())
            .filter(|&i| decon.pshells[i].ke_cutoff <= ke0)
            .collect();
        let rcut = dense
            .iter()
            .map(|&i| decon.pshells[i].rcut)
            .fold(0.0f64, f64::max);
        levels.push(GridLevel {
            mesh,
            dense,
            sparse,
            rcut,
        });
        if reached_target {
            break;
        }
    }
    Ok(levels)
}

/// `cart_powers(l)`, re-exported for callers building `PshellGridTable`
/// slots (`crate::multigrid::collocate`).
pub fn pshell_cart_powers(l: i32) -> Vec<(u32, u32, u32)> {
    cart_powers(l as u32)
}
