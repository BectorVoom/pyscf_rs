//! Multigrid **v2** task list + pair-fused collocation —
//! `pyscf/pbc/dft/multigrid/multigrid_pair.py` + `_backend_c.py`'s
//! `TaskList`/`GridLevel_Info`/`build_task_list`/`init_gridlevel_info`/
//! `init_rs_grid` (plan 17-12, Tasks 1/2).
//!
//! # The four C destructors
//!
//! Upstream's `_backend_c.py` pairs every constructor with a matching
//! destructor: `del_task_list`, `del_gridlevel_info`, `del_rs_grid` (twice —
//! `RS_Grid` is built and freed on both the "dense" and the "sparse" side in
//! some call paths). **None of the four is ported.** This is a deliberate
//! port decision, not an omission: [`PairTaskList`], [`GridLevelSpec`] and
//! [`PairLevelValues`] below are ordinary Rust structs owning `Vec`s: they
//! are freed by `Drop` the moment their last owner goes out of scope, with
//! no `unsafe`, no reference counting and no possibility of the
//! double-free/leak class of bug the C API's four-function contract exists
//! to avoid. Same courtesy `17-03-SUMMARY.md` Task 5 pays `cell.py`'s
//! deleted Python reference cycle, and `17-11-SUMMARY.md`'s "what did not
//! ship" section pays FFTDF's `del_task_list` non-equivalent.
//!
//! # Why this is NOT a literal port of `build_task_list` (`multigrid.c:427`)
//!
//! Upstream's C task-list builder is 3 182 lines across
//! `multigrid.c`/`grid_collocate.c`/`grid_integrate.c` combined — hand
//! cache-blocked, restricted to a per-pair rcut SUBMESH (not the full
//! level mesh), and written against a Hermite-Gaussian recursion this
//! plan's time budget could not port line-by-line (RULE 2 in spirit, not
//! letter — see `.planning/phases/17-ksymm-multigrid/17-12-SUMMARY.md`'s
//! "what did not ship" section for the honest accounting). What ships here
//! is a MATHEMATICALLY FAITHFUL reformulation of the SAME physical
//! quantity, gated by construction (the adjoint identity,
//! `crates/pyscf-kernels/tests/multigrid_pair.rs`) rather than by literal
//! agreement with upstream's per-level pair membership:
//!
//! * **Gaussian product theorem.** A primitive pair `(p ∈ shell i, q ∈
//!   shell j, image L)` fuses into ONE combined Gaussian at
//!   `P = (alpha_p·A + alpha_q·(B+L)) / eta`, `eta = alpha_p+alpha_q`,
//!   `K = exp(-alpha_p·alpha_q/eta · |A-(B+L)|²)`.
//! * **Binomial shift expansion** ([`binom_shift`]) re-expresses
//!   `(x-Ax)^a (x-Bx-Lx)^b` in powers of `(x-Px)` — the standard identity
//!   `(x-A)^a = Σ_m C(a,m) (x-P)^m (P-A)^{a-m}`, multiplied out and
//!   collected by power of `(x-P)`. Applied separably per Cartesian axis,
//!   this gives every `(k1,k2,k3)` monomial term's coefficient with NO
//!   approximation — it is an EXACT polynomial identity, not a truncation.
//! * **Level assignment.** Each pair is assigned the COARSEST of the
//!   `NTASKS` grid levels whose plane-wave cutoff still resolves
//!   `max(pshell_i.ke_cutoff, pshell_j.ke_cutoff)` — reusing the
//!   per-PRIMITIVE cutoff [`crate::multigrid::tasks::build_pshells`]
//!   already computes via `estimate_ke_cutoff_pgto` (Task 1's ladder,
//!   `multi_grids_tasks`, `multigrid_pair.py:59-78`, IS a literal, direct
//!   port — only the per-pair MEMBERSHIP rule is a documented
//!   reformulation, not the level ladder itself).
//!
//! This is the same class of judgment call D-PBC-21/23/27 record: measure
//! and state the deviation rather than silently approximate it. The
//! consequence is priced in `crate::multigrid::pair (this module)`'s Gate E numbers
//! (v1-vs-v2 and v2-vs-FFTDF), reported honestly rather than assumed to
//! match upstream's own C task list bit-for-bit.
//!
//! # Scope: GAMMA POINT ONLY
//!
//! Same stated scope reduction as v1 (`crate::multigrid::numint`'s module
//! doc) and the same one this plan's own instructions sanction pending
//! 17-05's `KPoints` — Bloch-phase (k-point-resolved) pair collocation is
//! NOT ported.

use pyscf_algebra::AlgebraError;
use pyscf_kernels::multigrid_pair::{
    MAX_SLOTS_PER_INSTANCE, PairSlotBatch, PairSlotBatchDevice, PairSlotTable,
    collocate_pairs_integrate, collocate_pairs_rho,
};
use pyscf_pbc_gto::Cell;
use pyscf_pbc_gto::lattice::get_lattice_ls;
use tracing::span::EnteredSpan;

use crate::error::PbcDftError;
use crate::multigrid::tasks::{Decontracted, Pshell, pshell_cart_powers};

/// `NTASKS` — `_backend_c.py`/`multigrid_pair.py:47`, `pbc_dft_multigrid_ntasks`.
pub const NTASKS: usize = 4;
/// `KE_RATIO` — `multigrid_pair.py:48`, `pbc_dft_multigrid_ke_ratio`.
pub const KE_RATIO: f64 = 3.0;

/// One grid level's mesh + the plane-wave cutoff that defines it —
/// `GridLevel_Info` (`multigrid.c:20-40`; the struct itself, not its C
/// constructor/destructor, which Rust ownership replaces — see the module
/// doc).
#[derive(Debug, Clone, Copy)]
pub struct GridLevelSpec {
    pub mesh: [usize; 3],
    pub cutoff: f64,
}

/// `multi_grids_tasks(cell, ke_cutoff=None, ...)` — `multigrid_pair.py:59-78`,
/// a LITERAL port (unlike per-pair membership, see the module doc): `NTASKS`
/// levels, geometric in `ke_cutoff` by `KE_RATIO`, finest level pinned to
/// `cell.mesh` exactly.
///
/// # Errors
/// Propagates [`pyscf_pbc_tools::mesh::mesh_to_cutoff`] /
/// [`pyscf_pbc_tools::mesh::cutoff_to_mesh`] (singular lattice).
pub fn pair_grid_levels(cell: &Cell) -> Result<Vec<GridLevelSpec>, PbcDftError> {
    let a = cell.lattice_vectors();
    let fft_mesh = cell.mesh;
    let ke_cutoff = match cell.ke_cutoff {
        Some(k) => k,
        None => {
            let k3 = pyscf_pbc_tools::mesh::mesh_to_cutoff(&a, fft_mesh)?;
            k3.into_iter().fold(f64::NEG_INFINITY, f64::max)
        }
    };
    let mut cutoff = [0.0f64; NTASKS];
    cutoff[NTASKS - 1] = ke_cutoff;
    let mut ke1 = ke_cutoff;
    for i in (0..NTASKS - 1).rev() {
        ke1 /= KE_RATIO;
        cutoff[i] = ke1;
    }
    let mut levels = Vec::with_capacity(NTASKS);
    for (i, &c) in cutoff.iter().enumerate() {
        let mesh = if i == NTASKS - 1 {
            fft_mesh
        } else {
            pyscf_pbc_tools::mesh::cutoff_to_mesh(&a, c)?
        };
        levels.push(GridLevelSpec { mesh, cutoff: c });
    }
    Ok(levels)
}

/// The full v2 task list: `NTASKS` [`GridLevelSpec`]s, plus every
/// primitive-pair `(pi, pj)` (global [`Pshell`] indices into
/// `Decontracted::pshells`) assigned to each — `TaskList`
/// (`multigrid.c:145-311`; struct only, see the module doc for the
/// destructor note).
#[derive(Debug, Clone)]
pub struct PairTaskList {
    pub levels: Vec<GridLevelSpec>,
    /// `level_pairs[l]` — every `(pi, pj)` this level owns.
    pub level_pairs: Vec<Vec<(usize, usize)>>,
}

/// `EXTRA_PREC` — same constant `tasks.rs` already documents
/// (`multigrid_pair.py:52`, `pbc_gto_eval_gto_extra_precision`); used here as
/// the pair-overlap prefactor `K` screening threshold (relative to
/// `cell.precision`).
const EXTRA_PREC: f64 = 1e-2;

/// `build_task_list(cell, decon)` — Task 1's pair enumeration + level
/// assignment (module doc: a documented reformulation of `build_task_list`,
/// not a literal port).
///
/// For every ordered pair `(pi, pj)` of decontracted primitives
/// (`decon.pshells`, built by [`crate::multigrid::tasks::build_pshells`]),
/// every periodic image `L` with `|A-(B+L)| <= rcut_pi + rcut_pj` is a
/// candidate; images whose Gaussian-product prefactor `K` falls below
/// `cell.precision * EXTRA_PREC` are screened out. A pair with NO surviving
/// image is dropped entirely (its two members do not overlap at all, to the
/// cell's own precision).
///
/// # Errors
/// Propagates [`pair_grid_levels`] / [`get_lattice_ls`].
pub fn build_pair_task_list(
    cell: &Cell,
    decon: &Decontracted,
) -> Result<PairTaskList, PbcDftError> {
    let levels = pair_grid_levels(cell)?;
    let cutoffs: Vec<f64> = levels.iter().map(|l| l.cutoff).collect();
    let mut level_pairs: Vec<Vec<(usize, usize)>> = vec![Vec::new(); levels.len()];

    let threshold = cell.precision * EXTRA_PREC;
    let n = decon.pshells.len();
    for pi in 0..n {
        for pj in 0..n {
            let p = &decon.pshells[pi];
            let q = &decon.pshells[pj];
            let search_rcut = (p.rcut + q.rcut).max(1e-6);
            let ls = get_lattice_ls(cell, Some(search_rcut), None, false)?;
            let mut any = false;
            for l in &ls {
                let dx = p.center[0] - (q.center[0] + l[0]);
                let dy = p.center[1] - (q.center[1] + l[1]);
                let dz = p.center[2] - (q.center[2] + l[2]);
                let dist2 = dx * dx + dy * dy + dz * dz;
                let eta = p.alpha + q.alpha;
                let k = (-p.alpha * q.alpha / eta * dist2).exp();
                let scale = (p.coef * q.coef).abs();
                if pair_prescreen_bound(k, scale, dist2.sqrt(), p.l + q.l, eta) >= threshold {
                    any = true;
                    break;
                }
            }
            if !any {
                continue;
            }
            let pair_ke = p.ke_cutoff.max(q.ke_cutoff);
            let level_idx = cutoffs
                .iter()
                .position(|&c| c >= pair_ke)
                .unwrap_or(levels.len() - 1);
            level_pairs[level_idx].push((pi, pj));
        }
    }

    Ok(PairTaskList {
        levels,
        level_pairs,
    })
}

/// Binomial coefficient `C(n,k)`, `n` small (angular momenta this milestone
/// gates are `<= 4`), plain product form — no factorial overflow risk at
/// this scale.
fn binom(n: u32, k: u32) -> f64 {
    if k > n {
        return 0.0;
    }
    let k = k.min(n - k);
    let mut result = 1.0f64;
    for i in 0..k {
        result = result * (n - i) as f64 / (i + 1) as f64;
    }
    result
}

/// The binomial shift expansion: coefficient of `(x-P)^k`, `k=0..=(a+b)`, in
/// `(x-A)^a (x-B)^b`, given `pa = P-A`, `pb = P-B` (one Cartesian axis).
///
/// `(x-A)^a = Σ_m C(a,m) (x-P)^m (P-A)^{a-m}` (binomial theorem on
/// `x-A = (x-P)+(P-A)`), similarly for `(x-B)^b`; multiplying the two sums
/// and collecting by `m+n=k` gives this. Exact — no truncation.
pub fn binom_shift(a: u32, b: u32, pa: f64, pb: f64) -> Vec<f64> {
    let mut f = vec![0.0f64; (a + b + 1) as usize];
    for m in 0..=a {
        let cam = binom(a, m) * pa.powi((a - m) as i32);
        if cam == 0.0 {
            continue;
        }
        for nn in 0..=b {
            let cbn = binom(b, nn) * pb.powi((b - nn) as i32);
            f[(m + nn) as usize] += cam * cbn;
        }
    }
    f
}

/// Target edge (grid points per axis) of one [`GridBlock`] — upstream's
/// rcut sub-mesh idea turned inside out: instead of one sub-mesh per
/// Gaussian, one slot subset per block of the mesh. Every level is
/// streamed through the kernel one block at a time with the reduction
/// fused into the kernel (`collocate_pairs_rho` / `collocate_pairs_integrate`),
/// so the only per-launch buffers are the block's points, its selected
/// slots and ONE output value per point (rho) or per slot (pass2): peak
/// memory is bounded by the table itself, never by `slots × ngrids`,
/// which for the 25³ Gate-E cells is >100 GiB per level and was
/// exit-137'd in 17-12's first run (`17-12-SUMMARY.md`).
pub const BLOCK_EDGE: usize = 5;

/// One level's pair-fused GEOMETRY — host-side only, NO grid values —
/// the v2 analogue of upstream's per-level `TaskList` entry plus the
/// Hermite/Cartesian pair coefficients `grid_collocate_drv` contracts the
/// density matrix into before it touches the grid.
///
/// Three index spaces:
///
/// * **fused terms** `t` — one per `(pair instance (p,q,L), monomial
///   (k1,k2,k3))`: the logical object `C_t · (r-P)^k · exp(-eta|r-P|²)`.
/// * **slots** `s` — one per `(term, ci, cj)` with a non-zero geometric
///   coefficient; the routing between a fused term and the decontracted
///   AO pair it came from. The coefficient is `K · fx·fy·fz` ONLY — the
///   contraction coefficient and `common_fac_sp` already live in
///   `Decontracted::expand`, and `dm_p = E·dm·Eᵀ` / `v = Eᵀ·v_p·E` apply
///   them exactly once (the same convention `crate::multigrid::colloc`
///   documents at its `pshell_coef = 1.0`).
/// * **kernel slots** `k` — one per `(term, wrap image L1)`: what the
///   kernel actually evaluates. A fused Gaussian centred at `P` is a
///   PERIODIC function on the grid, `Σ_{L1} G_P(r - L1)`; upstream's C
///   collocation gets this by indexing its rcut sub-mesh modulo the mesh,
///   this port gets it by giving every image `P + L1` whose radius reaches
///   the cell its own kernel instance, sharing the term's coefficients
///   (the polynomial is in `r - P - L1`, and `pa`/`pb` are unchanged by a
///   common shift of `A` and `B+L`). Without it a Gaussian on an atom at
///   the cell origin keeps only its one in-cell octant.
///
/// `rho(r) = Σ_k C_{term(k)} · kslot_k(r)` with `C_t = Σ_{s→t}
/// dm_p[ci,cj]·coef_s`, and its transpose `v_p[ci,cj] += Σ_{s→t} coef_s ·
/// Σ_{k→t} ∫ w·kslot_k`. Every `Σ_{s→t}` / `Σ_{k→t}` is fixed-order
/// (table order), independent of any parallel split (D-PBC-17 shape).
#[derive(Debug)]
pub struct PairLevelTable {
    pub mesh: [usize; 3],
    pub ngrids: usize,
    /// `(ngrids, 3)` row-major, Bohr.
    pub coords: Vec<f64>,
    /// `inv(a)`, rows of `a` the lattice vectors: `frac = r · inv_a`.
    pub inv_a: [[f64; 3]; 3],
    /// Slab height between the two faces normal to each reciprocal axis.
    pub heights: [f64; 3],
    /// Number of fused terms.
    pub nterms: usize,
    /// Per slot: the GEOMETRIC coefficient (`K·fx·fy·fz`, see above).
    pub slot_coef: Vec<f64>,
    /// Per slot: the decontracted Cartesian AO row / column it routes to.
    pub slot_ci: Vec<u32>,
    pub slot_cj: Vec<u32>,
    /// Per slot: the fused term it feeds.
    pub slot_term: Vec<u32>,
    /// Per kernel slot, 3 entries: `(k1,k2,k3)`.
    pub kslot_pow: Vec<u32>,
    /// Per kernel slot: its kernel instance (one per `(p,q,L,L1)`).
    pub kslot_instance: Vec<u32>,
    /// Per kernel slot: the fused term it is an image of.
    pub kslot_term: Vec<u32>,
    /// Per kernel instance: `eta = alpha_p + alpha_q`.
    pub instance_alpha: Vec<f64>,
    /// Per kernel instance, 3 entries: the (image-shifted) centre `P + L1`.
    pub instance_center: Vec<f64>,
    /// Per kernel instance: the fused Gaussian's own cutoff radius
    /// ([`fused_radius`]) — what decides which [`GridBlock`]s it reaches.
    pub instance_radius: Vec<f64>,

    /// M-02: the spatial block partition of this level's mesh, and each
    /// block's kernel-slot reach list, computed ONCE at build time.
    ///
    /// Both are pure geometry — [`grid_blocks`]'s own doc says so ("the
    /// partition depends only on the mesh, never on the density, the thread
    /// count or the backend") — and both used to be recomputed inside
    /// [`pairlevel_rho`] AND inside [`pairlevel_pass2`], i.e. twice per level
    /// per density evaluation, i.e. twice per level per SCF cycle.
    /// `block_slots` in particular is `ninst * nblocks` slab-distance tests.
    ///
    /// Bit-exact: the same values in the same order, computed once instead of
    /// four times.
    pub blocks: Vec<GridBlock>,
    /// `block_sel[b]` — [`block_slots`] for `blocks[b]`.
    pub block_sel: Vec<Vec<u32>>,

    /// M-03: every block's table CONCATENATED, so one kernel launch covers the
    /// whole level instead of one launch per block.
    ///
    /// `None` when the concatenation would exceed [`BATCH_BUDGET_BYTES`], in
    /// which case the per-block streaming path is used unchanged — the
    /// fallback D-PBC-26 point 6 and 17-12's own OOM both argue for. The
    /// stored batch carries geometry only; `slot_coef` is filled per call.
    pub batches: Vec<BatchedLevel>,
}

/// The M-03 concatenated launch tables for one level, plus the two maps that
/// scatter a batched result back into mesh / kernel-slot order.
#[derive(Debug)]
pub struct BatchedLevel {
    /// Geometry only. M-12: the per-call coefficients are indexed INSIDE the
    /// kernel through `batch.slot_global`, so nothing per concatenated slot
    /// is built or moved per call.
    pub batch: PairSlotBatch,
    /// Concatenated point index -> this level's mesh grid index.
    pub point_global: Vec<u32>,
    /// M-06: invariant geometry, uploaded lazily on first use.
    pub device: std::sync::OnceLock<PairSlotBatchDevice>,
}

/// The largest concatenated batch M-03 will build, in bytes.
///
/// A block's kernel-slot list contains every slot whose instance REACHES that
/// block, so a diffuse Gaussian appears in many blocks and the concatenation is
/// a multiple — measured at roughly `0.35 * nkslots * nblocks` on the Gate-E
/// cells — of the level's own slot count, not equal to it. That is the same
/// quantity 17-12's `pair_level_tables_stream_under_budget` bounds per launch,
/// and the same reason its predecessor was SIGKILLed: a batch is a memory
/// trade for a launch-count win, and it has to be bounded.
///
/// 256 MiB is 17-12's own per-launch budget, reused rather than reinvented.
pub const BATCH_BUDGET_BYTES: usize = 256 * 1024 * 1024;

impl PairLevelTable {
    pub fn nslots(&self) -> usize {
        self.slot_term.len()
    }
    pub fn nkslots(&self) -> usize {
        self.kslot_term.len()
    }

    /// Number of kernel launches used by either direction for this level.
    /// Batched execution launches once per M-07 chunk; the oversized-block
    /// fallback launches once for each non-empty reach list.
    pub fn launch_count(&self) -> usize {
        if self.batches.is_empty() {
            self.block_sel.iter().filter(|sel| !sel.is_empty()).count()
        } else {
            self.batches.len()
        }
    }
}

#[inline]
fn pairlevel_launch_count(lv: &PairLevelTable) -> usize {
    lv.launch_count()
}

fn wrap_alg(e: AlgebraError) -> PbcDftError {
    PbcDftError::Core(pyscf_core::PyscfRsError::Core(
        pyscf_core::CoreError::InvalidMolecule(format!("multigrid pair collocate: {e}")),
    ))
}

fn backend_client() -> Result<pyscf_algebra::AlgebraClient, PbcDftError> {
    Ok(pyscf_algebra::select_backend()
        .map_err(|e| {
            PbcDftError::Core(pyscf_core::PyscfRsError::Core(
                pyscf_core::CoreError::InvalidMolecule(format!(
                    "multigrid pair collocate: backend selection failed: {e}"
                )),
            ))
        })?
        .client)
}

fn norm3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// 3×3 inverse (row-major), `None` when singular.
fn inv3(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det == 0.0 || !det.is_finite() {
        return None;
    }
    let d = 1.0 / det;
    Some([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * d,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * d,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * d,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * d,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * d,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * d,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * d,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * d,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * d,
        ],
    ])
}

/// `r · inv_a` — the fractional coordinate of a Cartesian point.
fn frac_of(inv_a: &[[f64; 3]; 3], r: [f64; 3]) -> [f64; 3] {
    [0usize, 1, 2].map(|i| r[0] * inv_a[0][i] + r[1] * inv_a[1][i] + r[2] * inv_a[2][i])
}

/// Radius beyond which `cmax · r^kmax · exp(-eta r²)` stays below `thr` —
/// the fused Gaussian's own cutoff (the same `_primitive_gto_cutoff`
/// question, asked of the PRODUCT). Two fixed-point sweeps on the
/// polynomial factor; `0.0` when the whole function is below `thr`.
pub fn fused_radius(cmax: f64, kmax: f64, eta: f64, thr: f64) -> f64 {
    if cmax <= 0.0 || thr <= 0.0 || eta <= 0.0 {
        return 0.0;
    }
    let lnratio = (cmax / thr).ln();
    if lnratio <= 0.0 {
        return 0.0;
    }
    let mut r = (lnratio / eta).sqrt();
    for _ in 0..2 {
        let arg = lnratio + kmax * r.max(1.0).ln();
        r = (arg.max(0.0) / eta).sqrt();
    }
    r
}

/// Distance a fractional point `f` sits outside the fractional box
/// `[lo, hi]`, measured along each reciprocal axis and reduced by `max` —
/// a LOWER bound on the true point-to-parallelepiped distance (each slab
/// distance is one), so "`<= r`" never drops a Gaussian that reaches the
/// box; it may keep a few that do not, which costs kernel work, never
/// accuracy.
fn slab_distance(f: [f64; 3], lo: [f64; 3], hi: [f64; 3], heights: &[f64; 3]) -> f64 {
    let mut d = 0.0f64;
    for i in 0..3 {
        let outside = if f[i] < lo[i] {
            lo[i] - f[i]
        } else if f[i] > hi[i] {
            f[i] - hi[i]
        } else {
            0.0
        };
        d = d.max(outside * heights[i]);
    }
    d
}

/// Every lattice image `P + L1` of a fused centre `P` whose radius-`r`
/// ball can reach the mesh — the periodic wrap of one kernel instance
/// (see [`PairLevelTable`]). `glo`/`ghi` is the mesh's OWN fractional
/// bounding box, measured from its coordinates: this port's uniform grid
/// is origin-centred (`get_uniform_grids`'s `fftfreq`-style fractions in
/// `[-0.5, 0.5)`), NOT `[0,1)` — assuming the latter silently dropped
/// every image on the negative side of the cell and cost 1e-3 of the
/// electron count on the Gate-E cells (found by the per-pair brute-force
/// diagnostic in `tests/multigrid2.rs`). Candidates are the integer box
/// `[floor(glo_i - f_i - r/h_i), ceil(ghi_i - f_i + r/h_i)]` around `P`'s
/// fractional coordinate `f`; each is kept by [`slab_distance`].
fn wrap_images(
    a: &[[f64; 3]; 3],
    inv_a: &[[f64; 3]; 3],
    heights: &[f64; 3],
    glo: [f64; 3],
    ghi: [f64; 3],
    p: [f64; 3],
    r: f64,
) -> Vec<[f64; 3]> {
    let f = frac_of(inv_a, p);
    let lo = [0usize, 1, 2].map(|i| (glo[i] - f[i] - r / heights[i]).floor() as i64);
    let hi = [0usize, 1, 2].map(|i| (ghi[i] - f[i] + r / heights[i]).ceil() as i64);
    let mut out = Vec::new();
    for m0 in lo[0]..=hi[0] {
        for m1 in lo[1]..=hi[1] {
            for m2 in lo[2]..=hi[2] {
                let fi = [f[0] + m0 as f64, f[1] + m1 as f64, f[2] + m2 as f64];
                if slab_distance(fi, glo, ghi, heights) > r {
                    continue;
                }
                let (m0f, m1f, m2f) = (m0 as f64, m1 as f64, m2 as f64);
                out.push([
                    p[0] + m0f * a[0][0] + m1f * a[1][0] + m2f * a[2][0],
                    p[1] + m0f * a[0][1] + m1f * a[1][1] + m2f * a[2][1],
                    p[2] + m0f * a[0][2] + m1f * a[1][2] + m2f * a[2][2],
                ]);
            }
        }
    }
    out
}

/// Upper bound on a fused pair-image's peak magnitude, for PRE-screening
/// only: `K · scale · (1+d)^{l_p+l_q} · max_r r^{kmax} e^{-eta r²}` — the
/// binomial shift can multiply the fused coefficient by up to
/// `|P-A|^{l_p} |P-(B+L)|^{l_q} <= d^{l_p+l_q}` (`d = |A-(B+L)|`), and the
/// monomial `(r-P)^k` peaks at `r² = k/(2 eta)`. Screening on `K·scale`
/// alone dropped far `p-p` images whose NEGATIVE shifted terms carry
/// real weight and left `∫rho` 7e-5 too large on the Gate-E silicon cell;
/// the exact per-instance cutoff ([`fused_radius`] on the true `cmax`)
/// then does the real screening.
fn pair_prescreen_bound(k: f64, scale: f64, d: f64, ltot: i32, eta: f64) -> f64 {
    let kmax = ltot as f64;
    let mono_peak = if kmax > 0.0 {
        (kmax / (2.0 * eta)).powf(kmax / 2.0) * (-kmax / 2.0).exp()
    } else {
        1.0
    };
    k * scale * (1.0 + d).powi(ltot) * mono_peak.max(1.0)
}

/// Build one level's [`PairLevelTable`] for `pairs` on `level`'s own mesh —
/// Task 2's host-side half (the kernel launch is deferred to
/// [`pairlevel_rho`] / [`pairlevel_pass2`], which stream it).
///
/// # Errors
/// Propagates grid construction / lattice-image enumeration; a singular
/// lattice.
pub fn build_pair_level_table(
    cell: &Cell,
    decon: &Decontracted,
    level: &GridLevelSpec,
    pairs: &[(usize, usize)],
) -> Result<PairLevelTable, PbcDftError> {
    let grids = crate::gen_grid::PeriodicGrids::uniform(cell, Some(level.mesh))?;
    let coords = grids.coords()?;
    let ngrids = coords.len();
    let mut coords_flat = Vec::with_capacity(ngrids * 3);
    for c in coords {
        coords_flat.push(c[0]);
        coords_flat.push(c[1]);
        coords_flat.push(c[2]);
    }

    let a = cell.lattice_vectors();
    let inv_a = inv3(&a).ok_or_else(|| {
        PbcDftError::Core(pyscf_core::PyscfRsError::Core(
            pyscf_core::CoreError::InvalidMolecule(
                "multigrid pair collocate: singular lattice".to_string(),
            ),
        ))
    })?;
    // Column `i` of inv(a) is the (unscaled) reciprocal direction; the
    // slab height is `1 / |b_i|`.
    let heights = [0usize, 1, 2].map(|i| 1.0 / norm3([inv_a[0][i], inv_a[1][i], inv_a[2][i]]));
    // The mesh's fractional bounding box, from its coordinates (see
    // [`wrap_images`] for why this is measured rather than assumed).
    let mut glo = [f64::INFINITY; 3];
    let mut ghi = [f64::NEG_INFINITY; 3];
    for g in 0..ngrids {
        let f = frac_of(
            &inv_a,
            [
                coords_flat[g * 3],
                coords_flat[g * 3 + 1],
                coords_flat[g * 3 + 2],
            ],
        );
        for i in 0..3 {
            glo[i] = glo[i].min(f[i]);
            ghi[i] = ghi[i].max(f[i]);
        }
    }

    let mut slot_coef = Vec::new();
    let mut slot_ci = Vec::new();
    let mut slot_cj = Vec::new();
    let mut slot_term = Vec::new();
    let mut nterms = 0usize;
    let mut kslot_pow = Vec::new();
    let mut kslot_instance = Vec::new();
    let mut kslot_term = Vec::new();
    let mut instance_alpha = Vec::new();
    let mut instance_center = Vec::new();
    let mut instance_radius = Vec::new();

    let threshold = cell.precision * EXTRA_PREC;

    for &(pi, pj) in pairs {
        let p: &Pshell = &decon.pshells[pi];
        let q: &Pshell = &decon.pshells[pj];
        let search_rcut = (p.rcut + q.rcut).max(1e-6);
        let ls = get_lattice_ls(cell, Some(search_rcut), None, false)?;
        let powers_i = pshell_cart_powers(p.l);
        let powers_j = pshell_cart_powers(q.l);
        let eta = p.alpha + q.alpha;
        // The AO-product magnitude (`E`'s scales included) — for SCREENING
        // and the cutoff radius only, never stored in a coefficient.
        let scale = (p.coef * q.coef).abs();
        // Per pair-instance monomial lookup: `(k1,k2,k3)` -> fused term,
        // each `k <= l_p + l_q`. Allocated lazily on first non-zero use so
        // a monomial nobody feeds costs no kernel work.
        let kmax = (p.l + q.l + 1) as usize;
        let mut term_of = vec![u32::MAX; kmax * kmax * kmax];
        let mut terms_here: Vec<(u32, [u32; 3])> = Vec::new();

        for l in &ls {
            let bshift = [q.center[0] + l[0], q.center[1] + l[1], q.center[2] + l[2]];
            let d = [
                p.center[0] - bshift[0],
                p.center[1] - bshift[1],
                p.center[2] - bshift[2],
            ];
            let dist2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let k = (-p.alpha * q.alpha / eta * dist2).exp();
            if pair_prescreen_bound(k, scale, dist2.sqrt(), p.l + q.l, eta) < threshold {
                continue;
            }
            let pcen = [
                (p.alpha * p.center[0] + q.alpha * bshift[0]) / eta,
                (p.alpha * p.center[1] + q.alpha * bshift[1]) / eta,
                (p.alpha * p.center[2] + q.alpha * bshift[2]) / eta,
            ];
            let pa = [
                pcen[0] - p.center[0],
                pcen[1] - p.center[1],
                pcen[2] - p.center[2],
            ];
            let pb = [
                pcen[0] - bshift[0],
                pcen[1] - bshift[1],
                pcen[2] - bshift[2],
            ];

            term_of.iter_mut().for_each(|t| *t = u32::MAX);
            terms_here.clear();
            let mut cmax_here = 0.0f64;

            for (ci, &(aix, aiy, aiz)) in powers_i.iter().enumerate() {
                for (cj, &(bjx, bjy, bjz)) in powers_j.iter().enumerate() {
                    let fx = binom_shift(aix, bjx, pa[0], pb[0]);
                    let fy = binom_shift(aiy, bjy, pa[1], pb[1]);
                    let fz = binom_shift(aiz, bjz, pa[2], pb[2]);
                    for (k1, &cx) in fx.iter().enumerate() {
                        if cx == 0.0 {
                            continue;
                        }
                        for (k2, &cy) in fy.iter().enumerate() {
                            if cy == 0.0 {
                                continue;
                            }
                            for (k3, &cz) in fz.iter().enumerate() {
                                if cz == 0.0 {
                                    continue;
                                }
                                let coef = k * cx * cy * cz;
                                if coef == 0.0 {
                                    continue;
                                }
                                cmax_here = cmax_here.max(coef.abs() * scale);
                                let key = (k1 * kmax + k2) * kmax + k3;
                                let t = if term_of[key] == u32::MAX {
                                    let t = nterms as u32;
                                    nterms += 1;
                                    term_of[key] = t;
                                    terms_here.push((t, [k1 as u32, k2 as u32, k3 as u32]));
                                    t
                                } else {
                                    term_of[key]
                                };
                                slot_coef.push(coef);
                                slot_ci.push((p.cart_ao0 + ci) as u32);
                                slot_cj.push((q.cart_ao0 + cj) as u32);
                                slot_term.push(t);
                            }
                        }
                    }
                }
            }
            if terms_here.is_empty() {
                continue;
            }

            // Periodic wrap of the fused Gaussian (see the struct doc). Its
            // own radius — where `|C|·r^k·exp(-eta r²)` drops below the
            // screening threshold — bounds the images, NOT the primitives'
            // radii: `eta >= max(alpha)` and `K <= 1`, so the fused
            // function is at least as compact as its tighter parent.
            let r_inst = fused_radius(cmax_here, (p.l + q.l) as f64, eta, threshold);
            if r_inst <= 0.0 {
                continue;
            }
            for c in wrap_images(&a, &inv_a, &heights, glo, ghi, pcen, r_inst) {
                let inst = instance_alpha.len() as u32;
                instance_alpha.push(eta);
                instance_center.push(c[0]);
                instance_center.push(c[1]);
                instance_center.push(c[2]);
                instance_radius.push(r_inst);
                for &(t, pw) in &terms_here {
                    kslot_pow.push(pw[0]);
                    kslot_pow.push(pw[1]);
                    kslot_pow.push(pw[2]);
                    kslot_instance.push(inst);
                    kslot_term.push(t);
                }
            }
        }
    }

    let mut table = PairLevelTable {
        mesh: level.mesh,
        ngrids,
        coords: coords_flat,
        inv_a,
        heights,
        nterms,
        slot_coef,
        slot_ci,
        slot_cj,
        slot_term,
        kslot_pow,
        kslot_instance,
        kslot_term,
        instance_alpha,
        instance_center,
        instance_radius,
        blocks: Vec::new(),
        block_sel: Vec::new(),
        batches: Vec::new(),
    };
    // M-02: the block partition and the per-block reach lists are geometry, so
    // they are built here, once, rather than inside each direction of each
    // density evaluation. `grid_blocks` / `block_slots` stay public as the
    // builders (and as the seam the `multigrid2.rs` gates measure through).
    table.blocks = grid_blocks(&table);
    table.block_sel = table
        .blocks
        .iter()
        .map(|b| block_slots(&table, b))
        .collect();
    // M-03: the concatenated single-launch tables, when they fit.
    table.batches = build_batched_levels(&table);
    Ok(table)
}

/// Concatenate every block's launch table into one — M-03.
///
/// Mirrors [`block_table`] block by block and slot by slot, so the batched
/// kernel sees each block's instances and slots in exactly the order the
/// per-block launch presented them. Returns `None` above
/// [`BATCH_BUDGET_BYTES`], leaving the caller on the streaming path.
fn batch_counts(lv: &PairLevelTable, range: std::ops::Range<usize>) -> (usize, usize, usize) {
    let mut npoints = 0;
    let mut nslots = 0;
    let mut ninst = 0;
    for bi in range {
        npoints += lv.blocks[bi].points.len();
        nslots += lv.block_sel[bi].len();
        let mut last = u32::MAX;
        for &k in &lv.block_sel[bi] {
            let inst = lv.kslot_instance[k as usize];
            if inst != last {
                ninst += 1;
                last = inst;
            }
        }
    }
    (npoints, nslots, ninst)
}

fn batch_bytes(npoints: usize, nslots: usize, ninst: usize, nblocks: usize) -> usize {
    // Geometry, both varying arrays, both outputs, and the two host scatter maps.
    npoints * (3 * 8 + 4 + 8 + 4)
        + nslots * (3 * 4 + 8 + 8 + 4)
        + ninst * (4 + 8 + 3 * 8 + 4)
        + (nblocks + 1) * 2 * 4
}

/// The most slots any one instance owns in any block's reach list — the
/// quantity [`MAX_SLOTS_PER_INSTANCE`] bounds.
///
/// An instance's slots are the distinct `(k1,k2,k3)` monomials of its pair,
/// so this is `C(l_p+l_q+3, 3)` for the level's widest pair: 10 at
/// `l_p+l_q = 2`, but 20 at 3 and 35 at 4.
fn max_instance_span(lv: &PairLevelTable) -> usize {
    let mut max = 0usize;
    for sel in &lv.block_sel {
        let mut run = 0usize;
        let mut last = u32::MAX;
        for &k in sel {
            let inst = lv.kslot_instance[k as usize];
            if inst == last {
                run += 1;
            } else {
                last = inst;
                run = 1;
            }
            max = max.max(run);
        }
    }
    max
}

fn build_batched_levels(lv: &PairLevelTable) -> Vec<BatchedLevel> {
    // M-08: the batched kernels hold one instance's slot accumulators in a
    // fixed `MAX_SLOTS_PER_INSTANCE`-wide register array. A level whose
    // widest pair reaches `l_p + l_q >= 3` — every polarized basis has
    // p·d pairs — exceeds it, and streams instead. Checked here rather than
    // left to `validate_batch`, which would turn the whole density
    // evaluation into an error at launch time.
    if max_instance_span(lv) > MAX_SLOTS_PER_INSTANCE {
        return Vec::new();
    }
    let mut ranges = Vec::new();
    let mut start = 0;
    while start < lv.blocks.len() {
        let mut end = start;
        while end < lv.blocks.len() {
            let (np, ns, ni) = batch_counts(lv, start..end + 1);
            if batch_bytes(np, ns, ni, end + 1 - start) > BATCH_BUDGET_BYTES {
                break;
            }
            end += 1;
        }
        if end == start {
            // The sole allowed streaming fallback: one block cannot fit.
            return Vec::new();
        }
        ranges.push(start..end);
        start = end;
    }
    ranges
        .into_iter()
        .filter_map(|range| build_batched_range(lv, range))
        .collect()
}

fn build_batched_range(lv: &PairLevelTable, range: std::ops::Range<usize>) -> Option<BatchedLevel> {
    let (npoints, nslots, _) = batch_counts(lv, range.clone());
    if npoints == 0 || nslots == 0 {
        return None;
    }

    let mut b = PairSlotBatch {
        coords_x: Vec::with_capacity(npoints),
        coords_y: Vec::with_capacity(npoints),
        coords_z: Vec::with_capacity(npoints),
        point_block: Vec::with_capacity(npoints),
        block_point0: Vec::with_capacity(range.len() + 1),
        block_inst0: Vec::with_capacity(range.len() + 1),
        inst_block: Vec::new(),
        instance_alpha: Vec::new(),
        instance_center: Vec::new(),
        // Filled below: one START per instance, then the final end, which is
        // exactly the `ninst + 1` prefix shape the kernel indexes.
        inst_slot0: Vec::new(),
        slot_global: Vec::with_capacity(nslots),
        // M-12: the level's packed powers, once per KERNEL slot — shared by
        // every chunk of the level (a `nkslots · 4` B copy per chunk).
        kslot_pow: (0..lv.nkslots())
            .map(|k| {
                let ix = lv.kslot_pow[k * 3];
                let iy = lv.kslot_pow[k * 3 + 1];
                let iz = lv.kslot_pow[k * 3 + 2];
                debug_assert!(ix < 256 && iy < 256 && iz < 256);
                ix | (iy << 8) | (iz << 16)
            })
            .collect(),
    };
    let mut point_global = Vec::with_capacity(npoints);

    b.block_point0.push(0);
    b.block_inst0.push(0);
    for (local_bi, global_bi) in range.enumerate() {
        let block = &lv.blocks[global_bi];
        let sel = &lv.block_sel[global_bi];
        for &g in &block.points {
            let g = g as usize;
            b.coords_x.push(lv.coords[g * 3]);
            b.coords_y.push(lv.coords[g * 3 + 1]);
            b.coords_z.push(lv.coords[g * 3 + 2]);
            b.point_block.push(local_bi as u32);
            point_global.push(g as u32);
        }
        // Same instance grouping `block_table` performs: a new instance opens
        // whenever the table-order slot list moves to a different one.
        let mut last_inst = u32::MAX;
        for &k in sel {
            let k = k as usize;
            let inst = lv.kslot_instance[k];
            if inst != last_inst {
                last_inst = inst;
                let i = inst as usize;
                b.instance_alpha.push(lv.instance_alpha[i]);
                b.instance_center
                    .extend_from_slice(&lv.instance_center[i * 3..i * 3 + 3]);
                b.inst_block.push(local_bi as u32);
                // This instance's slots start at the running slot count.
                b.inst_slot0.push(b.slot_global.len() as u32);
            }
            b.slot_global.push(k as u32);
        }
        b.block_point0.push(b.point_block.len() as u32);
        b.block_inst0.push(b.instance_alpha.len() as u32);
    }
    // One START was pushed per instance; the final end closes the prefix.
    b.inst_slot0.push(b.slot_global.len() as u32);
    debug_assert_eq!(b.inst_slot0.len(), b.instance_alpha.len() + 1);
    debug_assert_eq!(b.inst_slot0[0], 0);
    debug_assert!(
        b.inst_slot0
            .windows(2)
            .all(|w| (w[1] - w[0]) as usize <= MAX_SLOTS_PER_INSTANCE),
        "M-08 reverse register bound exceeded"
    );
    Some(BatchedLevel {
        batch: b,
        point_global,
        device: std::sync::OnceLock::new(),
    })
}

fn resident_batch<'a>(
    bl: &'a BatchedLevel,
    client: &pyscf_algebra::AlgebraClient,
) -> Result<&'a PairSlotBatchDevice, PbcDftError> {
    if let Some(device) = bl.device.get() {
        return Ok(device);
    }
    let candidate = PairSlotBatchDevice::new(client, &bl.batch).map_err(wrap_alg)?;
    // `get_or_init` rather than `set` + a second `get`: a concurrent caller
    // that won the race keeps its upload and both callers see it, so there is
    // no "initialization failed" state to invent an error for.
    Ok(bl.device.get_or_init(move || candidate))
}

/// Every non-empty level's [`PairLevelTable`], built ONCE per density
/// evaluation and shared by the forward (`rho`) and reverse (`pass2`)
/// directions; `None` where the level owns no pair.
///
/// # Errors
/// Propagates [`build_pair_level_table`].
pub fn build_pair_level_tables(
    cell: &Cell,
    decon: &Decontracted,
    task_list: &PairTaskList,
) -> Result<Vec<Option<PairLevelTable>>, PbcDftError> {
    task_list
        .levels
        .iter()
        .zip(task_list.level_pairs.iter())
        .map(|(level, pairs)| {
            if pairs.is_empty() {
                Ok(None)
            } else {
                build_pair_level_table(cell, decon, level, pairs).map(Some)
            }
        })
        .collect()
}

/// One spatial block of a level's mesh: an index box `[i0,i1)×[j0,j1)×
/// [k0,k1)` (x slowest, z fastest — `get_uniform_grids`'s order), its
/// point list, and its fractional bounding box.
#[derive(Debug)]
pub struct GridBlock {
    pub points: Vec<u32>,
    pub flo: [f64; 3],
    pub fhi: [f64; 3],
}

/// Partition `lv`'s mesh into [`GridBlock`]s of about [`BLOCK_EDGE`]
/// points per axis. Pure geometry — the partition depends only on the
/// mesh, never on the density, the thread count or the backend.
pub fn grid_blocks(lv: &PairLevelTable) -> Vec<GridBlock> {
    let [nx, ny, nz] = lv.mesh;
    let nb = |n: usize| n.div_ceil(BLOCK_EDGE).max(1);
    let edges = |n: usize| -> Vec<(usize, usize)> {
        let b = nb(n);
        (0..b)
            .map(|i| (i * n / b, (i + 1) * n / b))
            .filter(|(lo, hi)| hi > lo)
            .collect()
    };
    let mut blocks = Vec::new();
    for &(x0, x1) in &edges(nx) {
        for &(y0, y1) in &edges(ny) {
            for &(z0, z1) in &edges(nz) {
                let mut points = Vec::with_capacity((x1 - x0) * (y1 - y0) * (z1 - z0));
                let mut flo = [f64::INFINITY; 3];
                let mut fhi = [f64::NEG_INFINITY; 3];
                for ix in x0..x1 {
                    for iy in y0..y1 {
                        for iz in z0..z1 {
                            let g = (ix * ny + iy) * nz + iz;
                            points.push(g as u32);
                            let r = [lv.coords[g * 3], lv.coords[g * 3 + 1], lv.coords[g * 3 + 2]];
                            let f = frac_of(&lv.inv_a, r);
                            for i in 0..3 {
                                flo[i] = flo[i].min(f[i]);
                                fhi[i] = fhi[i].max(f[i]);
                            }
                        }
                    }
                }
                if !points.is_empty() {
                    blocks.push(GridBlock { points, flo, fhi });
                }
            }
        }
    }
    blocks
}

/// The kernel slots (in table order) whose instance's cutoff ball can
/// reach `block` — the per-block analogue of upstream's per-Gaussian
/// rcut sub-mesh. Everything left out is below the screening threshold on
/// every point of the block by construction of [`fused_radius`].
pub fn block_slots(lv: &PairLevelTable, block: &GridBlock) -> Vec<u32> {
    let ninst = lv.instance_alpha.len();
    let mut reach = vec![false; ninst];
    for (i, r) in reach.iter_mut().enumerate() {
        let c = [
            lv.instance_center[i * 3],
            lv.instance_center[i * 3 + 1],
            lv.instance_center[i * 3 + 2],
        ];
        let f = frac_of(&lv.inv_a, c);
        *r = slab_distance(f, block.flo, block.fhi, &lv.heights) <= lv.instance_radius[i];
    }
    (0..lv.nkslots() as u32)
        .filter(|&k| reach[lv.kslot_instance[k as usize] as usize])
        .collect()
}

/// The kernel-facing sub-table for one block: the selected kernel slots
/// `sel` (table order, hence still grouped by instance) with `coef` as
/// the per-slot coefficient, the instances they touch renumbered densely,
/// and the block's own point coordinates.
fn block_table(lv: &PairLevelTable, block: &GridBlock, sel: &[u32], coef: &[f64]) -> PairSlotTable {
    let nsel = sel.len();
    let mut table = PairSlotTable {
        coords: Vec::with_capacity(block.points.len() * 3),
        slot_pow: Vec::with_capacity(nsel * 3),
        slot_coef: Vec::with_capacity(nsel),
        slot_instance: Vec::with_capacity(nsel),
        instance_alpha: Vec::new(),
        instance_center: Vec::new(),
    };
    let mut last_inst = u32::MAX;
    for &k in sel {
        let k = k as usize;
        let inst = lv.kslot_instance[k];
        if inst != last_inst {
            last_inst = inst;
            let i = inst as usize;
            table.instance_alpha.push(lv.instance_alpha[i]);
            table
                .instance_center
                .extend_from_slice(&lv.instance_center[i * 3..i * 3 + 3]);
        }
        table
            .slot_pow
            .extend_from_slice(&lv.kslot_pow[k * 3..k * 3 + 3]);
        table.slot_coef.push(coef[k]);
        table
            .slot_instance
            .push(table.instance_alpha.len() as u32 - 1);
    }
    for &g in &block.points {
        let g = g as usize;
        table.coords.extend_from_slice(&lv.coords[g * 3..g * 3 + 3]);
    }
    table
}

/// `grid_collocate_drv`'s forward direction at one level —
/// `rho(r) = Σ_k C_{term(k)} · kslot_k(r)`, `C_t` the density-contracted
/// fused-term coefficient (see [`PairLevelTable`]).
///
/// Streamed block by block ([`grid_blocks`] / [`block_slots`]) through
/// `collocate_pairs_rho`, whose per-point sum runs over the kernel slots
/// that reach the block, in table order — a list fixed by geometry alone,
/// so the result is bit-identical under any rayon thread count or launch
/// geometry (D-PBC-17). **Ordering note:** that in-kernel sum is
/// sequential, not `oracle_sum`'s pairwise tree; the pairwise host
/// reduction needed every `(slot × point)` value materialised, which is
/// the memory shape this plan could not afford (`17-12-SUMMARY.md`).
///
/// # Errors
/// Propagates backend selection / the kernel's shape checks.
pub fn pairlevel_rho(
    lv: &PairLevelTable,
    decon: &Decontracted,
    dm_p: &[f64],
) -> Result<Vec<f64>, PbcDftError> {
    pairlevel_rho_with(lv, decon, dm_p, true)
}

/// [`pairlevel_rho`] with M-03's batched launch explicitly on or off.
///
/// The seam exists so the two routes can be compared IN ONE PROCESS at
/// `to_bits()` equality (`tests/multigrid_batch.rs`). M-03 claims bit-identity
/// by construction — same slot list, same order, same lane ownership — and a
/// claim like that is worth exactly as much as the test that checks it.
/// Production always passes `true`; `false` is the streaming fallback the
/// budget guard also selects.
///
/// # Errors
/// As [`pairlevel_rho`].
pub fn pairlevel_rho_with(
    lv: &PairLevelTable,
    decon: &Decontracted,
    dm_p: &[f64],
    use_batch: bool,
) -> Result<Vec<f64>, PbcDftError> {
    let ngrids = lv.ngrids;
    let nk = lv.nkslots();
    let mut rho = vec![0.0f64; ngrids];
    if nk == 0 || ngrids == 0 {
        return Ok(rho);
    }
    // Contract the density matrix into the fused terms — fixed slot order.
    let mut term_coef = vec![0.0f64; lv.nterms];
    for s in 0..lv.nslots() {
        let d = dm_p[lv.slot_ci[s] as usize * decon.nao_p + lv.slot_cj[s] as usize];
        term_coef[lv.slot_term[s] as usize] += d * lv.slot_coef[s];
    }
    let kcoef: Vec<f64> = lv
        .kslot_term
        .iter()
        .map(|&t| term_coef[t as usize])
        .collect();

    let client = backend_client()?;

    // M-03: ONE launch for the whole level when the concatenated tables fit.
    // Bit-identical to the per-block route below — each lane runs the same
    // slot list in the same order, and every output is written by one lane.
    if use_batch && !lv.batches.is_empty() {
        for bl in &lv.batches {
            // M-12: the per-kernel-slot coefficients go up as they are; the
            // kernel indexes them through the resident `slot_global`.
            let out = resident_batch(bl, &client)?
                .rho(&client, &kcoef)
                .map_err(wrap_alg)?;
            for (p, v) in out.into_iter().enumerate() {
                rho[bl.point_global[p] as usize] = v;
            }
        }
        return Ok(rho);
    }

    // M-02: `lv.blocks` / `lv.block_sel` instead of recomputing the partition
    // and every block's reach list on each call. Same blocks, same order.
    for (block, sel) in lv.blocks.iter().zip(&lv.block_sel) {
        if sel.is_empty() {
            continue;
        }
        let table = block_table(lv, block, sel, &kcoef);
        let out = collocate_pairs_rho(&client, &table).map_err(wrap_alg)?;
        for (pi, v) in out.into_iter().enumerate() {
            rho[block.points[pi] as usize] = v;
        }
    }
    Ok(rho)
}

/// Two-spin forward sweep. Batched tables traverse geometry once; the
/// streaming fallback retains the two proven single-channel paths.
pub fn pairlevel_rho2(
    lv: &PairLevelTable,
    decon: &Decontracted,
    dm_p: [&[f64]; 2],
) -> Result<[Vec<f64>; 2], PbcDftError> {
    if lv.batches.is_empty() {
        // The two channels share no state, so the fallback costs one channel's
        // wall time rather than two — the batched route's own advantage.
        let (a, b) = rayon::join(
            || pairlevel_rho_with(lv, decon, dm_p[0], false),
            || pairlevel_rho_with(lv, decon, dm_p[1], false),
        );
        return Ok([a?, b?]);
    }
    let mut term_coef = [vec![0.0f64; lv.nterms], vec![0.0f64; lv.nterms]];
    for s in 0..lv.nslots() {
        let idx = lv.slot_ci[s] as usize * decon.nao_p + lv.slot_cj[s] as usize;
        let t = lv.slot_term[s] as usize;
        term_coef[0][t] += dm_p[0][idx] * lv.slot_coef[s];
        term_coef[1][t] += dm_p[1][idx] * lv.slot_coef[s];
    }
    let kcoef = term_coef.map(|tc| {
        lv.kslot_term
            .iter()
            .map(|&t| tc[t as usize])
            .collect::<Vec<_>>()
    });
    let client = backend_client()?;
    let mut rho = [vec![0.0; lv.ngrids], vec![0.0; lv.ngrids]];
    for bl in &lv.batches {
        let out = resident_batch(bl, &client)?
            .rho2(&client, [&kcoef[0], &kcoef[1]])
            .map_err(wrap_alg)?;
        for spin in 0..2 {
            for (p, &v) in out[spin].iter().enumerate() {
                rho[spin][bl.point_global[p] as usize] = v;
            }
        }
    }
    Ok(rho)
}

/// `grid_integrate_drv`'s reverse direction ("pass2") at one level —
/// `v_p[ci,cj] += Σ_{s→t} coef_s · I_t`, `I_t = Σ_{k→t} Σ_r w[r]·kslot_k(r)`.
/// ADDS into `v_p`, does not overwrite.
///
/// Streamed block by block through `collocate_pairs_integrate`: every
/// kernel slot's grid integral is the fixed-order sum of its per-block
/// partial sums (blocks visited in [`grid_blocks`]'s order, points within
/// a block in mesh order) — a property of the mesh and the table, not of
/// the thread count (D-PBC-17; same ordering note as [`pairlevel_rho`]).
/// The image fold `Σ_{k→t}` and the final scatter into `v_p` run in fixed
/// table order.
///
/// # Errors
/// Propagates backend selection / the kernel's shape checks.
pub fn pairlevel_pass2(
    lv: &PairLevelTable,
    decon: &Decontracted,
    weight: &[f64],
    v_p: &mut [f64],
) -> Result<(), PbcDftError> {
    pairlevel_pass2_with(lv, decon, weight, v_p, true)
}

/// [`pairlevel_pass2`] with M-03's batched launch explicitly on or off — the
/// reverse-direction twin of [`pairlevel_rho_with`], and for the same reason.
///
/// # Errors
/// As [`pairlevel_pass2`].
pub fn pairlevel_pass2_with(
    lv: &PairLevelTable,
    decon: &Decontracted,
    weight: &[f64],
    v_p: &mut [f64],
    use_batch: bool,
) -> Result<(), PbcDftError> {
    debug_assert_eq!(weight.len(), lv.ngrids);
    let nk = lv.nkslots();
    if nk == 0 || lv.ngrids == 0 {
        return Ok(());
    }
    let ones = vec![1.0f64; nk];
    let mut kint = vec![0.0f64; nk];

    let client = backend_client()?;

    // M-03 — see `pairlevel_rho`. The host fold below visits the concatenated
    // slots in block-major, within-block-table order, which is exactly the
    // order the per-block loop accumulated in (D-PBC-17's fixed-order
    // property, preserved).
    let mut batched = false;
    if use_batch && !lv.batches.is_empty() {
        for bl in &lv.batches {
            let w: Vec<f64> = bl
                .point_global
                .iter()
                .map(|&g| weight[g as usize])
                .collect();
            let out = resident_batch(bl, &client)?
                .integrate(&client, &w)
                .map_err(wrap_alg)?;
            for (s, v) in out.into_iter().enumerate() {
                kint[bl.batch.slot_global[s] as usize] += v;
            }
        }
        batched = true;
    }

    // M-02 — see `pairlevel_rho`.
    if !batched {
        for (block, sel) in lv.blocks.iter().zip(&lv.block_sel) {
            if sel.is_empty() {
                continue;
            }
            let table = block_table(lv, block, sel, &ones);
            let w: Vec<f64> = block.points.iter().map(|&g| weight[g as usize]).collect();
            let out = collocate_pairs_integrate(&client, &table, &w).map_err(wrap_alg)?;
            for (j, v) in out.into_iter().enumerate() {
                kint[sel[j] as usize] += v;
            }
        }
    }

    let mut integrals = vec![0.0f64; lv.nterms];
    for k in 0..nk {
        integrals[lv.kslot_term[k] as usize] += kint[k];
    }
    for s in 0..lv.nslots() {
        let idx = lv.slot_ci[s] as usize * decon.nao_p + lv.slot_cj[s] as usize;
        v_p[idx] += lv.slot_coef[s] * integrals[lv.slot_term[s] as usize];
    }
    Ok(())
}

/// Two-spin reverse sweep sharing each resident geometry traversal.
pub fn pairlevel_pass2_2(
    lv: &PairLevelTable,
    decon: &Decontracted,
    weight: [&[f64]; 2],
    v_p: [&mut [f64]; 2],
) -> Result<(), PbcDftError> {
    let [vpa, vpb] = v_p;
    if lv.batches.is_empty() {
        // `vpa` and `vpb` are disjoint, so the two channels run concurrently
        // for the same reason `pairlevel_rho2`'s fallback does.
        let (a, b) = rayon::join(
            || pairlevel_pass2_with(lv, decon, weight[0], vpa, false),
            || pairlevel_pass2_with(lv, decon, weight[1], vpb, false),
        );
        a?;
        b?;
        return Ok(());
    }
    let client = backend_client()?;
    let mut kint = [vec![0.0f64; lv.nkslots()], vec![0.0f64; lv.nkslots()]];
    for bl in &lv.batches {
        let w = weight.map(|src| {
            bl.point_global
                .iter()
                .map(|&g| src[g as usize])
                .collect::<Vec<_>>()
        });
        let out = resident_batch(bl, &client)?
            .integrate2(&client, [&w[0], &w[1]])
            .map_err(wrap_alg)?;
        for spin in 0..2 {
            for (s, &v) in out[spin].iter().enumerate() {
                kint[spin][bl.batch.slot_global[s] as usize] += v;
            }
        }
    }
    let mut integrals = [vec![0.0f64; lv.nterms], vec![0.0f64; lv.nterms]];
    for spin in 0..2 {
        for k in 0..lv.nkslots() {
            integrals[spin][lv.kslot_term[k] as usize] += kint[spin][k];
        }
        for s in 0..lv.nslots() {
            let idx = lv.slot_ci[s] as usize * decon.nao_p + lv.slot_cj[s] as usize;
            let target = if spin == 0 { &mut *vpa } else { &mut *vpb };
            target[idx] += lv.slot_coef[s] * integrals[spin][lv.slot_term[s] as usize];
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Task 5 — `MultiGridNumInt2` assembly.
// ---------------------------------------------------------------------

use pyscf_algebra::CTensor;
use pyscf_pbc_gto::{get_coulg_at_gv, get_gv};

use crate::multigrid::numint::{extract_gspace_window, insert_gspace_window, mg_xc_parts};
use crate::multigrid::tasks;

/// `pbc.dft.multigrid.multigrid_pair.MultiGridNumInt` (re-exported upstream
/// as `MultiGridNumInt2`, `__init__.py:18`) — the v2 multigrid driver
/// (gamma point; see the module doc). **This is the class Phase 18's
/// `grad/rhf.py:44` / `grad/uhf.py:40` `assert isinstance(ni,
/// MultiGridNumInt2)` on — NOT `crate::multigrid::MultiGridNumInt` (v1)** —
/// recorded in `PBC-MASTER-PLAN.md §8.10` by this plan.
#[derive(Debug, Default)]
pub struct MultiGridNumInt2 {
    prepared: std::sync::Mutex<Option<(u64, std::sync::Arc<V2Tasks>)>>,
}

/// The cell-independent geometry one v2 density evaluation needs — M-02.
///
/// The decontraction, plus every non-empty level's [`PairLevelTable`]
/// (including, since M-02, that level's block partition and per-block reach
/// lists). All of it is a pure function of the cell, and all of it was rebuilt
/// on every call: `build_pair_level_tables` re-runs `get_lattice_ls` and the
/// full binomial-shift image enumeration for EVERY pshell pair, which on the
/// `25^3` reference cells is the bulk of the 7-9 s 17-12 measured per density
/// evaluation.
type V2Tasks = (Decontracted, Vec<Option<PairLevelTable>>);

/// [`MultiGridNumInt2::nr_rks`]'s return — same shape as v1's
/// `MgNrRksResult`.
#[derive(Debug, Clone)]
pub struct Mg2NrRksResult {
    pub nelec: f64,
    pub exc: f64,
    pub ecoul: f64,
    /// `nao x nao` row-major.
    pub veff: Vec<f64>,
}

/// [`MultiGridNumInt2::nr_uks`]'s return — the open-shell twin of
/// [`Mg2NrRksResult`].
#[derive(Debug, Clone)]
pub struct Mg2NrUksResult {
    /// `(n_alpha, n_beta)` from the numerical integration.
    pub nelec: (f64, f64),
    pub exc: f64,
    pub ecoul: f64,
    /// `[alpha, beta]`, each `nao x nao` row-major.
    pub veff: [Vec<f64>; 2],
}

impl MultiGridNumInt2 {
    pub fn new() -> Self {
        Self::default()
    }

    /// Drop the cached geometry — the `KNumInt::reset` idiom. A different cell
    /// is detected automatically by
    /// [`crate::multigrid::utils::cell_fingerprint`].
    pub fn reset(&self) {
        if let Ok(mut g) = self.prepared.lock() {
            *g = None;
        }
    }

    /// Decontract, build the pair task list, and build every non-empty
    /// level's [`PairLevelTable`] ONCE — shared by the forward and reverse
    /// directions of one density evaluation.
    fn build_tasks(&self, cell: &Cell) -> Result<std::sync::Arc<V2Tasks>, PbcDftError> {
        let key = crate::multigrid::utils::cell_fingerprint(cell);
        if let Ok(g) = self.prepared.lock()
            && let Some((k, v)) = g.as_ref()
            && *k == key
        {
            let _span = tracing::info_span!("pbc_mg_build_tasks_hit").entered();
            return Ok(std::sync::Arc::clone(v));
        }
        let _span = tracing::info_span!("pbc_mg_build_tasks_miss").entered();
        let decon = tasks::build_pshells(cell)?;
        let task_list = build_pair_task_list(cell, &decon)?;
        let tables = build_pair_level_tables(cell, &decon, &task_list)?;
        let out = std::sync::Arc::new((decon, tables));
        if let Ok(mut g) = self.prepared.lock() {
            *g = Some((key, std::sync::Arc::clone(&out)));
        }
        Ok(out)
    }

    /// `get_nuc(mydf)` — delegated to AFTDF, see `crate::multigrid::pp`.
    ///
    /// # Errors
    /// Propagates `crate::multigrid::pp::get_nuc`.
    pub fn get_nuc(&self, cell: &Cell) -> Result<Vec<f64>, PbcDftError> {
        crate::multigrid::pp::get_nuc(cell)
    }

    /// `get_pp(mydf)` — delegated to AFTDF, see `crate::multigrid::pp`.
    ///
    /// # Errors
    /// Propagates `crate::multigrid::pp::get_pp`.
    pub fn get_pp(&self, cell: &Cell) -> Result<Vec<f64>, PbcDftError> {
        crate::multigrid::pp::get_pp(cell)
    }

    /// `rho(G)` on `cell.mesh`, combined from every grid level's pair-fused
    /// collocation — the v2 analogue of `MultiGridNumInt::eval_rho_g`.
    ///
    /// # Errors
    /// Propagates task-list / collocation / FFT construction.
    pub fn eval_rho_g(&self, cell: &Cell, dm: &[f64]) -> Result<CTensor, PbcDftError> {
        let prep = self.build_tasks(cell)?;
        let (decon, tables) = (&prep.0, &prep.1);
        let dm_p = crate::multigrid::colloc::expand_dm(decon, dm);
        rho_g_from_pair_levels(cell, decon, tables, &dm_p)
    }

    /// [`MultiGridNumInt2::eval_rho_g`] over a caller-supplied
    /// [`PairTaskList`] — the diagnostic seam that lets a test re-run the
    /// same density with a different level assignment (e.g. every pair on
    /// the finest mesh) and attribute a residual to the level ladder
    /// rather than to the collocation.
    ///
    /// # Errors
    /// As [`MultiGridNumInt2::eval_rho_g`].
    pub fn eval_rho_g_with_task_list(
        &self,
        cell: &Cell,
        task_list: &PairTaskList,
        dm: &[f64],
    ) -> Result<CTensor, PbcDftError> {
        let decon = tasks::build_pshells(cell)?;
        let tables = build_pair_level_tables(cell, &decon, task_list)?;
        let dm_p = crate::multigrid::colloc::expand_dm(&decon, dm);
        rho_g_from_pair_levels(cell, &decon, &tables, &dm_p)
    }

    /// `get_j_kpts` at gamma — the v2 analogue of `MultiGridNumInt::get_j`.
    ///
    /// # Errors
    /// Propagates task-list / collocation / FFT construction.
    pub fn get_j(&self, cell: &Cell, dm: &[f64]) -> Result<Vec<f64>, PbcDftError> {
        let prep = self.build_tasks(cell)?;
        let (decon, tables) = (&prep.0, &prep.1);
        let dm_p = crate::multigrid::colloc::expand_dm(decon, dm);
        let rho_g = rho_g_from_pair_levels(cell, decon, tables, &dm_p)?;

        let mesh = cell.mesh;
        let gv = get_gv(cell, Some(mesh))?;
        let coulg = get_coulg_at_gv(cell, mesh, &gv)?;
        let mut vg = rho_g;
        for ((re, im), c) in vg.re.iter_mut().zip(vg.im.iter_mut()).zip(&coulg) {
            *re *= c;
            *im *= c;
        }
        let v_p = pass2_from_full_vg_pair(cell, decon, tables, &vg)?;
        Ok(crate::multigrid::colloc::contract_v(decon, &v_p))
    }

    /// `nr_rks(mydf, xc_code, dm, with_j=True)` at gamma — the v2 analogue
    /// of `MultiGridNumInt::nr_rks`. Same GGA route (upstream default
    /// `RHOG_HIGH_ORDER=False`: grad rho from G-space, `wv[0] -=
    /// i·Gv·wv[1:4]` fold) `crate::multigrid::numint::MultiGridNumInt::nr_rks`
    /// already documents.
    ///
    /// # Errors
    /// Propagates task-list / collocation / FFT / XC evaluation.
    #[allow(clippy::needless_range_loop)]
    pub fn nr_rks(
        &self,
        cell: &Cell,
        xc_code: &str,
        dm: &[f64],
    ) -> Result<Mg2NrRksResult, PbcDftError> {
        let prep = self.build_tasks(cell)?;
        let (decon, tables) = (&prep.0, &prep.1);
        let dm_p = crate::multigrid::colloc::expand_dm(decon, dm);
        let rho_g = rho_g_from_pair_levels(cell, decon, tables, &dm_p)?;

        // M-00: the middle — Coulomb, XC, the GGA G-space fold, the ordered
        // energy reductions — is shared with v1 and with `nr_uks`. See
        // `crate::multigrid::numint::mg_xc_parts`.
        let parts = mg_xc_parts(cell, xc_code, std::slice::from_ref(&rho_g))?;
        let v_p = pass2_from_full_vg_pair(cell, decon, tables, &parts.wv_freq0[0])?;
        let veff = crate::multigrid::colloc::contract_v(decon, &v_p);

        Ok(Mg2NrRksResult {
            nelec: parts.nelec[0],
            exc: parts.exc,
            ecoul: parts.ecoul,
            veff,
        })
    }

    /// `nr_uks(mydf, xc_code, [dm_a, dm_b], with_j=True)` at gamma — the v2
    /// analogue of [`crate::multigrid::MultiGridNumInt::nr_uks`]. **M-00.**
    ///
    /// Same structure and the same shared middle: the Coulomb term is built
    /// from the spin-summed density and both channels receive the same `vG`;
    /// only the XC evaluation and the two `pass2` sweeps are per spin. The
    /// pair tables are built once (M-02) and used by all four sweeps.
    ///
    /// # Errors
    /// Propagates task-list / collocation / FFT / XC evaluation.
    pub fn nr_uks(
        &self,
        cell: &Cell,
        xc_code: &str,
        dm: &[&[f64]; 2],
    ) -> Result<Mg2NrUksResult, PbcDftError> {
        let prep = self.build_tasks(cell)?;
        let (decon, tables) = (&prep.0, &prep.1);

        let dm_p = dm.map(|d| crate::multigrid::colloc::expand_dm(decon, d));
        let rho_g = rho_g_from_pair_levels2(cell, decon, tables, [&dm_p[0], &dm_p[1]])?;

        let parts = mg_xc_parts(cell, xc_code, &rho_g)?;
        let v_p = pass2_from_full_vg_pair2(
            cell,
            decon,
            tables,
            [&parts.wv_freq0[0], &parts.wv_freq0[1]],
        )?;
        let veff = v_p.map(|v| crate::multigrid::colloc::contract_v(decon, &v));

        Ok(Mg2NrUksResult {
            nelec: (parts.nelec[0], parts.nelec[1]),
            exc: parts.exc,
            ecoul: parts.ecoul,
            veff,
        })
    }
}

/// The per-level tracing span both directions of both spin counts open.
fn batch_geometry_bytes(batch: &PairSlotBatch) -> u64 {
    let f64_bytes = batch
        .coords_x
        .len()
        .saturating_add(batch.coords_y.len())
        .saturating_add(batch.coords_z.len())
        .saturating_add(batch.instance_alpha.len())
        .saturating_add(batch.instance_center.len())
        .saturating_mul(core::mem::size_of::<f64>());
    let u32_bytes = batch
        .point_block
        .len()
        .saturating_add(batch.block_point0.len())
        .saturating_add(batch.block_inst0.len())
        .saturating_add(batch.inst_block.len())
        .saturating_add(batch.inst_slot0.len())
        .saturating_add(batch.slot_global.len())
        .saturating_add(batch.kslot_pow.len())
        .saturating_mul(core::mem::size_of::<u32>());
    f64_bytes.saturating_add(u32_bytes) as u64
}

/// RULE T transfer model for the resident M-06 path. `before` is the former
/// per-call upload of invariant geometry plus a zero output, varying inputs,
/// and the read-back; `after` retains only varying inputs and read-backs.
fn pair_level_transfer_bytes(
    direction: &'static str,
    channels: usize,
    lv: &PairLevelTable,
) -> (u64, u64) {
    lv.batches
        .iter()
        .fold((0_u64, 0_u64), |(before, after), bl| {
            let batch = &bl.batch;
            let geometry = batch_geometry_bytes(batch);
            // M-12: the per-call coefficient is per KERNEL slot (forward) and
            // gone (reverse). `before` keeps the pre-M-06 model — the
            // per-concatenated-slot coefficient — so the column stays the
            // same reference it has been.
            let coef_before = batch.nslots().saturating_mul(8).saturating_mul(channels) as u64;
            let coef_after = batch.nkslots().saturating_mul(8).saturating_mul(channels) as u64;
            let points = batch.npoints().saturating_mul(8).saturating_mul(channels) as u64;
            let slots = batch.nslots().saturating_mul(8).saturating_mul(channels) as u64;
            let (varying_before, varying, output) = if direction == "forward" {
                (coef_before, coef_after, points)
            } else {
                (coef_before.saturating_add(points), points, slots)
            };
            let after_chunk = varying.saturating_add(output); // one read-back
            let before_chunk = geometry
                .saturating_add(varying_before)
                .saturating_add(output) // zero upload
                .saturating_add(output); // read-back
            (
                before.saturating_add(before_chunk),
                after.saturating_add(after_chunk),
            )
        })
}

fn pair_level_span(
    direction: &'static str,
    level: usize,
    channels: usize,
    lv: &PairLevelTable,
) -> EnteredSpan {
    let launches = pairlevel_launch_count(lv) as u64;
    let (transfer_bytes_before, transfer_bytes_after) =
        pair_level_transfer_bytes(direction, channels, lv);
    // Session 3: the concatenated table sizes, so the instrument can show
    // where the resident bytes go (per-slot data dominates).
    let batch_points = lv
        .batches
        .iter()
        .map(|b| b.batch.npoints() as u64)
        .sum::<u64>();
    let batch_instances = lv
        .batches
        .iter()
        .map(|b| b.batch.ninstances() as u64)
        .sum::<u64>();
    let batch_slots = lv
        .batches
        .iter()
        .map(|b| b.batch.nslots() as u64)
        .sum::<u64>();
    let kslots = lv.nkslots() as u64;
    match direction {
        "forward" => tracing::info_span!(
            "pbc_mg_forward_level",
            level = level as u64,
            launches,
            transfer_bytes_before,
            transfer_bytes_after,
            batch_points,
            batch_instances,
            batch_slots,
            kslots,
            mesh = ?lv.mesh,
        ),
        _ => tracing::info_span!(
            "pbc_mg_reverse_level",
            level = level as u64,
            launches,
            transfer_bytes_before,
            transfer_bytes_after,
            batch_points,
            batch_instances,
            batch_slots,
            kslots,
            mesh = ?lv.mesh,
        ),
    }
    .entered()
}

/// Transform one level's real-space `rho` to G-space, apply the level's grid
/// weight, and fold the result into the full-mesh `rho(G)`.
fn insert_level_rho_g(
    rho_r: Vec<f64>,
    lv: &PairLevelTable,
    level: usize,
    vol: f64,
    mesh: [usize; 3],
    rho_g: &mut CTensor,
) -> Result<(), PbcDftError> {
    let rr = CTensor::from_planes(rho_r, vec![0.0; lv.ngrids]);
    let mut freq = {
        let _fft_span =
            tracing::info_span!("pbc_mg_fft", level = level as u64, direction = "forward")
                .entered();
        pyscf_pbc_tools::fft(&rr, lv.mesh).map_err(crate::multigrid::numint::wrap_tools)?
    };
    let weight = vol / lv.ngrids as f64;
    for x in freq.re.iter_mut().chain(freq.im.iter_mut()) {
        *x *= weight;
    }
    insert_gspace_window(rho_g, mesh, &freq, lv.mesh);
    Ok(())
}

/// This level's window of a full-mesh G-space field, back in real space.
fn level_v_r(
    vg_full: &CTensor,
    mesh: [usize; 3],
    lv: &PairLevelTable,
    level: usize,
) -> Result<CTensor, PbcDftError> {
    let sub = extract_gspace_window(vg_full, mesh, lv.mesh);
    let _fft_span =
        tracing::info_span!("pbc_mg_fft", level = level as u64, direction = "reverse").entered();
    pyscf_pbc_tools::ifft(&sub, lv.mesh).map_err(crate::multigrid::numint::wrap_tools)
}

/// Every non-empty level, paired with its index.
fn each_level(tables: &[Option<PairLevelTable>]) -> impl Iterator<Item = (usize, &PairLevelTable)> {
    tables
        .iter()
        .enumerate()
        .filter_map(|(i, lv)| lv.as_ref().map(|v| (i, v)))
}

/// Combine every level's real-space pair-fused `rho` into `rho(G)` on
/// `cell.mesh` — the v2 analogue of `numint::rho_g_from_levels`.
fn rho_g_from_pair_levels(
    cell: &Cell,
    decon: &Decontracted,
    tables: &[Option<PairLevelTable>],
    dm_p: &[f64],
) -> Result<CTensor, PbcDftError> {
    let mesh = cell.mesh;
    let ngrids_full = mesh[0] * mesh[1] * mesh[2];
    let mut rho_g = CTensor::zeros(ngrids_full);
    let vol = cell.vol();
    for (level, lv) in each_level(tables) {
        let _level_span = pair_level_span("forward", level, 1, lv);
        let rho_r = pairlevel_rho(lv, decon, dm_p)?;
        insert_level_rho_g(rho_r, lv, level, vol, mesh, &mut rho_g)?;
    }
    Ok(rho_g)
}

fn rho_g_from_pair_levels2(
    cell: &Cell,
    decon: &Decontracted,
    tables: &[Option<PairLevelTable>],
    dm_p: [&[f64]; 2],
) -> Result<[CTensor; 2], PbcDftError> {
    let mesh = cell.mesh;
    let ngrids: usize = mesh.iter().product();
    let mut rho_g = [CTensor::zeros(ngrids), CTensor::zeros(ngrids)];
    let vol = cell.vol();
    for (level, lv) in each_level(tables) {
        let _level_span = pair_level_span("forward", level, 2, lv);
        let mut rho_r = pairlevel_rho2(lv, decon, dm_p)?;
        for spin in 0..2 {
            let r = core::mem::take(&mut rho_r[spin]);
            insert_level_rho_g(r, lv, level, vol, mesh, &mut rho_g[spin])?;
        }
    }
    Ok(rho_g)
}

/// Contract a G-space weight field on the full mesh into a decontracted
/// potential matrix, level by level — the v2 analogue of
/// `numint::pass2_from_full_vg`.
fn pass2_from_full_vg_pair(
    cell: &Cell,
    decon: &Decontracted,
    tables: &[Option<PairLevelTable>],
    vg_full: &CTensor,
) -> Result<Vec<f64>, PbcDftError> {
    let mesh = cell.mesh;
    let mut v_p = vec![0.0f64; decon.nao_p * decon.nao_p];
    for (level, lv) in each_level(tables) {
        let _level_span = pair_level_span("reverse", level, 1, lv);
        let v_r = level_v_r(vg_full, mesh, lv, level)?;
        pairlevel_pass2(lv, decon, &v_r.re, &mut v_p)?;
    }
    Ok(v_p)
}

fn pass2_from_full_vg_pair2(
    cell: &Cell,
    decon: &Decontracted,
    tables: &[Option<PairLevelTable>],
    vg_full: [&CTensor; 2],
) -> Result<[Vec<f64>; 2], PbcDftError> {
    let mesh = cell.mesh;
    let mut v_p = [
        vec![0.0f64; decon.nao_p * decon.nao_p],
        vec![0.0f64; decon.nao_p * decon.nao_p],
    ];
    for (level, lv) in each_level(tables) {
        let _level_span = pair_level_span("reverse", level, 2, lv);
        let vr = [
            level_v_r(vg_full[0], mesh, lv, level)?,
            level_v_r(vg_full[1], mesh, lv, level)?,
        ];
        let [va, vb] = &mut v_p;
        pairlevel_pass2_2(lv, decon, [&vr[0].re, &vr[1].re], [va, vb])?;
    }
    Ok(v_p)
}
