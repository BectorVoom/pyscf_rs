//! `pbc_intor` / `intor_cross` — the periodic 1-electron lattice-sum driver.
//!
//! **This is the core of Phase 10 (D-PBC-07).**
//!
//! Ports `pyscf/pbc/gto/cell.py:184-288` (`intor_cross`), `:289-372`
//! (`_intor_cross_screened`), `:2018-2037` (`Cell.pbc_intor`) and the C driver
//! semantics of `PBCnr2c_drv` / `PBCnr2c_fill_ks1`
//! (`pyscf/lib/pbc/fill_ints.c:1331-1454`).
//!
//! # What it computes
//!
//! ```text
//! out[k][c, i, j] = Σ_L exp(i·k·L) · <φ_i(r − R_i) | O_c | φ_j(r − R_j − L)>
//! ```
//!
//! Three conventions are pinned by upstream and must not drift:
//!
//! 1. **The KET is the shifted centre.** `fill_ints.c:1371` calls
//!    `shift_bas(..., jptrxyz, jL)` — only the `j` shell's atom moves, by `+L`.
//! 2. **The phase is `exp(+i k·L)`**, `cell.py:224`.
//! 3. **`Ls` comes from `max(cell1.rcut, cell2.rcut)`**, `cell.py:223`, NOT from
//!    the per-shell radii — those only ever screen, never extend, the sum.
//!
//! # How it computes it (D-PBC-07, the image-expansion route)
//!
//! cintx is a molecular integral library: it has no periodic operator and no
//! `_env`-mutating driver. Instead of shifting `PTR_COORD` in place per image,
//! this port builds, for each lattice image `L`, a small cross `BasisSet`
//! holding the bra shells followed by the ket shells translated by `L`
//! ([`pyscf_gto::build_image_expanded_cross_basis`]), so each lattice term is
//! an ordinary molecular shell-pair evaluation. `tests/cintx_cross_basis_smoke.rs`
//! is the R-02 probe that proved cintx accepts such a cross-basis pair.
//!
//! ## Deviation from PBC-MASTER-PLAN plan 10-03 — measured, deliberate
//!
//! The plan mandates the opposite arrangement: ONE basis holding cell-0 plus
//! **all** `nimgs` image blocks, indexed `[ish, nbas + l_idx*nbas + jsh]`, with
//! a per-`L` fallback only above a 20 000-shell memory guard. That is slower,
//! not faster, because a cintx `SessionRequest` costs **O(total shells in the
//! basis)**, not O(1). Measured on diamond / `gth-szv` (`nbas = 4`,
//! `nimgs = 767`, so 3 072 shells one-shot vs 8 shells per image):
//!
//! | basis | shells | per shell-pair evaluation |
//! |---|---|---|
//! | one image  |     8 |  ~20 µs |
//! | 10 images  |    44 |   32 µs |
//! | 100 images |   404 |   65 µs |
//! | 767 images | 3 072 |  400 µs |
//!
//! Per-image bases are therefore ~20x faster here AND hold O(nbas) shells at a
//! time instead of O(nimgs·nbas), so the memory guard the plan asked for is not
//! needed: [`PBC_INTOR_SHELL_WARN_LIMIT`] only warns.
//!
//! ## Summation order
//!
//! The images loop is the OUTER loop and each output element accumulates
//! sequentially in `Ls` order — the same order upstream's two `dgemm_` calls
//! reduce over (`fill_ints.c:1382-1385`, contracting the `nimgs` axis). The
//! order is fixed by the image list rather than by a thread schedule, so the
//! result is reproducible run-to-run and independent of `RAYON_NUM_THREADS`
//! (the FOUND-06 / D-PBC-17 property), and it is what makes an elementwise
//! comparison against upstream meaningful at 1e-12.
//!
//! # Screening (D-PBC-08)
//!
//! Upstream has TWO entry points: the plain `intor_cross` walks every
//! `(ish, jsh, L)`, while `_intor_cross_screened` consults a
//! [`crate::neighborlist::NeighborList`]. `Cell.pbc_intor` picks between them on
//! `cell.use_loose_rcut` (`cell.py:2035-2039`), and so does [`pbc_intor`] here.
//! [`PbcIntorOpts::screen`] exposes the choice directly.
//!
//! # Output layout
//!
//! `F-ORDER`, per component: element `(c, i, j)` lives at `c*ni*nj + i + j*ni`.
//! This matches [`pyscf_gto::IntorOutput`] and the rest of the workspace;
//! upstream's numpy arrays are C-order, so a caller comparing element-by-element
//! against `cell.pbc_intor(...)` must transpose (or compare a Hermitian matrix's
//! conjugate).

use crate::cell::Cell;
use crate::neighborlist::{NeighborList, build_neighbor_list};
use cintx_core::{BasisSet as CintxBasisSet, OperatorId, Representation};
use cintx_ops::resolver::Resolver;
use cintx_rs::SessionRequest;
use cintx_runtime::ExecutionOptions;
use pyscf_algebra::{AlgebraClient, CTensor, select_backend};
use pyscf_core::{CoreError, PyscfRsError};

/// Advisory ceiling on `nimgs · (nbas_bra + nbas_ket)` — the total number of
/// shell evaluations one `pbc_intor` call will request per component before it
/// starts to look like a mis-specified system rather than a big one.
///
/// Exceeding it only emits a `tracing::warn!`: the driver holds ONE image's
/// basis at a time (see the module docs), so there is no memory cliff to guard,
/// but a lattice sum that wide is worth telling the user about.
pub const PBC_INTOR_SHELL_WARN_LIMIT: usize = 20_000;

/// `abs(kpt).sum() < KPT_GAMMA_TOL` — upstream's gamma-point test
/// (`cell.py:277`), which is an L1 test, not an L2 one.
pub const KPT_GAMMA_TOL: f64 = 1e-9;

/// Above this the "imaginary part of a gamma-point matrix" warning fires.
/// Upstream drops the imaginary part unconditionally; this port drops it too but
/// says so first, because a large residue means the lattice sum was wrong.
pub const GAMMA_IMAG_WARN_TOL: f64 = 1e-9;

/// Options for [`intor_cross`] / [`pbc_intor`], mirroring upstream's kwargs.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PbcIntorOpts {
    /// Number of tensor components. `None` (the default) takes the operator's
    /// natural component count from the layout table (upstream
    /// `moleintor._get_intor_and_comp`).
    pub comp: Option<usize>,
    /// `0` (the default, and upstream's) = full matrix, the `s1` fill. `1` =
    /// compute the `i >= j` half and mirror it with `conj` — upstream's `s2`
    /// fill plus `lib.hermi_triu`.
    pub hermi: i32,
    /// Consult a [`NeighborList`] and skip pairs whose shell radii cannot reach.
    /// `false` (the default) reproduces upstream's plain `intor_cross` exactly;
    /// `Cell::pbc_intor` sets it from `cell.use_loose_rcut`, as upstream does.
    pub screen: bool,
    /// Range-separation parameter ω (libcint `env[PTR_RANGE_OMEGA]`), on the
    /// SAME sign convention as
    /// [`JkOpts::omega`](../../pyscf_pbc_df/traits/struct.JkOpts.html) and
    /// `rsdf_builder::omega`, so no second convention enters the workspace:
    ///
    /// * `Some(ω)`, `ω > 0` — long range, `erf(ω r)/r`
    /// * `Some(ω)`, `ω < 0` — short range, `erfc(|ω| r)/r`
    /// * `None` or `Some(0.0)` — full Coulomb (the default)
    ///
    /// This is upstream's `with cell.with_range_coulomb(omega):` around the
    /// `pbc_intor` call, not a distinct integral symbol — libcint has no
    /// `int2c2e_sr_*`. It reaches cintx as `ExecutionOptions::range_omega` and
    /// is therefore part of the WORKSPACE query, not just the kernel: short
    /// range doubles the Rys roots.
    ///
    /// Only the Coulomb families (`int2c2e`, `int3c2e`, `int2e`) honour it;
    /// cintx returns `UnsupportedApi` for anything else rather than silently
    /// evaluating the full-range operator.
    ///
    /// **The lattice-image list is NOT re-estimated from ω.** `Ls` still comes
    /// from the full-range `cell.rcut`, which is conservative under both
    /// branches (short range decays faster, long range has the same 1/r tail),
    /// so the sum is correct and merely longer than it needs to be. Tightening
    /// it is `rsdf_builder::omega::estimate_rs_2c2e_rcut`'s job, at the caller.
    pub omega: Option<f64>,
}

/// The k-resolved result of a periodic 1-electron integral.
#[derive(Debug, Clone, PartialEq)]
pub struct PbcIntorOutput {
    /// One planar-complex buffer per k-point, each `comp * ni * nj` long,
    /// F-order per component (see the module docs).
    pub kmats: Vec<CTensor>,
    /// Bra AO count.
    pub ni: usize,
    /// Ket AO count.
    pub nj: usize,
    /// Component count (1 for `int1e_ovlp`, 3 for `int1e_ipovlp`, …).
    pub comp: usize,
    /// `true` for every k-point that satisfied upstream's gamma test and whose
    /// imaginary plane was therefore dropped.
    pub gamma: Vec<bool>,
}

impl PbcIntorOutput {
    /// The matrix at k-point `k`.
    pub fn at(&self, k: usize) -> &CTensor {
        &self.kmats[k]
    }

    /// Number of k-points.
    pub fn nkpts(&self) -> usize {
        self.kmats.len()
    }

    /// Element `(i, j)` of component `c` at k-point `k`, as `(re, im)`.
    pub fn element(&self, k: usize, c: usize, i: usize, j: usize) -> (f64, f64) {
        let p = c * self.ni * self.nj + i + j * self.ni;
        (self.kmats[k].re[p], self.kmats[k].im[p])
    }

    /// Largest `|Im|` over every k-point — zero for a correctly assembled
    /// gamma-only calculation.
    pub fn max_abs_imag(&self) -> f64 {
        self.kmats
            .iter()
            .flat_map(|m| m.im.iter())
            .fold(0.0_f64, |a, v| a.max(v.abs()))
    }
}

/// The device client for the Bloch phase table and the contraction GEMMs.
/// Mirrors `crate::gv`'s `select_backend()` call site (ALG-06: a `pyscf-pbc-*`
/// crate names `pyscf_algebra`, never `cubecl-*`).
fn resolve_client(who: &str) -> Result<AlgebraClient, PyscfRsError> {
    Ok(select_backend()
        .map_err(|e| {
            PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "{who}: backend selection failed: {e}"
            )))
        })?
        .client)
}

/// Upstream's gamma test, `abs(kpt).sum() < 1e-9` (`cell.py:277`).
pub fn is_gamma(kpt: &[f64; 3]) -> bool {
    kpt[0].abs() + kpt[1].abs() + kpt[2].abs() < KPT_GAMMA_TOL
}

/// Periodic 1-electron integrals over a single cell — `cell.pbc_intor(intor,
/// comp, hermi, kpts)` (`cell.py:2018-2037`).
///
/// `kpts` is a list of ABSOLUTE k-points in 1/Bohr (see
/// [`Cell::make_kpts`](crate::kpts_mesh::make_kpts)); an empty slice is treated
/// as the single gamma point, matching upstream's `kpts=None` default.
///
/// Screening follows `cell.use_loose_rcut` unless the caller overrides
/// [`PbcIntorOpts::screen`].
///
/// # Errors
/// See [`intor_cross`].
pub fn pbc_intor(
    cell: &Cell,
    intor: &str,
    kpts: &[[f64; 3]],
    opts: PbcIntorOpts,
) -> Result<PbcIntorOutput, PyscfRsError> {
    intor_cross(intor, cell, cell, kpts, opts)
}

/// Periodic 1-electron integrals between two cells — `intor_cross(intor, cell1,
/// cell2, ...)` (`cell.py:184-288`).
///
/// Bra functions come from `cell1`, ket functions from `cell2`; the KET is the
/// half that gets translated by `L`.
///
/// # Errors
/// * [`CoreError::InvalidMolecule`] — an unbuilt cell, an intor name outside the
///   layout table, a cintx workspace/evaluate failure, or a shape overflow.
/// * [`PyscfRsError::NotYetImplemented`] — an intor family Phase 10 does not
///   cover (see [`SUPPORTED_INTORS`]), or a spinor representation.
pub fn intor_cross(
    intor: &str,
    cell1: &Cell,
    cell2: &Cell,
    kpts: &[[f64; 3]],
    opts: PbcIntorOpts,
) -> Result<PbcIntorOutput, PyscfRsError> {
    let ls = lattice_images(cell1, cell2)?;
    intor_cross_with_images(intor, cell1, cell2, kpts, opts, &ls, None)
}

/// The lattice-image list `intor_cross` sums over —
/// `Ls = cell1.get_lattice_Ls(rcut = max(cell1.rcut, cell2.rcut))`
/// (`cell.py:222-223`).
///
/// # Errors
/// As [`crate::lattice::get_lattice_ls`].
pub fn lattice_images(cell1: &Cell, cell2: &Cell) -> Result<Vec<[f64; 3]>, PyscfRsError> {
    let rcut = cell1.try_rcut()?.max(cell2.try_rcut()?);
    crate::lattice::get_lattice_ls(cell1, Some(rcut), None, true)
}

/// The 1-electron families Phase 10 ships. Anything else is a Phase-13 (AFTDF /
/// moment-weighted) or later concern and is refused loudly rather than
/// silently mis-evaluated.
pub const SUPPORTED_INTORS: &[&str] = &[
    "int1e_ovlp",
    "int1e_kin",
    "int1e_nuc",
    "int1e_r",
    "int1e_r2_origi",
    "int1e_r4_origi",
    "int1e_ipovlp",
    "int1e_ipkin",
    "int1e_ipnuc",
    // Plan 18-03 — the `_int_vnl` derivative halves of `pp_int.py:454`
    // (`vppnl_nuc_grad` via `ppnl_half_ip2`): `<p_0| r^n |d/dR j>` with the
    // origin on the projector centre. Same `ComponentLeadingFOrder { 3 }`
    // layout as the `ip` families above, resolved through
    // `pyscf_gto::layout_table`; same `gth-pp` (`unstable-source-api`) gate
    // and `oracle_covered = false` posture as the scalar `origi` half.
    // `int1e_ipovlp` doubles as the rank-0 derivative half (negated to turn
    // its bra derivative into the ket derivative `_int_vnl` needs —
    // `pp_int.py:457-460`), so it needs no new symbol.
    "int1e_r2_origi_ip2",
    "int1e_r4_origi_ip2",
    // Plan 14-01: `incore.fill_2c2e` is `auxcell.pbc_intor('int2c2e', ...)` —
    // the auxiliary metric of every Gaussian density fitting builder. It is an
    // arity-2 two-electron operator, so it goes through the same lattice sum as
    // the 1-electron families above.
    "int2c2e",
];

/// [`intor_cross`] against a caller-supplied image list and (optionally) a
/// pre-built neighbor list.
///
/// This is the entry point the GTH non-local pseudopotential uses (plan 10-06):
/// `_int_vnl` evaluates several operators over the SAME `(cell, fakecell, Ls)`
/// triple, and rebuilding `Ls` — an `O(nimgs · natm)` filter — per operator is
/// pure waste.
///
/// # Errors
/// As [`intor_cross`].
pub fn intor_cross_with_images(
    intor: &str,
    cell1: &Cell,
    cell2: &Cell,
    kpts: &[[f64; 3]],
    opts: PbcIntorOpts,
    ls: &[[f64; 3]],
    neighbor_list: Option<&NeighborList>,
) -> Result<PbcIntorOutput, PyscfRsError> {
    intor_cross_with_image_weights(intor, cell1, cell2, kpts, opts, ls, neighbor_list, None)
}

/// Evaluate lattice integrals with an optional real weight for each image.
///
/// Weights multiply the Bloch phase before accumulating shell blocks. Arbitrary
/// weights need not preserve Hermiticity, so weighted calls require `hermi = 0`.
/// Passing `None` preserves the unweighted accumulation exactly.
pub fn intor_cross_with_image_weights(
    intor: &str,
    cell1: &Cell,
    cell2: &Cell,
    kpts: &[[f64; 3]],
    opts: PbcIntorOpts,
    ls: &[[f64; 3]],
    neighbor_list: Option<&NeighborList>,
    image_weights: Option<&[f64]>,
) -> Result<PbcIntorOutput, PyscfRsError> {
    if let Some(weights) = image_weights {
        if weights.len() != ls.len() || weights.iter().any(|w| !w.is_finite()) {
            return Err(PyscfRsError::Core(CoreError::InvalidMolecule(
                "pbc_intor: image weights must be finite and match the image count".into(),
            )));
        }
        if opts.hermi != 0 {
            return Err(PyscfRsError::Core(CoreError::InvalidMolecule(
                "pbc_intor: image-weighted integrals require hermi = 0".into(),
            )));
        }
    }
    if !cell1.mol._built || !cell2.mol._built {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(
            "pbc_intor: both cells must be built".into(),
        )));
    }

    // ── name + component count ─────────────────────────────────────────
    let full_name = pyscf_gto::add_suffix(intor, cell1.mol.cart);
    let core_name = full_name
        .trim_end_matches("_sph")
        .trim_end_matches("_cart")
        .to_string();
    if !SUPPORTED_INTORS.contains(&core_name.as_str()) {
        return Err(PyscfRsError::NotYetImplemented {
            phase: 13,
            what: "this periodic 1-electron family is outside Phase 10 \
                   (see pyscf_pbc_gto::pbc_intor::SUPPORTED_INTORS)",
        });
    }
    let layout = pyscf_gto::layout_table::lookup(&full_name).ok_or_else(|| {
        PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "pbc_intor: unknown intor '{full_name}' (not in INTOR_LAYOUTS)"
        )))
    })?;
    let natural_comp = match layout {
        pyscf_gto::layout_table::IntorLayout::ScalarFOrder => 1usize,
        pyscf_gto::layout_table::IntorLayout::ComponentLeadingFOrder { components } => {
            components as usize
        }
    };
    let comp = opts.comp.unwrap_or(natural_comp);

    let representation = if full_name.ends_with("_cart") {
        Representation::Cart
    } else if full_name.ends_with("_sph") {
        Representation::Spheric
    } else {
        return Err(PyscfRsError::NotYetImplemented {
            phase: 19,
            what: "spinor representation for periodic 1-electron integrals",
        });
    };

    let descriptor = Resolver::descriptor_by_symbol(&full_name).map_err(|e| {
        PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "cintx-ops resolver does not know symbol '{full_name}': {e}"
        )))
    })?;
    if descriptor.entry.arity != 2 {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "pbc_intor supports arity-2 integrals only; '{full_name}' is arity {}",
            descriptor.entry.arity
        ))));
    }
    let operator = descriptor.id;

    // ── k-points and their Bloch phases (K-07) ─────────────────────────
    let owned_gamma = [[0.0_f64; 3]];
    let kpts: &[[f64; 3]] = if kpts.is_empty() { &owned_gamma } else { kpts };
    let nkpts = kpts.len();
    let nimgs = ls.len();

    let kflat: Vec<f64> = kpts.iter().flatten().copied().collect();
    let lflat: Vec<f64> = ls.iter().flatten().copied().collect();
    let client = resolve_client("pbc_intor")?;
    let (mut expkl_re, mut expkl_im) = pyscf_kernels::pbc::bloch_phase(&client, &kflat, &lflat)
        .map_err(|e| {
            PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "K-07 bloch_phase failed ({nkpts} kpts x {nimgs} images): {e}"
            )))
        })?;
    if let Some(weights) = image_weights {
        for k in 0..nkpts {
            for (m, &weight) in weights.iter().enumerate() {
                expkl_re[k * nimgs + m] *= weight;
                expkl_im[k * nimgs + m] *= weight;
            }
        }
    }

    // ── screening ──────────────────────────────────────────────────────
    let owned_nl;
    let nl: Option<&NeighborList> = if let Some(nl) = neighbor_list {
        Some(nl)
    } else if opts.screen {
        owned_nl = build_neighbor_list(cell1, Some(cell2), ls, None, None, 0, None)?;
        Some(&owned_nl)
    } else {
        None
    };
    if let Some(nl) = nl
        && (nl.nish != cell1.mol.nbas || nl.njsh != cell2.mol.nbas || nl.nimgs != nimgs)
    {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "pbc_intor: neighbor list shape ({}, {}, {}) does not match \
             (nbas1 {}, nbas2 {}, nimgs {nimgs})",
            nl.nish, nl.njsh, nl.nimgs, cell1.mol.nbas, cell2.mol.nbas,
        ))));
    }

    // ── output allocation ──────────────────────────────────────────────
    let ni = cell1.mol.nao_nr;
    let nj = cell2.mol.nao_nr;
    let per_k = comp
        .checked_mul(ni)
        .and_then(|v| v.checked_mul(nj))
        .ok_or_else(|| {
            PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "pbc_intor '{full_name}': shape overflow comp={comp} ni={ni} nj={nj}"
            )))
        })?;
    let mut kmats: Vec<CTensor> = (0..nkpts).map(|_| CTensor::zeros(per_k)).collect();

    if ni == 0 || nj == 0 || nimgs == 0 {
        return Ok(PbcIntorOutput {
            kmats,
            ni,
            nj,
            comp,
            gamma: kpts.iter().map(is_gamma).collect(),
        });
    }

    // ── the lattice sum ────────────────────────────────────────────────
    let total_shells = nimgs * (cell1.mol.nbas + cell2.mol.nbas);
    if total_shells > PBC_INTOR_SHELL_WARN_LIMIT {
        tracing::warn!(
            "pbc_intor('{full_name}'): the lattice sum spans {nimgs} images x \
             ({} + {}) shells = {total_shells} shell instances (advisory limit \
             {PBC_INTOR_SHELL_WARN_LIMIT}); check cell.rcut / cell.precision",
            cell1.mol.nbas,
            cell2.mol.nbas,
        );
    }

    lattice_sum(
        &LatticeSumCtx {
            operator,
            representation,
            full_name: &full_name,
            comp,
            ni,
            nj,
            nimgs,
            hermi: opts.hermi,
            omega: opts.omega,
        },
        cell1,
        cell2,
        ls,
        nl,
        &expkl_re,
        &expkl_im,
        &mut kmats,
    )?;

    // ── hermi_triu + gamma realification (cell.py:270-280) ─────────────
    let gamma: Vec<bool> = kpts.iter().map(is_gamma).collect();
    for (k, mat) in kmats.iter_mut().enumerate() {
        if opts.hermi != 0 {
            for c in 0..comp {
                hermi_triu(mat, c * ni * nj, ni, nj)?;
            }
        }
        if gamma[k] {
            let max_im = mat.im.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
            if max_im > GAMMA_IMAG_WARN_TOL {
                tracing::warn!(
                    "pbc_intor('{full_name}'): gamma-point matrix has max|Im| = {max_im:e} \
                     (> {GAMMA_IMAG_WARN_TOL:e}); upstream drops it regardless, but this \
                     usually means the lattice sum is incomplete"
                );
            }
            mat.im.iter_mut().for_each(|v| *v = 0.0);
        }
    }

    Ok(PbcIntorOutput {
        kmats,
        ni,
        nj,
        comp,
        gamma,
    })
}

/// Everything the lattice-sum body needs beyond the cells themselves, so its
/// signature stays short enough to read.
struct LatticeSumCtx<'a> {
    operator: OperatorId,
    representation: Representation,
    full_name: &'a str,
    comp: usize,
    ni: usize,
    nj: usize,
    nimgs: usize,
    hermi: i32,
    /// See [`PbcIntorOpts::omega`].
    omega: Option<f64>,
}

/// `Σ_L exp(i·k·L) · <i(0) | O | j(L)>` — the whole lattice sum.
///
/// Images are the OUTER loop (see the module docs on summation order): each
/// iteration builds the small `(bra | ket + L)` cross basis, walks the shell
/// pairs that survive screening, and folds every block into all `nkpts` output
/// matrices before the basis is dropped.
#[allow(clippy::too_many_arguments)]
fn lattice_sum(
    ctx: &LatticeSumCtx<'_>,
    cell1: &Cell,
    cell2: &Cell,
    ls: &[[f64; 3]],
    nl: Option<&NeighborList>,
    expkl_re: &[f64],
    expkl_im: &[f64],
    kmats: &mut [CTensor],
) -> Result<(), PyscfRsError> {
    let comp = ctx.comp;
    let ni = ctx.ni;
    let nj = ctx.nj;
    // ω belongs to the options the WORKSPACE is queried with, not only to the
    // kernel: short range doubles the Rys roots, and cintx rejects a ω that
    // changes between query and evaluate as backend contract drift.
    let opts = ExecutionOptions {
        range_omega: ctx.omega,
        ..ExecutionOptions::default()
    };

    // AO offsets/counts are image-independent — the shells are identical, only
    // their centres move — so they are read once, off image 0's basis.
    let (probe, nbas_a, nbas_b) = cross_basis(cell1, cell2, &ls[0])?;
    let meta = probe.meta();
    let bra_off: Vec<usize> = (0..nbas_a)
        .map(|s| meta.shell_offset(s).unwrap_or(0))
        .collect();
    let bra_cnt: Vec<usize> = (0..nbas_a).map(|s| meta.ao_count(s).unwrap_or(0)).collect();
    let ket_off: Vec<usize> = (0..nbas_b)
        .map(|s| meta.shell_offset(nbas_a + s).unwrap_or(0) - ni)
        .collect();
    let ket_cnt: Vec<usize> = (0..nbas_b)
        .map(|s| meta.ao_count(nbas_a + s).unwrap_or(0))
        .collect();
    drop(probe);

    // BAND-01: the per-(image, shell pair) real blocks are k-independent, so
    // they are computed once per (cells, intor, images, screen) and cached;
    // every later k-list — the band k-points after an SCF, a second
    // `get_bands` — pays only the Bloch-phase fold below. The fold replays the
    // SAME (image, ish, jsh) order, so each output element receives the
    // identical sequence of additions: bit-identical to the uncached path.
    let npair = nbas_a * nbas_b;
    let key = image_block_key(ctx, cell1, cell2, ls, nl);
    let fold = |kmats: &mut [CTensor], m: usize, ish: usize, jsh: usize, block: &[f64]| {
        let di = bra_cnt[ish];
        let dj = ket_cnt[jsh];
        let oi = bra_off[ish];
        let oj = ket_off[jsh];
        for (k, mat) in kmats.iter_mut().enumerate() {
            let pr = expkl_re[k * ctx.nimgs + m];
            let pi = expkl_im[k * ctx.nimgs + m];
            for c in 0..comp {
                let cb = c * di * dj;
                let co = c * ni * nj;
                for jj in 0..dj {
                    for ii in 0..di {
                        let v = block[cb + ii + jj * di];
                        let o = co + (oi + ii) + (oj + jj) * ni;
                        mat.re[o] += pr * v;
                        mat.im[o] += pi * v;
                    }
                }
            }
        }
    };

    // Streaming: when the blocks could not be cached anyway (cache off, or an
    // upper bound of their size over the cap), evaluate the images in waves
    // and fold each wave at once, instead of holding EVERY image's blocks
    // (~8 GB for a 54-atom DZVP overlap). Each output element still receives
    // its additions in ascending image order, so the result is bit-identical;
    // the peak drops to one wave's blocks.
    let cached = key.and_then(image_block_cache_get);
    let bound = ls.len().saturating_mul(ni * nj * comp);
    if cached.is_none() && (key.is_none() || bound > image_block_cache_max_f64()) {
        // Waves of `threads * MIN_PAIRS / npair` images (`PYSCF_PBC_INTOR_WAVE_IMAGES`
        // overrides it, for tests), with ONE set of cintx contexts for the
        // whole sum. Measured on a 54-atom DZVP overlap (CPU runtime): fresh
        // contexts per wave cost more memory than they save (21 GB peak vs
        // 15 GB with reused contexts and MALLOC_ARENA_MAX=2) — each context's
        // executor allocates on creation and its metadata cache is freed
        // only on drop, into fragmented malloc arenas.
        let wave = std::env::var("PYSCF_PBC_INTOR_WAVE_IMAGES")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(image_block_threads() * MIN_PAIRS_PER_THREAD / npair.max(1))
            .max(1);
        let ectxs = image_contexts(ctx, ls.len(), npair);
        let mut m0 = 0;
        while m0 < ls.len() {
            let m1 = (m0 + wave).min(ls.len());
            for (slot, block) in eval_image_range(
                ctx, cell1, cell2, ls, m0..m1, nl, nbas_a, nbas_b, &bra_cnt, &ket_cnt, &opts, &ectxs,
            )? {
                if block.is_empty() {
                    continue;
                }
                let (m, pair) = (slot / npair, slot % npair);
                fold(kmats, m, pair / nbas_b, pair % nbas_b, &block);
            }
            m0 = m1;
        }
        return Ok(());
    }
    let blocks = match cached {
        Some(b) => b,
        None => {
            let b = std::sync::Arc::new(eval_image_blocks(
                ctx, cell1, cell2, ls, nl, nbas_a, nbas_b, &bra_cnt, &ket_cnt, &opts,
            )?);
            if let Some(key) = key {
                image_block_cache_put(key, &b);
            }
            b
        }
    };

    for m in 0..ls.len() {
        for ish in 0..nbas_a {
            for jsh in 0..nbas_b {
                let block = &blocks[m * npair + ish * nbas_b + jsh];
                if block.is_empty() {
                    continue;
                }
                fold(kmats, m, ish, jsh, block);
            }
        }
    }
    Ok(())
}

/// Every surviving `(image, ish, jsh)` block of the lattice sum, flat at
/// `[m * nbas_a * nbas_b + ish * nbas_b + jsh]`; screened, skipped or empty
/// pairs stay empty vectors. Evaluated in the pre-BAND-01 order.
///
/// Images are split over threads (`PYSCF_NUM_THREADS`, else every core) in
/// contiguous chunks, each worker with its own cintx `EvaluationContext`, as
/// `aux_e2` does. Each block is a pure function of its `(image, ish, jsh)`
/// and lands in its own slot, and the Bloch fold that consumes them is
/// unchanged, so the result is bit-identical to the serial loop. A 54-atom
/// DZVP cell (477 images, 178 shells) spent tens of minutes here on one core.
#[allow(clippy::too_many_arguments)]
fn eval_image_blocks(
    ctx: &LatticeSumCtx<'_>,
    cell1: &Cell,
    cell2: &Cell,
    ls: &[[f64; 3]],
    nl: Option<&NeighborList>,
    nbas_a: usize,
    nbas_b: usize,
    bra_cnt: &[usize],
    ket_cnt: &[usize],
    opts: &ExecutionOptions,
) -> Result<Vec<Vec<f64>>, PyscfRsError> {
    let mut blocks: Vec<Vec<f64>> = vec![Vec::new(); ls.len() * nbas_a * nbas_b];
    let ectxs = image_contexts(ctx, ls.len(), nbas_a * nbas_b);
    for (slot, block) in eval_image_range(
        ctx, cell1, cell2, ls, 0..ls.len(), nl, nbas_a, nbas_b, bra_cnt, ket_cnt, opts, &ectxs,
    )? {
        blocks[slot] = block;
    }
    Ok(blocks)
}

/// The surviving blocks of images `range` as `(slot, block)` pairs in
/// ascending slot order — i.e. in `(image, ish, jsh)` order, the order the
/// Bloch fold consumes them. Threaded over contiguous image chunks.
#[allow(clippy::too_many_arguments)]
fn eval_image_range(
    ctx: &LatticeSumCtx<'_>,
    cell1: &Cell,
    cell2: &Cell,
    ls: &[[f64; 3]],
    range: std::ops::Range<usize>,
    nl: Option<&NeighborList>,
    nbas_a: usize,
    nbas_b: usize,
    bra_cnt: &[usize],
    ket_cnt: &[usize],
    opts: &ExecutionOptions,
    ectxs: &[cintx_rs::EvaluationContext],
) -> Result<Vec<(usize, Vec<f64>)>, PyscfRsError> {
    let nimg = range.len();
    let nthreads = ectxs.len().min(nimg.max(1));
    if nthreads <= 1 {
        // A single context: the serial scalar route uses cintx's default one
        // unless the pair batch needs an explicit context.
        let batch_ctx = ectxs.first().filter(|_| pair_batch_serves(ctx));
        let mut out = Vec::new();
        for m in range {
            out.extend(image_blocks_one(
                ctx, cell1, cell2, m, &ls[m], nl, nbas_a, nbas_b, bra_cnt, ket_cnt, opts, batch_ctx,
            )?);
        }
        return Ok(out);
    }
    let pairs = nimg * nbas_a * nbas_b;
    tracing::debug!(nimgs = nimg, pairs, nthreads, "pbc_intor: threaded image blocks");
    let chunk = nimg.div_ceil(nthreads);
    let parts: Vec<Result<Vec<(usize, Vec<f64>)>, PyscfRsError>> = std::thread::scope(|scope| {
        let handles: Vec<_> = range
            .clone()
            .step_by(chunk)
            .zip(ectxs)
            .map(|(m0, ectx)| {
                let m1 = (m0 + chunk).min(range.end);
                scope.spawn(move || {
                    let mut out = Vec::new();
                    for m in m0..m1 {
                        out.extend(image_blocks_one(
                            ctx, cell1, cell2, m, &ls[m], nl, nbas_a, nbas_b, bra_cnt, ket_cnt,
                            opts, Some(ectx),
                        )?);
                    }
                    Ok(out)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("pbc_intor image-block worker panicked"))
            .collect()
    });
    let mut out = Vec::new();
    for part in parts {
        out.extend(part?);
    }
    Ok(out)
}

/// `PYSCF_PBC_INTOR_PAIR_BATCH=1`: evaluate each image's surviving shell pairs
/// of `int1e_ovlp_sph` / `int1e_kin_sph` as ONE cintx `PairBatchRequest`
/// instead of one `SessionRequest` (one kernel launch) per pair. Off by
/// default: the batched kernel is a different code path, so its blocks are
/// gated against the scalar ones at rounding level
/// (`tests/pbc_intor_pair_batch.rs`), not bitwise.
fn pair_batch_serves(ctx: &LatticeSumCtx<'_>) -> bool {
    std::env::var("PYSCF_PBC_INTOR_PAIR_BATCH").is_ok_and(|v| v.trim() == "1")
        && ctx.representation == Representation::Spheric
        && ctx.omega.is_none()
        && ctx.comp == 1
        && matches!(ctx.full_name, "int1e_ovlp_sph" | "int1e_kin_sph")
}

/// [`image_blocks_one`] through one `PairBatchRequest` for the whole image:
/// the same surviving pairs in the same order, each block in the scalar
/// path's layout (the batch output is the per-tuple blocks concatenated).
#[allow(clippy::too_many_arguments)]
fn image_blocks_batched(
    ctx: &LatticeSumCtx<'_>,
    basis: &CintxBasisSet,
    m: usize,
    nl: Option<&NeighborList>,
    nbas_a: usize,
    nbas_b: usize,
    bra_cnt: &[usize],
    ket_cnt: &[usize],
    opts: &ExecutionOptions,
    ectx: &cintx_rs::EvaluationContext,
) -> Result<Vec<(usize, Vec<f64>)>, PyscfRsError> {
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for ish in 0..nbas_a {
        if bra_cnt[ish] == 0 {
            continue;
        }
        for jsh in 0..nbas_b {
            if ket_cnt[jsh] == 0 || (ctx.hermi != 0 && ish < jsh) {
                continue;
            }
            if let Some(nl) = nl
                && nl.per_image[m].binary_search(&(ish, jsh)).is_err()
            {
                continue;
            }
            pairs.push((ish, jsh));
        }
    }
    if pairs.is_empty() {
        return Ok(Vec::new());
    }
    let out = cintx_rs::PairBatchRequest::new(
        ctx.operator,
        ctx.representation,
        basis,
        pairs.iter().map(|&(i, j)| [i as u32, (nbas_a + j) as u32]),
        opts.clone(),
    )
    .evaluate_in(ectx)
    .map_err(|e| {
        PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "cintx pair batch failed for '{}' at image {m}: {e}",
            ctx.full_name
        )))
    })?;
    let mut blocks = Vec::with_capacity(pairs.len());
    for (n, &(ish, jsh)) in pairs.iter().enumerate() {
        let len = bra_cnt[ish] * ket_cnt[jsh] * ctx.comp;
        let start = out.offsets[n];
        let block = out.values.get(start..start + len).ok_or_else(|| {
            PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "cintx pair batch for '{}' returned {} values, pair {n} needs {start}..{}",
                ctx.full_name,
                out.values.len(),
                start + len
            )))
        })?;
        blocks.push((m * nbas_a * nbas_b + ish * nbas_b + jsh, block.to_vec()));
    }
    Ok(blocks)
}

/// The worker contexts of one lattice sum: one per thread, created ONCE and
/// reused by every wave. Each cintx `EvaluationContext` owns a cubecl
/// executor whose buffers it keeps; standing up fresh ones per wave grew a
/// 54-atom DZVP overlap from 8 to 13+ GB. A worker's context also costs
/// ~0.3 s, so small sums get one (or none, below).
fn image_contexts(ctx: &LatticeSumCtx<'_>, nimg: usize, npair: usize) -> Vec<cintx_rs::EvaluationContext> {
    let nthreads = image_block_threads()
        .min(nimg.max(1))
        .min((nimg * npair / MIN_PAIRS_PER_THREAD).max(1));
    if nthreads <= 1 && !pair_batch_serves(ctx) {
        return Vec::new();
    }
    (0..nthreads).map(|_| cintx_rs::EvaluationContext::new()).collect()
}

/// Fewest `(image, shell pair)` candidates worth a worker thread.
const MIN_PAIRS_PER_THREAD: usize = 200_000;

/// `PYSCF_NUM_THREADS`, else the core count.
fn image_block_threads() -> usize {
    if let Ok(v) = std::env::var("PYSCF_NUM_THREADS")
        && let Ok(n) = v.trim().parse::<usize>()
        && n > 0
    {
        return n;
    }
    std::thread::available_parallelism().map(std::num::NonZeroUsize::get).unwrap_or(1)
}

/// The surviving blocks of image `m` as `(slot, block)` pairs.
#[allow(clippy::too_many_arguments)]
fn image_blocks_one(
    ctx: &LatticeSumCtx<'_>,
    cell1: &Cell,
    cell2: &Cell,
    m: usize,
    l: &[f64; 3],
    nl: Option<&NeighborList>,
    nbas_a: usize,
    nbas_b: usize,
    bra_cnt: &[usize],
    ket_cnt: &[usize],
    opts: &ExecutionOptions,
    eval_ctx: Option<&cintx_rs::EvaluationContext>,
) -> Result<Vec<(usize, Vec<f64>)>, PyscfRsError> {
    let comp = ctx.comp;
    let mut out = Vec::new();
    // Nothing survives screening for this image -> skip the basis build too.
    if let Some(nl) = nl
        && nl.per_image[m].is_empty()
    {
        return Ok(out);
    }
    let (basis, _, _) = cross_basis(cell1, cell2, l)?;

    if let Some(ectx) = eval_ctx
        && pair_batch_serves(ctx)
    {
        return image_blocks_batched(ctx, &basis, m, nl, nbas_a, nbas_b, bra_cnt, ket_cnt, opts, ectx);
    }

    for ish in 0..nbas_a {
        let di = bra_cnt[ish];
        if di == 0 {
            continue;
        }
        for jsh in 0..nbas_b {
            let dj = ket_cnt[jsh];
            if dj == 0 {
                continue;
            }
            // hermi != 0: upstream's `s2` fill evaluates only the i >= j
            // half — `_nr2c_fill(..., ish0 = jsh)` at `fill_ints.c:1413`
            // starts the bra loop at the ket shell — and `lib.hermi_triu`
            // mirrors the rest. The test is on SHELL indices, matching
            // upstream; it is only meaningful when bra and ket are the same
            // shell list, which `hermi_triu`'s square check enforces.
            if ctx.hermi != 0 && ish < jsh {
                continue;
            }
            if let Some(nl) = nl
                && nl.per_image[m].binary_search(&(ish, jsh)).is_err()
            {
                continue;
            }

            let j_global = nbas_a + jsh;
            let shells = basis
                .shell_tuple_for_indices([ish, j_global])
                .map_err(|e| {
                    PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                        "shell_tuple_for_indices({ish}, {j_global}) failed for '{}': {e}",
                        ctx.full_name
                    )))
                })?;
            let request = SessionRequest::new(
                ctx.operator,
                ctx.representation,
                &basis,
                shells,
                opts.clone(),
            );
            let request = match eval_ctx {
                Some(c) => request.query_workspace_in(c),
                None => request.query_workspace(),
            };
            let outcome = request
            .map_err(|e| {
                PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                    "cintx workspace query failed for '{}' pair ({ish},{jsh}) \
                     at image {m}: {e}",
                    ctx.full_name
                )))
            })?
            .evaluate()
            .map_err(|e| {
                PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                    "cintx evaluate failed for '{}' pair ({ish},{jsh}) at image {m}: {e}",
                    ctx.full_name
                )))
            })?;

            let block = outcome.tensor.owned_values;
            let dmjc = di * dj * comp;
            if block.len() != dmjc {
                return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                    "cintx returned {} elements for '{}' pair ({ish},{jsh}), expected \
                     {dmjc} (di={di} dj={dj} comp={comp}, extents={:?})",
                    block.len(),
                    ctx.full_name,
                    outcome.tensor.extents,
                ))));
            }
            out.push((m * nbas_a * nbas_b + ish * nbas_b + jsh, block));
        }
    }
    Ok(out)
}

/// 128-bit fingerprint of everything the image blocks depend on, or `None`
/// when the cache is off (`PYSCF_PBC_INTOR_IMAGE_CACHE=0`).
fn image_block_key(
    ctx: &LatticeSumCtx<'_>,
    cell1: &Cell,
    cell2: &Cell,
    ls: &[[f64; 3]],
    nl: Option<&NeighborList>,
) -> Option<(u64, u64)> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    if std::env::var("PYSCF_PBC_INTOR_IMAGE_CACHE").is_ok_and(|v| v == "0") {
        return None;
    }
    let feed = |h: &mut DefaultHasher| {
        for cell in [cell1, cell2] {
            format!("{:?}", cell.mol._atom).hash(h);
            // `_basis` is a HashMap: key-sorted, so equal bases hash equal.
            let mut basis: Vec<_> = cell.mol._basis.iter().collect();
            basis.sort_by(|a, b| a.0.cmp(b.0));
            format!("{basis:?}").hash(h);
            cell.mol.cart.hash(h);
        }
        ctx.full_name.hash(h);
        ctx.comp.hash(h);
        ctx.hermi.hash(h);
        ctx.omega.map(f64::to_bits).hash(h);
        for l in ls {
            for x in l {
                x.to_bits().hash(h);
            }
        }
        nl.map(|nl| &nl.per_image).hash(h);
    };
    let mut a = DefaultHasher::new();
    feed(&mut a);
    let mut b = DefaultHasher::new();
    0x9e37_79b9_7f4a_7c15_u64.hash(&mut b);
    feed(&mut b);
    Some((a.finish(), b.finish()))
}

type ImageBlocks = std::sync::Arc<Vec<Vec<f64>>>;

/// The most `f64`s the image-block cache holds (256 MiB). A larger entry is
/// never cached; inserting past the cap empties the cache first.
const IMAGE_BLOCK_CACHE_MAX_F64: usize = 32 * 1024 * 1024;

/// The cache cap in f64s: `PYSCF_PBC_INTOR_IMAGE_CACHE_MB` MiB, else
/// [`IMAGE_BLOCK_CACHE_MAX_F64`]. A large cell's overlap and kinetic blocks
/// (GBs at nao ~900) otherwise miss and are rebuilt for every band chunk.
fn image_block_cache_max_f64() -> usize {
    static MAX: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MAX.get_or_init(|| {
        std::env::var("PYSCF_PBC_INTOR_IMAGE_CACHE_MB")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .map_or(IMAGE_BLOCK_CACHE_MAX_F64, |mb| mb * 1024 * 1024 / 8)
    })
}

type ImageBlockCache = std::sync::Mutex<Vec<((u64, u64), ImageBlocks)>>;

fn image_block_cache() -> &'static ImageBlockCache {
    static CACHE: std::sync::OnceLock<ImageBlockCache> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(Vec::new()))
}

fn image_block_cache_get(key: (u64, u64)) -> Option<ImageBlocks> {
    let cache = image_block_cache().lock().ok()?;
    cache.iter().find(|(k, _)| *k == key).map(|(_, v)| v.clone())
}

fn image_block_cache_put(key: (u64, u64), blocks: &ImageBlocks) {
    let size = |b: &ImageBlocks| b.iter().map(Vec::len).sum::<usize>();
    let new = size(blocks);
    if new > image_block_cache_max_f64() {
        return;
    }
    if let Ok(mut cache) = image_block_cache().lock() {
        let held: usize = cache.iter().map(|(_, v)| size(v)).sum();
        if held + new > image_block_cache_max_f64() {
            cache.clear();
        }
        cache.push((key, blocks.clone()));
    }
}

/// The `(bra shells | ket shells translated by `l`)` cross basis for ONE image.
fn cross_basis(
    cell1: &Cell,
    cell2: &Cell,
    l: &[f64; 3],
) -> Result<(std::sync::Arc<CintxBasisSet>, usize, usize), PyscfRsError> {
    pyscf_gto::build_image_expanded_cross_basis(
        &cell1.mol._atom,
        &cell1.mol._basis,
        cell1.mol.cart,
        &cell2.mol._atom,
        &cell2.mol._basis,
        cell2.mol.cart,
        std::slice::from_ref(l),
    )
}

/// `lib.hermi_triu(v, hermi=1)` on one F-order `(n, n)` component slice:
/// copy the lower triangle into the upper with a conjugate.
///
/// # Errors
/// [`CoreError::InvalidMolecule`] on a non-square block — Hermitian symmetry is
/// meaningless there and upstream would have raised too.
fn hermi_triu(mat: &mut CTensor, base: usize, ni: usize, nj: usize) -> Result<(), PyscfRsError> {
    if ni != nj {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "pbc_intor: hermi != 0 requires a square block, got {ni}x{nj}"
        ))));
    }
    for j in 0..nj {
        for i in 0..j {
            let lower = base + j + i * ni; // (j, i)
            let upper = base + i + j * ni; // (i, j)
            mat.re[upper] = mat.re[lower];
            mat.im[upper] = -mat.im[lower];
        }
    }
    Ok(())
}

impl Cell {
    /// `cell.pbc_intor(intor, comp, hermi, kpts)` — `cell.py:2018-2037`.
    ///
    /// Screening follows `self.use_loose_rcut`, exactly as upstream picks
    /// between `intor_cross` and `_intor_cross_screened`.
    ///
    /// # Errors
    /// As [`intor_cross`].
    pub fn pbc_intor(
        &self,
        intor: &str,
        kpts: &[[f64; 3]],
        comp: Option<usize>,
        hermi: i32,
    ) -> Result<PbcIntorOutput, PyscfRsError> {
        pbc_intor(
            self,
            intor,
            kpts,
            PbcIntorOpts {
                comp,
                hermi,
                screen: self.use_loose_rcut,
                omega: None,
            },
        )
    }
}
