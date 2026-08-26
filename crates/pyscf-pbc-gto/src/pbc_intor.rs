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
    let (expkl_re, expkl_im) =
        pyscf_kernels::pbc::bloch_phase(&client, &kflat, &lflat).map_err(|e| {
            PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "K-07 bloch_phase failed ({nkpts} kpts x {nimgs} images): {e}"
            )))
        })?;

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
    let opts = ExecutionOptions::default();

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

    for (m, l) in ls.iter().enumerate() {
        // Nothing survives screening for this image -> skip the basis build too.
        if let Some(nl) = nl
            && nl.per_image[m].is_empty()
        {
            continue;
        }
        let (basis, _, _) = cross_basis(cell1, cell2, l)?;

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
                let outcome = SessionRequest::new(
                    ctx.operator,
                    ctx.representation,
                    &basis,
                    shells,
                    opts.clone(),
                )
                .query_workspace()
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

                let block = &outcome.tensor.owned_values;
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
            }
        }
    }
    Ok(())
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
            },
        )
    }
}
