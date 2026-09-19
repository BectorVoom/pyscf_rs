//! KRKS k-point analytic nuclear gradient — `pyscf/pbc/grad/krks.py` (141 l).
//!
//! The k-point restricted Kohn–Sham nuclear gradient. Upstream declares it as
//! `class Gradients(rhf_grad.Gradients)` (`krks.py:119`): it extends 18-05's
//! **base**, so the assembly (`grad_elec`, `make_rdm1e`, `get_hcore`,
//! `hcore_generator`, `get_ovlp`, `get_jk`, `grad_nuc`, `kernel`) is 18-05's
//! verbatim and ONLY `get_veff` is replaced — the XC-grid term ([`get_vxc_sets`]
//! below) plus the Coulomb / hybrid-exchange terms ([`KrksGradients::veff`]).
//! [`KrksGradients`] mirrors that split: the assembly duplicates 18-05's
//! bodies (exactly as 18-06's [`KuhfGradients`](crate::kuhf::KuhfGradients)
//! does — Rust has no inheritance), while `veff` and the grid term are new.
//!
//! # The three refusals (Task 1 — read before "fixing")
//!
//! Three capabilities the molecular port already ships are `raise
//! NotImplementedError` here, and inheriting them would make this port *more
//! capable than upstream* on the drop-in surface — a fidelity defect:
//!
//! * **`grid_response`.** `krks.py:48-49` is `if ks_grad.grid_response: raise
//!   NotImplementedError`. The molecular `pyscf-grad::rks` documents
//!   `grid_response` as **fully supported** (Becke-weight-derivative
//!   `extra_force`). [`KrksGradients::veff`] **overrides that to a refusal**:
//!   setting [`KrksGradients::with_grid_response`] carries the flag, and `veff`
//!   refuses when it is set.
//! * **meta-GGA.** `krks.py:112` — `raise NotImplementedError("metaGGA")`.
//!   [`get_vxc_sets`] maps the `XcType` meta-GGA error to the named
//!   [`PbcGradError::NotYetImplemented`].
//! * **NLC / VV10.** `krks.py:110` — `raise NotImplementedError("NLC")`.
//!   [`get_vxc_sets`] guards VV10/NLC functional strings to the same named
//!   refusal (the XC backend has no NLC family, so without the guard an NLC
//!   string would fail to parse as a generic error — or worse, parse as its
//!   semilocal part and return a number).
//!
//! # Stress asymmetry (do NOT reconcile)
//!
//! The stress tensor **DOES** support meta-GGA (`rks_stress.py:176-178`, gated
//! by `test_rks_stress.py:426`). Gradient = LDA/GGA/HF/hybrid; stress =
//! LDA/GGA/MGGA. That is upstream's shape, not an oversight.
//!
//! # Upstream correspondence (`krks.py` line → here)
//!
//! | upstream | here |
//! |---|```
//! | `get_veff` `:31-64` | [`KrksGradients::veff`] |
//! | `get_vxc` `:68-117` (LDA `:75-88`, GGA `:90-104`, HF `:106-107`) | [`get_vxc_sets`] |
//! | `NLC` `:109-110`, `metaGGA` `:111-112` | [`classify_xc`] |
//! | `grid_response` refusal `:48-49` | [`KrksGradients::veff`] |
//! | `-vmat[:,0]` squeeze + sign `:114-117` | [`get_vxc_sets`] (negation + squeeze) |
//! | `wv[0] *= .5` `:101` | INSIDE [`pyscf_grad::rks::gga_grad_sum_add`] (cannot be dropped) |
//! | RSH branch `:58-63` | [`KrksGradients::veff`] via 18-04's `get_k_e1` `omega` |
//!
//! # Index orders, `.real` placement, scopes (inherited from 18-05)
//!
//! Every density is indexed **`ji`, not `ij`** (18-CONTEXT trap 9). The grid
//! term is accumulated COMPLEX at the AO level (faithful to upstream's
//! complex `vmat`); the `.real` lives INSIDE the `grad_elec` einsum
//! contractions (D-PBC-31 clause 4 — 18-05's `contract_*_atom` helpers, proven
//! bit-identical to complex-then-`.real`). `de[x] /= nkpts` is INSIDE the
//! atom loop, `extra_force` is added AFTER the division, and
//! `vppnl_nuc_grad/nkpts` covers the WHOLE array AFTER the loop
//! (18-CONTEXT trap 3).
//!
//! # The RSH branch and the two `omega` routes (Task 3)
//!
//! `krks.py:61-63` re-enters `get_k` under
//! `cell.with_range_coulomb(omega)` with coefficient `(alpha - hyb)`.
//! Here that route is 18-04's `get_k_e1` `omega: Option<f64>` parameter
//! (FFTDF, the same parameter `fft_jk.rs`'s energy path takes): no cell is
//! mutated, the long-range kernel is selected per call. This is UNRELATED to
//! the GDF `omega` carry-over still blocked at `gdf/jk.rs:674` (STATE.md) —
//! the two blockers must not be conflated.
//!
//! # Reductions and kernels (ALG-06, D-PBC-17, D-PBC-31)
//!
//! Every reduction routes through `pyscf_algebra::oracle_sum` over a
//! materialised partial buffer — never a bare `+=`.
//!
//! No CubeCL kernel is added here (ALG-06: `pyscf-pbc-grad` may not depend on
//! `cubecl-*` AT ALL; `xtask check-dependency-wall` enforces it). The CubeCL
//! manual (`manual/Cubecl/INDEX.md`: generics-`Float` kernels, reduction and
//! coalescing discipline) was read before writing; there is no device kernel
//! in this file for it to apply to. Device computation consumed:
//! [`fused_local_contraction`](crate::krhf::fused_local_contraction)'s
//! clause-10 shared `(natm,3)` primitive, and the deriv-2 collocation kernel
//! ([`eval_ao_deriv2_kpts`], itself a generics-`Float` kernel) for the GGA
//! Hessian rows `ao[4:10]`.
//!
//! # Grids and blocks
//!
//! The quadrature is the mean field's own grid (`mf.grids`, the uniform FFT
//! box by default — `Krks::from_df` documents it follows `cell.mesh`).
//! `max_memory` for the block partition comes from `PYSCF_MAX_MEMORY`
//! (18-CONTEXT §3.7: never by transcribing `krks.py:46-47`'s
//! `lib.current_memory()` arithmetic) via [`KNumInt`](pyscf_pbc_dft::numint::KNumInt)'s
//! block sizer. The LDA blocks are sized exactly (deriv-1 tables are 4
//! components, the sizer's GGA width); the GGA deriv-2 tables are 10 wide,
//! so each sized range is walked in three contiguous sub-ranges.

use pyscf_algebra::{CTensor, oracle_sum, select_backend};
use pyscf_core::{CoreError, PyscfRsError};
use pyscf_grad::rks::{d1_dot_add, gga_grad_sum_add};
use pyscf_pbc_df::JkOpts;
use pyscf_pbc_gto::{Cell, ExxDiv};
use pyscf_pbc_scf::types::{KDms, KMats};
use pyscf_pbc_scf::krdm::make_rdm1;
use pyscf_pbc_dft::krks::Krks;
use pyscf_pbc_dft::numint::KNumInt;
use pyscf_pbc_dft::xc::{
    eval_xc_eff_rks, eval_xc_eff_uks, is_hybrid_xc, rsh_and_hybrid_coeff, RhoEff, XcType,
};

use crate::error::PbcGradError;
use crate::gradients::{EnergyScanner, GradMatrices, Gradient, Gradients};
use crate::krhf::{
    contractions::{contract_h1_atom, contract_ovlp_atom, contract_vhf_atom},
    fused_local_contraction,
    hcore_deriv_matrices,
    make_rdm1e_kpts,
    precompute_hcore,
    HcoreFusedStats,
};

fn invalid(message: impl Into<String>) -> PyscfRsError {
    PyscfRsError::Core(CoreError::InvalidMolecule(message.into()))
}

fn lift_dft<T>(r: Result<T, pyscf_pbc_dft::PbcDftError>) -> Result<T, PyscfRsError> {
    r.map_err(|e| match e {
        pyscf_pbc_dft::PbcDftError::Core(inner) => inner,
        other => invalid(format!("KRKS gradient: DFT layer failed: {other}")),
    })
}

fn df_err(context: &'static str, e: pyscf_pbc_df::PbcDfError) -> PyscfRsError {
    match e {
        pyscf_pbc_df::PbcDfError::Core(c) => c,
        other => PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "KRKS gradient ({context}): density fitting failed: {other}"
        ))),
    }
}

/// Number of k-points, treating an empty list as the single gamma point
/// (the `pbc_intor` convention; same helper as 18-05).
fn nkpts_of(kpts: &[[f64; 3]]) -> usize {
    if kpts.is_empty() { 1 } else { kpts.len() }
}

// ---------------------------------------------------------------------------
// Task 1: functional classification — LDA / GGA / HF implemented, NLC and
// meta-GGA refused by name (krks.py:106-112).
// ---------------------------------------------------------------------------

/// What [`classify_xc`] admits into [`get_vxc_sets`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum XcKind {
    Lda,
    Gga,
    /// `xc = 'HF'`: upstream's `:106-107` `pass` — the grid term is zeros and
    /// the exchange comes from the hybrid branch of `get_veff`.
    Hf,
}

/// Uppercase substrings naming meta-GGA functionals (`krks.py:112`'s
/// `metaGGA` branch). Needed because the default xcfun backend parses no
/// meta-GGA token at all (MGGA names live only in the feature-gated libxc
/// parser): without this guard an MGGA string fails to parse as a generic
/// "unknown token" error instead of the named `metaGGA` refusal. Every entry
/// is a functional upstream's `ni._xc_type` classifies `MGGA` (checked
/// against `pyscf/dft/libxc.py` + `xc/utils.py`); no LDA/GGA/HF name contains
/// one ("M06" covers M06L/M062X/M06HF, "M05" covers M05/M052X, "SCAN" covers
/// RSCAN/R2SCAN).
const MGGA_MARKERS: &[&str] = &[
    "TPSS", "SCAN", "M06", "M05", "M11", "M08", "MN12", "MN15", "N12", "VSXC", "PKZB", "MS0",
    "MS1", "MS2", "MVS", "TASK", "BLOC", "HLE17",
];

/// `ni._xc_type(xc_code)` plus the two named refusals (`krks.py:109-112`).
///
/// NLC is guarded FIRST by string (the XC backend has no NLC family: without
/// the guard an NLC string fails to parse as a generic error). `metaGGA` is
/// guarded by [`MGGA_MARKERS`] first and mapped out of [`XcType::of`]'s error
/// second (the latter covers the `--features libxc` build, where MGGA tokens
/// parse and classify) — the same two-net shape `stress/rks.rs` uses.
///
/// # Errors
/// [`PbcGradError::NotYetImplemented`] for NLC and meta-GGA; otherwise the
/// backend parse failure.
fn classify_xc(xc_code: &str) -> Result<XcKind, PyscfRsError> {
    let up = xc_code.to_ascii_uppercase();
    if up.contains("VV10") || up.contains("NLC") {
        return Err(PbcGradError::NotYetImplemented {
            phase: 18,
            what: "KRKS/KUKS get_vxc NLC (VV10 non-local correlation): upstream \
                   krks.py:110 / kuks.py:116 raise NotImplementedError(\"NLC\")",
        }
        .into());
    }
    if MGGA_MARKERS.iter().any(|m| up.contains(m)) {
        return Err(mgga_refusal());
    }
    if up.trim() == "HF" {
        return Ok(XcKind::Hf);
    }
    match XcType::of(xc_code) {
        Ok(XcType::Lda) => Ok(XcKind::Lda),
        Ok(XcType::Gga) => Ok(XcKind::Gga),
        Err(e) if e.to_string().contains("meta-GGA") => Err(mgga_refusal()),
        Err(e) => Err(lift_dft::<()>(Err(e)).expect_err("classify_xc: Err branch")),
    }
}

/// The shared `metaGGA` refusal value (`krks.py:112` / `kuks.py:118`).
fn mgga_refusal() -> PyscfRsError {
    PbcGradError::NotYetImplemented {
        phase: 18,
        what: "KRKS/KUKS get_vxc metaGGA (tau-dependent): upstream krks.py:112 / \
               kuks.py:118 raise NotImplementedError(\"metaGGA\"); the stress tensor \
               DOES support meta-GGA (rks_stress.py:176-178) — that asymmetry is \
               upstream's shape, not an oversight",
    }
    .into()
}

// ---------------------------------------------------------------------------
// (x, set, k) shape handling — the 18-05 seam, nset-generic.
// ---------------------------------------------------------------------------

/// Take one half of an 18-04 gradient JK result (`[x][set][k]`, single set)
/// as `[x][k]` row-major matrices (18-05's `take_grad_mats`, repeated here so
/// `krhf.rs` is untouched by this plan).
pub(crate) fn take_grad_mats(
    mats: pyscf_pbc_df::fft_jk_grad::GradMats,
    nkpts: usize,
    nao: usize,
    what: &'static str,
) -> Result<GradMatrices, PyscfRsError> {
    if mats.len() != 3 || mats.iter().any(|s| s.len() != 1 || s[0].len() != nkpts) {
        return Err(invalid(format!(
            "KRKS get_jk: {what} has the wrong (x, set, k) shape for nkpts = {nkpts}"
        )));
    }
    Ok(std::array::from_fn(|x| {
        mats[x][0]
            .iter()
            .map(|m| {
                debug_assert_eq!(m.re.len(), nao * nao);
                m.clone()
            })
            .collect()
    }))
}

/// The nset-generic reshape (18-06's `take_sets`, repeated here so neither
/// `krhf.rs` nor `kuhf.rs` is touched by this plan).
pub(crate) fn take_grad_mats_sets(
    mats: pyscf_pbc_df::fft_jk_grad::GradMats,
    what: &str,
    nset: usize,
    nkpts: usize,
    nao: usize,
) -> Result<Vec<GradMatrices>, PyscfRsError> {
    if mats.len() != 3 || mats.iter().any(|s| s.len() != nset || s.iter().any(|k| k.len() != nkpts))
    {
        return Err(invalid(format!(
            "KRKS get_jk: {what} has the wrong (x, set, k) shape for nset = {nset}, nkpts = {nkpts}"
        )));
    }
    let mut out: Vec<GradMatrices> = Vec::with_capacity(nset);
    for s in 0..nset {
        out.push(std::array::from_fn(|x| {
            mats[x][s].iter().map(|m| m.clone()).collect::<KMats>()
        }));
    }
    for set in &out {
        for kxm in set.iter().flatten() {
            if kxm.re.len() != nao * nao || kxm.im.len() != nao * nao {
                return Err(invalid(format!(
                    "KRKS get_jk: {what} plane is not nao x nao = {} for nao = {nao}",
                    kxm.re.len()
                )));
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// GGA Hessian rows — the deriv-2 collocation (krks.py:91 `ao_deriv = 2`).
// ---------------------------------------------------------------------------

/// Per-k-point periodic second-derivative AO tables: `re[k]` / `im[k]` are
/// `10 * ngrids * nao` F-order per component (value, `gx, gy, gz`, `xx, xy,
/// xz, yy, yz, zz`) — the layout `pyscf-pbc-gto`'s `EvalAoKptsOutput` uses,
/// so rows slice-compatibly into both [`KNumInt::eval_rho`] (rows `0..4`)
/// and [`gga_grad_sum_add`](pyscf_grad::rks::gga_grad_sum_add) (rows `0..10`).
pub(crate) struct Deriv2Tables {
    pub re: Vec<Vec<f64>>,
    pub im: Vec<Vec<f64>>,
    pub ngrids: usize,
    pub nao: usize,
}

/// Collocate the deriv-2 table through `pyscf_kernels::pbc::eval_ao_deriv2`
/// (the 18-12-validated kernel: its Hessian rows satisfy the strain-FD
/// relation at the upstream `:240` bound).
///
/// The image list is the same one [`pyscf_pbc_gto::eval_ao_kpts`] would build
/// at `deriv = 2` (`estimate_rcut_for_eval` + `get_lattice_ls`), so the
/// lattice sum is converged to the same tolerance as the deriv-1 tables this
/// is combined with. Cartesian cells are refused: the kernel decodes them,
/// but no FD gate validates the Cartesian ordering here.
///
/// # Errors
/// The named Cartesian refusal; otherwise the backend selection, rcut, image
/// list and kernel failures.
pub(crate) fn eval_ao_deriv2_kpts(
    cell: &Cell,
    coords: &[[f64; 3]],
    kpts: &[[f64; 3]],
) -> Result<Deriv2Tables, PyscfRsError> {
    if cell.mol.cart {
        return Err(PbcGradError::NotYetImplemented {
            phase: 18,
            what: "KRKS/KUKS get_vxc GGA for Cartesian cells: the deriv-2 kernel \
                   ordering is FD-validated for spherical AOs only",
        }
        .into());
    }
    let nao = cell.mol.nao_nr;
    let ngrids = coords.len();
    let rcut = pyscf_pbc_gto::eval_gto::estimate_rcut_for_eval(cell, 2).map_err(|e| {
        invalid(format!("KRKS get_vxc GGA: rcut estimation failed: {e}"))
    })?;
    let rmax = rcut.iter().copied().fold(0.0_f64, f64::max);
    let ls = pyscf_pbc_gto::get_lattice_ls(cell, Some(rmax), None, false)
        .map_err(|e| invalid(format!("KRKS get_vxc GGA: lattice list failed: {e}")))?;
    let client = select_backend()
        .map_err(|e| invalid(format!("KRKS get_vxc GGA: backend selection failed: {e}")))?
        .client;
    let out = pyscf_kernels::pbc::eval_ao_deriv2(
        &client,
        coords,
        kpts,
        &ls,
        &cell.mol._atm,
        &cell.mol._bas,
        &cell.mol._env,
        &cell.mol.atom_coords(),
        true,
        Some(&rcut),
        false,
    )
    .map_err(|e| invalid(format!("KRKS get_vxc GGA: deriv-2 collocation failed: {e}")))?;
    if out.ngrids != ngrids || out.nao != nao || out.re.len() != kpts.len() {
        return Err(invalid(format!(
            "KRKS get_vxc GGA: deriv-2 table has ngrids = {}, nao = {}, nkpts = {} \
             for ngrids = {ngrids}, nao = {nao}, nkpts = {}",
            out.ngrids,
            out.nao,
            out.re.len(),
            kpts.len(),
        )));
    }
    for (k, (re, im)) in out.re.iter().zip(out.im.iter()).enumerate() {
        if re.len() != 10 * ngrids * nao || im.len() != 10 * ngrids * nao {
            return Err(invalid(format!(
                "KRKS get_vxc GGA: deriv-2 k-point {k} has {}/{} entries, need {}",
                re.len(),
                im.len(),
                10 * ngrids * nao,
            )));
        }
    }
    Ok(Deriv2Tables {
        re: out.re,
        im: out.im,
        ngrids,
        nao,
    })
}

// ---------------------------------------------------------------------------
// Task 2: `get_vxc` — the LDA and GGA grid terms (krks.py:68-117).
// ---------------------------------------------------------------------------

/// Validate one density set: `nkpts` finite `nao × nao` row-major blocks.
fn check_dm_set(dm: &KMats, nao: usize, nkpts: usize, what: &str) -> Result<(), PyscfRsError> {
    if dm.len() != nkpts
        || dm.iter().any(|m| {
            m.re.len() != nao * nao || m.im.len() != nao * nao || m.len() != nao * nao
        })
        || dm
            .iter()
            .any(|m| m.re.iter().chain(&m.im).any(|v| !v.is_finite()))
    {
        return Err(invalid(format!(
            "KRKS/KUKS get_vxc: {what} needs {nkpts} finite nao x nao = {} densities",
            nao * nao,
        )));
    }
    Ok(())
}

/// `get_vxc(ni, cell, grids, xc_code, dms, kpts)` — `krks.py:68-117` /
/// `kuks.py:70-120`, nset-generic over the density sets (1 for KRKS, 2 for
/// KUKS — the spin threading differs only in the caller, exactly as 18-06
/// Task 2).
///
/// Per grid block (`PYSCF_MAX_MEMORY`-sized): collocate the AO table
/// (`deriv1` for LDA, deriv-2 for GGA), build `rho` through
/// [`KNumInt::eval_rho`] (the `_gen_rho_evaluator` equivalent — BZ-averaged,
/// `hermi = 1` only), evaluate `eval_xc_eff` at `deriv = 1`, and accumulate
/// into `vmat` through the GRAD-10 shared molecular primitives
/// ([`d1_dot_add`] for LDA, `krks.py:87`; [`gga_grad_sum_add`] for GGA,
/// `:103`).
///
/// The return is **negated** (`-vmat`, `:114-117` — the sign lives here), one
/// `[x][k]` complex matrix family per set, column-accumulated then
/// transposed to row-major (an exact reorder).
///
/// The rho rows and the contraction rows come from the SAME table (the rho
/// block is a row-slice copy), so no cross-table roundoff enters — this is
/// the Rust analogue of upstream building `rho` from `ao_k2`'s own rows.
///
/// # Errors
/// The NLC / meta-GGA named refusals ([`classify_xc`]); otherwise the grid,
/// AO, rho, XC and shape failures.
pub(crate) fn get_vxc_sets(
    cell: &Cell,
    dm_sets: &KDms,
    xc_code: &str,
    grids: &pyscf_pbc_dft::gen_grid::PeriodicGrids,
    kpts: &[[f64; 3]],
) -> Result<Vec<GradMatrices>, PyscfRsError> {
    let kind = classify_xc(xc_code)?;
    let nao = cell.mol.nao_nr;
    let nset = dm_sets.len();
    if nset == 0 {
        return Err(invalid("KRKS/KUKS get_vxc: need at least one density set"));
    }
    let owned_gamma = [[0.0_f64; 3]];
    let kpts_nz: &[[f64; 3]] = if kpts.is_empty() { &owned_gamma } else { kpts };
    let nkpts = kpts_nz.len();
    for (s, dm) in dm_sets.iter().enumerate() {
        check_dm_set(dm, nao, nkpts, &format!("set {s}"))?;
    }
    let coords = lift_dft(grids.coords())?;
    let weights = lift_dft(grids.weights())?;
    let ngrids = coords.len();
    if weights.len() != ngrids {
        return Err(invalid(format!(
            "KRKS/KUKS get_vxc: grid has {ngrids} coords but {} weights",
            weights.len()
        )));
    }
    // Column-major accumulators `[3 * nao * nao]` per (set, k), zero start.
    let mut acc_re = vec![vec![0.0_f64; 3 * nao * nao]; nset * nkpts];
    let mut acc_im = vec![vec![0.0_f64; 3 * nao * nao]; nset * nkpts];
    if ngrids == 0 || matches!(kind, XcKind::Hf) {
        // HF (`krks.py:106-107` `pass`): the grid term is zeros; the return
        // below still negates (exact `-0.0` plane) and squeezes per set.
    } else {
        // `PYSCF_MAX_MEMORY`-sized block partition (18-CONTEXT §3.7). The
        // sizer assumes energy-table widths; deriv-1 tables are exactly its
        // GGA width (4 comps), while deriv-2 tables are 10 wide, so each
        // sized GGA range is walked in three contiguous sub-ranges.
        let ni = KNumInt::new(kpts_nz);
        let sized = ni.block_ranges(ngrids, XcType::Gga, nkpts);
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for (p0, p1) in sized {
            if matches!(kind, XcKind::Gga) {
                let len = p1 - p0;
                let third = (len + 2) / 3;
                let mut q0 = p0;
                while q0 < p1 {
                    let q1 = (q0 + third).min(p1);
                    ranges.push((q0, q1));
                    q0 = q1;
                }
            } else {
                ranges.push((p0, p1));
            }
        }
        // Scratch density tables are built per block; the XC potential is
        // pointwise, so per-block eval == full-grid eval sliced, bit for bit.
        for (p0, p1) in ranges {
            let blen = p1 - p0;
            let chunk = &coords[p0..p1];
            if matches!(kind, XcKind::Lda) {
                let ao = pyscf_pbc_gto::eval_ao_kpts(cell, "GTOval_sph_deriv1", chunk, kpts_nz)
                    .map_err(|e| {
                        invalid(format!("KRKS/KUKS get_vxc LDA: deriv-1 AO failed: {e}"))
                    })?;
                if ao.comp != 4 || ao.ngrids != blen || ao.nao != nao || ao.nkpts() != nkpts {
                    return Err(invalid(format!(
                        "KRKS/KUKS get_vxc LDA: deriv-1 table has comp = {}, ngrids = {}, \
                         nao = {}, nkpts = {} for 4/{blen}/{nao}/{nkpts}",
                        ao.comp, ao.ngrids, ao.nao, ao.nkpts(),
                    )));
                }
                // Rho rows are row-0 slice copies of THIS table.
                let rho_table = pyscf_pbc_gto::eval_gto::EvalAoKptsOutput {
                    kaos: ao
                        .kaos
                        .iter()
                        .map(|k| {
                            CTensor::from_planes(
                                k.re[..blen * nao].to_vec(),
                                k.im[..blen * nao].to_vec(),
                            )
                        })
                        .collect(),
                    ngrids: blen,
                    nao,
                    comp: 1,
                    gamma: vec![false; nkpts],
                };
                let mut rhos: Vec<RhoEff> = Vec::with_capacity(nset);
                for dm in dm_sets.iter() {
                    rhos.push(lift_dft(ni.eval_rho(&rho_table, dm, XcType::Lda)).map_err(
                        |e| invalid(format!("KRKS/KUKS get_vxc LDA: rho build failed: {e}")),
                    )?);
                }
                // `eval_xc_eff(deriv=1)` per spin shape; `wv = vxc * weight`.
                let mut wvs: Vec<Vec<f64>> = Vec::with_capacity(nset);
                if nset == 1 {
                    let vxc = lift_dft(eval_xc_eff_rks(xc_code, &rhos[0])).map_err(|e| {
                        invalid(format!("KRKS get_vxc LDA: XC evaluation failed: {e}"))
                    })?;
                    let row = vxc.row(0, 0);
                    wvs.push(
                        row.iter()
                            .enumerate()
                            .map(|(g, v)| v * weights[p0 + g])
                            .collect(),
                    );
                } else {
                    if rhos.len() != 2 {
                        return Err(invalid(
                            "KUKS get_vxc LDA: open-shell XC needs exactly 2 density sets",
                        ));
                    }
                    let vxc =
                        lift_dft(eval_xc_eff_uks(xc_code, &rhos[0], &rhos[1])).map_err(|e| {
                            invalid(format!("KUKS get_vxc LDA: XC evaluation failed: {e}"))
                        })?;
                    for s in 0..2 {
                        let row = vxc.row(s, 0);
                        wvs.push(
                            row.iter()
                                .enumerate()
                                .map(|(g, v)| v * weights[p0 + g])
                                .collect(),
                        );
                    }
                }
                let row_at = |k: usize, c: usize| {
                    let base = c * blen * nao;
                    (&ao.kaos[k].re[base..base + blen * nao], &ao.kaos[k].im[base..base + blen * nao])
                };
                for (s, wv) in wvs.iter().enumerate() {
                    for k in 0..nkpts {
                        let (r0, i0) = row_at(k, 0);
                        let (r1, i1) = row_at(k, 1);
                        let (r2, i2) = row_at(k, 2);
                        let (r3, i3) = row_at(k, 3);
                        d1_dot_add(
                            &mut acc_re[s * nkpts + k],
                            &mut acc_im[s * nkpts + k],
                            [r1, r2, r3],
                            Some([i1, i2, i3]),
                            r0,
                            Some(i0),
                            wv,
                            nao,
                            blen,
                        );
                    }
                }
            } else {
                // GGA (`krks.py:90-104`, `ao_deriv = 2`).
                let d2 = eval_ao_deriv2_kpts(cell, chunk, kpts_nz)?;
                if d2.ngrids != blen || d2.nao != nao || d2.re.len() != nkpts {
                    return Err(invalid(format!(
                        "KRKS/KUKS get_vxc GGA: deriv-2 table has ngrids = {}, nao = {}, \
                         nkpts = {} for {blen}/{nao}/{nkpts}",
                        d2.ngrids,
                        d2.nao,
                        d2.re.len(),
                    )));
                }
                let rho_table = pyscf_pbc_gto::eval_gto::EvalAoKptsOutput {
                    kaos: d2
                        .re
                        .iter()
                        .zip(d2.im.iter())
                        .map(|(re, im)| {
                            CTensor::from_planes(
                                re[..4 * blen * nao].to_vec(),
                                im[..4 * blen * nao].to_vec(),
                            )
                        })
                        .collect(),
                    ngrids: blen,
                    nao,
                    comp: 4,
                    gamma: vec![false; nkpts],
                };
                let mut rhos: Vec<RhoEff> = Vec::with_capacity(nset);
                for dm in dm_sets.iter() {
                    rhos.push(lift_dft(ni.eval_rho(&rho_table, dm, XcType::Gga)).map_err(
                        |e| invalid(format!("KRKS/KUKS get_vxc GGA: rho build failed: {e}")),
                    )?);
                }
                // UNSCALED `wv[c] = vxc[c] * weight`: the shared primitive
                // halves row 0 internally (upstream `krks.py:101`).
                let mut wv_sets: Vec<[Vec<f64>; 4]> = Vec::with_capacity(nset);
                if nset == 1 {
                    let vxc = lift_dft(eval_xc_eff_rks(xc_code, &rhos[0])).map_err(|e| {
                        invalid(format!("KRKS get_vxc GGA: XC evaluation failed: {e}"))
                    })?;
                    wv_sets.push(std::array::from_fn(|c| {
                        vxc.row(0, c)
                            .iter()
                            .enumerate()
                            .map(|(g, v)| v * weights[p0 + g])
                            .collect()
                    }));
                } else {
                    if rhos.len() != 2 {
                        return Err(invalid(
                            "KUKS get_vxc GGA: open-shell XC needs exactly 2 density sets",
                        ));
                    }
                    let vxc =
                        lift_dft(eval_xc_eff_uks(xc_code, &rhos[0], &rhos[1])).map_err(|e| {
                            invalid(format!("KUKS get_vxc GGA: XC evaluation failed: {e}"))
                        })?;
                    for s in 0..2 {
                        wv_sets.push(std::array::from_fn(|c| {
                            vxc.row(s, c)
                                .iter()
                                .enumerate()
                                .map(|(g, v)| v * weights[p0 + g])
                                .collect()
                        }));
                    }
                }
                let row_at = |k: usize, c: usize| {
                    let base = c * blen * nao;
                    (
                        &d2.re[k][base..base + blen * nao],
                        &d2.im[k][base..base + blen * nao],
                    )
                };
                for (s, wv) in wv_sets.iter().enumerate() {
                    let wv_rows: [&[f64]; 4] = [&wv[0], &wv[1], &wv[2], &wv[3]];
                    for k in 0..nkpts {
                        let (re_rows, im_rows): (Vec<&[f64]>, Vec<&[f64]>) = (0..10)
                            .map(|c| row_at(k, c))
                            .unzip();
                        let re_arr: [&[f64]; 10] = re_rows.try_into().expect("10 rows");
                        let im_arr: [&[f64]; 10] = im_rows.try_into().expect("10 rows");
                        gga_grad_sum_add(
                            &mut acc_re[s * nkpts + k],
                            &mut acc_im[s * nkpts + k],
                            re_arr,
                            Some(im_arr),
                            wv_rows,
                            nao,
                            blen,
                        );
                    }
                }
            }
        }
    }
    // `return -vmat` (`:114-117`): negate into row-major `[x][k]` families.
    let n2 = nao * nao;
    let mut out: Vec<GradMatrices> = Vec::with_capacity(nset);
    for s in 0..nset {
        out.push(std::array::from_fn(|x| {
            (0..nkpts)
                .map(|k| {
                    let mut re = vec![0.0_f64; n2];
                    let mut im = vec![0.0_f64; n2];
                    for i in 0..nao {
                        for j in 0..nao {
                            re[i * nao + j] = -acc_re[s * nkpts + k][x * n2 + i + j * nao];
                            im[i * nao + j] = -acc_im[s * nkpts + k][x * n2 + i + j * nao];
                        }
                    }
                    CTensor::from_planes(re, im)
                })
                .collect()
        }));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// `KrksGradients` — `class Gradients(rhf_grad.Gradients)` (krks.py:119).
// ---------------------------------------------------------------------------

/// KRKS k-point nuclear gradient — `pbc/grad/krks.py`'s `Gradients` class.
///
/// Borrowed mean field ([`Krks`], owning `with_df`, `cell`, `kpts`, `xc`,
/// `grids`, `exxdiv`) plus caller-held SCF products: the density `dm0` and
/// the energy-weighted density `dme0` are built by [`KrksGradients::new`]
/// from the converged orbitals through [`make_rdm1`] and 18-05's
/// [`make_rdm1e_kpts`] (`krks.py` inherits both from `krhf`). No SCF runs
/// here, so there is no convergence path and no second-solution noise.
///
/// `grid_response` defaults OFF and is a REFUSAL when set (`krks.py:48-49`;
/// see the module docs — the molecular default-off-but-supported term does
/// not exist here). `extra_force` is the shared zero default.
pub struct KrksGradients<'a> {
    mf: &'a Krks,
    dm0: KDms,
    dme0: KMats,
    atmlst: Option<Vec<usize>>,
    grid_response: bool,
}

impl<'a> KrksGradients<'a> {
    /// Build from converged orbitals. `mo_coeff` is column-major `nao × nmo`
    /// per k-point; `mo_energy`/`mo_occ` are per-orbital per k-point.
    ///
    /// # Errors
    /// [`CoreError::InvalidMolecule`] on k-point/shape disagreement or
    /// non-finite input.
    pub fn new(
        mf: &'a Krks,
        mo_energy: Vec<Vec<f64>>,
        mo_coeff: Vec<CTensor>,
        mo_occ: Vec<Vec<f64>>,
    ) -> Result<Self, PyscfRsError> {
        let nkpts = nkpts_of(mf.kpts());
        let nao = mf.cell().mol.nao_nr;
        if mo_coeff.len() != nkpts || mo_energy.len() != nkpts || mo_occ.len() != nkpts {
            return Err(invalid(format!(
                "KRKS gradient: {nkpts} k-points but {}/{}/{} coeff/energy/occ blocks",
                mo_coeff.len(),
                mo_energy.len(),
                mo_occ.len(),
            )));
        }
        let dm0 = vec![make_rdm1(&mo_coeff, &mo_occ, nao)];
        for m in &dm0[0] {
            if m.re.iter().chain(&m.im).any(|v| !v.is_finite()) {
                return Err(invalid("KRKS gradient: density is non-finite"));
            }
        }
        let dme0 = make_rdm1e_kpts(&mo_coeff, &mo_energy, &mo_occ, nao)?;
        Ok(Self {
            mf,
            dm0,
            dme0,
            atmlst: None,
            grid_response: false,
        })
    }

    /// Atom subset (`de = de[atmlst]`). `None` (default) is all atoms.
    pub fn with_atmlst(mut self, atmlst: Vec<usize>) -> Result<Self, PyscfRsError> {
        for &ia in &atmlst {
            if ia >= self.mf.cell().natm {
                return Err(invalid(format!(
                    "KRKS gradient: atmlst id {ia} out of range for {} atoms",
                    self.mf.cell().natm
                )));
            }
        }
        self.atmlst = Some(atmlst);
        Ok(self)
    }

    /// Set the `grid_response` flag (default OFF). Setting it to `true` is
    /// NOT support for the Becke-weight term — [`KrksGradients::veff`]
    /// refuses it by name (`krks.py:48-49`).
    pub fn with_grid_response(mut self, on: bool) -> Self {
        self.grid_response = on;
        self
    }

    fn atom_list(&self) -> Vec<usize> {
        match &self.atmlst {
            Some(list) => list.clone(),
            None => (0..self.mf.cell().natm).collect(),
        }
    }

    /// `get_ovlp(cell, kpts)` — inherited unchanged from 18-05's base
    /// (`krhf.py:114-115`): `-cell.pbc_intor('int1e_ipovlp', kpts)`,
    /// `[x][k]` row-major. (The Rust port repeats the 18-05 body because it
    /// borrows a different mf type.)
    pub fn overlap_deriv(&self) -> Result<GradMatrices, PyscfRsError> {
        let cell = self.mf.cell();
        let kpts = self.mf.kpts();
        let nao = cell.mol.nao_nr;
        let out = pyscf_pbc_gto::pbc_intor(cell, "int1e_ipovlp", kpts, Default::default())?;
        if out.comp != 3 || out.ni != nao || out.nj != nao {
            return Err(invalid(format!(
                "KRKS get_ovlp: int1e_ipovlp has comp = {}, ni/nj = {}/{} for nao = {nao}",
                out.comp, out.ni, out.nj
            )));
        }
        Ok(std::array::from_fn(|x| {
            out.kmats
                .iter()
                .map(|m| {
                    let mut re = vec![0.0_f64; nao * nao];
                    let mut im = vec![0.0_f64; nao * nao];
                    for i in 0..nao {
                        for j in 0..nao {
                            let (r, v) = (
                                m.re[x * nao * nao + i + j * nao],
                                m.im[x * nao * nao + i + j * nao],
                            );
                            re[i * nao + j] = -r;
                            im[i * nao + j] = -v;
                        }
                    }
                    CTensor::from_planes(re, im)
                })
                .collect()
        }))
    }

    /// The exchange-divergence treatment carried by the mean field.
    pub fn exxdiv(&self) -> Option<ExxDiv> {
        self.mf.exxdiv
    }

    /// `get_jk(dm, kpts)` — the inherited 18-04 route (FFTDF `get_jk_e1`,
    /// the ONLY density-fitting route with a gradient). Same options and
    /// same refusals as 18-05, including `kk_symmetry: false` (D-PBC-30
    /// clause 4b).
    pub fn jk_deriv(&self, dm: &KDms) -> Result<(GradMatrices, GradMatrices), PyscfRsError> {
        let mf = self.mf;
        let nkpts = nkpts_of(mf.kpts());
        let nao = mf.cell().mol.nao_nr;
        let res = mf
            .with_df
            .get_jk_e1(
                dm,
                mf.kpts(),
                JkOpts {
                    hermi: 1,
                    kpts_band: None,
                    with_j: true,
                    with_k: true,
                    exxdiv: mf.exxdiv,
                    omega: None,
                    kk_symmetry: false,
                },
                None,
            )
            .map_err(|e| df_err("get_jk", e))?;
        let vj = take_grad_mats(
            res.vj.ok_or_else(|| invalid("KRKS get_jk: missing vj"))?,
            nkpts,
            nao,
            "vj",
        )?;
        let vk = take_grad_mats(
            res.vk.ok_or_else(|| invalid("KRKS get_jk: missing vk"))?,
            nkpts,
            nao,
            "vk",
        )?;
        Ok((vj, vk))
    }

    /// `get_j(dm, kpts)` — routed directly (not split out of
    /// [`Self::jk_deriv`]), exactly as upstream routes it, so a non-FFTDF
    /// builder serves 18-04's named `get_j_e1` refusal rather than a
    /// `get_jk_e1` one.
    pub fn j_deriv(&self, dm: &KDms) -> Result<GradMatrices, PyscfRsError> {
        let mf = self.mf;
        let nkpts = nkpts_of(mf.kpts());
        let nao = mf.cell().mol.nao_nr;
        let vj = mf
            .with_df
            .get_j_e1(dm, mf.kpts(), None)
            .map_err(|e| df_err("get_j", e))?;
        take_grad_mats(vj, nkpts, nao, "vj")
    }

    /// `get_k(dm, kpts)` — routed directly, with `mf.exxdiv`.
    pub fn k_deriv(&self, dm: &KDms) -> Result<GradMatrices, PyscfRsError> {
        self.k_deriv_omega(dm, None)
    }

    /// `get_k` under a range-separated Coulomb kernel — the RSH re-entry
    /// (`krks.py:61-63`): `None`/zero `omega` is the full-range kernel; a
    /// nonzero `omega` selects 18-04's long-range `get_k_e1` route (the same
    /// parameter `fft_jk.rs`'s energy path takes — NOT the blocked GDF
    /// carry-over at `gdf/jk.rs:674`).
    pub fn k_deriv_omega(
        &self,
        dm: &KDms,
        omega: Option<f64>,
    ) -> Result<GradMatrices, PyscfRsError> {
        let mf = self.mf;
        let nkpts = nkpts_of(mf.kpts());
        let nao = mf.cell().mol.nao_nr;
        let vk = mf
            .with_df
            .get_k_e1(dm, mf.kpts(), None, mf.exxdiv, omega, None)
            .map_err(|e| df_err("get_k", e))?;
        take_grad_mats(vk, nkpts, nao, "vk")
    }

    /// `get_vxc` squeezed to the single set (`krks.py:114-115`:
    /// `return -vmat[:,0]`; the negation already happened in
    /// [`get_vxc_sets`]).
    pub fn vxc(&self, dm: &KDms) -> Result<GradMatrices, PyscfRsError> {
        if dm.len() != 1 {
            return Err(invalid(format!(
                "KRKS get_vxc: need exactly 1 density set, got {}",
                dm.len()
            )));
        }
        let mut out = get_vxc_sets(
            self.mf.cell(),
            dm,
            &self.mf.xc,
            &self.mf.grids,
            self.mf.kpts(),
        )?;
        if out.len() != 1 {
            return Err(invalid(format!(
                "KRKS get_vxc: grid core returned {} sets, need 1",
                out.len()
            )));
        }
        Ok(out.remove(0))
    }

    /// `get_veff(ks_grad, dm, kpts)` — `krks.py:31-64`:
    ///
    /// ```text
    /// vxc = get_vxc(...)                       (refuses grid_response first)
    /// vxc += get_j                              (pure functional)
    /// vxc += vj - vk * .5                       (hybrid; vk *= hyb, RSH re-entry)
    /// ```
    ///
    /// The `grid_response` refusal (`:48-49`) fires BEFORE any number is
    /// produced — this is the override Task 1 requires, not an inheritance
    /// of the molecular "fully supported" term.
    pub fn veff(&self, dm: &KDms) -> Result<GradMatrices, PyscfRsError> {
        if self.grid_response {
            return Err(PbcGradError::NotYetImplemented {
                phase: 18,
                what: "KRKS get_veff grid_response = True: upstream krks.py:48-49 raises \
                       NotImplementedError; the molecular pyscf-grad rks.rs documents \
                       grid_response as fully supported, so the periodic subclass \
                       OVERRIDES it to a refusal",
            }
            .into());
        }
        let xc = &self.mf.xc;
        let vxc = self.vxc(dm)?;
        if !lift_dft(is_hybrid_xc(xc))? {
            let vj = self.j_deriv(dm)?;
            Ok(combine_veff(&vxc, &vj))
        } else {
            let (omega, alpha, hyb) = lift_dft(rsh_and_hybrid_coeff(xc))?;
            let (vj, vk_full) = self.jk_deriv(dm)?;
            let mut vk = scale_mats(&vk_full, hyb);
            if omega != 0.0 {
                let vk_lr = self.k_deriv_omega(dm, Some(omega))?;
                add_scaled_mats_in_place(&mut vk, &vk_lr, alpha - hyb);
            }
            Ok(combine_veff_hybrid(&vxc, &vj, &vk))
        }
    }

    /// `grad_elec` — 18-05's assembly verbatim (upstream inherits it
    /// unchanged): clause-7 fused local-PP contraction, `/nkpts` INSIDE the
    /// atom loop, `extra_force` (zero here) AFTER the division, `vppnl` over
    /// the WHOLE array AFTER the loop.
    pub fn electronic_gradient(&self) -> Result<Gradient, PyscfRsError> {
        let mf = self.mf;
        let cell = mf.cell();
        let kpts = mf.kpts();
        let nkpts = nkpts_of(kpts) as f64;
        let nao = cell.mol.nao_nr;
        let dm0 = &self.dm0[0];

        let tables = precompute_hcore(cell, kpts)?;
        let s1 = self.overlap_deriv()?;
        let vhf = self.veff(&self.dm0)?;
        let mut fused_stats = HcoreFusedStats::default();
        let fused = fused_local_contraction(&tables, cell, kpts, dm0, &mut fused_stats)?;
        let slices = pyscf_gto::aoslice_by_atom(&cell.mol)
            .map_err(|e| invalid(format!("KRKS grad_elec: aoslice failed: {e}")))?;

        let atmlst = self.atom_list();
        let mut de = Vec::with_capacity(atmlst.len());
        for &ia in &atmlst {
            let (_, _, p0, p1) = slices.get(ia).copied().ok_or_else(|| {
                invalid(format!(
                    "KRKS grad_elec: no aoslice for atom {ia} (natm = {})",
                    cell.natm
                ))
            })?;
            let h1 = contract_h1_atom(&tables.h1, dm0, p0, p1, nao);
            let vv = contract_vhf_atom(&vhf, dm0, p0, p1, nao);
            let ss = contract_ovlp_atom(&s1, &self.dme0, p0, p1, nao);
            let mut row = [0.0_f64; 3];
            for x in 0..3 {
                row[x] = oracle_sum(&[fused[ia][x], h1[x], vv[x], ss[x]]) / nkpts;
            }
            let extra = self.extra_force(ia, &[])?;
            for x in 0..3 {
                row[x] = oracle_sum(&[row[x], extra[x]]);
            }
            de.push(row);
        }
        let vppnl = pyscf_pbc_gto::pseudo::vppnl_nuc_grad_kdm(cell, dm0, kpts)?;
        for (row, &ia) in de.iter_mut().zip(atmlst.iter()) {
            for x in 0..3 {
                row[x] = oracle_sum(&[row[x], vppnl[ia][x] / nkpts]);
            }
        }
        Ok(de)
    }

    /// `grad_nuc` — inherited unchanged from 18-05's base: 18-03's
    /// `ewald_nuc_grad`.
    pub fn nuclear_gradient(&self) -> Result<Gradient, PyscfRsError> {
        pyscf_pbc_gto::ewald_nuc_grad(self.mf.cell(), None, None)
    }

    /// `kernel` — inherited from `krhf.Gradients`: `grad_elec + grad_nuc`.
    pub fn kernel(&self) -> Result<Gradient, PyscfRsError> {
        let elec = self.electronic_gradient()?;
        let nuc = self.nuclear_gradient()?;
        let atmlst = self.atom_list();
        if nuc.len() != self.mf.cell().natm {
            return Err(invalid(
                "KRKS kernel: nuclear part has the wrong atom count",
            ));
        }
        Ok(elec
            .into_iter()
            .zip(atmlst)
            .map(|(e, ia)| {
                [
                    oracle_sum(&[e[0], nuc[ia][0]]),
                    oracle_sum(&[e[1], nuc[ia][1]]),
                    oracle_sum(&[e[2], nuc[ia][2]]),
                ]
            })
            .collect())
    }
}

/// `vxc + vj`, elementwise over `[x][k]` planes, both real and imaginary
/// planes through [`oracle_sum`] (D-PBC-17: no bare `+=`).
fn combine_veff(vxc: &GradMatrices, vj: &GradMatrices) -> GradMatrices {
    std::array::from_fn(|x| {
        vxc[x]
            .iter()
            .zip(vj[x].iter())
            .map(|(a, b)| {
                CTensor::from_planes(
                    a.re.iter()
                        .zip(&b.re)
                        .map(|(p, q)| oracle_sum(&[*p, *q]))
                        .collect(),
                    a.im
                        .iter()
                        .zip(&b.im)
                        .map(|(p, q)| oracle_sum(&[*p, *q]))
                        .collect(),
                )
            })
            .collect()
    })
}

/// `vxc + vj - vk * .5` (`krks.py:64`), same reduction discipline.
fn combine_veff_hybrid(
    vxc: &GradMatrices,
    vj: &GradMatrices,
    vk: &GradMatrices,
) -> GradMatrices {
    std::array::from_fn(|x| {
        vxc[x]
            .iter()
            .zip(vj[x].iter())
            .zip(vk[x].iter())
            .map(|((a, b), c)| {
                CTensor::from_planes(
                    a.re
                        .iter()
                        .zip(&b.re)
                        .zip(&c.re)
                        .map(|((p, q), r)| oracle_sum(&[*p, *q, -0.5 * *r]))
                        .collect(),
                    a.im
                        .iter()
                        .zip(&b.im)
                        .zip(&c.im)
                        .map(|((p, q), r)| oracle_sum(&[*p, *q, -0.5 * *r]))
                        .collect(),
                )
            })
            .collect()
    })
}

/// Scale every plane by `s` (upstream `vk *= hyb`).
fn scale_mats(mats: &GradMatrices, s: f64) -> GradMatrices {
    std::array::from_fn(|x| {
        mats[x]
            .iter()
            .map(|m| {
                CTensor::from_planes(
                    m.re.iter().map(|v| s * v).collect(),
                    m.im.iter().map(|v| s * v).collect(),
                )
            })
            .collect()
    })
}

/// `dst += s * src`, elementwise, through [`oracle_sum`] (the RSH
/// `(alpha - hyb)` accumulation, `krks.py:63`).
fn add_scaled_mats_in_place(dst: &mut GradMatrices, src: &GradMatrices, s: f64) {
    for x in 0..3 {
        for (d, v) in dst[x].iter_mut().zip(src[x].iter()) {
            for (dr, vr) in d.re.iter_mut().zip(v.re.iter()) {
                *dr = oracle_sum(&[*dr, s * *vr]);
            }
            for (di, vi) in d.im.iter_mut().zip(v.im.iter()) {
                *di = oracle_sum(&[*di, s * *vi]);
            }
        }
    }
}

impl<'a> Gradients for KrksGradients<'a> {
    fn cell(&self) -> &Cell {
        self.mf.cell()
    }

    fn kpts(&self) -> &[[f64; 3]] {
        self.mf.kpts()
    }

    fn grad_elec(&self) -> Result<Gradient, PyscfRsError> {
        self.electronic_gradient()
    }

    fn get_hcore(&self) -> Result<GradMatrices, PyscfRsError> {
        crate::krhf::get_hcore(self.mf.cell(), self.mf.kpts())
    }

    fn hcore_generator(&self, atom: usize) -> Result<[KMats; 3], PyscfRsError> {
        let tables = precompute_hcore(self.mf.cell(), self.mf.kpts())?;
        hcore_deriv_matrices(&tables, self.mf.cell(), self.mf.kpts(), atom)
    }

    fn get_ovlp(&self) -> Result<GradMatrices, PyscfRsError> {
        self.overlap_deriv()
    }

    fn get_jk(&self, dm: &KDms) -> Result<(GradMatrices, GradMatrices), PyscfRsError> {
        self.jk_deriv(dm)
    }

    fn get_j(&self, dm: &KDms) -> Result<GradMatrices, PyscfRsError> {
        self.j_deriv(dm)
    }

    fn get_k(&self, dm: &KDms) -> Result<GradMatrices, PyscfRsError> {
        self.k_deriv(dm)
    }

    fn grad_nuc(&self) -> Result<Gradient, PyscfRsError> {
        self.nuclear_gradient()
    }

    fn make_rdm1e(&self) -> Result<KDms, PyscfRsError> {
        Ok(vec![self.dme0.clone()])
    }

    /// Upstream inherits `as_scanner` (`krhf.Gradients`, `krks.py` via the
    /// class body — the tests drive `mf.nuc_grad_method().as_scanner()`).
    /// [`crate::scanner`] is KRHF-typed until 18-19 lands a KS-typed
    /// shared-state scanner, so this keeps the trait's named refusal rather
    /// than wrapping the wrong mean field. Gate B builds its energy closure
    /// explicitly instead.
    fn as_scanner(&self) -> Result<EnergyScanner, PyscfRsError> {
        Err(PbcGradError::NotYetImplemented {
            phase: 18,
            what: "KRKS as_scanner: the shared-state scanner is KRHF-typed until 18-19 \
                   lands a KS-typed one",
        }
        .into())
    }
}
