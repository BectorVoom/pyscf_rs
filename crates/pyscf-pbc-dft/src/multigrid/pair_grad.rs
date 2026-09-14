//! Multigrid **v2** gradient entry points —
//! `pyscf/pbc/dft/multigrid/multigrid_pair.py`'s `get_veff_ip1` (`:748-860`),
//! `get_nuc` with `deriv` (`:861-892`), `get_nuc_ip1` (`:893-895`) and
//! `get_nuc_nuc_grad` (`:896-921`), as methods on
//! [`crate::multigrid::pair::MultiGridNumInt2`] (plan 18-09, Tasks 1-2).
//!
//! # Scope: gamma point, closed shell, LDA/HF/GGA
//!
//! * **Gamma only.** Every method takes `kpts: &[[f64; 3]]` and refuses
//!   anything but the gamma point with
//!   [`PbcDftError::MultiGridRequiresGamma`](crate::error::PbcDftError).
//!   Rust has no `KPoints` type (17-09's deferral), so upstream's
//!   `isinstance(kpts, KPoints): raise` (`multigrid_pair.py:749`, `:902`)
//!   surfaces here as a refusal of any non-gamma k-list, at the same call
//!   sites. Honoured, not worked around.
//! * **LDA/HF/GGA only.** Anything above GGA raises, exactly as
//!   `multigrid_pair.py:766`: a meta-GGA errors out of
//!   [`XcType::of`](crate::xc::XcType) (the periodic AO evaluator produces
//!   value + deriv1 only), and an explicit `deriv > 1` is refused with
//!   [`PbcDftError::MultiGridDerivUnsupported`](crate::error::PbcDftError).
//! * **`vpplocG_part1` is absent.** Upstream adds `mydf.vpplocG_part1` to
//!   `vG` when present (`:783-785`). This port never populates that field:
//!   the pseudopotential enters through the AFTDF delegation
//!   (`crate::multigrid::pp`), so the add is a documented no-op, not a
//!   silent skip — there is no cache whose staleness could leak into the
//!   gradient.
//! * **The `rhoG` cache is a returned handle.** Upstream stashes
//!   `mydf.rhoG` inside `get_veff_ip1` (`:786-787`) for
//!   `vpploc_part1_nuc_grad` to read. [`Mg2VeffIp1Result::rho_g`] returns it
//!   instead; [`MultiGridNumInt2::vpploc_part1_nuc_grad`] takes it as
//!   `Option<&CTensor>`. An explicit argument, never a side channel.
//!
//! # Where the arithmetic lives (ALG-06)
//!
//! This crate names no `cubecl`. G-space orchestration (structure factors,
//! Coulomb products, FFT windows) is host code here; the reverse
//! ("pass2") collocation reuses the already-shipped derivative AO tables
//! ([`pyscf_pbc_gto::eval_gto::eval_ao_kpts`] with `"GTOval_sph_deriv1"`,
//! whose kernels live in `pyscf-kernels`), and the `(natm, 3)` G-space
//! reduction routes through the single clause-10 primitive
//! [`pyscf_kernels::pbc::multigrid_grad::contract_atom_grid`] (D-PBC-31
//! clause 10 — one implementation, shared with 18-05's fused `hcore`).
//!
//! # The `pass2_ip1` route note (read before "optimising" it)
//!
//! Upstream's `_get_j_pass2_ip1` contracts through `eval_mat` — multigrid
//! pair collocation with derivative AOs. [`pass2_ip1`] contracts the same
//! real-space potential through the exact derivative AO table instead. The
//! two differ by the multigrid collocation error of the reverse direction
//! (the 18-01 floor), so `get_veff_ip1` against a finite difference of
//! `nr_rks` gates at the floor, not at machine precision — while
//! `get_nuc_nuc_grad`, which is pure G-space analytic, gates at machine
//! precision. Rebuilding a derivative pair table to close that gap would
//! duplicate `build_pair_level_table`'s image/wrap/block geometry for a
//! second table that must then agree with the first; the FD gate measures
//! the gap instead of assuming it away.

use pyscf_algebra::{AlgebraError, CTensor, oracle_sum};
use pyscf_core::{CoreError, PyscfRsError};
use pyscf_pbc_gto::{Cell, get_coulg_at_gv, get_gv};

use crate::error::PbcDftError;
use crate::multigrid::numint::{mg_xc_parts, wrap_tools};
use crate::multigrid::pair::MultiGridNumInt2;
use crate::xc::XcType;

fn wrap_alg(e: AlgebraError) -> PbcDftError {
    PbcDftError::Core(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
        "multigrid v2 gradient: {e}"
    ))))
}

fn backend_client() -> Result<pyscf_algebra::AlgebraClient, PbcDftError> {
    pyscf_algebra::select_backend()
        .map(|s| s.client)
        .map_err(|e| {
            PbcDftError::Core(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "multigrid v2 gradient: backend selection failed: {e}"
            ))))
        })
}

fn bad_shape(what: &str) -> PbcDftError {
    PbcDftError::Core(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
        "multigrid v2 gradient: {what}"
    ))))
}

/// Gamma-point enforcement — the `KPoints` refusal at every new entry point.
///
/// Rust has no `KPoints` type (17-09's deferral), so upstream's
/// `isinstance(kpts, KPoints): raise` (`multigrid_pair.py:749`, `:902`;
/// `pp.py:137`, `:155`) is honoured as: an empty list (the gamma
/// convention `crate::multigrid::pp` already uses) or an all-gamma list
/// passes; anything else is refused with `MultiGridRequiresGamma`.
pub fn check_gamma_kpts(kpts: &[[f64; 3]]) -> Result<(), PbcDftError> {
    let bad = kpts
        .iter()
        .filter(|k| k[0] * k[0] + k[1] * k[1] + k[2] * k[2] >= 1e-18)
        .count();
    if bad > 0 {
        return Err(PbcDftError::MultiGridRequiresGamma { nkpts: kpts.len() });
    }
    Ok(())
}

/// [`MultiGridNumInt2::get_veff_ip1`]'s return.
///
/// `veff_ip1` is `(3, nao, nao)` row-major, `x` slowest
/// (`[((x * nao) + mu) * nao + nu]`), in the RAW upstream sign convention —
/// callers negate (`pbc/grad/rhf.py`'s `get_veff` returns
/// `-mf._numint.get_veff_ip1(...)`). `rho_g` is the `mydf.rhoG` of
/// `multigrid_pair.py:786-787`: the cached density handle, returned and
/// never stashed (see the module doc).
#[derive(Debug, Clone)]
pub struct Mg2VeffIp1Result {
    pub veff_ip1: Vec<f64>,
    pub nao: usize,
    pub rho_g: CTensor,
}

/// The reverse ("pass2") collocation at `deriv = 1` — the gamma-point
/// analogue of `_get_j_pass2_ip1` (`multigrid_pair.py:429-469`).
///
/// `v_true` is the PHYSICAL real-space potential on `cell.mesh`'s uniform
/// grid (no `weight` factor — callers unweight their G-space field first;
/// see below). Returns `(3, nao, nao)` row-major, `x` slowest, holding the
/// BRA-centre nuclear derivative only:
/// `out[x,mu,nu] = -w·Σ_g dx[g,mu]·x[g,nu]·v[g]` with `dx` the
/// electron-coordinate derivative row of `"GTOval_sph_deriv1"` and
/// `w = vol/ngrids`. The ket-side term is NOT included: the gradient
/// assembly (`_contract_vhf_dm`, bra-sliced, times 2) accounts for it by
/// symmetry, and the FD gate pins that convention.
///
/// # The weight lives here exactly once (read before touching it)
///
/// G-space fields in this codebase carry inconsistent normalisation by
/// construction: `mg_xc_parts`' rows are `fft(weight·vxc)` (WEIGHTED —
/// `ifft` gives `weight·v_true`), while analytic fields built here
/// (`nuclear_vg`, `vpploc_g_part1`) are bare Fourier components
/// (`ifft` gives `v_true` directly). `pass2_ip1` therefore takes the
/// physical potential and applies `vol/ngrids` itself — one place, one
/// factor, gated elementwise against `get_nuc`.
///
/// Fixed-order accumulation (x, mu, nu outer; grid innermost through
/// `oracle_sum`) — independent of any rayon thread count (D-PBC-17).
///
/// # Errors
/// Propagates the uniform-grid build and the AO evaluation; rejects a
/// `v_true` that does not sit on `cell.mesh`.
pub(crate) fn pass2_ip1(cell: &Cell, v_true: &[f64]) -> Result<Vec<f64>, PbcDftError> {
    let mesh = cell.mesh;
    let ngrids = mesh[0] * mesh[1] * mesh[2];
    if v_true.len() != ngrids {
        return Err(bad_shape("pass2_ip1: potential is not on cell.mesh"));
    }
    let weight = cell.vol() / ngrids as f64;

    let grids = crate::gen_grid::PeriodicGrids::uniform(cell, Some(mesh))?;
    let coords = grids.coords()?;
    let ao = pyscf_pbc_gto::eval_gto::eval_ao_kpts(cell, "GTOval_sph_deriv1", coords, &[])
        .map_err(PbcDftError::Core)?;
    if ao.nkpts() != 1 || ao.comp != 4 || ao.ngrids != ngrids {
        return Err(bad_shape(
            "pass2_ip1: deriv1 AO table has the wrong k-point/component/grid shape",
        ));
    }
    let nao = cell.mol.nao_nr;
    if ao.nao != nao {
        return Err(bad_shape(
            "pass2_ip1: deriv1 AO table has the wrong AO count",
        ));
    }
    let blk = &ao.kaos[0].re;
    // F-order per component: `p = c*ngrids*nao + g + mu*ngrids`
    // (`EvalAoKptsOutput::element`). Component 0 is the value, 1..3 the
    // electron-coordinate derivatives.
    let at = |c: usize, g: usize, mu: usize| blk[c * ngrids * nao + g + mu * ngrids];

    let mut out = vec![0.0f64; 3 * nao * nao];
    for x in 0..3 {
        for mu in 0..nao {
            for nu in 0..nao {
                let terms: Vec<f64> = (0..ngrids)
                    .map(|g| at(1 + x, g, mu) * at(0, g, nu) * v_true[g])
                    .collect();
                out[(x * nao + mu) * nao + nu] = -weight * oracle_sum(&terms);
            }
        }
    }
    Ok(out)
}

impl MultiGridNumInt2 {
    /// `_eval_rhoG` at `deriv = 0` (LDA/HF) or `1` (GGA);
    /// `deriv > 1` raises (`multigrid_pair.py:766`).
    ///
    /// Both accepted orders return the row-0 density: upstream's default
    /// `RHOG_HIGH_ORDER = False` route rebuilds `grad rho` from G-space
    /// downstream (`mg_xc_parts`), so no `rhodim = 4` table is needed here.
    /// A meta-GGA never reaches this branch — it errors out of
    /// [`XcType::of`] first, which IS the `:766` refusal.
    ///
    /// # Errors
    /// `MultiGridDerivUnsupported` for `deriv > 1`; propagates `eval_rho_g`.
    pub fn rho_g_with_deriv(
        &self,
        cell: &Cell,
        dm: &[f64],
        deriv: u32,
    ) -> Result<CTensor, PbcDftError> {
        if deriv > 1 {
            return Err(PbcDftError::MultiGridDerivUnsupported { deriv });
        }
        self.eval_rho_g(cell, dm)
    }

    /// `get_veff_ip1` (`multigrid_pair.py:748-860`) at the gamma point,
    /// closed shell: `_eval_rhoG` at `deriv = 0/1` → multiply by `coulG`
    /// (folded with XC inside `mg_xc_parts`, the `with_j` branch) →
    /// inverse-FFT to `rhoR` with the `1/weight` rescale handled inside
    /// `mg_xc_parts` → evaluate XC → collocate back through [`pass2_ip1`].
    ///
    /// `xc_code` is LDA (`"lda"`, `"pbe"`, …), `"HF"`/`"NONE"`/`""`
    /// (Coulomb only — upstream's `eval_xc_eff('HF', …)` is zero), or GGA.
    /// Meta-GGA raises via [`XcType::of`], non-gamma `kpts` raises via
    /// [`check_gamma_kpts`].
    ///
    /// # Errors
    /// Propagates the density build, the XC evaluation and [`pass2_ip1`].
    pub fn get_veff_ip1(
        &self,
        cell: &Cell,
        xc_code: &str,
        dm: &[f64],
        kpts: &[[f64; 3]],
    ) -> Result<Mg2VeffIp1Result, PbcDftError> {
        check_gamma_kpts(kpts)?;
        let code = xc_code.trim().to_ascii_uppercase();
        let hf = code == "HF" || code == "NONE" || code.is_empty();
        let deriv = if hf {
            0
        } else {
            match XcType::of(xc_code)? {
                XcType::Lda => 0,
                XcType::Gga => 1,
            }
        };
        let rho_g = self.rho_g_with_deriv(cell, dm, deriv)?;
        // `mydf.vpplocG_part1` (`:783-785`): never populated in this port
        // (the PP enters through the AFTDF delegation) — documented no-op.
        let wv = if hf {
            let mesh = cell.mesh;
            let gv = get_gv(cell, Some(mesh))?;
            let coulg = get_coulg_at_gv(cell, mesh, &gv)?;
            let mut vg = rho_g.clone();
            for g in 0..vg.re.len() {
                vg.re[g] *= coulg[g];
                vg.im[g] *= coulg[g];
            }
            vg
        } else {
            mg_xc_parts(cell, xc_code, std::slice::from_ref(&rho_g))?
                .wv_freq0
                .swap_remove(0)
        };
        let nao = cell.mol.nao_nr;
        // Unweight to the physical potential (see [`pass2_ip1`]'s weight
        // note): BOTH branches carry one `weight` factor — `mg_xc_parts`
        // rows by construction, and the HF branch's `vg = rhoG·coulG`
        // because the pair-route `rhoG` already carries it
        // (`insert_level_rho_g`).
        let mesh = cell.mesh;
        let ngrids = mesh[0] * mesh[1] * mesh[2];
        let vr = pyscf_pbc_tools::ifft(&wv, mesh).map_err(wrap_tools)?;
        let weight = cell.vol() / ngrids as f64;
        let v_true: Vec<f64> = vr.re.iter().map(|v| v / weight).collect();
        let veff_ip1 = pass2_ip1(cell, &v_true)?;
        debug_assert_eq!(veff_ip1.len(), 3 * nao * nao);
        Ok(Mg2VeffIp1Result {
            veff_ip1,
            nao,
            rho_g,
        })
    }

    /// `vneG = rhoG_nuc · coulG` — the nuclear-attraction G-space field that
    /// `get_nuc(deriv=1)` contracts (`multigrid_pair.py:861-892`).
    ///
    /// `rhoG_nuc` is the batched point-charge structure factor,
    /// `Σ_ia q_ia·e^{-iG·A_ia}` with `q_ia = -Z_ia` (upstream's
    /// `charge = -cell.atom_charges()`), computed once for all atoms — the
    /// `get_SI`-shaped `(natm, ngrids)` batch, in host code (no reduction
    /// involved, so no device round-trip is bought).
    fn nuclear_vg(&self, cell: &Cell) -> Result<CTensor, PbcDftError> {
        let mesh = cell.mesh;
        let ngrids = mesh[0] * mesh[1] * mesh[2];
        let gv = get_gv(cell, Some(mesh))?;
        let coulg = get_coulg_at_gv(cell, mesh, &gv)?;
        let coords = cell.mol.atom_coords();
        let charges = cell.atom_charges();
        let mut re = vec![0.0f64; ngrids];
        let mut im = vec![0.0f64; ngrids];
        for (a, q) in coords.iter().zip(charges.iter()) {
            let qi = -f64::from(*q);
            if qi == 0.0 {
                continue;
            }
            for (g, gv_g) in gv.iter().enumerate() {
                let theta = gv_g[0] * a[0] + gv_g[1] * a[1] + gv_g[2] * a[2];
                re[g] += qi * theta.cos();
                im[g] += -qi * theta.sin();
            }
        }
        for g in 0..ngrids {
            re[g] *= coulg[g];
            im[g] *= coulg[g];
        }
        Ok(CTensor::from_planes(re, im))
    }

    /// `get_nuc(mydf, kpts, deriv)` — `deriv = 0` is the existing nuclear
    /// matrix (same AFTDF delegation [`MultiGridNumInt2::get_nuc`] uses);
    /// `deriv = 1` is the ip1 contraction (`multigrid_pair.py:884-885`).
    ///
    /// # Errors
    /// `MultiGridDerivUnsupported` for `deriv > 1`; propagates the field
    /// build and [`pass2_ip1`].
    pub fn get_nuc_with_deriv(
        &self,
        cell: &Cell,
        kpts: &[[f64; 3]],
        deriv: u32,
    ) -> Result<Vec<f64>, PbcDftError> {
        check_gamma_kpts(kpts)?;
        match deriv {
            0 => crate::multigrid::pp::get_nuc(cell),
            1 => {
                let vneg = self.nuclear_vg(cell)?;
                // Bare Fourier components — `ifft` is the physical
                // potential, no unweighting (see [`pass2_ip1`]).
                let mesh = cell.mesh;
                let vr = pyscf_pbc_tools::ifft(&vneg, mesh).map_err(wrap_tools)?;
                pass2_ip1(cell, &vr.re)
            }
            _ => Err(PbcDftError::MultiGridDerivUnsupported { deriv }),
        }
    }

    /// `get_nuc_ip1` (`multigrid_pair.py:893-895`): `get_nuc` at `deriv = 1`.
    ///
    /// # Errors
    /// As [`MultiGridNumInt2::get_nuc_with_deriv`].
    pub fn get_nuc_ip1(&self, cell: &Cell, kpts: &[[f64; 3]]) -> Result<Vec<f64>, PbcDftError> {
        self.get_nuc_with_deriv(cell, kpts, 1)
    }

    /// `get_nuc_nuc_grad` (`multigrid_pair.py:896-921`) at the gamma point:
    /// `<p|d/dR(-Z/|r-R|)|q>·D`, one `(natm, 3)` array.
    ///
    /// Upstream carries a `# TODO improve performance` on its per-atom loop
    /// (`:916-919`): each iteration allocates a fresh `(3, ngrids)` complex
    /// array and re-evaluates the structure factor. This port batches it
    /// instead — one `(natm·3, ngrids)` field, one call into the shared
    /// clause-10 primitive
    /// [`contract_atom_grid`](pyscf_kernels::pbc::multigrid_grad::contract_atom_grid),
    /// one `1/vol` rescale. The field is spelled EXACTLY as upstream's loop
    /// body (`vG = 1j·e^{+iG·A}·q·coulG·G`), so the batched-vs-literal gate
    /// is bit-identical rather than epsilon-close:
    /// `field[ia,x,g] = (-q·c·Gx·sinθ, +q·c·Gx·cosθ)` with
    /// `θ = G·A_ia`, contracted against the bare electronic `rhoG`.
    ///
    /// `rho_g` models the hoisted density explicitly: `None` recomputes via
    /// `eval_rho_g` (what upstream does — it does NOT read `mydf.rhoG`
    /// here), `Some` reuses a caller-held handle (e.g. from
    /// [`Mg2VeffIp1Result::rho_g`).
    ///
    /// # Errors
    /// Propagates the density build and the clause-10 reduction.
    pub fn get_nuc_nuc_grad(
        &self,
        cell: &Cell,
        dm: &[f64],
        kpts: &[[f64; 3]],
        rho_g: Option<&CTensor>,
    ) -> Result<Vec<[f64; 3]>, PbcDftError> {
        check_gamma_kpts(kpts)?;
        let owned;
        let rho: &CTensor = match rho_g {
            Some(r) => r,
            None => {
                owned = self.eval_rho_g(cell, dm)?;
                &owned
            }
        };
        let mesh = cell.mesh;
        let ngrids = mesh[0] * mesh[1] * mesh[2];
        if rho.re.len() != ngrids || rho.im.len() != ngrids {
            return Err(bad_shape(
                "get_nuc_nuc_grad: cached rhoG is not on cell.mesh",
            ));
        }
        let gv = get_gv(cell, Some(mesh))?;
        let coulg = get_coulg_at_gv(cell, mesh, &gv)?;
        let coords = cell.mol.atom_coords();
        let charges = cell.atom_charges();
        let natm = coords.len();

        // The batched loop: upstream's per-iteration `vG` for every atom at
        // once, in fixed atom-major order (row `ia*3+x`).
        let mut field_re = vec![0.0f64; natm * 3 * ngrids];
        let mut field_im = vec![0.0f64; natm * 3 * ngrids];
        for (ia, (a, q)) in coords.iter().zip(charges.iter()).enumerate() {
            let qi = -f64::from(*q);
            for (g, gv_g) in gv.iter().enumerate() {
                let theta = gv_g[0] * a[0] + gv_g[1] * a[1] + gv_g[2] * a[2];
                let (s, c) = theta.sin_cos();
                let cr = coulg[g] * qi;
                for x in 0..3 {
                    let row = (ia * 3 + x) * ngrids + g;
                    field_re[row] = -cr * gv_g[x] * s;
                    field_im[row] = cr * gv_g[x] * c;
                }
            }
        }
        let client = backend_client()?;
        let grad = pyscf_kernels::pbc::multigrid_grad::contract_atom_grid(
            &client, natm, &field_re, &field_im, &rho.re, &rho.im,
        )
        .map_err(wrap_alg)?;
        let vol = cell.vol();
        Ok(grad
            .iter()
            .map(|row| [row[0] / vol, row[1] / vol, row[2] / vol])
            .collect())
    }
}
