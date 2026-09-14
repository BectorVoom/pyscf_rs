//! Multigrid **v2** pseudopotential gradient entry points —
//! `pyscf/pbc/dft/multigrid/pp.py`'s `get_vpploc_part1_ip1` (`:135-150`) and
//! `vpploc_part1_nuc_grad` (`:151-201`), as methods on
//! [`crate::multigrid::pair::MultiGridNumInt2`] (plan 18-09, Task 3).
//!
//! Both build on `_get_vpplocG_part1` (`pp.py:107-134`) — the part-1 local
//! term, which `crate::multigrid::pp`'s module doc identifies as the ONLY
//! genuinely multigrid-specific piece of the PP (everything else is the
//! AFTDF delegation). What ships here is that piece plus its two
//! derivatives, in G-space, at the gamma point.
//!
//! # The Gaussian-core model (read before comparing against `pp.py`)
//!
//! Upstream's `_get_vpplocG_part1` (with the default
//! `PP_WITH_RHO_CORE = True`) builds the core density in REAL space through
//! the `build_core_density` C kernel over a fake cell of s-type Gaussians
//! (`zeta = .5/rloc²`, unit-normalised, charge `-Z_ion`), then FFTs and
//! multiplies by `coulG`, and finally adds the `G = 0` term
//! `2π·Σ rloc²·Z`. [`MultiGridNumInt2::vpploc_g_part1`] evaluates the SAME
//! Gaussians directly in G-space — the Fourier transform of a normalised
//! s-Gaussian is analytic (`e^{-G²/4ζ}` times the structure factor) — which
//! is exact for the model and needs no C kernel. The two routes differ by
//! the fake cell's real-space `pgf_rcut` truncation (a `precision²`-level
//! effect), not by physics.
//!
//! Consequences, each stated so the gates can check it:
//!
//! * atoms WITHOUT a pseudopotential contribute nothing (upstream gives
//!   them a delta-like `alpha = 1e16` Gaussian of radius 0 — no grid
//!   support, no gradient);
//! * `vpploc_part1_nuc_grad`'s field is the CONJUGATED core-density
//!   derivative, `-i·Z·G·e^{+iG·A}·e^{-G²/4ζ}` — the conjugate because the
//!   clause-10 primitive contracts WITHOUT conjugation
//!   (`ReΣ F·D`), so the caller folds the conjugate into the field, exactly
//!   as upstream's `+i·e^{+iG·A}` point-charge spelling does for
//!   `get_nuc_nuc_grad`. (The unconjugated `dρ/dR` spelling differs in the
//!   imaginary part's sign and misses the frozen-density FD by `2·Im` —
//!   measured, not theorised.)
//!   In components: re `+Z·Gx·d·sinθ`, im `-Z·Gx·d·cosθ`;
//! * the sign is `+contract/vol`, i.e. `dE/dR` of the part-1 energy
//!   `ReΣ conj(rhoG_elec)·vpplocG_part1/vol`. Upstream's `grad *= -1`
//!   (`pp.py:199`) belongs to its backend's r-gradient convention
//!   (`int_gauss_charge_v_rs` integrates against `∇_r ρ`, and
//!   `dρ/dR = -∇_r ρ`); this port's field is `dρ/dR` (conjugated) directly,
//!   so no `-1`. The FD gate on the part-1 energy settles it either way.
//!
//! # Scope
//!
//! Gamma point only ([`check_gamma_kpts`](super::pair_grad::check_gamma_kpts)
//! — the `pp.py:137` / `:155` `KPoints` refusals); closed shell; no `cubecl`
//! (ALG-06 — orchestration here, kernels in `pyscf-kernels`).

use pyscf_algebra::CTensor;
use pyscf_pbc_gto::{Cell, get_coulg_at_gv, get_gv};

use crate::error::PbcDftError;
use crate::multigrid::pair::MultiGridNumInt2;
use crate::multigrid::pair_grad::{check_gamma_kpts, pass2_ip1};

/// `2π`, spelled out: the `G = 0` term of `_get_vpplocG_part1`
/// (`pp.py:132`) is `2·π·Σ rloc²·Z`.
const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

/// One atom's part-1 core model: `(_get_vpplocG_part1`'s fake-cell Gaussian
/// with `zeta = .5/rloc²`, charge `-Z_ion`).
struct CoreAtom {
    center: [f64; 3],
    /// `Σ nelec` — the core charge this Gaussian neutralises.
    z: f64,
    /// `.5 / rloc²`.
    zeta: f64,
}

/// The fake-cell core list: every pseudo atom with a usable `rloc`.
/// All-electron atoms are skipped (radius-0 delta cores — see module doc).
fn core_atoms(cell: &Cell) -> Vec<CoreAtom> {
    let coords = cell.mol.atom_coords();
    let mut out = Vec::new();
    for (ia, a) in coords.iter().enumerate() {
        let Some(ps) = cell.atom_pseudo(ia) else {
            continue;
        };
        if ps.rloc <= 0.0 {
            continue;
        }
        let z: f64 = ps.nelec.iter().map(|&n| f64::from(n)).sum();
        if z == 0.0 {
            continue;
        }
        out.push(CoreAtom {
            center: *a,
            z,
            zeta: 0.5 / (ps.rloc * ps.rloc),
        });
    }
    out
}

impl MultiGridNumInt2 {
    /// `_get_vpplocG_part1` (`pp.py:107-134`, `with_rho_core = True`) in
    /// G-space: `rhoG_core·coulG` plus the `G = 0` term (see the module doc
    /// for why G-space is exact for this model).
    ///
    /// # Errors
    /// Propagates the G-vector / Coulomb-kernel build.
    pub fn vpploc_g_part1(&self, cell: &Cell) -> Result<CTensor, PbcDftError> {
        let mesh = cell.mesh;
        let ngrids = mesh[0] * mesh[1] * mesh[2];
        let gv = get_gv(cell, Some(mesh))?;
        let coulg = get_coulg_at_gv(cell, mesh, &gv)?;
        let cores = core_atoms(cell);

        let mut re = vec![0.0f64; ngrids];
        let mut im = vec![0.0f64; ngrids];
        let mut g0 = 0.0f64;
        for core in &cores {
            g0 += TWO_PI * core.z / (2.0 * core.zeta);
            for (g, gv_g) in gv.iter().enumerate() {
                let g2 = gv_g[0] * gv_g[0] + gv_g[1] * gv_g[1] + gv_g[2] * gv_g[2];
                let damp = (-g2 / (4.0 * core.zeta)).exp();
                let theta =
                    gv_g[0] * core.center[0] + gv_g[1] * core.center[1] + gv_g[2] * core.center[2];
                let w = -core.z * damp;
                re[g] += w * theta.cos();
                im[g] += -w * theta.sin();
            }
        }
        for g in 0..ngrids {
            re[g] *= coulg[g];
            im[g] *= coulg[g];
        }
        // `pp.py:132` — the finite `G → 0` limit piece (`coulG[0] = 0`
        // killed the product there). `2π·Σ rloc²·Z`, and
        // `rloc² = 1/(2ζ)`.
        let mut g0idx = 0usize;
        let mut g0min = f64::INFINITY;
        for (g, gv_g) in gv.iter().enumerate() {
            let g2 = gv_g[0] * gv_g[0] + gv_g[1] * gv_g[1] + gv_g[2] * gv_g[2];
            if g2 < g0min {
                g0min = g2;
                g0idx = g;
            }
        }
        re[g0idx] += g0;
        Ok(CTensor::from_planes(re, im))
    }

    /// `get_vpploc_part1_ip1` (`pp.py:135-150`) at the gamma point: the
    /// part-1 local PP contracted through [`pass2_ip1`](super::pair_grad)
    /// to `(3, nao, nao)`.
    ///
    /// Upstream reads `mydf.vpplocG_part1` when populated and builds it
    /// otherwise; this port always builds (there is no `vpplocG_part1`
    /// cache field to go stale).
    ///
    /// # Errors
    /// Refuses non-gamma `kpts`; propagates the field build and `pass2_ip1`.
    pub fn get_vpploc_part1_ip1(
        &self,
        cell: &Cell,
        kpts: &[[f64; 3]],
    ) -> Result<Vec<f64>, PbcDftError> {
        check_gamma_kpts(kpts)?;
        let vg = self.vpploc_g_part1(cell)?;
        // Bare Fourier components — `ifft` is the physical potential
        // (see [`pass2_ip1`](super::pair_grad)'s weight note).
        let mesh = cell.mesh;
        let vr = pyscf_pbc_tools::ifft(&vg, mesh).map_err(crate::multigrid::numint::wrap_tools)?;
        pass2_ip1(cell, &vr.re)
    }

    /// `vpploc_part1_nuc_grad` (`pp.py:151-201`) at the gamma point: the
    /// part-1 force on the nuclei, one `(natm, 3)` array (zeros for atoms
    /// outside `atm_id` and for all-electron atoms).
    ///
    /// * `rho_g = Some(..)` reuses the density `get_veff_ip1` cached
    ///   (`mydf.rhoG`, `pp.py:161-164`); `None` recomputes it. Both
    ///   branches ship, and the test asserts they agree.
    /// * `atm_id = Some(..)` restricts the field to those atoms (upstream's
    ///   `fake_cell_vloc_part1(cell, atm_id, …)`); `None` covers all atoms.
    ///   Out-of-range ids are refused rather than clipped.
    ///
    /// # Errors
    /// Refuses non-gamma `kpts` and out-of-range `atm_id`; propagates the
    /// density build and the clause-10 reduction.
    #[allow(clippy::too_many_arguments)]
    pub fn vpploc_part1_nuc_grad(
        &self,
        cell: &Cell,
        dm: &[f64],
        kpts: &[[f64; 3]],
        rho_g: Option<&CTensor>,
        atm_id: Option<&[usize]>,
    ) -> Result<Vec<[f64; 3]>, PbcDftError> {
        check_gamma_kpts(kpts)?;
        let natm = cell.mol.atom_coords().len();
        if let Some(ids) = atm_id
            && let Some(&bad) = ids.iter().find(|&&ia| ia >= natm)
        {
            return Err(PbcDftError::Core(pyscf_core::PyscfRsError::Core(
                pyscf_core::CoreError::InvalidMolecule(format!(
                    "vpploc_part1_nuc_grad: atm_id {bad} out of range for {natm} atoms"
                )),
            )));
        }

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
            return Err(PbcDftError::Core(pyscf_core::PyscfRsError::Core(
                pyscf_core::CoreError::InvalidMolecule(
                    "vpploc_part1_nuc_grad: cached rhoG is not on cell.mesh".to_string(),
                ),
            )));
        }
        let gv = get_gv(cell, Some(mesh))?;
        let coulg = get_coulg_at_gv(cell, mesh, &gv)?;
        // The electronic potential once — the field carries the per-atom
        // core derivative (cf. `get_nuc_nuc_grad`, where the field carries
        // `coulG` instead; same primitive, dual association).
        let mut den_re = vec![0.0f64; ngrids];
        let mut den_im = vec![0.0f64; ngrids];
        for g in 0..ngrids {
            den_re[g] = coulg[g] * rho.re[g];
            den_im[g] = coulg[g] * rho.im[g];
        }

        let coords = cell.mol.atom_coords();
        let selected = |ia: usize| atm_id.is_none_or(|ids| ids.contains(&ia));
        // `conj(d(rhoG_core)/dR_iax) = -i·Z·Gx·e^{+iθ}·damp`, i.e.
        // re `+Z·Gx·d·sinθ`, im `-Z·Gx·d·cosθ` (module doc: the conjugate
        // is folded into the field because the primitive does not
        // conjugate).
        let mut field_re = vec![0.0f64; natm * 3 * ngrids];
        let mut field_im = vec![0.0f64; natm * 3 * ngrids];
        for (ia, a) in coords.iter().enumerate() {
            if !selected(ia) {
                continue;
            }
            let Some(ps) = cell.atom_pseudo(ia) else {
                continue;
            };
            if ps.rloc <= 0.0 {
                continue;
            }
            let z: f64 = ps.nelec.iter().map(|&n| f64::from(n)).sum();
            if z == 0.0 {
                continue;
            }
            let zeta = 0.5 / (ps.rloc * ps.rloc);
            for (g, gv_g) in gv.iter().enumerate() {
                let g2 = gv_g[0] * gv_g[0] + gv_g[1] * gv_g[1] + gv_g[2] * gv_g[2];
                let d = (-g2 / (4.0 * zeta)).exp();
                let theta = gv_g[0] * a[0] + gv_g[1] * a[1] + gv_g[2] * a[2];
                let (s, c) = theta.sin_cos();
                for x in 0..3 {
                    let row = (ia * 3 + x) * ngrids + g;
                    field_re[row] = z * gv_g[x] * d * s;
                    field_im[row] = -z * gv_g[x] * d * c;
                }
            }
        }
        let client = pyscf_algebra::select_backend()
            .map(|s| s.client)
            .map_err(|e| {
                PbcDftError::Core(pyscf_core::PyscfRsError::Core(
                    pyscf_core::CoreError::InvalidMolecule(format!(
                        "multigrid v2 gradient: backend selection failed: {e}"
                    )),
                ))
            })?;
        let grad = pyscf_kernels::pbc::multigrid_grad::contract_atom_grid(
            &client, natm, &field_re, &field_im, &den_re, &den_im,
        )
        .map_err(|e| {
            PbcDftError::Core(pyscf_core::PyscfRsError::Core(
                pyscf_core::CoreError::InvalidMolecule(format!("multigrid v2 gradient: {e}")),
            ))
        })?;
        let vol = cell.vol();
        Ok(grad
            .iter()
            .map(|row| [row[0] / vol, row[1] / vol, row[2] / vol])
            .collect())
    }
}
