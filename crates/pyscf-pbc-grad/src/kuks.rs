//! KUKS k-point analytic nuclear gradient — `pyscf/pbc/grad/kuks.py` (135 l).
//!
//! The k-point unrestricted Kohn–Sham nuclear gradient. Upstream declares it
//! as `class Gradients(uhf_grad.Gradients)` (`kuks.py:122`): it extends 18-06's
//! **base**, so the assembly is 18-06's verbatim with the spin index threaded
//! through three of the five `grad_elec` terms, and ONLY `get_veff` is
//! replaced — the XC-grid term ([`get_vxc_sets`](crate::krks::get_vxc_sets),
//! shared with KRKS per Task 2) plus the spin-threaded Coulomb /
//! hybrid-exchange terms ([`KuksGradients::veff`]).
//!
//! # The three refusals (Task 1 — read before "fixing")
//!
//! * **`grid_response`.** `kuks.py:49-50` is `if ks_grad.grid_response:
//!   raise NotImplementedError`, and `kuks.py:129` hard-sets
//!   `self.grid_response = False` in `__init__`. [`KuksGradients::veff`]
//!   **overrides the molecular "fully supported" term to a refusal**; the
//!   constructor default is `false`, matching the hard-set.
//! * **meta-GGA** (`kuks.py:118`) and **NLC** (`kuks.py:116`) refuse by name
//!   inside the shared [`get_vxc_sets`](crate::krks::get_vxc_sets) core.
//!
//! # Stress asymmetry (do NOT reconcile)
//!
//! Gradient = LDA/GGA/HF/hybrid; stress = LDA/GGA/MGGA — upstream's shape
//! (`kuks_stress.py` supports MGGA), not an oversight. See `krks.rs`.
//!
//! # Upstream correspondence (`kuks.py` line → here)
//!
//! | upstream | here |
//! |---|```
//! | `get_veff` `:32-67` | [`KuksGradients::veff`] |
//! | `get_vxc` `:70-120` (LDA `:77-93`, GGA `:98-113`, HF `:95-96`) | shared core, `krks.rs` |
//! | `NLC` `:115-116`, `metaGGA` `:117-118` | shared core, `krks.rs` |
//! | `grid_response` refusal `:49-50` + hard-set `:129` | [`KuksGradients::veff`] + default `false` |
//! | `return -vmat` (no squeeze) `:120` | shared core (one family per set) |
//! | `vj[:,0][:,None] + vj[:,1][:,None] - vk` `:58/:66` | [`KuksGradients::veff`] (broadcast J, no `.5`) |
//! | RSH branch `:60-65` | [`KuksGradients::veff`] via 18-04's `get_k_e1` `omega` |
//!
//! Note the `.5` asymmetry with KRKS (`krks.py:64` has `vj - vk * .5`): KUKS
//! subtracts the FULL `vk[s]` per spin (`:66`), because its two channels are
//! not spin-degenerate — the same factor 18-06 documents for KUHF.
//!
//! # Which terms are spin-summed and which are not (`kuks.py` via `kuhf.py`)
//!
//! As in 18-06: the `h1ao` term contracts the spin-SUMMED density, the `vhf`
//! term contracts SPIN-RESOLVED (the spin sum living inside the einsum), and
//! the `s1` term contracts the summed energy-weighted density.
//!
//! # Reductions and kernels (ALG-06, D-PBC-17, D-PBC-31)
//!
//! Every reduction routes through `pyscf_algebra::oracle_sum` over a
//! materialised partial buffer — never a bare `+=`.
//!
//! No CubeCL kernel is added here (ALG-06: `pyscf-pbc-grad` may not depend on
//! `cubecl-*` AT ALL; `xtask check-dependency-wall` enforces it). The CubeCL
//! manual (`manual/Cubecl/INDEX.md`, generics-`Float` kernels) was read
//! before writing; there is no device kernel in this file for it to apply
//! to. Device computation consumed is 18-05's clause-10 shared primitive
//! (inside [`fused_local_contraction`](crate::krhf::fused_local_contraction))
//! and the deriv-2 collocation kernel (inside the shared grid core).

use pyscf_algebra::{CTensor, oracle_sum};
use pyscf_core::{CoreError, PyscfRsError};
use pyscf_pbc_df::JkOpts;
use pyscf_pbc_gto::{Cell, ExxDiv};
use pyscf_pbc_scf::types::{KDms, KMats};
use pyscf_pbc_scf::krdm::make_rdm1;
use pyscf_pbc_dft::kuks::Kuks;
use pyscf_pbc_dft::xc::{is_hybrid_xc, rsh_and_hybrid_coeff};

use crate::error::PbcGradError;
use crate::gradients::{EnergyScanner, GradMatrices, Gradient, Gradients};
use crate::krhf::{
    contractions::{contract_h1_atom, contract_ovlp_atom},
    fused_local_contraction, hcore_deriv_matrices, make_rdm1e_kpts, precompute_hcore,
    HcoreFusedStats,
};
use crate::krks::{get_vxc_sets, take_grad_mats_sets};
use crate::kuhf::{contract_vhf_atom_spin, sum_sets};

fn invalid(message: impl Into<String>) -> PyscfRsError {
    PyscfRsError::Core(CoreError::InvalidMolecule(message.into()))
}

fn lift_dft<T>(r: Result<T, pyscf_pbc_dft::PbcDftError>) -> Result<T, PyscfRsError> {
    r.map_err(|e| match e {
        pyscf_pbc_dft::PbcDftError::Core(inner) => inner,
        other => invalid(format!("KUKS gradient: DFT layer failed: {other}")),
    })
}

fn df_err(context: &'static str, e: pyscf_pbc_df::PbcDfError) -> PyscfRsError {
    match e {
        pyscf_pbc_df::PbcDfError::Core(c) => c,
        other => PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "KUKS gradient ({context}): density fitting failed: {other}"
        ))),
    }
}

/// Number of k-points, treating an empty list as the single gamma point
/// (the `pbc_intor` convention; same helper as 18-05/18-06).
fn nkpts_of(kpts: &[[f64; 3]]) -> usize {
    if kpts.is_empty() { 1 } else { kpts.len() }
}

// ---------------------------------------------------------------------------
// `KuksGradients` — `class Gradients(uhf_grad.Gradients)` (kuks.py:122).
// ---------------------------------------------------------------------------

/// KUKS k-point nuclear gradient — `pbc/grad/kuks.py`'s `Gradients` class.
///
/// Borrowed mean field ([`Kuks`], owning `with_df`, `cell`, `kpts`, `xc`,
/// `grids`, `exxdiv`) plus caller-held SCF products: the per-spin densities
/// `dm0` and energy-weighted densities `dme0` are built by
/// [`KuksGradients::new`] from the converged orbitals — `dm0` through the SCF
/// `make_rdm1` split into alpha/beta halves, `dme0` through 18-05's
/// restricted [`make_rdm1e_kpts`] applied to each spin's orbitals (the
/// `kuhf.py:80-84` pattern, inherited unchanged). No SCF runs here.
///
/// `grid_response` defaults OFF (matching `kuks.py:129`'s hard-set) and is a
/// REFUSAL when set (`kuks.py:49-50`). `extra_force` is the shared zero
/// default.
pub struct KuksGradients<'a> {
    mf: &'a Kuks,
    dm0: KDms,
    dme0: KDms,
    atmlst: Option<Vec<usize>>,
    grid_response: bool,
}

impl<'a> KuksGradients<'a> {
    /// Build from converged orbitals. Blocks are in `idx(set, k)` order —
    /// alpha's `nkpts` blocks then beta's: `mo_coeff` column-major
    /// `nao × nmo` per block, `mo_energy`/`mo_occ` per orbital per block.
    ///
    /// # Errors
    /// [`CoreError::InvalidMolecule`] on k-point/shape disagreement or
    /// non-finite input.
    pub fn new(
        mf: &'a Kuks,
        mo_energy: Vec<Vec<f64>>,
        mo_coeff: Vec<CTensor>,
        mo_occ: Vec<Vec<f64>>,
    ) -> Result<Self, PyscfRsError> {
        let nkpts = nkpts_of(mf.kpts());
        let nao = mf.cell().mol.nao_nr;
        if mo_coeff.len() != 2 * nkpts || mo_energy.len() != 2 * nkpts || mo_occ.len() != 2 * nkpts
        {
            return Err(invalid(format!(
                "KUKS gradient: {nkpts} k-points need {}/{}/{} coeff/energy/occ blocks (alpha, beta), got {}/{}/{}",
                2 * nkpts,
                2 * nkpts,
                2 * nkpts,
                mo_coeff.len(),
                mo_energy.len(),
                mo_occ.len(),
            )));
        }
        let dm0 = vec![
            make_rdm1(&mo_coeff[..nkpts], &mo_occ[..nkpts], nao),
            make_rdm1(&mo_coeff[nkpts..], &mo_occ[nkpts..], nao),
        ];
        for (s, set) in dm0.iter().enumerate() {
            if set.len() != nkpts
                || set.iter().any(|m| {
                    m.re.len() != nao * nao
                        || m.im.len() != nao * nao
                        || m.re.iter().chain(&m.im).any(|v| !v.is_finite())
                })
            {
                return Err(invalid(format!(
                    "KUKS gradient: spin set {s} density is misshapen or non-finite"
                )));
            }
        }
        let dme0 = vec![
            make_rdm1e_kpts(&mo_coeff[..nkpts], &mo_energy[..nkpts], &mo_occ[..nkpts], nao)?,
            make_rdm1e_kpts(&mo_coeff[nkpts..], &mo_energy[nkpts..], &mo_occ[nkpts..], nao)?,
        ];
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
                    "KUKS gradient: atmlst id {ia} out of range for {} atoms",
                    self.mf.cell().natm
                )));
            }
        }
        self.atmlst = Some(atmlst);
        Ok(self)
    }

    /// Set the `grid_response` flag (default OFF, matching `kuks.py:129`'s
    /// hard-set). Setting it to `true` is NOT support for the Becke-weight
    /// term — [`KuksGradients::veff`] refuses it by name (`kuks.py:49-50`).
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

    /// `get_ovlp(cell, kpts)` — inherited unchanged from 18-05's base via
    /// 18-06: `-cell.pbc_intor('int1e_ipovlp', kpts)`, `[x][k]` row-major.
    pub fn overlap_deriv(&self) -> Result<GradMatrices, PyscfRsError> {
        let cell = self.mf.cell();
        let kpts = self.mf.kpts();
        let nao = cell.mol.nao_nr;
        let out = pyscf_pbc_gto::pbc_intor(cell, "int1e_ipovlp", kpts, Default::default())?;
        if out.comp != 3 || out.ni != nao || out.nj != nao {
            return Err(invalid(format!(
                "KUKS get_ovlp: int1e_ipovlp has comp = {}, ni/nj = {}/{} for nao = {nao}",
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

    /// `get_jk(dm, kpts)` — the inherited 18-04 route, nset-generic: the
    /// per-set pairs `vj[s]`, `vk[s]`, each `[x][k]` row-major.
    /// `kk_symmetry: false` explicitly (D-PBC-30 clause 4b, exactly as 18-05).
    pub fn jk_deriv(
        &self,
        dm: &KDms,
    ) -> Result<(Vec<GradMatrices>, Vec<GradMatrices>), PyscfRsError> {
        let mf = self.mf;
        let nkpts = nkpts_of(mf.kpts());
        let nset = dm.len();
        let nao = mf.cell().mol.nao_nr;
        if nset == 0 {
            return Err(invalid("KUKS get_jk: need at least one spin set"));
        }
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
        let vj = take_grad_mats_sets(
            res.vj.ok_or_else(|| invalid("KUKS get_jk: missing vj"))?,
            "vj",
            nset,
            nkpts,
            nao,
        )?;
        let vk = take_grad_mats_sets(
            res.vk.ok_or_else(|| invalid("KUKS get_jk: missing vk"))?,
            "vk",
            nset,
            nkpts,
            nao,
        )?;
        Ok((vj, vk))
    }

    /// `get_j` per spin set, routed directly (same reason as 18-05's
    /// [`KrksGradients::j_deriv`](crate::krks::KrksGradients::j_deriv)).
    pub fn j_deriv_sets(&self, dm: &KDms) -> Result<Vec<GradMatrices>, PyscfRsError> {
        let mf = self.mf;
        let nkpts = nkpts_of(mf.kpts());
        let nset = dm.len();
        let nao = mf.cell().mol.nao_nr;
        if nset == 0 {
            return Err(invalid("KUKS get_j: need at least one spin set"));
        }
        let vj = mf
            .with_df
            .get_j_e1(dm, mf.kpts(), None)
            .map_err(|e| df_err("get_j", e))?;
        take_grad_mats_sets(vj, "vj", nset, nkpts, nao)
    }

    /// `get_k` per spin set under a range-separated kernel — the RSH
    /// re-entry (`kuks.py:63-65`); `None` is the full-range kernel.
    pub fn k_deriv_sets(
        &self,
        dm: &KDms,
        omega: Option<f64>,
    ) -> Result<Vec<GradMatrices>, PyscfRsError> {
        let mf = self.mf;
        let nkpts = nkpts_of(mf.kpts());
        let nset = dm.len();
        let nao = mf.cell().mol.nao_nr;
        if nset == 0 {
            return Err(invalid("KUKS get_k: need at least one spin set"));
        }
        let vk = mf
            .with_df
            .get_k_e1(dm, mf.kpts(), None, mf.exxdiv, omega, None)
            .map_err(|e| df_err("get_k", e))?;
        take_grad_mats_sets(vk, "vk", nset, nkpts, nao)
    }

    /// `get_vxc` without the squeeze (`kuks.py:120`: `return -vmat` — the
    /// negation already happened in the shared core): one `[x][k]` family
    /// per spin set.
    pub fn vxc(&self, dm: &KDms) -> Result<Vec<GradMatrices>, PyscfRsError> {
        if dm.len() != 2 {
            return Err(invalid(format!(
                "KUKS get_vxc: need exactly 2 spin sets, got {}",
                dm.len()
            )));
        }
        get_vxc_sets(
            self.mf.cell(),
            dm,
            &self.mf.xc,
            &self.mf.grids,
            self.mf.kpts(),
        )
    }

    /// `get_veff(ks_grad, dm, kpts)` — `kuks.py:32-67`:
    ///
    /// ```text
    /// vxc = get_vxc(...)                       (refuses grid_response first)
    /// vxc += vj[:,0][:,None] + vj[:,1][:,None]  (pure functional: broadcast J)
    /// vxc += vj[:,0][:,None] + vj[:,1][:,None] - vk   (hybrid; vk *= hyb, RSH)
    /// ```
    ///
    /// The Coulomb halves add in the fixed order `(alpha, beta)` and the
    /// exchange subtracts in FULL (no `.5` — `kuks.py:66`, the KUKS/KUHF
    /// factor). Spin threading is exactly 18-06 Task 2's.
    pub fn veff(&self, dm: &KDms) -> Result<Vec<GradMatrices>, PyscfRsError> {
        if self.grid_response {
            return Err(PbcGradError::NotYetImplemented {
                phase: 18,
                what: "KUKS get_veff grid_response = True: upstream kuks.py:49-50 raises \
                       NotImplementedError (and :129 hard-sets it False); the molecular \
                       pyscf-grad uks.rs documents grid_response as fully supported, so \
                       the periodic subclass OVERRIDES it to a refusal",
            }
            .into());
        }
        if dm.len() != 2 {
            return Err(invalid(format!(
                "KUKS get_veff: need exactly 2 spin sets, got {}",
                dm.len()
            )));
        }
        let xc = &self.mf.xc;
        let vxc = self.vxc(dm)?;
        if vxc.len() != 2 {
            return Err(invalid(format!(
                "KUKS get_veff: grid core returned {} sets, need 2",
                vxc.len()
            )));
        }
        if !lift_dft(is_hybrid_xc(xc))? {
            let vj = self.j_deriv_sets(dm)?;
            if vj.len() != 2 {
                return Err(invalid(format!(
                    "KUKS get_veff: jk route returned {} J sets, need 2",
                    vj.len()
                )));
            }
            Ok(combine_veff_u(&vxc, &vj, None))
        } else {
            let (omega, alpha, hyb) = lift_dft(rsh_and_hybrid_coeff(xc))?;
            let (vj, vk_full) = self.jk_deriv(dm)?;
            if vj.len() != 2 || vk_full.len() != 2 {
                return Err(invalid(format!(
                    "KUKS get_veff: jk route returned {}/{} sets, need 2/2",
                    vj.len(),
                    vk_full.len()
                )));
            }
            let mut vk: Vec<GradMatrices> = vk_full.iter().map(|m| scale_mats(m, hyb)).collect();
            if omega != 0.0 {
                let vk_lr = self.k_deriv_sets(dm, Some(omega))?;
                for (dst, src) in vk.iter_mut().zip(vk_lr.iter()) {
                    add_scaled_mats_in_place(dst, src, alpha - hyb);
                }
            }
            Ok(combine_veff_u(&vxc, &vj, Some(&vk)))
        }
    }

    /// `grad_elec` — 18-06's assembly verbatim (upstream inherits it from
    /// `uhf_grad.Gradients` unchanged): summed densities for the `h1`/`s1`
    /// terms, spin-resolved `vhf`, `/nkpts` INSIDE the atom loop,
    /// `extra_force` (zero here) AFTER the division, summed-`dm` `vppnl`
    /// over the WHOLE array AFTER the loop.
    pub fn electronic_gradient(&self) -> Result<Gradient, PyscfRsError> {
        let mf = self.mf;
        let cell = mf.cell();
        let kpts = mf.kpts();
        let nkpts = nkpts_of(kpts) as f64;
        let nao = cell.mol.nao_nr;

        let dm0_sf = sum_sets(&self.dm0, nao)?;
        let dme0_sf = sum_sets(&self.dme0, nao)?;

        let tables = precompute_hcore(cell, kpts)?;
        let s1 = self.overlap_deriv()?;
        let vhf = self.veff(&self.dm0)?;
        let mut fused_stats = HcoreFusedStats::default();
        let fused = fused_local_contraction(&tables, cell, kpts, &dm0_sf, &mut fused_stats)?;
        let slices = pyscf_gto::aoslice_by_atom(&cell.mol)
            .map_err(|e| invalid(format!("KUKS grad_elec: aoslice failed: {e}")))?;

        let atmlst = self.atom_list();
        let mut de = Vec::with_capacity(atmlst.len());
        for &ia in &atmlst {
            let (_, _, p0, p1) = slices.get(ia).copied().ok_or_else(|| {
                invalid(format!(
                    "KUKS grad_elec: no aoslice for atom {ia} (natm = {})",
                    cell.natm
                ))
            })?;
            let h1 = contract_h1_atom(&tables.h1, &dm0_sf, p0, p1, nao);
            let vv = contract_vhf_atom_spin(&vhf, &self.dm0, p0, p1, nao);
            let ss = contract_ovlp_atom(&s1, &dme0_sf, p0, p1, nao);
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
        let vppnl = pyscf_pbc_gto::pseudo::vppnl_nuc_grad_kdm(cell, &dm0_sf, kpts)?;
        for (row, &ia) in de.iter_mut().zip(atmlst.iter()) {
            for x in 0..3 {
                row[x] = oracle_sum(&[row[x], vppnl[ia][x] / nkpts]);
            }
        }
        Ok(de)
    }

    /// `grad_nuc` — inherited unchanged: 18-03's `ewald_nuc_grad`.
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
                "KUKS kernel: nuclear part has the wrong atom count",
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

/// `veff[s] = vxc[s] + vj[0] + vj[1] (- vk[s])` (`kuks.py:58/:66`): the
/// Coulomb halves add in the fixed order `(alpha, beta)`; the exchange
/// subtracts in FULL when present (no `.5`). Elementwise over `[x][k]`
/// planes through [`oracle_sum`].
fn combine_veff_u(
    vxc: &[GradMatrices],
    vj: &[GradMatrices],
    vk: Option<&[GradMatrices]>,
) -> Vec<GradMatrices> {
    (0..2)
        .map(|s| {
            std::array::from_fn(|x| {
                vxc[s][x]
                    .iter()
                    .zip(vj[0][x].iter())
                    .zip(vj[1][x].iter())
                    .enumerate()
                    .map(|(k, ((a, b), c))| {
                        let sub = vk.is_some();
                        let (kr, ki) = match vk {
                            Some(v) => (v[s][x][k].re.as_slice(), v[s][x][k].im.as_slice()),
                            None => (&[][..], &[][..]),
                        };
                        CTensor::from_planes(
                            a.re
                                .iter()
                                .zip(&b.re)
                                .zip(&c.re)
                                .enumerate()
                                .map(|(i, ((p, q), r))| {
                                    if sub {
                                        oracle_sum(&[*p, *q, *r, -kr[i]])
                                    } else {
                                        oracle_sum(&[*p, *q, *r])
                                    }
                                })
                                .collect(),
                            a.im
                                .iter()
                                .zip(&b.im)
                                .zip(&c.im)
                                .enumerate()
                                .map(|(i, ((p, q), r))| {
                                    if sub {
                                        oracle_sum(&[*p, *q, *r, -ki[i]])
                                    } else {
                                        oracle_sum(&[*p, *q, *r])
                                    }
                                })
                                .collect(),
                        )
                    })
                    .collect()
            })
        })
        .collect()
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
/// `(alpha - hyb)` accumulation, `kuks.py:65`).
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

impl<'a> Gradients for KuksGradients<'a> {
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

    /// The inherited `get_jk` route, squeezed to the single-set shape the
    /// trait carries (18-06's seam): the spin-resolved pair lives in
    /// [`KuksGradients::jk_deriv`]/[`KuksGradients::veff`] and is refused
    /// here by name rather than squeezed silently.
    fn get_jk(&self, dm: &KDms) -> Result<(GradMatrices, GradMatrices), PyscfRsError> {
        let (vj, vk) = self.jk_deriv(dm)?;
        if dm.len() != 1 || vj.len() != 1 || vk.len() != 1 {
            return Err(invalid(
                "KUKS get_jk: the trait seam carries single-set densities only; \
                 the spin-resolved pair lives in jk_deriv/veff",
            ));
        }
        Ok((vj.into_iter().next().expect("checked"), vk.into_iter().next().expect("checked")))
    }

    fn grad_nuc(&self) -> Result<Gradient, PyscfRsError> {
        self.nuclear_gradient()
    }

    fn make_rdm1e(&self) -> Result<KDms, PyscfRsError> {
        Ok(self.dme0.clone())
    }

    /// As for KRKS: the shared-state scanner is KRHF-typed until 18-19 lands
    /// a KS-typed one — the trait's named refusal, never the wrong mean
    /// field.
    fn as_scanner(&self) -> Result<EnergyScanner, PyscfRsError> {
        Err(PbcGradError::NotYetImplemented {
            phase: 18,
            what: "KUKS as_scanner: the shared-state scanner is KRHF-typed until 18-19 \
                   lands a KS-typed one",
        }
        .into())
    }
}
