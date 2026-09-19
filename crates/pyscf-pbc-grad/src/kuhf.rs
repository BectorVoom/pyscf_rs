//! KUHF k-point analytic nuclear gradient — `pyscf/pbc/grad/kuhf.py` (124 l).
//!
//! The k-point unrestricted Hartree–Fock nuclear gradient. Upstream declares
//! it as `class Gradients(krhf_grad.GradientsBase)` (`kuhf.py:86`): it extends
//! 18-05's **base**, not its `Gradients`, so `grad_elec`, `get_veff` and
//! `make_rdm1e` are replaced wholesale while `get_hcore`, `hcore_generator`,
//! `get_ovlp`, `get_jk`, `grad_nuc` and `as_scanner` are inherited unchanged.
//! [`KuhfGradients`] mirrors that split: the three replaced methods are new
//! bodies below; the inherited ones delegate to the same 18-03/18-04 seams
//! 18-05 calls, with the same options and the same refusals.
//!
//! Gate B drives [`KuhfGradients::kernel`] through [`crate::verify_fd`] on a
//! spin-polarised cell (`tests/kuhf.rs`); Gate C checks `lib.fp` against
//! upstream's committed constant on upstream's closed-shell diamond.
//!
//! `as_scanner` (`kuhf.py:102`, re-exported from `krhf.Gradients`) is the one
//! inherited seam with no Rust body yet: [`crate::scanner`] is KRHF-typed
//! (18-19) and there is no KUHF-typed shared-state scanner to wrap this
//! gradient in, so [`KuhfGradients`] keeps the trait's named refusal until
//! one lands. Nothing here routes around that — the Gate B harness builds
//! its energy closure explicitly instead.
//!
//! # Upstream correspondence (`kuhf.py` line → here)
//!
//! | upstream | here |
//! |---|```
//! | `grad_elec` `:29-72` (spin threading `:47-49`) | [`KuhfGradients::electronic_gradient`] |
//! | `get_veff` `:74-78` (`vj[0] + vj[1] - vk`) | [`KuhfGradients::veff`] |
//! | `make_rdm1e` `:80-84` (per-spin pair) | [`KuhfGradients::new`] via 18-05's [`make_rdm1e_kpts`] |
//! | inherited `get_hcore`/`hcore_generator`/`get_ovlp`/`get_jk`/`grad_nuc` | same 18-05 seams, same options |
//! | `kernel` (inherited from `krhf.Gradients`) | [`KuhfGradients::kernel`] |
//!
//! # Which terms are spin-summed and which are not (`:47-49`)
//!
//! * the `h1ao` term contracts against `dm0_sf = dm0[0] + dm0[1]` — **spin
//!   summed** ([`sum_sets`], then 18-05's fused local contraction and
//!   `contract_h1_atom`);
//! * the `vhf` term contracts **spin-resolved**, `vhf[s][:,:,p0:p1]` against
//!   `dm0[s][:,:,p0:p1]`, the spin sum living INSIDE the einsum
//!   ([`contract_vhf_atom_spin`]);
//! * the `s1` term contracts against `dme0_sf = dme0[0] + dme0[1]` — spin
//!   summed again ([`sum_sets`], then 18-05's `contract_ovlp_atom`).
//!
//! Summing the middle term too is the classic UHF gradient error: it is off
//! by exactly the exchange asymmetry, so it is *correct for a closed-shell
//! fixture* and wrong for every open-shell one. `tests/kuhf.rs` gates it on
//! a genuinely spin-polarised cell (open-shell HeH doublet Gate B) plus a
//! unit test that exhibits the summed-middle-term error on synthetic
//! spin-resolved inputs and shows it vanishing on closed-shell ones.
//!
//! # Index orders, `.real` placement, scopes (inherited from 18-05)
//!
//! Every density is indexed **`ji`, not `ij`** (18-CONTEXT trap 9). The
//! `.real` lives INSIDE each contraction (D-PBC-31 clause 4:
//! [`contract_vhf_atom_spin`] pushes one real partial per `(s,k,i,j)`
//! through [`oracle_sum`]; the complex accumulator does not exist).
//! `de[x] /= nkpts` is INSIDE the atom loop, `extra_force` is added AFTER
//! the division, and `vppnl_nuc_grad/nkpts` covers the WHOLE array AFTER the
//! loop (18-CONTEXT trap 3).
//!
//! # Reductions and kernels (ALG-06, D-PBC-17, D-PBC-31)
//!
//! Every reduction routes through `pyscf_algebra::oracle_sum` over a
//! materialised partial buffer — never a bare `+=`.
//!
//! No CubeCL kernel is added here (ALG-06: `pyscf-pbc-grad` may not depend on
//! `cubecl-*` AT ALL; `xtask check-dependency-wall` enforces it). The CubeCL
//! manual (`INDEX.md` + `Cubecl_generics.md`, generics-`Float` kernels) was
//! read before writing; there is no device kernel in this file for it to
//! apply to. The one device computation consumed is the clause-10 shared
//! `(natm,3)` primitive inside 18-05's [`fused_local_contraction`].

use pyscf_algebra::{CTensor, oracle_sum};
use pyscf_core::{CoreError, PyscfRsError};
use pyscf_pbc_df::JkOpts;
use pyscf_pbc_gto::Cell;
use pyscf_pbc_scf::types::{KDms, KMats};
use pyscf_pbc_scf::{Kuhf, krdm::make_rdm1};

use crate::gradients::{GradMatrices, Gradient, Gradients};
use crate::krhf::{
    contractions::{contract_h1_atom, contract_ovlp_atom},
    fused_local_contraction, hcore_deriv_matrices, make_rdm1e_kpts, precompute_hcore,
    HcoreFusedStats,
};

fn invalid(message: impl Into<String>) -> PyscfRsError {
    PyscfRsError::Core(CoreError::InvalidMolecule(message.into()))
}

fn df_err(context: &'static str, e: pyscf_pbc_df::PbcDfError) -> PyscfRsError {
    match e {
        pyscf_pbc_df::PbcDfError::Core(c) => c,
        other => PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "KUHF gradient ({context}): density fitting failed: {other}"
        ))),
    }
}

/// Number of k-points, treating an empty list as the single gamma point
/// (the `pbc_intor` convention; same helper as 18-05).
fn nkpts_of(kpts: &[[f64; 3]]) -> usize {
    if kpts.is_empty() { 1 } else { kpts.len() }
}

/// Spin-summed density `dm[0][k] + dm[1][k]` per k-point — upstream's `dm0_sf`
/// (`kuhf.py:40`) and `dme0_sf` (`:41`).
///
/// Elementwise [`oracle_sum`] over the ordered pair, so the sum is
/// deterministic. Both sets must be present with identical shapes.
///
/// # Errors
/// [`CoreError::InvalidMolecule`] unless `dm` holds exactly two sets of
/// equal-length `nao × nao` matrices.
pub fn sum_sets(dm: &KDms, nao: usize) -> Result<KMats, PyscfRsError> {
    if dm.len() != 2 {
        return Err(invalid(format!(
            "KUHF spin sum: need exactly 2 spin sets, got {}",
            dm.len()
        )));
    }
    let nkpts = dm[0].len();
    if dm[1].len() != nkpts {
        return Err(invalid(format!(
            "KUHF spin sum: alpha has {nkpts} k-blocks but beta has {}",
            dm[1].len()
        )));
    }
    let mut out = Vec::with_capacity(nkpts);
    for k in 0..nkpts {
        let (a, b) = (&dm[0][k], &dm[1][k]);
        if a.re.len() != nao * nao || a.im.len() != nao * nao || a.len() != b.len() {
            return Err(invalid(format!(
                "KUHF spin sum: k-point {k} shapes disagree (nao = {nao})"
            )));
        }
        out.push(CTensor::from_planes(
            a.re
                .iter()
                .zip(&b.re)
                .map(|(x, y)| oracle_sum(&[*x, *y]))
                .collect(),
            a.im
                .iter()
                .zip(&b.im)
                .map(|(x, y)| oracle_sum(&[*x, *y]))
                .collect(),
        ));
    }
    Ok(out)
}

/// `einsum('xskij,skji->x', vhf[:,:,:,p0:p1], dm0[:,:,:,p0:p1]).real * 2`
/// (`kuhf.py:48`; nabla was applied on the bra, `*2` for `nabla|ket>`).
///
/// The spin-RESOLVED middle term: `vhf[s][x,k,i,j]` contracts against
/// `dm0[s][k,j,i]` and the spin sum lives INSIDE the einsum — one real
/// partial per `(s,k,i,j)` into a single [`oracle_sum`], then `× 2`.
/// Summing `vhf` over spin BEFORE contracting is the classic UHF gradient
/// error (off by the exchange asymmetry); `tests/kuhf.rs` exhibits it.
///
/// `i` runs over atom A's ROWS on both factors (upstream slices the second
/// axis: `M[:,:,:,p0:p1]` is rows, exactly as 18-05's `contract_vhf_atom`);
/// the ket index `j` runs over ALL AOs. Densities are `ji`-indexed
/// (18-CONTEXT trap 9) and `.real` is taken inside (D-PBC-31 clause 4).
///
/// Both `vhf` and `dm0` hold exactly two spin sets; `vhf[s]` is `[x][k]`
/// row-major `nao × nao`, `dm0[s]` is per-k row-major.
pub fn contract_vhf_atom_spin(
    vhf: &[GradMatrices],
    dm0: &KDms,
    p0: usize,
    p1: usize,
    nao: usize,
) -> [f64; 3] {
    debug_assert_eq!(vhf.len(), 2, "KUHF vhf contraction needs 2 spin sets");
    debug_assert_eq!(dm0.len(), 2, "KUHF vhf contraction needs 2 spin sets");
    std::array::from_fn(|x| {
        let mut terms = Vec::new();
        for (v, d) in vhf.iter().zip(dm0.iter()) {
            for (vv, dd) in v[x].iter().zip(d.iter()) {
                for i in p0.min(nao)..p1.min(nao) {
                    for j in 0..nao {
                        let (ar, ai) = (vv.re[i * nao + j], vv.im[i * nao + j]);
                        let (br, bi) = (dd.re[j * nao + i], dd.im[j * nao + i]);
                        terms.push(ar * br - ai * bi);
                    }
                }
            }
        }
        2.0 * oracle_sum(&terms)
    })
}

// ---------------------------------------------------------------------------
// `KuhfGradients` — `class Gradients(krhf_grad.GradientsBase)` (kuhf.py:86)
// ---------------------------------------------------------------------------

/// KUHF k-point nuclear gradient — `pbc/grad/kuhf.py`'s `Gradients` class.
///
/// Borrowed mean field ([`Kuhf`], owning `with_df`, `cell`, `kpts`,
/// `exxdiv`) plus caller-held SCF products: the per-spin densities `dm0`
/// and energy-weighted densities `dme0` are built by [`KuhfGradients::new`]
/// from the converged orbitals — `dm0` through the SCF `make_rdm1` split
/// into alpha/beta halves, `dme0` through 18-05's restricted
/// [`make_rdm1e_kpts`] applied to each spin's orbitals (`kuhf.py:80-84`).
/// No SCF runs here, so there is no convergence path and no
/// second-solution noise.
///
/// `extra_force` is the shared zero default (upstream re-exports
/// `krhf_grad.Gradients.extra_force`, which returns `0` on the base).
pub struct KuhfGradients<'a> {
    mf: &'a Kuhf,
    dm0: KDms,
    dme0: KDms,
    atmlst: Option<Vec<usize>>,
}

impl<'a> KuhfGradients<'a> {
    /// Build from converged orbitals. Blocks are in `idx(set, k)` order —
    /// alpha's `nkpts` blocks then beta's (`kuhf.py`'s `(2, nkpts, ...)`
    /// stacking): `mo_coeff` column-major `nao × nmo` per block,
    /// `mo_energy`/`mo_occ` per orbital per block.
    ///
    /// # Errors
    /// [`CoreError::InvalidMolecule`] on k-point/shape disagreement or
    /// non-finite input.
    pub fn new(
        mf: &'a Kuhf,
        mo_energy: Vec<Vec<f64>>,
        mo_coeff: Vec<CTensor>,
        mo_occ: Vec<Vec<f64>>,
    ) -> Result<Self, PyscfRsError> {
        let nkpts = nkpts_of(mf.kpts());
        let nao = mf.cell().mol.nao_nr;
        if mo_coeff.len() != 2 * nkpts || mo_energy.len() != 2 * nkpts || mo_occ.len() != 2 * nkpts
        {
            return Err(invalid(format!(
                "KUHF gradient: {nkpts} k-points need {}/{}/{} coeff/energy/occ blocks (alpha, beta), got {}/{}/{}",
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
                    "KUHF gradient: spin set {s} density is misshapen or non-finite"
                )));
            }
        }
        // kuhf.py:80-84 — each spin through 18-05's restricted make_rdm1e.
        let dme0 = vec![
            make_rdm1e_kpts(&mo_coeff[..nkpts], &mo_energy[..nkpts], &mo_occ[..nkpts], nao)?,
            make_rdm1e_kpts(&mo_coeff[nkpts..], &mo_energy[nkpts..], &mo_occ[nkpts..], nao)?,
        ];
        Ok(Self {
            mf,
            dm0,
            dme0,
            atmlst: None,
        })
    }

    /// Atom subset (`de = de[atmlst]`). `None` (default) is all atoms.
    pub fn with_atmlst(mut self, atmlst: Vec<usize>) -> Result<Self, PyscfRsError> {
        for &ia in &atmlst {
            if ia >= self.mf.cell().natm {
                return Err(invalid(format!(
                    "KUHF gradient: atmlst id {ia} out of range for {} atoms",
                    self.mf.cell().natm
                )));
            }
        }
        self.atmlst = Some(atmlst);
        Ok(self)
    }

    fn atom_list(&self) -> Vec<usize> {
        match &self.atmlst {
            Some(list) => list.clone(),
            None => (0..self.mf.cell().natm).collect(),
        }
    }

    /// `get_ovlp(cell, kpts)` — inherited unchanged from 18-05's base
    /// (`krhf.py:114-115`): `-cell.pbc_intor('int1e_ipovlp', kpts)`,
    /// `[x][k]` row-major. (Upstream subclasses inherit the method; the Rust
    /// port repeats the 18-05 body because it borrows a different mf type.)
    pub fn overlap_deriv(&self) -> Result<GradMatrices, PyscfRsError> {
        let cell = self.mf.cell();
        let kpts = self.mf.kpts();
        let nao = cell.mol.nao_nr;
        let out = pyscf_pbc_gto::pbc_intor(cell, "int1e_ipovlp", kpts, Default::default())?;
        if out.comp != 3 || out.ni != nao || out.nj != nao {
            return Err(invalid(format!(
                "KUHF get_ovlp: int1e_ipovlp has comp = {}, ni/nj = {}/{} for nao = {nao}",
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

    /// `get_jk(dm, kpts)` — the inherited 18-04 route (`krhf.py:262`,
    /// `self.base.with_df.get_jk_e1(...)`), nset-generic: the per-set pairs
    /// `vj[s]`, `vk[s]`, each `[x][k]` row-major.
    ///
    /// The gradient entry point takes NO k-pair symmetry flag (D-PBC-30
    /// clause 4b, exactly as 18-05): the derivative sits on the bra, so the
    /// energy-path conjugate identity does not close. `kk_symmetry: false`
    /// is passed explicitly. A non-FFTDF builder serves 18-04's named
    /// refusal, never a fallback.
    pub fn jk_deriv(
        &self,
        dm: &KDms,
    ) -> Result<(Vec<GradMatrices>, Vec<GradMatrices>), PyscfRsError> {
        let mf = self.mf;
        let nkpts = nkpts_of(mf.kpts());
        let nset = dm.len();
        let nao = mf.cell().mol.nao_nr;
        if nset == 0 {
            return Err(invalid("KUHF get_jk: need at least one spin set"));
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
        let vj = take_sets(
            res.vj.ok_or_else(|| invalid("KUHF get_jk: missing vj"))?,
            "vj",
            nset,
            nkpts,
            nao,
        )?;
        let vk = take_sets(
            res.vk.ok_or_else(|| invalid("KUHF get_jk: missing vk"))?,
            "vk",
            nset,
            nkpts,
            nao,
        )?;
        Ok((vj, vk))
    }

    /// `get_veff(dm, kpts)` — `kuhf.py:74-78`: the spin-summed Coulomb minus
    /// the spin-resolved exchange, per spin `s`:
    /// `veff[s][x,k] = vj[0][x,k] + vj[1][x,k] - vk[s][x,k]`.
    /// The Coulomb halves add in the fixed order `(alpha, beta, -exchange)`.
    pub fn veff(&self, dm: &KDms) -> Result<Vec<GradMatrices>, PyscfRsError> {
        if dm.len() != 2 {
            return Err(invalid(format!(
                "KUHF get_veff: need exactly 2 spin sets, got {}",
                dm.len()
            )));
        }
        let (vj, vk) = self.jk_deriv(dm)?;
        if vj.len() != 2 || vk.len() != 2 {
            return Err(invalid(format!(
                "KUHF get_veff: jk route returned {}/{} sets, need 2/2",
                vj.len(),
                vk.len()
            )));
        }
        let mut out = Vec::with_capacity(2);
        for s in 0..2 {
            out.push(std::array::from_fn(|x| {
                vj[0][x]
                    .iter()
                    .zip(vj[1][x].iter())
                    .zip(vk[s][x].iter())
                    .map(|((a, b), c)| {
                        CTensor::from_planes(
                            a.re
                                .iter()
                                .zip(&b.re)
                                .zip(&c.re)
                                .map(|((p, q), r)| oracle_sum(&[*p, *q, -*r]))
                                .collect(),
                            a.im
                                .iter()
                                .zip(&b.im)
                                .zip(&c.im)
                                .map(|((p, q), r)| oracle_sum(&[*p, *q, -*r]))
                                .collect(),
                        )
                    })
                    .collect()
            }));
        }
        Ok(out)
    }

    /// `grad_elec` — `kuhf.py:29-72`, in upstream's exact order, with the
    /// clause-7 fused local-PP contraction (shared with 18-05) in place of
    /// the materialised `hcore_deriv(ia)` matrix:
    ///
    /// ```text
    /// dm0_sf/dme0_sf           spin-summed densities               (:40-41)
    /// h1ao/fused → de[x] += h-part, summed dm                     (:47 first)
    /// vhf       → de[x] += 2·vhf-part, SPIN-RESOLVED              (:48)
    /// s1        → de[x] -= 2·ovlp-part, summed dme0               (:49)
    ///             de[x] /= nkpts        INSIDE atom loop           (:50)
    ///             de    += extra_force  AFTER division             (:51, zero here)
    /// de += vppnl_nuc_grad(cell, dm0_sf)/nkpts  WHOLE array       (:52, after loop)
    /// ```
    ///
    /// Three scopes in five lines (18-CONTEXT trap 3): reproduce literally.
    pub fn electronic_gradient(&self) -> Result<Gradient, PyscfRsError> {
        let mf = self.mf;
        let cell = mf.cell();
        let kpts = mf.kpts();
        let nkpts = nkpts_of(kpts) as f64;
        let nao = cell.mol.nao_nr;

        // kuhf.py:40-41 — the summed halves.
        let dm0_sf = sum_sets(&self.dm0, nao)?;
        let dme0_sf = sum_sets(&self.dme0, nao)?;

        let tables = precompute_hcore(cell, kpts)?;
        let s1 = self.overlap_deriv()?;
        // kuhf.py:37 — spin-resolved, summed over spin only inside the einsum.
        let vhf = self.veff(&self.dm0)?;
        let mut fused_stats = HcoreFusedStats::default();
        let fused = fused_local_contraction(&tables, cell, kpts, &dm0_sf, &mut fused_stats)?;
        let slices = pyscf_gto::aoslice_by_atom(&cell.mol)
            .map_err(|e| invalid(format!("KUHF grad_elec: aoslice failed: {e}")))?;

        let atmlst = self.atom_list();
        let mut de = Vec::with_capacity(atmlst.len());
        for &ia in &atmlst {
            let (_, _, p0, p1) = slices.get(ia).copied().ok_or_else(|| {
                invalid(format!(
                    "KUHF grad_elec: no aoslice for atom {ia} (natm = {})",
                    cell.natm
                ))
            })?;
            let h1 = contract_h1_atom(&tables.h1, &dm0_sf, p0, p1, nao);
            let vv = contract_vhf_atom_spin(&vhf, &self.dm0, p0, p1, nao);
            let ss = contract_ovlp_atom(&s1, &dme0_sf, p0, p1, nao);
            let mut row = [0.0_f64; 3];
            for x in 0..3 {
                // kuhf.py:47/48/49 accumulate, :50 divides INSIDE the loop.
                row[x] = oracle_sum(&[fused[ia][x], h1[x], vv[x], ss[x]]) / nkpts;
            }
            // kuhf.py:51 — AFTER the division (a no-op here: the base-class
            // extra_force is zero).
            let extra = self.extra_force(ia, &[])?;
            for x in 0..3 {
                row[x] = oracle_sum(&[row[x], extra[x]]);
            }
            de.push(row);
        }
        // kuhf.py:52 — the whole array, after the loop (trap 3), against the
        // SUMMED density.
        let vppnl = pyscf_pbc_gto::pseudo::vppnl_nuc_grad_kdm(cell, &dm0_sf, kpts)?;
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
                "KUHF kernel: nuclear part has the wrong atom count",
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

/// Reshape one half of an 18-04 gradient J/K build into per-set
/// `[x][k]` matrices, validating the `(x, set, k)` shape and the
/// `nao × nao` planes.
fn take_sets(
    mats: pyscf_pbc_df::fft_jk_grad::GradMats,
    what: &str,
    nset: usize,
    nkpts: usize,
    nao: usize,
) -> Result<Vec<GradMatrices>, PyscfRsError> {
    if mats.len() != 3 || mats.iter().any(|s| s.len() != nset || s.iter().any(|k| k.len() != nkpts))
    {
        return Err(invalid(format!(
            "KUHF get_jk: {what} has the wrong (x, set, k) shape for nset = {nset}, nkpts = {nkpts}"
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
                    "KUHF get_jk: {what} plane is not nao x nao = {} for nao = {nao}",
                    kxm.re.len()
                )));
            }
        }
    }
    Ok(out)
}

impl<'a> Gradients for KuhfGradients<'a> {
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
    /// trait carries: serves 18-04's named refusal on non-FFTDF builders and
    /// the identical `get_jk_e1` call. A two-set density's spin-resolved
    /// pair lives in [`KuhfGradients::jk_deriv`]/[`KuhfGradients::veff`]
    /// and is refused here by name rather than squeezed silently.
    fn get_jk(&self, dm: &KDms) -> Result<(GradMatrices, GradMatrices), PyscfRsError> {
        let (vj, vk) = self.jk_deriv(dm)?;
        if dm.len() != 1 || vj.len() != 1 || vk.len() != 1 {
            return Err(invalid(
                "KUHF get_jk: the trait seam carries single-set densities only; \
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
}
