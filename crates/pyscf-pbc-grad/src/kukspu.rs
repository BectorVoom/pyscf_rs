//! KUKS+U k-point analytic nuclear gradient — `pyscf/pbc/grad/kukspu.py` (83 l).
//!
//! The unrestricted DFT+U gradient. Upstream declares it as
//! `class Gradients(kuks_grad.Gradients)` (`kukspu.py:76`) and imports
//! `generate_first_order_local_orbitals` straight from `krkspu`
//! (`kukspu.py:24`) — this module does the same, reusing
//! [`first_order_local_orbitals`](crate::krkspu::first_order_local_orbitals)
//! and [`make_coeff`](crate::krkspu::make_coeff) verbatim and porting only
//! the spin threading of [`hubbard_u_deriv1_uks`] (`_hubbard_U_deriv1`,
//! `:26-74`).
//!
//! # The spin threading (the whole plan)
//!
//! | restricted (`krkspu.py`) | unrestricted (`kukspu.py`) |
//! |---|---|
//! | one density set | **two** sets; `dm_deriv0`/`dm_deriv1` per spin (`:47-50`, `:64-65`) |
//! | `… - einsum('xij,ji->x', P1, P0).real * 2` | `… * 4` (`:73`) |
//!
//! The projectors are spin-independent (`kukspu.py:69-71` applies the same
//! `C_ao_lo` to both spins) — 17-08's D-17-08-02 again: no projector
//! rotation, and one appearing here is a bug.
//!
//! # Index orders, reductions, kernels
//!
//! As in [`crate::krkspu`]: upstream's einsum index order literally, complex
//! pairs accumulated through
//! [`oracle_sum`](pyscf_algebra::oracle_sum), no bare `+=`, no `cubecl-*`
//! (ALG-06). Gate B is **5e-6** (`18-CONTEXT §2.3`).

use pyscf_algebra::{CTensor, oracle_sum};
use pyscf_core::PyscfRsError;
use pyscf_pbc_dft::kspu::HubbardU;
use pyscf_pbc_gto::Cell;
use pyscf_pbc_scf::krdm::make_rdm1;
use pyscf_pbc_scf::types::KDms;

use crate::error::PbcGradError;
use crate::gradients::{Gradient, Gradients};
use crate::krkspu::{check_cfg, first_order_local_orbitals, ip_tables, make_coeff};
use crate::kuks::KuksGradients;

fn invalid(message: impl Into<String>) -> PyscfRsError {
    PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(message.into()))
}

fn lift_dft<T>(r: Result<T, pyscf_pbc_dft::PbcDftError>) -> Result<T, PyscfRsError> {
    r.map_err(|e| invalid(format!("KUKS+U gradient: DFT+U substrate failed: {e}")))
}

fn csum(terms: &[(f64, f64)]) -> (f64, f64) {
    let re: Vec<f64> = terms.iter().map(|(r, _)| *r).collect();
    let im: Vec<f64> = terms.iter().map(|(_, i)| *i).collect();
    (oracle_sum(&re), oracle_sum(&im))
}

/// `_hubbard_U_deriv1(mf, dm, kpts)` — `kukspu.py:26-74`.
///
/// `dm` is the spin pair (`dm[0]` alpha, `dm[1]` beta), each row-major
/// `dm[s][k]`, `nao × nao`, held fixed. Per-spin `dm_deriv0` (`:47-50`) and
/// `dm_deriv1` (`:65`), the spin-summed-free site reduction with `*2` on the
/// trace and **`*4`** on the `P1·P0` term (`:71-73`).
///
/// # Errors
/// A refused configuration, a shape disagreement, or a failed integral.
pub fn hubbard_u_deriv1_uks(
    cell: &Cell,
    dm: &KDms,
    kpts: &[[f64; 3]],
    cfg: &HubbardU,
) -> Result<Gradient, PyscfRsError> {
    use pyscf_pbc_dft::kspu::{make_minao_lo, reference_cell, set_u};

    check_cfg(cfg)?;
    if dm.len() != 2 {
        return Err(invalid(format!(
            "KUKS+U gradient: need exactly 2 spin density sets, got {}",
            dm.len()
        )));
    }
    let nao = cell.mol.nao_nr;
    let nkpts = kpts.len();
    if nkpts == 0 {
        return Err(PbcGradError::ShapeMismatch {
            expected: 1,
            got: 0,
        }
        .into());
    }
    for (s, set) in dm.iter().enumerate() {
        if set.len() != nkpts
            || set
                .iter()
                .any(|m| m.re.len() != nao * nao || m.im.len() != nao * nao)
        {
            return Err(PbcGradError::ShapeMismatch {
                expected: nao * nao,
                got: set.first().map(|m| m.re.len().min(m.im.len())).unwrap_or(0),
            }
            .into());
        }
        let _ = s;
    }

    let pcell = lift_dft(reference_cell(cell, &cfg.minao_ref))?;
    let c_lo = lift_dft(make_minao_lo(cell, &pcell, kpts))?;
    let resolved = lift_dft(set_u(&pcell, cfg))?;
    if resolved.indices.is_empty() {
        return Ok(vec![[0.0; 3]; cell.natm]);
    }
    let stack: Vec<usize> = resolved.indices.iter().flatten().copied().collect();
    let nu = stack.len();
    let nlo = c_lo[0].re.len() / nao;
    for &u in &stack {
        if u >= nlo {
            return Err(invalid(format!(
                "KUKS+U gradient: Hubbard index {u} exceeds nlo = {nlo}"
            )));
        }
    }
    let c0u = |ck: &CTensor, i: usize, a: usize| -> (f64, f64) {
        let q = i + stack[a] * nao;
        (ck.re[q], ck.im[q])
    };

    let ovlp0_f = pyscf_pbc_gto::get_ovlp(cell, kpts)
        .map_err(|e| invalid(format!("KUKS+U U1: get_ovlp failed: {e}")))?;
    let ovlp1 = pyscf_pbc_gto::pbc_intor(cell, "int1e_ipovlp", kpts, Default::default())
        .map_err(|e| invalid(format!("KUKS+U U1: int1e_ipovlp failed: {e}")))?;
    let mut s0: Vec<CTensor> = Vec::with_capacity(nkpts);
    for m in &ovlp0_f {
        let mut re = vec![0.0_f64; nao * nao];
        let mut im = vec![0.0_f64; nao * nao];
        for i in 0..nao {
            for j in 0..nao {
                re[i * nao + j] = m.re[i + j * nao];
                im[i * nao + j] = m.im[i + j * nao];
            }
        }
        s0.push(CTensor::from_planes(re, im));
    }

    let flo = first_order_local_orbitals(cell, &cfg.minao_ref, kpts)?;
    if flo.nlo != nlo {
        return Err(PbcGradError::ShapeMismatch {
            expected: nlo,
            got: flo.nlo,
        }
        .into());
    }
    let tables = ip_tables(cell, &pcell, kpts)?;
    let slices = pyscf_gto::aoslice_by_atom(&cell.mol)
        .map_err(|e| invalid(format!("KUKS+U U1: aoslice failed: {e}")))?;
    let mslices = pyscf_gto::aoslice_by_atom(&pcell.mol)
        .map_err(|e| invalid(format!("KUKS+U U1: MINAO aoslice failed: {e}")))?;
    let natm = cell.natm;

    let weight = 1.0 / nkpts as f64;
    let mut de = vec![[0.0_f64; 3]; natm];
    for (atm_id, &(_, _, p0, p1)) in slices.iter().enumerate() {
        let (_, _, q0, q1) = mslices[atm_id];
        let mut c1: Vec<[CTensor; 3]> = Vec::with_capacity(nkpts);
        for k in 0..nkpts {
            let full = make_coeff(&flo, k, p0, p1, q0, q1, &tables)?;
            c1.push(std::array::from_fn(|n| {
                let mut re = vec![0.0_f64; nao * nu];
                let mut im = vec![0.0_f64; nao * nu];
                for i in 0..nao {
                    for (b, &u) in stack.iter().enumerate() {
                        re[i * nu + b] = full[n].re[i * nlo + u];
                        im[i * nu + b] = full[n].im[i * nlo + u];
                    }
                }
                CTensor::from_planes(re, im)
            }));
        }
        for k in 0..nkpts {
            // `C_inv = C0^H · S0` (`:46`), spin-independent.
            let mut cinv_re = vec![0.0_f64; nu * nao];
            let mut cinv_im = vec![0.0_f64; nu * nao];
            for a in 0..nu {
                for b in 0..nao {
                    let mut terms = Vec::with_capacity(nao);
                    for i in 0..nao {
                        let (cr, ci) = c0u(&c_lo[k], i, a);
                        let (sr, si) = (s0[k].re[i * nao + b], s0[k].im[i * nao + b]);
                        terms.push((cr * sr + ci * si, cr * si - ci * sr));
                    }
                    let (rr, ri) = csum(&terms);
                    cinv_re[a * nao + b] = rr;
                    cinv_im[a * nao + b] = ri;
                }
            }
            // Per-spin `T_s = C_inv · dm_s[k]`, `dm_deriv0_s`, `SC1` (shared),
            // `dm_deriv1_s` (`:47-50`, `:61-65`).
            // SC1[x,p,i] is spin-independent — build once.
            let mut sc1_re = vec![0.0_f64; 3 * nao * nu];
            let mut sc1_im = vec![0.0_f64; 3 * nao * nu];
            for x in 0..3 {
                for p in 0..nao {
                    for i in 0..nu {
                        let mut terms = Vec::with_capacity(nao + 2 * (p1 - p0));
                        for q in 0..nao {
                            let (sr, si) = (s0[k].re[p * nao + q], s0[k].im[p * nao + q]);
                            let (cr, ci) = (c1[k][x].re[q * nu + i], c1[k][x].im[q * nu + i]);
                            terms.push((sr * cr - si * ci, sr * ci + si * cr));
                        }
                        for q in p0..p1 {
                            let (r, im) = ovlp1.element(k, x, q, p);
                            let (er, ei) = c0u(&c_lo[k], q, i);
                            terms.push((-(r * er + im * ei), -(im * er - r * ei)));
                        }
                        if (p0..p1).contains(&p) {
                            for q in 0..nao {
                                let (r, im) = ovlp1.element(k, x, p, q);
                                let (er, ei) = c0u(&c_lo[k], q, i);
                                terms.push((-(r * er - im * ei), -(r * ei + im * er)));
                            }
                        }
                        let (rr, ri) = csum(&terms);
                        sc1_re[(x * nao + p) * nu + i] = rr;
                        sc1_im[(x * nao + p) * nu + i] = ri;
                    }
                }
            }
            let mut row = [0.0_f64; 3];
            for dm_s in dm.iter() {
                // `T = C_inv · dm_s[k]`.
                let mut t_re = vec![0.0_f64; nu * nao];
                let mut t_im = vec![0.0_f64; nu * nao];
                for a in 0..nu {
                    for j in 0..nao {
                        let mut terms = Vec::with_capacity(nao);
                        for b in 0..nao {
                            let (cr, ci) = (cinv_re[a * nao + b], cinv_im[a * nao + b]);
                            let (dr, di) = (dm_s[k].re[b * nao + j], dm_s[k].im[b * nao + j]);
                            terms.push((cr * dr - ci * di, cr * di + ci * dr));
                        }
                        let (rr, ri) = csum(&terms);
                        t_re[a * nao + j] = rr;
                        t_im[a * nao + j] = ri;
                    }
                }
                // `dm_deriv0_s = T · C_inv^H`, full complex.
                let mut p0_re = vec![0.0_f64; nu * nu];
                let mut p0_im = vec![0.0_f64; nu * nu];
                for a in 0..nu {
                    for b in 0..nu {
                        let mut terms = Vec::with_capacity(nao);
                        for j in 0..nao {
                            let (tr, ti) = (t_re[a * nao + j], t_im[a * nao + j]);
                            let (cr, ci) = (cinv_re[b * nao + j], cinv_im[b * nao + j]);
                            terms.push((tr * cr + ti * ci, ti * cr - tr * ci));
                        }
                        let (rr, ri) = csum(&terms);
                        p0_re[a * nu + b] = rr;
                        p0_im[a * nu + b] = ri;
                    }
                }
                // `dm_deriv1_s = T · SC1`, full complex.
                let mut p1_re = vec![0.0_f64; 3 * nu * nu];
                let mut p1_im = vec![0.0_f64; 3 * nu * nu];
                for x in 0..3 {
                    for a in 0..nu {
                        for b in 0..nu {
                            let mut terms = Vec::with_capacity(nao);
                            for j in 0..nao {
                                let (tr, ti) = (t_re[a * nao + j], t_im[a * nao + j]);
                                let q = (x * nao + j) * nu + b;
                                let (sr, si) = (sc1_re[q], sc1_im[q]);
                                terms.push((tr * sr - ti * si, tr * si + ti * sr));
                            }
                            let (rr, ri) = csum(&terms);
                            p1_re[(x * nu + a) * nu + b] = rr;
                            p1_im[(x * nu + a) * nu + b] = ri;
                        }
                    }
                }
                let mut off = 0usize;
                for (idx, val) in resolved.indices.iter().zip(resolved.u_val.iter()) {
                    let n = idx.len();
                    let wgt = weight * val * 0.5;
                    for x in 0..3 {
                        let mut d_terms = Vec::with_capacity(n);
                        let mut p_terms = Vec::with_capacity(n * n);
                        for ai in 0..n {
                            let p = off + ai;
                            d_terms.push(p1_re[(x * nu + p) * nu + p]);
                            for bi in 0..n {
                                let q = off + bi;
                                let (ar, ai_) =
                                    (p1_re[(x * nu + p) * nu + q], p1_im[(x * nu + p) * nu + q]);
                                let (br, bi_) = (p0_re[q * nu + p], p0_im[q * nu + p]);
                                p_terms.push(ar * br - ai_ * bi_);
                            }
                        }
                        // `*2` on the trace, **`*4`** on the product (`:71-73`).
                        let term = oracle_sum(&[
                            wgt * 2.0 * oracle_sum(&d_terms),
                            -(wgt * 4.0 * oracle_sum(&p_terms)),
                        ]);
                        row[x] = oracle_sum(&[row[x], term]);
                    }
                    off += n;
                }
            }
            for x in 0..3 {
                de[atm_id][x] = oracle_sum(&[de[atm_id][x], row[x]]);
            }
        }
    }
    Ok(de)
}

// ---------------------------------------------------------------------------
// `Gradients(kuks_grad.Gradients)` — `kukspu.py:76-83`.
// ---------------------------------------------------------------------------

/// KUKS+U k-point nuclear gradient — `pbc/grad/kukspu.py`'s `Gradients` class.
///
/// 18-07's [`KuksGradients`] with `extra_force` extended by the per-spin `U`
/// term (`kukspu.py:81-83`), added per atom in [`kernel`](Self::kernel) where
/// upstream's `extra_force` sits.
pub struct KukspuGradients<'a> {
    base: KuksGradients<'a>,
    d_e_u: Gradient,
    atmlst: Option<Vec<usize>>,
}

impl<'a> KukspuGradients<'a> {
    /// Build from converged spin orbitals plus the Hubbard configuration.
    /// `mo_*` are in `idx(set, k)` order — alpha's `nkpts` blocks then
    /// beta's — the [`KuksGradients::new`](crate::kuks::KuksGradients::new)
    /// convention. The `U` term sees the same per-spin densities the base
    /// builds.
    ///
    /// # Errors
    /// Whatever [`KuksGradients::new`](crate::kuks::KuksGradients::new) or
    /// [`hubbard_u_deriv1_uks`] report.
    pub fn new(
        mf: &'a pyscf_pbc_dft::kuks::Kuks,
        mo_energy: Vec<Vec<f64>>,
        mo_coeff: Vec<CTensor>,
        mo_occ: Vec<Vec<f64>>,
        u: &HubbardU,
    ) -> Result<Self, PyscfRsError> {
        let nao = mf.cell().mol.nao_nr;
        let nkpts = mf.kpts().len();
        if mo_coeff.len() != 2 * nkpts {
            return Err(invalid(format!(
                "KUKS+U gradient: {nkpts} k-points need {} orbital blocks, got {}",
                2 * nkpts,
                mo_coeff.len()
            )));
        }
        let dm = vec![
            make_rdm1(&mo_coeff[..nkpts], &mo_occ[..nkpts], nao),
            make_rdm1(&mo_coeff[nkpts..], &mo_occ[nkpts..], nao),
        ];
        let d_e_u = if u.sites.is_empty() {
            vec![[0.0; 3]; mf.cell().natm]
        } else {
            hubbard_u_deriv1_uks(mf.cell(), &dm, mf.kpts(), u)?
        };
        let base = KuksGradients::new(mf, mo_energy, mo_coeff, mo_occ)?;
        Ok(Self {
            base,
            d_e_u,
            atmlst: None,
        })
    }

    /// Atom subset (`de = de[atmlst]`). `None` (default) is all atoms.
    pub fn with_atmlst(mut self, atmlst: Vec<usize>) -> Result<Self, PyscfRsError> {
        self.base = self.base.with_atmlst(atmlst.clone())?;
        self.atmlst = Some(atmlst);
        Ok(self)
    }

    /// `kernel` — 18-07's kernel plus the per-spin `U` rows
    /// (`kukspu.py:81-83`).
    pub fn kernel(&self) -> Result<Gradient, PyscfRsError> {
        let mut out = self.base.kernel()?;
        let list: Vec<usize> = match &self.atmlst {
            Some(l) => l.clone(),
            None => (0..self.d_e_u.len()).collect(),
        };
        if out.len() != list.len() {
            return Err(invalid(format!(
                "KUKS+U kernel: base returned {} rows for {} atoms",
                out.len(),
                list.len()
            )));
        }
        for (row, &ia) in out.iter_mut().zip(list.iter()) {
            for x in 0..3 {
                row[x] = oracle_sum(&[row[x], self.d_e_u[ia][x]]);
            }
        }
        Ok(out)
    }
}

impl<'a> Gradients for KukspuGradients<'a> {
    fn cell(&self) -> &Cell {
        self.base.cell()
    }

    fn kpts(&self) -> &[[f64; 3]] {
        self.base.kpts()
    }

    fn grad_elec(&self) -> Result<Gradient, PyscfRsError> {
        self.base.grad_elec()
    }

    fn grad_nuc(&self) -> Result<Gradient, PyscfRsError> {
        self.base.grad_nuc()
    }
}
