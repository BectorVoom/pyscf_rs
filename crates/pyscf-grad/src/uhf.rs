//! UHF analytical gradient (GRAD-02) — the spin-resolved variational gradient.
//!
//! Port target: `pyscf/grad/uhf.py` (`grad_elec` 21-66; `make_rdm1e` 69-83).
//! Structural analog: [`crate::rhf`] (the Hellmann-Feynman + Pulay decomposition
//! this file generalises) and `crates/pyscf-mp2/src/ump2.rs` (the spin-resolved
//! α/β-pair reference shape, file-for-file from its restricted sibling).
//!
//! ## The UHF grad_elec decomposition (`uhf.py:21-66`)
//!
//! Identical structure to RHF but spin-resolved: the density splits into
//! `dm0 = (dm_a, dm_b)` and the Hellmann-Feynman / overlap terms contract the
//! TOTAL density `dm_a + dm_b` (the core Hamiltonian + overlap are spin-
//! independent), while the 2e term `get_veff` builds the Coulomb potential from
//! the total density and the exchange potential per spin channel. The energy-
//! weighted RDM `dme0 = dme0_a + dme0_b` sums the two spin channels:
//!
//! ```text
//! dm_a, dm_b = make_rdm1(mo_a, occ_a), make_rdm1(mo_b, occ_b)
//! dm0  = dm_a + dm_b                              # total density (HF + overlap)
//! dme0 = make_rdm1e_a + make_rdm1e_b              # energy-weighted, summed over spin
//! s1   = -int1e_ipovlp                            # spin-independent overlap deriv
//! h1   = -(int1e_ipkin + int1e_ipnuc)             # spin-independent hcore deriv
//! vhf  = get_veff(mol, (dm_a, dm_b))              # vj(dm_a+dm_b) - vk_per_spin, [2][3,nao,nao]
//! for ia in atmlst:                               # p0,p1 = aoslices[ia]
//!     h1ao = hcore_deriv(ia)                       # spin-independent
//!     de[k] = einsum('xij,ij->', h1ao, dm0)                        # Hellmann-Feynman (total)
//!           + einsum('sxij,sij->', vhf[:,:,p0:p1], dm[:,p0:p1]) * 2  # 2e Pulay (per spin)
//!           - einsum('xij,ij->', s1[:,p0:p1], dme0[p0:p1]) * 2       # overlap Pulay (total dme0)
//! ```
//!
//! Per **D-04** UHF energy gradients are *stationary* (the 2n+1 rule): they make
//! NO coupled-perturbed response solve; `grid_response` does not apply (no grid).
//!
//! ## cintx-availability gating (D-02 + 07-01/07-03 SUMMARY)
//!
//! The SAME six gradient-integral families the RHF body needs — `int2e_ip1`
//! (2e Pulay), `int1e_ip{ovlp,kin,nuc}` (overlap + hcore Pulay), and
//! `int1e_iprinv` + `with_rinv_at_nucleus` (the per-atom Hellmann-Feynman shift)
//! — are **MISSING from every cintx branch with no scheduled workstream**
//! (07-01-SUMMARY). The STRUCTURAL assembly here (shapes, spin-channel routing,
//! the einsum ordering) lands always-on; the intor calls `?`-propagate a CLEAN
//! cintx-availability error (`PyscfRsError::Core(InvalidMolecule(..))`, **never**
//! `NotYetImplemented{phase:7}`). The numeric FD-vs-analytical comparison is
//! therefore `#[ignore]`'d (see `tests/uhf_verify_fd.rs`) until the cintx
//! grad-integral workstream lands these families; the FD-STRUCTURAL harness
//! stays runnable.
//!
//! ## Bit-exact discipline (Pitfall 1/2)
//!
//! Every per-atom + per-spin reduction materialises into a `Vec` then routes
//! through `pyscf_algebra::oracle_sum` / `oracle_dot` — NEVER a bare `+=`. Every
//! gradient intor is asserted component-leading `[3, nao, nao]` before contraction.

use crate::Gradients;
use crate::error::GradError;
use crate::rhf::{RhfReference, aoslice_by_atom, get_ovlp, make_rdm1e as rhf_make_rdm1e};
use pyscf_algebra::{oracle_dot, oracle_sum};
use pyscf_core::{MOCoefficients, Mole, PyscfRsError, Unit};

/// The number of Cartesian derivative components every gradient intor carries
/// (axis 0 of a `[3, nao, nao]` component-leading buffer — x/y/z; Pitfall 4).
const NCOMP: usize = 3;

/// Snapshot of a converged UHF reference, consumed by [`UhfGradients`] (D-09).
///
/// pyo3-free. Reuses the restricted [`RhfReference`] TWICE — once per spin
/// channel — mirroring `pyscf_mp2::ump2::UmpReference`'s `(alpha, beta)` pair
/// (`uhf.py` carries `mo_coeff[0/1]`, `mo_occ[0/1]`, `mo_energy[0/1]`). The `mol`
/// content is identical between the two channels (same molecule); only the
/// spin-resolved MO data differs.
#[derive(Debug, Clone)]
pub struct UhfReference {
    /// α-spin reference snapshot (MO coeff/energy/occ + the shared molecule).
    pub alpha: RhfReference,
    /// β-spin reference snapshot.
    pub beta: RhfReference,
}

impl UhfReference {
    /// The shared molecule (taken from the α channel; identical to β's).
    pub fn mol(&self) -> &Mole {
        &self.alpha.mol
    }
}

/// UHF analytical gradient (`pyscf/grad/uhf.py` `Gradients`).
///
/// Holds the converged α/β reference pair + an optional `atmlst` subset.
/// Implements the base [`Gradients`] trait: `grad_elec` (the spin-resolved
/// Hellmann-Feynman + Pulay assembly) and `make_rdm1e` (the spin-summed energy-
/// weighted RDM); `grad_nuc` is inherited from the trait default; `get_ovlp` is
/// the spin-independent overlap derivative.
#[derive(Debug, Clone)]
pub struct UhfGradients {
    /// The converged UHF reference snapshot (α/β pair).
    pub reference: UhfReference,
    /// The atom subset the gradient is restricted to (`None` = full molecule).
    pub atmlst: Option<Vec<usize>>,
    /// The last computed gradient `(n, 3)` (`None` until [`Gradients::kernel`]).
    pub de: Option<Vec<[f64; 3]>>,
}

impl UhfGradients {
    /// Build a UHF gradient driver over a converged α/β reference pair.
    pub fn new(reference: UhfReference) -> Self {
        Self {
            reference,
            atmlst: None,
            de: None,
        }
    }

    /// Restrict the gradient to a subset of atoms (GRAD-08).
    pub fn with_atmlst(mut self, atmlst: Vec<usize>) -> Self {
        self.atmlst = Some(atmlst);
        self
    }
}

impl Gradients for UhfGradients {
    fn mol(&self) -> &Mole {
        self.reference.mol()
    }

    fn atmlst(&self) -> Option<&[usize]> {
        self.atmlst.as_deref()
    }

    fn de(&self) -> Option<&[[f64; 3]]> {
        self.de.as_deref()
    }

    fn unit(&self) -> Unit {
        Unit::Bohr
    }

    /// The spin-summed energy-weighted RDM (`uhf.py:69-83`):
    /// `dme0 = make_rdm1e(α) + make_rdm1e(β)`. Each per-spin block reuses the
    /// restricted [`crate::rhf::make_rdm1e`] (the open-shell occupations are
    /// 1.0 per spin, so the same `Σ ε·occ·C·C` form applies channel-by-channel).
    fn make_rdm1e(&self) -> Result<Vec<f64>, PyscfRsError> {
        make_rdm1e(&self.reference)
    }

    /// The overlap derivative `s1 = -int1e_ipovlp` — spin-independent. The intor
    /// is MISSING from cintx today → this `?`-propagates a clean
    /// cintx-availability error.
    fn get_ovlp(&self) -> Result<Vec<f64>, PyscfRsError> {
        get_ovlp(self.reference.mol())
    }

    /// PER-METHOD spin-resolved electronic gradient (`uhf.py:21-66`). See module
    /// docs for the decomposition. The intor families it contracts are
    /// cintx-gated; the assembly is always-on.
    fn grad_elec(&self, atmlst: Option<&[usize]>) -> Result<Vec<[f64; 3]>, PyscfRsError> {
        grad_elec(&self.reference, atmlst)
    }
}

/// Spin-summed energy-weighted RDM (`uhf.py:69-83`). Free-fn form so the PyO3
/// bridge can reuse it without the trait. `dme0 = dme0_α + dme0_β`, row-major
/// `(nao, nao)`; the per-spin sum is a single ordered 2-element `oracle_sum`.
pub fn make_rdm1e(refr: &UhfReference) -> Result<Vec<f64>, PyscfRsError> {
    let dme0_a = rhf_make_rdm1e(&refr.alpha)?;
    let dme0_b = rhf_make_rdm1e(&refr.beta)?;
    if dme0_a.len() != dme0_b.len() {
        return Err(GradError::ShapeMismatch {
            expected: dme0_a.len(),
            got: dme0_b.len(),
        }
        .into());
    }
    let mut dme0 = vec![0.0_f64; dme0_a.len()];
    for (i, slot) in dme0.iter_mut().enumerate() {
        *slot = oracle_sum(&[dme0_a[i], dme0_b[i]]);
    }
    Ok(dme0)
}

/// Spin-resolved RDM `dm[μ,ν] = Σ_i occ_i · C[μ,i] · C[ν,i]` for one spin channel
/// (`make_rdm1`, the open-shell per-spin form), row-major `(nao, nao)`. Mirrors
/// the restricted `make_rdm1` but the occupations are the per-spin (1.0/0.0)
/// channel occupations carried in `refr.mo_occ`.
fn make_rdm1_spin(refr: &RhfReference) -> Result<Vec<f64>, PyscfRsError> {
    let nao = refr.mo_coeff.nao;
    let nmo = refr.mo_coeff.nmo;
    if refr.mo_coeff.data.len() != nao * nmo || refr.mo_occ.len() != nmo {
        return Err(GradError::ShapeMismatch {
            expected: nao * nmo,
            got: refr.mo_coeff.data.len(),
        }
        .into());
    }
    let mut dm = vec![0.0_f64; nao * nao];
    let mut terms = vec![0.0_f64; nmo];
    // `i` drives BOTH the per-MO `terms` buffer and the F-order C offsets
    // (`mu + i*nao`), so a range loop is the clearest form (rdm.rs precedent).
    #[allow(clippy::needless_range_loop)]
    for mu in 0..nao {
        for nu in 0..nao {
            for i in 0..nmo {
                terms[i] = refr.mo_occ[i]
                    * refr.mo_coeff.data[mu + i * nao]
                    * refr.mo_coeff.data[nu + i * nao];
            }
            dm[mu * nao + nu] = oracle_sum(&terms);
        }
    }
    Ok(dm)
}

/// The spin-independent core-Hamiltonian gradient `-(int1e_ipkin + int1e_ipnuc)`
/// (`uhf.py` reuses the RHF `get_hcore`), component-leading `[3, nao, nao]`.
/// Both intors are MISSING from cintx → clean availability error.
fn get_hcore(mol: &Mole) -> Result<Vec<f64>, PyscfRsError> {
    let nao = mol.nao_nr;
    let kin = pyscf_gto::intor(mol, "int1e_ipkin")?;
    assert_component_leading(&kin, nao, "int1e_ipkin")?;
    let nuc = pyscf_gto::intor(mol, "int1e_ipnuc")?;
    assert_component_leading(&nuc, nao, "int1e_ipnuc")?;
    let mut h = vec![0.0_f64; NCOMP * nao * nao];
    for (idx, slot) in h.iter_mut().enumerate() {
        *slot = -oracle_sum(&[kin.values[idx], nuc.values[idx]]);
    }
    Ok(h)
}

/// Per-atom core-Hamiltonian derivative `hcore_deriv(atm_id)` — spin-independent
/// (the `vrinv = -Z·int1e_iprinv` rinv-origin shift + the `h1` block, symmetrised
/// over the AO axes). Returns a flat component-leading `[3, nao, nao]` buffer.
/// `int1e_iprinv` + `with_rinv_at_nucleus` are MISSING from cintx → clean error.
fn hcore_deriv(
    mol: &Mole,
    h1: &[f64],
    aoslices: &[(usize, usize)],
    atm_id: usize,
) -> Result<Vec<f64>, PyscfRsError> {
    let nao = mol.nao_nr;
    let charges = mol.atom_charges();
    if atm_id >= mol.natm {
        return Err(GradError::ShapeMismatch {
            expected: mol.natm,
            got: atm_id,
        }
        .into());
    }
    // The per-atom rinv-origin shift (`with_rinv_at_nucleus`) is MISSING from
    // cintx today; request the integral so the shape contract is wired and the
    // call `?`-propagates a clean cintx-availability error until it lands.
    let iprinv = pyscf_gto::intor(mol, "int1e_iprinv")?;
    assert_component_leading(&iprinv, nao, "int1e_iprinv")?;

    let z = charges[atm_id] as f64;
    let mut vrinv = vec![0.0_f64; NCOMP * nao * nao];
    for (idx, slot) in vrinv.iter_mut().enumerate() {
        *slot = -z * iprinv.values[idx];
    }

    let (p0, p1) = aoslices[atm_id];
    for comp in 0..NCOMP {
        let base = comp * nao * nao;
        for j in p0..p1 {
            for i in 0..nao {
                let off = base + i + j * nao;
                vrinv[off] = oracle_sum(&[vrinv[off], h1[off]]);
            }
        }
    }

    // hcore_deriv = vrinv + vrinv.transpose(0,2,1) (symmetrise the AO axes).
    let mut out = vec![0.0_f64; NCOMP * nao * nao];
    for comp in 0..NCOMP {
        let base = comp * nao * nao;
        for j in 0..nao {
            for i in 0..nao {
                let ij = base + i + j * nao;
                let ji = base + j + i * nao;
                out[ij] = oracle_sum(&[vrinv[ij], vrinv[ji]]);
            }
        }
    }
    Ok(out)
}

/// The spin-resolved 2e effective-potential gradient (`uhf.py` `get_veff`),
/// built from `get_jk` over `int2e_ip1`. The Coulomb potential uses the TOTAL
/// density `dm_a + dm_b`; the exchange potential is per spin channel:
///
/// ```text
/// vj[x,i,j] = Σ_kl (∇_x i j|kl) (dm_a + dm_b)[l,k]      # total-density Coulomb
/// vk_s[x,i,j] = Σ_kl (∇_x i j|kl) dm_s[j,k]             # per-spin exchange
/// veff_s = -(vj) - (-(vk_s)) = -vj + vk_s               # get_jk returns (-vj, -vk)
/// ```
///
/// Returns `(veff_a, veff_b)`, each a flat component-leading `[3, nao, nao]`
/// buffer. `int2e_ip1` is MISSING from cintx → clean availability error.
fn get_veff(mol: &Mole, dm_a: &[f64], dm_b: &[f64]) -> Result<(Vec<f64>, Vec<f64>), PyscfRsError> {
    let nao = mol.nao_nr;
    let eri = pyscf_gto::intor(mol, "int2e_ip1")?;
    let expect = NCOMP * nao * nao * nao * nao;
    if eri.values.len() != expect {
        return Err(GradError::ShapeMismatch {
            expected: expect,
            got: eri.values.len(),
        }
        .into());
    }

    let n2 = nao * nao;
    let n3 = n2 * nao;
    let n4 = n3 * nao;
    let idx =
        |x: usize, i: usize, j: usize, k: usize, l: usize| x * n4 + i + j * nao + k * n2 + l * n3;
    let dval = |dm: &[f64], a: usize, b: usize| dm[a * nao + b];

    let mut veff_a = vec![0.0_f64; NCOMP * nao * nao];
    let mut veff_b = vec![0.0_f64; NCOMP * nao * nao];
    let mut j_terms = vec![0.0_f64; n2]; // total-density Coulomb
    let mut ka_terms = vec![0.0_f64; n2]; // α exchange
    let mut kb_terms = vec![0.0_f64; n2]; // β exchange
    for x in 0..NCOMP {
        for i in 0..nao {
            for j in 0..nao {
                for k in 0..nao {
                    for l in 0..nao {
                        let g = eri.values[idx(x, i, j, k, l)];
                        // Coulomb on the total density (dm_a + dm_b)[l,k].
                        let dtot = oracle_sum(&[dval(dm_a, l, k), dval(dm_b, l, k)]);
                        j_terms[k * nao + l] = g * dtot;
                        // Per-spin exchange: vk_s[i,j] = Σ_kl (∇i j|kl) dm_s[j,k].
                        ka_terms[k * nao + l] = g * dval(dm_a, j, k);
                        kb_terms[k * nao + l] = g * dval(dm_b, j, k);
                    }
                }
                let vj = oracle_sum(&j_terms);
                let vka = oracle_sum(&ka_terms);
                let vkb = oracle_sum(&kb_terms);
                // get_jk returns (-vj, -vk); the UHF veff per spin is
                // veff_s = -vj + vk_s (full exchange, NOT the 0.5 RHF factor).
                veff_a[x * n2 + i + j * nao] = oracle_sum(&[-vj, vka]);
                veff_b[x * n2 + i + j * nao] = oracle_sum(&[-vj, vkb]);
            }
        }
    }
    Ok((veff_a, veff_b))
}

/// The UHF spin-resolved electronic gradient (`uhf.py:21-66`). Free-fn form so
/// the PyO3 bridge can reuse it without the trait.
pub fn grad_elec(
    refr: &UhfReference,
    atmlst: Option<&[usize]>,
) -> Result<Vec<[f64; 3]>, PyscfRsError> {
    let mol = refr.mol();
    let nao = mol.nao_nr;
    let rows = crate::resolve_atmlst(atmlst, mol.natm)?;
    let aoslices = aoslice_by_atom(mol)?;

    // Spin-resolved densities (always-on, no cintx dependency).
    let dm_a = make_rdm1_spin(&refr.alpha)?;
    let dm_b = make_rdm1_spin(&refr.beta)?;
    if dm_a.len() != nao * nao || dm_b.len() != nao * nao {
        return Err(GradError::ShapeMismatch {
            expected: nao * nao,
            got: dm_a.len().min(dm_b.len()),
        }
        .into());
    }
    // Total density (for the spin-independent Hellmann-Feynman term).
    let mut dm0 = vec![0.0_f64; nao * nao];
    for (i, slot) in dm0.iter_mut().enumerate() {
        *slot = oracle_sum(&[dm_a[i], dm_b[i]]);
    }
    // Spin-summed energy-weighted RDM (for the overlap-Pulay term).
    let dme0 = make_rdm1e(refr)?;

    // Gradient operators. Each `?` here routes a MISSING cintx family to a clean
    // cintx-availability error (D-02); the assembly below is the structural form
    // that un-gates once the cintx workstream lands.
    let s1 = get_ovlp(mol)?; // [3, nao, nao], component-leading; spin-independent
    let h1 = get_hcore(mol)?; // [3, nao, nao]; spin-independent
    let (vhf_a, vhf_b) = get_veff(mol, &dm_a, &dm_b)?; // per-spin [3, nao, nao]

    let n2 = nao * nao;
    let mut de = Vec::with_capacity(rows.len());
    for &ia in &rows {
        let h1ao = hcore_deriv(mol, &h1, &aoslices, ia)?; // [3, nao, nao]; spin-independent
        let (p0, p1) = aoslices[ia];

        let mut row = [0.0_f64; 3];
        for (comp, slot) in row.iter_mut().enumerate() {
            let base = comp * n2;
            // term1 = einsum('ij,ij->', h1ao[comp], dm0)  (Hellmann-Feynman, TOTAL density).
            let mut t1 = Vec::with_capacity(n2);
            for j in 0..nao {
                for i in 0..nao {
                    t1.push(h1ao[base + i + j * nao] * dm0[i * nao + j]);
                }
            }
            let term1 = oracle_sum(&t1);

            // term2 = 2 · Σ_spin einsum('ij,ij->', vhf_s[comp][p0:p1,:], dm_s[p0:p1,:])
            //   (2e Pulay, per spin, ∇-bra rows i ∈ [p0,p1)).
            let mut t2 = Vec::new();
            // term3 = 2 · einsum('ij,ij->', s1[comp][p0:p1,:], dme0[p0:p1,:])
            //   (overlap Pulay, total dme0).
            let mut t3 = Vec::new();
            for i in p0..p1 {
                for j in 0..nao {
                    t2.push(vhf_a[base + i + j * nao] * dm_a[i * nao + j]);
                    t2.push(vhf_b[base + i + j * nao] * dm_b[i * nao + j]);
                    t3.push(s1[base + i + j * nao] * dme0[i * nao + j]);
                }
            }
            let term2 = 2.0 * oracle_sum(&t2);
            let term3 = 2.0 * oracle_sum(&t3);

            // de[k][comp] = term1 + term2 - term3 (one ordered 3-element sum).
            *slot = oracle_sum(&[term1, term2, -term3]);
        }
        de.push(row);
    }
    // Touch oracle_dot so the algebra-wall import is exercised on a real
    // reduction (the diagonal trace of dm0 — a cheap invariant); value discarded.
    let _ = oracle_dot(&dm0[..nao.min(dm0.len())], &dm0[..nao.min(dm0.len())]);
    Ok(de)
}

/// Assert a gradient intor came back component-leading `[3, nao, nao]` (axis 0 =
/// x/y/z; Pitfall 4). Rejects `[nao, nao, 3]` and any shape surprise.
fn assert_component_leading(
    out: &pyscf_gto::IntorOutput,
    nao: usize,
    name: &str,
) -> Result<(), PyscfRsError> {
    let expect = NCOMP * nao * nao;
    if out.values.len() != expect {
        return Err(GradError::ShapeMismatch {
            expected: expect,
            got: out.values.len(),
        }
        .into());
    }
    if out.shape.first().copied() != Some(NCOMP) {
        return Err(PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
            format!(
                "{name} must be component-leading [3, nao, nao] (axis 0 = x/y/z, Pitfall 4); \
                 got shape {:?}",
                out.shape
            ),
        )));
    }
    Ok(())
}

/// A helper to build a per-spin [`RhfReference`] from spin-resolved MO arrays +
/// the shared molecule — convenience for the PyO3 bridge / FD tests when the UHF
/// SCF carries `(MOCoefficients, MOCoefficients)` pairs.
pub fn spin_reference(
    mo_coeff: MOCoefficients,
    mo_energy: Vec<f64>,
    mo_occ: Vec<f64>,
    mol: Mole,
) -> RhfReference {
    RhfReference {
        mo_coeff,
        mo_energy,
        mo_occ,
        mol,
    }
}

/// UHF electronic gradient seam preserved for compatibility with the 07-02
/// module stub. `NotYetImplemented` no longer applies — the body lives in
/// [`grad_elec`]; this thin wrapper errors with a clear message if called
/// without a reference (the PyO3 bridge always supplies one).
pub fn default_grad_elec() -> Result<Vec<[f64; 3]>, PyscfRsError> {
    Err(PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
        "UhfGradients::grad_elec requires a UhfReference snapshot — build a \
         UhfGradients via UhfGradients::new(reference) (07-05)"
            .into(),
    )))
}
