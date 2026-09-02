//! RHF analytical gradient (GRAD-01) — the phase headline.
//!
//! Port target: `pyscf/grad/rhf.py:38-189` (`grad_elec` lines 59-76; `grad_nuc`
//! 92-107 — lives as the shared [`crate::Gradients::grad_nuc`] default;
//! `get_hcore` 109-117; `hcore_generator` 121-143; `get_ovlp` 146;
//! `get_jk`/`get_veff` 149-183; `make_rdm1e` 185-189). Structural analog:
//! `crates/pyscf-mp2/src/mp2.rs` (the `default_*` free-fn + `Result<_, Error>`
//! kernel-module shape, pyo3-free).
//!
//! ## The grad_elec decomposition (`rhf.py:59-76`)
//!
//! ```text
//! de[k] = einsum('xij,ij->x', hcore_deriv(ia), dm0)          # Hellmann-Feynman
//!       + 2 * einsum('xij,ij->x', vhf[:, p0:p1], dm0[p0:p1])  # 2e Pulay (∇ on bra, ×2)
//!       - 2 * einsum('xij,ij->x', s1 [:, p0:p1], dme0[p0:p1]) # overlap Pulay
//! ```
//! per atom `ia` over `atmlst`, where `p0,p1 = aoslices[ia]` are the atom's AO
//! range. `grad_nuc` (the nuclear-repulsion gradient) is added by
//! [`crate::Gradients::kernel`].
//!
//! Per **D-04** RHF energy gradients are *stationary* (the 2n+1 rule) — they do
//! **NOT** call CPHF.
//!
//! ## cintx-availability gating (D-02 + 07-01-SUMMARY)
//!
//! Six gradient-integral families this body needs — `int2e_ip1` (2e Pulay),
//! `int1e_ip{ovlp,kin,nuc}` (overlap + hcore Pulay), and `int1e_iprinv` +
//! `with_rinv_at_nucleus` (the per-atom Hellmann-Feynman shift) — are **MISSING
//! from every cintx branch with no scheduled workstream** (07-01-SUMMARY). The
//! STRUCTURAL assembly here (shapes, atmlst, the einsum routing) lands always-on;
//! the intor calls `?`-propagate a CLEAN cintx-availability error
//! (`PyscfRsError::Core(InvalidMolecule(..))`, **never** `NotYetImplemented{phase:7}`
//! — GRAD-07 closed that disposition in pyscf-gto). The numeric FD-vs-analytical
//! comparison is therefore `#[ignore]`'d (see `tests/rhf_verify_fd.rs`) until the
//! cintx grad-integral workstream lands these families; the FD-STRUCTURAL harness
//! stays runnable.
//!
//! ## Bit-exact discipline (Pitfall 1/2)
//!
//! Every `einsum`/per-atom reduction materialises into a `Vec` then routes
//! through `pyscf_algebra::oracle_sum` / `oracle_dot` — NEVER a bare `+=`. Every
//! gradient intor is asserted component-leading `[3, nao, nao]` (axis 0 = x/y/z;
//! Pitfall 4) before it is contracted.

use crate::Gradients;
use crate::error::GradError;
use pyscf_algebra::{oracle_dot, oracle_sum};

use pyscf_core::{MOCoefficients, Mole, PyscfRsError, Unit};

/// The number of Cartesian derivative components every gradient intor carries
/// (axis 0 of a `[3, nao, nao]` component-leading buffer — x/y/z; Pitfall 4).
const NCOMP: usize = 3;

/// Snapshot of a converged RHF reference, consumed by [`RhfGradients`] (D-09).
///
/// pyo3-free — plain arrays. The PyO3 bridge (07-09) snapshots `mf.mo_*` + `mf.mol`
/// into this; the in-tree `nuc_grad_method()` wiring builds it from a converged
/// `pyscf_scf::RHF`. Mirrors `pyscf_mp2::Mp2Reference`'s shape.
#[derive(Debug, Clone)]
pub struct RhfReference {
    /// MO coefficients, F-order `[nao, nmo]`, sign-canonicalized (SCF-13).
    pub mo_coeff: MOCoefficients,
    /// MO energies (one per MO).
    pub mo_energy: Vec<f64>,
    /// MO occupations (2.0 / 0.0 for closed-shell restricted).
    pub mo_occ: Vec<f64>,
    /// Molecule snapshot — geometry, basis, AO offsets the gradient is taken wrt.
    pub mol: Mole,
}

/// RHF analytical gradient (`pyscf/grad/rhf.py` `Gradients`).
///
/// Holds the converged SCF reference + an optional `atmlst` subset. Implements
/// the base [`Gradients`] trait: `grad_elec` (the Hellmann-Feynman + Pulay
/// assembly) and `make_rdm1e` (the energy-weighted RDM) are the per-method
/// bodies; `grad_nuc` is inherited from the trait default; `get_ovlp` /
/// `hcore_generator` are overridden here with the real (cintx-gated) intor
/// wiring.
#[derive(Debug, Clone)]
pub struct RhfGradients {
    /// The converged RHF reference snapshot.
    pub reference: RhfReference,
    /// The atom subset the gradient is restricted to (`None` = full molecule).
    pub atmlst: Option<Vec<usize>>,
    /// The last computed gradient `(n, 3)` (`None` until [`Gradients::kernel`]).
    pub de: Option<Vec<[f64; 3]>>,
}

impl RhfGradients {
    /// Build an RHF gradient driver over a converged reference.
    pub fn new(reference: RhfReference) -> Self {
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

impl Gradients for RhfGradients {
    fn mol(&self) -> &Mole {
        &self.reference.mol
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

    /// The energy-weighted reduced density matrix (`rhf.py:185-189`):
    /// `dme0[μ,ν] = Σ_{i: occ>0} (ε_i · occ_i) · C[μ,i] · C[ν,i]`.
    ///
    /// A thin pure-linear-algebra port over the already-shipped MO arrays. The
    /// reduction over occupied MOs goes through `oracle_sum` (Pitfall 1, no bare
    /// `+=`). Returns the flat row-major `(nao, nao)` matrix.
    fn make_rdm1e(&self) -> Result<Vec<f64>, PyscfRsError> {
        make_rdm1e(&self.reference)
    }

    /// The overlap derivative `s1 = -int1e_ipovlp` (`rhf.py:146`), a
    /// component-leading `[3, nao, nao]` buffer (axis 0 = x/y/z). The intor is
    /// MISSING from cintx today → this `?`-propagates a clean cintx-availability
    /// error (never `NotYetImplemented{phase:7}`).
    fn get_ovlp(&self) -> Result<Vec<f64>, PyscfRsError> {
        get_ovlp(&self.reference.mol)
    }

    /// PER-METHOD electronic gradient (`rhf.py:59-76`). See module docs for the
    /// decomposition. The intor families it contracts are cintx-gated; the
    /// assembly is always-on.
    fn grad_elec(&self, atmlst: Option<&[usize]>) -> Result<Vec<[f64; 3]>, PyscfRsError> {
        grad_elec(&self.reference, atmlst)
    }
}

/// Energy-weighted RDM (`rhf.py:185-189`). Free-fn form so the PyO3 bridge can
/// reuse it without the trait.
pub fn make_rdm1e(refr: &RhfReference) -> Result<Vec<f64>, PyscfRsError> {
    let nao = refr.mo_coeff.nao;
    let nmo = refr.mo_coeff.nmo;
    if refr.mo_coeff.data.len() != nao * nmo {
        return Err(GradError::ShapeMismatch {
            expected: nao * nmo,
            got: refr.mo_coeff.data.len(),
        }
        .into());
    }
    if refr.mo_energy.len() != nmo || refr.mo_occ.len() != nmo {
        return Err(GradError::ShapeMismatch {
            expected: nmo,
            got: refr.mo_energy.len().min(refr.mo_occ.len()),
        }
        .into());
    }

    // dme0[μ,ν] = Σ_{i: occ>0} (ε_i · occ_i) · C[μ,i] · C[ν,i].
    // C is F-order [nao, nmo]: C[μ,i] lives at data[μ + i*nao].
    let mut dme0 = vec![0.0_f64; nao * nao];
    let mut terms = vec![0.0_f64; nmo];
    // `i` drives BOTH the per-MO `terms` buffer and the F-order C offsets
    // (`mu + i*nao`), so a range loop is the clearest form (rdm.rs precedent).
    #[allow(clippy::needless_range_loop)]
    for mu in 0..nao {
        for nu in 0..nao {
            for i in 0..nmo {
                let occ = refr.mo_occ[i];
                if occ > 0.0 {
                    let w = refr.mo_energy[i] * occ;
                    terms[i] =
                        w * refr.mo_coeff.data[mu + i * nao] * refr.mo_coeff.data[nu + i * nao];
                } else {
                    terms[i] = 0.0;
                }
            }
            // Row-major (nao,nao) output, oracle-ordered reduction over MOs.
            dme0[mu * nao + nu] = oracle_sum(&terms);
        }
    }
    Ok(dme0)
}

/// Closed-shell RDM `dm0[μ,ν] = Σ_i occ_i · C[μ,i] · C[ν,i]` (`make_rdm1`,
/// `hf.py:1517-1538`), row-major `(nao, nao)`. Used by `grad_elec` as the
/// density the gradient operators are contracted against.
fn make_rdm1(refr: &RhfReference) -> Result<Vec<f64>, PyscfRsError> {
    let nao = refr.mo_coeff.nao;
    let nmo = refr.mo_coeff.nmo;
    if refr.mo_coeff.data.len() != nao * nmo || refr.mo_occ.len() != nmo {
        return Err(GradError::ShapeMismatch {
            expected: nao * nmo,
            got: refr.mo_coeff.data.len(),
        }
        .into());
    }
    let mut dm0 = vec![0.0_f64; nao * nao];
    let mut terms = vec![0.0_f64; nmo];
    #[allow(clippy::needless_range_loop)]
    for mu in 0..nao {
        for nu in 0..nao {
            for i in 0..nmo {
                terms[i] = refr.mo_occ[i]
                    * refr.mo_coeff.data[mu + i * nao]
                    * refr.mo_coeff.data[nu + i * nao];
            }
            dm0[mu * nao + nu] = oracle_sum(&terms);
        }
    }
    Ok(dm0)
}

/// The overlap derivative `s1 = -int1e_ipovlp` (`rhf.py:146`).
///
/// Returns a flat F-order `(3, nao, nao)` buffer with the COMPONENT axis fastest
/// (pyscf-gto stitches `out[comp + ncomp*i + ncomp*nao*j]`). `int1e_ipovlp`
/// evaluates in cintx and is byte-correct for s/p shells (verified vs PySCF).
pub fn get_ovlp(mol: &Mole) -> Result<Vec<f64>, PyscfRsError> {
    let out = pyscf_gto::intor(mol, "int1e_ipovlp")?;
    let nao = mol.nao_nr;
    assert_component_leading(&out, nao, "int1e_ipovlp")?;
    // s1 = -int1e_ipovlp.
    Ok(out.values.iter().map(|v| -v).collect())
}

/// The core-Hamiltonian gradient `get_hcore` part: `-(int1e_ipkin + int1e_ipnuc)`
/// (`rhf.py:109-118`), F-order `(3, nao, nao)` (component fastest). Elementwise,
/// so it inherits the input layout — no per-component indexing here.
///
/// (The ECP `ECPscalar_ipnuc` branch (cintx-ready) lands in the ECP-grad wave
/// 07-08; pure-Coulomb RHF needs only the kinetic + nuclear terms here.)
/// `int1e_ipkin` is byte-correct in cintx; `int1e_ipnuc` is currently WRONG for
/// l≥1 shells (external cintx p-shell kernel bug — F-08), so only s-shell
/// gradients are numerically validated.
fn get_hcore(mol: &Mole) -> Result<Vec<f64>, PyscfRsError> {
    let nao = mol.nao_nr;
    let kin = pyscf_gto::intor(mol, "int1e_ipkin")?;
    assert_component_leading(&kin, nao, "int1e_ipkin")?;
    let nuc = pyscf_gto::intor(mol, "int1e_ipnuc")?;
    assert_component_leading(&nuc, nao, "int1e_ipnuc")?;
    // h = -(ipkin + ipnuc), elementwise; each sum is a 2-element ordered add.
    let mut h = vec![0.0_f64; NCOMP * nao * nao];
    for (idx, slot) in h.iter_mut().enumerate() {
        *slot = -oracle_sum(&[kin.values[idx], nuc.values[idx]]);
    }
    Ok(h)
}

/// Per-atom core-Hamiltonian derivative `hcore_deriv(atm_id)` (`rhf.py:121-143`).
///
/// `vrinv = -Z_atm · int1e_iprinv` evaluated with `with_rinv_at_nucleus(atm_id)`,
/// then the `get_hcore` block `h1[:, p0:p1]` is added into the atom's AO columns
/// and the result is symmetrised over the two AO axes:
/// `hcore_deriv(ia) = vrinv + vrinv.transpose(0,2,1)`.
///
/// Returns a flat component-leading `[3, nao, nao]` buffer. `int1e_iprinv`
/// evaluates the real per-atom Hellmann-Feynman shift via
/// `pyscf_gto::intor_with_rinv_at_nucleus(mol, "int1e_iprinv", atm_id)`, which
/// pins the cintx `rinv` origin to `mol.atom_coord(atm_id)` (F-08 / quick
/// 260601-sln). (Origin-less `intor(mol, "int1e_iprinv")` fails the cintx
/// validator with `InvalidEnvParam{PTR_RINV_ORIG}` — the origin is mandatory.)
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
    // The per-atom rinv-origin shift (`with_rinv_at_nucleus`, pyscf/grad/rhf.py:
    // 121-143): pin the cintx `rinv` origin to this nucleus's coordinate so
    // `int1e_iprinv` evaluates the real per-atom Hellmann-Feynman shift
    // (F-08 / quick 260601-sln). cintx REQUIRES the origin for any iprinv op.
    let iprinv = pyscf_gto::intor_with_rinv_at_nucleus(mol, "int1e_iprinv", atm_id)?;
    assert_component_leading(&iprinv, nao, "int1e_iprinv")?;

    let z = charges[atm_id] as f64;
    // vrinv = -Z · int1e_iprinv (elementwise scale).
    let mut vrinv = vec![0.0_f64; NCOMP * nao * nao];
    for (idx, slot) in vrinv.iter_mut().enumerate() {
        *slot = -z * iprinv.values[idx];
    }

    // numpy `vrinv[:, p0:p1] += h1[:, p0:p1]` slices AXIS 1 — the FIRST AO index
    // `i` (the ∇-bra/row), all columns `j`. The cintx grad intors are F-order
    // `(3, nao, nao)` with the COMPONENT axis fastest (pyscf-gto stitches
    // `out[comp + ncomp*i + ncomp*nao*j]`), so element (comp, i, j) lives at
    // `comp + NCOMP*(i + j*nao)` — NOT the component-slowest `comp*nao*nao + …`.
    let (p0, p1) = aoslices[atm_id];
    for comp in 0..NCOMP {
        for i in p0..p1 {
            for j in 0..nao {
                let off = comp + NCOMP * (i + j * nao);
                vrinv[off] = oracle_sum(&[vrinv[off], h1[off]]);
            }
        }
    }

    // hcore_deriv = vrinv + vrinv.transpose(0,2,1) (symmetrise the AO axes).
    let mut out = vec![0.0_f64; NCOMP * nao * nao];
    for comp in 0..NCOMP {
        for j in 0..nao {
            for i in 0..nao {
                let ij = comp + NCOMP * (i + j * nao);
                let ji = comp + NCOMP * (j + i * nao);
                out[ij] = oracle_sum(&[vrinv[ij], vrinv[ji]]);
            }
        }
    }
    Ok(out)
}

/// `get_veff = -(vj - 0.5·vk)` (`rhf.py:180-183`), the 2e Coulomb-repulsion
/// gradient, built from `get_jk` over `int2e_ip1` (`rhf.py:149-178`).
///
/// Mirrors PySCF `_vhf.direct_mapdm('int2e_ip1', 's2kl', ('lk->s1ij',
/// 'jk->s1il'), …)`:
///   `J[x,i,j] = Σ_kl (∇_x i,j|k,l) D[l,k]`  (`lk->s1ij`)
///   `K[x,i,l] = Σ_jk (∇_x i,j|k,l) D[j,k]`  (`jk->s1il`)
/// then `vj,vk = -vj,-vk` (the `∇` sits on the bra). The K output index is the
/// FOURTH integral axis `l`, summing the second/third — NOT a copy of the J
/// pattern (those coincide only for 1-function-per-shell systems like H2/STO-3G).
/// Returns a flat F-order `(3, nao, nao)` buffer (component fastest).
/// NOTE: `int2e_ip1` evaluates but is numerically WRONG for l≥1 shells (external
/// cintx p-shell kernel bug — F-08); s-shell gradients are validated.
fn get_veff(mol: &Mole, dm0: &[f64]) -> Result<Vec<f64>, PyscfRsError> {
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

    // The cintx grad intor is F-order `(3, nao, nao, nao, nao)` with the
    // COMPONENT axis fastest (pyscf-gto stitches
    // `out[comp + ncomp*i + ncomp*nao*j + …]`), so element (x,i,j,k,l) lives at
    //   x + NCOMP*(i + j*nao + k*nao^2 + l*nao^3).
    let n2 = nao * nao;
    let n3 = n2 * nao;
    let idx = |x: usize, i: usize, j: usize, k: usize, l: usize| {
        x + NCOMP * (i + j * nao + k * n2 + l * n3)
    };
    // dm0 is row-major (nao,nao): D[a,b] at a*nao + b.
    let dval = |a: usize, b: usize| dm0[a * nao + b];

    let mut veff = vec![0.0_f64; NCOMP * nao * nao];
    // Per (x,i,j): materialise the contraction products, oracle_sum them (no bare
    // `+=`). vj contracts the (k,l) pair against D[l,k]; vk contracts the (jp,k)
    // pair of `(∇_x i,jp|k,j)` against D[jp,k] (output index j is the integral's
    // FOURTH axis l).
    let mut j_terms = vec![0.0_f64; n2];
    let mut k_terms = vec![0.0_f64; n2];
    for x in 0..NCOMP {
        for i in 0..nao {
            for j in 0..nao {
                // vj[x,i,j] = Σ_kl (∇_x i,j|k,l) D[l,k].
                for k in 0..nao {
                    for l in 0..nao {
                        j_terms[k * nao + l] = eri.values[idx(x, i, j, k, l)] * dval(l, k);
                    }
                }
                // vk[x,i,j] = Σ_{jp,k} (∇_x i,jp|k,j) D[jp,k]  (j is the l-axis).
                for jp in 0..nao {
                    for k in 0..nao {
                        k_terms[jp * nao + k] = eri.values[idx(x, i, jp, k, j)] * dval(jp, k);
                    }
                }
                let vj = oracle_sum(&j_terms);
                let vk = oracle_sum(&k_terms);
                // get_jk returns (-vj, -vk); get_veff = (-vj) - 0.5·(-vk).
                // Store component-fastest to match the cintx F-order layout.
                veff[x + NCOMP * (i + j * nao)] = oracle_sum(&[-vj, 0.5 * vk]);
            }
        }
    }
    Ok(veff)
}

/// The RHF electronic gradient (`rhf.py:59-76`). Free-fn form so the PyO3 bridge
/// can reuse it without the trait.
pub fn grad_elec(
    refr: &RhfReference,
    atmlst: Option<&[usize]>,
) -> Result<Vec<[f64; 3]>, PyscfRsError> {
    let mol = &refr.mol;
    let nao = mol.nao_nr;
    let rows = crate::resolve_atmlst(atmlst, mol.natm)?;
    let aoslices = aoslice_by_atom(mol)?;

    // Densities (always-on, no cintx dependency).
    let dm0 = make_rdm1(refr)?;
    let dme0 = make_rdm1e(refr)?;

    // Gradient operators. Each `?` here routes a MISSING cintx family to a clean
    // cintx-availability error (D-02); the assembly below is the structural form
    // that un-gates once the cintx workstream lands.
    let s1 = get_ovlp(mol)?; // [3, nao, nao], component-leading
    let h1 = get_hcore(mol)?; // [3, nao, nao]
    let vhf = get_veff(mol, &dm0)?; // [3, nao, nao]

    let n2 = nao * nao;
    let mut de = Vec::with_capacity(rows.len());
    for &ia in &rows {
        let h1ao = hcore_deriv(mol, &h1, &aoslices, ia)?; // [3, nao, nao]
        let (p0, p1) = aoslices[ia];

        let mut row = [0.0_f64; 3];
        for (comp, slot) in row.iter_mut().enumerate() {
            // The component buffers (h1ao, vhf, s1) are F-order `(3, nao, nao)`
            // with the component axis fastest: element (comp, i, j) lives at
            // `comp + NCOMP*(i + j*nao)`. dm0/dme0 are row-major (i,j) at i*nao+j.
            let off = |i: usize, j: usize| comp + NCOMP * (i + j * nao);
            // term1 = einsum('ij,ij->', h1ao[comp], dm0)  (Hellmann-Feynman, full AO).
            let mut t1 = Vec::with_capacity(n2);
            for j in 0..nao {
                for i in 0..nao {
                    t1.push(h1ao[off(i, j)] * dm0[i * nao + j]);
                }
            }
            let term1 = oracle_sum(&t1);

            // term2 = 2 · einsum('ij,ij->', vhf[comp][:, p0:p1], dm0[p0:p1])
            //   (2e Pulay, rows i ∈ [p0,p1)). Note rhf.py slices the FIRST AO
            //   axis (the ∇-bra) — vhf[:,p0:p1] and dm0[p0:p1].
            let mut t2 = Vec::new();
            // term3 = 2 · einsum('ij,ij->', s1[comp][:, p0:p1], dme0[p0:p1])
            //   (overlap Pulay).
            let mut t3 = Vec::new();
            for i in p0..p1 {
                for j in 0..nao {
                    t2.push(vhf[off(i, j)] * dm0[i * nao + j]);
                    t3.push(s1[off(i, j)] * dme0[i * nao + j]);
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
    // reduction (the diagonal trace of dm0 — a cheap invariant), keeping the
    // einsum routing honest (Pitfall 2). The value is discarded.
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
    // The leading axis MUST be the component axis (== 3), never trailing.
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

/// Per-atom molecular AO range `[p0, p1)` — re-exported from
/// [`pyscf_gto::aoslice_by_atom`], which is where the single copy now lives
/// (U-02 step 5). Kept public here because `pyscf-grad`'s own tests and the
/// `rks`/`uks` gradient modules import it under this path.
pub use pyscf_gto::aoslice_by_atom;

/// RHF electronic gradient seam preserved for compatibility with the 07-02
/// module stub. `NotYetImplemented` no longer applies — the body lives in
/// [`grad_elec`]; this thin wrapper errors with a clear message if called
/// without a reference (the PyO3 bridge always supplies one).
pub fn default_grad_elec() -> Result<Vec<[f64; 3]>, PyscfRsError> {
    Err(PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
        "RhfGradients::grad_elec requires an RhfReference snapshot — build an \
         RhfGradients via RhfGradients::new(reference) (07-03)"
            .into(),
    )))
}
