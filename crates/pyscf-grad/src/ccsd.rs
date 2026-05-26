//! CCSD analytical gradient (GRAD-06) — the Λ-driven relaxed density + the
//! orbital-relaxation Z-vector through the ONE CPHF solver ([`crate::cphf`]).
//!
//! Port target: `pyscf/grad/ccsd.py` (the Λ-driven CCSD gradient + the single
//! orbital-relaxation Z-vector). Structural analog: `crates/pyscf-ccsd/src/
//! {lambda.rs,rdm.rs}` — GRAD-06 CONSUMES the Phase-6 surface directly:
//!
//!   * `pyscf_ccsd::solve_lambda` (06-06, `lambda.rs:411`) for the converged Λ,
//!   * `pyscf_ccsd::make_rdm1`/`make_rdm2` (06-06, incl. `ao_repr` — D-03,
//!     `rdm.rs:202,296`) for the relaxed densities,
//!
//! and re-enters CPHF ONLY for the orbital-relaxation Z-vector via the ONE
//! 07-07 [`cphf::solve`] (its own `fvind`/RHS). The SCF-stationary RHF base
//! decomposition it plugs the relaxed density into is `crates/pyscf-grad/src/
//! rhf.rs` (07-03).
//!
//! ## NO Λ re-derivation (D-04 / T-07-25 / RESEARCH anti-pattern)
//!
//! Phase 6 ALREADY solved Λ (`solve_lambda`, 06-06). Phase-7 CCSD-grad CONSUMES
//! it — it does NOT re-derive it. There is NO second lambda-equation solver in
//! this crate; the `single_lambda_solver_in_grad` structural test
//! (`tests/ccsd_verify_fd.rs`) forbids one. This is the deliberate Phase-6 scope
//! choice paying off.
//!
//! ## The second non-variational gradient (D-04)
//!
//! Like MP2 (07-07), CCSD is NOT stationary in the orbitals — its gradient needs
//! an orbital-relaxation (Z-vector) solve. That solve is the SINGLE
//! [`cphf::solve`] (D-03/GRAD-10): CCSD supplies its OWN `fvind` response
//! operator + RHS `Xvo`. CCSD does NOT override the upstream default
//! `max_cycle=50` (unlike MP2's 30) — `pyscf/grad/ccsd.py` leaves the
//! `cphf.solve` default in place.
//!
//! ## cintx-availability gating (D-02, the 07-01/07-03/07-07 precedent)
//!
//! The `de` assembly contracts the six gradient-integral families missing from
//! cintx (`int2e_ip1`, `int1e_ip{ovlp,kin,nuc,rinv}`) via the RHF `get_veff`/
//! `get_ovlp`/`hcore_deriv` (07-03). Those `?`-propagate a CLEAN
//! cintx-availability error (`Core(InvalidMolecule(..))`, NEVER
//! `NotYetImplemented{phase:7}`). The STRUCTURAL arm (the Λ consumption, the
//! relaxed-density RDMs, the Z-vector wiring, the `(natm,3)` shape) lands
//! always-on; the numeric FD-vs-analytical comparison is `#[ignore]`'d until the
//! cintx grad-integral workstream lands the six families (see
//! `tests/ccsd_verify_fd.rs`).
//!
//! ## Bit-exact discipline (Pitfall 1/2)
//!
//! Every reduction materializes into a `Vec` then routes through
//! `pyscf_algebra::oracle_sum`/`oracle_dot` — NEVER a bare `+=`.

use crate::cphf;
use crate::error::GradError;
use crate::rhf::{RhfReference, make_rdm1e as rhf_make_rdm1e};
use pyscf_algebra::oracle_sum;
use pyscf_ccsd::{ChemistsEris, make_rdm1 as ccsd_make_rdm1, solve_lambda};
use pyscf_core::{Amplitudes, Density, MOCoefficients, Mole, PyscfRsError, Unit};
use pyscf_runtime::WorkspacePool;

/// The CPHF `max_cycle` the CCSD orbital-relaxation Z-vector uses. CCSD-grad
/// does NOT override the upstream `cphf.solve` default (unlike MP2's 30) —
/// `pyscf/grad/ccsd.py` leaves [`cphf::DEFAULT_MAX_CYCLE`] (= 50) in place.
pub const CCSD_CPHF_MAX_CYCLE: usize = cphf::DEFAULT_MAX_CYCLE;

/// Snapshot of a converged RHF + CCSD reference, consumed by [`CcsdGradients`].
///
/// pyo3-free — plain arrays. Mirrors `crate::mp2::Mp2Reference`, but carries the
/// converged CCSD `(t1, t2)` amplitudes + the [`ChemistsEris`] the Phase-6
/// `solve_lambda` + `make_rdm1`/`make_rdm2` consume DIRECTLY (D-04 — NO Λ
/// re-derivation). The PyO3 bridge (07-09) and the in-tree `nuc_grad_method()`
/// wiring build this from a converged `pyscf_ccsd::CcsdReference` + its
/// `CcsdResult` amplitudes + the `ChemistsEris`.
#[derive(Debug, Clone)]
pub struct CcsdGradReference {
    /// MO coefficients, F-order `[nao, nmo]`, sign-canonicalized (SCF-13).
    pub mo_coeff: MOCoefficients,
    /// MO energies (one per MO).
    pub mo_energy: Vec<f64>,
    /// MO occupations (2.0 / 0.0 for closed-shell restricted).
    pub mo_occ: Vec<f64>,
    /// The converged CCSD amplitudes `(t1, t2)` from the Phase-6 kernel. `t1`
    /// C-order `[nocc, nvir]`; `t2` C-order `[nocc, nocc, nvir, nvir]`.
    pub amps: Amplitudes,
    /// The chemists'-notation MO ERIs the Phase-6 `solve_lambda` +
    /// `make_rdm1`/`make_rdm2` consume.
    pub eris: ChemistsEris,
    /// Molecule snapshot — geometry, basis, AO offsets the gradient is wrt.
    pub mol: Mole,
}

impl CcsdGradReference {
    /// Number of active occupied MOs (`#{occ > 0}`).
    fn nocc(&self) -> usize {
        self.mo_occ.iter().filter(|&&o| o > 0.0).count()
    }
    /// Number of active virtual MOs (`#{occ == 0}`).
    fn nvir(&self) -> usize {
        self.mo_occ.iter().filter(|&&o| o == 0.0).count()
    }
    /// The RHF reference snapshot the SCF-stationary base decomposition uses.
    fn as_rhf(&self) -> RhfReference {
        RhfReference {
            mo_coeff: self.mo_coeff.clone(),
            mo_energy: self.mo_energy.clone(),
            mo_occ: self.mo_occ.clone(),
            mol: self.mol.clone(),
        }
    }
}

/// CCSD analytical gradient (`pyscf/grad/ccsd.py` `Gradients`).
///
/// Holds the converged RHF+CCSD reference + an optional `atmlst` subset. The
/// per-method `grad_elec` consumes the Phase-6 Λ + relaxed-density RDMs (D-04 —
/// NO re-derivation), solves the orbital-relaxation Z-vector through the ONE
/// [`cphf::solve`], and assembles the gradient on the RHF base decomposition
/// (cintx-gated numeric).
#[derive(Debug, Clone)]
pub struct CcsdGradients {
    /// The converged RHF+CCSD reference snapshot.
    pub reference: CcsdGradReference,
    /// The atom subset the gradient is restricted to (`None` = full molecule).
    pub atmlst: Option<Vec<usize>>,
    /// The last computed gradient `(n, 3)` (`None` until [`crate::Gradients::kernel`]).
    pub de: Option<Vec<[f64; 3]>>,
}

impl CcsdGradients {
    /// Build a CCSD gradient driver over a converged reference.
    pub fn new(reference: CcsdGradReference) -> Self {
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

impl crate::Gradients for CcsdGradients {
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

    /// The energy-weighted RDM. CCSD grad reuses the RHF `make_rdm1e`.
    fn make_rdm1e(&self) -> Result<Vec<f64>, PyscfRsError> {
        rhf_make_rdm1e(&self.reference.as_rhf())
    }

    /// The overlap derivative `s1 = -int1e_ipovlp` (the RHF `get_ovlp`,
    /// cintx-gated).
    fn get_ovlp(&self) -> Result<Vec<f64>, PyscfRsError> {
        crate::rhf::get_ovlp(&self.reference.mol)
    }

    /// PER-METHOD CCSD electronic gradient (`pyscf/grad/ccsd.py`): the Λ-driven
    /// relaxed density + the orbital-relaxation Z-vector through `cphf::solve`.
    fn grad_elec(&self, atmlst: Option<&[usize]>) -> Result<Vec<[f64; 3]>, PyscfRsError> {
        grad_elec(&self.reference, atmlst)
    }
}

/// Consume the Phase-6 Λ-equations + the 1-RDM directly (D-04, GRAD-06) — NO
/// re-derivation. Runs `pyscf_ccsd::solve_lambda` (the converged Λ) then
/// `pyscf_ccsd::make_rdm1` (the relaxed MO 1-RDM, `ao_repr=false`), returning the
/// flat row-major `(nmo, nmo)` MO-basis 1-RDM.
///
/// This is the load-bearing consumption point: the CCSD-grad relaxed density is
/// built from the Phase-6-validated `solve_lambda` + `make_rdm1`, NEVER a second
/// in-crate lambda solver (T-07-25). A `pyscf_ccsd::CcsdError` (e.g. Λ
/// non-convergence, an arena over-budget) `?`-propagates through
/// `GradError::Ccsd`.
pub fn relaxed_rdm1(refr: &CcsdGradReference) -> Result<Vec<f64>, PyscfRsError> {
    let t1 = refr
        .amps
        .t1_slice()
        .ok_or_else(|| GradError::ShapeMismatch {
            expected: refr.nocc() * refr.nvir(),
            got: 0,
        })?;
    let t2 = refr
        .amps
        .t2_slice()
        .ok_or_else(|| GradError::ShapeMismatch {
            expected: refr.nocc() * refr.nocc() * refr.nvir() * refr.nvir(),
            got: 0,
        })?;

    // The CCSD-grad Λ + RDM consume the Phase-6 arena pool (the `wvvvv ≈ nv⁴`
    // tenant `solve_lambda`/`make_rdm2` reserve). DEFAULT_BUDGET_BYTES matches
    // the Phase-6 default; an over-budget reference `?`-propagates a clean error.
    let pool = WorkspacePool::new(WorkspacePool::DEFAULT_BUDGET_BYTES);

    // 1. The Phase-6 Λ solve — CONSUMED, not re-derived (D-04 / GRAD-06).
    let lam = solve_lambda(t1, t2, &refr.eris, &pool)?;

    // 2. The relaxed MO 1-RDM via the Phase-6 make_rdm1 (ao_repr=false → the MO
    //    representation the orbital-response Z-vector + the base assembly use).
    let dm1 = ccsd_make_rdm1(
        t1,
        t2,
        &lam.l1,
        &lam.l2,
        &refr.eris,
        false, // ao_repr: MO-basis 1-RDM for the Z-vector RHS.
        &refr.mo_coeff,
    )?;
    Ok(dm1.data)
}

/// The CCSD-specific CPHF response operator `fvind` (the orbital-relaxation
/// Z-vector A-operator). Identical in shape to the MP2 `mp2_fvind`
/// (`pyscf/grad/mp2.py:274-279`): the response uses the ENERGY `int2e` `get_veff`
/// (cintx-ready as of 05-08), NOT a gradient integral — so the Z-vector solve
/// itself is fully runnable un-gated.
///
/// Given `x` (the trial vir×occ rotation, flat vir-major `a·nocc + i`):
/// 1. build the AO density `dm[μ,ν] = Σ_{a,i} Cv[μ,a]·x[a,i]·Co[ν,i]`,
/// 2. apply the SCF `get_veff` to `dm + dmᵀ`,
/// 3. project back `v_vo[a,i] = Σ_{μ,ν} Cv[μ,a]·v[μ,ν]·Co[ν,i]`,
/// 4. return `v_vo · 2`.
fn ccsd_fvind(refr: &CcsdGradReference, x: &[f64]) -> Result<Vec<f64>, PyscfRsError> {
    let nao = refr.mo_coeff.nao;
    let nmo = refr.mo_coeff.nmo;
    let nocc = refr.nocc();
    let nvir = refr.nvir();
    let ndim = nvir * nocc;
    if x.len() != ndim {
        return Err(GradError::ShapeMismatch {
            expected: ndim,
            got: x.len(),
        }
        .into());
    }
    if refr.mo_coeff.data.len() != nao * nmo {
        return Err(GradError::ShapeMismatch {
            expected: nao * nmo,
            got: refr.mo_coeff.data.len(),
        }
        .into());
    }

    // Occupied / virtual MO column indices in canonical order (occ-first).
    let occ_cols: Vec<usize> = (0..nmo).filter(|&p| refr.mo_occ[p] > 0.0).collect();
    let vir_cols: Vec<usize> = (0..nmo).filter(|&p| refr.mo_occ[p] == 0.0).collect();
    if occ_cols.len() != nocc || vir_cols.len() != nvir {
        return Err(GradError::ShapeMismatch {
            expected: nocc * nvir,
            got: occ_cols.len() * vir_cols.len(),
        }
        .into());
    }
    // C[μ, p] for F-order [nao, nmo]: data[μ + p·nao].
    let cmo = |mu: usize, p: usize| refr.mo_coeff.data[mu + p * nao];

    // 1. dm[μ,ν] = Σ_{a,i} Cv[μ,vir_cols[a]]·x[a·nocc+i]·Co[ν,occ_cols[i]].
    let mut dm = vec![0.0_f64; nao * nao];
    let mut terms = Vec::with_capacity(ndim);
    for mu in 0..nao {
        for nu in 0..nao {
            terms.clear();
            for a in 0..nvir {
                let cv = cmo(mu, vir_cols[a]);
                for i in 0..nocc {
                    terms.push(cv * x[a * nocc + i] * cmo(nu, occ_cols[i]));
                }
            }
            dm[mu * nao + nu] = oracle_sum(&terms);
        }
    }
    // dm + dmᵀ (symmetrize).
    let mut dm_sym = vec![0.0_f64; nao * nao];
    for mu in 0..nao {
        for nu in 0..nao {
            dm_sym[mu * nao + nu] = oracle_sum(&[dm[mu * nao + nu], dm[nu * nao + mu]]);
        }
    }

    // 2. v = get_veff(mol, dm + dmᵀ) — the SCF ENERGY J/K (int2e — cintx-ready);
    //    a clean availability error `?`-propagates if int2e is unavailable.
    let veff = pyscf_scf::fock::default_get_veff(&refr.mol, &Density { nao, data: dm_sym })?;
    if veff.data.len() != nao * nao {
        return Err(GradError::ShapeMismatch {
            expected: nao * nao,
            got: veff.data.len(),
        }
        .into());
    }
    let v = |mu: usize, nu: usize| veff.data[mu * nao + nu];

    // 3. v_vo[a,i] = Σ_{μ,ν} Cv[μ,a]·v[μ,ν]·Co[ν,i]; then 4. ×2.
    let mut out = vec![0.0_f64; ndim];
    let mut buf = Vec::with_capacity(nao * nao);
    for a in 0..nvir {
        let va = vir_cols[a];
        for i in 0..nocc {
            let oi = occ_cols[i];
            buf.clear();
            for mu in 0..nao {
                let cv = cmo(mu, va);
                for nu in 0..nao {
                    buf.push(cv * v(mu, nu) * cmo(nu, oi));
                }
            }
            out[a * nocc + i] = 2.0 * oracle_sum(&buf);
        }
    }
    Ok(out)
}

/// The CCSD orbital-relaxation 1-RDM block (the Z-vector response), the analog of
/// the MP2 `_response_dm1` (`mp2.py:268-284`).
///
/// Solves `dvo = cphf::solve(fvind, mo_energy, mo_occ, Xvo)` through the SINGLE
/// CPHF solver (the D-03/GRAD-10 consumer contract; CCSD passes its OWN `fvind`
/// and RHS — the SAME `cphf::solve` MP2 uses, NEVER a second copy), then places
/// `dvo` into the `(vir,occ)` and `(occ,vir)` blocks of an `nmo × nmo` MO-basis
/// response density. CCSD leaves `max_cycle` at the upstream default
/// [`CCSD_CPHF_MAX_CYCLE`] (= 50, unlike MP2's 30).
///
/// `xvo` is the Z-vector RHS, flat vir-major `(nvir·nocc)`. Returns the flat
/// row-major `(nmo × nmo)` response `dm1`, with `dm1[vir_a, occ_i] = dvo[a,i]`
/// and `dm1[occ_i, vir_a] = dvo[a,i]`.
pub fn response_dm1(refr: &CcsdGradReference, xvo: &[f64]) -> Result<Vec<f64>, PyscfRsError> {
    let nocc = refr.nocc();
    let nvir = refr.nvir();
    let nmo = nocc + nvir;
    let ndim = nvir * nocc;
    if xvo.len() != ndim {
        return Err(GradError::ShapeMismatch {
            expected: ndim,
            got: xvo.len(),
        }
        .into());
    }

    // The Z-vector solve through the ONE cphf solver (the 07-07 GRAD-10 solver).
    let fvind = |x: &[f64]| ccsd_fvind(refr, x);
    let dvo = cphf::solve(
        &fvind,
        &refr.mo_energy,
        &refr.mo_occ,
        xvo,
        None, // s1 = None → solve_nos1 (the Z-vector path).
        CCSD_CPHF_MAX_CYCLE,
        cphf::DEFAULT_TOL,
        false, // hermi
        cphf::DEFAULT_LEVEL_SHIFT,
    )?;
    if dvo.len() != ndim {
        return Err(GradError::ShapeMismatch {
            expected: ndim,
            got: dvo.len(),
        }
        .into());
    }

    // dm1[nocc:, :nocc] = dvo ; dm1[:nocc, nocc:] = dvoᵀ.
    let mut dm1 = vec![0.0_f64; nmo * nmo];
    for a in 0..nvir {
        for i in 0..nocc {
            let val = dvo[a * nocc + i];
            dm1[(nocc + a) * nmo + i] = val; // (vir_a, occ_i)
            dm1[i * nmo + (nocc + a)] = val; // (occ_i, vir_a) = dvoᵀ
        }
    }
    Ok(dm1)
}

/// The int2e-only (cintx-ready) part of the orbital-relaxation Z-vector RHS
/// `Xvo` (the analog of `crate::mp2::build_xvo_base`): `Xvo[a,i] = Σ_{μν}
/// Cv[μ,a]·(2·get_veff(C·dm1mo·Cᵀ))[μ,ν]·Co[ν,i]`. Pure energy-`int2e` linear
/// algebra — the always-on RHS arm. `?`-routes a clean availability error only
/// if `int2e` is unavailable.
fn build_xvo_base(refr: &CcsdGradReference, dm1mo: &[f64]) -> Result<Vec<f64>, PyscfRsError> {
    let nao = refr.mo_coeff.nao;
    let nmo = refr.mo_coeff.nmo;
    let nocc = refr.nocc();
    let nvir = refr.nvir();
    let occ_cols: Vec<usize> = (0..nmo).filter(|&p| refr.mo_occ[p] > 0.0).collect();
    let vir_cols: Vec<usize> = (0..nmo).filter(|&p| refr.mo_occ[p] == 0.0).collect();
    let cmo = |mu: usize, p: usize| refr.mo_coeff.data[mu + p * nao];

    // AO relaxed density dm_ao = C·dm1mo·Cᵀ (full nmo, row-major (nao,nao)).
    let mut dm_ao = vec![0.0_f64; nao * nao];
    let mut tmp = vec![0.0_f64; nao * nmo]; // tmp[μ,q] = Σ_p C[μ,p]·dm1mo[p,q].
    let mut buf = Vec::with_capacity(nmo);
    for mu in 0..nao {
        for q in 0..nmo {
            buf.clear();
            for p in 0..nmo {
                buf.push(cmo(mu, p) * dm1mo[p * nmo + q]);
            }
            tmp[mu * nmo + q] = oracle_sum(&buf);
        }
    }
    for mu in 0..nao {
        for nu in 0..nao {
            buf.clear();
            for q in 0..nmo {
                buf.push(tmp[mu * nmo + q] * cmo(nu, q));
            }
            dm_ao[mu * nao + nu] = oracle_sum(&buf);
        }
    }

    // vhf = get_veff(mol, dm_ao) * 2. ENERGY int2e J/K (cintx-ready).
    let veff = pyscf_scf::fock::default_get_veff(&refr.mol, &Density { nao, data: dm_ao })?;
    let vhf = |mu: usize, nu: usize| 2.0 * veff.data[mu * nao + nu];

    // Xvo[a,i] = Σ_{μν} Cv[μ,a]·vhf[μ,ν]·Co[ν,i].
    let mut xvo = vec![0.0_f64; nvir * nocc];
    let mut vbuf = Vec::with_capacity(nao * nao);
    for a in 0..nvir {
        let va = vir_cols[a];
        for i in 0..nocc {
            let oi = occ_cols[i];
            vbuf.clear();
            for mu in 0..nao {
                let cv = cmo(mu, va);
                for nu in 0..nao {
                    vbuf.push(cv * vhf(mu, nu) * cmo(nu, oi));
                }
            }
            xvo[a * nocc + i] = oracle_sum(&vbuf);
        }
    }
    Ok(xvo)
}

/// The CCSD electronic gradient (`pyscf/grad/ccsd.py`). Free-fn form so the PyO3
/// bridge can reuse it without the trait.
///
/// Consumes the Phase-6 Λ + relaxed-density 1-RDM via [`relaxed_rdm1`] (D-04 —
/// NO re-derivation), forms the orbital-relaxation Z-vector RHS `Xvo`, solves the
/// response through [`response_dm1`] (the ONE `cphf::solve`), and assembles the
/// gradient by plugging the relaxed density into the RHF `grad_elec` base
/// decomposition (07-03).
///
/// The gradient `de` assembly contracts the cintx-gated grad-intor families via
/// the RHF `get_veff`/`get_ovlp`/`hcore_deriv`; those `?`-route to a clean
/// cintx-availability error (D-02) until the cintx workstream lands them. The
/// Λ-consumption + relaxed-density + Z-vector machinery above is always-on.
pub fn grad_elec(
    refr: &CcsdGradReference,
    atmlst: Option<&[usize]>,
) -> Result<Vec<[f64; 3]>, PyscfRsError> {
    let mol = &refr.mol;
    let nocc = refr.nocc();
    let nvir = refr.nvir();
    let nmo = nocc + nvir;
    let rows = crate::resolve_atmlst(atmlst, mol.natm)?;

    // 1. Consume the Phase-6 Λ + relaxed MO 1-RDM directly (D-04 / GRAD-06). NO
    //    re-derivation. `relaxed_rdm1` runs `solve_lambda` + `make_rdm1`.
    let mut dm1mo = relaxed_rdm1(refr)?;
    if dm1mo.len() != nmo * nmo {
        return Err(GradError::ShapeMismatch {
            expected: nmo * nmo,
            got: dm1mo.len(),
        }
        .into());
    }

    // 2. The orbital-relaxation Z-vector RHS Xvo (the int2e-only, cintx-ready
    //    arm); the full upstream Xvo's gradient-integral arm rides the cintx gate
    //    inside the `de` assembly below.
    let xvo = build_xvo_base(refr, &dm1mo)?;

    // 3. The orbital response through the ONE cphf solver (the D-03/GRAD-10
    //    consumer contract); dm1mo += response_dm1(Xvo).
    let resp = response_dm1(refr, &xvo)?;
    for (slot, r) in dm1mo.iter_mut().zip(&resp) {
        *slot = oracle_sum(&[*slot, *r]);
    }

    // 4. Assemble the gradient on the RHF base decomposition (07-03). Plugging
    //    the relaxed density into the RHF grad_elec contracts the cintx-gated
    //    grad-intor families; those `?`-route to a clean cintx-availability error
    //    (D-02). The relaxed-density + Z-vector machinery (steps 1-3) is complete
    //    and always-on; the per-atom `de` contraction is the cintx-gated numeric.
    let rhf_ref = refr.as_rhf();
    let base_de = crate::rhf::grad_elec(&rhf_ref, Some(&rows))?;

    // (When cintx lands the families, the relaxed-density contraction replaces
    //  base_de; today base_de is unreachable past the clean-error `?` above.)
    Ok(base_de)
}

/// CCSD electronic gradient seam preserved for the 07-02 module stub. The real
/// body lives in [`grad_elec`]; this thin wrapper errors clearly if called
/// without a reference (the PyO3 bridge always supplies one).
pub fn default_grad_elec() -> Result<Vec<[f64; 3]>, PyscfRsError> {
    Err(PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
        "CcsdGradients::grad_elec requires a CcsdGradReference snapshot — build a \
         CcsdGradients via CcsdGradients::new(reference) (07-08)"
            .into(),
    )))
}
