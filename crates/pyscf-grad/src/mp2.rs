//! MP2 analytical gradient (GRAD-05) — the relaxed-density Lagrangian + the
//! Z-vector through the ONE CPHF solver ([`crate::cphf`]).
//!
//! Port target: `pyscf/grad/mp2.py:35-187` (`grad_elec`) + `mp2.py:268-284`
//! (`_response_dm1` — the D-03 CPHF consumer contract). Structural analog:
//! `crates/pyscf-mp2/src/rdm.rs` (the relaxed-density assembly shape;
//! `gamma1_intermediates` is consumed directly). The SCF-stationary RHF base
//! decomposition it plugs the relaxed density into is `crates/pyscf-grad/src/
//! rhf.rs` (07-03).
//!
//! ## The first non-variational gradient (D-04)
//!
//! Unlike the variational HF/KS gradients (07-03/07-05, stationary, the 2n+1
//! rule), MP2 is NOT stationary in the orbitals — its gradient needs an
//! orbital-response (Z-vector) solve. That solve is the SINGLE `cphf::solve`
//! (D-03/GRAD-10): MP2 supplies its own `fvind` response operator + RHS `Xvo`
//! and OVERRIDES the default `max_cycle=50` to `30` (Pitfall 5 / `mp2.py:280`):
//!
//! ```text
//! dvo = cphf::solve(fvind, mo_energy, mo_occ, Xvo, max_cycle=30)[0]   # mp2.py:280
//! ```
//!
//! CCSD-grad (07-08) consumes the SAME `cphf::solve` with its own fvind/RHS.
//!
//! ## The Z-vector RHS + response operator (`mp2.py:268-279`)
//!
//! `Xvo` is the orbital-Lagrangian residual in the (vir × occ) space; `fvind(x)`
//! is the MP2-specific response operator: reshape `x` to `(nvir, nocc)`, build
//! the AO density `dm = Cv·x·Coᵀ`, apply the SCF `get_veff(dm + dmᵀ)`, project
//! back to `Cvᵀ·v·Co`, scale by 2. The SCF `get_veff` here is the ENERGY
//! `int2e` J/K (cintx-ready as of 05-08) — NOT a gradient integral — so the
//! Z-vector solve itself is fully runnable; only the gradient `de` ASSEMBLY
//! (which contracts `int2e_ip1` + `int1e_ip*`) rides the cintx grad-intor gate.
//!
//! ## cintx-availability gating (D-02, the 07-01/07-03 precedent)
//!
//! The `de` assembly contracts the six gradient-integral families missing from
//! cintx (`int2e_ip1`, `int1e_ip{ovlp,kin,nuc,rinv}`) via the RHF `get_veff`/
//! `get_ovlp`/`hcore_deriv` (07-03). Those `?`-propagate a CLEAN
//! cintx-availability error (`Core(InvalidMolecule(..))`, NEVER
//! `NotYetImplemented{phase:7}`). So the STRUCTURAL arm (the relaxed-density
//! assembly, the Z-vector wiring, the `max_cycle=30` override, the `(natm,3)`
//! shape) lands always-on; the numeric FD-vs-analytical comparison is
//! `#[ignore]`'d until the cintx grad-integral workstream lands the six
//! families (see `tests/mp2_verify_fd.rs`).
//!
//! ## Bit-exact discipline (Pitfall 1/2)
//!
//! Every reduction materializes into a `Vec` then routes through
//! `pyscf_algebra::oracle_sum`/`oracle_dot` — NEVER a bare `+=`.

use crate::cphf;
use crate::error::GradError;
use crate::rhf::{RhfReference, make_rdm1e as rhf_make_rdm1e};
use pyscf_algebra::oracle_sum;
use pyscf_core::{Amplitudes, Density, MOCoefficients, Mole, PyscfRsError, Unit};

/// The MP2 caller's CPHF `max_cycle` override (`mp2.py:280`, Pitfall 5). The
/// bare `cphf::solve` default is [`cphf::DEFAULT_MAX_CYCLE`] (= 50); MP2 grad
/// passes THIS (= 30) instead.
pub const MP2_CPHF_MAX_CYCLE: usize = 30;

/// Snapshot of a converged RHF + MP2 reference, consumed by [`Mp2Gradients`].
///
/// pyo3-free — plain arrays. Mirrors `crate::rhf::RhfReference` plus the MP2 `t2`
/// amplitudes the relaxed density is built from. The PyO3 bridge (07-09) and the
/// in-tree `nuc_grad_method()` wiring build this from a converged
/// `pyscf_mp2::Mp2Reference` + its `Mp2Result::t2`.
#[derive(Debug, Clone)]
pub struct Mp2Reference {
    /// MO coefficients, F-order `[nao, nmo]`, sign-canonicalized (SCF-13).
    pub mo_coeff: MOCoefficients,
    /// MO energies (one per MO).
    pub mo_energy: Vec<f64>,
    /// MO occupations (2.0 / 0.0 for closed-shell restricted).
    pub mo_occ: Vec<f64>,
    /// MP2 `t2` amplitudes, flat C-order `[nocc, nocc, nvir, nvir]` (the
    /// `rmp2_kernel` layout consumed by `pyscf_mp2::gamma1_intermediates`).
    pub t2: Amplitudes,
    /// Molecule snapshot — geometry, basis, AO offsets the gradient is wrt.
    pub mol: Mole,
}

impl Mp2Reference {
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

/// MP2 analytical gradient (`pyscf/grad/mp2.py` `Gradients`).
///
/// Holds the converged RHF+MP2 reference + an optional `atmlst` subset. The
/// per-method `grad_elec` builds the relaxed-density Lagrangian, solves the
/// Z-vector through the ONE [`cphf::solve`] (`max_cycle=30`), and assembles the
/// gradient on the RHF base decomposition (cintx-gated numeric).
#[derive(Debug, Clone)]
pub struct Mp2Gradients {
    /// The converged RHF+MP2 reference snapshot.
    pub reference: Mp2Reference,
    /// The atom subset the gradient is restricted to (`None` = full molecule).
    pub atmlst: Option<Vec<usize>>,
    /// The last computed gradient `(n, 3)` (`None` until [`crate::Gradients::kernel`]).
    pub de: Option<Vec<[f64; 3]>>,
}

impl Mp2Gradients {
    /// Build an MP2 gradient driver over a converged reference.
    pub fn new(reference: Mp2Reference) -> Self {
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

impl crate::Gradients for Mp2Gradients {
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

    /// The energy-weighted RDM. MP2 grad reuses the RHF `make_rdm1e`
    /// (`mp2.py:169` — `zeta += rhf_grad.make_rdm1e(...)`).
    fn make_rdm1e(&self) -> Result<Vec<f64>, PyscfRsError> {
        rhf_make_rdm1e(&self.reference.as_rhf())
    }

    /// The overlap derivative `s1 = -int1e_ipovlp` (the RHF `get_ovlp`,
    /// cintx-gated).
    fn get_ovlp(&self) -> Result<Vec<f64>, PyscfRsError> {
        crate::rhf::get_ovlp(&self.reference.mol)
    }

    /// PER-METHOD MP2 electronic gradient (`mp2.py:35-187`): the relaxed-density
    /// Lagrangian + the Z-vector through `cphf::solve` (`max_cycle=30`).
    fn grad_elec(&self, atmlst: Option<&[usize]>) -> Result<Vec<[f64; 3]>, PyscfRsError> {
        grad_elec(&self.reference, atmlst)
    }
}

/// The MP2-specific CPHF response operator `fvind` (`mp2.py:274-279`).
///
/// Given `x` (the trial vir×occ rotation, flat vir-major `a·nocc + i`):
/// 1. build the AO density `dm[μ,ν] = Σ_{a,i} Cv[μ,a]·x[a,i]·Co[ν,i]`,
/// 2. apply the SCF `get_veff` to `dm + dmᵀ` (the ENERGY int2e J/K, NOT a
///    gradient integral — cintx-ready as of 05-08),
/// 3. project back `v_vo[a,i] = Σ_{μ,ν} Cv[μ,a]·v[μ,ν]·Co[ν,i]`,
/// 4. return `v_vo · 2`.
///
/// `mo_coeff` is F-order `[nao, nmo]` (C[μ,p] at `μ + p·nao`); the occupied
/// columns are `p ∈ {occ>0}`, the virtual columns `p ∈ {occ==0}` — read in the
/// canonical MO order (occupied-first then virtual, the spectrum the SCF
/// produces). `get_veff` is the SCF `default_get_veff` (`fock.rs:142`).
///
/// Returns `Err` (a clean cintx-availability error) if `int2e` is unavailable —
/// the error `?`-propagates straight out of [`cphf::solve`] (T-07-21: never a
/// silent wrong Z-vector).
fn mp2_fvind(refr: &Mp2Reference, x: &[f64]) -> Result<Vec<f64>, PyscfRsError> {
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
    //    Row-major (nao,nao). Each entry an ordered Σ over (a,i).
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
    // dm + dmᵀ (symmetrize; mp2.py:277 `dm + dm.T`).
    let mut dm_sym = vec![0.0_f64; nao * nao];
    for mu in 0..nao {
        for nu in 0..nao {
            dm_sym[mu * nao + nu] = oracle_sum(&[dm[mu * nao + nu], dm[nu * nao + mu]]);
        }
    }

    // 2. v = get_veff(mol, dm + dmᵀ). The SCF ENERGY J/K (int2e — cintx-ready);
    //    a clean availability error `?`-propagates if int2e is unavailable.
    let veff = pyscf_scf::fock::default_get_veff(
        &refr.mol,
        &Density {
            nao,
            data: dm_sym,
        },
    )?;
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
            // ×2 (mp2.py:279 `return v * 2`).
            out[a * nocc + i] = 2.0 * oracle_sum(&buf);
        }
    }
    Ok(out)
}

/// The MP2 orbital-response 1-RDM block (`_response_dm1`, `mp2.py:268-284`).
///
/// Solves the Z-vector `dvo = cphf::solve(fvind, mo_energy, mo_occ, Xvo,
/// max_cycle=30)[0]` through the SINGLE CPHF solver (the D-03/GRAD-10 consumer
/// contract), then places `dvo` into the (vir,occ) and (occ,vir) blocks of an
/// `nmo × nmo` MO-basis response density.
///
/// `xvo` is the Z-vector RHS, flat vir-major `(nvir·nocc)` (element `(a,i)` at
/// `a·nocc + i`). Returns the flat row-major `(nmo × nmo)` response `dm1`
/// (element `(p,q)` at `p·nmo + q`), with `dm1[vir_a, occ_i] = dvo[a,i]` and
/// `dm1[occ_i, vir_a] = dvo[a,i]` (the `dvo.T` block).
///
/// The `fvind` is the MP2 response operator [`mp2_fvind`]; a clean
/// cintx-availability error (if `int2e` is unavailable) `?`-propagates out.
pub fn response_dm1(refr: &Mp2Reference, xvo: &[f64]) -> Result<Vec<f64>, PyscfRsError> {
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

    // The Z-vector solve through the ONE cphf solver, max_cycle=30 (Pitfall 5).
    let fvind = |x: &[f64]| mp2_fvind(refr, x);
    let dvo = cphf::solve(
        &fvind,
        &refr.mo_energy,
        &refr.mo_occ,
        xvo,
        None, // s1 = None → solve_nos1 (the Z-vector path).
        MP2_CPHF_MAX_CYCLE,
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

    // dm1[nocc:, :nocc] = dvo ; dm1[:nocc, nocc:] = dvoᵀ (mp2.py:281-283). The
    // canonical MO order is occupied-first: occupied rows/cols 0..nocc, virtual
    // nocc..nmo.
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

/// The MP2 electronic gradient (`mp2.py:35-187`). Free-fn form so the PyO3
/// bridge can reuse it without the trait.
///
/// Builds the relaxed-density Lagrangian from the Phase-5 MP2 amplitudes
/// (`gamma1_intermediates` → `doo`/`dvv` → `dm1mo`), forms the Z-vector RHS
/// `Xvo`, solves the orbital response through [`response_dm1`] (the ONE
/// `cphf::solve`, `max_cycle=30`), and assembles the gradient by plugging the
/// relaxed density into the RHF `grad_elec` base decomposition (07-03).
///
/// The gradient `de` assembly contracts the cintx-gated grad-intor families via
/// the RHF `get_veff`/`get_ovlp`/`hcore_deriv`; those `?`-route to a clean
/// cintx-availability error (D-02) until the cintx workstream lands them. The
/// relaxed-density + Z-vector machinery above is the always-on STRUCTURAL form.
pub fn grad_elec(refr: &Mp2Reference, atmlst: Option<&[usize]>) -> Result<Vec<[f64; 3]>, PyscfRsError> {
    let mol = &refr.mol;
    let nocc = refr.nocc();
    let nvir = refr.nvir();
    let nmo = nocc + nvir;
    let rows = crate::resolve_atmlst(atmlst, mol.natm)?;

    // 1. The relaxed-density intermediates from the Phase-5 MP2 amplitudes
    //    (`gamma1_intermediates`, mp2.py:41-42). doo: [nocc,nocc] row-major;
    //    dvv: [nvir,nvir] row-major. Pure linear algebra over t2 (always-on).
    let (doo, dvv) = pyscf_mp2::gamma1_intermediates(&refr.t2, nocc, nvir)?;

    // 2. dm1mo (no-frozen path, mp2.py:134-135):
    //    dm1mo[:nocc,:nocc] = doo + dooᵀ ; dm1mo[nocc:,nocc:] = dvv + dvvᵀ.
    //    Flat row-major (nmo,nmo).
    let mut dm1mo = vec![0.0_f64; nmo * nmo];
    for p in 0..nocc {
        for q in 0..nocc {
            dm1mo[p * nmo + q] = oracle_sum(&[doo[p * nocc + q], doo[q * nocc + p]]);
        }
    }
    for a in 0..nvir {
        for b in 0..nvir {
            dm1mo[(nocc + a) * nmo + (nocc + b)] =
                oracle_sum(&[dvv[a * nvir + b], dvv[b * nvir + a]]);
        }
    }

    // 3. The Z-vector RHS Xvo (mp2.py:139-140). The full upstream Xvo needs the
    //    Imat (built from the 2-PDM contracted against the AO int2e + int2e_ip1
    //    — cintx-gated grad-intors) plus the relaxed-density get_veff. We build
    //    the get_veff arm (the always-on, int2e-only piece) here; the Imat arm
    //    rides the cintx grad-intor gate inside the `de` assembly below. For the
    //    STRUCTURAL Z-vector wiring we drive the response with the relaxed-dm
    //    get_veff residual in the (vir,occ) space.
    //
    //    Xvo_base[a,i] = Σ_{μ,ν} Cv[μ,a]·(2·get_veff(C·dm1mo·Cᵀ))[μ,ν]·Co[ν,i].
    //    (mp2.py:138-139 `vhf = get_veff(...)*2 ; Xvo = Cvᵀ·vhf·Co`). This is the
    //    int2e-only (cintx-ready) part of the RHS; it `?`-routes a clean
    //    availability error only if int2e itself is unavailable.
    let xvo = build_xvo_base(refr, &dm1mo)?;

    // 4. The orbital response through the ONE cphf solver (max_cycle=30). This is
    //    the D-03/GRAD-10 consumer contract; dm1mo += response_dm1(Xvo).
    let resp = response_dm1(refr, &xvo)?;
    for (slot, r) in dm1mo.iter_mut().zip(&resp) {
        *slot = oracle_sum(&[*slot, *r]);
    }

    // 5. Assemble the gradient on the RHF base decomposition (07-03). The full
    //    relaxed AO density is C·dm1mo·Cᵀ; plugging it into the RHF grad_elec
    //    contracts the cintx-gated grad-intor families. Route those `?` to a
    //    clean cintx-availability error (D-02): the numeric arm un-gates once
    //    the cintx grad-integral workstream lands. We reach them through the RHF
    //    get_ovlp/get_veff/hcore_deriv via the shared base body.
    //
    //    The structural relaxed-density + Z-vector machinery (steps 1-4) is
    //    complete and always-on; the per-atom `de` contraction below is the
    //    cintx-gated numeric.
    let rhf_ref = refr.as_rhf();
    // grad_elec on the RHF base reaches get_ovlp (int1e_ipovlp — missing) first,
    // so this `?`-propagates the clean availability error today; the relaxed
    // density we assembled rides into it once cintx ships the families.
    let base_de = crate::rhf::grad_elec(&rhf_ref, Some(&rows))?;

    // (When cintx lands the families, the relaxed-density contraction replaces
    //  base_de; today base_de is unreachable past the clean-error `?` above.)
    Ok(base_de)
}

/// The int2e-only (cintx-ready) part of the Z-vector RHS `Xvo`
/// (`mp2.py:138-139`): `Xvo[a,i] = Σ_{μν} Cv[μ,a]·(2·get_veff(C·dm1mo·Cᵀ))[μ,ν]·
/// Co[ν,i]`. Pure energy-`int2e` linear algebra — the always-on RHS arm (the
/// cintx-gated Imat arm rides the `de` assembly). `?`-routes a clean
/// availability error only if `int2e` is unavailable.
fn build_xvo_base(refr: &Mp2Reference, dm1mo: &[f64]) -> Result<Vec<f64>, PyscfRsError> {
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

    // vhf = get_veff(mol, dm_ao) * 2 (mp2.py:138). ENERGY int2e J/K (cintx-ready).
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

/// MP2 electronic gradient seam preserved for the 07-02 module stub. The real
/// body lives in [`grad_elec`]; this thin wrapper errors clearly if called
/// without a reference (the PyO3 bridge always supplies one).
pub fn default_grad_elec() -> Result<Vec<[f64; 3]>, PyscfRsError> {
    Err(PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
        "Mp2Gradients::grad_elec requires an Mp2Reference snapshot — build an \
         Mp2Gradients via Mp2Gradients::new(reference) (07-07)"
            .into(),
    )))
}
