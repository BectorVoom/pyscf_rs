//! Conventional DF-MP2: `DFRMP2(:RMP2)` / `DFUMP2(:UMP2)` — swap the ERI
//! source to the `pyscf-df` `DfIntegrals.b_uvq` 3-tensor (port
//! `pyscf/mp/dfmp2.py` + `pyscf/mp/dfump2.py`).
//!
//! ## What this is (D-06 primary, MP2-04 conventional half)
//!
//! Upstream `DFRMP2` *subclasses* `mp2.RMP2` (`dfmp2.py:124`); the ONLY thing
//! that changes vs in-core RMP2 is the integral source — `ao2mo()` returns a
//! density-fitted `(ia|jb)` block instead of the full-AO transform. The
//! closed-form correlation-energy kernel ([`crate::mp2::rmp2_kernel`] /
//! [`crate::ump2::ump2_kernel`]) is reused VERBATIM. This file therefore
//! provides a *different* [`Mp2OverrideHooks::ao2mo`] implementation (the
//! "swap-the-source" pattern) plus thin `dfrmp2_kernel` / `dfump2_kernel`
//! wrappers that wire it into the existing kernels.
//!
//! ## The DF contraction (D-06)
//!
//! The density-fitted Chemist's `(ia|jb)` integral is
//!
//! ```text
//! (ia|jb) = Σ_Q  B^Q_ia · B^Q_jb
//! ```
//!
//! where `B^Q_ia` is the AO→MO transform of the DF B-tensor `b_uvq` into the
//! (occupied, virtual) MO block for auxiliary index `Q`:
//!
//! ```text
//! B^Q_ia = Σ_{μ,ν}  C_μ^i · b_uvq[μ, ν, Q] · C_ν^a
//! ```
//!
//! Upstream forms the MO-transformed tensor `ovL` (shape `[nocc, nvir, naux]`)
//! then contracts `(ia|jb) = Σ_L iaL·jbL` through the C driver
//! `libmp.MP2_contract_d` (`dfmp2.py:65,96`). We port the MATH of that driver
//! as an [`oracle_dot`] over the auxiliary axis — NO C dependency, NO `+=`
//! accumulation (T-05-05-FP, Pitfall 1/2).
//!
//! ## Aux-basis default (A2 CONFIRMED)
//!
//! DF-MP2 uses the `*-ri` (mp2fit) auxiliary basis via [`pyscf_df::default_ri`]
//! — NOT the JK-fit aux used by DF-HF. Upstream `dfmp2.py:136` uses
//! `make_auxbasis(mol, mp2fit=True)`. Using the wrong aux silently shifts the
//! DF energy (T-05-05-AUX), so the choice is pinned by an acceptance grep gate.
//!
//! ## cintx#11 gating (D-05)
//!
//! [`pyscf_df::cholesky_eri`] builds `b_uvq` from `int3c2e_sph`, which is a
//! cintx-ops upstream gap (returns the gated error today). [`df_ao2mo`] and the
//! DF kernels `?`-propagate that error — they NEVER panic and NEVER substitute
//! a zero B-tensor (the Phase-4 CR-02 silent-substitution lesson; T-05-05-FFI).
//! When cintx lands the `int3c2e_sph` base op, the SAME code returns the
//! numeric DF-MP2 result with no change.

use crate::frozen::{self, Frozen};
use crate::hooks::{ChemistsEris, Mp2OverrideHooks};
use crate::mp2::{Mp2Reference, Mp2Result, rmp2_kernel};
use crate::ump2::{UmpReference, UmpResult, ump2_kernel};
use pyscf_algebra::{oracle_dot, oracle_sum};
use pyscf_core::{MOCoefficients, PyscfRsError};
use pyscf_df::DfIntegrals;

/// Per-atom atomic numbers `Z` for the reference's molecule (used by
/// [`Frozen::Auto`] chemcore resolution). Derived from `Mole::atom_charges()`.
fn reference_elements(refr: &Mp2Reference) -> Vec<u32> {
    refr.mol
        .atom_charges()
        .into_iter()
        .map(|z| z.max(0) as u32)
        .collect()
}

/// Build the active occupied OR virtual MO column subset honouring the frozen
/// mask. `want_occupied = true` selects active occupied columns
/// (`mo_occ > 0 AND active`); `false` selects active virtual columns
/// (`mo_occ == 0 AND active`). Returns the subset as column-major
/// `[nao, n_selected]` (mirrors `crate::mp2::mo_subset`, kept local so the DF
/// path has no cross-module private dependency).
fn mo_subset(
    refr: &Mp2Reference,
    mask: &[bool],
    want_occupied: bool,
) -> Result<MOCoefficients, PyscfRsError> {
    let nao = refr.mo_coeff.nao;
    let nmo = refr.mo_coeff.nmo;
    let mut data = Vec::new();
    for col in 0..nmo {
        let active = mask.get(col).copied().unwrap_or(false);
        if !active {
            continue;
        }
        let occ = refr.mo_occ.get(col).copied().unwrap_or(0.0);
        let is_occ = occ > 0.0;
        if is_occ != want_occupied {
            continue;
        }
        let start = col * nao;
        let end = start + nao;
        if end > refr.mo_coeff.data.len() {
            return Err(crate::error::Mp2Error::ShapeMismatch {
                expected: refr.mo_coeff.data.len(),
                got: end,
            }
            .into());
        }
        data.extend_from_slice(&refr.mo_coeff.data[start..end]);
    }
    let n_selected = data.len().checked_div(nao.max(1)).unwrap_or(0);
    Ok(MOCoefficients {
        nao,
        nmo: n_selected,
        data,
        energies: Vec::new(),
        occupations: Vec::new(),
    })
}

/// MO-transform the DF B-tensor into the `(occ,vir)` block: `B^Q_ia`.
///
/// Input `b_uvq` is AO-basis, ROW-MAJOR `[nao, nao, naux]` — element
/// `(μ, ν, Q)` lives at `b_uvq[μ*nao*naux + ν*naux + Q]` (the [`DfIntegrals`]
/// layout). `co.data` / `cv.data` are column-major `[nao, nocc]` / `[nao, nvir]`
/// MO blocks (element `C[ao, mo]` at `ao + mo*nao`).
///
/// Output `ovl` is ROW-MAJOR `[nocc, nvir, naux]` — element `(i, a, Q)` at
/// `ovl[(i*nvir + a)*naux + Q]`. Computed as
/// `B^Q_ia = Σ_{μ,ν} C_μ^i · b_uvq[μ,ν,Q] · C_ν^a` via a materialize-then-sum
/// two-index transform: for each `(i, Q)` first contract μ into a half-transform
/// `half[ν] = Σ_μ C_μ^i·b[μ,ν,Q]`, then contract ν
/// (`B^Q_ia = Σ_ν half[ν] · C_ν^a`). Every reduction materializes products into
/// a scratch `Vec` then folds via [`oracle_sum`] — NO `+=` accumulation
/// (T-05-05-FP / -LAYOUT).
fn transform_b_to_ov(
    df: &DfIntegrals,
    co: &MOCoefficients,
    cv: &MOCoefficients,
) -> Result<Vec<f64>, PyscfRsError> {
    let nao = df.nao;
    let naux = df.naux;
    let nocc = co.nmo;
    let nvir = cv.nmo;

    // Shape guards (T-05-05-FFI / caller→df_ao2mo boundary): the B-tensor and
    // the MO blocks must agree on nao; b_uvq must be exactly [nao,nao,naux].
    if co.nao != nao || cv.nao != nao {
        return Err(crate::error::Mp2Error::ShapeMismatch {
            expected: nao,
            got: co.nao.max(cv.nao),
        }
        .into());
    }
    if df.b_uvq.len() != nao * nao * naux {
        return Err(crate::error::Mp2Error::ShapeMismatch {
            expected: nao * nao * naux,
            got: df.b_uvq.len(),
        }
        .into());
    }

    let mut ovl = vec![0.0_f64; nocc * nvir * naux];

    // Scratch reused across the loop bodies (materialize-then-oracle_sum).
    let mut prod_mu: Vec<f64> = vec![0.0; nao]; // C_μ^i · b[μ,ν,Q] terms over μ
    let mut prod_nu: Vec<f64> = vec![0.0; nao]; // half[ν] · C_ν^a terms over ν
    // half_inu[ν] = Σ_μ C_μ^i · b[μ,ν,Q] for a fixed (i, Q).
    let mut half_inu: Vec<f64> = vec![0.0; nao];

    for i in 0..nocc {
        let ci = &co.data[i * nao..(i + 1) * nao]; // C_μ^i, length nao
        for q in 0..naux {
            // half_inu[ν] = Σ_μ C_μ^i · b_uvq[μ,ν,Q] (materialize → oracle_sum).
            for (nu, half) in half_inu.iter_mut().enumerate() {
                for (mu, pm) in prod_mu.iter_mut().enumerate() {
                    *pm = ci[mu] * df.b_uvq[mu * nao * naux + nu * naux + q];
                }
                *half = oracle_sum(&prod_mu);
            }
            // B^Q_ia = Σ_ν half_inu[ν] · C_ν^a (materialize → oracle_sum).
            for a in 0..nvir {
                let ca = &cv.data[a * nao..(a + 1) * nao]; // C_ν^a, length nao
                for (nu, pn) in prod_nu.iter_mut().enumerate() {
                    *pn = half_inu[nu] * ca[nu];
                }
                let b_q_ia = oracle_sum(&prod_nu);
                ovl[(i * nvir + a) * naux + q] = b_q_ia;
            }
        }
    }
    Ok(ovl)
}

/// DF `ao2mo`: assemble the Chemist's `(ia|jb)` block from the DF B-tensor.
///
/// Transforms `df.b_uvq` into the MO `(occ,vir)` block `B^Q_ia` (see
/// [`transform_b_to_ov`]) then assembles
/// `(ia|jb) = Σ_Q B^Q_ia · B^Q_jb` via [`oracle_dot`] over the auxiliary axis
/// `Q` for every `(ia, jb)` pair. This is the MATH of the upstream C driver
/// `libmp.MP2_contract_d` (`dfmp2.py:65`) — ported as an oracle reduction, NO
/// C dependency, NO `+=` accumulation (T-05-05-FP).
///
/// ## Flat-index contracts (T-05-05-LAYOUT)
/// - input `df.b_uvq`: ROW-MAJOR `[nao,nao,naux]`, `(μ,ν,Q)` at
///   `μ*nao*naux + ν*naux + Q`.
/// - intermediate `ovl` (from [`transform_b_to_ov`]): ROW-MAJOR
///   `[nocc,nvir,naux]`, `(i,a,Q)` at `(i*nvir + a)*naux + Q`.
/// - output [`ChemistsEris::ovov`]: C-order `[nocc,nvir,nocc,nvir]`, `(i,a,j,b)`
///   at `((i*nvir + a)*nocc + j)*nvir + b` (the layout
///   [`crate::mp2::rmp2_kernel`] consumes).
///
/// The frozen mask selects the active occupied (`co`) / virtual (`cv`) columns
/// — DF-MP2 honours `frozen` exactly like in-core RMP2.
///
/// # Errors
/// - [`PyscfRsError`] (via [`crate::error::Mp2Error::ShapeMismatch`]) on a
///   B-tensor / MO-block / occupied-count mismatch — never indexes OOB.
/// - Any error the caller `?`-propagated from [`pyscf_df::cholesky_eri`]
///   (notably the `int3c2e_sph` cintx#11 gate) is surfaced by the caller that
///   builds the [`DfIntegrals`]; `df_ao2mo` itself receives an already-built
///   `df` and never substitutes a zero buffer (T-05-05-FFI).
pub fn df_ao2mo(
    refr: &Mp2Reference,
    frozen: &Frozen,
    df: &DfIntegrals,
) -> Result<ChemistsEris, PyscfRsError> {
    let elements = reference_elements(refr);
    let mask = frozen::frozen_mask(frozen, &refr.mo_occ, &refr.mo_energy, &elements)?;

    let co = mo_subset(refr, &mask, true)?;
    let cv = mo_subset(refr, &mask, false)?;
    let nocc = co.nmo;
    let nvir = cv.nmo;
    let naux = df.naux;

    // ovl[(i,a,Q)] = B^Q_ia, ROW-MAJOR [nocc,nvir,naux].
    let ovl = transform_b_to_ov(df, &co, &cv)?;

    // (ia|jb) = Σ_Q B^Q_ia · B^Q_jb (oracle_dot over the Q axis).
    let mut ovov = vec![0.0_f64; nocc * nvir * nocc * nvir];
    for i in 0..nocc {
        for a in 0..nvir {
            let ia_base = (i * nvir + a) * naux;
            let b_ia = &ovl[ia_base..ia_base + naux];
            for j in 0..nocc {
                for b in 0..nvir {
                    let jb_base = (j * nvir + b) * naux;
                    let b_jb = &ovl[jb_base..jb_base + naux];
                    // Σ_Q B^Q_ia · B^Q_jb (deterministic Q-fold; NO `+=`).
                    let val = oracle_dot(b_ia, b_jb);
                    ovov[((i * nvir + a) * nocc + j) * nvir + b] = val;
                }
            }
        }
    }

    Ok(ChemistsEris { ovov, nocc, nvir })
}

/// The conventional DF-MP2 closed-shell driver state (`DFRMP2(mp2.RMP2)`,
/// `dfmp2.py:124`): an SCF reference snapshot plus the pre-assembled DF
/// B-tensor (built by the caller via
/// `pyscf_df::cholesky_eri(mol, pyscf_df::default_ri(basis))` — A2: the mp2fit
/// `*-ri` aux, NOT the JK-fit aux).
///
/// `DFRMP2` reuses the RMP2 base ([`rmp2_kernel`]) and only swaps the ERI
/// source via its [`Mp2OverrideHooks::ao2mo`] impl (delegates to [`df_ao2mo`]).
#[derive(Debug, Clone)]
pub struct DFRMP2 {
    /// Converged SCF reference (the RMP2 base reference).
    pub reference: Mp2Reference,
    /// Pre-assembled DF B-integrals (`*-ri` aux). Built by the caller; the
    /// `int3c2e_sph` cintx#11 gate is `?`-propagated by the caller's
    /// `cholesky_eri`, never panicked or zero-substituted here.
    pub df: DfIntegrals,
}

impl Mp2OverrideHooks for DFRMP2 {
    /// Swap the ERI source: build `(ia|jb)` from the DF B-tensor instead of the
    /// full-AO `int2e` transform (D-06). Delegates to [`df_ao2mo`].
    fn ao2mo(&self, refr: &Mp2Reference, frozen: &Frozen) -> Result<ChemistsEris, PyscfRsError> {
        df_ao2mo(refr, frozen, &self.df)
    }
}

/// Run the conventional closed-shell DF-MP2 kernel: wire [`df_ao2mo`] into the
/// reused [`rmp2_kernel`] (`DFRMP2.kernel`, `dfmp2.py:39`).
///
/// This is the "swap the ERI source" contract — the closed-form RMP2 math runs
/// verbatim with the DF `(ia|jb)` block in place of the in-core block.
///
/// # Errors
/// `?`-propagates any error from [`df_ao2mo`] / [`rmp2_kernel`] (notably a DF
/// B-tensor shape mismatch). The `int3c2e_sph` cintx#11 gate is surfaced
/// earlier when the caller builds `df` via `cholesky_eri`; this never panics
/// and never substitutes a zero buffer (T-05-05-FFI).
pub fn dfrmp2_kernel(
    refr: &Mp2Reference,
    frozen: &Frozen,
    df: &DfIntegrals,
    with_t2: bool,
) -> Result<Mp2Result, PyscfRsError> {
    let hooks = DfRmp2Hooks { df };
    rmp2_kernel(refr, frozen, &hooks, with_t2)
}

/// Borrowing hooks wrapper so `dfrmp2_kernel` can run without cloning the DF
/// B-tensor into a [`DFRMP2`]. Same swap-the-source behaviour.
struct DfRmp2Hooks<'a> {
    df: &'a DfIntegrals,
}

impl Mp2OverrideHooks for DfRmp2Hooks<'_> {
    fn ao2mo(&self, refr: &Mp2Reference, frozen: &Frozen) -> Result<ChemistsEris, PyscfRsError> {
        df_ao2mo(refr, frozen, self.df)
    }
}

/// The conventional DF-MP2 open-shell driver state (`DFUMP2`, `dfump2.py`): an
/// unrestricted α/β reference pair plus the per-spin DF B-tensors. Each spin
/// channel's `ao2mo` transforms its OWN B-tensor into its `(ia|jb)` block; the
/// reused [`ump2_kernel`] then consumes the three spin-block ERIs.
///
/// The αβ opposite-spin block is built from the α occupied/virtual MO transform
/// of the α B-tensor on the (i,a) side and the β transform of the β B-tensor on
/// the (J,B) side — i.e. `(ia|JB) = Σ_Q B^Q_ia(α) · B^Q_JB(β)` (the upstream
/// `dfump2.py` cross-spin contraction). The same auxiliary basis is shared
/// across spins (one `*-ri` aux for the molecule), so a single
/// [`DfIntegrals`] drives both channels.
#[derive(Debug, Clone)]
pub struct DFUMP2 {
    /// α/β unrestricted reference pair (the UMP2 base reference).
    pub reference: UmpReference,
    /// DF B-integrals (`*-ri` aux), shared across spin channels.
    pub df: DfIntegrals,
}

/// Build the αβ opposite-spin DF `(ia|JB)` block:
/// `(ia|JB) = Σ_Q B^Q_ia(α) · B^Q_JB(β)`.
///
/// `ovl_a` is the α MO-transformed B-tensor (ROW-MAJOR `[nocc_a,nvir_a,naux]`)
/// and `ovl_b` the β one (`[nocc_b,nvir_b,naux]`). Output is C-order
/// `[nocc_a,nvir_a,nocc_b,nvir_b]`, `(i,a,J,B)` at
/// `((i*nvir_a + a)*nocc_b + J)*nvir_b + B` — the αβ layout
/// [`crate::ump2::ump2_kernel`] consumes. Q-fold via [`oracle_dot`].
fn assemble_cross_spin(
    ovl_a: &[f64],
    nocc_a: usize,
    nvir_a: usize,
    ovl_b: &[f64],
    nocc_b: usize,
    nvir_b: usize,
    naux: usize,
) -> Vec<f64> {
    let mut ovov = vec![0.0_f64; nocc_a * nvir_a * nocc_b * nvir_b];
    for i in 0..nocc_a {
        for a in 0..nvir_a {
            let ia_base = (i * nvir_a + a) * naux;
            let b_ia = &ovl_a[ia_base..ia_base + naux];
            for jj in 0..nocc_b {
                for bb in 0..nvir_b {
                    let jb_base = (jj * nvir_b + bb) * naux;
                    let b_jb = &ovl_b[jb_base..jb_base + naux];
                    let val = oracle_dot(b_ia, b_jb);
                    ovov[((i * nvir_a + a) * nocc_b + jj) * nvir_b + bb] = val;
                }
            }
        }
    }
    ovov
}

/// Build the three spin-block DF `(ia|jb)` ERIs (αα, αβ, ββ) for one
/// [`DFUMP2`] reference, then run the reused [`ump2_kernel`]
/// (`DFUMP2.kernel`, `dfump2.py`).
///
/// The αα and ββ same-spin blocks are [`df_ao2mo`] on the respective spin
/// reference; the αβ block is the cross-spin Q-contraction
/// [`assemble_cross_spin`]. The shared `*-ri` aux drives both channels.
///
/// # Errors
/// `?`-propagates any error from [`df_ao2mo`] / the MO transform / the reused
/// [`ump2_kernel`]. The `int3c2e_sph` cintx#11 gate is surfaced when the caller
/// builds `df`; this never panics and never substitutes a zero buffer
/// (T-05-05-FFI).
pub fn dfump2_kernel(
    refr: &UmpReference,
    frozen: &Frozen,
    df: &DfIntegrals,
    with_t2: bool,
) -> Result<UmpResult, PyscfRsError> {
    // Same-spin αα and ββ blocks via the closed-shell DF ao2mo per spin.
    let eris_aa = df_ao2mo(&refr.alpha, frozen, df)?;
    let eris_bb = df_ao2mo(&refr.beta, frozen, df)?;

    // αβ cross-spin block: transform the B-tensor into each spin's (occ,vir)
    // MO block, then Q-contract across spins.
    let elements_a = refr
        .alpha
        .mol
        .atom_charges()
        .into_iter()
        .map(|z| z.max(0) as u32)
        .collect::<Vec<u32>>();
    let mask_a = frozen::frozen_mask(
        frozen,
        &refr.alpha.mo_occ,
        &refr.alpha.mo_energy,
        &elements_a,
    )?;
    let elements_b = refr
        .beta
        .mol
        .atom_charges()
        .into_iter()
        .map(|z| z.max(0) as u32)
        .collect::<Vec<u32>>();
    let mask_b = frozen::frozen_mask(frozen, &refr.beta.mo_occ, &refr.beta.mo_energy, &elements_b)?;

    let co_a = mo_subset(&refr.alpha, &mask_a, true)?;
    let cv_a = mo_subset(&refr.alpha, &mask_a, false)?;
    let co_b = mo_subset(&refr.beta, &mask_b, true)?;
    let cv_b = mo_subset(&refr.beta, &mask_b, false)?;

    let ovl_a = transform_b_to_ov(df, &co_a, &cv_a)?;
    let ovl_b = transform_b_to_ov(df, &co_b, &cv_b)?;

    let eris_ab_ovov = assemble_cross_spin(
        &ovl_a, co_a.nmo, cv_a.nmo, &ovl_b, co_b.nmo, cv_b.nmo, df.naux,
    );
    let eris_ab = ChemistsEris {
        ovov: eris_ab_ovov,
        nocc: co_a.nmo,
        nvir: cv_a.nmo,
    };

    ump2_kernel(refr, frozen, &eris_aa, &eris_ab, &eris_bb, with_t2)
}
