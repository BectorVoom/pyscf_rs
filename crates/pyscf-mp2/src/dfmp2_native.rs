//! Native RI-MP2 fast path — port of `pyscf/mp/dfmp2_native.py`
//! (`ints3c_cholesky` + `emp2_rhf` + `solve_cphf_rhf`) and the open-shell
//! analog `pyscf/mp/dfump2_native.py` (`emp2_uhf`).
//!
//! ## What this is (MP2-04 native half, D-06 additional target)
//!
//! This is a SEPARATE, DISTINCT module from the conventional DF-MP2 path
//! ([`crate::dfmp2`]). Upstream exposes it on its OWN module path
//! `pyscf.mp.dfmp2_native` (module aliases within that namespace:
//! `MP2 = RMP2 = DFMP2 = DFRMP2`), and crucially its `DFRMP2` *subclasses
//! `lib.StreamObject`* (`dfmp2_native.py:31`) — NOT `mp2.RMP2` like the
//! conventional [`crate::dfmp2::DFRMP2`] does. So this is NOT the default
//! `mp.DFMP2` factory (which is the conventional path from plan 05-05); it is a
//! deliberate scope expansion the user chose ("Both" DF-MP2 variants, D-06).
//!
//! The native path differs from the conventional one in the SHAPE of the
//! contraction, not the answer: it walks occupied PAIRS `(i, j)` with `j ≤ i`
//! and forms a per-pair `Kab[a,b] = (ia|jb)` from the MO-transformed B-tensor,
//! whereas the conventional path materializes the full `(ia|jb)` ovov block and
//! runs the in-core RMP2 closed form. For `ps = pt = 1.0` the two produce the
//! SAME total correlation energy on the same integrals — the always-on
//! structural cross-check (plan 05-06 Task 2) asserts exactly that.
//!
//! ## The native B-tensor (`ints3c_cholesky` → `ints_cholesky[i, Q, b]`)
//!
//! Upstream `ints3c_cholesky` builds, per occupied `i`, a matrix
//! `ints_cholesky[i, Q, b]` — the 3-center MO integral `(i b | Q)` fitted by the
//! Cholesky factor `L⁻¹` of the `(P|Q)` Coulomb metric. That fitted MO tensor is
//! EXACTLY the DF B-tensor `B^Q_ib` that [`pyscf_df::cholesky_eri`] already
//! produces (Cholesky-of-`(P|Q)` forward-substitution into the 3-center AO
//! integral), MO-transformed into the `(occ, vir)` block. We therefore REUSE the
//! shipped 3c Cholesky (Don't-Hand-Roll: NO second Cholesky here) — the caller
//! builds a [`DfIntegrals`] via
//! `pyscf_df::cholesky_eri(mol, pyscf_df::default_ri(basis))` (A2: the mp2fit
//! `*-ri` aux — `make_auxbasis(mol, mp2fit=True)` at `dfmp2_native.py:74`, NOT
//! the JK-fit aux) and we transform `b_uvq` into the native `[i, Q, b]` layout.
//!
//! ## The native energy contraction (`emp2_rhf`, `dfmp2_native.py:374-427`)
//!
//! ```text
//! for i in occ_act:
//!   for j < i:
//!     Kab[a,b] = Σ_Q B^Q_ia · B^Q_jb            # = (ia|jb)
//!     DE[a,b]  = εi + εj − εa − εb
//!     Tab      = Kab / DE
//!     E += 2(ps+pt)·Σ_ab Tab·Kab − 2·pt·Σ_ab Tab·Kabᵀ
//!   # j == i diagonal pair:
//!   Kab[a,b] = Σ_Q B^Q_ia · B^Q_ib
//!   DE[a,b]  = 2εi − εa − εb
//!   Tab      = Kab / DE
//!   E += ps·Σ_ab Tab·Kab
//! ```
//!
//! With the default `ps = pt = 1.0` this is plain RI-MP2. Every reduction — the
//! per-Q `Kab` dot AND the energy `Σ Tab·Kab` — goes through
//! [`oracle_dot`]/[`oracle_sum`]; NO `+=` accumulation of the float energy
//! (T-05-06-FP, Pitfall 1/2). NO C dependency (`lib.dot`/`lib.einsum` ported as
//! oracle reductions).
//!
//! ## The native UHF contraction (`emp2_uhf`, `dfump2_native.py:272-355`)
//!
//! Same-spin (αα, ββ): `pt`-scaled, antisymmetrized `Tab = (Kab − Kabᵀ)/DE`,
//! over `j < i` pairs within a spin. Opposite-spin (αβ): `ps`-scaled, DIRECT
//! `Tab = Kab/DE`, over all `i(α) × j(β)` pairs. `E = E_αα + E_ββ + E_αβ`.
//!
//! ## cintx#11 gating (D-05)
//!
//! [`pyscf_df::cholesky_eri`] builds `b_uvq` from `int3c2e_sph`, a cintx-ops
//! upstream gap. The native kernels receive an already-built [`DfIntegrals`] and
//! `?`-propagate any shape error; the `int3c2e_sph` gate is surfaced at the
//! caller's `cholesky_eri` and propagates with `?` — NEVER a panic, NEVER a
//! zero-substitution (the Phase-4 CR-02 lesson; T-05-06-FFI). Structural tests
//! are always-on; the live-PySCF numeric parity lives in the cintx-gated
//! `dfmp2_native_energy` oracle arm registered in plan 05-01.

use crate::frozen::{self, Frozen};
use crate::mp2::{Mp2Reference, Mp2Result};
use crate::ump2::{UmpReference, UmpResult};
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
/// mask. `want_occupied = true` selects active occupied columns; `false` selects
/// active virtual columns. Returns the subset column-major `[nao, n_selected]`
/// (mirrors [`crate::dfmp2`]'s local `mo_subset` — kept local so the native path
/// has no cross-module private dependency).
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

/// Active occupied / virtual MO energies for one reference, in column order,
/// honouring the frozen mask. Returns `(e_occ, e_vir)`.
fn active_energies(
    refr: &Mp2Reference,
    frozen: &Frozen,
) -> Result<(Vec<f64>, Vec<f64>), PyscfRsError> {
    let elements = reference_elements(refr);
    let mask = frozen::frozen_mask(frozen, &refr.mo_occ, &refr.mo_energy, &elements)?;
    let mut e_occ = Vec::new();
    let mut e_vir = Vec::new();
    for (col, &active) in mask.iter().enumerate() {
        if !active {
            continue;
        }
        let occ = refr.mo_occ.get(col).copied().unwrap_or(0.0);
        let e = refr.mo_energy.get(col).copied().unwrap_or(0.0);
        if occ > 0.0 {
            e_occ.push(e);
        } else {
            e_vir.push(e);
        }
    }
    Ok((e_occ, e_vir))
}

/// MO-transform the DF B-tensor into the NATIVE `[i, Q, b]` layout:
/// `ints_cholesky[i, Q, b] = B^Q_ib` (upstream `ints3c_cholesky` indexing order
/// `[mo1, aux, mo2]`, `dfmp2_native.py:344`).
///
/// Input `df.b_uvq` is AO-basis ROW-MAJOR `[nao, nao, naux]` — element
/// `(μ, ν, Q)` at `μ*nao*naux + ν*naux + Q` (the [`DfIntegrals`] layout). `co` /
/// `cv` are column-major `[nao, nocc]` / `[nao, nvir]` MO blocks.
///
/// Output is ROW-MAJOR `[nocc, naux, nvir]` — element `(i, Q, b)` at
/// `(i*naux + Q)*nvir + b`. Computed as
/// `B^Q_ib = Σ_{μ,ν} C_μ^i · b_uvq[μ,ν,Q] · C_ν^b` via the same
/// materialize-then-[`oracle_sum`] two-index transform [`crate::dfmp2`] uses —
/// only the OUTPUT stride differs (native wants aux as the middle index so each
/// occupied `i` slice `ints[i, :, :]` is a contiguous `[naux, nvir]` matrix, the
/// I/O-optimal layout upstream picks for the per-`i` `Kab` contraction). NO `+=`
/// accumulation (T-05-06-FP).
fn transform_b_to_iqb(
    df: &DfIntegrals,
    co: &MOCoefficients,
    cv: &MOCoefficients,
) -> Result<Vec<f64>, PyscfRsError> {
    let nao = df.nao;
    let naux = df.naux;
    let nocc = co.nmo;
    let nvir = cv.nmo;

    // Shape guards (T-05-06-FFI caller→transform boundary).
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

    let mut ints = vec![0.0_f64; nocc * naux * nvir];

    // Scratch reused across the loop bodies (materialize-then-oracle_sum).
    let mut prod_mu: Vec<f64> = vec![0.0; nao];
    let mut prod_nu: Vec<f64> = vec![0.0; nao];
    let mut half_inu: Vec<f64> = vec![0.0; nao]; // half[ν] = Σ_μ C_μ^i·b[μ,ν,Q]

    for i in 0..nocc {
        let ci = &co.data[i * nao..(i + 1) * nao]; // C_μ^i
        for q in 0..naux {
            // half_inu[ν] = Σ_μ C_μ^i · b_uvq[μ,ν,Q].
            for (nu, half) in half_inu.iter_mut().enumerate() {
                for (mu, pm) in prod_mu.iter_mut().enumerate() {
                    *pm = ci[mu] * df.b_uvq[mu * nao * naux + nu * naux + q];
                }
                *half = oracle_sum(&prod_mu);
            }
            // B^Q_ib = Σ_ν half_inu[ν] · C_ν^b → ints[(i*naux + Q)*nvir + b].
            for b in 0..nvir {
                let cb = &cv.data[b * nao..(b + 1) * nao]; // C_ν^b
                for (nu, pn) in prod_nu.iter_mut().enumerate() {
                    *pn = half_inu[nu] * cb[nu];
                }
                ints[(i * naux + q) * nvir + b] = oracle_sum(&prod_nu);
            }
        }
    }
    Ok(ints)
}

/// Build the per-pair `Kab[a,b] = Σ_Q B^Q_ia · B^Q_jb = (ia|jb)` from two native
/// `[i, Q, b]` slices.
///
/// `ints_i` is the `[naux, nvir]` slice for occupied `i` (`ints[i, :, :]`) and
/// `ints_j` the slice for `j`. The result is the `[nvir_i, nvir_j]` matrix
/// (row-major `K[a*nvir_j + b]`). This is the MATH of upstream
/// `Kab = lib.dot(ints3c_ia.T, ints3c_jb)` (`dfmp2_native.py:415`) — a Q-fold
/// per `(a,b)` through [`oracle_dot`], NO `+=` accumulation (T-05-06-FP).
fn kab_from_slices(
    ints_i: &[f64],
    ints_j: &[f64],
    naux: usize,
    nvir_i: usize,
    nvir_j: usize,
) -> Vec<f64> {
    // Gather each virtual column a (resp. b) as a contiguous length-naux vector
    // so the Q-fold is a single oracle_dot. ints_i is row-major [naux, nvir_i]:
    // element (Q, a) at Q*nvir_i + a.
    let mut col_a: Vec<f64> = vec![0.0; naux];
    let mut col_b: Vec<f64> = vec![0.0; naux];
    let mut k = vec![0.0_f64; nvir_i * nvir_j];
    for a in 0..nvir_i {
        for (q, ca) in col_a.iter_mut().enumerate() {
            *ca = ints_i[q * nvir_i + a];
        }
        for b in 0..nvir_j {
            for (q, cb) in col_b.iter_mut().enumerate() {
                *cb = ints_j[q * nvir_j + b];
            }
            k[a * nvir_j + b] = oracle_dot(&col_a, &col_b);
        }
    }
    k
}

/// Native RI-MP2 closed-shell correlation energy (`emp2_rhf`,
/// `dfmp2_native.py:374-427`), with `ps = pt = 1.0` (plain RI-MP2).
///
/// Maps upstream `ints3c_cholesky` to [`pyscf_df::cholesky_eri`] (the shipped 3c
/// Cholesky — NO second Cholesky here) MO-transformed into the native
/// `[i, Q, b]` layout, then walks occupied pairs `(i, j ≤ i)` forming the
/// per-pair `Kab = (ia|jb)` and accumulating
/// `E += 2·(ps+pt)·Σ Tab·Kab − 2·pt·Σ Tab·Kabᵀ` for `j < i` and
/// `E += ps·Σ Tab·Kab` for the `j == i` diagonal.
///
/// EVERY reduction (the per-Q `Kab` fold AND the per-pair energy `Σ Tab·Kab`)
/// goes through [`oracle_dot`]/[`oracle_sum`]; the float energy is the
/// [`oracle_sum`] of the per-pair contributions, never a running `+=`
/// (T-05-06-FP, bit-exact + thread-count invariant).
///
/// # Errors
/// `?`-propagates a B-tensor / MO-block / occupied-count mismatch (via
/// [`crate::error::Mp2Error::ShapeMismatch`]) and any frozen-mask resolution
/// error — NEVER panics, NEVER indexes OOB (T-05-06-FFI). The `int3c2e_sph`
/// cintx#11 gate is surfaced by the caller that builds `df`; this function
/// receives an already-built tensor and never substitutes a zero buffer.
pub fn emp2_rhf(
    refr: &Mp2Reference,
    frozen: &Frozen,
    df: &DfIntegrals,
) -> Result<f64, PyscfRsError> {
    // SCS factors fixed at the plain-MP2 default (ps = pt = 1.0). The native
    // upstream surfaces ps/pt as method attributes (SCSDFRMP2); the SCS split is
    // already provided generically by crate::mp2::scs_energy, so the native fast
    // path keeps the energy at the plain default.
    let ps = 1.0_f64;
    let pt = 1.0_f64;

    let elements = reference_elements(refr);
    let mask = frozen::frozen_mask(frozen, &refr.mo_occ, &refr.mo_energy, &elements)?;
    let co = mo_subset(refr, &mask, true)?;
    let cv = mo_subset(refr, &mask, false)?;
    let nocc = co.nmo;
    let nvir = cv.nmo;
    let naux = df.naux;

    let (e_occ, e_vir) = active_energies(refr, frozen)?;
    if e_occ.len() != nocc || e_vir.len() != nvir {
        return Err(crate::error::Mp2Error::ShapeMismatch {
            expected: nocc * nvir,
            got: e_occ.len() * e_vir.len(),
        }
        .into());
    }

    // ints[i, Q, b] = B^Q_ib (native layout).
    let ints = transform_b_to_iqb(df, &co, &cv)?;
    let stride = naux * nvir; // per-i slice length

    // Per-pair contributions accumulated into a Vec then oracle_sum'd (NO `+=`).
    let mut e_terms: Vec<f64> = Vec::new();

    // Scratch reused across pairs.
    let mut tab: Vec<f64> = vec![0.0; nvir * nvir];
    let mut kab_t: Vec<f64> = vec![0.0; nvir * nvir]; // Kabᵀ reordered to [a,b]

    for i in 0..nocc {
        let ints_i = &ints[i * stride..(i + 1) * stride];
        // j < i pairs.
        for j in 0..i {
            let ints_j = &ints[j * stride..(j + 1) * stride];
            let kab = kab_from_slices(ints_i, ints_j, naux, nvir, nvir);
            // DE[a,b] = εi + εj − εa − εb; Tab = Kab/DE.
            for a in 0..nvir {
                for b in 0..nvir {
                    let de = e_occ[i] + e_occ[j] - e_vir[a] - e_vir[b];
                    tab[a * nvir + b] = kab[a * nvir + b] / de;
                    kab_t[a * nvir + b] = kab[b * nvir + a];
                }
            }
            // E += 2(ps+pt)·Σ Tab·Kab − 2·pt·Σ Tab·Kabᵀ.
            let direct = oracle_dot(&tab, &kab);
            let exch = oracle_dot(&tab, &kab_t);
            e_terms.push(2.0 * (ps + pt) * direct);
            e_terms.push(-2.0 * pt * exch);
        }
        // j == i diagonal pair.
        let kab = kab_from_slices(ints_i, ints_i, naux, nvir, nvir);
        for a in 0..nvir {
            for b in 0..nvir {
                let de = 2.0 * e_occ[i] - e_vir[a] - e_vir[b];
                tab[a * nvir + b] = kab[a * nvir + b] / de;
            }
        }
        // E += ps·Σ Tab·Kab.
        e_terms.push(ps * oracle_dot(&tab, &kab));
    }

    Ok(oracle_sum(&e_terms))
}

/// Native RI-MP2 open-shell correlation energy (`emp2_uhf`,
/// `dfump2_native.py:272-355`), with `ps = pt = 1.0`.
///
/// `refr` carries the α/β reference pair; `df_a`/`df_b` are the per-spin DF
/// B-tensors (same shared `*-ri` aux). Same-spin (αα, ββ) channels are `pt`-
/// scaled and antisymmetrized (`Tab = (Kab − Kabᵀ)/DE`, over `j < i` pairs);
/// the opposite-spin (αβ) channel is `ps`-scaled and DIRECT (`Tab = Kab/DE`,
/// over all `i(α) × j(β)` pairs). `E = E_αα + E_ββ + E_αβ`.
///
/// Every reduction goes through [`oracle_dot`]/[`oracle_sum`]; the energy is the
/// [`oracle_sum`] of per-pair contributions (NO `+=`; T-05-06-FP).
///
/// # Errors
/// `?`-propagates a shape / frozen-mask error from either spin channel — never
/// panics, never indexes OOB (T-05-06-FFI).
pub fn emp2_uhf(
    refr: &UmpReference,
    frozen: &Frozen,
    df_a: &DfIntegrals,
    df_b: &DfIntegrals,
) -> Result<f64, PyscfRsError> {
    let ps = 1.0_f64;
    let pt = 1.0_f64;

    // Per-spin MO transform into the native [i, Q, b] layout.
    let mask_a = {
        let el = reference_elements(&refr.alpha);
        frozen::frozen_mask(frozen, &refr.alpha.mo_occ, &refr.alpha.mo_energy, &el)?
    };
    let mask_b = {
        let el = reference_elements(&refr.beta);
        frozen::frozen_mask(frozen, &refr.beta.mo_occ, &refr.beta.mo_energy, &el)?
    };
    let co_a = mo_subset(&refr.alpha, &mask_a, true)?;
    let cv_a = mo_subset(&refr.alpha, &mask_a, false)?;
    let co_b = mo_subset(&refr.beta, &mask_b, true)?;
    let cv_b = mo_subset(&refr.beta, &mask_b, false)?;

    let (eo_a, ev_a) = active_energies(&refr.alpha, frozen)?;
    let (eo_b, ev_b) = active_energies(&refr.beta, frozen)?;

    let nocc_a = co_a.nmo;
    let nvir_a = cv_a.nmo;
    let nocc_b = co_b.nmo;
    let nvir_b = cv_b.nmo;
    let naux_a = df_a.naux;
    let naux_b = df_b.naux;

    if eo_a.len() != nocc_a || ev_a.len() != nvir_a || eo_b.len() != nocc_b || ev_b.len() != nvir_b
    {
        return Err(crate::error::Mp2Error::ShapeMismatch {
            expected: nocc_a * nvir_a + nocc_b * nvir_b,
            got: eo_a.len() * ev_a.len() + eo_b.len() * ev_b.len(),
        }
        .into());
    }

    let ints_a = transform_b_to_iqb(df_a, &co_a, &cv_a)?;
    let ints_b = transform_b_to_iqb(df_b, &co_b, &cv_b)?;

    let mut e_terms: Vec<f64> = Vec::new();

    // Same-spin αα and ββ (pt-scaled, antisymmetrized, j < i pairs).
    same_spin_native(
        &ints_a,
        naux_a,
        nocc_a,
        nvir_a,
        &eo_a,
        &ev_a,
        pt,
        &mut e_terms,
    );
    same_spin_native(
        &ints_b,
        naux_b,
        nocc_b,
        nvir_b,
        &eo_b,
        &ev_b,
        pt,
        &mut e_terms,
    );

    // Opposite-spin αβ (ps-scaled, DIRECT, all i(α) × j(β) pairs).
    let stride_a = naux_a * nvir_a;
    let stride_b = naux_b * nvir_b;
    let mut tab: Vec<f64> = vec![0.0; nvir_a * nvir_b];
    for i in 0..nocc_a {
        let ints_i = &ints_a[i * stride_a..(i + 1) * stride_a];
        for j in 0..nocc_b {
            let ints_j = &ints_b[j * stride_b..(j + 1) * stride_b];
            let kab = kab_from_slices(ints_i, ints_j, naux_a.min(naux_b), nvir_a, nvir_b);
            for a in 0..nvir_a {
                for b in 0..nvir_b {
                    let de = eo_a[i] + eo_b[j] - ev_a[a] - ev_b[b];
                    tab[a * nvir_b + b] = kab[a * nvir_b + b] / de;
                }
            }
            e_terms.push(ps * oracle_dot(&tab, &kab));
        }
    }

    Ok(oracle_sum(&e_terms))
}

/// One same-spin native channel (`dfump2_native.py:308-331`): `pt`-scaled,
/// antisymmetrized `Tab = (Kab − Kabᵀ)/DE`, over `j < i` pairs. Pushes per-pair
/// `pt·Σ Tab·Kab` contributions into `e_terms` (folded by the caller's
/// [`oracle_sum`]; NO `+=`).
#[allow(clippy::too_many_arguments)]
fn same_spin_native(
    ints: &[f64],
    naux: usize,
    nocc: usize,
    nvir: usize,
    e_occ: &[f64],
    e_vir: &[f64],
    pt: f64,
    e_terms: &mut Vec<f64>,
) {
    let stride = naux * nvir;
    let mut tab: Vec<f64> = vec![0.0; nvir * nvir];
    for i in 0..nocc {
        let ints_i = &ints[i * stride..(i + 1) * stride];
        for j in 0..i {
            let ints_j = &ints[j * stride..(j + 1) * stride];
            let kab = kab_from_slices(ints_i, ints_j, naux, nvir, nvir);
            // Tab = (Kab − Kabᵀ)/DE.
            for a in 0..nvir {
                for b in 0..nvir {
                    let de = e_occ[i] + e_occ[j] - e_vir[a] - e_vir[b];
                    tab[a * nvir + b] = (kab[a * nvir + b] - kab[b * nvir + a]) / de;
                }
            }
            e_terms.push(pt * oracle_dot(&tab, &kab));
        }
    }
}

/// The native closed-shell DF-MP2 driver state
/// (`DFRMP2(lib.StreamObject)`, `dfmp2_native.py:31` — a DISTINCT class from the
/// conventional [`crate::dfmp2::DFRMP2`]: it does NOT subclass `mp2.RMP2`).
///
/// Holds the SCF reference snapshot plus the pre-assembled native B-tensor
/// (built by the caller via
/// `pyscf_df::cholesky_eri(mol, pyscf_df::default_ri(basis))` — A2: the mp2fit
/// `*-ri` aux). [`NativeDFRMP2::kernel`] wraps [`emp2_rhf`].
#[derive(Debug, Clone)]
pub struct NativeDFRMP2 {
    /// Converged SCF reference.
    pub reference: Mp2Reference,
    /// Pre-assembled DF B-integrals (`*-ri` aux). The `int3c2e_sph` cintx#11
    /// gate is `?`-propagated by the caller's `cholesky_eri`, never panicked or
    /// zero-substituted here.
    pub df: DfIntegrals,
    /// Frozen-core spec.
    pub frozen: Frozen,
}

impl NativeDFRMP2 {
    /// Run the native RI-MP2 fast path: [`emp2_rhf`] over the stored B-tensor.
    ///
    /// Returns an [`Mp2Result`] carrying the native correlation energy. The
    /// native fast path computes the ENERGY only (the per-pair contraction does
    /// not materialize the full amplitude block), so `e_ss`/`e_os` are not
    /// separately resolved and `t2` is `None` (the conventional [`crate::dfmp2`]
    /// path provides the spin-resolved decomposition + amplitudes). `e_ss`/`e_os`
    /// are reported as `(0, e_corr)` placeholders — the native deliverable is
    /// `e_corr` (MP2-04 native half); callers needing the SS/OS split use the
    /// conventional path.
    ///
    /// # Errors
    /// `?`-propagates any error from [`emp2_rhf`].
    pub fn kernel(&self) -> Result<Mp2Result, PyscfRsError> {
        let e_corr = emp2_rhf(&self.reference, &self.frozen, &self.df)?;
        Ok(Mp2Result {
            e_corr,
            e_ss: 0.0,
            e_os: e_corr,
            t2: None,
        })
    }
}

/// The native open-shell DF-MP2 driver state (`DFUMP2(DFRMP2)`,
/// `dfump2_native.py:30`). Holds the α/β reference pair plus the per-spin native
/// B-tensors (shared `*-ri` aux). [`NativeDFUMP2::kernel`] wraps [`emp2_uhf`].
#[derive(Debug, Clone)]
pub struct NativeDFUMP2 {
    /// α/β unrestricted reference pair.
    pub reference: UmpReference,
    /// α-spin DF B-integrals.
    pub df_alpha: DfIntegrals,
    /// β-spin DF B-integrals.
    pub df_beta: DfIntegrals,
    /// Frozen-core spec.
    pub frozen: Frozen,
}

impl NativeDFUMP2 {
    /// Run the native open-shell RI-MP2 fast path: [`emp2_uhf`] over the stored
    /// per-spin B-tensors. Returns a [`UmpResult`] carrying the native
    /// correlation energy. As with [`NativeDFRMP2::kernel`] the native path
    /// computes the total `e_corr` only; the per-channel `e_aa`/`e_ab`/`e_bb`
    /// split + amplitudes are the conventional [`crate::dfmp2`] path's surface,
    /// so they are reported as `(0, e_corr, 0)` placeholders and `t2 = None`.
    ///
    /// # Errors
    /// `?`-propagates any error from [`emp2_uhf`].
    pub fn kernel(&self) -> Result<UmpResult, PyscfRsError> {
        let e_corr = emp2_uhf(&self.reference, &self.frozen, &self.df_alpha, &self.df_beta)?;
        Ok(UmpResult {
            e_corr,
            e_aa: 0.0,
            e_ab: e_corr,
            e_bb: 0.0,
            t2: None,
        })
    }
}

/// CPHF-relaxed-RDM response solve (`solve_cphf_rhf`, `dfmp2_native.py:752-779`)
/// — STATUS-MARKER STUB (D-06 "the relaxed RDM may be staged behind a status
/// marker").
///
/// ## Decision (D-06): energy fast-path is the core MP2-04 deliverable
///
/// The native RI-MP2 ENERGY (`emp2_rhf`/`emp2_uhf`) is the core MP2-04 native
/// half and ships fully wired in this plan. The CPHF-relaxed 1-RDM
/// (`make_rdm1_relaxed`) is the OPTIONAL native extra: it requires the full
/// orbital-gradient machinery (`rmp2_densities_contribs` +
/// `orbgrad_from_Gamma`, `dfmp2_native.py:479-724`) PLUS a working
/// `fock_response_rhf` (which itself needs arity-4 `int2e` — the same cintx#11
/// gap the conventional path's `default_ao2mo` waits on). Wiring the full
/// gradient surface here would be premature: the response RHS `Lvo` is not yet
/// producible un-gated.
///
/// ## The intended landing
///
/// When the relaxed RDM is wired (a follow-on once arity-4 `int2e` lands), the
/// CPHF linear system
///
/// ```text
/// (εi − εa)·z[a,i] − Σ_bj [4(ai|bj) − (ab|ij) − (aj|ib)]·z[b,j] = Lvo[a,i]
/// ```
///
/// is solved through [`pyscf_algebra::solve_linear`] (already wired in
/// pyscf-algebra, lib.rs:56) — the response operator assembled into a dense
/// `A·z = Lvo` system and handed to `solve_linear(&a, &lvo, n)`. This keeps the
/// solve on the algebra wall (no hand-rolled iterative CPHF here) and `?`-
/// propagates any [`pyscf_algebra::AlgebraError`] — never panics (T-05-06-FFI).
///
/// # Errors
/// Returns [`crate::error::Mp2Error::NotYetImplemented`] (`plan = 5`) — the
/// status marker. NEVER panics, NEVER substitutes a wrong density.
pub fn solve_cphf_rhf(_refr: &Mp2Reference, _lvo: &[f64]) -> Result<Vec<f64>, PyscfRsError> {
    Err(crate::error::Mp2Error::NotYetImplemented { plan: 5 }.into())
}
