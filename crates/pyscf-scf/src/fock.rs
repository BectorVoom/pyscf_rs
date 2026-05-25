//! Fock build — h_core / S / J,K / V_HF / F. Body filled in plan 03-11.
//!
//! Source:
//!   - `pyscf/scf/hf.py:316-345` — `get_hcore` (kin + nuc)
//!   - `pyscf/scf/hf.py:346-360` — `get_ovlp`
//!   - `pyscf/scf/hf.py:957-1033` — `get_jk` (J + K via int2e)
//!   - `pyscf/scf/hf.py:1034-1085` — `get_veff` (J - 0.5K for RHF)
//!   - `pyscf/scf/hf.py:1086-1135` — `get_fock` (DIIS adapter slot)
//!
//! Notes:
//!   - All AO-basis matrices (h_core, S, D, F, J, K) are symmetric for
//!     closed-shell RHF, so F-order (intor's output) and row-major
//!     (`Density.data`'s documented convention) give bit-identical
//!     buffers. No transpose is needed when copying `IntorOutput.values`
//!     into `Density.data`.
//!   - `default_get_jk` requires the Phase 2 arity-4 `int2e_sph` dispatcher
//!     (plan 02-09 verification rollup). Until that lands the function
//!     propagates `NotYetImplemented{phase:2}` from `pyscf_gto::intor`.
//!     Plan 03-05 (DF-HF) provides an override path that bypasses
//!     int2e_sph entirely by routing through pyscf-df's DfIntegrals.
//!   - `default_get_fock` builds the BASE Fock (`F = h1e + V_HF`); the
//!     CDIIS extrapolation step has been hoisted one layer up into
//!     `crate::kernel_impl::scf_loop`, which calls
//!     `crate::diis_adapter::diis_step` AFTER `hooks.get_fock(...)`.
//!     This keeps the SCF-08 hook signature simple and lets any
//!     OverrideHooks impl share the kernel-loop DIIS without
//!     reimplementing it.

use pyscf_core::{Density, Mole, PyscfRsError};

/// `h_core(mol) = T + V_nuc` (1-electron Hamiltonian).
///
/// Source: `pyscf/scf/hf.py:316-345`. Returns an `nao×nao` `Density` (this
/// type is overloaded for any nao×nao matrix in pyscf-core's Phase 1
/// surface — see `pyscf-core/src/density.rs`).
pub fn default_get_hcore(mol: &Mole) -> Result<Density, PyscfRsError> {
    let t = pyscf_gto::intor(mol, "int1e_kin")?;
    let v = pyscf_gto::intor(mol, "int1e_nuc")?;
    let nao = mol.nao_nr;
    if t.values.len() != nao * nao || v.values.len() != nao * nao {
        return Err(PyscfRsError::Core(
            pyscf_core::CoreError::DimensionMismatch {
                expected: nao * nao,
                actual: t.values.len(),
            },
        ));
    }
    let data: Vec<f64> = t.values.iter().zip(&v.values).map(|(a, b)| a + b).collect();
    Ok(Density { nao, data })
}

/// `S = int1e_ovlp` (overlap matrix).
pub fn default_get_ovlp(mol: &Mole) -> Result<Density, PyscfRsError> {
    let s = pyscf_gto::intor(mol, "int1e_ovlp")?;
    let nao = mol.nao_nr;
    if s.values.len() != nao * nao {
        return Err(PyscfRsError::Core(
            pyscf_core::CoreError::DimensionMismatch {
                expected: nao * nao,
                actual: s.values.len(),
            },
        ));
    }
    Ok(Density {
        nao,
        data: s.values,
    })
}

/// `(J, K)` build from `int2e_sph` and the density matrix.
///
/// J[μν] = Σ_λσ (μν|λσ) D[λσ]
/// K[μν] = Σ_λσ (μλ|νσ) D[λσ]
///
/// Phase 2 status (plan 02-09 verification rollup gap): `pyscf_gto::intor`
/// returns `NotYetImplemented{phase:2}` for arity-4 ERIs. This function
/// propagates that error verbatim until the rollup ships. Override path:
/// implement `OverrideHooks::get_jk` via pyscf-df (plan 03-05) — DfHooks
/// bypasses int2e_sph entirely.
///
/// All λσ-reductions go through `pyscf_algebra::oracle_sum` (Pitfall 9
/// mitigation) once the ERIs are available.
pub fn default_get_jk(mol: &Mole, dm: &Density) -> Result<(Density, Density), PyscfRsError> {
    let nao = mol.nao_nr;
    if dm.nao != nao {
        return Err(PyscfRsError::Core(
            pyscf_core::CoreError::DimensionMismatch {
                expected: nao,
                actual: dm.nao,
            },
        ));
    }
    // Fetch (μν|λσ) ERIs. For now this returns NotYetImplemented{phase:2}
    // (plan 02-09 rollup gap); the `?` propagates that to the kernel
    // caller, which surfaces a structured error rather than panicking.
    let eri = pyscf_gto::intor(mol, "int2e")?;
    let expected = nao * nao * nao * nao;
    if eri.values.len() != expected {
        return Err(PyscfRsError::Core(
            pyscf_core::CoreError::DimensionMismatch {
                expected,
                actual: eri.values.len(),
            },
        ));
    }
    // intor returns F-order `[nao, nao, nao, nao]`. For arity-4 the
    // exact F-order index of (μ, ν, λ, σ) is `μ + ν*nao + λ*nao^2 + σ*nao^3`.
    // ERIs are 8-fold symmetric under (μν|λσ) == (νμ|λσ) == (μν|σλ) ==
    // (λσ|μν), so this layout works regardless of intor's choice between
    // chemist's and physicist's notation conventions.
    //
    // J[μν] = Σ_{λσ} eri[μ,ν,λ,σ] · D[λ,σ]
    // K[μν] = Σ_{λσ} eri[μ,λ,ν,σ] · D[λ,σ]
    let mut j_data = vec![0.0_f64; nao * nao];
    let mut k_data = vec![0.0_f64; nao * nao];
    let mut j_terms = vec![0.0_f64; nao * nao];
    let mut k_terms = vec![0.0_f64; nao * nao];
    let n2 = nao * nao;
    let n3 = n2 * nao;
    for mu in 0..nao {
        for nu in 0..nao {
            for lambda in 0..nao {
                for sigma in 0..nao {
                    // (μν|λσ) F-order index
                    let i_j = mu + nu * nao + lambda * n2 + sigma * n3;
                    // (μλ|νσ) for K
                    let i_k = mu + lambda * nao + nu * n2 + sigma * n3;
                    let d_ls = dm.data[lambda * nao + sigma];
                    j_terms[lambda * nao + sigma] = eri.values[i_j] * d_ls;
                    k_terms[lambda * nao + sigma] = eri.values[i_k] * d_ls;
                }
            }
            // Reduction over (λ, σ) — Pitfall 9 mitigation.
            j_data[mu * nao + nu] = pyscf_algebra::oracle_sum(&j_terms);
            k_data[mu * nao + nu] = pyscf_algebra::oracle_sum(&k_terms);
        }
    }
    Ok((Density { nao, data: j_data }, Density { nao, data: k_data }))
}

/// `V_HF = J - 0.5 · K` (RHF closed-shell effective potential).
pub fn default_get_veff(mol: &Mole, dm: &Density) -> Result<Density, PyscfRsError> {
    let (j, k) = default_get_jk(mol, dm)?;
    let nao = j.nao;
    let data: Vec<f64> = j
        .data
        .iter()
        .zip(&k.data)
        .map(|(a, b)| a - 0.5 * b)
        .collect();
    Ok(Density { nao, data })
}

/// `F = h1e + V_HF` — BASE Fock (pre-DIIS).
///
/// This function builds the plain Roothaan Fock matrix. The CDIIS
/// extrapolation step lives in `crate::diis_adapter::diis_step` and is
/// invoked from `crate::kernel_impl::scf_loop` AFTER `hooks.get_fock(...)`,
/// so an `OverrideHooks` impl that overrides `get_fock` does NOT need to
/// reimplement DIIS — the cycle loop handles extrapolation around the
/// hook's return value.
///
/// The `_diis_state` parameter is reserved for future use (e.g. passing
/// the Diis<FockSubspace> handle through the hook). Plan 03-04 keeps it
/// inert; plan 03-07 (PyO3 bridge) may surface it for Python-side
/// override implementations.
///
/// Source: `pyscf/scf/hf.py:1086-1135` (the upstream `get_fock` slot —
/// upstream embeds DIIS inside; we hoist it one layer up to the cycle
/// loop so the SCF-08 hook signature stays simple).
pub fn default_get_fock(
    h1e: &Density,
    _s1e: &Density,
    vhf: &Density,
    _dm: &Density,
    _cycle: i32,
    _diis_state: Option<&Density>,
) -> Result<Density, PyscfRsError> {
    let nao = h1e.nao;
    if vhf.nao != nao {
        return Err(PyscfRsError::Core(
            pyscf_core::CoreError::DimensionMismatch {
                expected: nao,
                actual: vhf.nao,
            },
        ));
    }
    let data: Vec<f64> = h1e.data.iter().zip(&vhf.data).map(|(a, b)| a + b).collect();
    Ok(Density { nao, data })
}
