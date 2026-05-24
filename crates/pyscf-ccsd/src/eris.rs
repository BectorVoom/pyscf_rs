//! `ChemistsEris` — the Rust port of upstream `ccsd._ChemistsERIs`
//! (`pyscf/cc/ccsd.py:1389`).
//!
//! CCSD consumes a far richer ERI block set than MP2's single `ovov`: the
//! amplitude equations (`update_amps`, Wave 1) and the intermediates
//! (`rintermediates`, Wave 1) reference `oooo`, `ovoo`, `oovv`, `ovov`,
//! `ovvo`, `ovvv` and the `vvvv` block (the `nv⁴` arena tenant), plus the
//! Fock matrix and MO energies.
//!
//! Every block is a flat `Vec<f64>` in C-order (row-major, last index fastest)
//! over the active occupied/virtual ranges. The flat-index offset for each
//! block is doc-commented below (copying the discipline of
//! `pyscf-mp2/src/hooks.rs:22-30`); consumers MUST validate `block.len()`
//! against the expected product BEFORE indexing (V5 — `CcsdError::ShapeMismatch`,
//! never an OOB panic since `#![forbid(unsafe_code)]`).
//!
//! `o` = active occupied (length `nocc`), `v` = active virtual (length `nvir`).

/// Chemist's-notation ERI blocks for (R)CCSD, the port of `_ChemistsERIs`.
///
/// All `(pq|rs)` blocks are in chemist's notation and flat C-order. Naming
/// follows upstream: each letter is `o` (occupied) or `v` (virtual) and gives
/// the index range of the corresponding slot in the `(pq|rs)` integral.
#[derive(Debug, Clone)]
pub struct ChemistsEris {
    /// `(oo|oo)` block, flat `(i,j,k,l)` C-order, length `nocc^4`.
    /// Element `(i,j,k,l)` at `((i*nocc + j)*nocc + k)*nocc + l`.
    pub oooo: Vec<f64>,

    /// `(ov|oo)` block, flat `(i,a,j,k)` C-order, length `nocc*nvir*nocc*nocc`.
    /// Element `(i,a,j,k)` at `((i*nvir + a)*nocc + j)*nocc + k`.
    pub ovoo: Vec<f64>,

    /// `(oo|vv)` block, flat `(i,j,a,b)` C-order, length `nocc*nocc*nvir*nvir`.
    /// Element `(i,j,a,b)` at `((i*nocc + j)*nvir + a)*nvir + b`.
    pub oovv: Vec<f64>,

    /// `(ov|ov)` block, flat `(i,a,j,b)` C-order, length `nocc*nvir*nocc*nvir`.
    /// Element `(i,a,j,b)` at `((i*nvir + a)*nocc + j)*nvir + b`. This is the
    /// `(ia|jb)` block the MP2 seed (`init_amps`) divides by the denominators.
    pub ovov: Vec<f64>,

    /// `(ov|vo)` block, flat `(i,a,b,j)` C-order, length `nocc*nvir*nvir*nocc`.
    /// Element `(i,a,b,j)` at `((i*nvir + a)*nvir + b)*nocc + j`.
    pub ovvo: Vec<f64>,

    /// `(ov|vv)` block, flat `(i,a,b,c)` C-order, length `nocc*nvir*nvir*nvir`.
    /// Element `(i,a,b,c)` at `((i*nvir + a)*nvir + b)*nvir + c`.
    pub ovvv: Vec<f64>,

    /// `(vv|vv)` block, flat `(a,b,c,d)` C-order, length `nvir^4`.
    ///
    /// THE primary arena tenant (`Wabef ≈ nv⁴`). For this scaffold it is an
    /// in-core dense buffer; Wave-4 swaps its SOURCE — `dfccsd` (06 DF wave)
    /// reads it from the DF `vvL` B-tensor and `direct` (06 AO-direct wave)
    /// streams it on the fly — but the consumer-facing layout stays this flat
    /// `(a,b,c,d)` C-order: element `(a,b,c,d)` at
    /// `((a*nvir + b)*nvir + c)*nvir + d`.
    pub vvvv: Vec<f64>,

    /// Fock matrix in the MO basis, flat `(p,q)` C-order over ALL active MOs
    /// (`nocc + nvir`), length `(nocc+nvir)^2`. Element `(p,q)` at
    /// `p*(nocc+nvir) + q`. For a converged canonical reference this is
    /// diagonal (= `mo_energy`); CCSD keeps the full matrix because the
    /// `cc_Foo`/`cc_Fvv`/`cc_Fov` intermediates reference the off-diagonal
    /// (non-canonical / Brueckner) blocks.
    pub fock: Vec<f64>,

    /// MO energies for the active orbitals, length `nocc + nvir` (occupied
    /// first, then virtual). The `εi + εj − εa − εb` denominators of the
    /// amplitude equations read from here.
    pub mo_energy: Vec<f64>,

    /// Number of (active) occupied orbitals.
    pub nocc: usize,
    /// Number of (active) virtual orbitals.
    pub nvir: usize,
}
