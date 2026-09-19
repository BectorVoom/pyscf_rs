//! GTO-04: `make_env` — flat-array projection per D-03.
//!
//! Source: `pyscf/gto/mole.py:961-1105` (Apache-2.0).
//! Slot constants from `cintx_compat::raw` via `pyscf_core::raw_layout` per D-03.
//!
//! Two-pass algorithm (Pitfall 4 — per-symbol grouping in first-occurrence order):
//!
//! 1. Pass 1: per-atom `_atm` row + xyz/zeta to `_env`.
//! 2. Per-symbol bas template — built ONCE per unique symbol (first-occurrence
//!    order), exponents + (normalised) coefficients appended to `_env`.
//! 3. Pass 2: per-atom clone the per-symbol templates, patch `ATOM_OF`.
//!
//! Coefficient normalisation applies both:
//!   - per-primitive radial `gto_norm(l, α) = 1 / sqrt(gaussian_int(2l+2, 2α))`
//!     (`pyscf/gto/mole.py:120-155`)
//!   - per-contraction `_nomalize_contracted_ao(l, exps, coeffs)` post-multiplies
//!     each contraction column by `1 / sqrt(c.T @ S @ c)` where
//!     `S[i,j] = gaussian_int(2l+2, e_i + e_j)` (`pyscf/gto/mole.py:1018-1027`)
//!
//! Closed-form gaussian_int (RESEARCH A2):
//!   `gaussian_int(n, α) = 0.5 * Γ((n+1)/2) / α^((n+1)/2)`
//! `Γ(x)` is `libm::tgamma(x)`.
//!
//! Pitfall 5: `KAPPA_OF = 0` for all sph/cart shells in v1.
//! Pitfall 2: `_atm.len() % ATM_SLOTS == 0` and `_bas.len() % BAS_SLOTS == 0`.

use pyscf_core::raw_layout::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, KAPPA_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF,
    POINT_NUC, PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_ZETA,
};
use pyscf_core::{ParsedAtom, ParsedBasis};
use std::collections::HashMap;

/// Output of `make_env` — the four flat arrays plus precomputed sizes.
///
/// Mirrors `pyscf/gto/mole.py:1029-1105` `make_env` return + `nao_nr` + `make_loc`
/// (`pyscf/gto/moleintor.py:804-820`).
#[derive(Debug, Default, Clone)]
pub struct MakeEnvOutput {
    /// `natm * ATM_SLOTS` i32 entries.
    pub _atm: Vec<i32>,
    /// `nbas * BAS_SLOTS` i32 entries.
    pub _bas: Vec<i32>,
    /// First `PTR_ENV_START` slots are libcint-reserved zeros; user data follows.
    pub _env: Vec<f64>,
    /// `nbas + 1` entries; cumulative AO offsets per shell.
    pub ao_loc_nr: Vec<i32>,
    /// Total AO count across all shells (sum of `(2l+1)*nctr` for sph,
    /// `(l+1)(l+2)/2 * nctr` for cart).
    pub nao_nr: usize,
}

/// Project `(atoms, basis_per_symbol)` to libcint flat arrays.
///
/// `cart=true` produces Cartesian-basis dimensions; `cart=false` (default)
/// produces spherical-basis dimensions per upstream PySCF semantics.
pub fn make_env(
    atoms: &[ParsedAtom],
    basis: &HashMap<String, ParsedBasis>,
    cart: bool,
) -> MakeEnvOutput {
    // Reserve the libcint global-param slots (PTR_ENV_START = 20 zeros).
    let mut out = MakeEnvOutput {
        _env: vec![0.0_f64; PTR_ENV_START],
        ..Default::default()
    };

    // ===== Pass 1: per-atom _atm rows + xyz to _env =====
    for (sym_with_suffix, xyz) in atoms {
        let alpha: String = sym_with_suffix
            .chars()
            .take_while(|c| c.is_alphabetic())
            .collect();
        let charge = crate::format_atom::charge_for_symbol(&alpha).unwrap_or(0);
        let nuclear_model = POINT_NUC; // v1: always point nucleus
        let zeta = 0.0_f64; // unused for point nuclei

        let ptr_coord = out._env.len();
        // Append xyz (3 doubles).
        out._env.extend_from_slice(xyz);
        // Append zeta placeholder (1 double; finite-nucleus model would write here).
        out._env.push(zeta);

        // Append _atm row (6 i32).
        let row_start = out._atm.len();
        out._atm.resize(row_start + ATM_SLOTS, 0);
        out._atm[row_start + CHARGE_OF] = charge;
        out._atm[row_start + PTR_COORD] = ptr_coord as i32;
        out._atm[row_start + NUC_MOD_OF] = nuclear_model;
        out._atm[row_start + PTR_ZETA] = (ptr_coord + 3) as i32;
        // Slots PTR_FRAC_CHARGE (4) and 5 stay zero per upstream.
    }

    // ===== Per-symbol bas template (Pitfall 4: first-occurrence order) =====
    // basdic: symbol → Vec<bas_row_template (8 i32 each)>
    let mut basdic: HashMap<String, Vec<[i32; BAS_SLOTS]>> = HashMap::new();
    for (sym_with_suffix, _) in atoms {
        let alpha: String = sym_with_suffix
            .chars()
            .take_while(|c| c.is_alphabetic())
            .collect();
        let upper = alpha.to_ascii_uppercase();
        if basdic.contains_key(&upper) {
            continue;
        }
        // Build bas template + append normalised exponents/coeffs to _env for THIS symbol.
        let parsed = basis
            .get(&upper)
            .expect("format_basis ensures every atom symbol is in basis map");
        let templates = make_bas_env_for_symbol(parsed, &mut out._env);
        basdic.insert(upper, templates);
    }

    // ===== Pass 2: per-atom clone the per-symbol template, patch ATOM_OF =====
    for (atom_id, (sym_with_suffix, _)) in atoms.iter().enumerate() {
        let alpha: String = sym_with_suffix
            .chars()
            .take_while(|c| c.is_alphabetic())
            .collect();
        let upper = alpha.to_ascii_uppercase();
        let templates = &basdic[&upper];
        for tpl in templates {
            let mut row = *tpl;
            row[ATOM_OF] = atom_id as i32;
            out._bas.extend_from_slice(&row);
        }
    }

    // Pitfall 2 sanity (Phase 2 success criterion 2 prerequisite).
    debug_assert_eq!(out._atm.len() % ATM_SLOTS, 0);
    debug_assert_eq!(out._bas.len() % BAS_SLOTS, 0);

    // Compute nao_nr + ao_loc_nr from _bas.
    let nbas = out._bas.len() / BAS_SLOTS;
    let mut ao_loc = Vec::with_capacity(nbas + 1);
    let mut acc: i32 = 0;
    ao_loc.push(0);
    for i in 0..nbas {
        let l = out._bas[i * BAS_SLOTS + ANG_OF];
        let nctr = out._bas[i * BAS_SLOTS + NCTR_OF];
        let dim_per_ctr = if cart {
            ((l + 1) * (l + 2)) / 2
        } else {
            2 * l + 1
        };
        acc += dim_per_ctr * nctr;
        ao_loc.push(acc);
    }
    out.nao_nr = acc as usize;
    out.ao_loc_nr = ao_loc;

    out
}

/// Per-symbol: build bas templates + append normalised exponents/coeffs to `_env`.
/// Returns `Vec<bas_row_template>` with `PTR_EXP` / `PTR_COEFF` pointing into `_env`;
/// `ATOM_OF` stays 0 (caller patches per atom).
fn make_bas_env_for_symbol(parsed: &ParsedBasis, _env: &mut Vec<f64>) -> Vec<[i32; BAS_SLOTS]> {
    let mut templates = Vec::with_capacity(parsed.shells.len());
    for shell in &parsed.shells {
        let l = shell.l;
        let nprim = shell.exponents.len();
        let nctr = shell.coeffs.len(); // each row in coeffs is one contraction column

        // Apply normalisation transforms: gto_norm + _nomalize_contracted_ao.
        // `final_coeffs[c_idx][p_idx]` — per-contraction-column ordering matches
        // libcint F-order (coefficient matrix flattened column-major).
        let final_coeffs = normalise_contractions(l, &shell.exponents, &shell.coeffs);

        let ptr_exp = _env.len();
        _env.extend_from_slice(&shell.exponents);
        let ptr_coeff = _env.len();
        // Append in F-order: for each contraction column c, append nprim primitives.
        for col in &final_coeffs {
            _env.extend_from_slice(col);
        }

        let mut row = [0_i32; BAS_SLOTS];
        row[ATOM_OF] = 0; // patched per atom in pass 2
        row[ANG_OF] = l as i32;
        row[NPRIM_OF] = nprim as i32;
        row[NCTR_OF] = nctr as i32;
        row[KAPPA_OF] = 0; // Pitfall 5: always 0 for sph/cart in v1
        row[PTR_EXP] = ptr_exp as i32;
        row[PTR_COEFF] = ptr_coeff as i32;
        // slot 7 stays 0 per upstream

        templates.push(row);
    }
    templates
}

/// `scipy.special.gamma(l + 1.5)` (cephes) for `l = 0..=15` — the scalar gamma
/// `gaussian_int` divides by. cephes is one ulp off the correctly-rounded value
/// (`gamma(1.5)` = `0x1.c5bf891b4ef6ap-1`, one ulp below `math.gamma(1.5)`), so
/// it is tabulated as raw bits instead of recomputed. Probed from the vendored
/// scipy (2026-09-18, Task 6).
const CEPHES_GAMMA_HALF: [f64; 16] = [
    f64::from_bits(0x3fec5bf891b4ef6a), // l=0  gamma(1.5)
    f64::from_bits(0x3ff544fa6d47b390), // l=1  gamma(2.5)
    f64::from_bits(0x400a96390899a075), // l=2  gamma(3.5)
    f64::from_bits(0x40274371e7866c66), // l=3  gamma(4.5)
    f64::from_bits(0x404a2be0247739f2), // l=4  gamma(5.5)
    f64::from_bits(0x4071fe2a1911f7d6), // l=5  gamma(6.5)
    f64::from_bits(0x409d3d0468bd32bd), // l=6  gamma(7.5)
    f64::from_bits(0x40cb693422315f91), // l=7  gamma(8.5)
    f64::from_bits(0x40fd1fc76454758a), // l=8  gamma(9.5)
    f64::from_bits(0x41314ade639225ca), // l=9  gamma(10.5)
    f64::from_bits(0x4166b243e2afd19a), // l=10 gamma(11.5)
    f64::from_bits(0x41a05020caee5ea6), // l=11 gamma(12.5)
    f64::from_bits(0x41d97d333d1473e4), // l=12 gamma(13.5)
    f64::from_bits(0x421581a33b8941c8), // l=13 gamma(14.5)
    f64::from_bits(0x42537d7bedf4639d), // l=14 gamma(15.5)
    f64::from_bits(0x4292e1900e84c081), // l=15 gamma(16.5)
];

/// `gaussian_int(n, α) = scipy.special.gamma((n+1)/2) / (2 α^((n+1)/2))`
/// (`pyscf/gto/mole.py:120`), specialised to `n = 2l+2` so the gamma is
/// [`CEPHES_GAMMA_HALF`][l] and `α^(l+1.5)` is numpy's SVML `pow`
/// ([`crate::svml_pow::svml_pow8`]) — not glibc `pow`. `α` is the value raised
/// to `l+1.5` (`2·expnt` for `gto_norm`, `expnt_i + expnt_j` for the S-matrix).
pub(crate) fn gaussian_int(l: u8, alpha: f64) -> f64 {
    let n1 = f64::from(l) + 1.5;
    CEPHES_GAMMA_HALF[l as usize] / (2.0 * crate::svml_pow::svml_pow8(alpha, n1))
}

/// Per-primitive radial normalisation:
/// `gto_norm(l, α) = 1 / sqrt(gaussian_int(2l+2, 2α))`
/// (`pyscf/gto/mole.py:125-155`).
pub(crate) fn gto_norm(l: u8, alpha: f64) -> f64 {
    1.0 / gaussian_int(l, 2.0 * alpha).sqrt()
}

/// Per-primitive radial `gto_norm` + per-contraction `_nomalize_contracted_ao`.
///
/// Source: `pyscf/gto/mole.py:120-155` (`gto_norm`) + `1018-1027`
/// (`_nomalize_contracted_ao`), reproduced in numpy's exact rounding:
///
///   * `cs = einsum('pi,p->pi', cs, gto_norm(es))` — elementwise multiply;
///   * `ee = es_i + es_j`, `ee = gaussian_int(2l+2, ee)`;
///   * `s1 = 1/sqrt(einsum('pi,pq,qi->i', cs, ee, cs))` — the einsum sums
///     p-outer then q-inner, product `(c[p]·ee[p,q])·c[q]`, plain left-to-right
///     accumulation (no FMA);
///   * `cs = einsum('pi,i->pi', cs, s1)` — elementwise multiply.
///
/// Returns coefficients indexed `[contraction_column][primitive]` (F-order on
/// flatten — column-major, matching libcint convention).
pub(crate) fn normalise_contractions(
    l: u8,
    exponents: &[f64],
    raw_coeffs: &[Vec<f64>],
) -> Vec<Vec<f64>> {
    let nprim = exponents.len();
    let nctr = raw_coeffs.len();

    // `gto_norm` on the exponent array, then the elementwise scale.
    let prim_norm: Vec<f64> = exponents.iter().map(|&a| gto_norm(l, a)).collect();
    let mut scaled: Vec<Vec<f64>> = (0..nctr)
        .map(|c| {
            (0..nprim)
                .map(|p| raw_coeffs[c][p] * prim_norm[p])
                .collect()
        })
        .collect();

    // `_nomalize_contracted_ao`: einsum('pi,pq,qi->i') in numpy's order.
    for col in scaled.iter_mut() {
        let mut norm_sq = 0.0_f64;
        for p in 0..nprim {
            for q in 0..nprim {
                let s = gaussian_int(l, exponents[p] + exponents[q]);
                norm_sq += (col[p] * s) * col[q];
            }
        }
        let inv = 1.0 / norm_sq.sqrt();
        for v in col.iter_mut() {
            *v *= inv;
        }
    }
    scaled
}

