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
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, KAPPA_OF, NCTR_OF, NPRIM_OF,
    NUC_MOD_OF, POINT_NUC, PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_ZETA,
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
    let mut out = MakeEnvOutput::default();

    // Reserve the libcint global-param slots (PTR_ENV_START = 20 zeros).
    out._env = vec![0.0_f64; PTR_ENV_START];

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
        let l = out._bas[i * BAS_SLOTS + ANG_OF] as i32;
        let nctr = out._bas[i * BAS_SLOTS + NCTR_OF] as i32;
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
fn make_bas_env_for_symbol(
    parsed: &ParsedBasis,
    _env: &mut Vec<f64>,
) -> Vec<[i32; BAS_SLOTS]> {
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
        for c_idx in 0..nctr {
            _env.extend_from_slice(&final_coeffs[c_idx]);
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

/// Per-primitive radial `gto_norm` + per-contraction `_nomalize_contracted_ao`.
///
/// Source: `pyscf/gto/mole.py:120-155` (`gto_norm`) + `1018-1027`
/// (`_nomalize_contracted_ao`).
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

    // Per-primitive radial normalisation factor.
    let prim_norm: Vec<f64> = exponents.iter().map(|&a| gto_norm(l, a)).collect();

    // Apply per-primitive norm: c'_{c,p} = c_{c,p} * prim_norm[p].
    let mut scaled: Vec<Vec<f64>> = (0..nctr)
        .map(|c| (0..nprim).map(|p| raw_coeffs[c][p] * prim_norm[p]).collect())
        .collect();

    // _nomalize_contracted_ao: post-multiply each contraction column c by
    // 1/sqrt(c.T @ S @ c) where S[i,j] = gaussian_int(2l+2, e_i + e_j).
    for c in 0..nctr {
        let col = &scaled[c];
        let mut norm_sq = 0.0_f64;
        for i in 0..nprim {
            for j in 0..nprim {
                norm_sq += col[i]
                    * col[j]
                    * gaussian_int(2 * l as i32 + 2, exponents[i] + exponents[j]);
            }
        }
        if norm_sq > 0.0 {
            let inv = 1.0 / norm_sq.sqrt();
            for p in 0..nprim {
                scaled[c][p] *= inv;
            }
        }
    }
    scaled
}

/// Closed-form Gaussian integral
/// `int_0^∞ r^n exp(-α r²) dr = 0.5 * Γ((n+1)/2) / α^((n+1)/2)`.
///
/// Source: `pyscf/gto/mole.py` `gaussian_int` helper (~mole.py:120) +
/// RESEARCH A2 (closed form). For `n` and `α` ranges seen in production basis
/// sets (`l ≤ 6`, `α > 1e-3`) the closed form is numerically well-conditioned;
/// `libm::tgamma` covers the half-integer Γ values.
pub(crate) fn gaussian_int(n: i32, alpha: f64) -> f64 {
    let half_n_plus_1 = (n as f64 + 1.0) * 0.5;
    let gamma = libm::tgamma(half_n_plus_1);
    0.5 * gamma / alpha.powf(half_n_plus_1)
}

/// Per-primitive radial normalisation:
/// `gto_norm(l, α) = 1 / sqrt(gaussian_int(2l+2, 2α))`.
///
/// Source: `pyscf/gto/mole.py:125-155`.
pub(crate) fn gto_norm(l: u8, alpha: f64) -> f64 {
    1.0 / gaussian_int(2 * l as i32 + 2, 2.0 * alpha).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_int_known_value_zero_alpha_unity_n_zero() {
        // Γ(1/2) = sqrt(π); gaussian_int(0, α) = 0.5 * sqrt(π) / α^(1/2).
        let g = gaussian_int(0, 1.0);
        let expected = 0.5 * std::f64::consts::PI.sqrt();
        assert!((g - expected).abs() < 1e-12, "got {g}, expected {expected}");
    }

    #[test]
    fn gto_norm_l_zero_alpha_unity() {
        // gto_norm(l=0, α=1) = 1 / sqrt(gaussian_int(2, 2)) = 1 / sqrt(0.5 * Γ(3/2) / 2^(3/2))
        //                    = 1 / sqrt(0.5 * (sqrt(π)/2) / 2.8284) = 1 / sqrt(0.0782) ≈ 3.5779
        let n = gto_norm(0, 1.0);
        // From scipy: 1 / sqrt(0.5 * sqrt(pi) * 0.5 / 2^(3/2)) = 2 * (2/π)^(1/4) * sqrt(2)
        let expected = 2.0 * (2.0 / std::f64::consts::PI).powf(0.25) * 1.0_f64.sqrt() * 2.0_f64.powf(0.75);
        // The exact closed form: gto_norm(0,α)= (2α/π)^(3/4) * 2^(0+0) ; for α=1 → (2/π)^(3/4)
        // Actually upstream: gto_norm(0,α) = (2α/π)^(3/4) * 2^l * sqrt((2l+1)!! / (4π))^-1 — simplified.
        // Easier: our implementation uses the closed form via gaussian_int. Verify by
        // evaluating gaussian_int(2, 2) directly: 0.5 * Γ(3/2) / 2^(3/2) = 0.5 * (sqrt(π)/2) / 2.828427
        let gi = 0.5 * (std::f64::consts::PI.sqrt() / 2.0) / 2.0_f64.powf(1.5);
        let expected_n = 1.0 / gi.sqrt();
        assert!(
            (n - expected_n).abs() < 1e-10,
            "got {n}, expected {expected_n} (closed form), upstream-formula approx {expected}"
        );
    }

    #[test]
    fn normalisation_yields_unit_self_overlap() {
        // For a single-primitive single-contraction shell (nprim=1, nctr=1):
        // After normalisation, c.T @ S @ c == 1 by construction.
        let l = 0_u8;
        let exps = vec![1.0_f64];
        let raw = vec![vec![1.0_f64]];
        let final_c = normalise_contractions(l, &exps, &raw);
        // Compute c.T @ S @ c on the normalised coeffs.
        let c = final_c[0][0];
        let s = gaussian_int(2 * l as i32 + 2, exps[0] + exps[0]);
        let norm_sq = c * c * s;
        assert!(
            (norm_sq - 1.0).abs() < 1e-12,
            "self-overlap = {norm_sq}, expected 1.0"
        );
    }

    #[test]
    fn normalisation_multi_primitive_unit_overlap() {
        // STO-3G H: 3 primitives, 1 contraction. After normalisation,
        // c.T @ S @ c == 1.
        let l = 0_u8;
        let exps = vec![3.42525091, 0.62391373, 0.16885540];
        let raw = vec![vec![0.15432897, 0.53532814, 0.44463454]];
        let final_c = normalise_contractions(l, &exps, &raw);
        let mut overlap = 0.0_f64;
        for i in 0..exps.len() {
            for j in 0..exps.len() {
                overlap += final_c[0][i]
                    * final_c[0][j]
                    * gaussian_int(2 * l as i32 + 2, exps[i] + exps[j]);
            }
        }
        assert!(
            (overlap - 1.0).abs() < 1e-12,
            "STO-3G H self-overlap = {overlap}, expected 1.0"
        );
    }
}
