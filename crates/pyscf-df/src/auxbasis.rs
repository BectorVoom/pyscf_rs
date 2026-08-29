//! `DEFAULT_AUXBASIS` resolution table — port of `pyscf/df/addons.py`.
//!
//! Source-of-truth: `pyscf/df/addons.py:DEFAULT_AUXBASIS`. The full upstream
//! table covers ~50 orbital-basis entries; Phase 3 plan 03-05 ships the
//! subset that the Phase 3 test corpus (H2O/cc-pVDZ, benzene/6-31G*,
//! water-trimer/sto-3g) actually exercises. Universal fallback for unknown
//! keys: `'weigend'` (the Weigend / Häser / Ahlrichs 2002 universal aux).
//!
//! Future expansion (~200 additional entries) is a mechanical hand-port
//! if a user request surfaces — the table layout is stable.

use std::collections::HashMap;
use std::sync::OnceLock;

static TABLE: OnceLock<HashMap<&'static str, (&'static str, &'static str)>> = OnceLock::new();

/// Get the static `DEFAULT_AUXBASIS` lookup table. Maps a canonical
/// orbital-basis name to a `(jkfit_name, ri_name)` pair. Keys are lower-cased,
/// dash-preserved (matches upstream's exact `'cc-pvdz'` / `'def2-svp'` style).
#[allow(non_snake_case)]
pub fn DEFAULT_AUXBASIS() -> &'static HashMap<&'static str, (&'static str, &'static str)> {
    TABLE.get_or_init(|| {
        let mut m: HashMap<&'static str, (&'static str, &'static str)> = HashMap::new();
        // === Karlsruhe def2 family ===
        m.insert("def2-svp", ("def2-svp-jkfit", "def2-svp-ri"));
        m.insert("def2-svpd", ("def2-svp-jkfit", "def2-svp-ri"));
        m.insert("def2-tzvp", ("def2-tzvp-jkfit", "def2-tzvp-ri"));
        m.insert("def2-tzvpp", ("def2-tzvpp-jkfit", "def2-tzvpp-ri"));
        m.insert("def2-tzvpd", ("def2-tzvp-jkfit", "def2-tzvp-ri"));
        m.insert("def2-qzvp", ("def2-qzvp-jkfit", "def2-qzvp-ri"));
        m.insert("def2-qzvpp", ("def2-qzvpp-jkfit", "def2-qzvpp-ri"));
        // === Dunning correlation-consistent (cc-pVxZ) ===
        m.insert("cc-pvdz", ("cc-pvdz-jkfit", "cc-pvdz-ri"));
        m.insert("cc-pvtz", ("cc-pvtz-jkfit", "cc-pvtz-ri"));
        m.insert("cc-pvqz", ("cc-pvqz-jkfit", "cc-pvqz-ri"));
        m.insert("cc-pv5z", ("cc-pv5z-jkfit", "cc-pv5z-ri"));
        m.insert("aug-cc-pvdz", ("aug-cc-pvdz-jkfit", "aug-cc-pvdz-ri"));
        m.insert("aug-cc-pvtz", ("aug-cc-pvtz-jkfit", "aug-cc-pvtz-ri"));
        m.insert("aug-cc-pvqz", ("aug-cc-pvqz-jkfit", "aug-cc-pvqz-ri"));
        m.insert("aug-cc-pv5z", ("aug-cc-pv5z-jkfit", "aug-cc-pv5z-ri"));
        // === Pople (defaults to weigend universal) ===
        m.insert("6-31g", ("weigend", "weigend"));
        m.insert("6-31g*", ("weigend", "weigend"));
        m.insert("6-31g**", ("weigend", "weigend"));
        m.insert("6-31+g*", ("weigend", "weigend"));
        m.insert("6-311g", ("weigend", "weigend"));
        m.insert("6-311g*", ("weigend", "weigend"));
        m.insert("6-311g**", ("weigend", "weigend"));
        m.insert("6-311+g**", ("weigend", "weigend"));
        m.insert("3-21g", ("weigend", "weigend"));
        // === Minimal ===
        // Upstream `DEFAULT_AUXBASIS['sto3g'] = ('def2-svp-jkfit', 'def2-svp-ri')`.
        // The Phase-3 table said `weigend` here, which is the UNIVERSAL fallback,
        // not this row. Found by plan 14-01 when `make_auxbasis` on the He-fcc
        // reference cell went through it. The upstream-faithful lookup — with
        // canonical keys, the BSE tier and the ETB fallback — is
        // `crate::make_auxbasis::predefined_auxbasis`; prefer it over the two
        // `default_*` helpers below, which exist only for the Phase-3 API.
        m.insert("sto-3g", ("def2-svp-jkfit", "def2-svp-ri"));
        m.insert("sto-6g", ("weigend", "weigend"));
        // === ANO ===
        m.insert("ano", ("weigend", "weigend"));
        m
    })
}

/// Resolve auxbasis name for JK-fit (Coulomb + exchange fitting).
///
/// Returns the entry's first tuple element if the basis is in the table;
/// universal fallback to `'weigend'` (Weigend / Häser / Ahlrichs 2002 aux,
/// covers any orbital basis at acceptable accuracy) otherwise.
pub fn default_jkfit(basis: &str) -> &'static str {
    DEFAULT_AUXBASIS()
        .get(basis)
        .map(|(jkfit, _ri)| *jkfit)
        .unwrap_or("weigend")
}

/// Resolve auxbasis name for RI (resolution-of-identity, used by MP2/CCSD-RI).
///
/// Returns the entry's second tuple element if the basis is in the table;
/// universal fallback to `'weigend'` otherwise.
pub fn default_ri(basis: &str) -> &'static str {
    DEFAULT_AUXBASIS()
        .get(basis)
        .map(|(_jkfit, ri)| *ri)
        .unwrap_or("weigend")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cc_pvdz_in_table() {
        assert_eq!(default_jkfit("cc-pvdz"), "cc-pvdz-jkfit");
        assert_eq!(default_ri("cc-pvdz"), "cc-pvdz-ri");
    }

    #[test]
    fn unknown_falls_back() {
        assert_eq!(default_jkfit("not-a-real-basis"), "weigend");
    }

    #[test]
    fn table_has_phase3_minimum() {
        // Phase 3 corpus minimum: must cover cc-pvdz, def2-svp, 6-31g*, sto-3g.
        let t = DEFAULT_AUXBASIS();
        assert!(t.contains_key("cc-pvdz"));
        assert!(t.contains_key("def2-svp"));
        assert!(t.contains_key("6-31g*"));
        assert!(t.contains_key("sto-3g"));
    }
}
