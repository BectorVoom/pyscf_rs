//! Test fixtures — string keys identifying canonical pyscf-rs ↔ upstream pairs.
//!
//! Each fixture is a (atom-string, basis-name, spin) triple consumed by both
//! upstream `pyscf.M(...)` (via Python::attach) and pyscf-rs `pyscf_gto::M(...)`.
//! Keeping the geometry/basis lookup behind a `&str` key means every oracle
//! test reads identically — `oracle_check!("scf_rhf_energy", H2O_CC_PVDZ, 1e-6)`.
//!
//! Per plan 03-08 must_haves: H2O_CC_PVDZ + BENZENE_6_31GS +
//! WATER_TRIMER_CC_PVDZ + H2O_TRIPLET_CCPVDZ are required.

/// Closed-shell water in the cc-pVDZ basis — the workhorse fixture.
pub const H2O_CC_PVDZ: &str = "h2o_ccpvdz";

/// Benzene in 6-31G* — medium-size aromatic for DF-HF coverage.
pub const BENZENE_6_31GS: &str = "benzene_631gs";

/// Water trimer in cc-pVDZ — multi-atom Mulliken/dipole coverage.
pub const WATER_TRIMER_CC_PVDZ: &str = "water_trimer_ccpvdz";

/// Open-shell H2O — spin=2 triplet state for UHF coverage (Arm 2).
pub const H2O_TRIPLET_CCPVDZ: &str = "h2o_triplet_ccpvdz";

/// Atom string for a fixture name. Same string is fed to upstream
/// `pyscf.M(atom=...)` AND `pyscf_gto::M(MoleBuildArgs { atom:
/// AtomInput::String(...) })`.
///
/// The `<fixture>_<mode>` variant (used by init_guess arm) strips the
/// trailing `_<mode>` token so the base fixture's atom resolves.
pub fn atom(name: &str) -> &'static str {
    // Strip the grid-level suffix `@levelN` (plan 04-04 grid_weights arm) so
    // the base fixture's geometry resolves.
    let name = name.split('@').next().unwrap_or(name);
    match name {
        "h2o_ccpvdz" | "h2o_triplet_ccpvdz" => H2O_GEOMETRY,
        "benzene_631gs" => BENZENE_GEOMETRY,
        "water_trimer_ccpvdz" => WATER_TRIMER_GEOMETRY,
        // init_guess token forms: "<fixture>_<mode>" — accept any
        // h2o_ccpvdz_* suffix as the H2O base geometry.
        s if s.starts_with("h2o_ccpvdz_") => H2O_GEOMETRY,
        other => panic!("pyscf-oracle: unknown fixture '{}'", other),
    }
}

/// Basis-set name for a fixture.
pub fn basis(name: &str) -> &'static str {
    let name = name.split('@').next().unwrap_or(name);
    match name {
        "h2o_ccpvdz" | "h2o_triplet_ccpvdz" => "cc-pvdz",
        "benzene_631gs" => "6-31g*",
        "water_trimer_ccpvdz" => "cc-pvdz",
        s if s.starts_with("h2o_ccpvdz_") => "cc-pvdz",
        other => panic!("pyscf-oracle: unknown fixture '{}'", other),
    }
}

/// Spin multiplicity hint (2S = nα − nβ). 0 for closed shell; 2 for triplet.
/// Strips the `@<suffix>` (grid-level or XC functional) before matching.
pub fn spin(name: &str) -> i32 {
    let name = name.split('@').next().unwrap_or(name);
    match name {
        "h2o_triplet_ccpvdz" => 2,
        _ => 0,
    }
}

/// XC functional token extracted from the fixture-name suffix
/// (e.g. `"h2o_ccpvdz@b3lyp"` → `"b3lyp"`). Used by the DFT-01
/// `rks_energy`/`uks_energy` oracle arms (plan 04-06). Defaults to `"svwn"`
/// when no `@<xc>` suffix is present.
pub fn xc(name: &str) -> &str {
    name.split('@').nth(1).unwrap_or("svwn")
}

/// init_guess mode token extracted from fixture-name suffix
/// (e.g. `"h2o_ccpvdz_minao"` → `"minao"`). Default is `"minao"` when no
/// suffix is present.
pub fn init_guess_mode(name: &str) -> &str {
    name.strip_prefix("h2o_ccpvdz_").unwrap_or("minao")
}

const H2O_GEOMETRY: &str = "O 0.0 0.0 0.0; H 0.757 0.587 0.0; H -0.757 0.587 0.0";

const BENZENE_GEOMETRY: &str = concat!(
    "C  0.0000  1.3970  0.0000; ",
    "C  1.2099  0.6985  0.0000; ",
    "C  1.2099 -0.6985  0.0000; ",
    "C  0.0000 -1.3970  0.0000; ",
    "C -1.2099 -0.6985  0.0000; ",
    "C -1.2099  0.6985  0.0000; ",
    "H  0.0000  2.4810  0.0000; ",
    "H  2.1486  1.2405  0.0000; ",
    "H  2.1486 -1.2405  0.0000; ",
    "H  0.0000 -2.4810  0.0000; ",
    "H -2.1486 -1.2405  0.0000; ",
    "H -2.1486  1.2405  0.0000",
);

const WATER_TRIMER_GEOMETRY: &str = concat!(
    "O  -1.4220  -0.7060  0.0000; H  -1.4220  -0.1390 -0.8060; H  -0.5340  -1.0370  0.0000; ",
    "O   1.4220  -0.7060  0.0000; H   0.5340  -1.0370  0.0000; H   2.0220  -0.1390 -0.8060; ",
    "O   0.0000   1.4120  0.0000; H  -0.6000   1.7430  0.8060; H   0.6000   1.7430  0.8060",
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_constants_are_unique_strings() {
        assert_ne!(H2O_CC_PVDZ, BENZENE_6_31GS);
        assert_ne!(H2O_CC_PVDZ, WATER_TRIMER_CC_PVDZ);
        assert_ne!(H2O_CC_PVDZ, H2O_TRIPLET_CCPVDZ);
    }

    #[test]
    fn atom_lookup_returns_geometry_for_known_fixtures() {
        assert!(atom(H2O_CC_PVDZ).starts_with("O 0.0"));
        assert!(atom(BENZENE_6_31GS).starts_with("C  0.0000  1.3970"));
        assert!(atom(WATER_TRIMER_CC_PVDZ).contains("H "));
        assert!(atom(H2O_TRIPLET_CCPVDZ).starts_with("O 0.0"));
    }

    #[test]
    fn basis_lookup_matches_expected() {
        assert_eq!(basis(H2O_CC_PVDZ), "cc-pvdz");
        assert_eq!(basis(BENZENE_6_31GS), "6-31g*");
        assert_eq!(basis(H2O_TRIPLET_CCPVDZ), "cc-pvdz");
    }

    #[test]
    fn spin_triplet_h2o() {
        assert_eq!(spin(H2O_CC_PVDZ), 0);
        assert_eq!(spin(H2O_TRIPLET_CCPVDZ), 2);
    }

    #[test]
    fn init_guess_mode_extracts_suffix() {
        assert_eq!(init_guess_mode("h2o_ccpvdz_minao"), "minao");
        assert_eq!(init_guess_mode("h2o_ccpvdz_1e"), "1e");
        assert_eq!(init_guess_mode("h2o_ccpvdz_huckel"), "huckel");
        // Default when no suffix.
        assert_eq!(init_guess_mode("h2o_ccpvdz"), "minao");
    }

    #[test]
    fn atom_resolves_init_guess_suffixed_fixture_to_base_geometry() {
        // <fixture>_<mode> form must still resolve geometry — the
        // suffixed key is what oracle_check! arm 4 passes in.
        assert_eq!(atom("h2o_ccpvdz_minao"), atom(H2O_CC_PVDZ));
        assert_eq!(atom("h2o_ccpvdz_1e"), atom(H2O_CC_PVDZ));
    }

    #[test]
    #[should_panic(expected = "unknown fixture")]
    fn atom_panics_on_unknown_fixture() {
        let _ = atom("does_not_exist");
    }
}
