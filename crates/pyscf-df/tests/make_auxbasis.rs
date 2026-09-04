//! `pyscf_df::make_auxbasis` — the DF auxiliary-basis resolution chain
//! (plan 14-01 Task 3).
//!
//! Targets are measurements from vendored PySCF 2.12.1, recorded in
//! `.planning/phases/14-gdf-mdf-rsdf-rsjk/measurements/README.md`:
//!
//! ```text
//! PYTHONPATH=. .venv/bin/python -c "
//! from pyscf.df import addons; import sys
//! sys.path.insert(0,'.planning/phases/14-gdf-mdf-rsdf-rsjk/measurements')
//! from _cells import diamond, he_fcc
//! print(addons.make_auxbasis(diamond())); print(addons.make_auxbasis(he_fcc()))"
//! ```

use std::collections::HashMap;

use pyscf_core::{ParsedAtom, ParsedBasis};
use pyscf_df::{make_auxbasis, predefined_auxbasis};

fn charge_of(sym: &str) -> Option<usize> {
    pyscf_gto::format_atom::charge_for_symbol(sym).and_then(|z| usize::try_from(z).ok())
}

fn load(name: &str, sym: &str) -> ParsedBasis {
    pyscf_gto::basis::load_basis(name, sym).expect("basis loads")
}

fn setup(
    sym: &str,
    orbital: &str,
) -> (
    Vec<ParsedAtom>,
    HashMap<String, String>,
    HashMap<String, ParsedBasis>,
) {
    let atoms: Vec<ParsedAtom> = vec![(sym.to_string(), [0.0, 0.0, 0.0])];
    let names = HashMap::from([(sym.to_string(), orbital.to_string())]);
    let parsed = HashMap::from([(sym.to_string(), load(orbital, sym))]);
    (atoms, names, parsed)
}

/// The Phase-3 table had `sto-3g -> weigend`; upstream says `def2-svp-jkfit`,
/// and its keys are `_format_basis_name`-canonical, not dash-preserving.
#[test]
fn predefined_auxbasis_matches_upstream_psi4_table() {
    assert_eq!(
        predefined_auxbasis("sto-3g", true, false),
        Some("def2-svp-jkfit")
    );
    assert_eq!(
        predefined_auxbasis("STO-3G", true, false),
        Some("def2-svp-jkfit")
    );
    assert_eq!(
        predefined_auxbasis("sto-3g", true, true),
        Some("def2-svp-ri")
    );
    assert_eq!(
        predefined_auxbasis("cc-pvdz", true, false),
        Some("cc-pvdz-jkfit")
    );
    assert_eq!(
        predefined_auxbasis("def2-svp", true, false),
        Some("def2-svp-jkfit")
    );
    // gth-szv is in neither table — this is what sends diamond to the ETB route.
    assert_eq!(predefined_auxbasis("gth-szv", true, false), None);
}

/// He-fcc/`sto-3g` takes the NAMED route: `def2-svp-jkfit`, 9 shells, 23 AOs.
#[test]
fn helium_sto3g_resolves_to_def2_svp_jkfit() {
    let (atoms, names, parsed) = setup("He", "sto-3g");
    let aux = make_auxbasis(&atoms, &names, &parsed, charge_of, true, false).expect("auxbasis");
    let he = aux.get("He").expect("He entry");
    assert_eq!(he.shells.len(), 9, "4 s + 3 p + 2 d");
    let nao: usize = he.shells.iter().map(|s| 2 * s.l as usize + 1).sum();
    assert_eq!(nao, 23, "measured auxcell.nao for the one-atom He cell");

    // It must be the LOADED def2-universal-jkfit, not an ETB set: the ETB
    // route for a single s-shell orbital basis would give s functions only.
    let want = load("def2-svp-jkfit", "He");
    assert_eq!(he.shells.len(), want.shells.len());
    for (g, w) in he.shells.iter().zip(want.shells.iter()) {
        assert_eq!(g.l, w.l);
        assert_eq!(g.exponents, w.exponents);
    }
}

/// diamond/`gth-szv` takes the ETB route: 18 shells, 54 AOs per carbon, so
/// `auxcell.nao = 108` for the two-atom cell (measured upstream).
#[test]
fn carbon_gth_szv_resolves_to_even_tempered() {
    let (atoms, names, parsed) = setup("C", "gth-szv");
    let aux = make_auxbasis(&atoms, &names, &parsed, charge_of, true, false).expect("auxbasis");
    let c = aux.get("C").expect("C entry");
    assert_eq!(c.shells.len(), 18);
    let nao: usize = c.shells.iter().map(|s| 2 * s.l as usize + 1).sum();
    assert_eq!(nao, 54);
    assert_eq!(nao * 2, 108, "the diamond cell has two carbons");
    // Every ETB shell is a single uncontracted primitive.
    assert!(c.shells.iter().all(|s| s.exponents.len() == 1));
    assert!((c.shells[0].exponents[0] - 7.6024170048).abs() < 1e-10);
}

/// A mixed cell exercises the `placed.len() != uniq.len()` branch: upstream
/// generates ETB for EVERY element and then re-overrides the placed ones.
#[test]
fn mixed_elements_keep_the_named_route_where_it_applies() {
    let atoms: Vec<ParsedAtom> = vec![
        ("C".to_string(), [0.0, 0.0, 0.0]),
        ("He".to_string(), [0.0, 0.0, 2.0]),
    ];
    let names = HashMap::from([
        ("C".to_string(), "gth-szv".to_string()),
        ("He".to_string(), "sto-3g".to_string()),
    ]);
    let parsed = HashMap::from([
        ("C".to_string(), load("gth-szv", "C")),
        ("He".to_string(), load("sto-3g", "He")),
    ]);
    let aux = make_auxbasis(&atoms, &names, &parsed, charge_of, true, false).expect("auxbasis");
    assert_eq!(aux.len(), 2);
    assert_eq!(aux["He"].shells.len(), 9, "He keeps def2-svp-jkfit");
    assert_eq!(aux["C"].shells.len(), 18, "C falls back to ETB");
}
