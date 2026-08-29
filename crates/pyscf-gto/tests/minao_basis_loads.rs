//! MINAO is stored as a Python MODULE, not as NWChem text.
//!
//! `pyscf/gto/basis/minao.py` holds one nested-list literal per element, and
//! `basis/__init__.py:665-676` imports it with `importlib`. The ALIAS table has
//! always advertised `"minao"`, but the loader only knew NWChem and CP2K, so
//! every `load_basis("minao", …)` failed with a file-not-found on the
//! extensionless path. DFT+U's local-orbital projection (`krkspu.py:161-176`)
//! is the first consumer that needs it.

use pyscf_gto::basis::load_basis;

#[test]
fn minao_loads_for_a_range_of_elements() {
    for (symbol, min_shells) in [("H", 1usize), ("Si", 2), ("Ni", 3), ("C", 2)] {
        let b = load_basis("minao", symbol).unwrap_or_else(|e| panic!("minao/{symbol}: {e}"));
        assert!(
            b.shells.len() >= min_shells,
            "minao/{symbol} produced {} shells, expected at least {min_shells}",
            b.shells.len()
        );
        for sh in &b.shells {
            assert!(!sh.exponents.is_empty(), "minao/{symbol}: empty shell");
            assert!(
                sh.exponents.iter().all(|e| *e > 0.0),
                "minao/{symbol}: non-positive exponent"
            );
            for ctr in &sh.coeffs {
                assert_eq!(
                    ctr.len(),
                    sh.exponents.len(),
                    "minao/{symbol}: a contraction has {} coefficients for {} exponents",
                    ctr.len(),
                    sh.exponents.len()
                );
            }
        }
        println!("minao/{symbol}: {} shells", b.shells.len());
    }
}

/// The `.py` fallback must not shadow a real extensionless file: an ordinary
/// NWChem set still loads through the primary path.
#[test]
fn the_python_module_fallback_does_not_disturb_nwchem_sets() {
    let b = load_basis("sto-3g", "C").expect("sto-3g/C");
    assert!(!b.shells.is_empty());
}
