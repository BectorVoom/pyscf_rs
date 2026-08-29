//! `pyscf_df::etb` — the even-tempered auxiliary basis (plan 14-01 Task 3).
//!
//! Every target here is a MEASUREMENT from vendored PySCF 2.12.1, recorded in
//! `.planning/phases/14-gdf-mdf-rsdf-rsjk/measurements/README.md`. Re-derive
//! them with:
//!
//! ```text
//! PYTHONPATH=. .venv/bin/python -c "
//! from pyscf import gto; from pyscf.df import addons
//! print(addons._aug_etb_element(6, gto.basis.load('gth-szv','C'), 2.0))"
//! ```

use pyscf_core::{ParsedBasis, ShellSpec};
use pyscf_df::etb::{ETB_BETA, aug_etb_element, expand_etbs};

/// `gto.basis.load('gth-szv', 'C')` — two contracted shells, four primitives each.
fn gth_szv_carbon() -> ParsedBasis {
    let exps = vec![4.3362376436, 1.2881838513, 0.4037767149, 0.1187877657];
    ParsedBasis {
        shells: vec![
            ShellSpec {
                l: 0,
                exponents: exps.clone(),
                coeffs: vec![vec![0.1490797872, -0.0292640031, -0.688204051, -0.3964426906]],
            },
            ShellSpec {
                l: 1,
                exponents: exps,
                coeffs: vec![vec![-0.0878123619, -0.27755603, -0.4712295093, -0.4058039291]],
            },
        ],
    }
}

/// `sto-3g` He — one contracted s shell.
fn sto3g_helium() -> ParsedBasis {
    ParsedBasis {
        shells: vec![ShellSpec {
            l: 0,
            exponents: vec![6.36242139, 1.158923, 0.31364979],
            coeffs: vec![vec![0.15432897, 0.53532814, 0.44463454]],
        }],
    }
}

#[test]
fn carbon_gth_szv_etb_matches_upstream() {
    let etb = aug_etb_element(6, &gth_szv_carbon(), ETB_BETA).expect("etb");
    // Upstream: [(0, 6, 0.2375755314, 2.0), (1, 6, 0.2375755314, 2.0),
    //            (2, 6, 0.2375755314, 2.0)]
    assert_eq!(etb.len(), 3, "l = 0, 1, 2");
    for (i, b) in etb.iter().enumerate() {
        assert_eq!(b.l as usize, i);
        assert_eq!(b.n, 6, "six primitives per l");
        assert!(
            (b.emin - 0.2375755314).abs() < 1e-12,
            "emin[l={i}] = {} != 0.2375755314",
            b.emin
        );
        assert_eq!(b.beta, 2.0);
    }
}

#[test]
fn carbon_etb_expands_to_18_shells_descending() {
    let etb = aug_etb_element(6, &gth_szv_carbon(), ETB_BETA).expect("etb");
    let basis = expand_etbs(&etb);
    // 6 exponents x 3 angular momenta, one uncontracted shell each.
    assert_eq!(basis.shells.len(), 18);
    // 6*1 + 6*3 + 6*5 = 54 spherical AOs per carbon; the two-atom diamond cell
    // therefore has auxcell.nao = 108 (measured upstream).
    let nao: usize = basis
        .shells
        .iter()
        .map(|s| 2 * s.l as usize + 1)
        .sum();
    assert_eq!(nao, 54);

    // `gto.expand_etbs` emits DESCENDING exponents; the AO ordering of every
    // downstream index depends on it.
    let s_exps: Vec<f64> = basis
        .shells
        .iter()
        .filter(|s| s.l == 0)
        .map(|s| s.exponents[0])
        .collect();
    let want = [
        7.6024170048,
        3.8012085024,
        1.9006042512,
        0.9503021256,
        0.4751510628,
        0.2375755314,
    ];
    assert_eq!(s_exps.len(), want.len());
    for (got, w) in s_exps.iter().zip(want.iter()) {
        assert!((got - w).abs() < 1e-10, "{got} != {w}");
    }
}

/// He/`sto-3g` never reaches the ETB route in practice — `make_auxbasis` places
/// it via the Psi4 table (`def2-svp-jkfit`, see `tests/make_auxbasis.rs`). What
/// the ETB generator itself does with a one-shell s-only orbital basis is
/// nonetheless worth pinning, because it is the `l_max_aux = 0` corner and a
/// port that silently produced p or d functions here would be wrong.
#[test]
fn helium_sto3g_etb_is_s_only() {
    let etb = aug_etb_element(2, &sto3g_helium(), ETB_BETA).expect("etb");
    // l_max_orb = 0 -> l_max_aux = 0, so s functions only.
    assert_eq!(etb.len(), 1);
    assert_eq!(etb[0].l, 0);
    let basis = expand_etbs(&etb);
    assert!(basis.shells.iter().all(|s| s.l == 0));
    assert_eq!(basis.shells.len(), etb[0].n);
    // emin = 2 * min(exponents) = 2 * 0.31364979.
    assert!(
        (etb[0].emin - 2.0 * 0.31364979).abs() < 1e-12,
        "emin = {}",
        etb[0].emin
    );
}

/// `_aug_etb_element` reads `CONFIGURATION.count(0)`, which is where
/// `CONFIGURATION` and `NRSRHF_CONFIGURATION` diverge. Sc (Z = 21) is the first
/// element where they do.
#[test]
fn configuration_table_is_not_the_nrsrhf_one() {
    use pyscf_df::CONFIGURATION;
    assert_eq!(CONFIGURATION.len(), 119);
    assert_eq!(CONFIGURATION[6], [4, 2, 0, 0], "C");
    assert_eq!(CONFIGURATION[21], [8, 12, 1, 0], "Sc — NRSRHF has [8, 13, 0, 0]");
    assert_eq!(CONFIGURATION[24], [7, 12, 5, 0], "Cr — NRSRHF has [8, 12, 4, 0]");
}
