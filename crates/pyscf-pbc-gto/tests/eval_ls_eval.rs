//! The EVAL-variant image list for `pbc_eval_gto`.
//!
//! Fix record: `eval_ao_kpts` built its lattice sum with the generic
//! tools-variant `get_lattice_ls` (`pbc.py:601-661`) instead of the eval
//! variant (`eval_gto.py:192-257`) that upstream's `pbc_eval_gto` consumes,
//! and without the norm sort (`eval_gto.py:137-138`) the C screener assumes
//! (`grid_ao.c:58-62`). On the He-fcc gate cell (`sto-3g`, `precision` 1e-8)
//! the tools list has 343 images (`discard = false`) where upstream's eval
//! list has 246 — the extra far images carry ~1e-13 of tail mass into every
//! Bloch AO table, which is the dominant term of the `get_nuc` oracle gap
//! (measured 8.6e-13 on AO tables, 3.3e-13 on `Hcore`).
//!
//! Oracle numbers below were recorded from the vendored upstream PySCF 2.12.1:
//!
//! ```text
//! c = gto.Cell(); c.a = [[0,h,h],[h,0,h],[h,h,0]] (h = 2.834589)
//! c.atom = [('He',(0,0,0))]; c.basis = 'sto-3g'; c.unit = 'Bohr'
//! c.precision = 1e-8; c.build()
//! extract_pgto_params(c, 'diffuse') -> es = [0.31364979], cs = [0.47081682]
//! eval_gto._estimate_rcut(c, 0)     -> [9.30364041]
//! eval_gto.get_lattice_Ls(c, rcut = rcut.max()) -> 246 images,
//!     first row [0,0,0], norms nondecreasing (after the call-site sort)
//! ```
//!
//! The all-electron He cell is used deliberately: no pseudopotential, so the
//! eval path under test is exactly the one `FFTDF.get_nuc` exercises.

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_gto::{
    ALattice, Cell, CellBuildArgs, PgtoOp, estimate_rcut_for_eval, extract_pgto_params,
    get_lattice_ls, get_lattice_ls_eval,
};
use pyscf_pbc_tools::mat3::norm3;

/// He on the fcc lattice of `krhf_bands_oracle.rs`, all-electron `sto-3g`.
fn he_cell() -> Cell {
    let h = 2.834589;
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![("He".into(), [0.0, 0.0, 0.0])]),
            basis: BasisInput::Name("sto-3g".into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]),
        precision: 1e-8,
        ..Default::default()
    })
    .expect("He cell must build")
}

/// `mole.py:4336-4366` parity: the `diffuse` selector, not min-exponent.
#[test]
fn diffuse_selector_matches_upstream_record() {
    let cell = he_cell();
    let (es, cs) = extract_pgto_params(&cell, PgtoOp::Diffuse);
    assert_eq!(es.len(), 1);
    assert!(
        (es[0] - 0.31364979).abs() < 1e-8,
        "diffuse exponent {} != upstream 0.31364979",
        es[0]
    );
    assert!(
        (cs[0] - 0.47081682).abs() < 1e-8,
        "diffuse coefficient {} != upstream 0.47081682",
        cs[0]
    );
    // ... and the eval rcut built on it.
    let rcut = estimate_rcut_for_eval(&cell, 0).expect("eval rcut");
    assert!(
        (rcut[0] - 9.30364041).abs() < 1e-6,
        "eval rcut {} != upstream 9.30364041",
        rcut[0]
    );
}

/// The eval list on the gate cell: 246 images, origin first, norms sorted.
#[test]
fn eval_list_matches_upstream_record() {
    let cell = he_cell();
    let rcut = estimate_rcut_for_eval(&cell, 0).expect("eval rcut");
    let rmax = rcut.iter().copied().fold(0.0_f64, f64::max);
    let mut ls = get_lattice_ls_eval(&cell, rmax).expect("eval Ls");
    assert_eq!(
        ls.len(),
        246,
        "eval image count {} != upstream-recorded 246",
        ls.len()
    );
    // eval_gto.py:137-138 — the call-site stable norm sort.
    ls.sort_by(|a, b| norm3(a).total_cmp(&norm3(b)));
    assert_eq!(ls[0], [0.0, 0.0, 0.0], "origin must sort first");
    let mut prev = 0.0_f64;
    for l in &ls {
        let n = norm3(l);
        assert!(n >= prev, "eval Ls must be norm-sorted after the sort");
        prev = n;
    }
}

/// The eval list is NOT the tools list: different construction, different
/// count at the same cutoff. This pins the fixed bug (they were conflated).
#[test]
fn eval_list_differs_from_tools_list() {
    let cell = he_cell();
    let rcut = estimate_rcut_for_eval(&cell, 0).expect("eval rcut");
    let rmax = rcut.iter().copied().fold(0.0_f64, f64::max);
    let eval_ls = get_lattice_ls_eval(&cell, rmax).expect("eval Ls");
    let tools_ls =
        get_lattice_ls(&cell, Some(rmax), None, false).expect("tools Ls");
    assert_eq!(tools_ls.len(), 343);
    assert_eq!(eval_ls.len(), 246);
    assert_ne!(eval_ls, tools_ls);
}
