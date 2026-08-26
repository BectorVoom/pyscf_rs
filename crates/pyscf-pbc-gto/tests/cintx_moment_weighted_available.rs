//! Plans 10-05 / 10-06, **Task 0** — the cintx fail-open check
//! (PBC-MASTER-PLAN §2.4 + risk R-13).
//!
//! GTH pseudopotentials need two moment-weighted cintx families:
//!
//! | symbol | consumer |
//! |---|---|
//! | `int1e_r2_origi`, `int1e_r4_origi` | `pp_int.py:626` `_int_vnl` -> `get_pp_nl` |
//! | `int3c1e_r{2,4,6}_origk` | `pp_int.py:150` `get_pp_loc_part2` |
//!
//! §2.4 records them as `oracle_covered: false` with NO dispatch arm, and warns
//! that `center_3c1e.rs:1469` FALLS THROUGH (`_ => {}`) for an unrecognised
//! operator name — so an unimplemented symbol may return the **unweighted**
//! parent integral instead of erroring. A silently-wrong `<i|r²|j>` would make
//! `get_pp_nl` plausible and wrong.
//!
//! This file therefore asserts BOTH halves of availability:
//!
//! 1. the symbol resolves and evaluates at all, and
//! 2. its value is not identical to the unweighted parent.
//!
//! Disposition per §2.4: an Err means the family is genuinely unimplemented and
//! the dependent numeric gates get `#[ignore]`d; equality with the parent means
//! a shipped-API silent-wrong-answer bug and work STOPS.

use cintx_core::Representation;
use cintx_ops::resolver::Resolver;
use cintx_rs::SessionRequest;
use cintx_runtime::ExecutionOptions;
use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};

/// A two-centre fixture with a non-trivial separation, so an `r^n` weight
/// cannot coincide with the unweighted integral by symmetry.
///
/// `gth-szv` carbon is `1s + 1p` per atom, so the shells run
/// `[0]=s@C0, [1]=p@C0, [2]=s@C1, [3]=p@C1` and [`CROSS`] is the s(C0)/s(C1)
/// pair — the same-atom pairs `(0,1)` are zero by symmetry for BOTH the
/// weighted and the unweighted operator and would make the comparison vacuous.
const CROSS: (usize, usize) = (0, 2);

fn fixture() -> pyscf_core::Mole {
    M(MoleBuildArgs {
        atom: AtomInput::String("C 0 0 0; C 0 0 2.4".into()),
        basis: BasisInput::Name("gth-szv".into()),
        unit: Unit::Bohr,
        ..Default::default()
    })
    .expect("fixture builds")
}

fn eval2c(
    mol: &pyscf_core::Mole,
    symbol: &str,
    ish: usize,
    jsh: usize,
) -> Result<Vec<f64>, String> {
    let descriptor =
        Resolver::descriptor_by_symbol(symbol).map_err(|e| format!("resolver: {e}"))?;
    let basis = mol.cintx_basis().map_err(|e| format!("basis: {e}"))?;
    let shells = basis
        .shell_tuple_for_indices([ish, jsh])
        .map_err(|e| format!("shells: {e}"))?;
    let outcome = SessionRequest::new(
        descriptor.id,
        Representation::Spheric,
        &basis,
        shells,
        ExecutionOptions::default(),
    )
    .query_workspace()
    .map_err(|e| format!("query_workspace: {e}"))?
    .evaluate()
    .map_err(|e| format!("evaluate: {e}"))?;
    Ok(outcome.tensor.owned_values.clone())
}

/// Report, never fail: this test's job is to make the cintx state VISIBLE in the
/// log of every run, so a later numeric failure in `get_pp_nl` is immediately
/// attributable. The hard assertions live in the two tests below it.
#[test]
fn report_moment_weighted_availability() {
    let mol = fixture();
    for symbol in [
        "int1e_ovlp_sph",
        "int1e_r2_origi_sph",
        "int1e_r4_origi_sph",
        "int3c1e_sph",
        "int3c1e_r2_origk_sph",
        "int3c1e_r4_origk_sph",
        "int3c1e_r6_origk_sph",
    ] {
        match Resolver::descriptor_by_symbol(symbol) {
            Ok(d) => println!(
                "{symbol}: RESOLVES (arity {}, oracle_covered {})",
                d.entry.arity, d.entry.oracle_covered
            ),
            Err(e) => println!("{symbol}: UNRESOLVED ({e})"),
        }
    }
    for symbol in ["int1e_r2_origi_sph", "int1e_r4_origi_sph"] {
        match eval2c(&mol, symbol, CROSS.0, CROSS.1) {
            Ok(v) => println!("{symbol}: evaluates -> {:?}", &v[..v.len().min(4)]),
            Err(e) => println!("{symbol}: BLOCKED -> {e}"),
        }
    }
}

/// `int1e_r2_origi` must evaluate AND differ from the unweighted `int1e_ovlp`.
#[test]
fn int1e_r2_origi_is_available_and_not_the_unweighted_parent() {
    let mol = fixture();
    let unweighted = eval2c(&mol, "int1e_ovlp_sph", CROSS.0, CROSS.1)
        .expect("int1e_ovlp is shipped and must succeed");
    match eval2c(&mol, "int1e_r2_origi_sph", CROSS.0, CROSS.1) {
        Err(e) => panic!(
            "BLOCKED on cintx Wave 0.5 (PBC-MASTER-PLAN §2.4 / R-13): \
             int1e_r2_origi_sph did not evaluate: {e}"
        ),
        Ok(v) => {
            assert_eq!(v.len(), unweighted.len(), "component count changed");
            assert!(
                v.iter()
                    .zip(unweighted.iter())
                    .any(|(a, b)| (a - b).abs() > 1e-12),
                "cintx FAIL-OPEN: int1e_r2_origi returned the unweighted int1e_ovlp \
                 ({v:?}); this is a silent-wrong-answer bug in a shipped cintx API — \
                 escalate to cintx Wave 0.5 task W0-05 before trusting any value"
            );
        }
    }
}

/// Same for `int1e_r4_origi`, and additionally `<i|r^4|j> != <i|r^2|j>` — a
/// dispatcher that maps every `origi` variant onto one code path would pass the
/// test above but fail this one.
#[test]
fn int1e_r4_origi_is_available_and_distinct_from_r2() {
    let mol = fixture();
    let ovlp = eval2c(&mol, "int1e_ovlp_sph", CROSS.0, CROSS.1).expect("int1e_ovlp must succeed");
    let r2 = match eval2c(&mol, "int1e_r2_origi_sph", CROSS.0, CROSS.1) {
        Ok(v) => v,
        Err(e) => panic!("BLOCKED on cintx Wave 0.5: int1e_r2_origi_sph: {e}"),
    };
    let r4 = match eval2c(&mol, "int1e_r4_origi_sph", CROSS.0, CROSS.1) {
        Ok(v) => v,
        Err(e) => panic!("BLOCKED on cintx Wave 0.5: int1e_r4_origi_sph: {e}"),
    };
    assert!(
        r4.iter()
            .zip(ovlp.iter())
            .any(|(a, b)| (a - b).abs() > 1e-12),
        "cintx FAIL-OPEN: int1e_r4_origi returned the unweighted int1e_ovlp"
    );
    assert!(
        r4.iter().zip(r2.iter()).any(|(a, b)| (a - b).abs() > 1e-12),
        "cintx FAIL-OPEN: int1e_r4_origi and int1e_r2_origi returned the same values"
    );
}

// ---------------------------------------------------------------------------
// General contractions — a SECOND fail-open surface, found and FIXED 2026-08-26
// ---------------------------------------------------------------------------
//
// §2.4's fail-open warning is about the operator NAME falling through to the
// unweighted parent. A second one was found on 2026-08-26: both families
// mishandled a shell with `nctr > 1` — `origi` silently returned zeros for
// every contraction pair but (0,0) (and got that one wrong too), `origk`
// panicked in `cintx-cubecl/src/transform/c2s.rs:684`. The Cartesian->spherical
// step sized its output from the shell's angular momentum and forgot the
// contraction axis.
//
// **cintx has since fixed both** (`kernels/unstable/{origi,origk,shared}.rs`,
// plus its own `origi_genctr_parity` / `origk_genctr_parity` oracle tests). The
// tests below are the pyscf-rs side of that fix: they pin the corrected values
// against libcint on a general-contraction fixture, so a regression shows up
// here as a numeric failure rather than as a wrong pseudopotential.
//
// `Li`/`gth-szv` — one `s` shell, TWO contractions — is the only general
// contraction among the elements the PBC-MASTER-PLAN §9.2 reference systems
// use, which is why LiF was the single blocked system while the bug stood.

/// A general-contraction fixture: `Li`/`gth-szv` is `nctr = 2`, `l = 0`, so each
/// shell carries 2 AOs and a shell pair is a 2x2 block.
fn general_contraction_fixture() -> pyscf_core::Mole {
    M(MoleBuildArgs {
        atom: AtomInput::String("Li 0 0 0; Li 0 0 3.0".into()),
        basis: BasisInput::Name("gth-szv".into()),
        unit: Unit::Bohr,
        ..Default::default()
    })
    .expect("fixture builds")
}

fn eval_at(mol: &pyscf_core::Mole, symbol: &str, idx: &[usize]) -> Vec<f64> {
    let descriptor = Resolver::descriptor_by_symbol(symbol)
        .unwrap_or_else(|e| panic!("resolver does not know {symbol}: {e}"));
    let basis = mol.cintx_basis().expect("cintx basis");
    let shells = match idx {
        [i, j] => basis.shell_tuple_for_indices([*i, *j]),
        [i, j, k] => basis.shell_tuple_for_indices([*i, *j, *k]),
        other => panic!("unsupported arity {}", other.len()),
    }
    .expect("shell tuple");
    SessionRequest::new(
        descriptor.id,
        Representation::Spheric,
        &basis,
        shells,
        ExecutionOptions::default(),
    )
    .query_workspace()
    .expect("query_workspace")
    .evaluate()
    .expect("evaluate")
    .tensor
    .owned_values
    .clone()
}

/// Compare against libcint element-wise, relative to `max(|reference|, 1)` so
/// the near-zero entries are held to an absolute bound instead of an
/// unreachable relative one.
fn assert_matches_libcint(symbol: &str, got: &[f64], want: &[f64]) {
    assert_eq!(
        got.len(),
        want.len(),
        "{symbol}: cintx returned {} values, libcint has {} — the contraction \
         axis is being dropped again",
        got.len(),
        want.len()
    );
    let mut worst = 0.0_f64;
    for (g, w) in got.iter().zip(want.iter()) {
        worst = worst.max((g - w).abs() / w.abs().max(1.0));
    }
    println!("{symbol}: max rel |delta| vs libcint = {worst:e}");
    assert!(
        worst < 1e-12,
        "{symbol} disagrees with libcint by {worst:e} on a general contraction\n  \
         cintx:   {got:?}\n  libcint: {want:?}"
    );
}

/// `int1e_r{2,4}_origi` on a general contraction, against libcint.
///
/// Reference: `pyscf.gto.M(atom='Li 0 0 0; Li 0 0 3.0', basis='gth-szv',
/// unit='B').intor(name)[0:2, 2:4].ravel(order='F')`, PySCF 2.12.1.
///
/// The broken values were `[19.13, 0, 0, 0]` for `r2` — note that even element 0
/// was wrong, so "the first entry looks plausible" was never a safe check.
#[test]
fn origi_matches_libcint_for_general_contractions() {
    let mol = general_contraction_fixture();
    assert_matches_libcint(
        "int1e_ovlp_sph",
        &eval_at(&mol, "int1e_ovlp_sph", &[0, 1]),
        &[
            0.019058498870202594,
            0.16378771195073671,
            0.16378771195073671,
            0.789635394261138,
        ],
    );
    assert_matches_libcint(
        "int1e_r2_origi_sph",
        &eval_at(&mol, "int1e_r2_origi_sph", &[0, 1]),
        &[
            0.07864376666091163,
            1.611693169830101,
            0.2842951727874758,
            17.157351280779032,
        ],
    );
    assert_matches_libcint(
        "int1e_r4_origi_sph",
        &eval_at(&mol, "int1e_r4_origi_sph", &[0, 1]),
        &[
            0.5998858261728383,
            17.500177490883054,
            -2.448987668624486,
            641.5486765816063,
        ],
    );
}

/// `int3c1e_r{2,4,6}_origk` on a general contraction, against libcint.
///
/// Reference: `gto.moleintor.getints3c(name, m._atm, m._bas, m._env,
/// shls_slice=(0,1,0,1,1,2)).ravel(order='F')`, PySCF 2.12.1. This is the family
/// that used to PANIC, so the length assertion in `assert_matches_libcint` is
/// doing real work: `2x2x2 = 8` values, not 1.
#[test]
fn origk_matches_libcint_for_general_contractions() {
    let mol = general_contraction_fixture();
    assert_matches_libcint(
        "int3c1e_sph",
        &eval_at(&mol, "int3c1e_sph", &[0, 0, 1]),
        &[
            -0.0023142045183919546,
            -0.0005921696933314895,
            -0.0005921696933314896,
            -0.007615297051821254,
            -0.04716640878642131,
            0.0007370995579713977,
            0.0007370995579713993,
            -0.026854262065596962,
        ],
    );
    assert_matches_libcint(
        "int3c1e_r2_origk_sph",
        &eval_at(&mol, "int3c1e_r2_origk_sph", &[0, 0, 1]),
        &[
            -0.017235690887229056,
            -0.0008166030301062022,
            -0.0008166030301062022,
            -0.014386261050353205,
            -0.43369880058863997,
            -0.000717847102356495,
            -0.0007178471023564933,
            -0.43934074950879454,
        ],
    );
    assert_matches_libcint(
        "int3c1e_r4_origk_sph",
        &eval_at(&mol, "int3c1e_r4_origk_sph", &[0, 0, 1]),
        &[
            -0.1444053179095048,
            -0.0012776461780831957,
            -0.001277646178083206,
            -0.05025290930862128,
            -4.244629327217809,
            -0.23557815548874803,
            -0.23557815548874791,
            -11.50205184462882,
        ],
    );
    assert_matches_libcint(
        "int3c1e_r6_origk_sph",
        &eval_at(&mol, "int3c1e_r6_origk_sph", &[0, 0, 1]),
        &[
            -1.3176419201683167,
            -0.04069105311928278,
            -0.04069105311928285,
            1.6122305260295695,
            -44.24107435214476,
            -7.16952029881965,
            -7.16952029881965,
            -428.2229134124223,
        ],
    );
}
