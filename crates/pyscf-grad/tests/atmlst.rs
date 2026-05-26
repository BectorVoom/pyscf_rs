//! Structural tests for `atmlst` row-subsetting (GRAD-08) + the GradScanner
//! seam, on a TRIVIAL reference `Gradients` impl — proving the base-trait
//! contract (D-09) before any method body lands (07-03+).
//!
//! Run scoped + single-threaded:
//!   cargo test -p pyscf-grad --locked -- --test-threads=1 atmlst

use pyscf_core::{Energy, Mole, ParsedAtom, PyscfRsError};
use pyscf_grad::scanner::GradScanner;
use pyscf_grad::{Gradients, resolve_atmlst};

/// Build a minimal 3-atom (H3, linear) mock Mole carrying only what the base
/// trait reads: `natm`, `_atom` (→ `atom_coords()` + the symbol→charge
/// fallback). No basis build is triggered (the gradient base API needs only
/// geometry + nuclear charges for `grad_nuc`).
fn mock_h3() -> Mole {
    let atom: Vec<ParsedAtom> = vec![
        ("H".to_string(), [0.0, 0.0, 0.0]),
        ("H".to_string(), [0.0, 0.0, 1.4]),
        ("H".to_string(), [0.0, 0.0, 2.8]),
    ];
    Mole {
        natm: 3,
        _atom: atom,
        ..Default::default()
    }
}

/// A trivial reference gradient whose `grad_elec` returns a FIXED, easily-
/// recognisable full `(natm, 3)` array (row `ia` = `[10*ia, 10*ia+1, 10*ia+2]`),
/// subset to the requested `atmlst` rows. This makes row-identity assertions
/// unambiguous: row `k` of the result must equal the full-gradient row at
/// `atmlst[k]`.
struct RefGrad {
    mol: Mole,
}

impl Gradients for RefGrad {
    fn mol(&self) -> &Mole {
        &self.mol
    }
    fn atmlst(&self) -> Option<&[usize]> {
        None
    }
    fn de(&self) -> Option<&[[f64; 3]]> {
        None
    }

    fn grad_elec(&self, atmlst: Option<&[usize]>) -> Result<Vec<[f64; 3]>, PyscfRsError> {
        // Full gradient = recognisable per-row pattern, then subset.
        let rows = resolve_atmlst(atmlst, self.mol.natm)?;
        Ok(rows
            .iter()
            .map(|&ia| {
                let base = 10.0 * ia as f64;
                [base, base + 1.0, base + 2.0]
            })
            .collect())
    }

    fn make_rdm1e(&self) -> Result<Vec<f64>, PyscfRsError> {
        Ok(vec![])
    }

    // Override grad_nuc to ZERO so the kernel = grad_elec exactly, isolating the
    // row-subsetting assertion from the nuclear-repulsion contribution.
    fn grad_nuc(&self, atmlst: Option<&[usize]>) -> Result<Vec<[f64; 3]>, PyscfRsError> {
        let rows = resolve_atmlst(atmlst, self.mol.natm)?;
        Ok(vec![[0.0, 0.0, 0.0]; rows.len()])
    }
}

#[test]
fn kernel_none_returns_all_rows() {
    let g = RefGrad { mol: mock_h3() };
    let de = g.kernel(None).expect("kernel(None) should succeed");
    assert_eq!(de.len(), 3, "atmlst=None must return all natm rows");
    assert_eq!(de[0], [0.0, 1.0, 2.0]);
    assert_eq!(de[1], [10.0, 11.0, 12.0]);
    assert_eq!(de[2], [20.0, 21.0, 22.0]);
}

#[test]
fn kernel_subset_returns_exactly_those_rows() {
    // GRAD-08 headline: kernel(atmlst=[1,2]) -> a (2,3) array equal to rows
    // [1,2] of the full (3,3) gradient.
    let g = RefGrad { mol: mock_h3() };
    let subset = [1usize, 2usize];
    let de = g.kernel(Some(&subset)).expect("kernel([1,2]) should succeed");

    assert_eq!(de.len(), 2, "atmlst=[1,2] must return exactly 2 rows");
    assert_eq!(de[0], [10.0, 11.0, 12.0], "row 0 must be full-gradient row 1");
    assert_eq!(de[1], [20.0, 21.0, 22.0], "row 1 must be full-gradient row 2");

    // And the order follows atmlst, not the natural atom order.
    let reordered = [2usize, 0usize];
    let de2 = g.kernel(Some(&reordered)).expect("kernel([2,0]) should succeed");
    assert_eq!(de2[0], [20.0, 21.0, 22.0]);
    assert_eq!(de2[1], [0.0, 1.0, 2.0]);
}

#[test]
fn kernel_rejects_out_of_range_atmlst() {
    // T-07-06: an index >= natm returns GradError::ShapeMismatch, never an OOB
    // panic.
    let g = RefGrad { mol: mock_h3() };
    let err = g
        .kernel(Some(&[1usize, 3usize]))
        .expect_err("index 3 >= natm=3 must be rejected");
    assert!(
        format!("{err}").contains("shape mismatch"),
        "out-of-range atmlst must surface ShapeMismatch, got: {err}"
    );
}

#[test]
fn grad_nuc_default_is_translationally_invariant() {
    // The SHARED default grad_nuc (not overridden here) must sum to ~0 over all
    // atoms (Newton's third law / translational invariance) on a neutral system.
    struct PlainGrad {
        mol: Mole,
    }
    impl Gradients for PlainGrad {
        fn mol(&self) -> &Mole {
            &self.mol
        }
        fn atmlst(&self) -> Option<&[usize]> {
            None
        }
        fn de(&self) -> Option<&[[f64; 3]]> {
            None
        }
        fn grad_elec(&self, atmlst: Option<&[usize]>) -> Result<Vec<[f64; 3]>, PyscfRsError> {
            let rows = resolve_atmlst(atmlst, self.mol.natm)?;
            Ok(vec![[0.0, 0.0, 0.0]; rows.len()])
        }
        fn make_rdm1e(&self) -> Result<Vec<f64>, PyscfRsError> {
            Ok(vec![])
        }
    }

    let g = PlainGrad { mol: mock_h3() };
    let nuc = g.grad_nuc(None).expect("grad_nuc should succeed");
    assert_eq!(nuc.len(), 3);
    let sum_z: Vec<f64> = (0..3).map(|c| nuc.iter().map(|r| r[c]).sum()).collect();
    for s in sum_z {
        assert!(s.abs() < 1e-12, "nuclear forces must sum to zero, got {s}");
    }
    // The linear H3 force on atom 1 (middle) is zero by symmetry.
    assert!(nuc[1][2].abs() < 1e-12, "middle atom z-force must vanish");
}

#[test]
fn grad_scanner_returns_energy_and_gradient_tuple() {
    // The GradScanner seam (07-04 consumer): Mole -> (e_tot, de). On a new
    // geometry it re-runs the energy + kernel; de has shape (natm, 3).
    let energy = Box::new(|mol: &Mole| -> Result<Energy, PyscfRsError> {
        // Trivial energy = sum of z-coords (a stand-in for the as_scanner seam).
        let z: Vec<f64> = mol.atom_coords().iter().map(|c| c[2]).collect();
        Ok(Energy(pyscf_algebra::oracle_sum(&z)))
    });
    let grad = Box::new(
        |mol: &Mole, atmlst: Option<&[usize]>| -> Result<Vec<[f64; 3]>, PyscfRsError> {
            let rows = resolve_atmlst(atmlst, mol.natm)?;
            // Each atom's z-force is 1 (dE/dz = 1), others 0.
            Ok(rows.iter().map(|_| [0.0, 0.0, 1.0]).collect())
        },
    );

    let scanner = GradScanner::new(energy, grad);
    let mol = mock_h3();

    let (e_tot, de) = scanner.eval(&mol, None).expect("scanner.eval should succeed");
    assert_eq!(de.len(), mol.natm, "de must have shape (natm, 3)");
    // sum of z = 0 + 1.4 + 2.8 = 4.2
    assert!((e_tot.0 - 4.2).abs() < 1e-12, "e_tot = {}", e_tot.0);
    assert_eq!(de[0], [0.0, 0.0, 1.0]);

    // atmlst subset flows through the scanner too.
    let (_e, de_sub) = scanner
        .eval(&mol, Some(&[2usize]))
        .expect("scanner.eval subset should succeed");
    assert_eq!(de_sub.len(), 1);
    assert_eq!(de_sub[0], [0.0, 0.0, 1.0]);
}
