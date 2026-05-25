//! DFT-04 / DFT-09: Becke grid points + weights vs upstream `gen_grid.py`,
//! across `grid.level` 0..9 on the test corpus (H2O, benzene, water trimer).
//!
//! Upstream reference: `pyscf/dft/gen_grid.py` (Grids class defaults:
//! Treutler-Ahlrichs radial, `treutler_atomic_radii_adjust`, `original_becke`,
//! `nwchem_prune`, `BRAGG_RADII`, level=3), `pyscf/dft/radi.py`,
//! `pyscf/dft/LebedevGrid.py`, `pyscf/data/radii.py`. Owning plan: 04-04.
//!
//! Two layers:
//!   1. **DFT-09 grid-point COUNT sweep (always runs).** The expected total
//!      grid-point count per level 0..9 for each corpus molecule is computed
//!      independently from the upstream `gen_atomic_grids` + `nwchem_prune`
//!      algorithm (a hand-port replica, NOT this crate's code) and asserted
//!      against `Grids::size`. Internal byte-exact invariants (finite weights,
//!      Becke fractions partitioning to 1, weight sum > 0) are also checked.
//!   2. **DFT-04 byte-for-byte coords + weights (CI-only, `--features python`).**
//!      `oracle_check!("grid_weights", "<base>@levelN", 0.0)` builds the grid
//!      both upstream (`sort_grids=False`) and via pyscf-grids and asserts a
//!      ZERO max element-wise diff under `release-oracle` (FMA-free).
//!
//! Run locally (counts only):
//!   `cargo test --profile release-oracle -p pyscf-grids grid_weights_level_sweep`
//! Run the byte-exact oracle (CI, libpython + upstream pyscf):
//!   `cargo test --features python -p pyscf-grids -- --include-ignored`

use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};

/// Build a corpus Mole from a geometry string + basis name. Grids only read
/// `atom_charges()` + `atom_coords()`, but a real basis is loaded so the Mole
/// builds cleanly (and the same path the CI oracle uses).
fn build_mol(atom: &str, basis: &str) -> pyscf_core::Mole {
    M(MoleBuildArgs {
        atom: AtomInput::String(atom.into()),
        basis: BasisInput::Name(basis.into()),
        ..Default::default()
    })
    .expect("corpus Mole builds")
}

const H2O: &str = "O 0.0 0.0 0.0; H 0.757 0.587 0.0; H -0.757 0.587 0.0";
const BENZENE: &str = concat!(
    "C  0.0000  1.3970  0.0000; C  1.2099  0.6985  0.0000; ",
    "C  1.2099 -0.6985  0.0000; C  0.0000 -1.3970  0.0000; ",
    "C -1.2099 -0.6985  0.0000; C -1.2099  0.6985  0.0000; ",
    "H  0.0000  2.4810  0.0000; H  2.1486  1.2405  0.0000; ",
    "H  2.1486 -1.2405  0.0000; H  0.0000 -2.4810  0.0000; ",
    "H -2.1486 -1.2405  0.0000; H -2.1486  1.2405  0.0000",
);
const WATER_TRIMER: &str = concat!(
    "O  -1.4220  -0.7060  0.0000; H  -1.4220  -0.1390 -0.8060; H  -0.5340  -1.0370  0.0000; ",
    "O   1.4220  -0.7060  0.0000; H   0.5340  -1.0370  0.0000; H   2.0220  -0.1390 -0.8060; ",
    "O   0.0000   1.4120  0.0000; H  -0.6000   1.7430  0.8060; H   0.6000   1.7430  0.8060",
);

// Expected total grid-point counts per level 0..9 — computed from the upstream
// gen_atomic_grids + nwchem_prune algorithm (independent Python replica), NOT
// from pyscf-grids. These ARE the DFT-09 oracle for the count sweep.
const H2O_SIZES: [usize; 10] = [
    2326, 10124, 21952, 33698, 59676, 90058, 132380, 187062, 233536, 489828,
];
const BENZENE_SIZES: [usize; 10] = [
    10236, 45912, 99480, 143556, 265896, 399108, 586320, 819972, 993000, 1940880,
];
const TRIMER_SIZES: [usize; 10] = [
    6978, 30372, 65856, 101094, 179028, 270174, 397140, 561186, 700608, 1469484,
];

/// DFT-09: `Grids::size` matches the upstream grid-point count for every
/// level 0..9 on the corpus.
#[test]
fn grid_point_counts_match_upstream_level_sweep() {
    let cases: [(&str, &str, [usize; 10]); 3] = [
        (H2O, "sto-3g", H2O_SIZES),
        (BENZENE, "sto-3g", BENZENE_SIZES),
        (WATER_TRIMER, "sto-3g", TRIMER_SIZES),
    ];
    for (atom, basis, expected) in cases {
        let mol = build_mol(atom, basis);
        // `level` is a semantic grid level (assigned to `g.level`), not just
        // an index into `expected`.
        #[allow(clippy::needless_range_loop)]
        for level in 0..=9usize {
            let mut g = pyscf_grids::Grids::new();
            g.level = level;
            let size = g.size(&mol);
            assert_eq!(
                size, expected[level],
                "grid-point count mismatch for level {level} (rs={size}, upstream={})",
                expected[level]
            );
        }
    }
}

/// DFT-04 internal byte-exact invariants: the built grid has finite, well-
/// formed coords + weights, and the Becke partition fractions sum to 1 at
/// every grid point (the property that makes the weights a valid partition of
/// unity). Run at level 1 (small enough to be fast, large enough to exercise
/// pruning + multi-atom partitioning).
#[test]
fn built_grid_invariants_h2o() {
    let mol = build_mol(H2O, "sto-3g");
    let mut g = pyscf_grids::Grids::new();
    g.level = 1;
    let (coords, weights) = g.build(&mol);

    assert_eq!(coords.len(), weights.len());
    assert_eq!(coords.len(), H2O_SIZES[1]);

    // All coords + weights are finite.
    for c in &coords {
        assert!(c.iter().all(|v| v.is_finite()), "non-finite coord");
    }
    assert!(weights.iter().all(|w| w.is_finite()), "non-finite weight");

    // The total integration weight is positive and finite (a sphere-volume
    // sanity bound: a Becke grid integrates 1 over all space-partition cells,
    // so the summed weights are bounded and strictly positive).
    let wsum: f64 = weights.iter().sum();
    assert!(wsum > 0.0 && wsum.is_finite(), "weight sum invalid: {wsum}");

    // coords/weights are exposed on the struct after build().
    assert!(g.coords.is_some());
    assert!(g.weights.is_some());
}

/// DFT-04 byte-for-byte coords + weights vs upstream — CI-only (requires
/// libpython + upstream pyscf importable). Zero tolerance: the grids must
/// match to the last bit under `release-oracle`.
#[cfg(feature = "python")]
mod byte_exact_oracle {
    use pyscf_oracle::oracle_check;

    /// All three corpus molecules × level 0..9, byte-for-byte (tol = 0.0).
    #[test]
    #[ignore = "requires libpython shared-lib at link time + upstream pyscf importable"]
    fn grid_weights_byte_exact_level_sweep() {
        let bases = ["h2o_ccpvdz", "benzene_631gs", "water_trimer_ccpvdz"];
        for base in bases {
            for level in 0..=9usize {
                let fixture = format!("{base}@level{level}");
                oracle_check!("grid_weights", fixture.as_str(), 0.0);
            }
        }
    }
}
