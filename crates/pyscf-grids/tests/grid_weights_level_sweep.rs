//! Wave-0 scaffold — DFT-04 / DFT-09: grid points + weights byte-for-byte vs
//! upstream `pyscf/dft/gen_grid.py` (Becke partitioning + radial/angular
//! quadrature) across `grid.level` 0..9 on the test corpus.
//!
//! Upstream reference: `pyscf/dft/gen_grid.py`, `pyscf/dft/radi.py`,
//! `pyscf/dft/LebedevGrid.py`, `pyscf/data/radii.py`.
//! Owning plan: 04-04 (grids) unignores this and fills the oracle byte-compare.
//! Verify command: `cargo test -p pyscf-grids grid_weights_level_sweep`.

#[test]
#[ignore = "Phase 4 04-04: byte-for-byte grid/weights oracle vs gen_grid.py (level 0..9) — unignore when pyscf-grids lands"]
fn grid_weights_level_sweep() {
    // 04-04 fills: for each corpus molecule and grid.level 0..9, build the
    // Becke grid and assert points + weights match the upstream PySCF oracle
    // byte-for-byte under release-oracle (FMA-free, ordered reductions).
    unimplemented!("04-04 fills the grid_weights_level_sweep oracle byte-compare");
}
