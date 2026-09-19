# 20-04 FIX SUMMARY — KS drivers grid XC on `cell.mesh`, not the DF mesh

**Shipped:** 2026-09-14 (landed in commit `5e6dd65`; the executor was cut off by
a session limit before writing this summary, which the orchestrator wrote from
its logs `target/fix2004_{build,tests,gatec_gdf,bench_check}.log`).

## Defect (from 20-04, class (a))
`Krks/Kuks/Kroks/Kgks::from_df` built the default XC uniform grid on
`with_df.mesh()`. For GDF that is the RS long-range mesh (`[13,13,13]` vs
`cell.mesh` `[35,35,35]` on `si [2,2,2]`), so ksymm Gate C on GDF compared two
different XC quadratures: 1.432e-06 (17-VERIFICATION:126, NOT MET).

## Change
- `crates/pyscf-pbc-dft/src/{krks,kuks,kroks,kgks}.rs`: `PeriodicGrids::uniform(with_df.cell(), None)`
  (resolves `cell.try_mesh()`), with upstream provenance
  (`pbc/dft/rks.py:272`, `gen_grid.py:72`; the explicit exceptions —
  `density_fit()` Becke grids, `multigrid_numint` — noted in the comment).
- Callers that relied on the coupling now pin `mf.grids` explicitly, matching
  what their oracle scripts do (`mf.grids.mesh = mesh`):
  `crates/pyscf-pbc-dft/tests/{gate.rs,gate_openshell.rs,modules.rs,smoke.rs}`,
  `crates/pyscf-bench/src/bin/{krks_repro.rs,krks_profile.rs}`.

## Verification
| command | result |
|---|---|
| `cargo test --release -p pyscf-pbc-dft --test gate --test gate_openshell --test gdf_ksymm_bisect --test krks_ksymm --test modules --test multigrid_device_scatter --test multigrid_kpts --test multigrid_kpts_oracle --test multigrid_scf --test smoke` | exit 0 (krks_ksymm 7 passed, modules 8, device_scatter 4, multigrid_kpts 6, oracle 1, multigrid_scf 3, smoke 13) |
| `PYSCF_ORACLE_VENV=1 … --test krks_ksymm -- --ignored --exact krks_ibz_energy_matches_full_bz_on_gdf` | exit 0, **`|dE| = 2.1997159649345122e-10`** (e_full −7.774588781504, e_ibz −7.774588781724), bound 1e-8 → **Gate C GDF MET** |
| `cargo build --release -p pyscf-bench` | exit 101 — `krks_profile.rs:1694` calls `KsNumInt::unfold_kdms`, which exists only on `KNumInt` (`numint.rs:585`). That line is not part of this fix's hunks (pre-existing bench drift); recorded, not fixed. |

## Not done by the executor
- The `#[ignore]`d upstream oracle gates in `gate.rs`/`gate_openshell.rs`
  (KRKS Si PBE 6.45e-12 floor) were not re-run after the grid pin; routed to
  20-02's finalisation.
