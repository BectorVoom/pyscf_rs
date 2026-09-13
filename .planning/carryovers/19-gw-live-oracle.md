# Carryover — Phase 19 live-oracle arms NOT RUN

**Source:** `.planning/phases/19-periodic-response-relativistic/19-VERIFICATION.md`
(19-19 rollup, 2026-09-13). Nothing below is absorbed into a pass.

## GW live-vs-upstream QP arms (Gate C, per route)

- **What:** `krgw_cd`, `kugw_ac`, `kgw_slow` QP energies vs live upstream
  PySCF 2.12.1 GDF numbers at the per-route 1e-4 floor (19-01 Task 3 /
  19-11 Task 2 / 19-12 Task 3 / 19-13 Task 3).
- **Status:** NOT RUN. The shipped gates are analytic-model arms (CD
  closed-form root at 1e-9; UAC vs bisection inside 1e-4; slow-vs-AC inside
  1e-4) plus oracle-free identities (closed-shell U==R; supercell
  equivalence; `gw_slow` alias bit-identity). No committed live-GW fixture
  exists except 19-10's AC one (`tests/fixtures/krgw_ac_diamond_311.json`).
- **Unblock:** generate CD/UAC fixtures from the vendored 2.12.1 tree on a
  small diamond/GDF cell (same shape as 19-10's fixture: sigma/W rows +
  per-orb grids + `vk`/`vmf` diagonals + QP energies, grid pinned `nw = 100`,
  contour `eta = 1e-3` pinned), then gate at 1e-4 per route.
- **Note:** needs `PYSCF_ORACLE_VENV` + GDF integrals; the DF `W` build
  (`sr_loop` dielectric) is the documented injection seam, not ported.

## Ignored live arms elsewhere in the phase (NOT RUN, not failures)

- `pyscf-pbc-scf` `newton_ah` 1 ignored live arm (19-04).
- `pyscf-pbc-tdscf` `uhf` 1 ignored live arm (19-08).
- `pyscf-pbc-adc` `ea` 1 ignored arm (19-16).
- Each needs the oracle venv; each is recorded here rather than folded.

## Fixed during rollup (no carryover — closed 2026-09-13)

- `kadc_base::t2_first_order_matches_defining_equation` FAILED at rollup
  (sign flip: test used positive-gap denominators, upstream
  `kadc_rhf_amplitudes.py:101-108` uses `_get_epq fac=[1,-1]`, negative
  gaps — bit patterns differed by exactly 2⁶³). Test corrected to the
  upstream convention; `pyscf-pbc-adc --test kadc_base` 7/7. Implementation
  was already correct (live-block comparison in `tests/ip.rs` had pinned it).
