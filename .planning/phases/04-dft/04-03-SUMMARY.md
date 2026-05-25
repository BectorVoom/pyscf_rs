---
phase: 04-dft
plan: 03
subsystem: kernels
tags: [eval_gto, cart2sph, spherical-harmonics, deriv1, gradient, gga, libcint, oracle-sum, dft-grid]

# Dependency graph
requires:
  - phase: 02-gto
    provides: "eval_gto s-shell host CPU path (eval_gto_sph) + AlgebraClient-typed surface; make_env libcint-normalised coefficients; ao_loc_nr"
  - phase: 01-foundation
    provides: "pyscf_algebra::oracle_sum (FMA-free ordered reduction); release-oracle profile; AlgebraClient/select_backend"
provides:
  - "l >= 1 (p/d/f/g) cart->sph AO evaluation in pyscf_kernels::eval_gto_sph (was a Phase 2 zero-stub)"
  - "GTOval_sph_deriv1 (value + 3 Cartesian gradient components) — the GGA grid-loop dRho input"
  - "pyscf-kernels::eval_gto_sph_deriv1 public AlgebraClient-typed surface ([4, ngrids, nao] layout)"
  - "pyscf-gto::eval_gto(\"GTOval_sph_deriv1\") dispatches to the real kernel (no longer NotYetImplemented)"
affects: [04-dft grid loop (D-07), RKS/UKS density evaluation, GGA functionals (PBE/B3LYP), 07-grad]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Embedded libcint g_trans_cart2sph c2s tables (L0..L4) with provenance; FROZEN f64, byte-identical to cintx-cubecl C2S_L*"
    - "Self-contained kernel-level oracle: independent longhand reference (different code path) + analytic structural identities + finite-difference cross-check — passes without a live Python/numpy/pyscf install"
    - "Hand-built libcint _atm/_bas/_env flat arrays in a pyscf-kernels test (cannot depend on pyscf-gto::M — algebra-wall dep inversion)"

key-files:
  created:
    - "crates/pyscf-kernels/tests/eval_gto_lge1.rs"
    - "crates/pyscf-gto/tests/eval_gto_deriv1_oracle.rs"
  modified:
    - "crates/pyscf-kernels/src/eval_gto.rs"
    - "crates/pyscf-kernels/src/lib.rs"
    - "crates/pyscf-gto/src/eval_gto.rs"
    - "crates/pyscf-gto/tests/eval_gto_smoke.rs"
    - "tests/oracle/test_eval_gto.py"

key-decisions:
  - "Embed the libcint Condon-Shortley c2s matrices in pyscf-kernels (small FROZEN const data) rather than add a cintx-cubecl dep — keeps the dep graph unchanged, no libxc/build-time risk, matches the existing inlined-y00 pattern"
  - "Keep the host CPU path as the FMA-free oracle target; NO #[cube] kernel added (Phase 8 owns GPU per D-07)"
  - "Cartesian deriv1 (GTOval_cart_deriv1) stays deferred — it needs the cartesian ao_loc/nao (more AOs than spherical); v1 DFT uses GTOval_sph_deriv1 only"
  - "Route contracted radial sums (>2 prims) through oracle_sum for both value and deriv1 e/e2a (Pitfall 3 / FOUND-06)"

patterns-established:
  - "Pattern 1: cart->sph eval = cartesian monomials (upstream GTOshell_eval_grid_cart loop order) × ordered radial, then g_trans_cart2sph per component"
  - "Pattern 2: deriv1 analytic gradient = e2a·q·mono + e·lq·q^(lq-1)·(other monomials), c2s-transformed component-wise"

requirements-completed: [DFT-01, DFT-10]

# Metrics
duration: ~25min
completed: 2026-05-22
---

# Phase 4 Plan 03: l>=1 eval_gto + GTOval_sph_deriv1 Summary

**Landed the deferred p/d/f cart->sph AO-on-grid evaluation and the GTOval_sph_deriv1 (value + dRho gradient) kernel in pyscf-kernels, wired into pyscf-gto — unblocking rho and grad-rho for real corpus molecules.**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-05-22T03:10Z (approx)
- **Completed:** 2026-05-22T03:34Z
- **Tasks:** 2 (both TDD: RED → GREEN)
- **Files modified:** 7 (2 created, 5 modified)

## Accomplishments
- `l >= 1` cart->sph AO evaluation in `pyscf_kernels::eval_gto_sph` — was a Phase 2 zero-stub; now evaluates p/d/f/g shells element-wise via the libcint `g_trans_cart2sph` transform with the upstream cartesian-monomial ordering.
- `GTOval_sph_deriv1` (`eval_gto_sph_deriv1`): value + 3 Cartesian gradient components per AO (`[4, ngrids, nao]` layout) — the GGA grad-rho input. Verified vs an independent reference AND a finite-difference cross-check.
- `pyscf-gto::eval_gto("GTOval_sph_deriv1", ...)` now dispatches into the kernel instead of returning `NotYetImplemented{phase:4}`.
- Algebra wall intact (`check-dependency-wall` PASS); no `#[cube]` kernel added (host path stays the FMA-free oracle target).

## Task Commits

Each task TDD-committed atomically (test → feat):

1. **Task 1 (RED): failing l>=1 oracle** - `e445289` (test)
2. **Task 1 (GREEN): l>=1 cart->sph eval** - `416ae81` (feat)
3. **Task 2 (RED): failing GTOval_sph_deriv1 oracle** - `9d8cd91` (test)
4. **Task 2 (GREEN): GTOval_sph_deriv1 + pyscf-gto wiring** - `73ea5e4` (feat)
5. **Deviation: close Phase 2 l>=1 xfail in python oracle** - `9e1c5ef` (test)

## Files Created/Modified
- `crates/pyscf-kernels/src/eval_gto.rs` - l>=1 cart->sph value path + `eval_gto_sph_deriv1`; embedded c2s tables L0..L4; `common_fac_sp`/`cart_powers`/`ncart`/`nsph` helpers.
- `crates/pyscf-kernels/src/lib.rs` - re-export `eval_gto_sph_deriv1`.
- `crates/pyscf-kernels/tests/eval_gto_lge1.rs` - **created**: self-contained l>=1 oracle (hand-built s+p+d fixture, independent reference, p/d structural identities).
- `crates/pyscf-gto/src/eval_gto.rs` - `GTOval_sph_deriv1` dispatch into the kernel; variant-table + EvalGtoOutput layout docs; `GTOval_cart_deriv1` stays deferred.
- `crates/pyscf-gto/tests/eval_gto_deriv1_oracle.rs` - **created**: 4-component deriv1 oracle + value==GTOval_sph bit check + finite-difference cross-check.
- `crates/pyscf-gto/tests/eval_gto_smoke.rs` - refreshed the obsolete deriv1-NYI smoke test to assert the new 4-component shape.
- `tests/oracle/test_eval_gto.py` - removed the now-stale `@pytest.mark.xfail` on the cc-pVDZ p-shell parity test.

## Decisions Made
- Embedded the libcint c2s matrices in-crate rather than adding a `cintx-cubecl` dependency: keeps `pyscf-kernels`' dep graph (and the libxc-free guarantee) untouched, and the tables are FROZEN const data matching the existing inlined-`y00` pattern.
- Host CPU path only — no `#[cube]` kernel (Phase 8 GPU per D-07); the host path is the FMA-free oracle target.
- `GTOval_cart_deriv1` deferred (needs cartesian `ao_loc`/`nao`); v1 DFT is spherical.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Refreshed an obsolete smoke test asserting deriv1 == NotYetImplemented**
- **Found during:** Task 2 (deriv1 wiring)
- **Issue:** `eval_gto_smoke.rs::deriv1_returns_not_yet_implemented_phase_4` asserted `GTOval_sph_deriv1` returns `NotYetImplemented{phase:4}` — now false because deriv1 lands. Left as-is it would fail.
- **Fix:** Replaced with `deriv1_sph_returns_four_component_result` (shape `[4,ngrids,nao]` + component-0 == GTOval_sph bit-check) and added `deriv1_cart_returns_not_yet_implemented_phase_4` for the still-deferred cart variant.
- **Files modified:** crates/pyscf-gto/tests/eval_gto_smoke.rs
- **Verification:** `cargo test -p pyscf-gto --test eval_gto_smoke` → 12 passed.
- **Committed in:** 73ea5e4 (Task 2 commit)

**2. [Rule 1 - Bug] Closed the stale Phase 2 l>=1 xfail in the python oracle harness**
- **Found during:** post-Task-2 cleanup
- **Issue:** `tests/oracle/test_eval_gto.py::test_eval_gto_h2o_ccpvdz_includes_p_shells` was `@pytest.mark.xfail` ("l>=1 stubs to zero; Phase 4 extends"). With l>=1 landed it would XPASS, and the xfail/import became stale.
- **Fix:** Removed the `xfail` marker + unused `pytest` import; refreshed the module docstring. (Live-Python harness; runs in CI with numpy+pyscf, not in this sandbox.)
- **Files modified:** tests/oracle/test_eval_gto.py
- **Verification:** `python3 -m py_compile` OK (numpy/pyscf unavailable here; CI runs the assertion).
- **Committed in:** 9e1c5ef

---

**Total deviations:** 2 auto-fixed (both Rule 1 — stale tests made correct after the behavior change).
**Impact on plan:** Both are direct consequences of the planned behavior change (deriv1 now works). No scope creep.

## Issues Encountered
- **No live Python oracle in this environment:** numpy/scipy/pyscf are not importable, so the Phase 2 `dump_*_for_oracle` + pytest path cannot run here. The plan's verify commands are pure-Rust `cargo test`. Resolved by writing self-contained Rust oracles: an *independent* longhand reference (different code path from the kernel — naive cartesian loop + explicit c2s matrices) plus convention-locking analytic identities and a finite-difference gradient cross-check. The cross-implementation diff is tight to ~1e-12.
- **pyscf-kernels cannot depend on pyscf-gto** (algebra-wall dep inversion → cycle). Resolved by hand-constructing the libcint `_atm/_bas/_env`/`ao_loc` flat arrays with `gto_norm`-normalised coefficients directly in the kernel test.

## Verification
- `cargo test --profile release-oracle -p pyscf-kernels eval_gto_lge1` → 4 passed.
- `cargo test --profile release-oracle -p pyscf-gto eval_gto_deriv1_oracle` → 4 passed.
- `cargo run -p xtask --bin check-dependency-wall` → PASS (no cubecl leak).
- Full `cargo test -p pyscf-kernels` (5) and `cargo test -p pyscf-gto` (all green) — no regressions.
- **libxc was NEVER compiled** (all commands scoped to `-p pyscf-kernels` / `-p pyscf-gto`, default features).

## Known Stubs
- `GTOval_cart_deriv1` → `NotYetImplemented{phase:4}` (intentional; needs cartesian ao_loc — v1 DFT uses the spherical deriv1).
- `GTOval_sph_deriv2` / `GTOval_cart_deriv2` → `NotYetImplemented{phase:4}` (intentional; deriv2 not needed for v1 DFT energy).
- `GTOval_ip*` / `GTOval_ig*` → `NotYetImplemented{phase:7}` (intentional; Phase 7 grad).
- `l > 4` (h+ shells) in the cart->sph transform panics loudly (intentional fail-fast; v1 corpus tops out at f, l=3).

## Threat Flags
None — pure compute-kernel extension. The T-04-03 numerical-correctness mitigation (l>=1 AO ordering vs `ao_loc_nr`) is covered by the element-wise oracle + the p/d structural-identity tests. No new network/auth/file/parsing surface.

## Next Phase Readiness
- The DFT grid loop (D-07) can now evaluate rho on any corpus molecule (p/d/f shells) and grad-rho via `GTOval_sph_deriv1` for GGA functionals.
- GPU variant of these kernels remains a Phase 8 item (host path is the oracle reference).

## Self-Check: PASSED

- Created files exist: `crates/pyscf-kernels/tests/eval_gto_lge1.rs`, `crates/pyscf-gto/tests/eval_gto_deriv1_oracle.rs`, `crates/pyscf-kernels/src/eval_gto.rs`, `.planning/phases/04-dft/04-03-SUMMARY.md`.
- Commits exist: e445289, 416ae81, 9d8cd91, 73ea5e4, 9e1c5ef.

---
*Phase: 04-dft*
*Completed: 2026-05-22*
