---
phase: 04-dft
plan: 12
subsystem: dft
tags: [ks-energy-cache, dm-fingerprint, hashing, scf-convergence, cr-04]

# Dependency graph
requires:
  - phase: 04-dft
    provides: KsHooks/DfKsHooks per-cycle Exc cache (DFT-01/DFT-07) whose key this plan fixes
provides:
  - Injective u64 content-hash KS energy-cache fingerprint (CR-04 closed)
  - dm_fingerprint(&Density) -> u64 in hooks.rs and df_dft.rs (DefaultHasher over f64 bits)
  - Exact-equality cache-hit guard (no 1e-12 float approximation) in both energy_elec paths
affects: [04-dft gap-closure, RKS/UKS bit-exact energy gate, DF-DFT energy gate]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Cache keys over floating-point data hash the raw f64 bit pattern (v.to_bits()) for injective, deterministic, bit-exact comparison — no L1-norm/approximate keys"

key-files:
  created: []
  modified:
    - crates/pyscf-dft/src/hooks.rs
    - crates/pyscf-dft/src/df_dft.rs

key-decisions:
  - "Used std::collections::hash_map::DefaultHasher (SipHash, stdlib) over the f64 bits — no new crate dependency, satisfying the no-install threat disposition (T-04-12-SC)"
  - "Cache hit requires exact u64 equality, replacing the (c.dm_fingerprint - dm_fingerprint(dm)).abs() < 1e-12 float guard; any density change misses the cache and recomputes the grid loop (the safety-net fallback is retained)"

patterns-established:
  - "f64-data fingerprints hash to_bits() (not value): -0.0 != 0.0 and distinct NaN payloads hash differently — the cache reuses Exc only for a byte-identical density"

requirements-completed: [DFT-01]

# Metrics
duration: 6min
completed: 2026-05-23
---

# Phase 04 Plan 12: KS Energy-Cache Injective Fingerprint (CR-04) Summary

**Replaced the non-injective Σ|D| (L1-norm) KS energy-cache key with a u64 DefaultHasher content hash of the f64 bits in both KsHooks and DfKsHooks, eliminating stale-Exc false cache hits at µHartree convergence.**

## Performance

- **Duration:** ~6 min
- **Started:** 2026-05-23T04:46:00Z
- **Completed:** 2026-05-23T04:52:00Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 2

## Accomplishments
- Closed BLOCKER CR-04: the KS energy cache could return a stale XC energy from a prior SCF cycle when two distinct density matrices share an L1 norm differing by less than 1e-12 — exactly the regime the 1µHartree bit-exact gate operates in.
- `dm_fingerprint(&Density) -> u64` now hashes each element's raw bit pattern (`v.to_bits().hash(&mut h)`) via `std::collections::hash_map::DefaultHasher`, in both `hooks.rs` (KsHooks) and `df_dft.rs` (DfKsHooks) with an identical scheme.
- Cache-hit guards in `KsHooks::energy_elec` and `DfKsHooks::energy_elec` now use exact `u64 ==` comparison (no float approximation); a changed density misses the cache and recomputes the grid loop.
- New test `dm_fingerprint_is_injective` proves two density matrices with identical Σ|D|=4 but different entries produce different fingerprints, and that an identical density is deterministic.
- No new crate dependency (stdlib hasher) — Cargo.toml untouched; the no-install threat disposition (T-04-12-SC) holds.

## Task Commits

Each task was committed atomically (TDD: test → fix):

1. **Task 1 (RED): add failing dm_fingerprint_is_injective** - `f278ef9` (test)
2. **Task 1 (GREEN): injective u64 content-hash KS energy-cache key** - `ad7b164` (fix)

**Plan metadata:** (this docs commit)

## Files Created/Modified
- `crates/pyscf-dft/src/hooks.rs` - `dm_fingerprint` returns `u64` (DefaultHasher over f64 bits); `KsEnergyCache.dm_fingerprint: u64`; `KsHooks::energy_elec` cache guard uses `==`; new `dm_fingerprint_is_injective` test.
- `crates/pyscf-dft/src/df_dft.rs` - identical `dm_fingerprint -> u64` scheme; `DfKsEnergyCache.dm_fingerprint: u64`; `DfKsHooks::energy_elec` cache guard uses `==`.

## Decisions Made
- **stdlib DefaultHasher over the f64 bits.** No external hashing crate (ahash/FxHash) was added — the plan and threat register both mandate a no-dependency fix. `DefaultHasher` (SipHash 1-3) is sufficient: the cache is a single-entry per-cycle correctness guard, not a hot-path hash map, so collision-resistance quality dominates over throughput.
- **Hash the bits, not the value.** `v.to_bits()` makes the fingerprint bit-exact: `-0.0` and `0.0` produce different keys and distinct NaN payloads differ. The cache reuses an `Exc` only for a density that is byte-for-byte the one it scored — the strongest possible correctness guarantee for the SCF energy at convergence.
- **Retained the cache mechanism and its miss-fallback.** The cache still avoids a double grid-loop evaluation per SCF cycle; only the key changed. The `energy_elec` cache-miss branch (fresh `ks_veff` recompute) is unchanged as the safety net.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None. RED confirmed the collision (`4.0 == 4.0` for both fixtures); GREEN passed all 43 lib unit tests plus every integration suite (cam_b3lyp_h2o_rsh, df_dft_match, dtype_f32_smoke, ks_chkfile_roundtrip, numint_signatures, parse_xc_parity, rks_uks_bitexact, vv10_energy_match, wgpu_f64_fallback) with `cargo test -p pyscf-dft` exiting 0. Clippy on `-p pyscf-dft --tests` produced only pre-existing workspace noise (`cintx` patch-not-used, `fma4` target-feature) unrelated to these changes.

## Known Stubs
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- CR-04 closed. Remaining Phase 04 gap-closure: Wave 6 — 04-13 (CR-02); Wave 7 — 04-14 (CR-01).
- libxc is never compiled in the default build; this plan added no dependency, so the ~6h libxc_rs compile remains untriggered.

## Self-Check: PASSED

- FOUND: `.planning/phases/04-dft/04-12-SUMMARY.md`
- FOUND: `f278ef9` (RED — test commit)
- FOUND: `ad7b164` (GREEN — fix commit)

## TDD Gate Compliance

- RED gate: `f278ef9` (`test(04-12): ...`) — failing test committed before implementation.
- GREEN gate: `ad7b164` (`fix(04-12): ...`) — implementation makes the test pass.
- REFACTOR gate: not needed (implementation already minimal).

---
*Phase: 04-dft*
*Completed: 2026-05-23*
