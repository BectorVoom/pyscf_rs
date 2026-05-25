---
phase: 05-mp2
plan: 04
subsystem: mp2
tags: [ump2, open-shell, spin-block, rdm, density-matrix, oracle-sum, mp2]

# Dependency graph
requires:
  - phase: 05-mp2 (plan 05-03)
    provides: "rmp2_kernel, ChemistsEris, Mp2Reference/Mp2Result, Frozen, get_nocc, oracle_dot/oracle_sum reduction idiom"
  - phase: 05-mp2 (plan 05-01)
    provides: "ump2.rs/rdm.rs stubs, ump2_structural.rs scaffold, Mp2Error bridge"
provides:
  - "ump2_kernel: open-shell UMP2 correlation energy from spin-resolved (aa/ab/bb) (ia|jb) blocks (e_corr = e_aa + e_bb + e_ab)"
  - "UmpAmplitudes { t2aa, t2ab, t2bb } spin-resolved amplitude triple (no in-repo analog)"
  - "UmpReference (alpha/beta Mp2Reference pair) + UmpResult (spin-decomposed energy + amplitudes)"
  - "make_rdm1 (MO/AO 1-RDM via doo/dvv) + gamma1_intermediates"
  - "make_rdm2 (nmo^4 Chemist 2-RDM with occ/vir/frozen sub-block placement)"
affects: [05-05-dfmp2, 05-06-dfmp2-native, 05-07-pyo3, 06-ccsd, 07-grad]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Spin-resolved triple container (UmpAmplitudes) parallels single-channel pyscf-core::Amplitudes"
    - "Per-spin-channel materialize-then-oracle_sum reduction (T-05-04-FP)"
    - "nmo^4 RDM fancy-indexing via explicit oidx/vidx flat-index maps (T-05-04-LAYOUT/FFI)"
    - "ao_repr deferral returns NotYetImplemented rather than a silently-wrong tensor"

key-files:
  created: []
  modified:
    - crates/pyscf-mp2/src/ump2.rs
    - crates/pyscf-mp2/src/rdm.rs
    - crates/pyscf-mp2/src/lib.rs
    - crates/pyscf-mp2/src/hooks.rs
    - crates/pyscf-mp2/tests/ump2_structural.rs

key-decisions:
  - "UmpReference reuses Mp2Reference twice (alpha/beta) mirroring ump2.py mo_*[0/1] rather than a flattened pair"
  - "ump2_kernel takes the three ChemistsEris blocks DIRECTLY (synthetic-block test path) — the pyscf-py bridge / default_ao2mo build them per channel once int2e lands"
  - "make_rdm2 ao_repr=true returns NotYetImplemented (nmo^4 AO back-transform deferred to the Phase-7 gradient consumer) — never a silently-wrong tensor"
  - "RDM free functions ship now; the hooks RDM seam stays cintx#11-gated because producing t2 needs default_ao2mo's NotYetImplemented int2e"

patterns-established:
  - "Spin-block kernel: same-spin antisymmetrized 0.5·(direct − exchange); opposite-spin direct-only (no exchange)"
  - "Antisymmetrized same-spin amplitudes vanish on a·b-symmetric integral blocks (physical fact baked into test design)"

requirements-completed: [MP2-02, MP2-05]

# Metrics
duration: 18min
completed: 2026-05-23
---

# Phase 5 Plan 04: Open-shell UMP2 + MP2 RDMs Summary

**Open-shell `ump2_kernel` summing spin-resolved aa/ab/bb (ia|jb) blocks into `e_aa + e_bb + e_ab`, a new `UmpAmplitudes { t2aa, t2ab, t2bb }` triple, and the MP2 1-RDM (doo/dvv) + nmo^4 2-RDM — all reductions through oracle_sum, all structural tests always-on.**

## Performance

- **Duration:** ~18 min
- **Started:** 2026-05-23T17:00Z (approx)
- **Completed:** 2026-05-23T17:18Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- `ump2_kernel` (port of `pyscf/mp/ump2.py:35-109`): the open-shell MP2 correlation energy as the sum of three spin channels — same-spin αα/ββ (antisymmetrized `0.5·Σ t2·(ia|jb) − 0.5·Σ t2·(ib|ja)`) and opposite-spin αβ (direct only `Σ t2·(ia|JB)`), each channel transforming its own α/β orbital energies. `e_corr = oracle_sum([e_aa, e_ab, e_bb])`.
- `UmpAmplitudes` spin-resolved amplitude triple (`t2aa`/`t2ab`/`t2bb`) — no in-repo analog; the single-channel `pyscf-core::Amplitudes` does not cover spin-resolved storage. Flat-index layout doc-commented per block.
- `UmpReference` (alpha/beta `Mp2Reference` pair) + `UmpResult` (spin-decomposed `e_aa`/`e_ab`/`e_bb` + optional amplitudes).
- `gamma1_intermediates` (port `mp2.py:175-203`): the doo (occ-occ) and dvv (vir-vir) correlation blocks.
- `make_rdm1` (port `mp2.py:151` → `ccsd_rdm._make_rdm1:246`): the MO-basis `nmo×nmo` 1-RDM from doo/dvv with `ao_repr` (C·γ·Cᵀ back-transform) and `with_frozen` (core-diagonal embedding) flags.
- `make_rdm2` (port `mp2.py:275-348`): the `nmo^4` Chemist-notation 2-RDM with dovov occ/vir sub-block placement + Chemist transpose, the dm1 contribution, the separable HF `+4/−2` part, and the frozen `oidx`/`vidx` fancy-index maps.
- Every reduction routed through `oracle_sum`/`oracle_dot` (no `+=` accumulation) → bit-exact, thread-count invariant (verified RAYON_NUM_THREADS=1 vs 8 identical).

## Task Commits

Each task was committed atomically:

1. **Task 1: UMP2 spin-block kernel + UmpAmplitudes container** - `a8af9b2` (feat) — TDD (test + impl landed together: RED via the asymmetric-block test, GREEN via the kernel)
2. **Task 2: make_rdm1 + make_rdm2 + gamma1_intermediates** - `4dace6f` (feat)

**Plan metadata:** (this commit) `docs(05-04)`

## Files Created/Modified

- `crates/pyscf-mp2/src/ump2.rs` - `ump2_kernel`, `UmpAmplitudes`, `UmpReference`, `UmpResult`, the same-spin/opposite-spin channel helpers, per-channel oracle_sum reductions, flat-index docs.
- `crates/pyscf-mp2/src/rdm.rs` - `gamma1_intermediates`, `make_rdm1` (MO/AO, frozen-aware), `make_rdm2` (nmo^4, sub-block placement), + 5 always-on toy tests.
- `crates/pyscf-mp2/src/lib.rs` - re-exports of the new UMP2 + RDM surface (and `Mp2OverrideHooks`).
- `crates/pyscf-mp2/src/hooks.rs` - documented the RDM hook seams as cintx#11-gated wrappers over the shipped free functions.
- `crates/pyscf-mp2/tests/ump2_structural.rs` - 3 always-on synthetic-block tests: hand-computed `e_corr`, α=β reduction to closed-shell `rmp2_kernel`, asymmetric `t2aa != t2bb`.

## Decisions Made

- **UmpReference layout:** reuse `Mp2Reference` twice (`alpha`/`beta`) rather than a flattened pair, mirroring upstream `ump2.py`'s `mo_*[0]`=α / `mo_*[1]`=β indexing. Each channel carries its own coefficients/energies/occupations.
- **Synthetic-block kernel path:** `ump2_kernel` takes the three `ChemistsEris` blocks directly (already transformed). This is the always-on test path (no `intor`); the pyscf-py bridge (05-07) and `default_ao2mo` build them per channel once arity-4 `int2e` lands (cintx#11).
- **`make_rdm2` ao_repr deferral:** the `nmo^4` AO back-transform is deferred to the Phase-7 gradient consumer and returns `NotYetImplemented{plan:4}` rather than a silently-wrong tensor (T-05-04-FFI). The MO-basis path (the one the gradient layer needs first) is complete.
- **RDM hooks seam:** the free functions `make_rdm1`/`make_rdm2` ship now; the `Mp2OverrideHooks::make_rdm{1,2}` seams stay cintx#11-gated because producing the `t2` for an arbitrary reference requires `default_ao2mo`'s `int2e` (NotYetImplemented{phase:2}). Once the integral lands, the bridge calls `rmp2_kernel(..,with_t2=true)` then the free function with no change to this crate.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Test arithmetic / symmetric-block degeneracy in the asymmetric t2 test**
- **Found during:** Task 1 (UMP2 kernel, RED phase)
- **Issue:** The first draft of `ump2_asymmetric_input_gives_distinct_t2` used a·b-symmetric same-spin blocks (`g[a,b]==g[b,a]`). The antisymmetrized same-spin amplitude `t2i[j,a,b] − t2i[j,b,a]` then vanishes identically (a real physical fact), so `t2aa == t2bb == [0,0,0,0]` regardless of the distinct β data, falsely failing the "distinct" assertion.
- **Fix:** Switched the same-spin test blocks to a·b-asymmetric values so the antisymmetrizer produces non-zero, channel-distinct amplitudes. Documented the physical reason in the test.
- **Files modified:** crates/pyscf-mp2/tests/ump2_structural.rs
- **Verification:** `ump2_asymmetric_input_gives_distinct_t2` passes; `t2aa != t2bb`.
- **Committed in:** a8af9b2 (Task 1)

**2. [Rule 1 - Bug] make_rdm2 sub-block test had a wrong hand value**
- **Found during:** Task 2 (make_rdm2)
- **Issue:** The first sub-block placement test used nvir=1 and asserted `dovov == 1.5`; with a single virtual the antisymmetrizer collapses to `(2·0.5 − 0.5)·2 = 1.0`, and a single-virtual block cannot exercise the (a,b)↔(b,a) transpose.
- **Fix:** Rewrote the test with nvir=2 and an asymmetric single-pair amplitude so both the dovov value and the Chemist transpose are verified at distinct flat indices.
- **Files modified:** crates/pyscf-mp2/src/rdm.rs
- **Verification:** `make_rdm2_known_subblock_placement` asserts the exact dovov values (1.0 and −0.2) at the dovov and transposed positions.
- **Committed in:** 4dace6f (Task 2)

**3. [Rule 3 - Blocking] clippy `needless_range_loop` / `too_many_arguments` under -D warnings**
- **Found during:** Tasks 1 and 2
- **Issue:** The nested multi-index tensor loops (which compute 4-D flat indices AND index energy slices in the same body) tripped `needless_range_loop`; `make_rdm1`/`make_rdm2` exceed the 7-argument clippy limit because they carry the upstream surface (t2, nocc, nmo, frozen, ao_repr, with_frozen, mo_coeff, mo_occ).
- **Fix:** Added scoped `#[allow(clippy::needless_range_loop)]` (the raw indices are load-bearing for the flat-index discipline, T-05-04-LAYOUT) and `#[allow(clippy::too_many_arguments)]` (the count is the upstream contract), each with an explanatory comment.
- **Files modified:** crates/pyscf-mp2/src/ump2.rs, crates/pyscf-mp2/src/rdm.rs
- **Verification:** `cargo clippy -p pyscf-mp2 --tests -- -D warnings` exits 0.
- **Committed in:** a8af9b2, 4dace6f

---

**Total deviations:** 3 auto-fixed (2 test bugs, 1 blocking lint). No scope creep — all confined to the plan's named files.
**Impact on plan:** The two test-bug fixes hardened the always-on assertions (the symmetric-block degeneracy is a genuine physics subtlety worth documenting); the lint allows are scoped + justified.

## Issues Encountered

- The closed-shell reduction invariant required reasoning about the exact relationship between the open-shell spin decomposition and the closed-shell `e_ss`/`e_os`: for a spin-symmetric reference `e_ab(open) == e_os(closed)` and `e_aa(open) + e_bb(open) == e_ss(closed)`. The test asserts both, plus total `e_corr` equality.
- Confirmed the MP2 1-RDM trace conservation analytically: `Tr(doo) == −Tr(dvv)` (the doo/dvv traces are the same sum under index relabeling), so `Tr(γ) == 2·nocc` exactly, enabling the `Tr(γ) == nelec` toy assertion.

## Verification

- `cargo test -p pyscf-mp2 --locked`: 30 tests green (19 lib + 2 ccsd_import + 5 rmp2_structural + 4 ump2_structural).
- `cargo clippy -p pyscf-mp2 --tests -- -D warnings`: clean.
- `cargo fmt -p pyscf-mp2 -- --check`: clean.
- `xtask check-no-fma`: PASS (no FMA in release-oracle asm).
- `xtask check-dependency-wall`: PASS (cubecl-* containment intact; no cubecl in pyscf-mp2).
- Determinism: all bit-exact assertions identical under `RAYON_NUM_THREADS=1` and `=8`.

## Next Phase Readiness

- The UMP2 + RDM surface is complete and always-on at the structural level. The pyscf-py bridge (05-07) wires `ump2_kernel` and the RDM free functions to the Python `UMP2`/`make_rdm{1,2}` dispatch.
- Numeric energy/RDM parity remains CI-gated behind cintx#11 (arity-4 `int2e` for in-core; arity-3 `int3c2e_sph` for DF) — the same gate the RMP2 numeric oracle waits on. No code change needed when it lands.
- `make_rdm2` `ao_repr=true` (the `nmo^4` AO back-transform) is deferred to the Phase-7 gradient consumer; the MO-basis 2-RDM the gradients need first is shipped.

## Self-Check: PASSED

- Files exist: `crates/pyscf-mp2/src/ump2.rs`, `crates/pyscf-mp2/src/rdm.rs`, `.planning/phases/05-mp2/05-04-SUMMARY.md`.
- Commits exist: `a8af9b2` (Task 1), `4dace6f` (Task 2).
- Symbols present: `pub fn ump2_kernel`, `pub struct UmpAmplitudes`, `pub fn make_rdm1`, `pub fn make_rdm2`.

---
*Phase: 05-mp2*
*Completed: 2026-05-23*
