---
phase: 05-mp2
plan: 06
subsystem: mp2
tags: [mp2, ri-mp2, df-mp2, dfmp2_native, density-fitting, cholesky, oracle, rust]

# Dependency graph
requires:
  - phase: 05-05
    provides: "DFRMP2/DFUMP2 + df_ao2mo + dfrmp2_kernel (conventional DF-MP2 path; the cross-check anchor)"
  - phase: 05-03
    provides: "Mp2Reference, ChemistsEris, Frozen, frozen_mask, rmp2_kernel"
  - phase: 05-04
    provides: "UmpReference, UmpResult, ump2_kernel (open-shell base)"
  - phase: 03 (df)
    provides: "pyscf_df::cholesky_eri (3c Cholesky B-tensor), DfIntegrals, default_ri (mp2fit aux)"
  - phase: algebra
    provides: "oracle_sum/oracle_dot (bit-exact reductions), solve_linear (CPHF landing)"
provides:
  - "Native RI-MP2 energy fast path (RHF + UHF) on a distinct module path (dfmp2_native)"
  - "emp2_rhf: occupied-pair native contraction (port dfmp2_native.py:374-427)"
  - "emp2_uhf: open-shell native contraction (port dfump2_native.py:272-355)"
  - "NativeDFRMP2 / NativeDFUMP2 driver structs with energy kernel()"
  - "solve_cphf_rhf status-marker stub (documents the solve_linear relaxed-RDM landing, D-06)"
  - "Always-on native↔conventional cross-check (un-gated correctness anchor, T-05-06-XCHECK)"
affects: [05-07 (pyo3 overlay exposes pyscf.mp.dfmp2_native), ccsd, gradients]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Native [i,Q,b] B-tensor layout (ints_cholesky order [mo1,aux,mo2]) vs conventional [i,a,Q]"
    - "Occupied-pair (j<=i) contraction with per-pair Kab=(ia|jb) through oracle_dot"
    - "Cross-path agreement test: native == conventional on identical synthetic B-tensor (relative tolerance)"
    - "Status-marker stub documenting the algebra-wall (solve_linear) landing for a deferred optional surface"

key-files:
  created:
    - "crates/pyscf-mp2/tests/dfmp2_native_structural.rs"
  modified:
    - "crates/pyscf-mp2/src/dfmp2_native.rs"
    - "crates/pyscf-mp2/src/lib.rs"

key-decisions:
  - "Native path reuses pyscf_df::cholesky_eri (shipped 3c Cholesky) — NO second Cholesky (Don't-Hand-Roll)"
  - "Native fast path computes e_corr only (no SS/OS split, no t2); the conventional dfmp2 path owns the decomposition + amplitudes"
  - "solve_cphf_rhf staged behind NotYetImplemented{plan:5} status marker (D-06: relaxed RDM is the optional native extra; energy is the core MP2-04 deliverable)"
  - "Native uses default_ri (mp2fit *-ri aux), NOT default_jkfit"
  - "ps = pt = 1.0 fixed (plain RI-MP2); generic SCS split lives in crate::mp2::scs_energy"

patterns-established:
  - "Pattern 1: native vs conventional energy cross-check is the un-gated correctness anchor (no cintx#11 needed)"
  - "Pattern 2: relative tolerance for cross-path float agreement (different reduction orders → bit-close not bit-identical)"

requirements-completed: [MP2-04]

# Metrics
duration: 18min
completed: 2026-05-23
---

# Phase 5 Plan 06: Native RI-MP2 Fast Path Summary

**Pure-Rust native RI-MP2 (`emp2_rhf`/`emp2_uhf`) ported from `pyscf/mp/dfmp2_native.py` on a distinct `dfmp2_native` module path, reusing the shipped pyscf-df 3c Cholesky B-tensor, with an always-on synthetic cross-check proving native == conventional DF-MP2.**

## Performance

- **Duration:** ~18 min
- **Started:** 2026-05-23T08:30Z
- **Completed:** 2026-05-23
- **Tasks:** 2
- **Files modified:** 3 (1 created, 2 modified)

## Accomplishments

- **Native RI-MP2 energy fast path (R + U)** ported from upstream `dfmp2_native.py`/`dfump2_native.py` as a pure-Rust contraction — NO C dependency (`lib.dot`/`lib.einsum` reductions ported to `oracle_dot`/`oracle_sum`).
- **`emp2_rhf`** walks occupied PAIRS `(i, j ≤ i)`, forms per-pair `Kab[a,b] = Σ_Q B^Q_ia·B^Q_jb = (ia|jb)`, and accumulates `2(ps+pt)·ΣTab·Kab − 2pt·ΣTab·Kabᵀ` (j<i) + `ps·ΣTab·Kab` (j==i diagonal), every reduction through the oracle (no bare `+=`).
- **`emp2_uhf`** ports the open-shell native math: same-spin (αα, ββ) `pt`-scaled antisymmetrized (`(Kab−Kabᵀ)/DE`, j<i pairs) + opposite-spin (αβ) `ps`-scaled direct (all i(α)×j(β) pairs).
- **Distinct module path:** `NativeDFRMP2`/`NativeDFUMP2` re-exported under `Native*` names — `pyscf.mp.dfmp2_native` is NOT the default `mp.DFMP2` factory (which is the conventional 05-05 path). Native upstream `DFRMP2` subclasses `lib.StreamObject`, not `mp2.RMP2`.
- **Reuses the shipped 3c Cholesky:** `transform_b_to_iqb` MO-transforms `pyscf_df::cholesky_eri`'s `b_uvq` into the native `[i,Q,b]` layout — NO second Cholesky (Don't-Hand-Roll).
- **Always-on cross-path agreement** (T-05-06-XCHECK): native `emp2_rhf` equals conventional `dfrmp2_kernel` `e_corr` on the same synthetic B-tensor across three (nocc,nvir,naux) shapes + the 1×1 case — catches a divergent native math port WITHOUT cintx#11 live integrals.
- **CPHF relaxed-RDM** (`solve_cphf_rhf`) staged behind a documented `NotYetImplemented{plan:5}` status marker, with the intended `pyscf_algebra::solve_linear` landing documented (D-06: the relaxed RDM is the optional native extra; energy is the core MP2-04 deliverable).

## Task Commits

Each task was committed atomically:

1. **Task 1: Native RI-MP2 energy fast path (emp2_rhf + ints3c_cholesky reuse)** — `bcf95f5` (feat)
2. **Task 2: Native DF-MP2 structural tests (always-on synthetic + cross-check)** — `2c0cca3` (test)

**Plan metadata:** (this commit) (docs: complete plan)

## Files Created/Modified

- `crates/pyscf-mp2/src/dfmp2_native.rs` — Native RI-MP2 path: `emp2_rhf`, `emp2_uhf`, `transform_b_to_iqb` (`[i,Q,b]` MO transform), `kab_from_slices` (per-pair `(ia|jb)` Q-fold), `same_spin_native` helper, `NativeDFRMP2`/`NativeDFUMP2` driver structs + `kernel()`, `solve_cphf_rhf` status-marker stub.
- `crates/pyscf-mp2/src/lib.rs` — Re-export `NativeDFRMP2`, `NativeDFUMP2`, `emp2_rhf`, `emp2_uhf`, `solve_cphf_rhf` under a clearly-namespaced block.
- `crates/pyscf-mp2/tests/dfmp2_native_structural.rs` — 5 always-on arms: native↔conventional cross-check (3 shapes), driver-kernel wiring, UHF longhand-reference match + driver wiring, single-orbital cross-check, int3c2e_sph gate propagation (no panic).

## Decisions Made

- **Native computes `e_corr` only** — the per-pair contraction does not materialize the full amplitude block, so `NativeDFRMP2::kernel` reports `(e_ss, e_os) = (0, e_corr)` placeholders and `t2 = None`. Callers needing the SS/OS split + amplitudes use the conventional `dfmp2` path. This mirrors upstream's separate `make_rdm1`/density-contribs machinery.
- **`solve_cphf_rhf` status-marker stub** — the full relaxed-RDM surface needs the orbital-gradient machinery (`rmp2_densities_contribs`/`orbgrad_from_Gamma`) plus a working `fock_response_rhf` (which itself needs arity-4 `int2e`, the same cintx#11 gap). The response RHS `Lvo` is not producible un-gated, so wiring it now would be premature. The intended `solve_linear`-based landing is documented in the stub doc-comment.
- **`ps = pt = 1.0` fixed** (plain RI-MP2); the generic SCS split is already provided by `crate::mp2::scs_energy`, so the native fast path keeps the default factors.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Cross-path agreement test used an absolute tolerance unfit for the synthetic energy magnitude**
- **Found during:** Task 2 (native structural tests)
- **Issue:** The native↔conventional cross-check asserted `(native - conv).abs() < 1e-10`. The synthetic toy reference has near-degenerate denominators producing an energy magnitude of ~3.3e6, so the two paths (which sum the SAME integrals in DIFFERENT reduction orders — bit-close but not bit-identical) differed by ~5e-7 absolute = ~1.5e-13 relative. The absolute tolerance falsely failed.
- **Fix:** Switched to a relative tolerance `(native - conv).abs() / native.abs().max(1.0) < 1e-10` (12+ significant figures). The cross-check still proves the native math agrees with the conventional path; only the comparison was corrected to be scale-aware.
- **Files modified:** crates/pyscf-mp2/tests/dfmp2_native_structural.rs
- **Verification:** All 5 native arms pass; the cross-check holds across (2,2,3), (2,3,4), (3,2,3) and the 1×1 case.
- **Committed in:** 2c0cca3 (Task 2 commit)

**2. [Rule 3 - Blocking] `needless_range_loop` clippy lint in the UHF longhand reference**
- **Found during:** Task 2 (native structural tests)
- **Issue:** The opposite-spin loop in the longhand `reference_emp2_uhf` indexes the energy slices with raw `i`/`j` while also computing the flat `[i,Q,b]` offset — `cargo clippy --tests -- -D warnings` flagged `needless_range_loop`.
- **Fix:** Added `#[allow(clippy::needless_range_loop)]` with a comment explaining the raw indices are load-bearing for the flat-index layout (mirrors the identical allow in the production `ump2.rs`).
- **Files modified:** crates/pyscf-mp2/tests/dfmp2_native_structural.rs
- **Verification:** `cargo clippy -p pyscf-mp2 --tests -- -D warnings` exits 0.
- **Committed in:** 2c0cca3 (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (1 test-tolerance bug, 1 blocking clippy lint)
**Impact on plan:** Both confined to the test file; the native production math was correct on the first build. No scope creep.

## Issues Encountered

- The native B-tensor layout is `[i, Q, b]` (`ints_cholesky` order `[mo1,aux,mo2]`), differing from the conventional path's `[i, a, Q]`. Resolved by a dedicated `transform_b_to_iqb` (same materialize-then-`oracle_sum` math, different output stride) so each occupied `i` slice is a contiguous `[naux, nvir]` matrix — the I/O-optimal layout upstream picks for the per-`i` `Kab` contraction.

## Known Stubs

| Stub | File | Line | Reason |
|------|------|------|--------|
| `solve_cphf_rhf` → `NotYetImplemented{plan:5}` | crates/pyscf-mp2/src/dfmp2_native.rs | (status-marker fn) | Intentional (D-06): the CPHF relaxed-RDM is the optional native extra; it needs the orbital-gradient machinery + arity-4 `int2e` (cintx#11) to produce the response RHS un-gated. The energy fast path (the core MP2-04 native deliverable) is fully wired. The intended `solve_linear` landing is documented in the stub. |

The native RI-MP2 ENERGY (the plan's core deliverable) is NOT stubbed — it is fully wired and cross-checked against the conventional path un-gated. The only stub is the deliberately-deferred optional relaxed-RDM surface.

## Verification

- `cargo build -p pyscf-mp2 --locked` exits 0.
- `cargo test -p pyscf-mp2 --locked` green (19 lib + 2 ccsd + 5 dfmp2_native + 5 dfmp2 + 5 rmp2 + 4 ump2 = all pass).
- `cargo test -p pyscf-mp2 --locked dfmp2_native` selects + passes all 5 native arms.
- `cargo clippy -p pyscf-mp2 --locked --tests -- -D warnings` exits 0.
- `cargo fmt -p pyscf-mp2 --check` clean.
- `check-dependency-wall` PASS (pyscf-mp2 stays cubecl-free).
- Acceptance greps: `pub fn emp2_rhf`, `cholesky_eri|DfIntegrals`, `default_ri` all present; 22 oracle reduction sites, 0 bare `+=` energy accumulation.

## Next Phase Readiness

- The native RI-MP2 energy path (R + U) is complete and ships on a distinct module path. The pyscf-py overlay (05-07) can expose `pyscf.mp.dfmp2_native` separately from the default `mp.DFMP2` factory via the `Native*` re-exports.
- Numeric live-PySCF parity stays in the cintx#11-gated `dfmp2_native_energy` oracle arm (registered in 05-01); it flips on with no code change once arity-3 `int3c2e_sph` lands.
- The CPHF relaxed-RDM is a deferred optional follow-on (status-marker stub) — it lands once arity-4 `int2e` is available to produce the response RHS.

## Self-Check: PASSED

All created/modified files exist on disk; both task commits (`bcf95f5`, `2c0cca3`) are present in the git log.

---
*Phase: 05-mp2*
*Completed: 2026-05-23*
