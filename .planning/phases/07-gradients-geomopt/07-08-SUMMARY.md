---
phase: 07-gradients-geomopt
plan: 08
subsystem: gradients
tags: [gradients, ccsd, ecp, lambda, z-vector, cphf, relaxed-density, ECPscalar_ipnuc, ECPscalar_iprinv, oracle_sum, single-cphf, GRAD-06, GRAD-07, D-04, cintx-gated]

# Dependency graph
requires:
  - phase: 07-gradients-geomopt
    plan: 07
    provides: "cphf::solve — the SINGLE matrix-free Krylov CPHF/CPKS solver (D-03/GRAD-10) + the Fvind contract; the relaxed-density→Xvo→cphf::solve→response_dm1 Z-vector pattern (CCSD reuses it verbatim with its own fvind/RHS, max_cycle=50 vs MP2's 30); build_xvo_base int2e-only RHS arm"
  - phase: 07-gradients-geomopt
    plan: 03
    provides: "RhfReference + the grad_elec base decomposition (Hellmann-Feynman + 2e Pulay + overlap Pulay); make_rdm1e; get_ovlp/get_hcore/hcore_deriv (the get_hcore path the ECP ipnuc term folds into); the structural-lands / numeric-#[ignore]'d cintx-gating precedent (D-02); assert_component_leading [3,nao,nao]"
  - phase: 07-gradients-geomopt
    plan: 01
    provides: "CintxEcpEngine::ecp_int1e_ipnuc (ECPscalar_ipnuc cintx-READY, un-gated [3,nao,nao]); ECPscalar_iprinv MISSING → clean cintx-availability error; the 6 grad-intor families (int2e_ip1 + int1e_ip{ovlp,kin,nuc,rinv}) MISSING → clean cintx-availability error, never NotYetImplemented{phase:7}"
  - phase: 06-ccsd
    provides: "solve_lambda (the converged Λ, lambda.rs:411) + make_rdm1/make_rdm2 (incl. ao_repr, rdm.rs:202,296) + ChemistsEris + CcsdReference — CONSUMED DIRECTLY by CCSD-grad (D-04, NO re-derivation)"
  - phase: 01-foundation
    provides: "pyscf_algebra::oracle_sum/oracle_dot (deterministic reductions); pyscf_runtime::WorkspacePool (the Phase-6 arena solve_lambda/make_rdm consume)"
provides:
  - "CcsdGradients + CcsdGradReference — the CCSD relaxed-density gradient driver (GRAD-06, the SECOND non-variational grad after MP2): consumes the Phase-6 solve_lambda + make_rdm1 directly (NO Λ re-derivation, D-04/T-07-25) and re-enters the ONE cphf::solve for the orbital-relaxation Z-vector (max_cycle=50, the upstream default — NOT MP2's 30)"
  - "ccsd::relaxed_rdm1 — the Phase-6 consumption point (runs solve_lambda → make_rdm1(ao_repr=false), returns the MO-basis relaxed 1-RDM); ccsd::response_dm1 (the Z-vector through cphf::solve); ccsd::grad_elec (the RHF-base assembly); CCSD_CPHF_MAX_CYCLE (= 50)"
  - "ecp::get_hcore_ecp — the get_hcore '+ ECPscalar_ipnuc' term (cintx-READY, numeric UN-GATED): real [3,nao,nao] for an ECP molecule, all-zero for a non-ECP molecule (never a panic); normalises the engine's component-inner buffer to the RHF component-leading F-order"
  - "ecp::hcore_deriv_ecp — the hcore_deriv '+ ECPscalar_iprinv' per-atom term routed to a CLEAN cintx-availability error (MISSING from cintx, T-07-27 — never a panic/silent-zero/NotYetImplemented{phase:7})"
affects: [07-09-pyo3-bridge, 07-10-oracle-closeout]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "CCSD relaxed-density grad CONSUMES the Phase-6 surface (D-04/T-07-25): solve_lambda + make_rdm1 run via pyscf-ccsd; the single_lambda_solver_in_grad source-scan forbids a second `fn ...lambda...` declaration in pyscf-grad"
    - "ONE-solver consumer contract (D-03/GRAD-10): CCSD reuses the SAME cphf::solve as MP2 with its own fvind (ENERGY int2e get_veff, cintx-ready) + RHS; CCSD leaves max_cycle at the upstream default 50 (MP2 overrides to 30 — Pitfall 5)"
    - "ECP-gradient hcore split (07-01 D-02): ECPscalar_ipnuc (get_hcore term) cintx-ready → numeric un-gates; ECPscalar_iprinv (hcore_deriv per-atom term) MISSING → clean availability error; both dispatch through CintxEcpEngine"
    - "ECP ipnuc layout normalisation: the engine's component-INNER buffer (data[comp + p*3 + q*3*nao]) is repacked to the RHF component-leading F-order (out[comp*nao*nao + i + j*nao]) so the term folds into the RHF get_hcore path"
    - "no-ECP molecule → zero contribution (NOT an error): get_hcore_ecp maps EcpEngineNotAvailable to an all-zero [3,nao,nao] buffer"

key-files:
  created:
    - crates/pyscf-grad/tests/ccsd_verify_fd.rs
    - crates/pyscf-grad/tests/ecp_verify_fd.rs
  modified:
    - crates/pyscf-grad/src/ccsd.rs
    - crates/pyscf-grad/src/ecp.rs
    - crates/pyscf-grad/src/lib.rs

key-decisions:
  - "CCSD-grad CONSUMES the Phase-6 solve_lambda + make_rdm1 directly (D-04/GRAD-06/T-07-25) — there is NO second lambda-equation solver in pyscf-grad. The single_lambda_solver_in_grad test source-scans src/ for any `fn ...lambda...` declaration and asserts NONE (the body only CALLS pyscf_ccsd::solve_lambda)."
  - "CCSD leaves the cphf max_cycle at the upstream default 50 (CCSD_CPHF_MAX_CYCLE = cphf::DEFAULT_MAX_CYCLE), distinct from MP2's 30 override — pyscf/grad/ccsd.py does NOT override the cphf.solve default the way mp2.py:280 does."
  - "The CCSD Z-vector fvind uses the ENERGY int2e get_veff (cintx-ready as of 05-08), identical in shape to the MP2 fvind — so the orbital-relaxation Z-vector solve runs end-to-end un-gated; only the gradient de ASSEMBLY (int2e_ip1 + int1e_ip*) rides the cintx grad-intor gate."
  - "ECPscalar_ipnuc numeric UN-GATES now (cintx-ready, 07-01): the ecp_ipnuc structural test runs the REAL cintx ECP-gradient path on Cu/LANL2DZ and asserts finite, non-zero, component-leading values. ECPscalar_iprinv stays gated (MISSING — clean availability error, T-07-27)."
  - "A non-ECP molecule contributes an all-zero ECP-grad hcore term (get_hcore_ecp maps EcpEngineNotAvailable → zero buffer), NOT an error — so the RHF get_hcore fold is a no-op for non-ECP molecules."

patterns-established:
  - "CcsdGradReference shape (mo_coeff/mo_energy/mo_occ + the converged Amplitudes + the ChemistsEris + mol): the pyo3-free converged-RHF+CCSD snapshot the PyO3 bridge (07-09) builds from a CcsdReference + CcsdResult + the eris"
  - "CCSD-grad reuses the grad-local as_rhf() projection (RhfReference) onto the RHF base decomposition the relaxed density plugs into — verbatim the MP2 07-07 pattern"

requirements-completed: [GRAD-06, GRAD-07]

# Metrics
duration: 8min
completed: 2026-05-26
---

# Phase 7 Plan 08: CCSD Gradient (GRAD-06) + ECP Gradient (GRAD-07) Summary

**Completed the gradient method surface (D-09 order — CCSD then ECP last). CCSD-grad (`CcsdGradients`, GRAD-06, the SECOND non-variational gradient after MP2) CONSUMES the Phase-6 surface DIRECTLY — `pyscf_ccsd::solve_lambda` for the converged Λ and `pyscf_ccsd::make_rdm1` for the relaxed MO 1-RDM (NO Λ re-derivation in pyscf-grad, D-04/T-07-25; a source-scan test forbids a second lambda solver) — and re-enters the ONE 07-07 `cphf::solve` for the orbital-relaxation Z-vector with its own `fvind` (the ENERGY `int2e` `get_veff`, cintx-ready) + RHS, leaving `max_cycle` at the upstream default 50 (NOT MP2's 30). ECP-grad (GRAD-07) wires the `get_hcore` `+ ECPscalar_ipnuc` term (cintx-READY per 07-01 — numeric UN-GATES, real `[3,nao,nao]` on an ECP molecule, all-zero on a non-ECP molecule, never a panic) through the Phase-2 `CintxEcpEngine`, normalising the engine's component-inner buffer into the RHF component-leading F-order so it folds into `get_hcore` (07-03); the `hcore_deriv` `+ ECPscalar_iprinv` per-atom term (MISSING from cintx) routes to a CLEAN cintx-availability error (never a panic/silent-zero/`NotYetImplemented{phase:7}`, T-07-27). Both are FD-gated (D-01); the CCSD numeric + the end-to-end ECP numeric stay `#[ignore]`'d on the six missing cintx grad-intor families per the 07-01/07-03/07-07 precedent. This closes the GTO-05 arc (Phase 2 wired ECP eval; Phase 7 wires ECP gradient).**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-05-26T04:26Z (post context-load)
- **Completed:** 2026-05-26T04:34Z
- **Tasks:** 2 (both `type="auto" tdd="true"`)
- **Files:** 5 (2 created, 3 modified)

## The CCSD-grad API (recorded for 07-09 PyO3 bridge + 07-10 oracle close-out)

```rust
pub struct CcsdGradReference {
    pub mo_coeff: MOCoefficients, pub mo_energy: Vec<f64>, pub mo_occ: Vec<f64>,
    pub amps: Amplitudes,        // the converged CCSD (t1, t2)
    pub eris: ChemistsEris,      // the Phase-6 MO ERIs solve_lambda/make_rdm1 consume
    pub mol: Mole,
}
pub struct CcsdGradients { pub reference: CcsdGradReference, pub atmlst: Option<Vec<usize>>, pub de: Option<Vec<[f64;3]>> }
impl CcsdGradients { pub fn new(reference) -> Self; pub fn with_atmlst(self, Vec<usize>) -> Self; }
impl Gradients for CcsdGradients { /* make_rdm1e/get_ovlp/grad_elec; grad_nuc + kernel inherited */ }

// Free-fn forms (PyO3-bridge reusable, no trait):
pub fn relaxed_rdm1(refr: &CcsdGradReference) -> Result<Vec<f64>, PyscfRsError>;   // runs solve_lambda → make_rdm1(ao_repr=false); MO-basis (nmo,nmo)
pub fn response_dm1(refr: &CcsdGradReference, xvo: &[f64]) -> Result<Vec<f64>, PyscfRsError>; // Z-vector via cphf::solve; (nmo,nmo)
pub fn grad_elec(refr: &CcsdGradReference, atmlst: Option<&[usize]>) -> Result<Vec<[f64;3]>, PyscfRsError>;
pub const CCSD_CPHF_MAX_CYCLE: usize = 50;   // == cphf::DEFAULT_MAX_CYCLE (NOT MP2's 30)
```

### The CCSD Z-vector wiring (07-09 builds the reference; the de assembly un-gates with cintx)

```text
(l1,l2) = pyscf_ccsd::solve_lambda(t1, t2, eris, pool)              # the Phase-6 Λ — CONSUMED, not re-derived
dm1mo   = pyscf_ccsd::make_rdm1(t1,t2,l1,l2,eris, ao_repr=false, C) # the relaxed MO 1-RDM
Xvo     = build_xvo_base(refr, dm1mo)                              # = Cvᵀ·(2·get_veff(C·dm1mo·Cᵀ))·Co  (int2e — cintx-ready)
dvo     = cphf::solve(ccsd_fvind, mo_energy, mo_occ, Xvo, None, 50, 1e-9, false, 0.0)  # ← the ONE solver, max_cycle=50
dm1mo  += response_dm1(dvo)                                        # dm1[vir,occ]=dvo ; dm1[occ,vir]=dvoᵀ
de      = rhf::grad_elec(as_rhf(refr), atmlst)                     # ← cintx-gated (reaches get_ovlp first)
```

## The ECP-grad API (recorded for 07-09 — the gate structure the bridge wires)

```rust
// get_hcore '+ ECPscalar_ipnuc' (cintx-READY, numeric UN-GATED). Real [3,nao,nao]
// for an ECP molecule; ALL-ZERO for a non-ECP molecule (never a panic).
pub fn get_hcore_ecp(mol: &Mole) -> Result<Vec<f64>, PyscfRsError>;   // component-leading [3,nao,nao] F-order

// hcore_deriv '+ ECPscalar_iprinv' per-atom term (MISSING — clean availability error, T-07-27).
pub fn hcore_deriv_ecp(mol: &Mole, atm_id: usize) -> Result<Vec<f64>, PyscfRsError>;  // Err(clean cintx-availability)
```

## Accomplishments

- **`CcsdGradients` + `CcsdGradReference` (GRAD-06)** — the CCSD relaxed-density gradient driver, the SECOND non-variational grad (D-04). `grad_elec` consumes the Phase-6 Λ + relaxed 1-RDM via `relaxed_rdm1` (which runs `solve_lambda` then `make_rdm1(ao_repr=false)` — CONSUMED, never re-derived), forms the orbital-relaxation Z-vector RHS `Xvo`, solves the response through the ONE `cphf::solve` at `max_cycle=50`, and assembles `de` on the RHF base decomposition. `make_rdm1e` reuses the RHF energy-weighted RDM.
- **`ccsd::relaxed_rdm1` + `response_dm1` + `ccsd_fvind` + `build_xvo_base`** — the consumption + Z-vector machinery. `ccsd_fvind` uses the ENERGY `int2e` `get_veff` (cintx-ready, identical in shape to `mp2_fvind`), so the Z-vector solve runs end-to-end un-gated; the structural test confirms the real solve returns a transposed (vir,occ)/(occ,vir) MO response density.
- **`ecp::get_hcore_ecp` (GRAD-07, cintx-READY)** — the `get_hcore` `+ ECPscalar_ipnuc` term. Dispatches `CintxEcpEngine::ecp_int1e_ipnuc(mol, "ECPscalar_ipnuc")`, normalises the engine's component-inner buffer (`data[comp + p*3 + q*3*nao]`) to the RHF component-leading F-order (`out[comp*nao*nao + i + j*nao]`), and maps a non-ECP molecule's `EcpEngineNotAvailable` to an all-zero buffer. The numeric UN-GATES: the structural test runs the REAL cintx path on Cu/LANL2DZ and asserts finite, non-zero, `[3,nao,nao]` values.
- **`ecp::hcore_deriv_ecp` (GRAD-07, cintx-GATED)** — the `hcore_deriv` `+ ECPscalar_iprinv` per-atom term. Routes through the engine (which is MISSING `iprinv`, 07-01) to a CLEAN cintx-availability error — never a panic, never a silent zero, never `NotYetImplemented{phase:7}` (T-07-27).
- **`tests/ccsd_verify_fd.rs` (6 always-on + 1 `#[ignore]`'d)** — `make_rdm1e` shape (`dme0[0,0] = ε0·occ0 = −1.0`), `ccsd_lambda_is_consumed_from_phase6_not_rederived` (drives `relaxed_rdm1` → real Λ solve + RDM), `ccsd_zvector_routes_through_the_one_cphf_solver` (response_dm1 transpose-block check), `kernel()` (natm,3)-or-clean-error, atmlst subset, the `single_lambda_solver_in_grad` GRAD-10/T-07-25 source-scan, plus the `#[ignore]`'d numeric FD arm.
- **`tests/ecp_verify_fd.rs` (3 always-on + 1 `#[ignore]`'d)** — `ecp_ipnuc_term_is_component_leading_for_an_ecp_molecule` (REAL cintx ipnuc on Cu/LANL2DZ, finite + non-zero + [3,nao,nao]), `ecp_ipnuc_term_is_zero_contribution_for_a_non_ecp_molecule` (He → all-zero), `ecp_iprinv_per_atom_term_routes_to_the_gated_arm` (clean availability error), plus the `#[ignore]`'d end-to-end numeric arm.
- **Gates green:** `cargo test -p pyscf-grad --locked -- --test-threads=1` → 51 passed / 6 ignored / 0 failed (the 6 cintx-gated numeric arms across rhf/uhf/rks/uks/mp2/ccsd + the ECP end-to-end). `cargo clippy -p pyscf-grad --tests --locked` clean. `check-dependency-wall` PASS (no `cubecl-*` in pyscf-grad, T-07-SC).

## Numeric-gating decision (the plan's required record — which arms un-gate vs stay cintx-gated)

| Arm | Status | Reason |
|-----|--------|--------|
| CCSD Λ consumption (`solve_lambda` + `make_rdm1`) | **always-on** | Pure post-SCF linear algebra (Phase-6 surface); no grad-intor dependency. |
| CCSD orbital-relaxation Z-vector (`cphf::solve` + `ccsd_fvind` + `build_xvo_base`) | **always-on** | `ccsd_fvind` rides the ENERGY `int2e` `get_veff` (cintx-ready, 05-08). |
| CCSD numeric `de` assembly (FD-vs-analytical) | **`#[ignore]`'d** | Reaches the RHF `get_ovlp` (`int1e_ipovlp` — MISSING) first; `kernel()` `?`-propagates a clean availability error. Un-gates with the six cintx grad-intor families. |
| ECP `get_hcore` ipnuc term (`ECPscalar_ipnuc`) | **un-gated** | cintx-READY (07-01); the structural test runs the real cintx path on Cu/LANL2DZ. |
| ECP `hcore_deriv` iprinv term (`ECPscalar_iprinv`) | **gated (clean error)** | MISSING from every cintx branch (07-01); routes to a clean availability error. |
| ECP end-to-end numeric (FD-vs-analytical) | **`#[ignore]`'d** | The full ECP gradient FD still needs the RHF base `de` assembly (the six missing families) + `iprinv`. |

**The 07-09 bridge must keep the CCSD numeric + the ECP end-to-end numeric `#[ignore]`'d** and pair any "drop the `#[ignore]`" with a cintx-side availability note confirming `int2e_ip1` + `int1e_ip{ovlp,kin,nuc,rinv}` (+ `ECPscalar_iprinv` for the ECP end-to-end) shipped. The CCSD Λ consumption + Z-vector solve and the **ECP ipnuc term** are NOT gated and need no such pairing.

## Task Commits

1. **Task 1: CCSD gradient — Phase-6 Λ + RDMs + orbital-relaxation Z-vector (GRAD-06)** — `4f61a2d` (feat)
2. **Task 2: ECP gradient hcore term — ECPscalar_ipnuc ready + iprinv gated (GRAD-07)** — `d98541c` (feat)

## Files Created/Modified

- `crates/pyscf-grad/src/ccsd.rs` — replaced the `NotYetImplemented{wave:7}` stub with `CcsdGradients`/`CcsdGradReference` + `relaxed_rdm1`/`response_dm1`/`ccsd_fvind`/`build_xvo_base`/`grad_elec` + `CCSD_CPHF_MAX_CYCLE` (the Phase-6-consuming relaxed-density grad).
- `crates/pyscf-grad/src/ecp.rs` — replaced the `NotYetImplemented{wave:8}` stub with `get_hcore_ecp` (the cintx-ready ipnuc `get_hcore` term + layout normalisation + the non-ECP zero-contribution map) + `hcore_deriv_ecp` (the cintx-gated iprinv `hcore_deriv` term).
- `crates/pyscf-grad/src/lib.rs` — flat re-exports for `CcsdGradients`/`CcsdGradReference`/`CCSD_CPHF_MAX_CYCLE` + `get_hcore_ecp`/`hcore_deriv_ecp`; refreshed the module-status doc (ccsd/ecp landed in 07-08).
- `crates/pyscf-grad/tests/ccsd_verify_fd.rs` — NEW: 6 always-on CCSD tests incl. the `single_lambda_solver_in_grad` GRAD-10/T-07-25 source-scan + 1 `#[ignore]`'d numeric FD arm.
- `crates/pyscf-grad/tests/ecp_verify_fd.rs` — NEW: 3 always-on ECP tests (ipnuc real / ipnuc zero / iprinv gated) + 1 `#[ignore]`'d end-to-end numeric arm.

## Decisions Made

- **CCSD-grad CONSUMES `solve_lambda` + `make_rdm1` directly (D-04/GRAD-06/T-07-25).** There is NO second lambda-equation solver in `pyscf-grad`; the `single_lambda_solver_in_grad` test source-scans `src/` for any `fn ...lambda...` declaration and asserts NONE. The body only CALLS `pyscf_ccsd::solve_lambda` (a path-prefixed call, never a declaration).
- **CCSD leaves `max_cycle` at the upstream default 50** (`CCSD_CPHF_MAX_CYCLE = cphf::DEFAULT_MAX_CYCLE`), distinct from MP2's 30 override — `pyscf/grad/ccsd.py` does not override the `cphf.solve` default.
- **The CCSD Z-vector `fvind` uses the ENERGY `int2e` `get_veff` (cintx-ready), identical in shape to `mp2_fvind`** — so the orbital-relaxation Z-vector runs end-to-end un-gated; only the gradient `de` assembly is cintx-gated.
- **ECP ipnuc numeric UN-GATES now (cintx-ready, 07-01)** — the structural test runs the REAL cintx ECP-gradient path on Cu/LANL2DZ. ECP iprinv stays gated (clean availability error, T-07-27).
- **A non-ECP molecule contributes an all-zero ECP-grad hcore term** (`EcpEngineNotAvailable` → zero buffer), NOT an error — so the RHF `get_hcore` fold is a no-op for non-ECP molecules.
- **Reductions go through `oracle_sum`/`oracle_dot`, NOT `pyscf_algebra::gemm`** (still a Phase-2 `NotYetImplemented` stub) — the established RHF/MP2/CCSD precedent.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] clippy `doc_list_item` on three ccsd.rs response_dm1 doc lines**
- **Found during:** Task 1 (clippy gate)
- **Issue:** A `+ RHS` at the start of a doc continuation line made clippy parse `(vir,occ)`/`(occ,vir)` continuations as un-indented doc list items.
- **Fix:** Reworded `+ RHS` → `and RHS` and code-quoted `(vir,occ)`/`(occ,vir)` so no line starts with a list-marker character.
- **Files modified:** `crates/pyscf-grad/src/ccsd.rs`
- **Verification:** `cargo clippy -p pyscf-grad --tests --locked` clean.
- **Committed in:** `4f61a2d` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 3 — clippy-gate blocking fix, mechanical, no behavior change). **Impact on plan:** none — the CCSD + ECP gradients land exactly as specified.

## Known Stubs

None that block the plan's goal. The `ccsd::default_grad_elec` / `ecp::default_grad_ecp` free fns (07-02 seams) are retained as thin error-returning wrappers for any caller lacking a reference/molecule; the real bodies are the reference-consuming forms (the PyO3 bridge 07-09 always supplies them). The CCSD numeric FD arm + the ECP end-to-end numeric FD arm are `#[ignore]`'d (cintx-gated, documented above) — intentional, un-gate with the cintx grad-intor workstream. The ECP ipnuc term itself is NOT a stub — it runs the real cintx path.

## Issues Encountered

- The cintx `ecp_int1e_ipnuc` Density carries a component-INNER layout (`data[comp + p*3 + q*3*nao]`), distinct from the RHF path's component-leading F-order (`comp*nao*nao + i + j*nao`). `get_hcore_ecp` normalises it so the term folds into the RHF `get_hcore` path. No upstream change needed; recorded as a pattern.
- `pyscf_algebra::gemm`/`gemv` remain Phase-2 `NotYetImplemented` stubs — all contractions route through `oracle_sum`/`oracle_dot` (the working algebra-wall primitives). No code change; consistent with the 07-03/07-07 precedent.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- **07-09 (PyO3 bridge):** snapshot the converged `mycc.mo_*` + `mycc.mol` + the CCSD `(t1,t2)` amplitudes + the `ChemistsEris` into `CcsdGradReference`; wire `mycc.nuc_grad_method()` → `CcsdGradients`; expose `kernel`. For ECP molecules, fold `get_hcore_ecp(mol)` into the RHF `get_hcore` term and route `hcore_deriv_ecp` per atom. The free-fn forms (`relaxed_rdm1`, `response_dm1`, `grad_elec`, `get_hcore_ecp`, `hcore_deriv_ecp`) are bridge-reusable without the trait.
- **07-10 (oracle close-out):** the CCSD numeric + the ECP end-to-end numeric arms are `#[ignore]`'d on the six missing cintx grad-intor families (+ `ECPscalar_iprinv` for ECP end-to-end). The ECP **ipnuc** term un-gates now (cintx-ready). The upstream byte-identity arms wire the same gate structure.
- **Coordination note (D-02 hinge):** any "drop the `#[ignore]`" on the CCSD numeric / ECP end-to-end MUST be paired with a cintx-side availability note confirming the families shipped. The CCSD Λ consumption + Z-vector solve and the ECP ipnuc term are NOT gated and need no such pairing.

## Self-Check: PASSED

- `crates/pyscf-grad/src/ccsd.rs` exists (contains `solve_lambda` consumption ×10 + `cphf::solve` + `CCSD_CPHF_MAX_CYCLE`).
- `crates/pyscf-grad/src/ecp.rs` exists (contains `ECPscalar_ipnuc` + `ECPscalar_iprinv` references ×14).
- `crates/pyscf-grad/tests/ccsd_verify_fd.rs` exists (calls `verify_fd`; `single_lambda_solver_in_grad` source-scan passes).
- `crates/pyscf-grad/tests/ecp_verify_fd.rs` exists (calls `verify_fd`).
- Both task commits (`4f61a2d`, `d98541c`) present in git history.
- `cargo test -p pyscf-grad --locked -- --test-threads=1`: 51 passed, 6 ignored, 0 failed.
- `cargo clippy -p pyscf-grad --tests --locked`: clean. `check-dependency-wall`: PASS.

---
*Phase: 07-gradients-geomopt*
*Completed: 2026-05-26*
