---
phase: 07-gradients-geomopt
plan: 03
subsystem: gradients
tags: [gradients, rhf, grad_elec, make_rdm1e, hcore_deriv, get_veff, verify_fd, GRAD-01, oracle_sum, cintx-gated]

# Dependency graph
requires:
  - phase: 07-gradients-geomopt
    plan: 01
    provides: "cintx grad-intor availability split (int2e_ip1 + int1e_ip* MISSING → clean cintx-availability error, never NotYetImplemented{phase:7})"
  - phase: 07-gradients-geomopt
    plan: 02
    provides: "base Gradients trait (grad_elec/make_rdm1e/grad_nuc/get_ovlp/kernel + resolve_atmlst); verify_fd FD harness; GradScanner seam"
  - phase: 03-scf
    provides: "RHF + as_scanner energy closure (SCF-12); make_rdm1 (C·diag(occ)·Cᵀ, oracle_sum) shape; MOCoefficients F-order layout"
  - phase: 02-gto-integrals
    provides: "intor dispatcher (int1e_ip*/int2e_ip1 component-leading); ao_loc_nr + _bas[ATOM_OF]; M/MoleBuildArgs"
provides:
  - "RhfGradients (base Gradients impl): grad_elec (Hellmann-Feynman + 2e Pulay + overlap Pulay), make_rdm1e (energy-weighted RDM), get_ovlp; grad_nuc inherited from trait default"
  - "RhfReference snapshot (mo_coeff/mo_energy/mo_occ/mol) — the pyo3-free converged-SCF reference the gradient + PyO3 bridge (07-09) consume"
  - "hcore_deriv (per-atom rinv shift port), get_hcore, get_veff (int2e_ip1 J/K → vj-0.5vk), aoslice_by_atom (ATOM_OF + ao_loc_nr walk) — all component-leading [3,nao,nao]"
  - "rhf_verify_fd gate: always-on STRUCTURAL arm (5 tests) + #[ignore]'d NUMERIC arm (verify_fd disp=1e-4 tol=1e-6) un-gating on the cintx grad-intor workstream"
affects: [07-04-geomopt, 07-05-uhf-rks-uks-grad, 07-06-mp2-grad, 07-07-ccsd-grad, 07-08-df-grad, 07-09-pyo3-bridge]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Reference-snapshot gradient driver: RhfGradients { reference: RhfReference, atmlst, de } implements the base Gradients trait; grad_elec/make_rdm1e are the per-method bodies, grad_nuc inherited"
    - "Structural-lands-regardless / numeric-#[ignore]'d split (D-02): the grad_elec assembly + intor wiring land always-on; the missing cintx families ?-propagate clean availability errors; the FD-vs-analytical numeric arm is #[ignore]'d until cintx ships them"
    - "assert_component_leading: every gradient intor guarded as [3,nao,nao] (axis 0 = x/y/z) before contraction (Pitfall 4)"
    - "Materialise-then-oracle_sum/oracle_dot for every einsum reduction; no bare += in any accumulation (Pitfall 1/2)"

key-files:
  created:
    - crates/pyscf-grad/tests/rhf_verify_fd.rs
  modified:
    - crates/pyscf-grad/src/rhf.rs

key-decisions:
  - "rhf_verify_fd NUMERIC arm is #[ignore]'d (cintx workstream pending): int2e_ip1 + int1e_ip{ovlp,kin,nuc,rinv} + with_rinv_at_nucleus are MISSING from cintx (07-01-SUMMARY, no scheduled workstream) so RhfGradients::kernel() cannot produce a numeric gradient yet; the STRUCTURAL arm + the wiring are complete and un-gate by dropping the #[ignore]"
  - "lib.rs left UNCHANGED: the 07-02 base Gradients trait already exposed grad_elec/make_rdm1e/get_ovlp as the required contract, so RhfGradients implements them with no trait edit (fewer changes than the plan's files_modified anticipated)"
  - "All matmuls/einsums route through oracle_sum/oracle_dot, NOT pyscf_algebra::gemm — gemm is still a Phase-2 NotYetImplemented stub; the einsum('xij,ij->x') contractions are oracle_dot/oracle_sum over flattened [ij] slices per component (the make_rdm1/rmp2_kernel precedent)"
  - "D-04 honored: RhfGradients makes NO CPHF call (RHF energy gradients are stationary, the 2n+1 rule)"

patterns-established:
  - "RhfReference is the per-method converged-SCF snapshot shape every variational grad (UHF/RKS/UKS, 07-05) and the geomopt scanner (07-04) reuse"
  - "is_clean_cintx_availability_error test helper: asserts a missing-family error is Core(InvalidMolecule(..)) and NOT NotYetImplemented{phase:7} (the D-02 contract every downstream grad test reuses)"

requirements-completed: [GRAD-01]

# Metrics
duration: 24min
completed: 2026-05-26
---

# Phase 7 Plan 03: RHF Analytical Gradient Body (GRAD-01) Summary

**Ported the RHF analytical gradient — the phase headline — into `crates/pyscf-grad/src/rhf.rs`: `RhfGradients` implements the base `Gradients` trait with `grad_elec` (Hellmann-Feynman + 2e Pulay + overlap Pulay), `make_rdm1e` (energy-weighted RDM), `hcore_deriv`/`get_hcore`/`get_veff`/`get_ovlp` (component-leading `[3,nao,nao]` intor wiring), and `aoslice_by_atom`; every reduction is oracle-ordered, no CPHF call (D-04), and the six missing cintx grad-intor families `?`-route to a clean cintx-availability error. The `rhf_verify_fd` gate lands its always-on STRUCTURAL arm (5 tests) and an `#[ignore]`'d NUMERIC arm that un-gates on the cintx grad-integral workstream.**

## Performance

- **Duration:** ~24 min
- **Started:** 2026-05-26T02:44Z (post context-load)
- **Completed:** 2026-05-26T03:08Z
- **Tasks:** 2 (both `type="auto"`; Task 1 `tdd="true"` — body + structural assertions)
- **Files modified:** 2 (1 created, 1 modified)

## RhfGradients API (recorded for 07-04 / 07-05 / 07-09)

```rust
pub struct RhfReference { pub mo_coeff: MOCoefficients, pub mo_energy: Vec<f64>, pub mo_occ: Vec<f64>, pub mol: Mole }

pub struct RhfGradients { pub reference: RhfReference, pub atmlst: Option<Vec<usize>>, pub de: Option<Vec<[f64; 3]>> }
impl RhfGradients {
    pub fn new(reference: RhfReference) -> Self;
    pub fn with_atmlst(self, atmlst: Vec<usize>) -> Self;
}
impl Gradients for RhfGradients { /* mol/atmlst/de/unit + make_rdm1e/get_ovlp/grad_elec; grad_nuc + kernel inherited */ }

// Free-fn forms (PyO3-bridge reusable, no trait):
pub fn make_rdm1e(refr: &RhfReference) -> Result<Vec<f64>, PyscfRsError>;      // energy-weighted RDM, row-major (nao,nao)
pub fn grad_elec(refr: &RhfReference, atmlst: Option<&[usize]>) -> Result<Vec<[f64;3]>, PyscfRsError>;
pub fn get_ovlp(mol: &Mole) -> Result<Vec<f64>, PyscfRsError>;                  // s1 = -int1e_ipovlp, [3,nao,nao]
pub fn aoslice_by_atom(mol: &Mole) -> Result<Vec<(usize, usize)>, PyscfRsError>; // per-atom (p0,p1) AO range
```

## The grad_elec decomposition (port of `pyscf/grad/rhf.py:59-76`, for 07-05/07-06)

```text
dm0  = make_rdm1(refr)                       # Σ_i occ_i C[μ,i] C[ν,i], row-major (nao,nao)
dme0 = make_rdm1e(refr)                       # Σ_{i:occ>0} (ε_i·occ_i) C[μ,i] C[ν,i]
s1   = -int1e_ipovlp                           # get_ovlp, [3,nao,nao] component-leading
h1   = -(int1e_ipkin + int1e_ipnuc)            # get_hcore, [3,nao,nao]
vhf  = get_veff(mol, dm0)                      # int2e_ip1 J/K → vj - 0.5·vk, [3,nao,nao]
for ia in atmlst:                              # p0,p1 = aoslices[ia]
    h1ao = hcore_deriv(ia)                     # -Z·int1e_iprinv (rinv@nucleus) + h1 block, symmetrised
    de[k][x] = einsum('ij,ij->', h1ao[x], dm0)             # Hellmann-Feynman (full AO)
             + 2·einsum('ij,ij->', vhf[x][p0:p1], dm0[p0:p1])   # 2e Pulay (∇-bra rows)
             - 2·einsum('ij,ij->', s1 [x][p0:p1], dme0[p0:p1])  # overlap Pulay
# grad_nuc(atmlst) added by Gradients::kernel (the shared trait default)
```
Every `einsum` materialises its product terms into a `Vec` then `oracle_sum`s them (the per-atom `de[k][x]` is a single ordered 3-element sum `[term1, term2, -term3]`); no bare `+=`.

## Accomplishments

- **`RhfGradients` + `RhfReference`** — the reference-snapshot gradient driver implementing the base `Gradients` trait (`grad_elec`/`make_rdm1e`/`get_ovlp` per-method; `grad_nuc`/`kernel` inherited). The `RhfReference` shape (mo_coeff/mo_energy/mo_occ/mol) is the pyo3-free converged-SCF snapshot every downstream variational grad reuses.
- **`make_rdm1e`** (energy-weighted RDM, `rhf.py:185-189`) — thin pure-linear-algebra port; `dme0[μν] = Σ_{i:occ>0}(ε_i·occ_i)·C[μ,i]·C[ν,i]`, oracle-ordered over MOs. Verified: the H2/STO-3G identity reference gives `dme0[0,0] = ε0·occ0 = -1.0`.
- **`get_ovlp` / `get_hcore` / `hcore_deriv` / `get_veff`** — the component-leading `[3,nao,nao]` intor wiring: `s1=-int1e_ipovlp`; `h1=-(int1e_ipkin+int1e_ipnuc)`; per-atom `vrinv=-Z·int1e_iprinv` + h1-block + AO-axis symmetrisation; `get_jk` over `int2e_ip1` → `vj-0.5·vk`. Each guarded by `assert_component_leading` (Pitfall 4) and `?`-propagating a clean cintx-availability error for the missing families.
- **`aoslice_by_atom`** — clean port of the `pyscf-scf` `_bas[ATOM_OF]` + `ao_loc_nr` walk; H2/STO-3G partitions into `[0,1)`,`[1,2)` covering nao.
- **`rhf_verify_fd` gate** — STRUCTURAL arm always-on (5 tests: make_rdm1e shape, grad_nuc Newton's-3rd-law z-force, aoslice partition, `kernel()` returns (natm,3)-or-clean-error, `grad_elec` clean-error routing) + `#[ignore]`'d NUMERIC arm wiring `verify_fd(disp=1e-4, tol=1e-6)` against the SCF `as_scanner` energy.
- **Gates green:** `cargo test -p pyscf-grad --locked -- --test-threads=1` → 14 passed / 1 ignored / 0 failed; clippy clean; `check-dependency-wall` PASS (no `cubecl-*` in pyscf-grad, T-07-SC).

## rhf_verify_fd gating decision (the plan's required record)

**The NUMERIC arm is `#[ignore]`'d (cintx workstream PENDING).** Per 07-01-SUMMARY the six families the RHF analytical gradient contracts — `int2e_ip1` (2e Pulay), `int1e_ip{ovlp,kin,nuc}` (overlap + hcore Pulay), and `int1e_iprinv` + the `with_rinv_at_nucleus` origin shift (per-atom Hellmann-Feynman) — are MISSING from every cintx branch with **no scheduled workstream**. So `RhfGradients::kernel()` cannot produce a numeric gradient today: it `?`-propagates a clean `Core(InvalidMolecule(..))` cintx-availability error (never `NotYetImplemented{phase:7}`, the disposition GRAD-07 closed) the moment `grad_elec` reaches `get_ovlp`. The STRUCTURAL arm (always-on) proves the assembly shapes + the clean-error contract; the NUMERIC arm (`#[ignore]`'d, full `verify_fd` wiring complete) un-gates by simply dropping the `#[ignore]` once a future cintx grad-integral workstream lands the six families. The always-on FD-STRUCTURAL gate (D-01) proceeds regardless.

## Task Commits

1. **Task 1: Port RHF grad_elec + hcore_deriv + make_rdm1e + get_veff + get_ovlp + aoslice_by_atom** — `e774f8a` (feat)
2. **Task 2: Wire rhf_verify_fd (STRUCTURAL always-on + NUMERIC #[ignore]'d) against the 07-02 FD harness** — `3a7de15` (test)

## Files Created/Modified

- `crates/pyscf-grad/src/rhf.rs` — replaced the 07-02 `NotYetImplemented{wave:3}` stub with the full `RhfGradients`/`RhfReference` body: `grad_elec`, `make_rdm1`, `make_rdm1e`, `get_ovlp`, `get_hcore`, `hcore_deriv`, `get_veff`, `aoslice_by_atom`, `assert_component_leading` (529 lines).
- `crates/pyscf-grad/tests/rhf_verify_fd.rs` — NEW: 5 always-on structural tests + 1 `#[ignore]`'d numeric FD gate (302 lines).

## Decisions Made

- **NUMERIC arm `#[ignore]`'d, STRUCTURAL arm always-on (D-02).** See the gating record above — the canonical 07-01 cintx-availability split.
- **`lib.rs` left unchanged.** The 07-02 base `Gradients` trait already exposed `grad_elec`/`make_rdm1e`/`get_ovlp` as the required contract; `RhfGradients` implements them directly. The plan's `files_modified` anticipated a `lib.rs` edit, but the trait shape was sufficient — fewer changes is the correct outcome (no scope creep, no over-editing).
- **Reductions go through `oracle_sum`/`oracle_dot`, NOT `pyscf_algebra::gemm`.** `gemm` is still a Phase-2 `NotYetImplemented` stub; the gradient einsums `einsum('xij,ij->x', A, B)` are `oracle_dot`/`oracle_sum` over flattened `[ij]` slices per component — the established `make_rdm1`/`rmp2_kernel` precedent. This keeps the algebra-wall discipline (no bare `+=`, deterministic reduction order) without depending on the unwired dense-GEMM path.
- **D-04: no CPHF.** `RhfGradients` makes no CPHF call — RHF energy gradients are stationary.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added two `#[allow(clippy::needless_range_loop)]` in `make_rdm1`/`make_rdm1e`**
- **Found during:** Task 1 (clippy gate)
- **Issue:** Clippy flagged the MO-reduction loops where `i` indexes BOTH the per-MO `terms` buffer AND the F-order `mo_coeff.data[μ + i*nao]` offsets — `needless_range_loop` fires but the range loop is the clearest form (iterating a single slice can't drive two distinct strided indices).
- **Fix:** Added `#[allow(clippy::needless_range_loop)]` with a comment, matching the established `pyscf-scf/src/rdm.rs` precedent for the identical flat-array reduction pattern.
- **Files modified:** `crates/pyscf-grad/src/rhf.rs`
- **Verification:** `cargo clippy -p pyscf-grad --locked` clean.
- **Committed in:** `e774f8a` (Task 1 commit)

**2. [Rule 3 - Blocking] De-indented one doc-comment list item in the test file**
- **Found during:** Task 2 (clippy gate)
- **Issue:** Clippy `doc_overindented_list_items` flagged a 6-space-indented continuation line in the module doc-comment.
- **Fix:** Re-indented to 4 spaces.
- **Files modified:** `crates/pyscf-grad/tests/rhf_verify_fd.rs`
- **Verification:** `cargo clippy -p pyscf-grad --tests --locked` clean.
- **Committed in:** `3a7de15` (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 3 — clippy-gate blocking fixes, mechanical, no behavior change). **Impact on plan:** none — the RHF gradient body + the FD gate land exactly as specified; only lint-conformance attributes were added. The single planned-but-untouched file (`lib.rs`) is a documented decision, not a deviation (the trait already carried the contract).

## Known Stubs

None that block the plan's goal. The `default_grad_elec()` free fn (07-02 seam) is retained as a thin error-returning wrapper for any caller that lacks an `RhfReference`; the real body is `grad_elec(refr, atmlst)` + the `RhfGradients` trait impl. This is intentional (the PyO3 bridge 07-09 always supplies a reference) and documented in-source.

## Issues Encountered

- `pyscf_algebra::gemm` is a Phase-2 `NotYetImplemented` stub — confirmed at read time, so all contractions route through `oracle_sum`/`oracle_dot` (the working algebra-wall primitives), not `gemm`. No code change needed; recorded as a decision above.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- **07-04 (geomopt scanner):** `GradScanner` can wrap an `RhfGradients`-backed gradient closure + an SCF `as_scanner` energy closure; the `(natm,3)` `kernel` shape is fixed. (The scanner's gradient values stay cintx-gated until the grad-intor workstream lands.)
- **07-05 (UHF/RKS/UKS grad):** reuse the `RhfReference` snapshot shape + the `grad_elec` decomposition (spin-resolve the densities for UHF; add the XC-potential + Becke-weight terms for RKS/UKS); the oracle-ordered einsum routing + `assert_component_leading` discipline carry over verbatim.
- **07-06 / 07-07 (MP2/CCSD grad):** consume `RhfReference` + the relaxed/energy-weighted-RDM contractions; their Z-vector/Λ arms add the CPHF solver (07-09 / D-03) on top of this stationary base.
- **07-09 (PyO3 bridge):** snapshot `mf.mo_*` + `mf.mol` into `RhfReference`, wire `mf.nuc_grad_method()` → `RhfGradients`, expose `kernel`. The free-fn forms (`make_rdm1e`, `grad_elec`, `get_ovlp`, `aoslice_by_atom`) are bridge-reusable without the trait.
- **Coordination note (D-02 hinge):** the NUMERIC arm stays `#[ignore]`'d for the six missing families; any "drop the `#[ignore]`" MUST be paired with a cintx-side availability note confirming the families shipped.

## Self-Check: PASSED

- `crates/pyscf-grad/src/rhf.rs` exists (529 lines, contains `grad_elec`/`make_rdm1e`/`hcore_deriv`/`get_veff`).
- `crates/pyscf-grad/tests/rhf_verify_fd.rs` exists (302 lines, calls `verify_fd` with `DEFAULT_DISP`/`FD_TOL`).
- Both task commits (`e774f8a`, `3a7de15`) present in git history.
- `cargo test -p pyscf-grad --locked -- --test-threads=1`: 14 passed, 1 ignored, 0 failed.
- `cargo clippy -p pyscf-grad --tests --locked`: clean. `check-dependency-wall`: PASS.

---
*Phase: 07-gradients-geomopt*
*Completed: 2026-05-26*
