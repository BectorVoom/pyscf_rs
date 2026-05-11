---
status: partial
phase: 03-scf-pyo3-bindings
source: [03-VERIFICATION.md]
started: 2026-05-12T00:00:00Z
updated: 2026-05-12T00:00:00Z
---

## Current Test

[awaiting human testing — see Gaps for code-level gaps that block these tests]

## Tests

### 1. µHartree numeric parity vs upstream PySCF
expected: `scf.RHF(mol).kernel()` on H2O/cc-pVDZ + benzene/6-31G* converges to upstream PySCF total energy to ≤1 µHartree under `release-oracle`; UHF matches for open-shell systems; GHF runs to completion.
why_human: Requires maturin + upstream pyscf install. Numerical claim cannot be verified at the µHartree level by inspection of the Rust source alone; it depends on FMA-free codegen + actual ERIs + cross-platform LAPACK + sign canonicalization composing correctly at runtime.
result: [pending — also blocked by int2e_sph + int3c2e_sph gaps; see Gaps]

### 2. python3.13t free-threading parity
expected: `maturin develop --no-default-features --features free-threading` builds; `import pyscf._native` succeeds under python3.13t (no-GIL); BIND-05 GIL-release seam works in free-threaded mode.
why_human: Requires the python3.13t interpreter (deadsnakes PPA or uv-installed); not available in this verification environment. The GIL-release seam can only be exercised under the experimental free-threaded build.
result: [pending — CI job `python313t-smoke` wired in .github/workflows/ci.yml]

### 3. BIND-04 NumPy stride contiguity policy
expected: `python/pyscf/tests/test_scf_stride_fuzz.py` passes — non-contiguous NumPy arrays handed to PyO3 boundary get is_c_contiguous-checked + .to_owned()-copied when needed.
why_human: BIND-04 numpy contiguity policy must be exercised through actual PyO3 invocation; the test body exists but cannot run without maturin. The stride-fuzz CI job is wired to do this in plan 03-09's xplat-uhartree job graph but couldn't run in this verifier environment.
result: [pending — CI job `stride-fuzz` wired]

### 4. BIND-07 end-to-end subclass dispatch
expected: Python subclass of `PyRHF` that overrides any of the 11 hooks has its override invoked via `slf.call_method1` (verified by side-effect assertion).
why_human: End-to-end subclass dispatch round-trip requires the wheel, an installed upstream pyscf (for the H2O fixture's Mole.dumps() round-trip on the Python side), AND a working RHF.kernel() (which today is blocked by the int2e_sph gap).
result: [pending — also blocked by get_init_guess gap; see Gaps]

### 5. BIND-09 panic → Python exception runtime chain
expected: A panic inside `pyscf._native` surfaces in Python as `PyscfRsRuntimeError` with `.kind` and `.source_chain` attributes; chained source preserved.
why_human: Requires maturin develop; cannot be exercised through Rust unit tests alone because the create_exception!-generated PyscfRsRuntimeError + the Python overlay's PyscfRsError subclass interaction is a runtime contract.
result: [pending — CI job `maturin-smoke` wired]

## Summary

total: 5
passed: 0
issues: 0
pending: 5
skipped: 0
blocked: 0

## Gaps

Code-level gaps identified by verifier that block the µHartree numeric claim (these are NOT runtime/UAT issues — they are missing code that needs a follow-up plan):

- gap: int3c2e_sph returns zero-filled buffer (cintx-ops upstream gap)
  status: failed
  artifact: crates/pyscf-gto/src/intor.rs:459-476
  reason: Documented but means cholesky_eri produces all-zero B-buffer; DF-HF kernel converges to (1e + nuc) only, not a real DF-HF energy.
  next: cintx-ops base operator id OR alternate integral path for DF-HF
- gap: PyOverrideBridge::get_init_guess short-circuits to NoOverrides instead of slf.call_method1
  status: failed
  artifact: crates/pyscf-py/src/bridge.rs:93-109
  reason: BIND-07 / ROADMAP §SC4 names get_init_guess in dispatch list. REVIEW.md WR-01 includes a verbatim fix patch.
  next: 20-line patch replacing NoOverrides delegate with Python::attach + call_method1
- gap: init_guess minao/atom/huckel modes return InitGuessNotYetImplemented
  status: failed
  artifact: crates/pyscf-scf/src/init_guess.rs:14-21
  reason: SCF-05 explicitly enumerates all 5 modes. 3 of 5 are stubs. test_scf_init_guess.py xfails them.
  next: Port upstream pyscf init_guess_by_{minao,atom,huckel} bodies (~100-150 lines)
- gap: PyRHF::kernel does not auto-write chkfile + no PyRHF.from_chk pymethod
  status: failed
  artifact: crates/pyscf-py/src/scf.rs:230-275
  reason: SCF-10 promises "mf.chkfile = path writes HDF5 file on convergence". Rust primitives shipped but PyO3 wiring missing.
  next: Add auto-write check at end of PyRHF::kernel + #[pymethods] from_chk(path) constructor
- gap: mulliken_meta returns NotYetImplemented{phase:3}
  status: failed
  artifact: crates/pyscf-scf/src/analyze.rs
  reason: SCF-09 / ROADMAP §SC6 names mulliken_meta explicitly. mulliken_pop + dip_moment + analyze ship as real bodies; mulliken_meta is the only NYI in this group.
  next: Port upstream meta-Löwdin population analysis
