---
status: partial
phase: 03-scf-pyo3-bindings
source: [03-VERIFICATION.md]
started: 2026-05-12T00:00:00Z
updated: 2026-05-24T15:00:00Z
---

## Current Test

[awaiting human testing — all code-level gaps are now closed; these items require the Python toolchain (maturin + upstream pyscf + h5py) / CI runners, which are unavailable in the sandbox]

## Tests

### 1. µHartree numeric parity vs upstream PySCF on the test corpus (SCF-01/02/03)
expected: `maturin develop --profile release-oracle && pytest python/pyscf/tests/test_scf_rhf_h2o.py -x` exits 0 with H2O/cc-pVDZ RHF total energy matching upstream `pyscf.scf.RHF(mol).kernel()` to ≤1 µHartree; UHF matches for open-shell; GHF runs to completion. Full corpus (H2O, benzene/6-31G*, water-trimer) passes.
why_human: Requires maturin + upstream pyscf install. The µHartree claim depends on FMA-free codegen + actual ERIs + cross-platform LAPACK + sign canonicalization composing correctly at runtime — not verifiable by Rust-source inspection. CI job: `xplat-uhartree`.
result: [pending]

### 2. Cross-platform µHartree parity Linux x86_64 + macOS aarch64 (SCF-13, Pitfall 12)
expected: `xplat-uhartree` matrix job (ubuntu-latest + macos-14) exits 0 with maturin>=1.4 + `--profile release-oracle` (no `--release` fallback); total energies agree within 1 µHartree across platforms.
why_human: Requires a macOS aarch64 CI runner + maturin build. All infrastructure (canonicalize_signs, oracle_sum reductions, release-oracle profile, CI job wiring) is in place; the numerical assertion cannot execute without the build environment. CI job: `xplat-uhartree` matrix.
result: [pending]

### 3. python3.13t free-threading smoke (BIND-05)
expected: `maturin develop --no-default-features --features free-threading` builds; `import pyscf._native` succeeds under python3.13t (no-GIL); the GIL-release seam works without deadlock or segfault.
why_human: Requires the python3.13t interpreter (deadsnakes PPA or uv-installed); not available in this environment. CI job: `python313t-smoke`.
result: [pending]

### 4. BIND-04 NumPy stride-fuzz
expected: `pytest python/pyscf/tests/test_scf_stride_fuzz.py -x` exits 0; four stride variants (C-contig, transpose, slice-stride 2, slice-offset) of the same density matrix produce bit-identical `mf.get_veff` bytes via `np.testing.assert_array_equal`.
why_human: The BIND-04 NumPy contiguity policy must be exercised through actual PyO3 invocation; the test body exists but cannot run without maturin. CI job: `stride-fuzz`.
result: [pending]

### 5. BIND-07 subclass-override dispatch round-trip
expected: `pytest python/pyscf/tests/test_scf_override_dispatch.py` exits 0; a `CountedHF` subclass of `scf.RHF` that overrides `get_veff` shows the override called ≥1 time per SCF cycle; the `get_init_guess` override is also dispatched (bridge.rs:95-125).
why_human: The end-to-end round-trip requires the wheel and an installed upstream pyscf for `Mole.dumps()` serialization; cannot run without maturin. CI job: `maturin-smoke`.
result: [pending]

### 6. ORACLE-08 chkfile h5py↔hdf5 round-trip + BIND-09 panic→exception
expected: `test_scf_chkfile.py::test_chkfile_rs_writes_h5py_reads` and `test_chkfile_upstream_writes_pyscf_rs_reads` pass after `maturin develop` + h5py install (stale xfail markers removed); `test_panic_to_exception.py::test_rust_panic_becomes_python_exception` passes with `PyscfRsRuntimeError` bearing `.kind` and `.source_chain`.
why_human: Requires maturin + h5py; cross-language HDF5 byte-identity and the PyO3 panic→exception bridge are runtime contracts not exercisable via Rust unit tests. CI job: `maturin-smoke`.
result: [pending]

## Summary

total: 6
passed: 0
issues: 0
pending: 6
skipped: 0
blocked: 0

## Gaps

(No code-level gaps remain. The code-level gaps recorded in the 2026-05-12 revision —
`init_guess` minao/atom/huckel stubs, `mulliken_meta` NYI, `get_init_guess` dispatch,
chkfile auto-write/from_chk, and the int3c2e zero-buffer — are all resolved: minao in 03-13,
atom/huckel in 03-14, mulliken_meta in 03-15, the PyO3 dispatch/chkfile wiring earlier in the
phase, and int3c2e in the 03-12/03-13 numeric closure. The 6 items above are runtime/CI-gated
checks requiring the Python toolchain, not missing code.)
