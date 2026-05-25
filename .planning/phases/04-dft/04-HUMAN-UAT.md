---
status: partial
phase: 04-dft
source: [04-VERIFICATION.md]
started: 2026-05-23T05:30:57Z
updated: 2026-05-23T05:30:57Z
---

## Current Test

[awaiting human testing — blocked on Phase-2 ERI environment: maturin build + live pyscf + working 2e integrals]

## Tests

### 1. Python subclass override get_veff invoked every SCF cycle
expected: A Python subclass of RKS/UKS that overrides `get_veff` has its override called on every SCF iteration through the PyO3 bridge dispatch in `bridge.rs`; confirmed by cycle-count assertion in a live Python test.
result: [pending — requires maturin + live pyscf + Phase-2 int2e_sph/int3c2e_sph ERIs]

### 2. define_xc_ override invoked per cycle
expected: A Python subclass that sets `define_xc_` has that override invoked per SCF cycle via the bridge; confirmed in a live Python environment.
result: [pending — same environment constraint as above]

## Summary

total: 2
passed: 0
issues: 0
pending: 2
skipped: 0
blocked: 0

## Gaps
