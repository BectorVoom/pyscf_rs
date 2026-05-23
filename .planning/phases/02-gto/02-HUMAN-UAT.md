---
status: partial
phase: 02-gto
source: [02-VERIFICATION.md]
started: 2026-05-23T11:30:00Z
updated: 2026-05-23T11:30:00Z
---

## Current Test

[awaiting human testing]

## Tests

### 1. Upstream byte-identity for mol.intor('int1e_ecp') on Cu/LANL2DZ
expected: pyscf-rs int1e_ecp matrix agrees with upstream PySCF to atol=1e-10 on Cu/LANL2DZ
result: [pending]
why_human: tests/oracle/test_ecp_int1e.py requires numpy + upstream pyscf venv (tests/oracle/requirements.txt), unavailable in the default sandbox. cintx itself pins atol=1e-12 vs vendored PySCF nr_ecp in cintx-oracle/tests/safe_api_ecp_parity.rs, so byte-identity is indirectly verified at source.
command: pytest tests/oracle/test_ecp_int1e.py::test_cu_lanl2dz_int1e_ecp_byte_equal -v

### 2. Upstream byte-identity for _atm/_bas/_env/ao_loc_nr/nao_nr (H2O/cc-pVDZ, benzene/6-31G*, water-trimer/STO-3G)
expected: tests/oracle/test_byte_identity.py exits 0 — 15 byte-equal assertions (3 fixtures x 5 arrays)
result: [pending]
why_human: Requires upstream pyscf venv. Test exists at tests/oracle/test_byte_identity.py; CI owns the python-side byte-identity assertion.
command: pytest tests/oracle/test_byte_identity.py -v

### 3. mol.intor() arity-2 parity vs upstream (7 names) + Pitfall 8 F-order layout
expected: tests/oracle/test_intor_oracle.py exits 0 — 7 arity-2 integral names green at atol=1e-10, F-order layout preserved
result: [pending]
why_human: Requires upstream pyscf venv. Test exists at tests/oracle/test_intor_oracle.py.
command: pytest tests/oracle/test_intor_oracle.py -v

## Summary

total: 3
passed: 0
issues: 0
pending: 3
skipped: 0
blocked: 0

## Gaps
