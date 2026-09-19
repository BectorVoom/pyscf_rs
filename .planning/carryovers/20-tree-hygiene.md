# Carryover — build breakage and stale floor quotes left in the tree

**Source:** `.planning/phases/20-pbc-python-bindings/20-04-FIX-SUMMARY.md` (bench),
`20-19-D-SUMMARY.md` §Notes (stale floors), `20-19-AC-SUMMARY.md` D4; recorded by 20-18.

## (l) `pyscf-bench` does not compile

`cargo build --release -p pyscf-bench` → exit 101: `crates/pyscf-bench/src/bin/krks_profile.rs:1694`
(and `:1707`) call `mf.ni.unfold_kdms(...)` on a `KsNumInt`; the method exists only on `KNumInt`
(`crates/pyscf-pbc-dft/src/numint.rs`; the free function is `unfold_kdms_sym`). Pre-existing bench
drift, not part of the 20-04 fix hunks. Any `cargo build --workspace` / `cargo test --workspace`
job fails on it.

## (q) Stale floor quotes (the gates' tolerances are unchanged and correct)

20-19 D ported upstream's SCF `get_ovlp` precision rule (`pbc/scf/hf.py:47-55`); the periodic
oracle floors fell ~100× (KRHF Si gth 4.158e-12 → 2.931e-14; KRKS Si PBE 6.451e-12 → 4.086e-14;
KRHF bands mo_energy 6.10e-11 → 3.72e-13, get_bands 1.68e-11 → 4.23e-13; KUHF Li γ 1.493e-11 →
3.490e-12). The following still quote the old numbers:

- `crates/pyscf-pbc-dft/tests/gate.rs` doc comments (4.159e-12, 6.453e-12, …) and
  `crates/pyscf-pbc-dft/tests/gate_openshell.rs:107` ("`KRHF Si` sits at 4.158e-12 in `gate.rs`",
  and the "inherited from `get_pp`" explanation, which 20-19 D refuted);
- `.planning/phases/12-periodic-dft/12-VERIFICATION.md:77,79,113,142` (6.45e-12 and its §1c
  explanation) and `11-VERIFICATION.md:93,133` (diamond 4.00e-12, not re-measured after D);
- `.planning/phases/20-pbc-python-bindings/measurements/pbc-oracle-tiers.md:136,138,162,301,304`;
- `docs/pbc-status.md` and `python/pyscf/pbc/_unported.py` `FAMILIES` notes quote the 20-17 import
  count 41/195 (73/195 after 20-19 A);
- the user auto-memory `pbc-pseudopotential-oracle-floor` ("~4e-12 inherited from get_pp").

Rule for the fix: append the measured post-D number with provenance and mark the old one
superseded; never tighten a tolerance to the new floor without a re-measurement on the CI venv.
`measurements/README.md` §1 was updated by 20-18.
