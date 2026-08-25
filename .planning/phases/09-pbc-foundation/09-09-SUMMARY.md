---
phase: 09-pbc-foundation
plan: 09
type: summary
wave: 4
status: complete
completed: 2026-08-26
requirements: [PBC-ORACLE-01]
---

# 09-09 SUMMARY — Phase 9 verification rollup

**Phase 9 is CLOSED.** All seven success criteria in `09-CONTEXT.md` are
demonstrated green in `09-VERIFICATION.md`, every tier-2 reference number is
indexed with its generating snippet, and a venv-gated oracle compares this port
against live upstream PySCF for all five §9.2 systems.

## What shipped

### `crates/pyscf-pbc-gto/tests/oracle_phase9.rs` — the live-upstream oracle

Seven tests. Every one is `#[ignore]`d **and** short-circuits with a printed skip
line unless `PYSCF_ORACLE_VENV` is set, so:

* `cargo test --workspace` never touches Python;
* `cargo test -- --ignored` on a machine without an upstream venv still passes.

Each test spawns the gate interpreter on an embedded Python emitter (the script
is a `const &str` in the test file, so there is no second place to keep in sync)
and diffs the JSON it prints.

| test | criterion | what it compares |
|---|---|---|
| `cell_scalars_match_upstream` | 4 | `vol`, `rcut`, `mesh`, `b` (1e-12 rel / exact ints) + `b·aᵀ == 2π·I` |
| `gv_and_si_match_upstream` | 5 | `get_Gv([5,5,5])` and `get_SI` element-wise (1e-12) + `\|SI\| == 1` |
| `lattice_ls_match_upstream` | 6 | image COUNT exact + every component (1e-12) |
| `kpts_and_kconserv_match_upstream` | 6 | `make_kpts([2,2,2])`, `([3,2,1])` (1e-12); `get_kconserv` EXACT |
| `make_kpts_variants_match_upstream` | 6 | `with_gamma_point=False`, `wrap_around=True`, both, `scaled_center` |
| `ewald_matches_upstream` | 7 | `cell.ewald()` (1e-9 Ha), or the typed Phase-12 deferral for graphene |
| `angstrom_lattices_match_upstream_within_the_codata_gap` | — | the Angstrom path (see deviation 2) |

All five §9.2 systems in every test except the `make_kpts` variants (diamond).

`PYSCF_ORACLE_VENV` accepts `1`/`true`/`auto`/`yes` (→ `<root>/.venv/bin/python`),
a venv directory, or an interpreter path.

### `.planning/phases/09-pbc-foundation/09-VERIFICATION.md`

* §1 — one row per success criterion: criterion, proving command, observed
  value, PASS/FAIL. All seven PASS.
* §2 — the reference-value index: every tier-2 literal in the phase, the file it
  lives in, and where its generating Python snippet is recorded. Scalars are
  inlined (Ewald parameters, energies, the `η` scan, the plan-10-01 pseudised
  targets).
* §3 — the two standing caveats (the CODATA gap, all-electron charges).
* §4 — the deferred-branch table with owning plan and error type.
* §5 — carry-overs into Phase 10.
* §6 — the full command transcript.

### `python/pyscf/pbc/__init__.py`

An import-path shim with **nothing to re-export**. D-PBC-14 puts every periodic
PyO3 binding in Phase 20 plan 20-05: the Rust side is gated crate by crate first
and exposed in one pass. `pkgutil.extend_path` is called so submodules of the
vendored upstream `pyscf.pbc` that this file does not shadow still resolve.

### Bookkeeping

* `.planning/ROADMAP.md` — Phase 9 checkbox ticked with the verification link.
* `.planning/STATE.md` — progress to 9/9 (100%), a Phase 9 close-out decision row
  recording D-PBC-04 and R-02, two new blockers (cintx Wave 0.5, the CODATA
  gap), and the session-continuity entry.

## Green test command

```
cargo test --workspace                                             # all green
PYSCF_ORACLE_VENV=1 \
  cargo test -p pyscf-pbc-gto --test oracle_phase9 -- --ignored    # 7 passed
cargo test -p pyscf-pbc-gto --test oracle_phase9                   # 7 ignored (gate closed)
cargo run -p xtask --bin check-dependency-wall                     # PASS
cargo run -p xtask --bin check-forbidden-paths                     # PASS
```

Upstream: PySCF **2.12.1** (the vendored `<root>/pyscf` tree). Spot-checked
against the venv's 2.14.0 — every value identical.

## Deviations from the plan text

1. **The xtask binary names in the plan's verification block are wrong.**
   `check_dependency_wall` / `check_forbidden_paths` do not exist; cargo's own
   error names the real targets, `check-dependency-wall` and
   `check-forbidden-paths` (hyphens). Corrected everywhere.

2. **The oracle compares BOHR-specified geometry, plus a dedicated Angstrom
   test.** The plan asks for 1e-12 element-wise agreement. That is unreachable
   from Angstrom input: `pyscf_core::Unit::Ang` is CODATA-2014 and upstream is
   CODATA-2010, a 4.951e-9 relative lattice gap (plan 09-03). The oracle
   therefore rebuilds both sides from upstream's own Bohr numbers — the same
   resolution plans 09-05, 09-07 and 09-08 already use — so the 1e-12 gate
   measures the algorithm. The conversion path is NOT left uncovered:
   `angstrom_lattices_match_upstream_within_the_codata_gap` asserts the Angstrom
   deviation is EXACTLY the CODATA gap with a two-sided bound, so it fails loudly
   if either side ever changes its constant.

3. **`pseudo` is unset on both sides.** The plan's five systems carry
   `gth-pade`, but upstream `atom_charges()` then returns valence charges while
   this port returns all-electron `Z` until plan 10-01 (D-PBC-11). Comparing
   would have measured the missing GTH parser, not the Ewald code. The pseudised
   targets are committed in `ewald_reference::PSEUDISED_EWALD`.

4. **Graphene's `ewald()` is asserted as a DEFERRAL, not a value.** Upstream
   returns -44.57202102404764 through the `dimension == 2` truncated-Coulomb
   branch; this port returns `NotYetImplemented { phase: 12 }` per D-PBC-20. The
   oracle asserts exactly that pairing — a typed Phase-12 error against a real
   upstream number — so the test starts failing the moment plan 12-08 lands and
   forgets to update it.

5. **The plan says "write the pytest oracle".** It is a Rust test that spawns
   Python instead. Reason: the repo's existing `crates/pyscf-gto/tests/dump_*_for_oracle.rs`
   pattern splits the comparison across a Rust dumper and a Python harness, which
   means two files to keep in sync and a comparison that `cargo test` cannot run.
   Keeping the emitter as a `const &str` inside the Rust test gives one file, one
   command, and the `#[ignore]` + env-var gating the plan actually requires.

6. **Two extra tests beyond the plan's list.** `make_kpts_variants_match_upstream`
   (the four non-default flag combinations plan 09-07 shipped) and the CODATA-gap
   test above.

## One bug found and fixed in this plan's own harness

The first parallel run of the oracle failed 2 of 7 tests. Cause: `run_python`
wrote every emitted script to one fixed path, `<tmp>/pyscf_rs_oracle_<pid>/oracle.py`
— and `cargo test` runs these tests in parallel THREADS of ONE process, so a test
writing `ORACLE_ANG_PY` could clobber the `ORACLE_PY` another was about to
execute. Fixed with a `SCRIPT_SEQ` `AtomicUsize` giving each call its own
filename; verified stable over three consecutive parallel runs (7/7 each). Worth
recording because the failure only appears under parallelism — the first run,
with `--test-threads=1`, was green.

## Carry-overs

None for Phase 9 — it is closed. The items Phase 10 inherits are listed in
`09-VERIFICATION.md` §5; the load-bearing one is that **cintx Wave 0.5** (the 10
moment-weighted families `int3c1e_r{2,4,6}_origk`, `int1e_r{2,4}_origi`) is a
hard prerequisite for plan 10-05's GTH pseudopotentials. Plans 10-01…10-04 and
10-07 do not depend on it and can start immediately.
