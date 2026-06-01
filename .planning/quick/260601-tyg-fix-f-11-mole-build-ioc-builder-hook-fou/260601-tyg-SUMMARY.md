---
quick_id: 260601-tyg
slug: fix-f-11-mole-build-ioc-builder-hook-fou
description: "fix F-11: Mole::build IoC builder hook (FOUND-02 intact)"
date: 2026-06-02
status: complete
commits:
  - fa38417  # pyscf-core IoC registry (prior session, 2026-06-01)
  - cc7de9e  # pyscf-gto hook + py wiring + tests (this session)
---

# Quick Task 260601-tyg — F-11: `Mole::build()` IoC builder hook — COMPLETE

## Outcome

`pyscf_core::Mole::build()` no longer hard-errors on a direct call. It now
dispatches to `pyscf-gto`'s real builder through a runtime inversion-of-control
hook, so the upstream-faithful `mol.build()` entry point (and the canonical
`auxmol = mol.copy(); auxmol.basis = aux; auxmol.build()` rebuild pattern) works
end-to-end. **FOUND-02 is intact** — `pyscf-core` gained no dependency on
`pyscf-gto`; it holds only a process-global `fn(&mut Mole)` pointer.

This reclassifies audit row **F-11** from *won't-fix* → **FIXED** at the user's
explicit direction (IoC builder hook, FOUND-02 intact).

## What changed

**`pyscf-core` (registry — commit `fa38417`):**
- `MoleBuilderHook = fn(&mut Mole) -> Result<(), PyscfRsError>` + a
  `static MOLE_BUILDER: OnceLock<…>`.
- `register_mole_builder(hook)` (idempotent) + `mole_builder_is_registered()`.
- `Mole::build()` rewired: hook registered → invoke it (always, so rebuild
  works); else `_built` → idempotent `Ok`; else actionable `NotYetImplemented`.
- Re-exported from the crate root.

**`pyscf-gto` (hook + arming — commit `cc7de9e`):**
- `build_in_place(&mut Mole)` reconstructs `MoleBuildArgs` losslessly from the
  Mole's own fields — structured `_atom` (Bohr) → `AtomInput::Tuples` with
  `unit = Bohr` (no double conversion); `basis`/`ecp` via `strip_name_echo` of
  the `{:?}` echo (or a raw assigned name) → `BasisInput::Name`/`EcpInput` — then
  delegates to `build_from`.
- `register_mole_builder()` arms the core hook; auto-armed at the top of `M()`
  and `build_from()` and in the PyO3 `_native` module init.

**`pyscf-py`:** `pyscf_gto::register_mole_builder()` in the `#[pymodule]` init —
the whole Python surface has `Mole::build()` armed.

## Verification

- `pyscf-core` unit tests (9 pass) — incl. the **unregistered cold-start** arm
  (`build()` on a fresh Mole → `NotYetImplemented`, message points at
  `pyscf_gto::M`/`register_mole_builder`) and the idempotent-already-built arm.
  `pyscf-gto` is never linked in the core test binary, so this genuinely
  exercises the FOUND-02 fallback.
- `pyscf-gto/tests/mole_build_ioc.rs` (5 pass): front-door arms the hook;
  idempotent register; **copy-rebuild with a new basis** (sto-3g → cc-pvdz via
  `Mole::build()`, `nao_nr` matches a direct `M(cc-pvdz)`); same-basis rebuild is
  value-stable (no coordinate drift); direct build from structured atoms.
- Regression: `mole_copy`, `mole_construction`, `int3c2e_auxmol`,
  `intor_with_auxmol_smoke` (21 tests) all green — the `build_from` auto-arm
  didn't perturb the existing `M`/auxmol path.
- `pyscf-py` `cargo check` clean (PyO3 wiring compiles). No in-tree caller relies
  on `Mole::build()` erroring (grep-confirmed) → no behavioral regression.
- Builds run with `cargo +nightly` (workspace manifest needs nightly Cargo for
  the `pyscf-dft → libxc_rs profile-rustflags` feature). `pyscf-core`/`pyscf-gto`/
  `pyscf-py` dep graphs are all **0 libxc rows** — no ~6h libxc build triggered.
  Logs under `log/f11-*.log`.

## Documented limitations (not regressions)

- Only the `Name` basis/ecp form round-trips through the direct-build path (the
  dominant case; exactly what the upstream `auxmol` pattern uses). `PerElement`/
  `NwchemText`/text forms still go through `pyscf_gto::M`.
- After a hook rebuild, `mol.unit` reads `Bohr` (coordinates are stored in Bohr).
- Pure cold start with zero prior gto calls and no explicit
  `register_mole_builder()` still returns the (now actionable) NYI — strictly
  better than the old always-error, and FOUND-02-faithful.
- `#[ctor]`-style link-time auto-registration was intentionally avoided (no new
  dep; runtime arming via `M`/`build_from`/pymodule covers every real path).

## ⚠ Note on `fa38417` (shared-index race — prior session)

The pyscf-core registry commit `fa38417` (2026-06-01 21:40, a prior interrupted
run of this task) **also captured the deletion of 5 `.planning/research/*.md`
files** (ARCHITECTURE, FEATURES, PITFALLS, STACK, SUMMARY) — the shared-index
race documented in memory (`feedback_worktree_isolation_unreliable`): the commit
omitted `-- <pathspec>` and swept a concurrent agent's staged deletions. **No
content is lost** — all five remain in `.planning/research/v1.0-archive/` and in
commit `86a5bf8`; four also sit untracked on disk. This is prior-session
collateral, outside this task's scope; restoring their tracked state (if desired)
is a separate cleanup. This session's commit `cc7de9e` was pathspec-scoped and is
clean.
