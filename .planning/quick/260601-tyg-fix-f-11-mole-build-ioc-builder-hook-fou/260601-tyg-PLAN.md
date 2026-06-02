---
quick_id: 260601-tyg
slug: fix-f-11-mole-build-ioc-builder-hook-fou
description: "fix F-11: Mole::build IoC builder hook (FOUND-02 intact)"
date: 2026-06-01
status: planned
---

# Quick Task 260601-tyg — F-11: `Mole::build()` IoC builder hook

## Problem

`pyscf_core::Mole::build()` hard-errors with `NotYetImplemented{phase:2}` on a
direct call (`mole.rs:264`). Audit `AUDIT-FIX-2026-06-01.md` row F-11 classified
this *won't-fix* because `pyscf-core` is zero-compute-deps (FOUND-02): it cannot
depend on `pyscf-gto` (circular) and so cannot itself parse basis input.

**User decision (this task):** Implement an **inversion-of-control (IoC) builder
hook** so `Mole::build()` dispatches to `pyscf-gto`'s real builder *without*
`pyscf-core` gaining a compile-time dependency on `pyscf-gto`. FOUND-02 stays
intact.

## Design

A process-global function-pointer registry in `pyscf-core`:

- `pub type MoleBuilderHook = fn(&mut Mole) -> Result<(), PyscfRsError>;`
- `static MOLE_BUILDER: OnceLock<MoleBuilderHook>`
- `pub fn register_mole_builder(hook) -> Result<(), PyscfRsError>` (idempotent —
  ignores re-registration; only one impl ever exists in-tree).
- `Mole::build()` dispatch contract:
  - hook registered → call `hook(self)` (PySCF-faithful: always (re)builds, so the
    `mol.copy(); mol.basis = aux; mol.build()` pattern works), return `Ok(self)`.
  - hook **not** registered + already `_built` → `Ok(self)` (idempotent — the
    `pyscf_gto::M` front-door already populated it).
  - hook **not** registered + not built → keep the `NotYetImplemented` error, with
    an actionable message pointing at `pyscf_gto::M` / `register_mole_builder`.

`pyscf-gto` side:

- `build_in_place(&mut Mole)` reconstructs `MoleBuildArgs` **losslessly from the
  Mole's structured fields** and calls the existing `build_from`:
  - atom: `AtomInput::Tuples(mol._atom.clone())` with `unit: Unit::Bohr` (mol._atom
    is already stored in Bohr → avoids double unit conversion).
  - basis: strip the `{:?}` echo wrapper (`Name("x")` → `x`, same rule as
    `df_scf::extract_basis_name`) or take the raw string the user assigned →
    `BasisInput::Name`. Covers the canonical copy-rebuild/auxbasis case.
  - ecp: `EcpInput::None` when the echo is `"None"`, else best-effort `Name` strip.
  - scalars (`charge`, `spin`, `cart`, `verbose`, `max_memory`, `output`) read from
    the Mole.
- `pub fn register_mole_builder()` registers `build_in_place` (idempotent).
- Auto-register at the top of `M()` and `build_from()` so any gto front-door use
  arms the hook — the copy-rebuild source mol always comes through one of these.
- Wire `register_mole_builder()` into the PyO3 `#[pymodule]` init so the Python
  surface always has it armed.

### Why this respects FOUND-02

`pyscf-core` references only `fn(&mut Mole)` — a core-owned type. No
`pyscf-gto` types leak into core; `Cargo.toml` dep direction is unchanged
(`gto → core` only). The hook is *registered at runtime* by the higher layer.

### Known/documented limitations (not regressions)

- Basis re-derivation supports the `Name` form (and raw user-assigned names) — the
  dominant case and exactly what the upstream `auxmol` pattern uses. `PerElement`/
  `NwchemText`/etc. echo strings are not round-trippable; those users keep using
  `pyscf_gto::M`. Documented in the hook doc-comment.
- After a hook rebuild, `mol.unit` reads `Bohr` (coords are Bohr). Documented.
- Pure cold-start (`Mole::default()` + manual fields + `.build()` with **zero**
  prior gto calls and no `register_mole_builder()`) still returns the NYI error —
  strictly better than today (always errored) and FOUND-02-faithful.

## Tasks

### Task 1 — pyscf-core IoC registry + `Mole::build()` rewire
- **files:** `crates/pyscf-core/src/mole.rs`, `crates/pyscf-core/src/lib.rs` (re-export)
- **action:** Add `MoleBuilderHook` type, `MOLE_BUILDER: OnceLock`, `register_mole_builder`,
  `mole_builder_is_registered` (test helper), and rewrite `Mole::build()` per the
  dispatch contract above. Re-export the new public items from the crate root.
- **verify:** `cargo build -p pyscf-core` clean; existing pyscf-core tests pass.
- **done:** `Mole::build()` dispatches to a registered hook; unregistered behavior
  unchanged (idempotent-Ok if `_built`, else NYI).

### Task 2 — pyscf-gto hook impl + registration + PyO3 wiring + tests
- **files:** `crates/pyscf-gto/src/lib.rs`, `crates/pyscf-py/src/lib.rs` (pymodule init),
  new `crates/pyscf-gto/tests/mole_build_ioc.rs`
- **action:** Implement `build_in_place`, `register_mole_builder()`; auto-register in
  `M`/`build_from`; wire into PyO3 module init. Add tests:
  1. copy-rebuild auxbasis pattern (`M()` H2/sto-3g → clone → set basis "weigend" →
     `build()`) yields `nao_nr` matching a direct `M()` with the aux basis;
  2. `build()` on a freshly `M()`-built mol is a no-op-equivalent (idempotent values);
  3. cold-start (no registration) returns the NYI error (run in an isolated process/
     test that never calls gto front-door — or assert message text via the core unit
     test, since within the gto test binary the hook is auto-armed).
- **verify:** `cargo test -p pyscf-gto --test mole_build_ioc` green; `cargo build -p pyscf-py` clean.
- **done:** `Mole::build()` works end-to-end for the canonical pattern; PyO3 arms the hook.

## Out of scope
- Changing the `{:?}` echo storage in `build_from` (downstream `extract_basis_name`,
  `default_ri` depend on it).
- `PerElement`/text basis round-trip through the direct-build path.
- `#[ctor]`-style link-time auto-registration (avoids a new dep; runtime
  registration via M/build_from/pymodule covers every real path).
