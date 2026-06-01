---
quick_id: 260601-fmc
slug: implement-format-atom-callable-form
date: 2026-06-01
finding: F-12
status: complete
---

# Quick Task 260601-fmc: Fix F-12 — `format_atom` callable atom form (GTO-01.5)

## Goal

Resolve audit finding **F-12** (`AUDIT-FIX-2026-06-01.md`): the `format_atom`
callable atom form, previously classified *manual-only — "Needs Rust-closure API
design decision."* Make the design decision and implement the callable form
entirely in Rust (no PyO3 dependency), mirroring upstream PySCF's callable
`atom=` form.

## Design decision (user-confirmed)

`AtomInput::Callable` becomes a closure variant:

```rust
pub type AtomCallable = Arc<dyn Fn() -> Result<AtomInput, PyscfRsError> + Send + Sync>;
```

- **`Arc` (not `Box`)** keeps `AtomInput: Clone` (other variants already Clone;
  `MoleBuildArgs`/`Mole` clone freely). `Send + Sync` lets a built molecule cross
  threads.
- The closure **produces another `AtomInput`** (typically `String`/`Tuples`),
  resolved by re-entering `format_atom` — so the produced spec gets identical
  unit/origin/axes treatment.
- **One-level recursion guard:** a callable that returns another callable is
  rejected (`InvalidMolecule`). Bounds the recursion and keeps the "callable
  produces a concrete spec" contract.
- `#[derive(Debug)]` dropped from `AtomInput` (closure isn't `Debug`); replaced
  with a manual `Debug` impl that forwards every concrete variant and prints
  `Callable(<closure>)`.

## Tasks

1. **`crates/pyscf-gto/src/types.rs`** — add `AtomCallable` type alias; change
   `Callable` to `Callable(AtomCallable)`; manual `Debug` impl; `AtomInput::callable`
   constructor. (`verify`: `cargo +nightly test -p pyscf-gto` compiles; `done`:
   enum carries the closure, derives intact.)
2. **`crates/pyscf-gto/src/lib.rs`** — re-export `AtomCallable`.
3. **`crates/pyscf-gto/src/format_atom.rs`** — replace the `NotYetImplemented`
   arm with: invoke closure → reject nested callable → recurse. Update module doc.
4. **`crates/pyscf-gto/tests/mole_construction.rs`** — replace the stale
   `callable_form_returns_not_yet_implemented_phase_3` test with 4 tests: congruence
   oracle (callable→String == direct String), unit conversion through callable,
   nested-callable rejection, closure-error propagation.

## Verification

- `cargo +nightly test -p pyscf-gto --locked` → full suite green (pyscf-gto
  excludes libxc → no ~6h build).
- `rustfmt --edition 2024 --check` on touched files → clean.
- `cargo +nightly clippy -p pyscf-gto` → no findings on new code.

## Out of scope

- The PyO3 (`pyscf-py`) bridge that would accept a *Python* callable and adapt it
  into an `AtomCallable` — that is genuine Phase 3 BIND work, not this audit-fix.
  The Rust-native closure form is now the foundation it would build on.
