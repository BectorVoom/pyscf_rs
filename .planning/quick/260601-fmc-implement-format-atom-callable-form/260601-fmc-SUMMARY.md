---
quick_id: 260601-fmc
slug: implement-format-atom-callable-form
date: 2026-06-01
finding: F-12
status: complete
commit: 444d868
---

# Quick Task 260601-fmc: F-12 — `format_atom` callable atom form — SUMMARY

## Outcome: FIXED

Audit finding **F-12** (`format_atom` callable atom form, GTO-01.5) — previously
*manual-only, "Needs Rust-closure API design decision"* — is resolved. The
callable atom form is now a first-class, pure-Rust input form.

## What changed

- **`crates/pyscf-gto/src/types.rs`**
  - New `pub type AtomCallable = Arc<dyn Fn() -> Result<AtomInput, PyscfRsError> + Send + Sync>`.
  - `AtomInput::Callable` (was a unit variant returning `NotYetImplemented{phase:3}`)
    is now `Callable(AtomCallable)`.
  - `#[derive(Debug)]` dropped (closure isn't `Debug`); replaced with a manual
    `impl Debug` — concrete variants forward to their inner value, `Callable`
    prints `Callable(<closure>)`. `#[derive(Clone)]` retained (`Arc` is `Clone`).
  - New `AtomInput::callable(f)` constructor.
- **`crates/pyscf-gto/src/lib.rs`** — re-export `AtomCallable`.
- **`crates/pyscf-gto/src/format_atom.rs`** — `Callable(produce)` arm invokes the
  closure, **rejects a nested callable** (one-level guard → `InvalidMolecule`),
  then re-enters `format_atom` on the produced spec so it gets identical
  unit/origin/axes treatment. Module doc updated.
- **`crates/pyscf-gto/tests/mole_construction.rs`** — replaced the stale
  `callable_form_returns_not_yet_implemented_phase_3` test with 4:
  - `callable_form_returning_string_matches_direct` — congruence oracle: a
    callable returning `"H 0 0 0; H 0 0 1.4"` builds the byte-identical molecule
    (`_atom`, `natm`, `nelectron`) to passing that string directly.
  - `callable_form_honours_unit_conversion` — Å→Bohr applies to callable output.
  - `callable_returning_callable_is_rejected` — one-level recursion guard.
  - `callable_error_propagates` — closure error surfaces unchanged.

## Design decision (user-confirmed via AskUserQuestion)

`Arc<dyn Fn() -> Result<AtomInput, PyscfRsError> + Send + Sync>`, recursive with a
one-level guard. `Arc` (over `Box`) preserves `AtomInput: Clone`; the closure
produces another atom spec (faithful to upstream PySCF's callable `atom=` form)
resolved recursively, with nested callables rejected to bound recursion.

## Verification (in-sandbox; logs under `log/`)

- `cargo +nightly test -p pyscf-gto --locked` → full suite green, 0 failures
  (incl. the 4 new callable tests + the `AtomInput::callable` doctest).
  `log/f12-pyscf-gto-test.log`.
- `rustfmt --edition 2024 --check` on the 4 touched files → clean. `log/f12-fmt.log`.
- `cargo +nightly clippy -p pyscf-gto --locked` → no findings on new code
  (the lone "1 warning (1 duplicate)" is the workspace-wide `fma4`/proc-macro
  future-incompat notice every unchanged crate emits, not F-12 code).
  `log/f12-clippy.log`.
- `pyscf-gto` excludes libxc (no ~6h build).

## ⚠ Shared-working-tree race (commit provenance)

This quick task ran **concurrently with two other `claude` agents in the same
working tree** (one running quick task `si2`, F-07 UCCSD `urdm.rs`). At the moment
this task ran `git add <4 pyscf-gto files> && git commit`, the `si2` agent issued
its own commit against the shared index — **sweeping the F-12 `pyscf-gto` files
into the F-07 commit `444d868`** ("feat(quick-260601-si2): F-07 port spin-orbital
GCCSD RDMs into urdm.rs"). This task's own `git commit` then found nothing staged.

**Net effect:** the complete, tested F-12 change is committed (in `444d868`) and
correct, but it is co-mingled with F-07's `pyscf-ccsd/urdm.rs` change under an
F-07 commit message.

**Why not split it:** rewriting `444d868` (the live `HEAD`) via reset/rebase while
two other agents are actively committing to the same tree risks clobbering their
in-flight work and corrupting the index. The safe choice was to leave history
intact and document the provenance here. The F-12 `pyscf-gto` portion of `444d868`
is cleanly separable by path (`crates/pyscf-gto/*`) if a later single-writer
session wants to disentangle it.

**Lesson (see memory `feedback_worktree_isolation_unreliable`):** multi-agent
runs on a shared tree need single-writer commit discipline; even explicit
`git add <path>` does not protect against a concurrent `git commit` grabbing the
shared index. Prefer scoping commits with `git commit -- <pathspec>`.

## Out of scope

- PyO3 bridge to accept a *Python* callable and adapt it into an `AtomCallable`
  (Phase 3 BIND). The Rust-native closure form is the foundation it builds on.
