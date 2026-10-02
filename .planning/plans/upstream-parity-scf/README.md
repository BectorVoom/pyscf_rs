# Upstream parity of the periodic SCF — executor guide

You are implementing three fixes so that this Rust port gives the same numbers
as upstream PySCF 2.12.1. Work through `tasks/T01 … T17` **in order, one task
at a time**. Each task file is self-contained. Do not read `DESIGN.md`.

## The three fixes

| id | what is wrong today | fixed by |
|---|---|---|
| D2 | level shift uses `S − ½·S·D·S` for every SCF type | T02–T04 |
| D3 | no final diagonalisation after convergence (`conv_check`) | T05–T07 |
| D1 | non-local pseudopotential under FFTDF is evaluated analytically; upstream evaluates it on the FFT mesh in reciprocal space | T08–T15 |

T16–T17 validate. T01 is the baseline.

## Decisions already made — do not re-open them

1. D1 default route = `Reciprocal` (upstream). The old route stays as opt-in
   `PYSCF_PBC_FFTDF_PP_NL=realspace`.
2. D2 follows upstream exactly, including its restricted k-point form.
3. `conv_check` default = `true` in the library; the band pipeline turns it
   on for the LAST SCF stage only.
4. Kernels K-PP1 and K-PP2 are concrete `f64` and call `cube_math::double`
   (the repo's documented exception — see
   `crates/pyscf-kernels/src/pbc/ft_aopair.rs:27-37`). Kernel K-PP3 is generic
   `F: Float`.

## Rules for every task

1. Do the steps in the order written. Do not skip the "Verify" step.
2. Change only the files a task names. If you think another file must
   change, STOP and report (rule 9).
3. Tests go in separate files under `tests/`. Never add `mod tests` or
   `#[cfg(test)]` to a file under `src/`.
4. Before writing or editing any `#[cube]` kernel, read the CubeCL manual
   pages the task lists. The manual is at
   `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/`
   (index: `INDEX.md`).
5. If a build fails with an error that mentions `cubecl` or a `#[cube]`
   function, open
   `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/cubecl_error_solution_guide/`
   and follow the matching file BEFORE changing code. The three files there:
   - calling a normal Rust function inside `#[cube]` → E0433
   - `ArrayArg::from_raw_parts` needs an `unsafe` block → E0133
   - mismatched types → E0308 / E0599
6. Never run `cargo fmt` on the workspace. Format only the files you
   touched: `rustfmt --edition 2024 <file>`.
7. Never run two cargo commands at the same time. Never use bare `--tests`.
8. Do not commit, push, or start a Kaggle run unless a task says so.
9. **STOP rule.** If a Verify step fails and the task's "If it fails" list
   does not fix it within TWO attempts, stop. Write what you ran, the full
   error, and what you tried, to `tasks/REPORT-<task id>.md`. Do not
   continue to the next task. Do not weaken a tolerance, delete a test, or
   mark a test `#[ignore]` to get past a failure.
10. After each task, append one line to `PROGRESS.md` in this directory:
    `T07 done — <the number the Verify step printed>`.

## Commands

Run everything from the repository root
`/home/user/Documents/workspace/pyscf_rs`.

```bash
# TEST  — one named test target of one crate
CARGO_TARGET_DIR=target/gate CARGO_PROFILE_RELEASE_LTO=false \
  cargo test --release -p <crate> --test <target> -- --nocapture

# ORACLE — the same, against upstream PySCF (tests marked #[ignore])
PYSCF_ORACLE_VENV=1 CARGO_TARGET_DIR=target/gate CARGO_PROFILE_RELEASE_LTO=false \
  cargo test --release -p <crate> --test <target> -- --ignored --nocapture

# CHECK — fast compile check of one crate
CARGO_TARGET_DIR=target/gate CARGO_PROFILE_RELEASE_LTO=false cargo check --release -p <crate> --tests
```

Spell `CARGO_PROFILE_RELEASE_LTO=false` exactly (`off` rebuilds 500 crates).
The first build of `pyscf-pbc-dft` can take 20–60 minutes; that is normal.
A test run is finished only when you have seen `test result: ok` and the
command's exit status is 0.

## CubeCL rules (from the manual) — apply to every kernel in T11–T14

| rule | manual page |
|---|---|
| A helper called inside `#[cube]` must itself be `#[cube]`; no normal Rust functions | `cubecl_error_solution_guide/calling a “normal” Rust function …` |
| No `continue` in a kernel loop; wrap the body in `if` instead | `Cubecl_loop_control.md` |
| No `let x = if c { a } else { b };` — write `let mut x = a; if c { x = b; }` | `Cubecl_conditionals.md` |
| No `while { if … else … }` inside a kernel — use `for i in 0..n` (it aborts the CPU runtime) | repo finding, see `crates/pyscf-algebra/src/launch.rs` module docs |
| Generic kernel: `fn k<F: Float>(…)`, constants as `F::from_int(0)`; launched as `k::launch_unchecked::<F, R>` | `Cubecl_generics.md` |
| Every lane starts with a guard `if i < lanes { … }` (launches round up) | `Cubecl_grid_stride_loop.md` |
| Upload tables once per call, not per block; read results back in ONE `client.read(vec![…])` | `11_launch_overhead_and_transfers.md` §2, §3 |
| One launch covers all items; do not launch per item | `11_launch_overhead_and_transfers.md` §5 |
| `ArrayArg::from_raw_parts` only inside `unsafe { }` with a `// SAFETY:` comment | `cubecl_error_solution_guide/call to unsafe function …` |
| No `Array::new` local arrays inside a kernel | `13_memory_preallocation.md` + `crates/pyscf-algebra/src/launch.rs:200-215` |
| Pick the launch size with `pyscf_algebra::launch::launch_1d(client, lanes, work_per_lane)`; never hard-code a cube size | `Hardware-Adaptive_Launch_Geometry.md` |

## Task list

| task | title | crate |
|---|---|---|
| T01 | Baseline | — |
| T02 | Level shift: red test | pyscf-pbc-scf |
| T03 | Level shift: fix | pyscf-pbc-scf, pyscf-pbc-dft, pyscf-py |
| T04 | Level shift: device products | pyscf-pbc-scf |
| T05 | Final diagonalisation: red oracle test | pyscf-pbc-dft |
| T06 | Final diagonalisation: implement | pyscf-pbc-scf |
| T07 | Final diagonalisation: pipeline | pyscf-pbc-dft example |
| T08 | G-space pseudopotential: red oracle tests | pyscf-pbc-df, pyscf-pbc-dft |
| T09 | Projector tables on the host | pyscf-pbc-df |
| T10 | G-space route on the host + switch | pyscf-pbc-df |
| T11 | Kernel K-PP3 (fold) | pyscf-kernels |
| T12 | Kernel K-PP2 (projector block) | pyscf-kernels |
| T13 | Kernel K-PP1 (AO transform block) | pyscf-kernels, pyscf-pbc-df |
| T14 | Device driver + memory budget | pyscf-kernels, pyscf-pbc-df |
| T15 | Pipeline fingerprint, config, docs | example, tools/kaggle-t4 |
| T16 | Local validation against PySCF | — |
| T17 | T4 validation (owner starts the run) | — |

## Glossary

- **planar** — a complex array stored as two `f64` arrays, `re` and `im`.
  Never interleave.
- **row-major (nrow, ncol)** — element `(r, c)` is at index `r * ncol + c`.
- **lane** — one kernel thread; its index is `ABSOLUTE_POS`.
- **oracle** — a test that runs upstream PySCF through Python and compares.
- **Γ (gamma)** — the k-point `[0, 0, 0]`.
