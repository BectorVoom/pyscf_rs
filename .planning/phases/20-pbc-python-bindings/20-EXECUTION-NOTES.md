# Phase 20 — execution notes (read with 20-CONTEXT.md before any plan)

**Written:** 2026-09-13, at the start of execution. 20-CONTEXT.md was written
2026-09-12; the facts below moved since then or were found at kickoff. Where
this file and a PLAN disagree on a *fact*, this file wins; record the
deviation in the plan's SUMMARY.

## 1. Facts that moved since 20-CONTEXT

| CONTEXT / PLAN says | measured 2026-09-13 |
|---|---|
| Phase 19 "does not exist" (§1.2, §1.7) | **Phase 19 CLOSED** (`19-VERIFICATION.md`, commit `0f8b58d`). `pyscf-pbc-tdscf` 1,526 / `-gw` 1,286 / `-adc` 1,736 / `-x2c` 482 / `-eph` 95 src lines — no longer 13-line stubs. Still UNBOUND in Python. |
| Phase 18 "planned, never executed" | **Phase 18 IN PROGRESS, paused** (`18-IMPLEMENTATION-CHECKPOINT.md`). `pyscf-pbc-grad` 2,491 lines. Its work is **uncommitted in the working tree** (≈54 paths under `crates/pyscf-pbc-{df,dft,gto,grad}`, `crates/pyscf-kernels`, phase-18 measurements). |
| 161 ignored PBC gates | **170** (`grep -rc '#\[ignore' crates/pyscf-pbc-*/tests/*.rs`) |
| `cargo tree -p pyscf-py \| grep -c libxc` returns 0 (20-08, 20-13) | **547 today.** `pyscf-py` already takes `pyscf-dft` with default features, and `pyscf-dft`'s default has been `["libxc"]` since the 2026-08-28 XC default flip. libxc is ALREADY in the wheel. 20-08's "0" check and 20-13's "feature default OFF" posture are stale — see §3. |
| `pyscf-pbc-mpi`, `-geomopt` are 13-line stubs | still true |
| `python/pyscf/_native.abi3.so` | **stale — built 2026-08-20.** Rebuild before any Python test (§2). |

## 2. Build / test protocol (all plans)

* Tree compiles at kickoff: `cargo check -p pyscf-py` exit 0 (1m09s warm).
* **Python extension:** from repo root,
  ```bash
  export VIRTUAL_ENV=$PWD/.venv PATH=$PWD/.venv/bin:$PATH \
         CARGO_TARGET_DIR=$PWD/target/py CARGO_PROFILE_RELEASE_LTO=false CARGO_BUILD_JOBS=8
  maturin develop --release --skip-install
  ```
  writes `python/pyscf/_native.abi3.so` in place. Use EXACTLY this spelling
  (`LTO=false`, `target/py`) — any other value re-fingerprints and rebuilds the
  ~500 libxc kernel crates (memory: gate-target-dir-lto-spelling).
* Python tests: `.venv/bin/pytest python/pyscf/tests/<file> -q -p no:cacheprovider`.
* Rust gates: `CARGO_TARGET_DIR=$PWD/target/gate CARGO_PROFILE_RELEASE_LTO=false
  cargo test --release -p <crate> --test a --test b -j 6 -- --ignored`. Name every
  target; never bare `--tests`; confirm `exit=0`, never infer from process absence.
* Never `CARGO_TARGET_DIR` under `/tmp` (16 GB RAM tmpfs). Never `ulimit -v`;
  use `systemd-run --user --scope -p MemoryMax=16G` if a cap is needed.
* `rustfmt --edition 2024 <only the files you touched>`; never `cargo fmt`.
* Oracle is `pyscf==2.12.1` exactly (`.venv` has it; vendored tree at `pyscf/`).
* Code navigation: use the CodeGraph MCP tool `codegraph_explore` before
  grep/Read (user instruction for this phase).

## 3. Decisions taken at kickoff

* **D-20-A — no commits during execution.** The working tree carries Phase 18's
  uncommitted work in the same crates Phase 20 touches, and commits were not
  requested. Agents must NOT `git add`, `git commit`, `git stash`, `git restore`
  or `git checkout` anything. Never revert a file you did not edit.
* **D-20-B — libxc posture (restates 20-13 Task 1).** libxc is already linked
  into `pyscf-py` through `pyscf-dft` (§1). 20-13 adds `pyscf-pbc-dft` with its
  default features; the gate becomes "the `cargo tree -p pyscf-py` libxc line
  count does not change by more than the `pyscf-pbc-dft` edge itself" and "no
  xcfun default is introduced". 20-08's `grep -c libxc == 0` check is replaced by
  "the count is unchanged from 547 after adding the non-dft PBC deps".
* **D-20-C — the identity gate was written and run FIRST.**
  `python/pyscf/tests/test_pbc_identity_gate.py` (18 cases) ran on HEAD
  `0f8b58d` + the stale `.so` before any binding: **18 failed**
  (`ModuleNotFoundError: No module named 'pyscf._native.pbc'`), raw output in
  `measurements/identity-gate-pre-20-09.out`. This is 20-18 Task 1's
  "fails on the pre-20-09 tree" evidence, captured without a checkout.
* **D-20-D — 20-02 runtime budget.** A gate still running at 600 s is killed and
  recorded as `T3, >600 s, outcome not observed`; it is not run to completion
  in this phase. Other builds may run concurrently, so runtimes carry noise;
  borderline (within 2× of a tier edge) rows are flagged.
* **D-20-E — Phase-18 FFTDF exchange defect fixed in the working tree (2026-09-14).**
  `kscf::supercell_equivalence_holds` failed (k-point −10.347315387196 vs
  supercell/2 −10.531064341613). Bisected in a HEAD worktree
  (`measurements/kscf-supercell-regression.md`) to the uncommitted 18-04 hunk
  keying `Fftdf`'s `coulG` cache on the wrapped `kdiff_index` — `coulG` is not
  per-point invariant under `dk -> dk + b`. Re-keyed on raw `dk` bits
  (`crates/pyscf-pbc-df/src/fftdf.rs`), and `tests/fft_jk_grad.rs::
  coulg_build_counter_reads_nkpts` restated to "at most one build per distinct
  raw dk, fewer than nkpts²". After: delta **1.5650591933535907e-10**, k-point
  energy bit-identical to HEAD (−10.531064341456); `fft_jk_grad` 6 passed. Any
  Phase-18 exchange-gradient number measured with the class key must be
  re-measured by Phase 18.
