# Python suite triage: molecular failures vs Phase 20

**Measured:** 2026-09-14, HEAD `5e6dd65` plus the Phase-20 working tree.
**Scope:** the MOLECULAR files (`test_scf_*.py`, `test_panic_to_exception.py`, `test_intor_spinor.py`, `test_overlay_resolution.py`, `test_complex_boundary.py`). `test_pbc_*` files were excluded.
**`.so`:** 15:04:13 for run A. The concurrent 20-12 agent rebuilt it at 15:35:57, while run B was in progress (see Caveats).

## Verdict

**None of the 19 molecular failures is caused by Phase 20.** Every one fails the same way with Phase 20's `sys.modules` registrations removed. Only the error *text* of the 9 `upstream`-fixture errors changed.

Nothing was fixed. No Rust or Python file was edited, and no worktree build was needed (see "Why no HEAD~1 build").

## Method

1. Each file was run on its own with `-q --tb=short`. Outputs are in the scratchpad `ind/*.out`.
2. **A/B in one binary, in suite (alphabetical) order:**
   - **A** is the tree as-is.
   - **B** adds `-p strip_plugin`. This is a `sys.meta_path` finder that lets the `pyscf._native` extension initialise, then deletes every `pyscf._native.*` entry it put in `sys.modules`. That removes 22 entries: the 20-08 nested `pbc` tree, 20-14's `pbc.lib.*`, and 20-09 step 0b's 7 flat modules plus the 2 geomopt shims. Nothing is left behind before any overlay subpackage import runs, which recreates the pre-Phase-20 import state.
   - Both runs: **10 failed, 43 passed, 4 skipped, 3 xfailed, 9 errors**. `diff` of the failing-id lists: **identical**.
3. Scope of the Phase-20 diff (`git diff 0f8b58d`):
   - It touches none of `scf.rs`, `errors.rs`, `dft.rs`, `python/pyscf/__init__.py`, `python/pyscf/{gto,scf}/`, `conftest.py`, or any `test_scf_*` / `test_panic_*` file.
   - In `gto.rs`, `bridge.rs`, `numpy_io.rs` and `lib.rs` the only removed line is one `use` re-spelling. Everything else is additive.
4. CI mode: `PYSCF_RS_UPSTREAM_PYTHON=$PWD/.venv/bin/python`. The subprocess cwd is the repo root, so it imports vendored 2.12.1.
   - `test_scf_rhf_h2o.py` + `test_scf_rhf_ccpvdz.py` → **4 passed**, exit 0.

## Table

| test | error | cause class | evidence | action |
|---|---|---|---|---|
| `test_scf_analyze.py` ×3 (errors) | `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` | pre-existing (overlay path) | See "Root cause R1". B (pre-20-09 state) fails identically with `No module named 'pyscf._native.gto'; 'pyscf._native' is not a package`. | none |
| `test_scf_chkfile.py::test_chkfile_upstream_writes_pyscf_rs_reads` (error) | same | pre-existing (R1) | same | none |
| `test_scf_df.py` (error) | same | pre-existing (R1) | same | none |
| `test_scf_diis.py` (error) | same | pre-existing (R1) | same | none |
| `test_scf_rhf_benzene.py` (error) | same | pre-existing (R1) | same | none |
| `test_scf_uhf.py` (error) | same | pre-existing (R1) | same | none |
| `test_scf_xplat_uhartree.py` (error) | same | pre-existing (R1) | same | none |
| `test_scf_rhf_h2o.py` | `moleintor` (in-process oracle fallback) | environment | CI sets `PYSCF_RS_UPSTREAM_PYTHON` (`ci.yml:344-346`). With it set: passed. | none; run with the env var |
| `test_scf_rhf_ccpvdz.py` ×3 | 1st: `moleintor`; then `M() got an unexpected keyword argument 'verbose'` | environment (+R1) | The first failure leaves a half-initialised `_upstream_pyscf` in `sys.modules`. Its `gto` global is the native overlay `gto`, so later params call native `M(verbose=0)`. With the env var: 3 passed (`ci.yml:355-358`). | none; run with the env var |
| `test_scf_cross_dispatch.py` ×3 | `DID NOT RAISE PyscfRsError` | pre-existing (test drift) | `to_uks`/`to_rks` were wired to real UKS/RKS in `fbe1e35` (04-09, 2026-05-22). The test still expects the Phase-3 stub to raise. `scf.rs` is untouched by Phase 20. Identical in B. | none |
| `test_scf_ghf.py` | `'pyscf._native.scf.GHF' object has no attribute 'run'` | pre-existing (Rust surface gap) | `PyGHF` (`scf.rs:800-`) has no `run`. `scf.rs` is untouched by Phase 20. Identical in B. | none |
| `test_panic_to_exception.py` ×2 | raises `_native.PyscfRsRuntimeError(..., 'Core', ...)`, not `PyscfRsError` | pre-existing (design defect) | See "Root cause R2". `errors.rs` last changed `3bc7f75` (2026-05-23). Identical in B. | none |
| `test_intor_spinor.py` ×4 (SKIPPED in suite order; FAILED if run after an R1 file) | skip reason `No module named 'pyscf.gto.moleintor'`; out of order: `'Mole' object has no attribute 'intor'` | pre-existing (R1 + test-order coupling) | Same in-process loader. Its `_upstream_or_skip` reuses a half-built `sys.modules['_upstream_pyscf']` left by conftest. In B the skip text is the `'_native' is not a package` form. | none |

### Root cause R1: the in-process upstream loader

**The mechanism.**
- `conftest._load_upstream` executes vendored `pyscf/__init__.py` as `_upstream_pyscf`. Its *absolute* imports (`from pyscf import ao2mo`) resolve against the overlay `pyscf`.
- The overlay `__path__` (`extend_path`) falls through to `.venv/.../site-packages/pyscf`. That copy is **2.14.0**, not 2.12.1.
- There, `ao2mo/_ao2mo.py:19` does `from pyscf.gto.moleintor import ...`. That lands on the overlay package `python/pyscf/gto/__init__.py` (added `52b6965`, 2026-05-25), which has no `moleintor` and no `extend_path`.

**Before and after 20-09.**
- Before 20-09 step 0b, that overlay's body raised "`'pyscf._native' is not a package`".
- After 20-09 it imports fine, and the lookup of `moleintor` fails instead.
- Either way the loader cannot work. Broken since `52b6965`; masked in CI by the subprocess oracle.

### Root cause R2: the exception class

`create_exception!(_native, PyscfRsRuntimeError, PyException)` raises the base class. The Python `PyscfRsError(_PyscfRsBase)` is a *subclass*, so `pytest.raises(PyscfRsError)` can never catch it.

The `kind` is also `'Core'`, not `ConvergenceFailure`, because the non-convergence surfaces as `CoreError::InvalidMolecule`. So the kind assertion would fail too.

This matches 20-08 D3.

## Why no HEAD~1 build

- The strip A/B isolates the only Phase-20 change on the molecular import path (`sys.modules` registration).
- Every remaining failure lies in code that Phase 20 did not modify: `scf.rs`, `errors.rs`, the overlay `gto`/`scf` packages, `conftest.py` and the tests.
- The Phase-20 Rust edits reachable from molecular classes are additive getters (`PyMole.natm`, …). No failure touches them.

## Caveats / environment findings

- The `.so` was rebuilt by 20-12 at 15:35:57, while run B was in progress. The failing set was still identical, and 20-12's edits are `pbc/scf.rs`/`kbridge.rs`, which molecular tests do not reach.
- **`.venv` site-packages has `pyscf 2.14.0`** (installed 2026-08-20 by uv), and EXECUTION-NOTES §2 says 2.12.1.
  - Outside the repo root, `import pyscf` gives 2.14.0.
  - From the repo root, cwd `''` puts vendored `pyscf/` (2.12.1) first.
  - Subprocess oracles that inherit repo-root cwd therefore get 2.12.1. Any oracle launched from elsewhere, and the in-process loader's fall-through submodules, get 2.14.0.
- Test-order coupling: R1 leaves a partial `_upstream_pyscf` in `sys.modules`. That turns `test_intor_spinor.py`'s skips into failures when it runs after an R1 file (not the case in alphabetical suite order).

## Artifacts

Scratchpad (session-local):
- `ind/*.out` — the per-file runs;
- `A_current.out` / `B_strip.out` / `*.ids` — the A/B;
- `strip_plugin.py`;
- `oracle_env.out` — the CI-mode run.
