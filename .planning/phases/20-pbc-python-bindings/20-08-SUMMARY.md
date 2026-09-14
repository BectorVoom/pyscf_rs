# 20-08 SUMMARY — nested `_native.pbc.*` module tree (empty children)

**Shipped:** 2026-09-14. All four tasks done. Every verification command is green, with the deviations recorded below.

## Task 1–2 — the nested tree, importable

`crates/pyscf-py/src/pbc/mod.rs` (new) exposes `pub const PBC_MODULE = "pyscf._native.pbc"`,
`pub const PBC_CHILDREN: [(&str, &str); 10]` (child, owning plan) and
`pub fn register(py: Python<'_>, root: &Bound<'_, PyModule>) -> PyResult<()>`. It is called from `lib.rs` after the seven flat
submodules and declared `pub mod pbc;`.

The pattern, established once for 20-09…20-15:
1. `PyModule::new(py, "pyscf._native.pbc.<child>")`, so the full dotted `__name__` shows in `repr()` and pickling;
2. `parent.add_submodule(&child)`. PyO3 0.28 strips the dotted prefix, so the attribute is the short name;
3. `sys.modules["pyscf._native.pbc.<child>"] = child`, and the same for `pyscf._native.pbc`.

The children are `gto, scf, dft, df, symm, lib, tools, mp, cc, ci`. Each is empty apart from a `__doc__` naming its plan.
Classes added later must use `#[pyclass(module = "pyscf._native.pbc.<child>")]` (documented in the file header).

## Task 3 — deps

Added to `crates/pyscf-py/Cargo.toml`: `pyscf-pbc-{gto,scf,df,symm,lib,tools,mp,cc,ci,ao2mo}`. `pyscf-pbc-dft` was **not** added.

## Task 4 — overlay docstring

`python/pyscf/pbc/__init__.py` now lists what is registered (the nested tree and the 20-07 boundary) and what is not
(every child is empty, `__all__ = []`, and the identity gate still fails by design). `pkgutil.extend_path` is **kept** and marked as 20-17's decision.

## Verification (2026-09-14, rebuilt `.so`)

| command | result |
|---|---|
| `maturin develop --release --skip-install` (EXECUTION-NOTES §2 env) | **maturin exit=0**. Cargo reported `Finished release in 64m 32s`. Shell `real 547m` includes a system suspend (22:19, `journalctl`). `.so` = 589,570,345 B, Sep 14 06:24 |
| `.venv/bin/python -c "import pyscf._native.pbc.scf, pyscf._native.pbc.gto; print('ok')"` | `ok`, exit=0 |
| `.venv/bin/python -c "import pyscf._native.pbc.scf as m; print(m.__name__)"` | `pyscf._native.pbc.scf`, exit=0 |
| all 10 children: `importlib.import_module`, `sys.modules[...] is m`, `getattr(_native.pbc, c) is m`, no public attrs | `10 children ok, empty`, exit=0 |
| `cargo tree -p pyscf-py \| grep -c libxc` | **547** before and after (D-20-B); `grep -c pyscf-pbc-dft` → **0** |
| `cargo run -p xtask --bin check-dependency-wall` | exit=0 (both PASS lines) |
| `cargo run -p xtask --bin check-orphan-modules` | exit=0, `PASS — 431 source files, all reachable` |
| `check-catch-unwind` / `check-forbid-lazy-static` | exit=0 / exit=0 |
| `pytest test_complex_boundary.py test_overlay_resolution.py test_scf_smoke.py test_panic_to_exception.py test_scf_stride_fuzz.py test_intor_spinor.py` | **25 passed, 4 skipped, 2 failed** (both `test_panic_to_exception.py`, see D3) |
| `pytest test_pbc_identity_gate.py` | 18 failed, **as expected** (children empty). The failure mode moved, see D4 |

## Deviations / findings

- **D1 — "build under 2 minutes warm" did not hold, but not because libxc leaked in.** `pyscf-pbc-gto` takes
  `rmath = { path }` with default features. That changes rmath's feature list in pyscf-py's graph from `wide` to
  `default,wide`. The effective feature set is identical, but cargo's fingerprint changes, so `rmath` → `libxc-rkernel-math` → **all
  268 libxc kernels** rebuild once per target dir (about 65 min). The gth-pp default also turns on `cintx-*/unstable-source-api`.
  Measured by a `cargo tree -e features` diff with and without the new edges. libxc count unchanged at 547. This is a one-time
  cost, and `target/py` is warm again. Avoidable only by `default-features = false, features = ["wide"]` on that rmath edge inside
  `pyscf-pbc-gto`, which this plan may not touch (and which holds uncommitted Phase-18 edits).
- **D2 — libxc gate** restated per D-20-B: "547 unchanged", not "0".
- **D3 — `test_panic_to_exception.py` 2 failures are not caused by this plan.** Rust raises the base
  `_native.PyscfRsRuntimeError` (kind `'Core'`). The test expects the Python *subclass* `PyscfRsError`, which can never
  catch a base-class instance. `errors.rs`, `scf.rs` and `python/pyscf/__init__.py` are untouched here. It could not
  be A/B'd against a HEAD `.so`: the stale 2026-08-20 `.so` was overwritten, and the orchestrator's warm build failed at
  the leaf because it picked up 20-07's source with the pre-20-07 manifest.
- **D4 — the identity gate now fails one step later, on a pre-existing flat-module defect.** `import_module("pyscf._native.pbc.gto")` now
  succeeds. The failure is in the overlay path: upstream `pyscf.pbc.gto` → `pyscf.dft.radi` →
  `python/pyscf/dft/__init__.py:17 from pyscf._native.dft import RKS` → `ModuleNotFoundError: 'pyscf._native' is not a
  package`. The seven FLAT submodules were never entered in `sys.modules`, so `from pyscf._native.<flat> import X` fails
  whenever the `python/` overlay is the `pyscf` on the path (reproduced from `cwd=python/`). 20-17 / 20-18 will hit
  this. The fix is the same step-3 `sys.modules` line for the flat modules, but that is out of 20-08's scope and was not done.
