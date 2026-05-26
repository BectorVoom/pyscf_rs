"""pyscf.geomopt overlay — re-exports from pyscf._native.geomopt (BIND-02 / plan 07-09).

This is the pyscf-rs `_native` re-export overlay (NET-NEW for Phase 7 — no
`pyscf/geomopt/` overlay existed before this plan; only `cc`/`mp`/`scf`/`dft`/
`grad`). It mirrors `python/pyscf/cc/__init__.py`: the `optimize` entry point +
the `geometric_solver` / `berny_solver` solver shims resolve to the Rust PyO3
surface in `pyscf._native.geomopt`.

GEOMOPT-01 (the CRITICAL no-runtime-dep contract): the geometry optimizer is
FULLY NATIVE. This overlay and the Rust `_native.geomopt` engine pull in NEITHER
external optimizer package — both `geometric_solver` and `berny_solver` delegate
to the ONE native `pyscf_geomopt::optimize` BFGS+RFO engine (D-06 / T-07-20).
User scripts using the `geometric_solver` / `berny_solver` entry points run
UNCHANGED, with no external optimizer package installed. (The uninstall-the-
external-packages CI proof lands in 07-10.)

Entry shapes (mirroring `pyscf/geomopt/geometric_solver.py:96-192`, D-07):
  - `pyscf.geomopt.optimize(method, maxsteps=100, ...) -> Mole`  (GEOMOPT-02/03)
  - `geometric_solver.kernel(method, ...) -> (conv, Mole)`
  - `geometric_solver.optimize(method, ...) -> Mole`  (`== kernel(...)[1]`)
  - `berny_solver.kernel/optimize(method, ...)` — the same, a thin alias.

`method` may be a grad scanner (`g.as_scanner()`), a Gradients object (has
`as_scanner`), or an `mf`/post-SCF object (has `nuc_grad_method`); the Rust
bridge resolves it to the native `GradScanner` the optimizer drives. A non-None
`constraints` kwarg raises a clear error (the 07-06 `ConstraintsUnsupported`),
never a silent no-op (T-07-33); `maxsteps` defaults to 100 and is capped at the
bridge boundary (T-07-32).
"""

from pyscf._native.geomopt import (  # type: ignore[attr-defined]
    optimize,
    geometric_solver,
    berny_solver,
)

__all__ = ["optimize", "geometric_solver", "berny_solver"]
