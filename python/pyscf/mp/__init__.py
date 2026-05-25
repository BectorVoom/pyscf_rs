"""pyscf.mp overlay — re-exports from pyscf._native.mp (BIND-02 / plan 05-07).

This is the pyscf-rs `_native` re-export overlay (DISTINCT from the vendored
upstream `pyscf/mp/`). It mirrors `python/pyscf/scf/__init__.py`: every
user-facing MP2 class + the `MP2()` factory resolve to the Rust PyO3 surface in
`pyscf._native.mp`.

Factory dispatch (`MP2(mf, frozen=None)`, mirrors `pyscf/mp/__init__.py:MP2`):
  - RHF reference            -> RMP2
  - UHF reference            -> UMP2 (`mf.istype('UHF')`)
  - density-fitted reference -> DFMP2 (`getattr(mf, 'with_df', None)`)

Eager-snapshot contract (D-07): the PyO3 classes snapshot the converged SCF
reference from `mf` into plain Rust arrays at construction; the pyo3-free
`pyscf-mp2` kernels do the compute. Subclass overrides of `ao2mo`/`make_rdm1`/
`make_rdm2`/`energy` dispatch via the Python MRO (D-08, Pitfall 7).

`as_scanner()` (MP2-07) returns a Mole->energy callable that re-runs the
reference SCF at a new geometry, re-snapshots, and re-runs the MP2 kernel.
"""

from pyscf._native.mp import MP2, RMP2, UMP2, DFMP2  # type: ignore[attr-defined]

__all__ = ["MP2", "RMP2", "UMP2", "DFMP2"]
