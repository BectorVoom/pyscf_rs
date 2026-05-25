"""pyscf.cc overlay — re-exports from pyscf._native.cc (BIND-02 / plan 06-10).

This is the pyscf-rs `_native` re-export overlay (DISTINCT from the vendored
upstream `pyscf/cc/`). It mirrors `python/pyscf/mp/__init__.py`: every
user-facing CCSD class + the `CCSD()` factory resolve to the Rust PyO3 surface
in `pyscf._native.cc`.

Factory dispatch (`CCSD(mf, frozen=None)`, mirrors `pyscf/cc/__init__.py:83-139`):
  - RHF reference            -> RCCSD
  - UHF reference            -> UCCSD (`mf.istype('UHF')`)
  - density-fitted reference -> DFCCSD (`getattr(mf, 'with_df', None)`)

Cross-module dispatch (`mf.CCSD()` / `mf.density_fit().CCSD()`): upstream grafts
`CCSD` onto the SCF base class (`scf.hf.SCF.CCSD = CCSD`, cc/__init__.py:92).
Our `mf` objects are the Rust `pyscf._native.scf.{RHF,UHF,GHF}` pyclasses, so we
graft a `CCSD` method onto each so `mf.CCSD()` and `mf.density_fit().CCSD()`
resolve to the same `_native.cc.CCSD` factory (BIND-02). `density_fit()` already
returns an `mf` carrying `with_df`, so `mf.density_fit().CCSD()` routes to DFCCSD
via the factory's `with_df` branch.

Eager-snapshot contract (D-09): the PyO3 classes snapshot the converged SCF
reference from `mf` into plain Rust arrays at construction; the pyo3-free
`pyscf-ccsd` kernels do the compute. Subclass overrides of `ao2mo`/`update_amps`/
`make_rdm1`/`make_rdm2`/`energy` dispatch via the Python MRO (D-09, Pitfall 7);
the no-override default runs pure-Rust under `py.detach` (BIND-05) — the
`update_amps` default is the biggest GIL-released region in the project.

`as_scanner()` (CCSD-07) returns a Mole->energy callable that re-runs the
reference SCF at a new geometry, re-snapshots, and re-runs the CCSD kernel.
"""

from pyscf._native.cc import CCSD, RCCSD, UCCSD, DFCCSD, Scanner  # type: ignore[attr-defined]

__all__ = ["CCSD", "RCCSD", "UCCSD", "DFCCSD", "Scanner"]


def _graft_ccsd_onto_scf() -> None:
    """Graft `mf.CCSD()` onto the Rust SCF base classes (the upstream
    `scf.hf.SCF.CCSD = CCSD` cross-module dispatch, cc/__init__.py:92).

    `mf.CCSD(frozen=None)` forwards to the `_native.cc.CCSD` factory, which
    dispatches UHF->UCCSD / with_df->DFCCSD / else RCCSD. `mf.density_fit()`
    returns an `mf` carrying `with_df`, so `mf.density_fit().CCSD()` routes to
    DFCCSD automatically.
    """
    def _ccsd_method(self, frozen=None):  # type: ignore[no-untyped-def]
        return CCSD(self, frozen)

    _ccsd_method.__name__ = "CCSD"
    _ccsd_method.__qualname__ = "SCF.CCSD"

    try:
        from pyscf._native.scf import RHF, UHF, GHF  # type: ignore[attr-defined]
    except ImportError:  # pragma: no cover - _native always present in a wheel build
        return
    for cls in (RHF, UHF, GHF):
        # Only graft if the class does not already define a CCSD method (a
        # subclass override wins).
        if getattr(cls, "CCSD", None) is None:
            cls.CCSD = _ccsd_method


_graft_ccsd_onto_scf()
