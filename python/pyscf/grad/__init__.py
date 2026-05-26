"""pyscf.grad overlay — re-exports from pyscf._native.grad (BIND-02 / plan 07-09).

This is the pyscf-rs `_native` re-export overlay (NET-NEW for Phase 7 — no
`pyscf/grad/` overlay existed before this plan; only `cc`/`mp`/`scf`/`dft`). It
mirrors `python/pyscf/cc/__init__.py`: every user-facing gradient class + the
`Gradients()` factory resolve to the Rust PyO3 surface in `pyscf._native.grad`.

Factory dispatch (`Gradients(mf)`, the `mf.nuc_grad_method()` target — upstream
`pyscf/scf/hf.py:2484`):
  - MP2 post-SCF object       -> Mp2Gradients
  - CCSD post-SCF object      -> CcsdGradients
  - RKS / UKS reference (xc)  -> Rks/UksGradients
  - UHF reference             -> UhfGradients
  - otherwise (RHF-like)      -> RhfGradients (ECP folds into the HF path)

Cross-module dispatch (`mf.nuc_grad_method()`): upstream grafts `nuc_grad_method`
onto the SCF base class (`scf.hf.SCF.nuc_grad_method`, scf/hf.py:2484). Our `mf`
objects are the Rust `pyscf._native.scf.{RHF,UHF,GHF}` pyclasses, so we graft a
`nuc_grad_method` method onto each so `mf.nuc_grad_method().kernel()` resolves to
the `_native.grad.Gradients` factory (BIND-02).

Eager-snapshot contract (D-09): the PyO3 classes snapshot the converged SCF
reference from `mf` into plain Rust arrays at construction; the pyo3-free
`pyscf-grad` drivers do the compute. Subclass overrides of `grad_elec` dispatch
via the Python MRO (D-09, Pitfall 7); the no-override default runs pure-Rust
under `py.detach` (BIND-05).

`as_scanner()` returns a Mole -> (e_tot, de) callable (rhf.py:248-262 — a TUPLE,
distinct from the energy-only SCF/MP2/CCSD scanner) — the seam the native
geometry optimizer (`pyscf.geomopt`) drives its line-search on.

D-02 (cintx gating): the analytical-gradient NUMERIC rides the six grad-intor
families MISSING from cintx today (07-01); `kernel()` SURFACES a clean
cintx-availability error as a Python exception (never a panic across the FFI)
until the cintx grad-intor workstream lands. The bridge (wiring, snapshot,
dispatch, factory, graft) is always-on and structurally testable.
"""

from pyscf._native.grad import (  # type: ignore[attr-defined]
    Gradients,
    RhfGradients,
    UhfGradients,
    RksGradients,
    UksGradients,
    Mp2Gradients,
    CcsdGradients,
    Scanner,
)

__all__ = [
    "Gradients",
    "RhfGradients",
    "UhfGradients",
    "RksGradients",
    "UksGradients",
    "Mp2Gradients",
    "CcsdGradients",
    "Scanner",
]


def _graft_nuc_grad_onto_scf() -> None:
    """Graft `mf.nuc_grad_method()` onto the Rust SCF base classes (the upstream
    `scf.hf.SCF.nuc_grad_method` cross-module dispatch, scf/hf.py:2484).

    `mf.nuc_grad_method()` forwards to the `_native.grad.Gradients` factory,
    which dispatches MP2->Mp2Gradients / CCSD->CcsdGradients / UKS->UksGradients /
    RKS->RksGradients / UHF->UhfGradients / else RhfGradients. The RKS/UKS
    classes live in `_native.dft`, so we graft onto those as well.
    """
    def _nuc_grad_method(self):  # type: ignore[no-untyped-def]
        return Gradients(self)

    _nuc_grad_method.__name__ = "nuc_grad_method"
    _nuc_grad_method.__qualname__ = "SCF.nuc_grad_method"

    classes = []
    try:
        from pyscf._native.scf import RHF, UHF, GHF  # type: ignore[attr-defined]

        classes.extend([RHF, UHF, GHF])
    except ImportError:  # pragma: no cover - _native always present in a wheel build
        pass
    try:
        from pyscf._native.dft import RKS, UKS  # type: ignore[attr-defined]

        classes.extend([RKS, UKS])
    except ImportError:  # pragma: no cover
        pass

    for cls in classes:
        # Only graft if the class does not already define a nuc_grad_method (a
        # subclass override wins).
        if getattr(cls, "nuc_grad_method", None) is None:
            cls.nuc_grad_method = _nuc_grad_method


_graft_nuc_grad_onto_scf()
