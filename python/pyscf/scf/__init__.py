"""pyscf.scf overlay — re-exports from pyscf._native.scf (BIND-02).

Plan 03-07 makes this import unconditional. `density_fit` is a method on
RHF/UHF (`mf.density_fit(...)`), not a top-level function — the BIND-03
idiom `mf.density_fit().run()` covers the user-facing surface.
"""
from pyscf._native.scf import RHF, UHF, GHF  # type: ignore[attr-defined]

__all__ = ["RHF", "UHF", "GHF"]
