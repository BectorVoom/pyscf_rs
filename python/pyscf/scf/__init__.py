"""pyscf.scf overlay — re-exports from pyscf._native.scf (BIND-02)."""
try:
    from pyscf._native.scf import RHF, UHF, GHF, density_fit  # type: ignore[attr-defined]
except ImportError:
    # Plan 03-07 fills these in. Until then, accessing them raises a clean
    # ImportError rather than NameError.
    def _not_built(name: str):
        def _raise(*_a, **_kw):
            raise ImportError(
                f"pyscf.scf.{name}: pyscf-rs Rust cdylib not built. "
                "Run `maturin develop` from the repo root."
            )
        return _raise

    RHF = _not_built("RHF")
    UHF = _not_built("UHF")
    GHF = _not_built("GHF")
    density_fit = _not_built("density_fit")
