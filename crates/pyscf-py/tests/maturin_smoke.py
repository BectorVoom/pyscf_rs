"""Smoke: `maturin develop` produces an importable `pyscf._native` module.

BIND-01 (cdylib + abi3-py310) + BIND-02 (overlay resolution).

Run from CI: `maturin develop && python crates/pyscf-py/tests/maturin_smoke.py`.

This test file is a plain Python script (not a pytest fixture) so it can
run before pytest is installed and so the maturin CI job (plan 03-09)
has a single-file smoke gate.
"""

import sys


def test_module_imports():
    from pyscf._native import scf  # type: ignore[attr-defined]
    assert hasattr(scf, "RHF"), "pyscf._native.scf missing RHF (BIND-02)"
    assert hasattr(scf, "UHF"), "pyscf._native.scf missing UHF"
    assert hasattr(scf, "GHF"), "pyscf._native.scf missing GHF"
    assert hasattr(scf, "Scanner"), "pyscf._native.scf missing Scanner (SCF-12)"
    print("OK: pyscf._native.scf imports cleanly")


def test_overlay_resolution():
    from pyscf import scf
    from pyscf._native.scf import RHF as RHF_native  # type: ignore[attr-defined]

    assert scf.RHF is RHF_native, (
        f"pyscf.scf.RHF must resolve to pyscf._native.scf.RHF; "
        f"got {scf.RHF} vs {RHF_native}"
    )
    print("OK: `from pyscf import scf` resolves to overlay (BIND-02)")


def test_upstream_compat_shims():
    """BIND-02 idiom: `from pyscf.scf.hf import RHF` works as in upstream PySCF."""
    from pyscf.scf.hf import RHF as RHF_hf  # type: ignore[attr-defined]
    from pyscf.scf.uhf import UHF as UHF_uhf  # type: ignore[attr-defined]
    from pyscf.scf.ghf import GHF as GHF_ghf  # type: ignore[attr-defined]
    from pyscf._native.scf import RHF, UHF, GHF  # type: ignore[attr-defined]

    assert RHF_hf is RHF
    assert UHF_uhf is UHF
    assert GHF_ghf is GHF
    print("OK: upstream-compat shims (pyscf.scf.{hf,uhf,ghf}) resolve correctly")


def test_panic_to_exception_class():
    """BIND-09: PyscfRsRuntimeError is a real exception class on _native."""
    from pyscf._native import PyscfRsRuntimeError  # type: ignore[attr-defined]

    assert issubclass(PyscfRsRuntimeError, Exception), (
        "PyscfRsRuntimeError must subclass Exception (BIND-09)"
    )
    print("OK: PyscfRsRuntimeError exists as Exception subclass (BIND-09)")


def test_pyscf_rs_error_overlay():
    """BIND-09 overlay: pyscf.PyscfRsError grafts .kind and .source_chain."""
    from pyscf import PyscfRsError

    e = PyscfRsError("msg", "ConvergenceFailure", ["nested error 1", "nested error 2"])
    assert e.kind == "ConvergenceFailure"
    assert e.source_chain == ["nested error 1", "nested error 2"]
    print("OK: PyscfRsError overlay grafts .kind + .source_chain (BIND-09)")


if __name__ == "__main__":
    test_module_imports()
    test_overlay_resolution()
    test_upstream_compat_shims()
    test_panic_to_exception_class()
    test_pyscf_rs_error_overlay()
    print("\nAll BIND-01/02/09 maturin smoke tests passed.")
    sys.exit(0)
