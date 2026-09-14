"""Phase 20 gate — the `pyscf.pbc` identity contract (20-CONTEXT §1.1).

The original Phase-20 gate ("an unmodified upstream `pyscf.pbc` script runs on
pyscf-rs") is satisfied by the `pkgutil.extend_path` passthrough with zero work
done, because every `pyscf.pbc.*` name resolves to the vendored upstream
Python tree. This test is the non-vacuous restatement: each public name the
overlay exports must be the SAME object as its `pyscf._native.pbc.*` binding,
in the Phase-3 form of `test_overlay_resolution.py:20`.

It was written and run BEFORE 20-09 bound anything, and failed there (recorded
in `.planning/phases/20-pbc-python-bindings/20-VERIFICATION.md`), so a no-op
cannot pass it.
"""
import importlib

import pytest

# (overlay module, native module, names) — the surface 20-09 … 20-15 bind.
IDENTITY = [
    ("pyscf.pbc.gto", "pyscf._native.pbc.gto", ["Cell", "M"]),
    ("pyscf.pbc.df", "pyscf._native.pbc.df", ["FFTDF", "AFTDF", "GDF", "MDF", "RSDF"]),
    ("pyscf.pbc.scf", "pyscf._native.pbc.scf", ["KRHF", "KUHF", "KROHF", "KGHF"]),
    ("pyscf.pbc.dft", "pyscf._native.pbc.dft", ["KRKS", "KUKS", "KROKS", "KGKS"]),
    ("pyscf.pbc.symm", "pyscf._native.pbc.symm", ["KPoints"]),
    ("pyscf.pbc.mp", "pyscf._native.pbc.mp", ["KMP2"]),
    ("pyscf.pbc.cc", "pyscf._native.pbc.cc", ["KRCCSD"]),
]

CASES = [(ov, nat, name) for ov, nat, names in IDENTITY for name in names]


@pytest.mark.parametrize("overlay,native,name", CASES, ids=[f"{o}.{n}" for o, _, n in CASES])
def test_pbc_overlay_name_is_native_object(overlay, native, name):
    nat_mod = importlib.import_module(native)
    ov_mod = importlib.import_module(overlay)
    nat_obj = getattr(nat_mod, name)
    ov_obj = getattr(ov_mod, name)
    assert ov_obj is nat_obj, (
        f"{overlay}.{name} = {ov_obj!r} (from {getattr(ov_mod, '__file__', '?')}), "
        f"expected the native binding {nat_obj!r}"
    )
