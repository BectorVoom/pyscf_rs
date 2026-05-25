"""Shared pytest fixtures for pyscf-rs Phase 3 SCF tests.

Plan 03-10 fills the Wave-0 skip-stubs with real Mole-construction bodies.
Fixtures use the pyscf-rs overlay (`pyscf.gto.M`) for the rs side; upstream
PySCF is loaded via importlib under a different module name so both
libraries are importable in the same Python process (RESEARCH §Validation
Architecture lines 1313-1357 — in-process comparison is faster than
subprocess isolation and good enough for the µHartree contract because
the Rust kernel cannot mutate upstream Python state).
"""
import importlib.util
import json
import os
import subprocess
import sys
import textwrap

import pytest


def _load_upstream():
    """Load upstream PySCF (the original `pyscf/` tree under repo root)
    under the namespace `_upstream_pyscf` so it coexists with the overlay.

    The maturin install puts `python/pyscf/` on sys.path ahead of the
    upstream tree, so `import pyscf` resolves to the overlay. We bypass
    that by loading upstream's `__init__.py` via importlib directly.
    """
    # <repo>/python/pyscf/tests/conftest.py → walk up 3 dirs to <repo>
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    upstream_init = os.path.join(repo_root, "pyscf", "__init__.py")
    if not os.path.exists(upstream_init):
        pytest.skip(f"upstream pyscf not found at {upstream_init}")

    # Cache hit if already loaded.
    if "_upstream_pyscf" in sys.modules:
        return sys.modules["_upstream_pyscf"]

    spec = importlib.util.spec_from_file_location(
        "_upstream_pyscf",
        upstream_init,
        submodule_search_locations=[os.path.dirname(upstream_init)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_upstream_pyscf"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def upstream():
    """Upstream PySCF imported as `_upstream_pyscf` — coexists with overlay."""
    return _load_upstream()


@pytest.fixture
def upstream_rhf_energy():
    """Return upstream RHF energy, using a separate interpreter when provided."""

    def calculate(atom: str, basis: str):
        upstream_python = os.environ.get("PYSCF_RS_UPSTREAM_PYTHON")
        if upstream_python:
            code = textwrap.dedent(
                """
                import json
                import sys

                from pyscf import gto, scf

                request = json.load(sys.stdin)
                # verbose=0 silences PySCF's "converged SCF energy = ..." banner,
                # which otherwise pollutes stdout ahead of the JSON result. The
                # sentinel prefix makes extraction robust against any residual
                # C-level stdout (libcint/libxc) regardless of verbosity.
                mol = gto.M(atom=request["atom"], basis=request["basis"], verbose=0)
                mf = scf.RHF(mol).run()
                payload = {"converged": bool(mf.converged), "e_tot": float(mf.e_tot)}
                print("__PYSCF_RS_ORACLE__" + json.dumps(payload))
                """
            )
            proc = subprocess.run(
                [upstream_python, "-c", code],
                input=json.dumps({"atom": atom, "basis": basis}),
                text=True,
                capture_output=True,
                check=False,
            )
            if proc.returncode != 0:
                pytest.fail(
                    "upstream PySCF subprocess failed\n"
                    f"stdout:\n{proc.stdout}\n"
                    f"stderr:\n{proc.stderr}"
                )
            marker = "__PYSCF_RS_ORACLE__"
            line = next(
                (ln for ln in proc.stdout.splitlines() if ln.startswith(marker)),
                None,
            )
            if line is None:
                pytest.fail(
                    "upstream PySCF subprocess emitted no oracle result\n"
                    f"stdout:\n{proc.stdout}\n"
                    f"stderr:\n{proc.stderr}"
                )
            return json.loads(line[len(marker):])

        upstream = _load_upstream()
        mol = upstream.gto.M(atom=atom, basis=basis, verbose=0)
        mf = upstream.scf.RHF(mol).run()
        return {"converged": bool(mf.converged), "e_tot": float(mf.e_tot)}

    return calculate


@pytest.fixture
def h2o_sto3g_mol():
    """H2O / STO-3G in the pyscf-rs overlay for fast native SCF smoke gates."""
    from pyscf import gto

    return gto.M(
        atom="O 0.0 0.0 0.0; H 0.757 0.587 0.0; H -0.757 0.587 0.0",
        basis="sto-3g",
    )


@pytest.fixture
def h2o_mol():
    """H2O / cc-pVDZ in the pyscf-rs overlay.

    Used as the primary SCF test fixture (SCF-01 / SCF-04 / SCF-07 / SCF-10).
    Built via `pyscf.gto.M()`; the overlay routes to pyscf-rs's pyscf-gto
    (Phase 2) which implements the same `M(atom, basis)` signature as upstream.
    """
    from pyscf import gto  # overlay route → pyscf-rs's pyscf-gto (Phase 2)
    return gto.M(
        atom="O 0.0 0.0 0.0; H 0.757 0.587 0.0; H -0.757 0.587 0.0",
        basis="cc-pvdz",
    )


@pytest.fixture
def benzene_mol():
    """Benzene / 6-31G* — SCF-01 secondary corpus entry."""
    from pyscf import gto
    return gto.M(
        atom="""
            C  0.0000  1.3970  0.0000
            C  1.2099  0.6985  0.0000
            C  1.2099 -0.6985  0.0000
            C  0.0000 -1.3970  0.0000
            C -1.2099 -0.6985  0.0000
            C -1.2099  0.6985  0.0000
            H  0.0000  2.4810  0.0000
            H  2.1486  1.2405  0.0000
            H  2.1486 -1.2405  0.0000
            H  0.0000 -2.4810  0.0000
            H -2.1486 -1.2405  0.0000
            H -2.1486  1.2405  0.0000
        """,
        basis="6-31g*",
    )


@pytest.fixture
def water_trimer_mol():
    """Water trimer / cc-pVDZ — chkfile round-trip fixture (ORACLE-08)."""
    from pyscf import gto
    return gto.M(
        atom="""
            O  -1.4220  -0.7060  0.0000
            H  -1.4220  -0.1390 -0.8060
            H  -0.5340  -1.0370  0.0000
            O   1.4220  -0.7060  0.0000
            H   0.5340  -1.0370  0.0000
            H   2.0220  -0.1390 -0.8060
            O   0.0000   1.4120  0.0000
            H  -0.6000   1.7430  0.8060
            H   0.6000   1.7430  0.8060
        """,
        basis="cc-pvdz",
    )
