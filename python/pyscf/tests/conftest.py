"""Shared pytest fixtures for pyscf-rs Phase 3 SCF tests.

Plan 03-10 fills the Wave-0 skip-stubs with real Mole-construction bodies.
Fixtures use the pyscf-rs overlay (`pyscf.gto.M`) for the rs side; upstream
PySCF runs in a SEPARATE interpreter (carryover 20-molecular-python-suite-drift
item 1 — the in-process importlib loader is retired).

Why out-of-process: the overlay IS the `pyscf` package in this process
(molecular `_passthrough.py` fallthrough included), so executing the vendored
`pyscf/__init__.py` in-process binds the overlay's native `M`/`RHF`/`UHF`
and any "oracle" built from it compares pyscf-rs against itself (a vacuous
pass; guarded since 20-19 A, now removed outright). The subprocess runs with
`PYTHONPATH=<repo root>` from a neutral cwd, so `import pyscf` resolves to
the vendored 2.12.1 tree — asserted on every run — and never to site-packages
2.14.0 or the overlay.
"""
import json
import os
import subprocess
import sys
import tempfile
import textwrap

import pytest

#: Sentinel prefix marking the oracle payload line (C-level stdout from
#: libcint/libxc can precede it regardless of verbosity).
ORACLE_MARKER = "__PYSCF_RS_ORACLE__"

#: Hard cap per upstream evaluation (benzene/6-31G* is the slowest caller).
UPSTREAM_TIMEOUT = 1500


def _repo_root():
    # <repo>/python/pyscf/tests/conftest.py → walk up 3 dirs to <repo>
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", ".."))


def upstream_eval(script, request):
    """Run `script` against the VENDORED upstream PySCF and return its payload.

    `script` reads its JSON `request` from stdin and prints
    `ORACLE_MARKER + json.dumps(payload)` to stdout. The interpreter is
    `PYSCF_RS_UPSTREAM_PYTHON` when set (CI free-threaded escape hatch),
    else `sys.executable`; `PYTHONPATH=<repo root>` from a neutral cwd pins
    `import pyscf` to the vendored 2.12.1 tree, and a preamble asserts the
    version so a mis-resolved import fails loudly instead of comparing
    against the wrong PySCF.
    """
    interpreter = os.environ.get("PYSCF_RS_UPSTREAM_PYTHON") or sys.executable
    root = _repo_root()
    existing = os.environ.get("PYTHONPATH", "")
    env = dict(os.environ)
    env["PYTHONPATH"] = root + (os.pathsep + existing if existing else "")
    preamble = textwrap.dedent(
        """
        import pyscf
        assert pyscf.__version__ == "2.12.1", (
            "upstream oracle resolved PySCF %s from %s, expected vendored 2.12.1"
            % (pyscf.__version__, pyscf.__file__)
        )
        """
    )
    # Callers often indent the script inside the call — normalize once here.
    script = textwrap.dedent(script)
    try:
        proc = subprocess.run(
            [interpreter, "-c", preamble + script],
            input=json.dumps(request),
            text=True,
            capture_output=True,
            check=False,
            timeout=UPSTREAM_TIMEOUT,
            cwd=tempfile.gettempdir(),
            env=env,
        )
    except subprocess.TimeoutExpired as e:
        pytest.fail(f"upstream PySCF subprocess timed out after {UPSTREAM_TIMEOUT}s: {e}")
    if proc.returncode != 0:
        pytest.fail(
            "upstream PySCF subprocess failed\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    line = next(
        (ln for ln in proc.stdout.splitlines() if ln.startswith(ORACLE_MARKER)),
        None,
    )
    if line is None:
        pytest.fail(
            "upstream PySCF subprocess emitted no oracle result\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return json.loads(line[len(ORACLE_MARKER):])


@pytest.fixture
def run_upstream():
    """The out-of-process vendored oracle: `run_upstream(script, request)`."""

    return upstream_eval


@pytest.fixture
def upstream_rhf_energy():
    """Return upstream RHF energy via the out-of-process vendored oracle."""

    def calculate(atom: str, basis: str):
        return upstream_eval(
            textwrap.dedent(
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
            ),
            {"atom": atom, "basis": basis},
        )

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
