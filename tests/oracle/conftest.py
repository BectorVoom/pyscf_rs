"""pytest harness for upstream-PySCF byte-identity tests (Phase 2 GTO-04).

Tests in this directory load both upstream ``pyscf`` (in-process, from the
vendored ``pyscf/`` checkout at the repo root) and the pyscf-rs CLI/library
output (via ``cargo test``-driven JSON dumps), then diff
``_atm`` / ``_bas`` / ``_env`` / ``ao_loc_nr`` / ``nao_nr`` byte-for-byte
against upstream.

Gated by the ``release-oracle`` Cargo profile in the calling Rust tests
(see Phase 1 D-08 + this repo's profile entry in Cargo.toml).

Prerequisites
-------------
The runner needs upstream PySCF + numpy + scipy importable. Install via::

    pip install -r tests/oracle/requirements.txt

If the dev box has no SSL-enabled Python (e.g., a stripped pyenv build), see
``docs/env-vars.md`` "Test setup" appendix for the manual installation path.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def workspace_root() -> Path:
    """Repo root (parent of tests/oracle/)."""
    here = Path(__file__).resolve().parent
    return here.parent.parent


@pytest.fixture(scope="session")
def upstream_pyscf_path(workspace_root: Path) -> Path:
    """Path to the vendored upstream pyscf/ checkout."""
    return workspace_root / "pyscf"


@pytest.fixture(scope="session")
def upstream_pyscf(workspace_root: Path):
    """Import upstream pyscf from this repo's vendored copy.

    Adds the repo root to ``sys.path`` so ``import pyscf`` resolves to the
    vendored checkout at ``<workspace_root>/pyscf/`` rather than any
    site-packages copy. Cached at session scope so subsequent tests reuse
    the same module object.
    """
    root_str = str(workspace_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    import pyscf  # noqa: WPS433  (deliberate runtime import after sys.path mutation)

    return pyscf


@pytest.fixture
def h2_sto3g(upstream_pyscf):
    """Tiny smoke fixture: H2 at 1.4 Bohr, STO-3G."""
    return upstream_pyscf.M(
        atom="H 0 0 0; H 0 0 1.4",
        basis="sto-3g",
        unit="Bohr",
    )


@pytest.fixture
def h2o_ccpvdz(upstream_pyscf):
    """Mid-size fixture: water in cc-pVDZ. Used by the byte-identity oracle
    once plan 02-04 ships make_env."""
    return upstream_pyscf.M(
        atom="O 0 0 0; H 0 0.7 0.6; H 0 -0.7 0.6",
        basis="cc-pvdz",
    )


@pytest.fixture(scope="session")
def basis_path(workspace_root: Path) -> str:
    """``PYSCF_BASIS_PATH`` value pointing at the vendored ``pyscf/gto/basis/``
    directory. Tests can set this on a subprocess env when invoking the Rust
    side."""
    candidate = workspace_root / "pyscf" / "gto" / "basis"
    return str(candidate)


def pytest_collection_modifyitems(config, items):
    """Skip the entire oracle suite if upstream pyscf isn't importable.

    The harness is gated by Python availability (numpy / pyscf / scipy from
    requirements.txt). Without them, mark every collected test as skipped
    with a clear reason rather than failing on import errors.
    """
    try:
        import pyscf  # noqa: F401
    except ImportError as exc:
        skip_marker = pytest.mark.skip(
            reason=(
                "upstream pyscf not importable "
                f"({exc}); install tests/oracle/requirements.txt"
            )
        )
        for item in items:
            item.add_marker(skip_marker)


# Honour PYSCF_BASIS_PATH if the test session sets it; otherwise leave the
# Rust-side resolver to fall back to its CARGO_MANIFEST_DIR walk-up.
if os.environ.get("PYSCF_BASIS_PATH"):
    os.environ["PYSCF_BASIS_PATH"] = os.environ["PYSCF_BASIS_PATH"]
