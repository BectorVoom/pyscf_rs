"""GTO-05 (evaluation half) — ``int1e_ecp`` byte-identity vs upstream PySCF.

Gap-closure plan 02-10. Closes the evaluation half of GTO-05: the
cintx-backed ``CintxEcpEngine`` (returned by ``pyscf_gto::ecp_engine()``
since 02-10) evaluates the Type-1 + Type-2 ECP projector integral on
Cu/LANL2DZ and must match upstream ``mol.intor("int1e_ecp")`` byte-equal
(within the 1e-10 chemical-accuracy oracle tolerance; the cintx-side
``safe_api_ecp_parity.rs`` already pins 1e-12 byte-identity against vendored
PySCF ``nr_ecp``).

This test is auto-skipped when upstream pyscf is not importable (see
``conftest.py::pytest_collection_modifyitems``); it is the release-oracle
counterpart to the always-on in-tree gate
``crates/pyscf-gto/tests/ecp_int1e_oracle.rs``.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np


def _int1e_ecp_pyscfrs(
    atom: str,
    basis: str,
    ecp: str,
    workspace_root: Path,
) -> np.ndarray:
    """Invoke the pyscf-rs ``dump_intor_for_oracle`` helper with an ECP.

    Returns the F-order-reshaped ``int1e_ecp`` matrix. Raises
    ``AssertionError`` on cargo failure.
    """
    out_path = workspace_root / "tests" / "oracle" / ".tmp" / "cu_lanl2dz_int1e_ecp.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "PYSCF_RS_ORACLE_ATOM": atom,
            "PYSCF_RS_ORACLE_BASIS": basis,
            "PYSCF_RS_ORACLE_ECP": ecp,
            "PYSCF_RS_ORACLE_INTOR": "int1e_ecp",
            "PYSCF_RS_ORACLE_OUT": str(out_path),
        }
    )
    result = subprocess.run(
        [
            "cargo",
            "test",
            "--features",
            "release-oracle-tests",
            "-p",
            "pyscf-gto",
            "--test",
            "dump_intor_for_oracle",
            "--",
            "--ignored",
        ],
        cwd=str(workspace_root),
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "cargo test (dump_intor_for_oracle, int1e_ecp) failed for Cu/LANL2DZ:\n"
        f"--- stdout ---\n{result.stdout[-2000:]}\n"
        f"--- stderr ---\n{result.stderr[-2000:]}"
    )
    with open(out_path) as f:
        data = json.load(f)
    return np.asarray(data["values"], dtype=np.float64).reshape(
        tuple(data["shape"]), order="F"
    )


def test_cu_lanl2dz_int1e_ecp_byte_equal(upstream_pyscf, workspace_root):
    """Cu/LANL2DZ ``int1e_ecp`` parity vs upstream PySCF."""
    atom, basis, ecp = "Cu 0 0 0", "lanl2dz", "lanl2dz"
    mol = upstream_pyscf.M(
        atom=atom, basis=basis, ecp=ecp, unit="Bohr", verbose=0
    )
    upstream = mol.intor("int1e_ecp")
    rs = _int1e_ecp_pyscfrs(atom, basis, ecp, workspace_root)
    np.testing.assert_allclose(
        rs,
        upstream,
        atol=1e-10,
        rtol=1e-10,
        err_msg="GTO-05 evaluation half: Cu/LANL2DZ int1e_ecp parity vs upstream",
    )
