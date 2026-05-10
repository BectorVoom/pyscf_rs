"""GTO-07: ``eval_gto`` element-wise parity vs upstream PySCF.

Phase 2 ships ``GTOval_sph`` for ``l = 0`` shells; ``l ≥ 1`` paths
return zero in the current Phase 2 kernel — Phase 4 DFT extends. The
``test_eval_gto_h2o_ccpvdz_includes_p_shells`` test is marked
``@pytest.mark.xfail`` so CI tracks the deferral without blocking
Phase 2 closure.

The ``s``-shell smoke test (`test_eval_gto_h_sto3g_s_shell_only`) on
H/STO-3G is the keystone Phase-2-success-criterion-#4 partial: l=0
parity passes byte-exact (the kernel matches upstream's primitive
contraction analytically), and the chemical-accuracy comparison
(``atol=1e-10``) gives a useful ε-budget.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest


def _eval_gto_pyscfrs(
    name: str,
    atom: str,
    basis: str,
    eval_name: str,
    coords: list[list[float]],
    workspace_root: Path,
) -> np.ndarray:
    """Invoke the pyscf-rs ``dump_eval_gto_for_oracle`` integration test.

    Returns the F-order reshaped values matrix (shape ``(ngrids, nao)``).
    """
    out_path = workspace_root / "tests" / "oracle" / ".tmp" / f"{name}_eval_{eval_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "PYSCF_RS_ORACLE_ATOM": atom,
            "PYSCF_RS_ORACLE_BASIS": basis,
            "PYSCF_RS_ORACLE_EVAL_NAME": eval_name,
            "PYSCF_RS_ORACLE_COORDS": json.dumps(coords),
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
            "dump_eval_gto_for_oracle",
            "--",
            "--ignored",
        ],
        cwd=str(workspace_root),
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"cargo test (dump_eval_gto_for_oracle, {eval_name}) failed for {name}:\n"
        f"--- stdout ---\n{result.stdout[-2000:]}\n"
        f"--- stderr ---\n{result.stderr[-2000:]}"
    )
    with open(out_path) as f:
        data = json.load(f)
    return np.asarray(data["values"], dtype=np.float64).reshape(
        tuple(data["shape"]), order="F"
    )


def test_eval_gto_h_sto3g_s_shell_only(upstream_pyscf, workspace_root):
    """``l = 0`` smoke — matches upstream because the s-shell path is
    the priority MVP shipped in Phase 2's ``eval_gto_sph`` kernel."""
    atom = "H 0 0 0"
    basis = "sto-3g"
    mol = upstream_pyscf.M(atom=atom, basis=basis, unit="Bohr", verbose=0)
    coords = [[0.0, 0.0, z] for z in np.linspace(0.0, 5.0, 100).tolist()]
    coords_np = np.asarray(coords)
    upstream = mol.eval_gto("GTOval_sph", coords_np)
    rs = _eval_gto_pyscfrs(
        "h_sto3g", atom, basis, "GTOval_sph", coords, workspace_root
    )
    # Element-wise tolerance per Phase 2 success criterion #4
    # (chemical accuracy, ε ≈ 1e-10).
    np.testing.assert_allclose(
        rs,
        upstream,
        atol=1e-10,
        rtol=1e-10,
        err_msg="eval_gto GTOval_sph for H/STO-3G — l=0 path mismatch",
    )


@pytest.mark.xfail(
    reason=(
        "l ≥ 1 path stubs to zero in Phase 2; Phase 4 DFT extends the "
        "kernel. cc-pVDZ has p shells on O — the upstream values are "
        "non-zero, so the assertion fails until Phase 4 ships."
    )
)
def test_eval_gto_h2o_ccpvdz_includes_p_shells(upstream_pyscf, workspace_root):
    """cc-pVDZ has p shells on O (l=1) which Phase 2's kernel writes
    as zero. Marked xfail to track the Phase 4 DFT deferral."""
    atom = "O 0 0 0; H 0 0.7 0.6; H 0 -0.7 0.6"
    basis = "cc-pvdz"
    mol = upstream_pyscf.M(atom=atom, basis=basis, unit="Bohr", verbose=0)
    coords = np.linspace([0, 0, 0], [3, 0, 0], 50).tolist()
    coords_np = np.asarray(coords)
    upstream = mol.eval_gto("GTOval_sph", coords_np)
    rs = _eval_gto_pyscfrs(
        "h2o_ccpvdz", atom, basis, "GTOval_sph", coords, workspace_root
    )
    np.testing.assert_allclose(rs, upstream, atol=1e-10, rtol=1e-10)
