"""GTO-04: byte-identity of ``_atm`` / ``_bas`` / ``_env`` /
``ao_loc_nr`` / ``nao_nr`` vs upstream PySCF.

Per Phase 2 success criterion 1 (ROADMAP). Pitfall 17 mitigation: the
``ao_loc_nr`` byte-equal assertion catches off-by-one drift in basis
indexing — a single-shell off-by-one would corrupt every AO index from
that shell onward.

Strategy: each test invokes the pyscf-rs ``dump_arrays_for_oracle``
integration test via ``cargo test --features release-oracle-tests``,
which writes the molecule's flat arrays to a temp JSON. Python then
reads the JSON and ``numpy.testing.assert_array_equal``s against the
upstream ``pyscf.M(...)`` arrays.

The PR-CI corpus (3 fixtures) covers:
  * H2O / cc-pvdz — the keystone GTO-04 byte-identity (3 shells, 24 AOs)
  * benzene / 6-31G* — 12 atoms, polarisation functions on C and H
  * water-trimer / sto-3g — 9 atoms, 21 AOs, exercises the per-element
    map dispatch on a non-trivial basis-set re-use pattern
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

PR_CI_FIXTURES = [
    (
        "h2o_ccpvdz",
        "O 0 0 0; H 0 0.7 0.6; H 0 -0.7 0.6",
        "cc-pvdz",
    ),
    (
        "benzene_631gs",
        # Benzene at experimental geometry (D6h, C-C ~1.397 Bohr is too
        # short — used here as a fixture only; the byte-identity test
        # is geometry-agnostic so any non-degenerate geometry works).
        "C 0 1.397 0; C 1.21 0.6985 0; C 1.21 -0.6985 0; "
        "C 0 -1.397 0; C -1.21 -0.6985 0; C -1.21 0.6985 0; "
        "H 0 2.481 0; H 2.149 1.2405 0; H 2.149 -1.2405 0; "
        "H 0 -2.481 0; H -2.149 -1.2405 0; H -2.149 1.2405 0",
        "6-31g*",
    ),
    (
        "water_trimer_sto3g",
        "O 0 0 0; H 0 0.7 0.6; H 0 -0.7 0.6; "
        "O 3 0 0; H 3.7 0.6 0; H 2.3 0.6 0; "
        "O -3 0 0; H -3.7 0.6 0; H -2.3 0.6 0",
        "sto-3g",
    ),
]


def _upstream_arrays(upstream_pyscf, atom: str, basis: str) -> dict:
    mol = upstream_pyscf.M(atom=atom, basis=basis, unit="Bohr", verbose=0)
    return {
        "_atm": np.asarray(mol._atm, dtype=np.int32).flatten(),
        "_bas": np.asarray(mol._bas, dtype=np.int32).flatten(),
        "_env": np.asarray(mol._env, dtype=np.float64),
        "ao_loc_nr": np.asarray(mol.ao_loc_nr(), dtype=np.int32),
        "nao_nr": int(mol.nao_nr()),
    }


def _pyscf_rs_arrays(name: str, atom: str, basis: str, workspace_root: Path) -> dict:
    """Invoke the pyscf-rs ``dump_arrays_for_oracle`` integration test.

    Returns the parsed JSON arrays. Raises ``AssertionError`` on cargo
    failure with the captured stderr trimmed to 2KB.
    """
    out_path = workspace_root / "tests" / "oracle" / ".tmp" / f"{name}_pyscfrs.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "PYSCF_RS_ORACLE_ATOM": atom,
            "PYSCF_RS_ORACLE_BASIS": basis,
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
            "dump_arrays_for_oracle",
            "--",
            "--ignored",
            "--nocapture",
        ],
        cwd=str(workspace_root),
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"cargo test (dump_arrays_for_oracle) failed for fixture {name}:\n"
        f"--- stdout ---\n{result.stdout[-2000:]}\n"
        f"--- stderr ---\n{result.stderr[-2000:]}"
    )
    with open(out_path) as f:
        data = json.load(f)
    return {
        "_atm": np.asarray(data["_atm"], dtype=np.int32),
        "_bas": np.asarray(data["_bas"], dtype=np.int32),
        "_env": np.asarray(data["_env"], dtype=np.float64),
        "ao_loc_nr": np.asarray(data["ao_loc_nr"], dtype=np.int32),
        "nao_nr": int(data["nao_nr"]),
    }


@pytest.mark.parametrize("name,atom,basis", PR_CI_FIXTURES)
def test_atm_bas_env_byte_for_byte(
    upstream_pyscf, workspace_root, name, atom, basis
):
    """Phase 2 success criterion #1 — byte-identical flat arrays."""
    u = _upstream_arrays(upstream_pyscf, atom, basis)
    r = _pyscf_rs_arrays(name, atom, basis, workspace_root)
    np.testing.assert_array_equal(
        r["_atm"], u["_atm"], err_msg=f"_atm mismatch for {name}"
    )
    np.testing.assert_array_equal(
        r["_bas"], u["_bas"], err_msg=f"_bas mismatch for {name}"
    )
    # _env: f64 — bit-equal under release-oracle profile (Phase 1 D-08
    # FMA-free profile guarantees deterministic floating-point ops in
    # the make_env normalisation path).
    np.testing.assert_array_equal(
        r["_env"], u["_env"], err_msg=f"_env mismatch for {name}"
    )


@pytest.mark.parametrize("name,atom,basis", PR_CI_FIXTURES)
def test_ao_loc_nr_byte_for_byte(
    upstream_pyscf, workspace_root, name, atom, basis
):
    """Pitfall 17 mitigation: off-by-one basis indexing → ao_loc_nr drift.

    A single shell with the wrong AO count poisons every subsequent
    cumulative offset. Byte-identity to upstream is the cheapest signal.
    """
    u = _upstream_arrays(upstream_pyscf, atom, basis)
    r = _pyscf_rs_arrays(name, atom, basis, workspace_root)
    np.testing.assert_array_equal(
        r["ao_loc_nr"],
        u["ao_loc_nr"],
        err_msg=(
            f"ao_loc_nr mismatch for {name} "
            "(Pitfall 17 — off-by-one in basis indexing)"
        ),
    )
    assert r["nao_nr"] == u["nao_nr"], (
        f"nao_nr mismatch for {name}: pyscf-rs={r['nao_nr']} vs upstream={u['nao_nr']}"
    )
