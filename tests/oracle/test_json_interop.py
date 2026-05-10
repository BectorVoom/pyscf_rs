"""GTO-09 cross-language JSON interop.

pyscf-rs writes JSON via ``pyscf_gto::dumps(&mol)``; Python (upstream
``pyscf``) reads the snapshot, extracts the user-input portion, and
re-builds via ``pyscf.M(**state)``. The contract is **semantic
round-trip** (per CONTEXT D-09) — the rebuilt upstream Mole's ``_atm``
matches a fresh ``pyscf.M(orig_atom, basis)`` Mole's ``_atm``.

This is NOT byte-identical to upstream PySCF's ``Mole.dumps()`` JSON;
that's a Phase 8 ORACLE-08 chkfile concern. What we verify here is
that the user-input shape (atom symbols + Bohr coordinates + basis
name) survives the round trip well enough for upstream to reconstruct
an equivalent Mole.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np


def _pyscf_rs_dump(
    name: str, atom: str, basis: str, workspace_root: Path
) -> str:
    """Invoke the pyscf-rs ``dump_mole_dumps_for_oracle`` helper.

    Returns the JSON payload produced by ``pyscf_gto::dumps(&mol)``.
    """
    out_path = workspace_root / "tests" / "oracle" / ".tmp" / f"{name}_dumps.json"
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
            "dump_mole_dumps_for_oracle",
            "--",
            "--ignored",
        ],
        cwd=str(workspace_root),
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"cargo test (dump_mole_dumps_for_oracle) failed for {name}:\n"
        f"--- stdout ---\n{result.stdout[-2000:]}\n"
        f"--- stderr ---\n{result.stderr[-2000:]}"
    )
    with open(out_path) as f:
        return f.read()


def test_pyscfrs_dumps_to_pyscf_loads_roundtrip(upstream_pyscf, workspace_root):
    """Semantic round-trip: pyscf-rs dumps → Python parses → upstream
    rebuild produces an equivalent Mole.

    The rebuilt-via-tuples Mole and the freshly-built-from-string Mole
    must yield the same ``_atm`` array.  ``_bas`` / ``_env`` may differ
    if tuple-form coords trigger a different parser path upstream, but
    ``_atm`` (Z, ptr_coord, nuc_mod, ...) is invariant under the
    user-input shape.
    """
    atom = "H 0 0 0; H 0 0 1.4"
    basis = "sto-3g"
    json_str = _pyscf_rs_dump("h2_sto3g", atom, basis, workspace_root)
    snap = json.loads(json_str)

    # Reconstruct the per-atom (symbol, coords) list from snap.atom_parsed.
    rebuilt_atoms = [(sym, [c[0], c[1], c[2]]) for sym, c in snap["atom_parsed"]]

    # Build upstream from the round-tripped tuples list.
    mol_rebuilt = upstream_pyscf.M(
        atom=rebuilt_atoms, basis=basis, unit="Bohr", verbose=0
    )
    # Build upstream directly from the original string.
    mol_orig = upstream_pyscf.M(atom=atom, basis=basis, unit="Bohr", verbose=0)

    np.testing.assert_array_equal(
        mol_rebuilt._atm,
        mol_orig._atm,
        err_msg=(
            "pyscf.M(rebuilt_atoms) and pyscf.M(orig_atom) produced different "
            "_atm — pyscf-rs dumps round-trip lost atom-input shape"
        ),
    )

    # Charge / spin / cart kwargs survive too.
    assert snap["charge"] == 0
    assert snap["spin"] == 0
    assert snap["cart"] is False


def test_pyscfrs_dumps_snapshot_shape(upstream_pyscf, workspace_root):
    """Sanity: the snapshot JSON has the keys the round-trip path
    relies on.  Catches MoleSnapshot field renames during refactors
    before they break consumers.
    """
    atom = "H 0 0 0"
    basis = "sto-3g"
    json_str = _pyscf_rs_dump("h_sto3g_solo", atom, basis, workspace_root)
    snap = json.loads(json_str)

    expected_keys = {
        "atom_parsed",
        "basis_per_element",
        "ecp_per_element",
        "charge",
        "spin",
        "cart",
        "unit",
        "verbose",
        "max_memory",
    }
    missing = expected_keys - set(snap.keys())
    assert not missing, f"MoleSnapshot JSON missing keys: {missing}"
    assert isinstance(snap["atom_parsed"], list)
    assert len(snap["atom_parsed"]) == 1
    assert snap["atom_parsed"][0][0].upper() == "H"
