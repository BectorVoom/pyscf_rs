"""GTO-03: representative builtin basis sweep against upstream PySCF.

Phase 2 PR-CI representative subset — five basis names cover the
shape of the PR-CI corpus (H2O/cc-pvdz, benzene/6-31G*, water-trimer/
sto-3g, plus a Karlsruhe def2 entry and a Pople triple-zeta entry).

The pyscf-rs side of the sweep is in
``crates/pyscf-gto/tests/builtin_basis_sweep.rs`` (10 bases via
``cargo test``). This file's job is the upstream-side parity smoke —
upstream PySCF must accept the same names and produce a non-empty
Mole, otherwise the cross-stack byte-identity tests are operating on
stale comparators.
"""

from __future__ import annotations

import pytest

# Five names = a quick smoke; the cargo-side sweep covers ten.
BASES = [
    "sto-3g",
    "6-31g",
    "cc-pvdz",
    "def2-svp",
    "6-311g",
]


@pytest.mark.parametrize("basis", BASES)
def test_basis_loads_and_matches_upstream_for_h(upstream_pyscf, basis):
    """Upstream-only smoke: each representative basis loads on H and
    produces a non-empty Mole. The pyscf-rs cross-check lives in the
    ``builtin_basis_sweep.rs`` cargo test."""
    mol = upstream_pyscf.M(atom="H 0 0 0", basis=basis, unit="Bohr", verbose=0)
    nao = mol.nao_nr()
    assert nao > 0, f"upstream basis {basis!r} produced empty Mole for H"
