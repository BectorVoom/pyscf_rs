"""F-03 T5 — `Mole.intor_spinor` PyO3 surface + byte-identity vs upstream PySCF.

Exercises the complex `complex128` spinor-integral binding
(`crates/pyscf-py/src/gto.rs::intor_spinor`) through the overlay `pyscf.gto`:

  * structural contract — `complex128` dtype, F-order, shape `[n2c,n2c]` (1e) /
    `[n2c;4]` (2e), Hermiticity, ERI permutation symmetry;
  * byte-identity to upstream `mol.intor("…_spinor")` at atol 1e-10 on
    STO-3G (1e ovlp/kin/nuc + int2e).

Limitation (cintx): the cart→spinor transform is wired for SEGMENTED bases
only (`nctr==1`); general contraction (cc-pVDZ, 6-31g valence as nctr>1) raises.
For multi-shell-same-l bases the integrals are eigenvalue-identical to upstream
but the global AO ordering differs — tracked in the F-03 plan, not asserted here.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyscf import gto

ATOL = 1e-10
H2 = "H 0 0 0; H 0 0 1.4"
H2O = "O 0 0 0; H 0 1.4 1.1; H 0 -1.4 1.1"


def _upstream_or_skip():
    """Load upstream PySCF (vendored ``<repo>/pyscf``) under a private module
    name, or skip if its compiled C-libs aren't importable in this interpreter
    (e.g. a minimal overlay venv). In a full env (CI) the byte-identity
    assertions below run for real — they are confirmed to pass to <=2e-14
    (see the F-03 plan / commit notes for the manual cross-venv verification)."""
    import importlib.util
    import os
    import sys

    try:
        here = os.path.abspath(os.path.dirname(__file__))
        repo_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
        if "_upstream_pyscf" in sys.modules:
            return sys.modules["_upstream_pyscf"]
        init = os.path.join(repo_root, "pyscf", "__init__.py")
        spec = importlib.util.spec_from_file_location(
            "_upstream_pyscf", init, submodule_search_locations=[os.path.dirname(init)]
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_upstream_pyscf"] = mod
        spec.loader.exec_module(mod)
        import importlib

        mod.gto = importlib.import_module("_upstream_pyscf.gto")
        return mod
    except Exception as exc:  # noqa: BLE001 — any import/C-lib failure → skip
        sys.modules.pop("_upstream_pyscf", None)
        pytest.skip(f"upstream PySCF not importable here: {exc}")


def test_intor_spinor_structural_contract():
    """complex128, F-order, shape, Hermiticity, real-positive overlap diagonal."""
    mol = gto.M(atom=H2, basis="sto-3g", unit="Bohr")
    n2c = mol.nao_2c()
    assert n2c == 2 * mol.nao_nr()

    s = mol.intor_spinor("int1e_ovlp_spinor")
    assert s.dtype == np.complex128
    assert s.shape == (n2c, n2c)
    assert s.flags["F_CONTIGUOUS"]  # matches upstream order='F'
    assert np.max(np.abs(s - s.conj().T)) < ATOL  # Hermitian
    assert np.allclose(np.diag(s).imag, 0.0)
    assert np.all(np.diag(s).real > 0.0)

    eri = mol.intor_spinor("int2e_spinor")
    assert eri.dtype == np.complex128
    assert eri.shape == (n2c, n2c, n2c, n2c)
    assert np.all(np.isfinite(eri))
    # particle exchange (ij|kl) == (kl|ij); conjugation (ij|kl) == conj(ji|lk)
    assert np.max(np.abs(eri - np.transpose(eri, (2, 3, 0, 1)))) < ATOL
    assert np.max(np.abs(eri - np.transpose(eri, (1, 0, 3, 2)).conj())) < ATOL


def test_intor_spinor_name_normalisation():
    """A bare operator name routes to the spinor path identically."""
    mol = gto.M(atom=H2, basis="sto-3g", unit="Bohr")
    np.testing.assert_array_equal(
        mol.intor_spinor("int1e_ovlp"), mol.intor_spinor("int1e_ovlp_spinor")
    )


@pytest.mark.parametrize("op", ["int1e_ovlp_spinor", "int1e_kin_spinor", "int1e_nuc_spinor"])
def test_intor_spinor_1e_byte_identity_vs_upstream(op):
    """1e spinor integrals byte-match upstream PySCF on H2O/STO-3G (atol 1e-10)."""
    upstream = _upstream_or_skip()
    ours = gto.M(atom=H2O, basis="sto-3g", unit="Bohr").intor_spinor(op)
    up = upstream.gto.M(atom=H2O, basis="sto-3g", unit="Bohr").intor(op)
    assert ours.shape == up.shape
    err = np.max(np.abs(ours - up))
    assert err < ATOL, f"{op}: max|ours-upstream| = {err:.3e}"


def test_int2e_spinor_byte_identity_vs_upstream():
    """2e spinor ERIs byte-match upstream PySCF on H2/STO-3G (atol 1e-10)."""
    upstream = _upstream_or_skip()
    ours = gto.M(atom=H2, basis="sto-3g", unit="Bohr").intor_spinor("int2e_spinor")
    up = upstream.gto.M(atom=H2, basis="sto-3g", unit="Bohr").intor("int2e_spinor")
    assert ours.shape == up.shape
    err = np.max(np.abs(ours - up))
    assert err < ATOL, f"int2e_spinor: max|ours-upstream| = {err:.3e}"


def test_intor_spinor_general_contraction_raises():
    """General contraction (nctr>1) is unsupported by cintx — must raise cleanly."""
    mol = gto.M(atom="Ne 0 0 0", basis="cc-pvdz", unit="Bohr")
    with pytest.raises(Exception):
        mol.intor_spinor("int1e_ovlp_spinor")
