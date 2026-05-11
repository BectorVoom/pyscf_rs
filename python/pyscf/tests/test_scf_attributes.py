"""Phase 3 SCF-14 ≥ 30-attribute floor introspection from Python.

Covers: SCF-14 — ≥ 30-attribute floor on the SCF class is reachable from Python.

Plan 03-07 ships 31 #[getter] annotations on PyRHF alone (46 across
RHF/UHF/GHF combined). This test enumerates the floor and asserts that
every entry is reachable via `hasattr(mf, attr)`.
"""
from pyscf import scf


REQUIRED_ATTRS = [
    # Scalar SCF state (set by kernel()).
    "mo_coeff", "mo_energy", "mo_occ", "e_tot", "e_elec",
    "converged", "cycles",
    # Mol handle.
    "mol",
    # User-tunable params.
    "verbose", "chkfile", "max_memory",
    "direct_scf", "direct_scf_tol",
    "init_guess", "level_shift", "damp",
    "diis", "diis_space", "diis_start_cycle", "diis_damp", "diis_file",
    "max_cycle", "conv_tol", "conv_tol_grad",
    # DF + dispersion + extras.
    "with_df", "disp", "do_disp", "irrep_nelec", "nelec",
    "callback", "scf_summary",
]
# 31 attrs — meets the SCF-14 ≥ 30 floor.


def test_30_attribute_floor(h2o_mol):
    """Every REQUIRED_ATTR must be reachable via hasattr(mf, attr)."""
    mf = scf.RHF(h2o_mol)
    missing = [a for a in REQUIRED_ATTRS if not hasattr(mf, a)]
    assert not missing, f"missing attributes: {missing}"
    assert len(REQUIRED_ATTRS) >= 30, (
        f"REQUIRED_ATTRS must list ≥ 30 entries, has {len(REQUIRED_ATTRS)}"
    )


def test_scf_thirty_attribute_floor_introspectable(h2o_mol):
    """Alias matching the plan 03-02 stub name (kept for grep continuity)."""
    test_30_attribute_floor(h2o_mol)
