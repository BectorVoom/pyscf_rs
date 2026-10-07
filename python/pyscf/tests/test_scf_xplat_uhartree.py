"""Phase 3 SCF-13 cross-platform µHartree consistency (Pitfall 12 mitigation).

Covers: SCF-13 — Linux x86_64 + macOS aarch64 µHartree assertion on
H2O/cc-pVDZ. This file is the Python-side companion to the matrix-CI job
(plan 03-09 ships the GitHub Actions `xplat-uhartree` workflow); the
assertion runs on each platform and the matrix job's pass/fail is the
cross-platform contract.

Local-run behavior: this test passes whenever pyscf-rs's H2O/cc-pVDZ RHF
energy is within 1 µHartree of upstream PySCF's energy ON THE SAME
PLATFORM. CI then asserts that the energies on Linux x86_64 and macOS
aarch64 agree to 1 µHartree across platforms by running this same test
on both runners and comparing the captured energy values.
"""
from pyscf import scf

#: Upstream RHF energy, out of process (vendored oracle — see conftest.py).
ORACLE_RHF_ENERGY = """
import json
import sys

from pyscf import gto, scf

request = json.load(sys.stdin)
mol = gto.M(atom=request["atom"], basis=request["basis"], verbose=0)
mf = scf.RHF(mol).run()
payload = {"converged": bool(mf.converged), "e_tot": float(mf.e_tot)}
print("__PYSCF_RS_ORACLE__" + json.dumps(payload))
"""


def test_scf_xplat_uhartree_h2o_ccpvdz(h2o_mol, run_upstream):
    """H2O/cc-pVDZ RHF — same-platform |e_rs - e_up| < 1 µHartree."""
    mf_rs = scf.RHF(h2o_mol).run()
    assert mf_rs.converged

    up = run_upstream(ORACLE_RHF_ENERGY, {"atom": h2o_mol.atom, "basis": "cc-pvdz"})
    assert up["converged"]

    diff = abs(mf_rs.e_tot - up["e_tot"])
    assert diff < 1e-6, (
        f"SCF-13 same-platform |e_rs - e_up| = {diff:.3e} > 1 µHartree; "
        f"e_rs={mf_rs.e_tot:.12f} e_up={up['e_tot']:.12f}"
    )
    # Print the captured energy so CI workflow can scrape it for the
    # cross-platform diff (plan 03-09's xplat-uhartree job).
    print(f"SCF-13 H2O/cc-pVDZ e_tot (this platform) = {mf_rs.e_tot:.12f}")
