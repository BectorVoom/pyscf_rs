"""Phase 3 SCF-04 C-DIIS iteration count vs upstream (±1 cycle).

Covers: SCF-04 — C-DIIS converges in upstream iteration count ±1.

Default DIIS in pyscf-rs (plan 03-04) is Pulay's C-DIIS, matching the
upstream PySCF default; the iteration count is expected to agree within
±1 cycle for the same molecule + same convergence threshold.
"""
from pyscf import scf

#: Upstream RHF cycle count, out of process (vendored oracle — see conftest.py).
ORACLE_RHF_CYCLES = """
import json
import sys

from pyscf import gto, scf

request = json.load(sys.stdin)
mol = gto.M(atom=request["atom"], basis=request["basis"], verbose=0)
mf = scf.RHF(mol).run()
payload = {"converged": bool(mf.converged), "cycles": int(mf.cycles)}
print("__PYSCF_RS_ORACLE__" + json.dumps(payload))
"""


def test_scf_cdiis_iteration_count_within_one(h2o_mol, run_upstream):
    mf_rs = scf.RHF(h2o_mol).run()
    assert mf_rs.converged, "pyscf-rs RHF did not converge"

    up = run_upstream(ORACLE_RHF_CYCLES, {"atom": h2o_mol.atom, "basis": "cc-pvdz"})
    assert up["converged"]

    # Both pyscf-rs and upstream report a non-negative integer cycle count.
    rs_cycles = int(mf_rs.cycles)
    up_cycles = int(up["cycles"])
    diff = abs(rs_cycles - up_cycles)
    assert diff <= 1, (
        f"C-DIIS iteration count drift: pyscf-rs converged in {rs_cycles}, "
        f"upstream in {up_cycles} — diff = {diff} (> 1 cycle = SCF-04 fail)"
    )
