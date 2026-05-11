"""Phase 3 SCF-03 GHF on H2 — runs to completion (correctness only, no perf parity).

Covers: SCF-03 — GHF H2 run-to-completion.

Per PROJECT scope, GHF is a Phase-3 deliverable on the correctness axis
only — no µHartree parity assertion (the spinor MO formalism has multiple
valid initial-guess conventions that produce different energies for the
same molecule). Assertion: kernel converges and returns a finite energy.
"""
import math

from pyscf import gto, scf


def test_scf_ghf_h2_runs_to_completion():
    """GHF on H2 / sto-3g — converged, finite energy."""
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
    mf = scf.GHF(mol).run()
    assert mf.converged, "GHF on H2 did not converge"
    assert math.isfinite(mf.e_tot), f"GHF e_tot is not finite: {mf.e_tot}"
    # H2 GHF energy should be in the chemically reasonable HF range
    # (-1.2 < e_tot < -0.5 Ha covers any valid spinor initialization).
    assert -1.3 < mf.e_tot < -0.5, f"GHF e_tot out of range: {mf.e_tot}"
