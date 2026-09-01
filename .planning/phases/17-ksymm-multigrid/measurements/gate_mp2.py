#!/usr/bin/env python
"""Task 4 -- the post-SCF floors (17-09's gate). mp/test/test_ksym.py's He
cell at its own settings, then si at [2,2,2]. KRCCSD is recorded as
unmeasured because crates/pyscf-pbc-cc/src is still a 13-line stub (Phase 16
has not shipped) as of this measurement.
"""
import sys
import numpy as np
import pyscf
assert pyscf.__version__ == "2.12.1", pyscf.__version__
from pyscf.pbc import gto, scf, mp

print("=== He cell, test_ksym.py's own settings ===")
L = 2.
He = gto.Cell()
He.verbose = 0
He.a = np.eye(3)*L
He.atom = [['He', (L/2+0., L/2+0., L/2+0.)]]
He.basis = {'He': [[0, (4.0, 1.0)], [0, (1.0, 1.0)]]}
He.space_group_symmetry = True
He.build()

nk = [2,2,2]
kpts0 = He.make_kpts(nk)
kmf0 = scf.KRHF(He, kpts0, exxdiv=None).density_fit()
kmf0.kernel()
kmp2ref = mp.KMP2(kmf0)
kmp2ref.kernel()

kpts = He.make_kpts(nk, space_group_symmetry=True, time_reversal_symmetry=True)
kmf = scf.KRHF(He, kpts, exxdiv=None).density_fit()
kmf.kernel()
kmp2 = mp.KMP2(kmf)
kmp2.kernel()

d_ecorr = abs(kmp2.e_corr - kmp2ref.e_corr)
print(f"e_corr (ksymm)  = {kmp2.e_corr!r}")
print(f"e_corr (full BZ) = {kmp2ref.e_corr!r}")
print(f"|d e_corr| = {d_ecorr:.6e}")

dm1ref = kmp2ref.make_rdm1()
dm1 = kmp2.make_rdm1()
worst = 0.
for i, k in enumerate(kpts.ibz2bz):
    err = np.amax(np.absolute(dm1[i] - dm1ref[k]))
    worst = max(worst, err)
print(f"rdm1 max residual (ibz vs corresponding full-BZ) = {worst:.6e}")

# Also compare the SCF energies for context (upstream's ordering claim:
# post-SCF tighter than SCF).
d_escf = abs(kmf.e_tot - kmf0.e_tot)
print(f"|d E_scf| (same run) = {d_escf:.6e}")

print()
print("=== si at [2,2,2], gth-szv/gth-pade, KRHF(exxdiv=None) + KMP2, density_fit ===")
half = 5.4306/2; q = 5.4306/4
si = gto.Cell()
si.atom = f"Si 0. 0. 0.\nSi {q} {q} {q}"
si.a = [[0.,half,half],[half,0.,half],[half,half,0.]]
si.basis = 'gth-szv'; si.pseudo = 'gth-pade'
si.space_group_symmetry = True
si.verbose = 0
si.build()

nk2 = [2,2,2]
kpts0_si = si.make_kpts(nk2)
kmf0_si = scf.KRHF(si, kpts0_si, exxdiv=None).density_fit()
kmf0_si.kernel()
kmp2ref_si = mp.KMP2(kmf0_si)
kmp2ref_si.kernel()

kpts_si = si.make_kpts(nk2, space_group_symmetry=True, time_reversal_symmetry=True)
kmf_si = scf.KRHF(si, kpts_si, exxdiv=None).density_fit()
kmf_si.kernel()
kmp2_si = mp.KMP2(kmf_si)
kmp2_si.kernel()

d_ecorr_si = abs(kmp2_si.e_corr - kmp2ref_si.e_corr)
print(f"e_corr (ksymm)  = {kmp2_si.e_corr!r}")
print(f"e_corr (full BZ) = {kmp2ref_si.e_corr!r}")
print(f"|d e_corr| = {d_ecorr_si:.6e}")

dm1ref_si = kmp2ref_si.make_rdm1()
dm1_si = kmp2_si.make_rdm1()
worst_si = 0.
for i, k in enumerate(kpts_si.ibz2bz):
    err = np.amax(np.absolute(dm1_si[i] - dm1ref_si[k]))
    worst_si = max(worst_si, err)
print(f"rdm1 max residual (ibz vs corresponding full-BZ) = {worst_si:.6e}")
d_escf_si = abs(kmf_si.e_tot - kmf0_si.e_tot)
print(f"|d E_scf| (same run) = {d_escf_si:.6e}")

print()
print("=== ordering check: post-SCF tighter than SCF ===")
print(f"He:  |d e_corr|={d_ecorr:.3e}  vs  |d E_scf|={d_escf:.3e}  ->  "
      f"post-SCF tighter: {d_ecorr < d_escf}")
print(f"si:  |d e_corr|={d_ecorr_si:.3e}  vs  |d E_scf|={d_escf_si:.3e}  ->  "
      f"post-SCF tighter: {d_ecorr_si < d_escf_si}")

print()
print("=== KRCCSD (test_kccsd_ksymm.py) ===")
print("unmeasured, Phase 16 not shipped -- crates/pyscf-pbc-cc/src is a "
      "13-line stub (lib.rs + error.rs) as of this measurement (2026-09-01).")
