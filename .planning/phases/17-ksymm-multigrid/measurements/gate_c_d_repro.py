#!/usr/bin/env python
"""Task 3's run-to-run / thread-count spread, isolated into its own cheap
script (si, KRHF, FFTDF, gamma-centred, default mesh) after the combined
gate_c_d_part3.py run was cut short by lif's natural [81,81,81] mesh cost.
"""
import sys, os
import numpy as np
import pyscf
assert pyscf.__version__ == "2.12.1", pyscf.__version__
from pyscf.pbc import gto, scf

half = 5.4306/2; q = 5.4306/4
def build():
    c = gto.Cell()
    c.atom = f"Si 0. 0. 0.\nSi {q} {q} {q}"
    c.a = [[0.,half,half],[half,0.,half],[half,half,0.]]
    c.basis = 'gth-szv'; c.pseudo = 'gth-pade'
    c.verbose = 0
    c.build()
    return c

for threads in [1, 8]:
    os.environ['OMP_NUM_THREADS'] = str(threads)
    os.environ['RAYON_NUM_THREADS'] = str(threads)
    for run_idx in range(2):
        cell = build()
        kpts = cell.make_kpts([2,2,2], with_gamma_point=True)
        mf = scf.KRHF(cell, kpts)
        mf.conv_tol = 1e-11
        mf.chkfile = None
        mf.kernel()
        print(f"threads={threads} run={run_idx} e_tot={mf.e_tot!r} converged={mf.converged}")
        sys.stdout.flush()
