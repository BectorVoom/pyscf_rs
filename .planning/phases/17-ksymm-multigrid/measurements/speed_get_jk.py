#!/usr/bin/env python
"""Task 6.1 -- get_jk at nkpts vs nkpts_ibz (D-PBC-26, 17-CONTEXT §8).

si [4,4,4] (nkpts=64). Time upstream's KRHF.get_jk called once at the full
k-mesh vs called once at only the kpts_ibz subset (same DF object, same
density SHAPE -- not physically meaningful for the IBZ-subset call, this is a
pure cost-bound measurement), for GDF and FFTDF both. There is no upstream
code path that does this (khf_ksymm.get_jk always calls the full-BZ route) --
this bounds what D-PBC-26's fast path in 17-07 can plausibly gain.
"""
import sys, time
import numpy as np
import pyscf
assert pyscf.__version__ == "2.12.1", pyscf.__version__
from pyscf.pbc import gto, scf, df as pdf

half = 5.4306/2; q = 5.4306/4
cell = gto.Cell()
cell.atom = f"Si 0. 0. 0.\nSi {q} {q} {q}"
cell.a = [[0.,half,half],[half,0.,half],[half,half,0.]]
cell.basis = 'gth-szv'; cell.pseudo = 'gth-pade'
cell.space_group_symmetry = True
cell.verbose = 0
# DEVIATION from the plan's literal si default mesh: FFTDF get_jk with_k=True
# at 64 k-points and the default mesh (36^3) is an O(nkpts^2 * ngrids) sweep
# that does not finish in this measurement's time budget. This is a pure
# WALL-CLOCK RATIO measurement (full-BZ vs IBZ-subset, same DF object, same
# mesh on both sides) -- the ratio is not expected to depend on the mesh
# resolution, so the mesh is coarsened to make the measurement tractable.
# Recorded as a deviation in measurements/README.md and 17-01-SUMMARY.md.
cell.mesh = [9, 9, 9]
cell.build()

kpts_full = cell.make_kpts([4,4,4])
kpts_ibz_obj = cell.make_kpts([4,4,4], space_group_symmetry=True, time_reversal_symmetry=True)
nkpts = len(kpts_full)
nkpts_ibz = kpts_ibz_obj.nkpts_ibz
print(f"nkpts = {nkpts}   nkpts_ibz = {nkpts_ibz}   ratio nkpts/nkpts_ibz = {nkpts/nkpts_ibz:.3f}")
sys.stdout.flush()

nao = cell.nao
np.random.seed(3)

def make_dm(n):
    dm = np.random.random((n, nao, nao)) + 1j*np.random.random((n, nao, nao)) * 0.1
    dm = dm + dm.conj().transpose(0,2,1)
    return dm

dm_full = make_dm(nkpts)
dm_ibz_shape = dm_full[:nkpts_ibz]  # same shape as an IBZ-only call, cost bound only

for route in ['FFTDF', 'GDF']:
    print(f"building {route}..."); sys.stdout.flush()
    if route == 'FFTDF':
        mydf = pdf.FFTDF(cell, kpts=kpts_full)
    else:
        mydf = pdf.GDF(cell, kpts=kpts_full)
        mydf.build()

    t0 = time.time()
    vj_full, vk_full = mydf.get_jk(dm_full, kpts=kpts_full, with_j=True, with_k=True)
    t_full = time.time() - t0

    kpts_ibz_arr = np.asarray(kpts_ibz_obj.kpts_ibz)
    t0 = time.time()
    vj_ibz, vk_ibz = mydf.get_jk(dm_ibz_shape, kpts=kpts_ibz_arr, with_j=True, with_k=True)
    t_ibz = time.time() - t0

    print(f"[{route}] get_jk(full {nkpts} kpts)  = {t_full:.4f} s")
    print(f"[{route}] get_jk(ibz  {nkpts_ibz} kpts)  = {t_ibz:.4f} s")
    print(f"[{route}] wall-time ratio full/ibz = {t_full/t_ibz:.3f}x   "
          f"(ideal bound nkpts/nkpts_ibz = {nkpts/nkpts_ibz:.3f}x)")
    sys.stdout.flush()
