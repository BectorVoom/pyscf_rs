#!/usr/bin/env python
"""Gate B -- the transform floor, against ONE converged SCF.
17-CONTEXT §2.2 Gate B / 17-01-PLAN.md Task 2.

si (PBC-MASTER-PLAN §9.2, diamond structure, a=5.4306 Ang), gth-szv/gth-pade,
KRKS(LDA,VWN), [3,3,3] k-mesh, conv_tol=1e-11.  transform_dm,
transform_mo_occ, transform_mo_energy, transform_1e_operator (via get_fock),
symmetrize_density, and make_rdm1(transform_mo_coeff(...)) against the SAME
converged run's full-BZ arrays -- no second SCF, so no convergence noise.

Do NOT compare mo_coeff elementwise (17-CONTEXT §3.1) -- recorded anyway as a
demonstration that it IS large.
"""
import sys
import numpy as np
import pyscf
assert pyscf.__version__ == "2.12.1", pyscf.__version__
from pyscf.pbc import gto, scf
from pyscf.pbc.scf import khf
from pyscf.pbc.lib import kpts as libkpts

cell = gto.Cell()
half = 5.4306 / 2.0
quarter = 5.4306 / 4.0
cell.atom = f"Si 0. 0. 0.\nSi {quarter} {quarter} {quarter}"
cell.a = [[0., half, half], [half, 0., half], [half, half, 0.]]
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.verbose = 4
cell.build()

kpts0 = cell.make_kpts([3, 3, 3])
kmf = scf.KRKS(cell, kpts0)
kmf.xc = 'lda,vwn'
kmf.conv_tol = 1e-11
kmf.kernel()
print(f"e_tot = {kmf.e_tot!r}")
print(f"converged = {kmf.converged}")

kpts = libkpts.make_kpts(cell, kpts0, space_group_symmetry=True, time_reversal_symmetry=True)
print(f"nkpts = {kpts.nkpts}  nkpts_ibz = {kpts.nkpts_ibz}")

results = {}

# transform_dm
dms_ibz = kmf.make_rdm1()[kpts.ibz2bz]
dms_bz = kpts.transform_dm(dms_ibz)
results['transform_dm'] = abs(dms_bz - kmf.make_rdm1()).max()

# transform_mo_coeff -> make_rdm1 (the correct comparison per 17-CONTEXT §3.1)
mo_coeff_ibz = np.asarray(kmf.mo_coeff)[kpts.ibz2bz]
mo_coeff_bz = kpts.transform_mo_coeff(mo_coeff_ibz)
dms_bz2 = khf.make_rdm1(mo_coeff_bz, kmf.mo_occ)
results['make_rdm1(transform_mo_coeff)'] = abs(dms_bz2 - kmf.make_rdm1()).max()

# DEMONSTRATION ONLY -- elementwise mo_coeff comparison, expected to be LARGE
results['mo_coeff_elementwise_DEMO_ONLY'] = abs(np.asarray(kmf.mo_coeff) - mo_coeff_bz).max()

# transform_mo_occ
mo_occ_ibz = kpts.check_mo_occ_symmetry(kmf.mo_occ)
mo_occ_bz = kpts.transform_mo_occ(mo_occ_ibz)
results['transform_mo_occ'] = abs(mo_occ_bz - np.asarray(kmf.mo_occ)).max()

# transform_mo_energy
mo_energy_ibz = np.asarray(kmf.mo_energy)[kpts.ibz2bz]
mo_energy_bz = kpts.transform_mo_energy(mo_energy_ibz)
results['transform_mo_energy'] = abs(mo_energy_bz - np.asarray(kmf.mo_energy)).max()

# transform_1e_operator (fock) -- via transform_fock which calls transform_1e_operator
fock_ibz = kmf.get_fock()[kpts.ibz2bz]
fock_bz = kpts.transform_fock(fock_ibz)
results['transform_1e_operator(fock)'] = abs(fock_bz - kmf.get_fock()).max()

# symmetrize_density
rho0 = kmf.get_rho()
dms_ibz2 = kmf.make_rdm1()[kpts.ibz2bz]
nao = dms_ibz2.shape[-1]
rho = 0.
for k in range(kpts.nkpts_ibz):
    rho_k = khf.get_rho(kmf, dms_ibz2[k].reshape((-1, nao, nao)),
                         kpts=kpts.kpts_ibz[k].reshape((-1, 3)))
    rho += kpts.symmetrize_density(rho_k, k, cell.mesh)
rho *= 1.0 / kpts.nkpts
results['symmetrize_density'] = abs(rho - rho0).max()

print()
print("=== Gate B residuals (abs().max(), against the SAME converged run) ===")
for k, v in results.items():
    print(f"  {k:35s} {v:.6e}")

print()
expect_tight = [k for k in results if k not in ('mo_coeff_elementwise_DEMO_ONLY',)]
worst = max(results[k] for k in expect_tight)
print(f"worst (excluding the DEMO row) = {worst:.6e}")
if worst < 1e-12:
    print("All linear-map transforms land at >=1e-12 as 17-CONTEXT §2.2 expects.")
else:
    print("WARNING: at least one transform did NOT reach 1e-12 -- investigate before writing the gate.")
