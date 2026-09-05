#!/usr/bin/env python
"""Staggered-mesh KMP2 oracle (Phase 15 follow-up).

Reproduces the fixture committed in `pyscf/pbc/mp/kmp2_stagger.py:353-420`
(H2 dimer, gth-szv/gth-pade, ke_cutoff=100, 2x2x2 k-points, exxdiv='ewald')
and prints every number the Rust `Kmp2Stagger` test is gated against:

  * the FFTDF submesh energy (flag_submesh=True)
  * the FFTDF full-mesh energy (flag_submesh=False, get_bands + vcut_sph)
  * the standard KMP2 energy on the same mean field
  * the residual of each against the constant embedded in upstream's source

Run from the workspace root:

    PYTHONPATH=$PWD .venv/bin/python -u \
      .planning/phases/15-periodic-ao2mo-kmp2/measurements/stagger.py
"""

import numpy as np

import pyscf
from pyscf.pbc import df, gto, mp, scf
from pyscf.pbc.mp.kmp2_stagger import KMP2_stagger

assert pyscf.__version__ == "2.12.1", pyscf.__version__
print("pyscf.__version__=%s" % pyscf.__version__)

# Committed constants, `kmp2_stagger.py:385/390/395`.
SRC_SUBMESH = -0.0160902544091997
SRC_FULLMESH = -0.0140289970302513
SRC_STANDARD = -0.0143904878990777


def build_cell():
    cell = gto.Cell()
    cell.pseudo = "gth-pade"
    cell.basis = "gth-szv"
    cell.ke_cutoff = 100
    cell.atom = """
        H 3.00   3.00   2.10
        H 3.00   3.00   3.90
        """
    cell.a = """
        6.0   0.0   0.0
        0.0   6.0   0.0
        0.0   0.0   6.0
        """
    cell.unit = "B"
    cell.verbose = 0
    cell.build()
    return cell


cell = build_cell()
print("mesh=%s" % (list(map(int, cell.mesh)),))
print("nao=%d" % cell.nao_nr())

kpts = cell.make_kpts([2, 2, 2], with_gamma_point=True)
kmf = scf.KRHF(cell, kpts, exxdiv="ewald")
kmf.conv_tol = 1e-11
ehf = kmf.kernel()
print("e_tot_krhf=%.17g" % ehf)
print("converged=%s" % kmf.converged)

kmp = KMP2_stagger(kmf, flag_submesh=True)
e_submesh = kmp.kernel()
print("stagger_submesh_fftdf=%.17g" % e_submesh)
print("stagger_submesh_fftdf_residual=%.6g" % abs(e_submesh - SRC_SUBMESH))
print("stagger_submesh_nkpts_ov=%d" % kmp.nkpts_ov)
print("stagger_submesh_kpts_idx_occ=%s" % (list(map(int, kmp.kpts_idx_occ)),))
print("stagger_submesh_kpts_idx_vir=%s" % (list(map(int, kmp.kpts_idx_vir)),))

kmp = KMP2_stagger(kmf, flag_submesh=False)
e_fullmesh = kmp.kernel()
print("stagger_fullmesh_fftdf=%.17g" % e_fullmesh)
print("stagger_fullmesh_fftdf_residual=%.6g" % abs(e_fullmesh - SRC_FULLMESH))
print("stagger_fullmesh_nkpts_ov=%d" % kmp.nkpts_ov)

kmp = mp.KMP2(kmf)
e_standard, _ = kmp.kernel()
print("standard_kmp2_fftdf=%.17g" % e_standard)
print("standard_kmp2_fftdf_residual=%.6g" % abs(e_standard - SRC_STANDARD))

# The same three numbers on a GDF mean field, so the Rust GDF/Lov stagger route
# is gated against its own upstream value (15-CONTEXT §2.2, the two-route rule).
kmf_gdf = scf.KRHF(cell, kpts, exxdiv="ewald")
kmf_gdf.with_df = df.GDF(cell, kpts).build()
kmf_gdf.conv_tol = 1e-11
ehf_gdf = kmf_gdf.kernel()
print("e_tot_krhf_gdf=%.17g" % ehf_gdf)
kmp = KMP2_stagger(kmf_gdf, flag_submesh=True)
print("stagger_submesh_gdf=%.17g" % kmp.kernel())
print("stagger_submesh_gdf_with_df_ints=%s" % kmp.with_df_ints)

# Exercise the submesh route through the four-index AO2MO path as well, to show
# the Lov and AO2MO routes agree on the staggered kernel exactly as they do on
# the plain one (15-VERIFICATION row 3).
kmp = KMP2_stagger(kmf_gdf, flag_submesh=True)
kmp.with_df_ints = False
print("stagger_submesh_gdf_ao2mo=%.17g" % kmp.kernel())

# The scaled staggered k-points, so the Rust `staggered_submesh` map can be
# diffed element-wise rather than only structurally.
kmp = KMP2_stagger(kmf, flag_submesh=True)
occ = np.asarray(cell.get_scaled_kpts(kmp.kpts_occ))
vir = np.asarray(cell.get_scaled_kpts(kmp.kpts_vir))
print("stagger_scaled_occ=%s" % np.array2string(occ, precision=12))
print("stagger_scaled_vir=%s" % np.array2string(vir, precision=12))
