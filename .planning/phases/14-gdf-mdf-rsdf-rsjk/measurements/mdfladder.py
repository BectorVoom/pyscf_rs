"""Phase-14 Gate 2: MDF is the builder that legitimately converges to FFTDF.

MDF = GDF (Gaussian fit) + the AFT residual on a plane-wave mesh. Raising that
mesh must drive E_MDF toward E_FFTDF; GDF alone cannot. This records the ladder
the port is gated against.
"""
import time
import numpy as np
from _cells import diamond, he_fcc
from pyscf.pbc import scf, df


def ladder(cell, kmesh, label, meshes, fft_mesh):
    kpts = cell.make_kpts(kmesh)
    mf = scf.KRHF(cell, kpts); mf.with_df = df.FFTDF(cell, kpts)
    mf.with_df.mesh = fft_mesh; mf.exxdiv='ewald'; mf.conv_tol=1e-11
    e_fft = mf.kernel()
    g = scf.KRHF(cell, kpts); g.with_df = df.GDF(cell, kpts)
    g.exxdiv='ewald'; g.conv_tol=1e-11
    e_gdf = g.kernel()
    print(f"\n=== {label} kmesh={kmesh} ===", flush=True)
    print(f"  FFTDF(mesh={fft_mesh}) E = {e_fft:.14f}", flush=True)
    print(f"  GDF                    E = {e_gdf:.14f}  |dE vs FFTDF| = {abs(e_gdf-e_fft):.3e}",
          flush=True)
    for m in meshes:
        t = time.time()
        mf2 = scf.KRHF(cell, kpts)
        d = df.MDF(cell, kpts); d.mesh = [m]*3
        mf2.with_df = d; mf2.exxdiv='ewald'; mf2.conv_tol=1e-11
        e = mf2.kernel()
        print(f"  MDF mesh {m:2d}            E = {e:.14f}  |dE vs FFTDF| = "
              f"{abs(e-e_fft):.3e}  conv={mf2.converged}  {time.time()-t:.1f}s", flush=True)


ladder(diamond(), [2,2,2], "diamond/gth-szv", [7, 11, 15, 21, 27], [31]*3)
ladder(he_fcc(), [2,2,2], "He-fcc/sto-3g", [7, 11, 15, 21, 27], [31]*3)
