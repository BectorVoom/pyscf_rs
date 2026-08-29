"""Gate 2, on the route plan 14-06 actually ships: `_CCMDFBuilder`.

`mdfladder.py` was run with `df.MDF`'s DEFAULT builder, and `MDF._prefer_ccdf`
is `False` (mdf.py:79) — so every row of `mdfladder.out` is the RANGE-SEPARATED
`_RSMDFBuilder`, which plan 14-07, not 14-06, is responsible for. Plan 14-06
ports `_CCMDFBuilder` (mdf.py:354), exactly as 14-02 ported `_CCGDFBuilder`.

This records the same ladder on the CC route, plus MDF's own default mesh, which
`14-06-PLAN.md` states as `[7,7,7]` and which is in fact `[11,11,11]` on diamond
2x2x2 and `[9,9,9]` on He-fcc — mesh 7 is simply `mdfladder.py`'s lowest rung.
"""
import time
import numpy as np
from _cells import diamond, he_fcc
from pyscf.pbc import scf, df


def ladder(cell, kmesh, label, meshes, fft_mesh):
    kpts = cell.make_kpts(kmesh)
    mf = scf.KRHF(cell, kpts)
    mf.with_df = df.FFTDF(cell, kpts)
    mf.with_df.mesh = fft_mesh
    mf.exxdiv = 'ewald'
    mf.conv_tol = 1e-11
    e_fft = mf.kernel()

    g = scf.KRHF(cell, kpts)
    d = df.GDF(cell, kpts)
    d._prefer_ccdf = True
    g.with_df = d
    g.exxdiv = 'ewald'
    g.conv_tol = 1e-11
    e_gdf = g.kernel()

    dflt = df.MDF(cell, kpts)
    dflt._prefer_ccdf = True
    dflt.build()
    print(f"\n=== {label} kmesh={kmesh}  (CC route, _prefer_ccdf = True) ===", flush=True)
    print(f"  MDF default mesh       = {list(dflt.mesh)}", flush=True)
    print(f"  FFTDF(mesh={fft_mesh}) E = {e_fft:.14f}", flush=True)
    print(f"  GDF (CC)               E = {e_gdf:.14f}  |dE vs FFTDF| = "
          f"{abs(e_gdf - e_fft):.3e}", flush=True)
    for m in meshes:
        t = time.time()
        mf2 = scf.KRHF(cell, kpts)
        d2 = df.MDF(cell, kpts)
        d2._prefer_ccdf = True
        d2.mesh = [m] * 3
        mf2.with_df = d2
        mf2.exxdiv = 'ewald'
        mf2.conv_tol = 1e-11
        e = mf2.kernel()
        print(f"  MDF(CC) mesh {m:2d}        E = {e:.14f}  |dE vs FFTDF| = "
              f"{abs(e - e_fft):.3e}  conv={mf2.converged}  {time.time() - t:.1f}s",
              flush=True)


ladder(he_fcc(), [2, 2, 2], "He-fcc/sto-3g", [7, 9, 11, 15, 21], [31] * 3)
ladder(he_fcc(), [1, 1, 1], "He-fcc/sto-3g gamma", [7, 11, 15, 21], [31] * 3)
ladder(diamond(), [2, 2, 2], "diamond/gth-szv", [7, 11, 15, 21, 27], [31] * 3)
ladder(diamond(), [1, 1, 1], "diamond/gth-szv gamma", [7, 11, 13, 15, 21], [31] * 3)
