"""Does exclude_dd_block change the ANSWER or only the ROUTE?

This decides whether Phase 14 must port ft_ao._RangeSeparatedCell (~600 lines
that D-PBC-21 declined in Phase 13) or can again port the definition.
"""
import time
import numpy as np
from _cells import diamond, he_fcc
from pyscf.pbc import scf, df
from pyscf.pbc.df import gdf_builder, df as pbcdf


def cderi_and_energy(cell, kmesh, exclude_dd, label):
    kpts = cell.make_kpts(kmesh)
    t = time.time()
    d = pbcdf.GDF(cell, kpts)
    auxcell = pbcdf.make_modrho_basis(cell, d.auxbasis, None)
    b = gdf_builder._CCGDFBuilder(cell, auxcell, kpts)
    b.exclude_dd_block = exclude_dd
    d._prefer_ccdf = True
    d._dfbuilder = b
    # drive the build through the builder we configured
    import tempfile, os
    fn = tempfile.mktemp(dir=os.getcwd(), suffix='.h5')
    b.make_j3c(fn, aosym='s2', j_only=False)
    d._cderi = fn
    d.auxcell = auxcell
    mf = scf.KRHF(cell, kpts)
    mf.with_df = d
    mf.exxdiv = 'ewald'
    mf.conv_tol = 1e-11
    e = mf.kernel()
    # fingerprint of cderi at (0,0)
    fp = None
    for LpqR, LpqI, sign in d.sr_loop((kpts[0], kpts[0]), compact=False):
        fp = (float(np.linalg.norm(LpqR)), float(np.linalg.norm(LpqI)))
        break
    os.unlink(fn)
    print(f"  {label} exclude_dd_block={exclude_dd}: E = {e:.14f} conv={mf.converged} "
          f"cderi|R|={fp[0]:.12f} |I|={fp[1]:.3e}  {time.time()-t:.1f}s", flush=True)
    return e, fp


for label, cell, kmesh in [("diamond 2x2x2", diamond(), [2,2,2]),
                           ("diamond gamma", diamond(), [1,1,1]),
                           ("He-fcc 2x2x2 ", he_fcc(), [2,2,2])]:
    print(f"=== {label} ===", flush=True)
    e1, f1 = cderi_and_energy(cell, kmesh, True, label)
    e2, f2 = cderi_and_energy(cell, kmesh, False, label)
    print(f"  |dE| = {abs(e1-e2):.3e}   d|cderi_R| = {abs(f1[0]-f2[0]):.3e}", flush=True)
