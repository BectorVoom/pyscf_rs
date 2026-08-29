"""df.GDF's DEFAULT builder is _RSGDFBuilder, not _CCGDFBuilder.

Phase 14 ports _CCGDFBuilder first (plan 14-02) and _RSGDFBuilder later (14-07),
so the oracle target for the port's first GDF is upstream with
_prefer_ccdf = True. This measures the gap between the two routes.
"""
import numpy as np
from _cells import diamond, he_fcc
from pyscf.pbc import scf, df

for label, cell, kmesh in [("diamond 2x2x2", diamond(), [2,2,2]),
                           ("diamond gamma", diamond(), [1,1,1]),
                           ("He-fcc 2x2x2 ", he_fcc(), [2,2,2])]:
    kpts = cell.make_kpts(kmesh)
    out = {}
    for tag in (False, True):
        mf = scf.KRHF(cell, kpts)
        d = df.GDF(cell, kpts); d._prefer_ccdf = tag
        mf.with_df = d; mf.exxdiv = 'ewald'; mf.conv_tol = 1e-11
        out[tag] = mf.kernel()
        print(f"  {label} _prefer_ccdf={tag!s:5s} E = {out[tag]:.14f} conv={mf.converged}",
              flush=True)
    print(f"  {label} |dE(RS route - CC route)| = {abs(out[False]-out[True]):.3e}\n",
          flush=True)
