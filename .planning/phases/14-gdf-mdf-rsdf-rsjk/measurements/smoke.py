import sys, time
import numpy as np
from _cells import diamond
from pyscf.pbc import df
import pyscf
print("pyscf", pyscf.__version__, pyscf.__file__)
cell = diamond()
kpts = cell.make_kpts([2,2,2])
print("default mesh", cell.mesh, "nao", cell.nao)
for name, cls in [("GDF", df.GDF), ("MDF", df.MDF), ("RSDF", df.RSDF)]:
    t = time.time()
    try:
        d = cls(cell, kpts); d.build()
        aux = d.auxcell.nao if getattr(d, 'auxcell', None) is not None else '?'
        print(f"{name:5s} ok  naux={d.get_naoaux()}  auxcell.nao={aux}  mesh={getattr(d,'mesh',None)}  {time.time()-t:.1f}s")
    except Exception as e:
        print(f"{name:5s} FAIL {type(e).__name__}: {e}")
