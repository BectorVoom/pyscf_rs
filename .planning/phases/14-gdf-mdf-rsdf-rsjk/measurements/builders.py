"""Phase-14 Gate 1: does every DF builder give the same KRHF energy, and to what?

The ROADMAP asks for 1e-15; the master plan's 14-09 asks for 1e-6. Neither is a
measurement. This script produces the number.
"""
import time
import numpy as np
from _cells import diamond, he_fcc
from pyscf.pbc import scf, df

BUILDERS = [("FFTDF", df.FFTDF), ("AFTDF", df.AFTDF), ("GDF", df.GDF),
            ("MDF", df.MDF), ("RSDF", df.RSDF)]


def run(cell, kmesh, label, mesh=None):
    kpts = cell.make_kpts(kmesh)
    print(f"\n=== {label}  kmesh={kmesh}  nkpts={len(kpts)}  nao={cell.nao} "
          f"default mesh={cell.mesh}  override={mesh} ===", flush=True)
    out = {}
    for name, cls in BUILDERS:
        t = time.time()
        mf = scf.KRHF(cell, kpts)
        mf.with_df = cls(cell, kpts)
        if mesh is not None and name in ("FFTDF", "AFTDF"):
            mf.with_df.mesh = mesh
        mf.exxdiv = 'ewald'
        mf.conv_tol = 1e-11
        e = mf.kernel()
        out[name] = e
        print(f"  {name:6s} E = {e:.14f}  conv={mf.converged}  {time.time()-t:6.1f}s",
              flush=True)
    print(f"  --- pairwise |dE| ({label}) ---", flush=True)
    names = [n for n, _ in BUILDERS]
    for i, a in enumerate(names):
        for b in names[i+1:]:
            print(f"  |{a:5s} - {b:5s}| = {abs(out[a]-out[b]):.3e}", flush=True)
    return out


if __name__ == '__main__':
    run(diamond(), [2, 2, 2], "diamond/gth-szv 2x2x2", mesh=[31]*3)
    run(diamond(), [1, 1, 1], "diamond/gth-szv gamma", mesh=[31]*3)
    run(he_fcc(), [2, 2, 2], "He-fcc/sto-3g 2x2x2 (all-electron)")
