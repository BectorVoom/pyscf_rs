"""The two blind spots in 14-VERIFICATION's GDF gates, measured.

14-VERIFICATION gates GDF on `he_fcc` (sto-3g -> ONE AO) and on diamond at
GAMMA. Neither can see a defect that needs BOTH nao > 1 AND ki != kj:

  * one AO makes `nao_pair == nao*nao`, so the `s2` store and the `s1` square
    are the same array and no (mu, nu) packing can be wrong;
  * at gamma every k-pair is diagonal, where `(L | mu^k nu^k)` IS Hermitian in
    (mu, nu) and `lib.ANTIHERMI` on one block is the whole story.

Off the diagonal it is not: `_KPair3CLoader.__getitem__` (df.py:990-1009) hands
`PBCunpack_tril_triu` (lib/pbc/fill_ints.c:1460-1483) `tril` from the (ki, kj)
block and `triu` from the (kj, ki) one.

This script records, for He/6-31g (nao = 2) at [1,1,2]:

  1. the four stored `j3c` blocks, so the port's store can be diffed elementwise;
  2. `sr_loop(..., compact=True)` for all four k-pairs -- (0,1) and (1,0) DIFFER,
     which is what makes serving one from the other wrong;
  3. `get_nuc` on GDF / AFTDF / FFTDF at the pinned [9,9,9] mesh and at the
     cell's own estimate, which is how `_CCNucBuilder`'s mesh independence shows
     up as a number;
  4. the converged KRHF energies, exxdiv=None, both DF routes;
  5. the same on the DIAMOND anchor, which is the pseudopotential leg -- its
     mesh already IS `estimate_mesh`, so it isolates the packing defect from the
     nuclear-mesh one.

Run:
  PYTHONPATH=$PWD .venv/bin/python -u \
    .planning/phases/14-gdf-mdf-rsdf-rsjk/measurements/offgamma_multiao.py
"""
import numpy as np
import pyscf
from pyscf.pbc import df, gto, scf

assert pyscf.__version__ == "2.12.1", pyscf.__version__
np.set_printoptions(precision=12, linewidth=200)


def helium_631g(mesh=(9, 9, 9)):
    """He-fcc as `_cells.he_fcc`, but 6-31g (TWO AOs) and a PINNED mesh."""
    h = 2.834589
    c = gto.Cell()
    c.a = [[0., h, h], [h, 0., h], [h, h, 0.]]
    c.atom = [('He', (0., 0., 0.))]
    c.basis = '6-31g'
    c.unit = 'Bohr'
    c.verbose = 0
    if mesh is not None:
        c.mesh = list(mesh)
    return c.build()


cell = helium_631g()
kpts = cell.make_kpts([1, 1, 2])
nao = cell.nao_nr()
print(f"cell.mesh = {cell.mesh}  nao = {nao}  nkpts = {len(kpts)}")

mydf = df.GDF(cell, kpts)
mydf.build()
print(f"auxcell nao = {mydf.auxcell.nao_nr()}")
print(f"auxcell basis = {mydf.auxcell._basis}\n")

print("--- 1. the stored j3c blocks, keyed ki*nkpts+kj ---")
import h5py
with h5py.File(mydf._cderi, 'r') as f:
    print(f"aosym = {f['aosym'][()]}")
    for key in sorted(f['j3c'].keys(), key=int):
        dat = f['j3c'][key]['0'][()]
        print(f"j3c/{key} {dat.shape} {dat.dtype}")
        print(dat)
print()

print("--- 2. sr_loop(compact=True) -- (0,1) and (1,0) are DIFFERENT ---")
for ki in range(len(kpts)):
    for kj in range(len(kpts)):
        for LpqR, LpqI, sign in mydf.sr_loop((kpts[ki], kpts[kj]), compact=True):
            print(f"sr_loop({ki},{kj}) sign={sign} shape={LpqR.shape}")
            print(LpqR)
print()

print("--- 3. get_nuc: _CCNucBuilder does not depend on cell.mesh ---")
for meshspec in ((9, 9, 9), None):
    c = helium_631g(meshspec)
    k = c.make_kpts([1, 1, 2])
    row = {name: complex(builder(c, k).get_nuc(k)[0][0, 0]).real
           for name, builder in (("gdf", df.GDF), ("aftdf", df.AFTDF),
                                 ("fftdf", df.FFTDF))}
    print(f"cell.mesh = {c.mesh}  " +
          "  ".join(f"{n} nuc[0][0,0] = {v:.12f}" for n, v in row.items()))
print()

print("--- 4. KRHF, exxdiv=None, conv_tol=1e-11 ---")
for name, builder in (("fftdf", df.FFTDF), ("gdf", df.GDF)):
    mf = scf.KRHF(cell, kpts, exxdiv=None)
    mf.with_df = builder(cell, kpts)
    mf.conv_tol = 1e-11
    e = mf.kernel()
    print(f"{name}: E = {e!r}  converged = {mf.converged}")


print()
print("--- 5. the diamond anchor: same k-mesh, pseudopotential, nao = 8 ---")
h, q = 3.370137329, 1.685068664391
dia = gto.Cell()
dia.a = [[0., h, h], [h, 0., h], [h, h, 0.]]
dia.atom = [('C', (0., 0., 0.)), ('C', (q, q, q))]
dia.basis = 'gth-szv'
dia.pseudo = 'gth-pade'
dia.unit = 'Bohr'
dia.verbose = 0
dia.build()
dkpts = dia.make_kpts([1, 1, 2])
print(f"cell.mesh = {dia.mesh}  nao = {dia.nao_nr()}  nkpts = {len(dkpts)}")
energies = {}
for name, builder in (("fftdf", df.FFTDF), ("gdf", df.GDF)):
    mydf = builder(dia, dkpts)
    mydf.build()
    print(f"{name}: get_pp[0][0,0] = {complex(mydf.get_pp(dkpts)[0][0, 0]).real:.14e}")
    mf = scf.KRHF(dia, dkpts, exxdiv=None)
    mf.with_df = mydf
    mf.conv_tol = 1e-11
    energies[name] = mf.kernel()
    print(f"{name}: E = {energies[name]!r}  converged = {mf.converged}")
print(f"|E_FFTDF - E_GDF| = {abs(energies['fftdf'] - energies['gdf']):.6e}  "
      "-- the DF FITTING error, a property of the auxiliary basis")
