import numpy as np
from pyscf.pbc import gto, scf, df

h=3.37032; q=1.68516
cell = gto.Cell()
cell.a = [[0.,h,h],[h,0.,h],[h,h,0.]]
cell.atom = [('C',(0.,0.,0.)),('C',(q,q,q))]
cell.basis='gth-szv'; cell.pseudo='gth-pade'; cell.unit='Bohr'; cell.verbose=0
cell.build()
kpts = cell.make_kpts([2,2,2])
print("default mesh", cell.mesh)
for m in [15,21,31,41,55,71]:
    mesh=[m]*3
    f = scf.KRHF(cell, kpts); f.with_df = df.FFTDF(cell, kpts); f.with_df.mesh=mesh
    f.exxdiv='ewald'; f.conv_tol=1e-12; e1=f.kernel()
    a = scf.KRHF(cell, kpts); a.with_df = df.AFTDF(cell, kpts); a.with_df.mesh=mesh
    a.exxdiv='ewald'; a.conv_tol=1e-12; e2=a.kernel()
    print("GATE2 mesh %2d  E_FFTDF %.14f  E_AFTDF %.14f  |dE| %.3e  conv %s/%s"
          % (m, e1, e2, abs(e1-e2), f.converged, a.converged))
