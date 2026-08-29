import numpy as np
from pyscf.pbc import gto, scf, df, tools

h=3.37032; q=1.68516
cell = gto.Cell()
cell.a=[[0.,h,h],[h,0.,h],[h,h,0.]]
cell.atom=[('C',(0.,0.,0.)),('C',(q,q,q))]
cell.basis='gth-szv'; cell.pseudo='gth-pade'; cell.unit='Bohr'; cell.verbose=0
cell.build()
kpts=cell.make_kpts([2,2,2])
mf=scf.KRHF(cell,kpts); dm=mf.get_init_guess()
print("madelung", tools.madelung(cell,kpts))
for mesh in [[21]*3,[31]*3]:
    f=df.FFTDF(cell,kpts); f.mesh=mesh
    a=df.AFTDF(cell,kpts); a.mesh=mesh
    for exx in [None,'ewald']:
        vjf,vkf=f.get_jk(dm,kpts=kpts,exxdiv=exx)
        vja,vka=a.get_jk(dm,kpts=kpts,exxdiv=exx)
        print("CTL mesh %2d exxdiv=%-6s dvj %.3e dvk %.3e"
              % (mesh[0],str(exx),abs(vjf-vja).max(),abs(vkf-vka).max()), flush=True)
