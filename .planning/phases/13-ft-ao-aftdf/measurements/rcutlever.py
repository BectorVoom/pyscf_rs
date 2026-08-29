import numpy as np
from pyscf.pbc import gto, scf, df
from pyscf.pbc.df import ft_ao

h=3.37032; q=1.68516
cell = gto.Cell()
cell.a=[[0.,h,h],[h,0.,h],[h,h,0.]]
cell.atom=[('C',(0.,0.,0.)),('C',(q,q,q))]
cell.basis='gth-szv'; cell.pseudo='gth-pade'; cell.unit='Bohr'; cell.verbose=0
cell.build()
kpts=cell.make_kpts([2,2,2])
mf=scf.KRHF(cell,kpts); dm=mf.get_init_guess()
orig = ft_ao.estimate_rcut
s0 = cell.pbc_intor('int1e_ovlp', kpts=np.zeros(3))

for scale in [1.0, 1.5, 2.0]:
    ft_ao.estimate_rcut = lambda c, precision=None, _s=scale: orig(c, precision)*_s
    # Gate 1 at this rcut
    ftg = ft_ao.ft_aopair(cell, np.zeros((1,3)), kpti_kptj=np.zeros((2,3)))
    g1 = abs(ftg[0]-s0).max()
    mesh=[31]*3
    f=df.FFTDF(cell,kpts); f.mesh=mesh
    a=df.AFTDF(cell,kpts); a.mesh=mesh
    vjf,vkf=f.get_jk(dm,kpts=kpts,exxdiv='ewald')
    vja,vka=a.get_jk(dm,kpts=kpts,exxdiv='ewald')
    print("LEVER rcut x%.1f (=%.2f)  GATE1 %.3e   dvj %.3e  dvk %.3e"
          % (scale, orig(cell).max()*scale, g1, abs(vjf-vja).max(), abs(vkf-vka).max()), flush=True)
ft_ao.estimate_rcut = orig
