import numpy as np
from pyscf.pbc import gto, scf, df
from pyscf.pbc.df import ft_ao

h=3.37032; q=1.68516
cell = gto.Cell()
cell.a = [[0.,h,h],[h,0.,h],[h,h,0.]]
cell.atom = [('C',(0.,0.,0.)),('C',(q,q,q))]
cell.basis='gth-szv'; cell.pseudo='gth-pade'; cell.unit='Bohr'; cell.verbose=0
cell.build()
print("default mesh", cell.mesh, "nao", cell.nao_nr(), "rcut", cell.rcut)

# Gate 1 on upstream: ft_aopair[G=0] vs pbc_intor int1e_ovlp
s = cell.pbc_intor('int1e_ovlp', kpts=np.zeros(3))
ft = ft_ao.ft_aopair(cell, np.zeros((1,3)), kpti_kptj=np.zeros((2,3)))
print("GATE1 gamma  ft[G=0] vs int1e_ovlp: %.3e" % abs(ft[0]-s).max())
kpts = cell.make_kpts([2,2,2])
for ik,k in enumerate(kpts[:3]):
    sk = cell.pbc_intor('int1e_ovlp', kpts=k)
    ftk = ft_ao.ft_aopair(cell, np.zeros((1,3)), kpti_kptj=np.array([k,k]))
    print("GATE1 k=%d  %.3e" % (ik, abs(ftk[0]-sk).max()))
print("estimate_rcut max", ft_ao.estimate_rcut(cell).max())

mf = scf.KRHF(cell, kpts); mf.exxdiv='ewald'
dm = mf.get_init_guess()
for mesh in [[15]*3,[21]*3,[31]*3,[41]*3]:
    f = df.FFTDF(cell, kpts); f.mesh=mesh
    a = df.AFTDF(cell, kpts); a.mesh=mesh
    vjf, vkf = f.get_jk(dm, kpts=kpts, exxdiv='ewald')
    vja, vka = a.get_jk(dm, kpts=kpts, exxdiv='ewald')
    print("JK mesh %s  dvj %.3e  dvk %.3e" % (mesh[0], abs(vjf-vja).max(), abs(vkf-vka).max()))
