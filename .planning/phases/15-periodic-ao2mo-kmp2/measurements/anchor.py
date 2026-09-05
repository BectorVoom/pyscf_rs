from pyscf import __version__, pbc

assert __version__ == "2.12.1", __version__
cell = pbc.gto.Cell()
cell.atom = """
C 0.000000000000 0.000000000000 0.000000000000
C 1.685068664391 1.685068664391 1.685068664391
"""
cell.basis = "gth-szv"
cell.pseudo = "gth-pade"
cell.a = """
0.000000000 3.370137329 3.370137329
3.370137329 0.000000000 3.370137329
3.370137329 3.370137329 0.000000000
"""
cell.unit = "B"
cell.verbose = 0
cell.build()
kpts = cell.make_kpts([1, 1, 2])
mf = pbc.scf.KRHF(cell, kpts=kpts, exxdiv=None)
mf.conv_tol = 1e-11
mf.kernel()
mp = pbc.mp.KMP2(mf)
e_corr, _ = mp.kernel(with_t2=False)
reference = -0.204721432828996
print(f"pyscf.__version__={__version__}")
print(f"e_hf={mf.e_tot:.17g}")
print(f"e_corr={e_corr:.17g}")
print(f"anchor_residual={abs(e_corr-reference):.17g}")
print(f"e_corr_ss={mp.e_corr_ss:.17g}")
print(f"e_corr_os={mp.e_corr_os:.17g}")
