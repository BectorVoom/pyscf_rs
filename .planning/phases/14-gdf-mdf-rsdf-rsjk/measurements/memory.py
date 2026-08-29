"""Phase-14 Gate 4: is GDF under 20% of FFTDF memory at mesh [40,40,40]?

FFTDF's resident cost is the AO table on the uniform grid,
nkpts * ngrids * nao * 16 B (complex128); GDF's is the _cderi store, which is a
FILE, so we measure the file and also state the in-core equivalent.
"""
import os, tempfile
import numpy as np
from _cells import diamond, he_fcc
from pyscf.pbc import df


def report(cell, kmesh, label, mesh=(40, 40, 40)):
    kpts = cell.make_kpts(kmesh)
    nkpts, nao = len(kpts), cell.nao
    ngrids = int(np.prod(mesh))
    fft_ao = nkpts * ngrids * nao * 16
    fft_rho = ngrids * 16
    d = df.GDF(cell, kpts)
    fn = tempfile.mktemp(dir=os.getcwd(), suffix='.h5')
    d._cderi_to_save = fn
    d.build()
    size = os.path.getsize(d._cderi)
    naux = d.get_naoaux()
    nao_pair = nao * (nao + 1) // 2
    incore = nkpts * nkpts * naux * nao_pair * 16
    print(f"=== {label} kmesh={kmesh} nkpts={nkpts} nao={nao} naux={naux} ===")
    print(f"  FFTDF AO table  @mesh{list(mesh)}  = {fft_ao/2**20:10.2f} MiB "
          f"(+{fft_rho/2**20:.2f} MiB rho)")
    print(f"  GDF  _cderi file                   = {size/2**20:10.2f} MiB")
    print(f"  GDF  incore upper bound            = {incore/2**20:10.2f} MiB")
    print(f"  ratio file/FFTDF                   = {100*size/fft_ao:10.2f} %")
    print(f"  ratio incore/FFTDF                 = {100*incore/fft_ao:10.2f} %")
    os.unlink(d._cderi)


report(diamond(), [2, 2, 2], "diamond/gth-szv")
report(diamond(), [3, 3, 3], "diamond/gth-szv")
report(he_fcc(), [2, 2, 2], "He-fcc/sto-3g")
