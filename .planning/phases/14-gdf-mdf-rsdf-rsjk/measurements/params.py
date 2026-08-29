"""Phase-14 reference parameters + tensor fingerprints the port must reproduce.

Everything here is oracle-free-reproducible on the Rust side once 14-01..14-03
land; recording it now means the port is written against measured numbers rather
than against a guess.
"""
import numpy as np
from _cells import diamond, he_fcc
from pyscf.pbc.df import gdf_builder, df as pbcdf, incore
from pyscf.pbc.df import aft


def fingerprint(a):
    a = np.asarray(a)
    return dict(shape=list(a.shape), dtype=str(a.dtype),
                norm=float(np.linalg.norm(a)),
                maxabs=float(abs(a).max()) if a.size else 0.0,
                sum_re=float(np.asarray(a).real.sum()))


def report(cell, kmesh, label):
    kpts = cell.make_kpts(kmesh)
    print(f"\n=== {label}  kmesh={kmesh} ===", flush=True)
    auxcell = pbcdf.make_modrho_basis(cell, None, None)
    print("  auxcell.nao      =", auxcell.nao)
    print("  auxcell.nbas     =", auxcell.nbas)
    print("  auxcell l/exps   =",
          [(int(auxcell.bas_angular(i)), [float(x) for x in auxcell.bas_exp(i)])
           for i in range(min(auxcell.nbas, 12))])
    b = gdf_builder._CCGDFBuilder(cell, auxcell, kpts)
    b.build()
    print("  eta              =", b.eta)
    print("  mesh             =", b.mesh)
    print("  ke_cutoff        =", b.ke_cutoff)
    print("  fused_cell.nao   =", b.fused_cell.nao)
    print("  fused_cell.nbas  =", b.fused_cell.nbas)
    print("  direct_scf_tol   =", b.direct_scf_tol)
    print("  has_long_range   =", b.has_long_range())
    print("  exclude_dd_block =", b.exclude_dd_block)

    # 2-centre metric at the unique k-differences
    uniq = np.zeros((1, 3))
    j2c = b.get_2c2e(uniq)[0]
    print("  j2c(kpt=0)       =", fingerprint(j2c))
    w = np.linalg.eigvalsh(np.asarray(j2c))
    print("  j2c eig min/max  =", float(w.min()), float(w.max()))

    # auxbar / vbar
    vbar = b.fuse(gdf_builder.auxbar(b.fused_cell))
    print("  auxbar nnz       =", int((vbar != 0).sum()), " fingerprint",
          fingerprint(vbar))

    # eta estimators, standalone
    print("  estimate_eta_min          =", gdf_builder.estimate_eta_min(cell))
    print("  estimate_ke_cutoff_for_eta(eta) =",
          gdf_builder.estimate_ke_cutoff_for_eta(cell, b.eta))
    print("  estimate_eta_for_ke_cutoff(ke)  =",
          gdf_builder.estimate_eta_for_ke_cutoff(cell, b.ke_cutoff))
    print("  incore.estimate_rcut(cell,aux)  =",
          incore.estimate_rcut(cell, auxcell).max())
    print("  gdf_builder.estimate_rcut       =",
          gdf_builder.estimate_rcut(b.rs_cell, b.fused_cell).max())
    print("  cell.rcut                       =", cell.rcut)

    # the fully-built GDF cderi
    d = pbcdf.GDF(cell, kpts)
    d.build()
    print("  naoaux           =", d.get_naoaux())
    for (ki, kj) in [(0, 0)]:
        for LpqR, LpqI, sign in d.sr_loop((kpts[ki], kpts[kj]), compact=False):
            print(f"  cderi[{ki},{kj}] R  =", fingerprint(LpqR), "sign", sign)
            print(f"  cderi[{ki},{kj}] I  =", fingerprint(LpqI))
            break
    return d


if __name__ == '__main__':
    report(diamond(), [2, 2, 2], "diamond/gth-szv")
    report(diamond(), [1, 1, 1], "diamond/gth-szv gamma")
    report(he_fcc(), [2, 2, 2], "He-fcc/sto-3g (all-electron)")
