import numpy as np
from _cells import diamond, he_fcc
from pyscf.pbc.df import ft_ao, gdf_builder, rsdf_builder, df as pbcdf

for label, cell in [("diamond", diamond()), ("he_fcc", he_fcc())]:
    kpts = cell.make_kpts([2,2,2])
    auxcell = pbcdf.make_modrho_basis(cell, None, None)
    b = gdf_builder._CCGDFBuilder(cell, auxcell, kpts); b.build()
    rs = b.rs_cell
    print(f"=== {label} ===")
    print("  ke_cutoff", b.ke_cutoff, "RCUT_THRESHOLD", rsdf_builder.RCUT_THRESHOLD)
    print("  rs_cell.nbas", rs.nbas, "cell.nbas", cell.nbas, "rs_cell.nao", rs.nao)
    print("  bas_type", rs.bas_type, " SMOOTH=", ft_ao.SMOOTH_BASIS)
    print("  any smooth:", bool(np.any(rs.bas_type == ft_ao.SMOOTH_BASIS)))
    print("  sh_loc", rs.sh_loc)
    print("  bvk_kmesh", b.bvk_kmesh)
    print("  supmol.nbas", b.supmol.nbas, "supmol.nao", b.supmol.nao)
    print("  supmol_ft.nbas", b.supmol_ft.nbas)
    j2c = b.get_2c2e(np.zeros((1,3)))[0]
    cd = b.decompose_j2c(j2c)
    print("  j2ctag", cd[2], " j2c shape", np.asarray(cd[0]).shape,
          " negative", None if cd[1] is None else np.asarray(cd[1]).shape)
    print("  linear_dep_threshold", b.linear_dep_threshold, "j2c_eig_always", b.j2c_eig_always)
    # aux basis exps per atom
    print("  auxcell shells:", [(int(auxcell.bas_atom(i)), int(auxcell.bas_angular(i)),
                                 float(auxcell.bas_exp(i)[0])) for i in range(auxcell.nbas)][:8])
