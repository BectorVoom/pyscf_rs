"""Plan 14-07 Task 0 — the omega machinery, measured BEFORE porting.

Phase-9 precedent: every estimator the range-separated builder depends on is
recorded here first, so 7a's tests assert against upstream numbers rather than
against re-derived ones. The whole scheme's accuracy is one `omega` away, and a
wrong `omega` shows up as a plausible-looking 1e-6.

The `_RSGDFBuilder` `j2c`/`cderi` fingerprints `14-07-PLAN.md` also asks for are
NOT here: see `omega.out` and `14-07-SUMMARY.md`. They need a SHORT-RANGE
`int3c2e`/`int2c2e`, which the cintx safe API cannot request (no `range_omega`
knob on `ExecutionOptions` — the same gap `crates/pyscf-gto/src/range_coulomb.rs`
records as Open Question A5 / cintx#11). Recording targets the port cannot
reach would misrepresent the phase's state.
"""
import numpy as np
from _cells import diamond, he_fcc
from pyscf import gto
from pyscf.pbc import gto as pgto
from pyscf.pbc.df import rsdf_builder, ft_ao, aft
from pyscf.pbc.df import df as pdf


def row(label, value):
    if isinstance(value, np.ndarray):
        if value.size <= 8:
            print(f"  {label:44s} {np.array2string(value, precision=15)}")
        else:
            print(f"  {label:44s} min {value.min():.15g}  max {value.max():.15g} "
                  f"  n {value.size}")
    elif isinstance(value, (list, tuple)):
        print(f"  {label:44s} {list(value)}")
    elif isinstance(value, float):
        print(f"  {label:44s} {value:.15g}")
    else:
        print(f"  {label:44s} {value}")


def report(cell, kmesh, label):
    kpts = cell.make_kpts(kmesh)
    print(f"\n=== {label}  kmesh={kmesh} ===", flush=True)
    row("OMEGA_MIN", float(rsdf_builder.OMEGA_MIN))
    row("RCUT_THRESHOLD", float(rsdf_builder.RCUT_THRESHOLD))
    row("cell.precision", float(cell.precision))
    row("cell.rcut", float(cell.rcut))
    row("cell.vol", float(cell.vol))
    row("cell.nao", int(cell.nao))

    omega, mesh, ke_cutoff = rsdf_builder._guess_omega(cell, kpts, None)
    row("_guess_omega -> omega", float(omega))
    row("_guess_omega -> mesh", [int(x) for x in mesh])
    row("_guess_omega -> ke_cutoff", float(ke_cutoff))

    om_min = rsdf_builder.estimate_omega_min(cell)
    row("estimate_omega_min", float(om_min))
    row("estimate_ke_cutoff_for_omega(OMEGA_MIN)",
        float(rsdf_builder.estimate_ke_cutoff_for_omega(cell, rsdf_builder.OMEGA_MIN)))
    row("estimate_ke_cutoff_for_omega(omega)",
        float(rsdf_builder.estimate_ke_cutoff_for_omega(cell, omega)))
    row("estimate_omega_for_ke_cutoff(ke_cutoff)",
        float(rsdf_builder.estimate_omega_for_ke_cutoff(cell, ke_cutoff)))
    row("estimate_omega_for_ke_cutoff(20.0)",
        float(rsdf_builder.estimate_omega_for_ke_cutoff(cell, 20.0)))

    row("_round_off_to_odd_mesh([8,9,10])",
        [int(x) for x in rsdf_builder._round_off_to_odd_mesh([8, 9, 10])])
    row("_estimate_meshz", int(rsdf_builder._estimate_meshz(cell)))

    auxcell = pdf.make_modrho_basis(cell, None, None)
    row("auxcell.nao", int(auxcell.nao))
    row("estimate_rs_2c2e_rcut(auxcell, omega)",
        float(rsdf_builder.estimate_rs_2c2e_rcut(auxcell, omega)))
    row("estimate_rs_2c2e_rcut(auxcell, 0)",
        float(rsdf_builder.estimate_rs_2c2e_rcut(auxcell, 0)))

    # `estimate_rcut` / `estimate_ft_rcut` take a _RangeSeparatedCell upstream.
    # This port has none (D-PBC-21/23), so the numbers that matter to it are the
    # ones taken on the PLAIN cell — recorded both ways so the difference is on
    # the record rather than assumed away.
    rs_cell = ft_ao._RangeSeparatedCell.from_cell(cell, ke_cutoff,
                                                  rsdf_builder.RCUT_THRESHOLD)
    rs_auxcell = ft_ao._RangeSeparatedCell.from_cell(auxcell, ke_cutoff)
    for name, c, ac in [("plain cell", cell, auxcell), ("rs_cell", rs_cell, rs_auxcell)]:
        r = rsdf_builder.estimate_rcut(c, ac, omega, exclude_dd_block=False)
        row(f"estimate_rcut ({name}) max", float(np.max(r)))
        row(f"estimate_rcut ({name}) all", np.asarray(r, dtype=float))
        f = rsdf_builder.estimate_ft_rcut(c, exclude_dd_block=False)
        row(f"estimate_ft_rcut ({name}) max", float(np.max(f)))
        row(f"estimate_ft_rcut ({name}) all", np.asarray(f, dtype=float))

    row("_gaussian_int(auxcell)[:6]",
        np.asarray(rsdf_builder._gaussian_int(auxcell), dtype=float)[:6])

    # weighted_coulG_LR / _SR: the identity that catches an erf/erfc swap.
    class _Stub:
        pass
    stub = _Stub()
    stub.cell = cell
    stub.omega = omega
    stub.mesh = [int(x) for x in mesh]
    stub.kpts = kpts
    stub.max_memory = 4000
    stub.weighted_coulG = lambda kpt, exx, m, om=None: aft.weighted_coulG(
        stub, kpt, exx, m, om)
    stub.weighted_coulG_SR = lambda kpt, exx, m: rsdf_builder._RSGDFBuilder.weighted_coulG_SR(
        stub, kpt, exx, m)
    stub.weighted_coulG_LR = lambda kpt, exx, m: rsdf_builder._RSGDFBuilder.weighted_coulG_LR(
        stub, kpt, exx, m)
    lr = stub.weighted_coulG_LR(np.zeros(3), False, stub.mesh)
    sr = stub.weighted_coulG_SR(np.zeros(3), False, stub.mesh)
    full = aft.weighted_coulG(stub, np.zeros(3), False, stub.mesh)
    # G = 0 is special-cased in the LR branch; report the residual both ways.
    d = np.abs(lr + sr - full)
    row("max|LR + SR - full| (k=0)", float(d.max()))
    row("max|LR + SR - full| (k=0, G != 0)", float(d[1:].max()))
    row("weighted_coulG_LR[:4] (k=0)", np.asarray(lr, dtype=float)[:4])
    row("weighted_coulG_SR[:4] (k=0)", np.asarray(sr, dtype=float)[:4])
    if len(kpts) > 1:
        k = kpts[1]
        lr = stub.weighted_coulG_LR(k, False, stub.mesh)
        sr = stub.weighted_coulG_SR(k, False, stub.mesh)
        full = aft.weighted_coulG(stub, k, False, stub.mesh)
        row("max|LR + SR - full| (k = kpts[1])", float(np.abs(lr + sr - full).max()))


report(he_fcc(), [2, 2, 2], "He-fcc/sto-3g")
report(he_fcc(), [1, 1, 1], "He-fcc/sto-3g gamma")
report(diamond(), [2, 2, 2], "diamond/gth-szv")
report(diamond(), [1, 1, 1], "diamond/gth-szv gamma")
