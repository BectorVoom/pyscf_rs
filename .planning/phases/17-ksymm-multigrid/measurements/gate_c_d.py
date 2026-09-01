#!/usr/bin/env python
"""Gates C and D -- the two-SCF energy floor, mesh PINNED (17-01-PLAN.md Task 3).

For si, diamond (deep grid: KRHF/KRKS x gamma/monkhorst x FFTDF/GDF x sym on/off)
and lif, he_fcc, graphene (broad sweep: KRHF, FFTDF, gamma-centred only, sym on/off)
-- resource-scoped as recorded in measurements/README.md -- record
|E(ksymm) - E(full BZ)| with cell.mesh PINNED identically on both sides
(17-CONTEXT §3.3), plus what the mesh WOULD have been unpinned and the energy
delta that alone produces, plus run-to-run / thread-count spread on one
representative configuration.
"""
import sys, time, os
import numpy as np
import pyscf
assert pyscf.__version__ == "2.12.1", pyscf.__version__
from pyscf.pbc import gto, scf

def build_cell(kind, mesh=None):
    c = gto.Cell()
    if kind == 'si':
        half = 5.4306/2; q = 5.4306/4
        c.atom = f"Si 0. 0. 0.\nSi {q} {q} {q}"
        c.a = [[0.,half,half],[half,0.,half],[half,half,0.]]
        c.basis = 'gth-szv'; c.pseudo = 'gth-pade'
    elif kind == 'diamond':
        half = 3.5668/2; q = 3.5668/4
        c.atom = f"C 0. 0. 0.\nC {q} {q} {q}"
        c.a = [[0.,half,half],[half,0.,half],[half,half,0.]]
        c.basis = 'gth-szv'; c.pseudo = 'gth-pade'
    elif kind == 'lif':
        half = 4.03/2
        c.atom = f"Li 0. 0. 0.\nF {half} {half} {half}"
        c.a = [[0.,half,half],[half,0.,half],[half,half,0.]]
        c.basis = 'gth-szv'; c.pseudo = 'gth-pade'
    elif kind == 'he_fcc':
        half = 3.0/2
        c.atom = f"He 0. 0. 0."
        c.a = [[0.,half,half],[half,0.,half],[half,half,0.]]
        c.basis = 'gth-szv'; c.pseudo = 'gth-pade'
    elif kind == 'graphene':
        a_gr = 2.46; c_vac = 20.0
        a1 = np.array([a_gr,0.,0.]); a2 = np.array([-a_gr/2., a_gr*np.sqrt(3)/2.,0.])
        atom2 = (a1 + 2*a2)/3.
        c.atom = f"C 0. 0. 0.\nC {atom2[0]:.10f} {atom2[1]:.10f} {atom2[2]:.10f}"
        c.a = [list(a1), list(a2), [0.,0.,c_vac]]
        c.dimension = 2
        c.basis = 'gth-szv'; c.pseudo = 'gth-pade'
    else:
        raise ValueError(kind)
    c.space_group_symmetry = True
    c.verbose = 0
    if mesh is not None:
        c.mesh = [int(x) for x in mesh]
    c.build()
    return c

def default_mesh(kind):
    """The mesh cell.build() picks with no explicit mesh given (before any
    symmetrize_mesh enlargement from space_group_symmetry)."""
    c0 = build_cell(kind, mesh=None)
    return [int(x) for x in c0.mesh]

def run_scf(cell, kpts, method, df_route, conv_tol=1e-11):
    if method == 'KRHF':
        mf = scf.KRHF(cell, kpts)
    elif method == 'KRKS':
        mf = scf.KRKS(cell, kpts)
        mf.xc = 'lda,vwn'
    else:
        raise ValueError(method)
    if df_route == 'GDF':
        mf = mf.density_fit()
    mf.conv_tol = conv_tol
    mf.chkfile = None
    mf.kernel()
    return mf.e_tot, mf.converged

def pair(kind, mesh, kmesh_type, method, df_route):
    with_gamma = (kmesh_type == 'gamma')
    cell = build_cell(kind, mesh=mesh)
    kpts_sym = cell.make_kpts([2,2,2], with_gamma_point=with_gamma,
                               space_group_symmetry=True, time_reversal_symmetry=True)
    kpts_full = cell.make_kpts([2,2,2], with_gamma_point=with_gamma)
    e_sym, conv_sym = run_scf(cell, kpts_sym, method, df_route)
    e_full, conv_full = run_scf(cell, kpts_full, method, df_route)
    return e_sym, e_full, conv_sym, conv_full

results = []

def record(kind, mesh, kmesh_type, method, df_route):
    t0 = time.time()
    e_sym, e_full, cs, cf = pair(kind, mesh, kmesh_type, method, df_route)
    dt = time.time() - t0
    d = abs(e_sym - e_full)
    row = dict(kind=kind, mesh=mesh, kmesh_type=kmesh_type, method=method,
               df=df_route, e_sym=e_sym, e_full=e_full, diff=d,
               conv_sym=cs, conv_full=cf, wall=dt)
    results.append(row)
    print(f"[{kind:9s} mesh={mesh} {kmesh_type:9s} {method:5s} {df_route:5s}] "
          f"E_sym={e_sym:.12f} E_full={e_full:.12f} |dE|={d:.3e} "
          f"conv=({cs},{cf}) wall={dt:.1f}s")
    sys.stdout.flush()

print("### mesh pinned to the default (non-symmetry) build for each system ###")
pinned_mesh = {}
for kind in ['si', 'diamond', 'lif', 'he_fcc', 'graphene']:
    m = default_mesh(kind)
    pinned_mesh[kind] = m
    print(f"  {kind:9s} default mesh = {m}")
sys.stdout.flush()

print()
print("### deep grid: si, diamond -- KRHF/KRKS x gamma/monkhorst x FFTDF/GDF ###")
for kind in ['si', 'diamond']:
    m = pinned_mesh[kind]
    for method in ['KRHF', 'KRKS']:
        for kmesh_type in ['gamma', 'monkhorst']:
            for df_route in ['FFTDF', 'GDF']:
                record(kind, m, kmesh_type, method, df_route)

print()
print("### broad sweep: lif, he_fcc, graphene -- KRHF, FFTDF, gamma-centred only ###")
for kind in ['lif', 'he_fcc', 'graphene']:
    m = pinned_mesh[kind]
    record(kind, m, 'gamma', 'KRHF', 'FFTDF')

print()
print("### mesh-unpinning demonstration: what changes if space_group_symmetry=True")
print("### is allowed to enlarge cell.mesh via symmetrize_mesh (17-CONTEXT §3.3) ###")
for kind in ['si', 'diamond', 'lif', 'he_fcc', 'graphene']:
    pinned = pinned_mesh[kind]
    c_unpinned = build_cell(kind, mesh=None)  # let symmetrize_mesh do its thing
    unpinned_mesh = list(c_unpinned.mesh)
    kpts_sym = c_unpinned.make_kpts([2,2,2], space_group_symmetry=True, time_reversal_symmetry=True)
    e_unpinned, conv_u = run_scf(c_unpinned, kpts_sym, 'KRHF', 'FFTDF')
    c_pinned = build_cell(kind, mesh=pinned)
    kpts_sym_p = c_pinned.make_kpts([2,2,2], space_group_symmetry=True, time_reversal_symmetry=True)
    e_pinned, conv_p = run_scf(c_pinned, kpts_sym_p, 'KRHF', 'FFTDF')
    dmesh = abs(e_unpinned - e_pinned)
    print(f"  {kind:9s} pinned_mesh={pinned}  unpinned_mesh={unpinned_mesh}  "
          f"E(pinned)={e_pinned:.12f}  E(unpinned)={e_unpinned:.12f}  "
          f"|dE from mesh alone|={dmesh:.3e}")
    sys.stdout.flush()

print()
print("### run-to-run / thread-count spread: si, KRHF, FFTDF, gamma-centred, mesh pinned ###")
m = pinned_mesh['si']
for threads in [1, 8]:
    os.environ['OMP_NUM_THREADS'] = str(threads)
    os.environ['RAYON_NUM_THREADS'] = str(threads)
    for run_idx in range(2):
        cell = build_cell('si', mesh=m)
        kpts_full = cell.make_kpts([2,2,2], with_gamma_point=True)
        e, conv = run_scf(cell, kpts_full, 'KRHF', 'FFTDF')
        print(f"  threads={threads} run={run_idx} e_tot={e!r} converged={conv}")
        sys.stdout.flush()

print()
print("=== SUMMARY TABLE ===")
print(f"{'kind':9s} {'mesh':16s} {'kmesh':9s} {'method':5s} {'df':5s} {'|dE|':>12s}")
for r in results:
    print(f"{r['kind']:9s} {str(r['mesh']):16s} {r['kmesh_type']:9s} {r['method']:5s} {r['df']:5s} {r['diff']:12.3e}")
