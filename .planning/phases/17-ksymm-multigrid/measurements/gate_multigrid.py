#!/usr/bin/env python
"""Gate E -- the multigrid floor (17-01-PLAN.md Task 5).

MultiGridNumInt (v1) and MultiGridNumInt (v2, imported as MultiGridNumInt2)
against the reference pbc.dft.numint / FFTDF, on diamond and si, at two
meshes each, so the phase learns whether the residual is a mesh-convergence
artefact or a definitional one. Also records the v1-vs-v2 difference.
"""
import sys, time
import numpy as np
import pyscf
assert pyscf.__version__ == "2.12.1", pyscf.__version__
from pyscf.pbc import gto, df, dft
from pyscf.pbc.dft import multigrid
from pyscf.pbc.dft.multigrid import multigrid as multigrid_v1_mod

def build_cell(kind, mesh):
    c = gto.Cell()
    if kind == 'diamond':
        half = 3.5668/2; q = 3.5668/4
        c.atom = f"C 0. 0. 0.\nC {q} {q} {q}"
        c.a = [[0.,half,half],[half,0.,half],[half,half,0.]]
        c.basis = 'gth-szv'; c.pseudo = 'gth-pade'
    elif kind == 'si':
        half = 5.4306/2; q = 5.4306/4
        c.atom = f"Si 0. 0. 0.\nSi {q} {q} {q}"
        c.a = [[0.,half,half],[half,0.,half],[half,half,0.]]
        c.basis = 'gth-szv'; c.pseudo = 'gth-pade'
    else:
        raise ValueError(kind)
    c.mesh = [int(x) for x in mesh]
    c.verbose = 0
    c.build()
    return c

for kind in ['diamond', 'si']:
    c0 = build_cell(kind, mesh=[1,1,1])  # dummy, just to read the default mesh
    default_mesh = None
c0_diamond = gto.M(a=np.eye(3)*3.5668,
                    atom='C 0. 0. 0.\nC 1.8 1.8 1.8', basis='gth-szv',
                    pseudo='gth-pade', verbose=0)
c0_si = gto.M(a=[[0.,2.7153,2.7153],[2.7153,0.,2.7153],[2.7153,2.7153,0.]],
              atom=f'Si 0. 0. 0.\nSi {5.4306/4} {5.4306/4} {5.4306/4}',
              basis='gth-szv', pseudo='gth-pade', verbose=0)
default_meshes = {'diamond': [int(x) for x in c0_diamond.mesh],
                   'si': [int(x) for x in c0_si.mesh]}
print("default meshes:", default_meshes)

def two_meshes(kind):
    d = default_meshes[kind]
    coarse = [max(int(round(x*0.6)), 12) for x in d]
    return [d, coarse]

results = []

def report(kind, mesh, label, ref, out, extra=""):
    d = abs(ref - out).max() if hasattr(ref, 'shape') else abs(ref - out)
    print(f"  [{kind:8s} mesh={mesh} {label:28s}] max|diff|={d:.3e} {extra}")
    sys.stdout.flush()
    results.append((kind, tuple(mesh), label, d))
    return d

for kind in ['diamond', 'si']:
    for mesh in two_meshes(kind):
        print(f"--- {kind} mesh={mesh} ---")
        cell = build_cell(kind, mesh)

        # get_pp: FFTDF reference vs v1 vs v2
        ref_pp = df.FFTDF(cell).get_pp()
        v1 = multigrid.MultiGridNumInt(cell)
        out_pp_v1 = v1.get_pp()
        report(kind, mesh, "get_pp v1 vs FFTDF", ref_pp, out_pp_v1)

        v2 = multigrid.MultiGridNumInt2(cell)
        out_pp_v2 = v2.get_pp(return_full=True)
        report(kind, mesh, "get_pp v2 vs FFTDF", ref_pp, out_pp_v2)
        report(kind, mesh, "get_pp v1 vs v2", out_pp_v1, out_pp_v2)

        # get_nuc: FFTDF vs v1 (module-level get_nuc for v2 needs the
        # dedicated grad-context path; skip if unavailable)
        ref_nuc = df.FFTDF(cell).get_nuc()
        out_nuc_v1 = v1.get_nuc()
        report(kind, mesh, "get_nuc v1 vs FFTDF", ref_nuc, out_nuc_v1)

        # get_j: FFTDF vs v1, on a random Hermitian density matrix
        np.random.seed(2)
        nao = cell.nao
        dm = np.random.random((nao, nao)) * .2
        dm = dm + dm.T + np.eye(nao)
        ref_j = df.FFTDF(cell).get_jk(dm, with_k=False)[0]
        out_j_v1 = v1.get_j(dm)
        report(kind, mesh, "get_j v1 vs FFTDF", ref_j, out_j_v1)

        # vxc / exc / ecoul for LDA and a GGA, v1 vs reference numint
        ni_ref = dft.numint.NumInt()
        grids = dft.gen_grid.UniformGrids(cell)
        for xc in ['lda,vwn', 'pbe,pbe']:
            n0, exc0, vxc0 = ni_ref.nr_rks(cell, grids, xc, dm)
            n1, exc1, vxc1 = multigrid_v1_mod.nr_rks(v1, xc, dm)
            dvxc = abs(vxc0 - vxc1).max()
            dexc = abs(exc0 - exc1)
            print(f"  [{kind:8s} mesh={mesh} v1 nr_rks xc={xc:9s}] "
                  f"|dvxc|={dvxc:.3e} |dexc|={dexc:.3e}")
            results.append((kind, tuple(mesh), f"v1 vxc {xc}", dvxc))
            results.append((kind, tuple(mesh), f"v1 exc {xc}", dexc))
            sys.stdout.flush()

print()
print("=== converged KRKS e_tot: reference numint vs multigrid v1 vs v2 ===")
for kind in ['diamond', 'si']:
    mesh = default_meshes[kind]
    cell = build_cell(kind, mesh)
    for xc in ['lda,vwn']:
        mf_ref = dft.RKS(cell)
        mf_ref.xc = xc
        mf_ref.conv_tol = 1e-10
        mf_ref.chkfile = None
        t0 = time.time()
        mf_ref.kernel()
        t_ref = time.time() - t0

        mf_v1 = dft.RKS(cell)
        mf_v1.xc = xc
        mf_v1.conv_tol = 1e-10
        mf_v1.chkfile = None
        mf_v1._numint = multigrid.MultiGridNumInt(cell)
        t0 = time.time()
        mf_v1.kernel()
        t_v1 = time.time() - t0

        mf_v2 = dft.RKS(cell)
        mf_v2.xc = xc
        mf_v2.conv_tol = 1e-10
        mf_v2.chkfile = None
        mf_v2._numint = multigrid.MultiGridNumInt2(cell)
        t0 = time.time()
        mf_v2.kernel()
        t_v2 = time.time() - t0

        print(f"  {kind:8s} xc={xc:9s} E_ref={mf_ref.e_tot:.10f} (t={t_ref:.1f}s) "
              f"E_v1={mf_v1.e_tot:.10f} (t={t_v1:.1f}s, |dE|={abs(mf_v1.e_tot-mf_ref.e_tot):.3e}, ratio={t_ref/max(t_v1,1e-9):.2f}x) "
              f"E_v2={mf_v2.e_tot:.10f} (t={t_v2:.1f}s, |dE|={abs(mf_v2.e_tot-mf_ref.e_tot):.3e}, ratio={t_ref/max(t_v2,1e-9):.2f}x)")
        sys.stdout.flush()

print()
print("=== SUMMARY: mesh dependence (does the residual shrink with mesh?) ===")
for kind in ['diamond', 'si']:
    rows = [r for r in results if r[0] == kind]
    for label in sorted(set(r[2] for r in rows)):
        vals = [(r[1], r[3]) for r in rows if r[2] == label]
        print(f"  {kind:8s} {label:28s}: {vals}")
