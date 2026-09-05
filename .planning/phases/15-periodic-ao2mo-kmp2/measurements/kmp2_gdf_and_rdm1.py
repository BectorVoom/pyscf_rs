"""Two `tests/kmp2.rs` assertions, measured against upstream instead of guessed.

Both were written when this port's GDF mean field was 1.461e-1 Ha from
upstream's (`15-VERIFICATION.md` row 4, fixed 2026-09-05 --
`14-VERIFICATION.md §11`), and both encode something upstream does not do:

  1. `fft_matches_upstream_and_gdf_integral_routes_agree` pinned the GDF KMP2
     `e_corr` to WHAT THIS PORT PRODUCED, by its own comment, because there was
     no point gating a correlation energy on a broken reference. With the mean
     field fixed there is a real oracle, so this records it -- for both of
     upstream's routes, `with_df_ints` True (the `Lov` route) and False (the
     four-index AO2MO route).

  2. `diamond_anchor_and_without_t2` asserted `Tr(gamma_k) == nelec` for EVERY
     k-point. That identity does not hold per k-point in k-point MP2 and
     upstream does not satisfy it: the occupied- and virtual-block corrections
     cancel only after averaging over k. This records the per-k traces so the
     test can assert what is actually true, `(1/Nk) sum_k Tr(gamma_k) == nelec`.

Run:
  PYTHONPATH=$PWD .venv/bin/python -u \
    .planning/phases/15-periodic-ao2mo-kmp2/measurements/kmp2_gdf_and_rdm1.py
"""
import numpy as np
import pyscf
from pyscf.pbc import df, gto, mp, scf

assert pyscf.__version__ == "2.12.1", pyscf.__version__


def helium_631g():
    h = 2.834589
    c = gto.Cell()
    c.a = [[0., h, h], [h, 0., h], [h, h, 0.]]
    c.atom = [('He', (0., 0., 0.))]
    c.basis = '6-31g'
    c.unit = 'Bohr'
    c.mesh = [9, 9, 9]
    c.verbose = 0
    return c.build()


def diamond_anchor():
    h, q = 3.370137329, 1.685068664391
    c = gto.Cell()
    c.a = [[0., h, h], [h, 0., h], [h, h, 0.]]
    c.atom = [('C', (0., 0., 0.)), ('C', (q, q, q))]
    c.basis = 'gth-szv'
    c.pseudo = 'gth-pade'
    c.unit = 'Bohr'
    c.verbose = 0
    return c.build()


print("--- 1. KMP2 e_corr on He/6-31g [1,1,2], exxdiv=None, conv_tol=1e-11 ---")
cell = helium_631g()
kpts = cell.make_kpts([1, 1, 2])
for name, builder in (("fftdf", df.FFTDF), ("gdf", df.GDF)):
    mf = scf.KRHF(cell, kpts, exxdiv=None)
    mf.with_df = builder(cell, kpts)
    mf.conv_tol = 1e-11
    e_hf = mf.kernel()
    print(f"{name}: e_hf   = {e_hf!r}")
    for with_df_ints in ((True, False) if name == "gdf" else (False,)):
        pt = mp.KMP2(mf)
        pt.with_df_ints = with_df_ints
        print(f"{name}: e_corr = {pt.kernel()[0]!r}   "
              f"(with_df_ints={with_df_ints})")

print()
print("--- 2. per-k RDM1 traces on the diamond anchor [1,1,2] ---")
print("    Tr(gamma_k) is NOT nelec per k-point; only the k-AVERAGE is.")
dia = diamond_anchor()
dkpts = dia.make_kpts([1, 1, 2])
mf = scf.KRHF(dia, dkpts, exxdiv=None)
mf.conv_tol = 1e-11
mf.kernel()
pt = mp.KMP2(mf)
e_corr, t2 = pt.kernel()
print(f"e_corr = {e_corr!r}")
dm = pt.make_rdm1(t2)
traces = [np.trace(d).real for d in dm]
for k, t in enumerate(traces):
    print(f"  Tr(gamma_{k}) = {t!r}")
print(f"  mean          = {sum(traces) / len(traces)!r}")
print(f"  cell.nelectron = {dia.nelectron}")
print(f"  |mean - nelec| = {abs(sum(traces) / len(traces) - dia.nelectron):.3e}")
