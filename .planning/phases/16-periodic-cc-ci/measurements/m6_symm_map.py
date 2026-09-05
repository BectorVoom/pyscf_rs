#!/usr/bin/env python
"""16-01 Task 6 — price the `symm_map` saving (D-PBC-29 clause 3).

    PYTHONPATH=$PWD .venv/bin/python -u \
      .planning/phases/16-periodic-cc-ci/measurements/m6_symm_map.py

`kccsd_rhf.py:783` builds `khelper.symm_map` and `:798-805` transforms only the
ORBIT REPRESENTATIVES, filling the rest by `transform_symm`. `16-REVIEW.md §3`
derives the saving as "a genuine ~4x"; this measures it, and reports the number
even if it is materially below 4 — `15-REVIEW.md D-15-R-04` is the precedent for
a corollary whose arithmetic was backwards though its conclusion survived.

Two quantities:
  1. `len(symm_map)` against `nkpts**3` at nkpts = 1, 2, 4, 8, 27 (and 64/125
     if they finish) — the COUNT ratio, plus the cost of `build_symm_map`
     itself, which is `O(nkpts^3)` and is why `kccsd_rhf.py:512` builds it
     lazily;
  2. the WALL CLOCK of the `_ERIS` integral loop as written vs a patched
     version that transforms all `nkpts**3` triples directly.
"""

import sys
import time

import numpy as np

import pyscf

assert pyscf.__version__ == "2.12.1", pyscf.__version__

from pyscf.pbc import cc as pbcc
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbcscf
from pyscf.pbc.lib import kpts_helper


def diamond(mesh=(15, 15, 15)):
    a0 = 3.5668
    q = a0 / 4.0
    cell = pbcgto.Cell()
    cell.atom = [("C", (0.0, 0.0, 0.0)), ("C", (q, q, q))]
    cell.a = np.array(
        [[0.0, a0 / 2, a0 / 2], [a0 / 2, 0.0, a0 / 2], [a0 / 2, a0 / 2, 0.0]]
    )
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pade"
    cell.unit = "A"
    cell.mesh = list(mesh)
    cell.verbose = 0
    cell.build()
    return cell


def counts(nk_list):
    print("\n=== orbit counts and build_symm_map cost ===", flush=True)
    cell = diamond()
    for nk in nk_list:
        kpts = cell.make_kpts(nk)
        n = len(kpts)
        t0 = time.time()
        kh = kpts_helper.KptsHelper(cell, kpts, init_symm_map=False)
        t1 = time.time()
        kh.build_symm_map()
        t2 = time.time()
        reps = len(kh.symm_map)
        sizes = [len(v) for v in kh.symm_map.values()]
        distinct = sum(len(set(v)) for v in kh.symm_map.values())
        print(
            f"  nk {nk} nkpts {n:4d}  nkpts^3 {n ** 3:8d}  representatives {reps:8d}  "
            f"ratio {n ** 3 / reps:6.4f}  orbit sizes {sorted(set(sizes))}  "
            f"distinct-members {distinct}  kconserv {t1 - t0:.4f} s  "
            f"build_symm_map {t2 - t1:.4f} s",
            flush=True,
        )


def eris_wallclock(nk):
    """`_ERIS` as written vs an all-triples build, same fixture."""
    print(f"\n=== _ERIS wall clock, diamond gth-szv nk={nk} ===", flush=True)
    cell = diamond()
    kpts = cell.make_kpts(nk)
    mf = pbcscf.KRHF(cell, kpts, exxdiv=None)
    mf.conv_tol = 1e-10
    mf.kernel()
    cc = pbcc.KRCCSD(cc_mf := mf)
    nkpts = len(kpts)

    t0 = time.time()
    eris_symm = pbcc.kccsd_rhf._ERIS(cc, cc.mo_coeff, method="incore")
    t1 = time.time()
    print(f"  symmetry loop ({len(cc.khelper.symm_map)} representatives of "
          f"{nkpts ** 3} triples): {t1 - t0:.3f} s", flush=True)

    # The all-triples build: monkey-patch `build_symm_map` so every triple is
    # its own representative with a singleton orbit and operation 0.
    from collections import OrderedDict

    def all_triples(self, kptlist=None):
        self._operation = np.zeros((self.nkpts,) * 3, dtype=int)
        self.symm_map = OrderedDict()
        for kp in range(self.nkpts):
            for kq in range(self.nkpts):
                for kr in range(self.nkpts):
                    self.symm_map[(kp, kq, kr)] = [(kp, kq, kr)]

    orig = kpts_helper.KptsHelper.build_symm_map
    kpts_helper.KptsHelper.build_symm_map = all_triples
    try:
        cc2 = pbcc.KRCCSD(cc_mf)
        t2 = time.time()
        eris_all = pbcc.kccsd_rhf._ERIS(cc2, cc2.mo_coeff, method="incore")
        t3 = time.time()
    finally:
        kpts_helper.KptsHelper.build_symm_map = orig
    print(f"  all-triples loop ({nkpts ** 3} transforms): {t3 - t2:.3f} s", flush=True)
    print(f"  SPEED RATIO (all / symm) = {(t3 - t2) / (t1 - t0):.3f}x", flush=True)

    for name in ["oooo", "ooov", "oovv", "ovov", "voov", "vovv", "vvvv"]:
        a, b = getattr(eris_symm, name), getattr(eris_all, name)
        if a is None or b is None:
            continue
        d = float(abs(np.asarray(a) - np.asarray(b)).max())
        print(f"  max|{name}_symm - {name}_all| = {d:.17g}", flush=True)
    print("  (the symmetry loop must be BIT-identical to the all-triples one; "
          "16-05 test 5 asserts this)", flush=True)


if __name__ == "__main__":
    print(f"pyscf {pyscf.__version__} at {pyscf.__file__}", flush=True)
    args = sys.argv[1:]
    if not args or "counts" in args:
        counts([[1, 1, 1], [1, 1, 2], [1, 2, 2], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
    if not args or "wall" in args:
        eris_wallclock([2, 2, 2])
