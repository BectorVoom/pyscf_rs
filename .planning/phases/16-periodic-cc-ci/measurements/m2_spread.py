#!/usr/bin/env python
"""16-01 Task 2 — the run-to-run / thread / convergence / precision spread.

    PYTHONPATH=$PWD .venv/bin/python -u \
      .planning/phases/16-periodic-cc-ci/measurements/m2_spread.py [section...]

`e_corr` is an ITERATED quantity: two independently converged CCSD runs take
different DIIS paths, so the achievable floor is bounded below by that spread —
not by anything a document asserts. Sections:

  repeat    the same script 5x, same threads
  threads   OMP_NUM_THREADS 1 / 4 / 8 (re-exec, one per child)
  convtol   the conv_tol x conv_tol_normt ladder, and whether t1/t2 plateau
            at the same point as e_corr
  precision cell.precision 1e-8 (default) vs 1e-10 vs 1e-12 -- 17-01 Gate B
            found the floor was integral-screening-limited, not
            convergence-limited, and that is the most transferable correction
            in the project

FIXTURE. `16-01` asks for `diamond` `gth-szv` 2x2x2. At `cell.precision = 1e-8`
that cell's DEFAULT mesh is [47,47,47] and one KRHF at [1,1,2] alone costs 79 s,
so a 2x2x2 conv_tol ladder is hours. The mesh is therefore PINNED at [15,15,15]
for the ladder sections and the pin is reported with every number: a spread
measurement does not need the converged-basis energy, it needs two runs of the
same thing. The `precision` section is the one that varies the pin, since that
is the quantity it is measuring.
"""

import os
import subprocess
import sys
import time

import numpy as np

import pyscf

assert pyscf.__version__ == "2.12.1", pyscf.__version__

from pyscf.pbc import cc as pbcc
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbcscf

MESH = [15, 15, 15]
NK = [2, 2, 2]


def diamond(mesh=None, precision=None):
    """§9.2 `diamond` — C2 fcc a = 3.5668 A, gth-szv/gth-pade."""
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
    cell.verbose = 0
    if precision is not None:
        cell.precision = precision
    if mesh is not None:
        cell.mesh = mesh
    cell.build()
    return cell


def krccsd(cell, nk=NK, conv_tol=1e-9, conv_tol_normt=1e-7, scf_conv=1e-10):
    kpts = cell.make_kpts(nk)
    mf = pbcscf.KRHF(cell, kpts, exxdiv=None)
    mf.conv_tol = scf_conv
    ehf = mf.kernel()
    mycc = pbcc.KRCCSD(mf)
    mycc.conv_tol = conv_tol
    mycc.conv_tol_normt = conv_tol_normt
    ecc, t1, t2 = mycc.kernel()
    return ehf, ecc, t1, t2


def fp(x):
    """`lib.fp`-shaped fingerprint of a nested amplitude list."""
    from pyscf import lib

    flat = np.hstack([np.asarray(b).ravel() for b in np.asarray(x, dtype=object).ravel()]) \
        if isinstance(x, list) else np.asarray(x).ravel()
    return lib.fp(flat)


def section_repeat():
    print(f"# diamond gth-szv nk={NK} mesh={MESH} precision=default(1e-8)", flush=True)
    vals = []
    for i in range(5):
        t0 = time.time()
        ehf, ecc, t1, t2 = krccsd(diamond(mesh=MESH))
        vals.append(ecc)
        print(f"  run {i}  ehf {ehf!r}  e_corr {ecc!r}  ({time.time() - t0:.1f} s)", flush=True)
    vals = np.array(vals)
    print(
        f"REPEAT-SPREAD max-min = {vals.max() - vals.min():.17g}   "
        f"std = {vals.std():.17g}   mean = {vals.mean()!r}",
        flush=True,
    )


def section_threads():
    if os.environ.get("_M2_CHILD"):
        ehf, ecc, _, _ = krccsd(diamond(mesh=MESH))
        print(f"THREADS {os.environ.get('OMP_NUM_THREADS')} {ecc!r}", flush=True)
        return
    vals = {}
    for n in ["1", "4", "8"]:
        env = dict(os.environ, OMP_NUM_THREADS=n, MKL_NUM_THREADS=n,
                   OPENBLAS_NUM_THREADS=n, _M2_CHILD="1")
        out = subprocess.run(
            [sys.executable, "-u", __file__, "threads"], env=env,
            capture_output=True, text=True,
        )
        line = [l for l in out.stdout.splitlines() if l.startswith("THREADS")]
        if not line:
            print(f"  threads {n}: FAILED\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}", flush=True)
            continue
        vals[n] = float(line[0].split()[2])
        print(f"  OMP_NUM_THREADS={n}  e_corr {vals[n]!r}", flush=True)
    if len(vals) > 1:
        v = np.array(list(vals.values()))
        print(f"THREAD-SPREAD max-min = {v.max() - v.min():.17g}", flush=True)


def section_convtol():
    print(f"# diamond gth-szv nk={NK} mesh={MESH}", flush=True)
    from pyscf import lib

    ref = None
    for ct, ctn in [(1e-7, 1e-5), (1e-8, 1e-6), (1e-9, 1e-7), (1e-10, 1e-8), (1e-11, 1e-9)]:
        t0 = time.time()
        ehf, ecc, t1, t2 = krccsd(diamond(mesh=MESH), conv_tol=ct, conv_tol_normt=ctn)
        f1 = lib.fp(np.hstack([np.asarray(b).ravel() for b in t1]))
        f2 = lib.fp(np.hstack([np.asarray(b).ravel() for b in np.asarray(t2).ravel()]))
        if ref is None:
            ref = (ecc, f1, f2)
        print(
            f"  conv_tol {ct:.0e} normt {ctn:.0e}  e_corr {ecc!r}  "
            f"d(e_corr) {abs(ecc - ref[0]):.3e}  "
            f"fp(t1) {f1!r}  fp(t2) {f2!r}  ({time.time() - t0:.1f} s)",
            flush=True,
        )
        ref = (ecc, f1, f2)
    print("  (d(e_corr) is against the PREVIOUS ladder rung: the plateau is "
          "where it stops shrinking)", flush=True)


def section_precision():
    print(f"# diamond gth-szv nk={NK}, cell.precision varied, mesh left DEFAULT", flush=True)
    for prec in [1e-8, 1e-10, 1e-12]:
        cell = diamond(precision=prec)
        t0 = time.time()
        ehf, ecc, _, _ = krccsd(cell)
        print(
            f"  precision {prec:.0e}  mesh {list(cell.mesh)}  ehf {ehf!r}  "
            f"e_corr {ecc!r}  ({time.time() - t0:.1f} s)",
            flush=True,
        )


SECTIONS = {
    "repeat": section_repeat,
    "threads": section_threads,
    "convtol": section_convtol,
    "precision": section_precision,
}

if __name__ == "__main__":
    print(f"pyscf {pyscf.__version__} at {pyscf.__file__}", flush=True)
    for name in sys.argv[1:] or list(SECTIONS):
        print(f"\n===== {name} =====", flush=True)
        SECTIONS[name]()
