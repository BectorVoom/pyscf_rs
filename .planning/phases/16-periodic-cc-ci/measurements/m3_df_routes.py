#!/usr/bin/env python
"""16-01 Task 3 — the DF-route split of `KRCCSD e_corr`.

    PYTHONPATH=$PWD .venv/bin/python -u \
      .planning/phases/16-periodic-cc-ci/measurements/m3_df_routes.py

`kccsd_rhf.py:37` imports `GDF, RSGDF` and branches the whole `_ERIS` build on
the mean field's DF class — the same split `kmp2.py:69` makes, which Phase 14
measured at 4.5e-6 Ha at the SCF level on diamond (standing memory
`rsdf-gdf-disagree-on-diamond`). **The Phase-16 gate is therefore stated PER
ROUTE**; a single "matches upstream" number that does not name its backend is
untestable.

Reported per route: the mean-field energy, `KRCCSD e_corr`, and the pairwise
spread of the four routes against each other — the quantity 16-14 compares this
port's own inter-route gap to.
"""

import sys
import time

import numpy as np

import pyscf

assert pyscf.__version__ == "2.12.1", pyscf.__version__

from pyscf.pbc import cc as pbcc
from pyscf.pbc import df as pbcdf
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbcscf

MESH = [15, 15, 15]


def diamond(mesh=MESH):
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
    cell.mesh = mesh
    cell.verbose = 0
    cell.build()
    return cell


def he_fcc(mesh=MESH):
    """The ALL-ELECTRON control — no pseudopotential, so `get_nuc` runs.

    **`gth-szv` cannot host this measurement.** One He atom in `gth-szv` has a
    single AO, so `nocc = 1` and `nvir = 0`: `KRCCSD` has no virtual space at
    all and dies with `IndexError: index 2 is out of bounds for axis 0 with
    size 2`. Phase 15 hit exactly this (`STATE.md`: "`he_fcc` `gth-szv` cannot
    host the `Lov`/MO-first oracles ... `nvir = 0` and the block is empty") and
    used `6-31g`; the same substitution is made here, with the same reason.
    """
    a0 = 3.0
    cell = pbcgto.Cell()
    cell.atom = [("He", (0.0, 0.0, 0.0))]
    cell.a = np.array(
        [[0.0, a0 / 2, a0 / 2], [a0 / 2, 0.0, a0 / 2], [a0 / 2, a0 / 2, 0.0]]
    )
    cell.basis = "6-31g"
    cell.unit = "A"
    cell.mesh = mesh
    cell.verbose = 0
    cell.build()
    return cell


ROUTES = {
    "FFTDF": pbcdf.FFTDF,
    "GDF": pbcdf.GDF,
    "MDF": pbcdf.MDF,
    "RSDF": pbcdf.RSDF,
}


def table(name, cellf, nk):
    print(f"\n===== {name} nk={nk} mesh={MESH} precision=default(1e-8) =====", flush=True)
    out = {}
    for route, cls in ROUTES.items():
        t0 = time.time()
        cell = cellf()
        kpts = cell.make_kpts(nk)
        mf = pbcscf.KRHF(cell, kpts, exxdiv=None)
        mf.with_df = cls(cell, kpts)
        mf.conv_tol = 1e-10
        try:
            ehf = mf.kernel()
            mycc = pbcc.KRCCSD(mf)
            mycc.conv_tol = 1e-9
            ecorr = mycc.kernel()[0]
        except Exception as exc:  # noqa: BLE001 - the failure IS the measurement
            print(f"  {route:6s} FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue
        out[route] = (ehf, ecorr)
        print(
            f"  {route:6s} e_hf {ehf!r}  e_corr {ecorr!r}  ({time.time() - t0:.1f} s)",
            flush=True,
        )
    names = list(out)
    print("  pairwise |Δe_corr| between routes:", flush=True)
    for i, ra in enumerate(names):
        for rb in names[i + 1:]:
            print(
                f"    {ra:6s} vs {rb:6s}  {abs(out[ra][1] - out[rb][1]):.6e}  "
                f"(mean field {abs(out[ra][0] - out[rb][0]):.6e})",
                flush=True,
            )
    return out


if __name__ == "__main__":
    print(f"pyscf {pyscf.__version__} at {pyscf.__file__}", flush=True)
    which = sys.argv[1:] or ["diamond112", "he112", "diamond222"]
    if "diamond112" in which:
        table("diamond gth-szv", diamond, [1, 1, 2])
    if "he112" in which:
        table("he_fcc 6-31g (all-electron control)", he_fcc, [1, 1, 2])
    if "diamond222" in which:
        table("diamond gth-szv", diamond, [2, 2, 2])
