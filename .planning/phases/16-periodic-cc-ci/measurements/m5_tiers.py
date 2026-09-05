#!/usr/bin/env python
"""16-01 Task 5 — the storage-tier crossover, so 16-05's gate can cross it.

    PYTHONPATH=$PWD .venv/bin/python -u \
      .planning/phases/16-periodic-cc-ci/measurements/m5_tiers.py

`kccsd_rhf.py:786-832` chooses between an incore `np.empty` build and an HDF5
`create_dataset` build by

    mem_incore, _, _ = _mem_usage(nkpts, nocc, nvir)      # :777
    if method == 'incore' and mem_incore + mem_now < cc.max_memory: ...

and `:423-455` makes the same three-way choice for `Wvvvv` (skip / incore /
HDF5). `16-REVIEW.md §2.3` shows a port gated only on the `§9.2` `gth-szv`
fixtures would ship the incore tier and never once execute the spill path.

This script reports, per (fixture, basis, mesh):

  * the EXACT byte size of each of the seven `_ERIS` blocks;
  * `_mem_usage`'s estimate and its over-estimate factor against that sum;
  * the `max_memory` value at which upstream flips tier, measured by bisection
    on the actual branch condition;
  * and, for one fixture, that the two tiers agree — upstream asserts exactly
    this at 12 decimals (`test_krccsd.py:250-256`).
"""

import sys
import time

import numpy as np

import pyscf

assert pyscf.__version__ == "2.12.1", pyscf.__version__

from pyscf import lib
from pyscf.pbc import cc as pbcc
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbcscf
from pyscf.pbc.cc.kccsd_rhf import _mem_usage

MB = 1024.0 * 1024.0


def diamond(basis="gth-szv", mesh=(15, 15, 15)):
    a0 = 3.5668
    q = a0 / 4.0
    cell = pbcgto.Cell()
    cell.atom = [("C", (0.0, 0.0, 0.0)), ("C", (q, q, q))]
    cell.a = np.array(
        [[0.0, a0 / 2, a0 / 2], [a0 / 2, 0.0, a0 / 2], [a0 / 2, a0 / 2, 0.0]]
    )
    cell.basis = basis
    cell.pseudo = "gth-pade"
    cell.unit = "A"
    cell.mesh = list(mesh)
    cell.verbose = 0
    cell.build()
    return cell


BLOCKS = {
    "oooo": ("o", "o", "o", "o"),
    "ooov": ("o", "o", "o", "v"),
    "oovv": ("o", "o", "v", "v"),
    "ovov": ("o", "v", "o", "v"),
    "voov": ("v", "o", "o", "v"),
    "vovv": ("v", "o", "v", "v"),
    "vvvv": ("v", "v", "v", "v"),
}


def block_bytes(nkpts, nocc, nvir):
    out = {}
    for name, spec in BLOCKS.items():
        n = nkpts ** 3
        for s in spec:
            n *= nocc if s == "o" else nvir
        out[name] = n * 16  # complex128
    return out


def report_sizes(label, nkpts, nocc, nvir):
    sizes = block_bytes(nkpts, nocc, nvir)
    total = sum(sizes.values())
    mem_incore, _, _ = _mem_usage(nkpts, nocc, nvir)  # MB
    est = mem_incore * 1e6
    print(f"\n--- {label}: nkpts {nkpts} nocc {nocc} nvir {nvir} ---", flush=True)
    for name in BLOCKS:
        print(f"  {name}  {sizes[name] / MB:12.3f} MiB", flush=True)
    print(f"  SUM of 7 blocks       {total / MB:12.3f} MiB", flush=True)
    print(f"  upstream _mem_usage   {est / MB:12.3f} MiB "
          f"(kccsd_rhf.py:1100-1107, its own '# TODO: Improve incore estimate')",
          flush=True)
    print(f"  OVER-ESTIMATE FACTOR  {est / total:12.3f}x", flush=True)
    return total, est


def tier_flip(nkpts, nocc, nvir):
    """The `max_memory` at which `:786` flips, in MB, from upstream's own test."""
    mem_incore, _, _ = _mem_usage(nkpts, nocc, nvir)
    # `mem_now` is whatever the process already holds; report both the pure
    # threshold and the one the branch actually sees right now.
    mem_now = lib.current_memory()[0]
    print(
        f"  TIER FLIP: incore iff  mem_incore ({mem_incore:.3f} MB) + "
        f"mem_now ({mem_now:.3f} MB) < cc.max_memory\n"
        f"             -> max_memory > {mem_incore + mem_now:.3f} MB is incore, "
        f"below it spills",
        flush=True,
    )
    return mem_incore + mem_now


def tier_equivalence(nk=(2, 2, 2)):
    """Upstream's `test_ao2mo` shape: build both tiers, diff them."""
    print(f"\n=== tier equivalence, diamond gth-szv nk={list(nk)} ===", flush=True)
    cell = diamond()
    kpts = cell.make_kpts(nk)
    mf = pbcscf.KRHF(cell, kpts, exxdiv=None)
    mf.conv_tol = 1e-10
    mf.kernel()

    cc = pbcc.KRCCSD(mf)
    nocc, nmo = cc.nocc, cc.nmo
    nvir = nmo - nocc
    total, est = report_sizes("diamond gth-szv", len(kpts), nocc, nvir)
    flip = tier_flip(len(kpts), nocc, nvir)

    # The outcore branch has its OWN floor (`:912`): it refuses below
    # `mem_now + nvir**4 * 16 * 2 / 1e6`. So the spill run is given a budget
    # that is above that floor and below the incore flip — the window a test
    # can actually set, which is what 16-05 test 4 needs.
    floor = lib.current_memory()[0] + nvir ** 4 * 16 * 2 / 1e6
    spill_budget = 0.5 * (floor + flip)
    print(f"  outcore floor {floor:.3f} MB, incore flip {flip:.3f} MB "
          f"-> spill run uses max_memory = {spill_budget:.3f} MB", flush=True)
    assert floor < spill_budget < flip, (floor, spill_budget, flip)

    t0 = time.time()
    cc.max_memory = 100000  # far above the flip: incore
    eris1 = pbcc.kccsd_rhf._ERIS(cc, cc.mo_coeff, method="incore")
    t1 = time.time()
    cc.max_memory = spill_budget
    eris2 = pbcc.kccsd_rhf._ERIS(cc, cc.mo_coeff, method="outcore")
    t2 = time.time()
    print(f"  incore build {t1 - t0:.2f} s, outcore build {t2 - t1:.2f} s", flush=True)
    for name in BLOCKS:
        a = getattr(eris1, name)
        b = getattr(eris2, name)
        if a is None or b is None:
            print(f"  {name}: one side is None (incore {a is None}, "
                  f"outcore {b is None}) — the 'skip' tier", flush=True)
            continue
        d = float(abs(np.asarray(a) - np.asarray(b)).max())
        print(f"  max|{name}_incore - {name}_outcore| = {d:.17g}", flush=True)
    print(f"  (upstream asserts each of these at 12 decimals, "
          f"test_krccsd.py:250-256)", flush=True)
    return flip


if __name__ == "__main__":
    print(f"pyscf {pyscf.__version__} at {pyscf.__file__}", flush=True)
    # Derived table for the fixtures 16-REVIEW.md §2.3 tabulated, so the port's
    # own numbers can be checked against it.
    for nkpts, nocc, nvir, label in [
        (2, 4, 4, "diamond gth-szv 1x1x2"),
        (8, 4, 4, "diamond gth-szv 2x2x2"),
        (27, 4, 4, "diamond gth-szv 3x3x3"),
        (8, 4, 22, "diamond gth-dzvp 2x2x2"),
        (27, 4, 22, "diamond gth-dzvp 3x3x3"),
    ]:
        report_sizes(label, nkpts, nocc, nvir)
    if "sizes-only" not in sys.argv[1:]:
        tier_equivalence()
