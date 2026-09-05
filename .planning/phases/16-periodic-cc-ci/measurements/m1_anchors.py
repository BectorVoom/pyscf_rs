#!/usr/bin/env python
"""16-01 Task 1 — reproduce upstream's OWN committed KRCCSD/KGCCSD anchors.

    PYTHONPATH=$PWD .venv/bin/python -u \
      .planning/phases/16-periodic-cc-ci/measurements/m1_anchors.py

Every anchor prints, to 17 significant digits: the value upstream's suite pins
in its source, the value the code actually produces today, the residual, and
the decimal place `assertAlmostEqual` actually asserts. The gap between "what
upstream pins" and "what upstream asserts" is the first half of the Phase-16
floor (`16-CONTEXT §2`).
"""

import sys
import time

import numpy as np

import pyscf

assert pyscf.__version__ == "2.12.1", pyscf.__version__

from pyscf.pbc import cc as pbcc
from pyscf.pbc import scf as pbcscf
from pyscf.pbc.tools import make_test_cell
from pyscf.pbc.tools.pbc import super_cell


def report(name, pinned, got, decimals, source):
    resid = abs(got - pinned)
    print(
        f"ANCHOR {name}\n"
        f"  source     {source}\n"
        f"  pinned     {pinned!r}\n"
        f"  produced   {got!r}\n"
        f"  residual   {resid:.17g}\n"
        f"  asserted   {decimals} decimals (tol {0.5 * 10 ** -decimals:.1e})\n"
        f"  verdict    {'REPRODUCES' if resid < 0.5 * 10 ** -decimals else 'DOES NOT REPRODUCE'}",
        flush=True,
    )


def run_kcell(cell, nk):
    """`test_krccsd.py:150-166` verbatim."""
    abs_kpts = cell.make_kpts(nk, wrap_around=True)
    kmf = pbcscf.KRHF(cell, abs_kpts, exxdiv=None)
    kmf.conv_tol = 1e-14
    ekpt = kmf.scf()
    cc = pbcc.kccsd_rhf.RCCSD(kmf)
    cc.conv_tol = 1e-8
    ecc, t1, t2 = cc.kernel()
    return ekpt, ecc


def anchor_311():
    """`test_krccsd.py:172-180` test_311_n1_high_cost."""
    t0 = time.time()
    cell = make_test_cell.test_cell_n1(7.0, [9] * 3)
    escf, ecc = run_kcell(cell, (3, 1, 1))
    report("hf_311", -0.92687629918229486, escf, 8, "test_krccsd.py:177/:179")
    report("cc_311", -0.042702177586414237, ecc, 6, "test_krccsd.py:178/:180")
    print(f"  wall       {time.time() - t0:.1f} s", flush=True)


def anchor_frozen_n3():
    """`test_krccsd.py:206-232` test_frozen_n3, plus its supercell half."""
    t0 = time.time()
    cell = make_test_cell.test_cell_n3([12] * 3)
    nk = (1, 1, 2)
    abs_kpts = cell.make_kpts(nk, with_gamma_point=True)
    kmf = pbcscf.KRHF(cell, abs_kpts, exxdiv=None)
    kmf.conv_tol = 1e-9
    ehf = kmf.scf()
    cc = pbcc.kccsd_rhf.RCCSD(kmf, frozen=[[0], [0, 1]])
    cc.diis_start_cycle = 1
    ecc, t1, t2 = cc.kernel()
    report("ehf_bench", -8.648503065380389, ehf, 6, "test_krccsd.py:210/:225")
    report("ecc_bench", -0.100045112503651, ecc, 6, "test_krccsd.py:211/:226")
    print(f"  wall(kpts) {time.time() - t0:.1f} s", flush=True)

    t1w = time.time()
    mf = super_cell(cell, nk).RHF(exxdiv=None).run()
    report("ehf_bench_supercell/2", -8.648503065380389, mf.e_tot / 2, 5,
           "test_krccsd.py:229")
    ccs = pbcc.RCCSD(mf, frozen=[0, 1, 2])
    ccs.diis_start_cycle = 1
    report("ecc_bench_supercell/2", -0.100045112503651, ccs.kernel()[0] / 2, 6,
           "test_krccsd.py:232")
    print(f"  wall(super) {time.time() - t1w:.1f} s", flush=True)


def anchor_cu_metallic():
    """`test_krccsd.py:338`/`:356` — ecc2_bench / ecc3_bench, mesh [7,7,7].

    Set up EXACTLY as `test_cu_metallic_high_cost` (`:404-419`) does, including
    `scaled_center=[0,0,0]` + `wrap_around=True` and `conv_tol_grad = 1e-6`.
    Note that test carries `@unittest.skip('Results not match')` at `:403` —
    upstream has itself disabled the assertions these anchors come from.
    """
    t0 = time.time()
    cell = make_test_cell.test_cell_cu_metallic([7] * 3)
    assert list(cell.mesh) == [7, 7, 7]
    nk = [1, 1, 2]
    kmf = pbcscf.KRHF(cell, exxdiv=None)
    kmf.kpts = cell.make_kpts(nk, scaled_center=[0.0, 0.0, 0.0], wrap_around=True)
    kmf.conv_tol_grad = 1e-6
    ehf = kmf.scf()
    report("ehf_bench_cu", -52.5393701339723, ehf, 6, "test_krccsd.py:407/:416")
    print(f"  scf wall   {time.time() - t0:.1f} s", flush=True)

    t1w = time.time()
    mycc = pbcc.kccsd_rhf.RCCSD(kmf, frozen=[[2, 3], [0, 1]])
    mycc.diis_start_cycle = 1
    mycc.iterative_damping = 0.05
    mycc.max_cycle = 5
    eris = mycc.ao2mo()
    eris.mo_energy = [f.diagonal().real for f in eris.fock]
    ecc2, _, _ = mycc.kernel(eris=eris)
    report("ecc2_bench", -0.7651806468801496, ecc2, 6, "test_krccsd.py:333/:338")
    print(f"  wall       {time.time() - t1w:.1f} s", flush=True)

    t2w = time.time()
    mycc = pbcc.kccsd_rhf.RCCSD(kmf, frozen=[[1, 17], [0]])
    mycc.diis_start_cycle = 1
    mycc.iterative_damping = 0.05
    mycc.max_cycle = 5
    eris = mycc.ao2mo()
    eris.mo_energy = [f.diagonal().real for f in eris.fock]
    ecc3, _, _ = mycc.kernel(eris=eris)
    report("ecc3_bench", -0.76794053711557086, ecc3, 6, "test_krccsd.py:351/:356")
    ew, _ = mycc.ipccsd(nroots=3, koopmans=True, kptlist=[1])
    for i, pin in enumerate(
        [-3.028339571372944, -2.850636489429295, -2.801491561537961]
    ):
        report(f"ip_root_{i}", pin, ew[0][i], 3, f"test_krccsd.py:{359 + i}")
    ew, _ = mycc.eaccsd(nroots=3, koopmans=True, kptlist=[1])
    for i, pin in enumerate(
        [3.266064683223669, 3.281390137070985, 3.426297911456726]
    ):
        report(f"ea_root_{i}", pin, ew[0][i], 2, f"test_krccsd.py:{364 + i}")
    print(f"  wall       {time.time() - t2w:.1f} s", flush=True)


def anchor_supercell_equivalence():
    """`test_krccsd.py:441-484` — the oracle-free supercell identity and (T)."""
    from pyscf.pbc import dft as pbcdft

    t0 = time.time()
    n = 14
    cell = make_test_cell.test_cell_n3([n] * 3)
    nk = [1, 1, 2]
    kpts = cell.make_kpts(nk)
    kpts -= kpts[0]
    kks = pbcdft.KRKS(cell, kpts)
    ekks = kks.kernel()
    khf = pbcscf.KRHF(cell, kpts)
    khf.__dict__.update(kks.__dict__)
    mycc = pbcc.KRCCSD(khf)
    eris = mycc.ao2mo()
    ekcc, _, _ = mycc.kernel(eris=eris)
    ekcc_t = mycc.ccsd_t(eris=eris)

    supcell = super_cell(cell, nk)
    rks = pbcdft.RKS(supcell)
    rks.kernel()
    rhf = pbcscf.RHF(supcell)
    rhf.__dict__.update(rks.__dict__)
    mycc = pbcc.RCCSD(rhf)
    eris = mycc.ao2mo()
    ercc, _, _ = mycc.kernel(eris=eris)
    ercc_t = mycc.ccsd_t(eris=eris)

    nkprod = np.prod(nk)
    report("ercc/prod(nk)", -0.15632445245405927, ercc / nkprod, 4,
           "test_krccsd.py:478")
    print(
        f"SUPERCELL-EQUIVALENCE (oracle-free, upstream asserts 5 decimals at :479)\n"
        f"  ekcc           {ekcc!r}\n"
        f"  ercc/prod(nk)  {ercc / nkprod!r}\n"
        f"  |diff|         {abs(ekcc - ercc / nkprod):.17g}",
        flush=True,
    )
    report("ercc_t/prod(nk)", -0.00114619248449, ercc_t / nkprod, 5,
           "test_krccsd.py:481")
    print(
        f"(T)-SUPERCELL-EQUIVALENCE (upstream asserts 6 decimals at :482)\n"
        f"  ekcc_t          {ekcc_t!r}\n"
        f"  ercc_t/prod(nk) {ercc_t / nkprod!r}\n"
        f"  |diff|          {abs(ekcc_t - ercc_t / nkprod):.17g}\n"
        f"  ekks            {ekks!r}\n"
        f"  wall            {time.time() - t0:.1f} s",
        flush=True,
    )


ANCHORS = {
    "311": anchor_311,
    "frozen_n3": anchor_frozen_n3,
    "cu": anchor_cu_metallic,
    "supercell": anchor_supercell_equivalence,
}

if __name__ == "__main__":
    print(f"pyscf {pyscf.__version__} at {pyscf.__file__}", flush=True)
    wanted = sys.argv[1:] or list(ANCHORS)
    for name in wanted:
        print(f"\n===== {name} =====", flush=True)
        ANCHORS[name]()
