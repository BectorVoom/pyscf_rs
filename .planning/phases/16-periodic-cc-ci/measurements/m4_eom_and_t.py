#!/usr/bin/env python
"""16-01 Task 4 — EOM roots and (T): separate floors, separately measured.

    PYTHONPATH=$PWD .venv/bin/python -u \
      .planning/phases/16-periodic-cc-ci/measurements/m4_eom_and_t.py [section...]

Sections:

  eom      EOM-IP/EA roots on diamond [1,1,2] and [2,2,2]: run-to-run spread,
           spread across `nroots`, and spread across the Davidson `conv_tol`.
           Upstream asserts these at THREE decimals (`test_krccsd.py:359-366`,
           `test_eom_krccsd.py`); this measures whether that is pessimism or
           necessity.
  triples  `kccsd_t_rhf` vs `kccsd_t_rhf_slow` on the SAME amplitudes. Same
           input, same formula, two implementations — so this is the tight one,
           and it is the gate 16-08 is held to. `kccsd_t_rhf.py:236` runs the C
           kernel `_ccsd.libcc.CCsd_zcontract_t3T`; `kccsd_t_rhf_slow.py` is the
           loop-explicit form this port is gated against (`16-CONTEXT §1.7`).
  ee       `EOMEESinglet` runs; `EOMEETriplet` / `EOMEESpinFlip` / UHF `EOMEE`
           and `_IMDS.make_ee` still refuse. The traceback text is recorded —
           16-10/16-11 quote these line numbers in their refusal payloads and
           an oracle-gated test asserts upstream still raises.
"""

import sys
import time
import traceback

import numpy as np

import pyscf

assert pyscf.__version__ == "2.12.1", pyscf.__version__

from pyscf.pbc import cc as pbcc
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbcscf
from pyscf.pbc.cc import eom_kccsd_rhf, eom_kccsd_uhf

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
    cell.mesh = list(mesh)
    cell.verbose = 0
    cell.build()
    return cell


def converged_cc(nk, conv_tol=1e-9):
    cell = diamond()
    kpts = cell.make_kpts(nk)
    mf = pbcscf.KRHF(cell, kpts, exxdiv=None)
    mf.conv_tol = 1e-10
    mf.kernel()
    mycc = pbcc.KRCCSD(mf)
    mycc.conv_tol = conv_tol
    eris = mycc.ao2mo()
    mycc.kernel(eris=eris)
    return mycc, eris


def section_eom(nk):
    print(f"\n=== EOM-IP/EA, diamond gth-szv nk={nk} mesh={MESH} ===", flush=True)
    mycc, eris = converged_cc(nk)
    print(f"  e_corr {mycc.e_corr!r}", flush=True)

    def roots(kind, nroots, conv_tol):
        # `RCCSD.ipccsd` takes no `conv_tol`; the Davidson threshold comes from
        # the EOM object (`eom_kccsd_ghf.py:kernel` reads `eom.conv_tol`), so
        # the knob is set there and the kernel driven directly.
        cls = eom_kccsd_rhf.EOMIP if kind == "ip" else eom_kccsd_rhf.EOMEA
        eom = cls(mycc)
        eom.conv_tol = conv_tol
        e, _ = eom.kernel(nroots=nroots, koopmans=True, kptlist=[0], eris=eris)
        return np.asarray(e).ravel()

    base = {}
    for kind in ("ip", "ea"):
        t0 = time.time()
        a = roots(kind, 3, 1e-7)
        b = roots(kind, 3, 1e-7)
        base[kind] = a
        print(
            f"  {kind.upper()} nroots=3 conv_tol=1e-7  {[repr(x) for x in a]}  "
            f"({time.time() - t0:.1f} s)",
            flush=True,
        )
        print(f"     run-to-run max|Δ| {np.abs(a - b).max():.3e}", flush=True)
        for ct in (1e-6, 1e-8, 1e-9):
            c = roots(kind, 3, ct)
            print(f"     conv_tol {ct:.0e} vs 1e-7: max|Δ| {np.abs(c - a).max():.3e}",
                  flush=True)
        for nr in (2, 4):
            d = roots(kind, nr, 1e-7)
            m = min(nr, 3)
            print(f"     nroots {nr} vs 3 (first {m}): max|Δ| "
                  f"{np.abs(d[:m] - a[:m]).max():.3e}", flush=True)
    return base


def section_triples(nk):
    print(f"\n=== (T): kccsd_t_rhf vs kccsd_t_rhf_slow, nk={nk} ===", flush=True)
    from pyscf.pbc.cc import kccsd_t_rhf, kccsd_t_rhf_slow

    mycc, eris = converged_cc(nk)
    t0 = time.time()
    fast = kccsd_t_rhf.kernel(mycc, eris, mycc.t1, mycc.t2)
    t1 = time.time()
    slow = kccsd_t_rhf_slow.kernel(mycc, eris, mycc.t1, mycc.t2)
    t2 = time.time()
    print(f"  e_corr        {mycc.e_corr!r}", flush=True)
    print(f"  fast  (kccsd_t_rhf, C kernel at :236)  {fast!r}  ({t1 - t0:.2f} s)",
          flush=True)
    print(f"  slow  (kccsd_t_rhf_slow)               {slow!r}  ({t2 - t1:.2f} s)",
          flush=True)
    print(f"  |fast - slow| = {abs(fast - slow):.17g}", flush=True)
    print(f"  relative      = {abs(fast - slow) / abs(slow):.3e}", flush=True)

    # And the spin-orbital (T) on the same system, via KGCCSD.
    try:
        from pyscf.pbc.cc import kccsd, kccsd_t
        gcc = kccsd.KGCCSD(mycc._scf)
        gcc.conv_tol = 1e-9
        geris = gcc.ao2mo()
        gcc.kernel(eris=geris)
        et = kccsd_t.kernel(gcc, geris)
        print(f"  KGCCSD e_corr {gcc.e_corr!r}", flush=True)
        print(f"  spin-orbital (T) (kccsd_t)             {et!r}", flush=True)
        print(f"  |spinorb - rhf| = {abs(et - fast):.17g}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  spin-orbital (T) FAILED: {type(exc).__name__}: {exc}", flush=True)


def _record_refusal(label, fn):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - the exception IS the measurement
        tb = traceback.format_exc().strip().splitlines()
        print(f"  {label}: RAISES {type(exc).__name__}", flush=True)
        for line in tb[-4:]:
            print(f"      {line}", flush=True)
        return
    print(f"  {label}: DID NOT RAISE — the refusal this port ships would be "
          f"outliving its reason", flush=True)


def section_ee(nk):
    print(f"\n=== EOM-EE surface and refusals, nk={nk} ===", flush=True)
    mycc, eris = converged_cc(nk)

    t0 = time.time()
    try:
        ee = eom_kccsd_rhf.EOMEESinglet(mycc)
        size = ee.vector_size()
        e, _ = ee.kernel(nroots=2, kptlist=[0])
        print(f"  EOMEESinglet (eom_kccsd_rhf.py:1425): vector_size {size}  "
              f"roots {np.asarray(e).ravel()!r}  ({time.time() - t0:.1f} s)",
              flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  EOMEESinglet FAILED: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()

    _record_refusal(
        "EOMEE.vector_size (eom_kccsd_rhf.py:1417)",
        lambda: eom_kccsd_rhf.EOMEE(mycc).vector_size(),
    )
    _record_refusal(
        "EOMEETriplet.kernel (eom_kccsd_rhf.py:1483)",
        lambda: eom_kccsd_rhf.EOMEETriplet(mycc).kernel(nroots=1, kptlist=[0]),
    )
    _record_refusal(
        "EOMEESpinFlip.kernel (eom_kccsd_rhf.py:1489)",
        lambda: eom_kccsd_rhf.EOMEESpinFlip(mycc).kernel(nroots=1, kptlist=[0]),
    )
    _record_refusal(
        "eom_kccsd_uhf.EOMEE (class must not exist, 16-CONTEXT §1.5)",
        lambda: getattr(eom_kccsd_uhf, "EOMEE"),
    )
    _record_refusal(
        "eom_kccsd_uhf._IMDS.make_ee (eom_kccsd_uhf.py:1120)",
        lambda: eom_kccsd_uhf._IMDS.make_ee(object.__new__(eom_kccsd_uhf._IMDS)),
    )


SECTIONS = {
    "eom": lambda: section_eom([1, 1, 2]),
    "eom222": lambda: section_eom([2, 2, 2]),
    "triples": lambda: section_triples([1, 1, 2]),
    "ee": lambda: section_ee([1, 1, 2]),
}

if __name__ == "__main__":
    print(f"pyscf {pyscf.__version__} at {pyscf.__file__}", flush=True)
    for name in sys.argv[1:] or ["triples", "ee", "eom"]:
        SECTIONS[name]()
