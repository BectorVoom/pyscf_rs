#!/usr/bin/env python
"""17-09 Task 4 oracle -- `KsymAdaptedRCCSD` (`kccsd_rhf_ksymm.py`).

17-01 could not measure this: `crates/pyscf-pbc-cc/src` was a 13-line stub on
2026-09-01 (`gate_mp2.out`'s last section). Phase 16 has since shipped
`KRCCSD`, so the CC half of 17-09 is unblocked and this is its floor.

The fixture is upstream's own (`cc/test/test_kccsd_ksymm.py:25-41`): a two-GTO
He in a 2 Bohr cube at `[2,2,2]`, `KRHF(exxdiv=None).density_fit()`. nocc = 1,
nvir = 1 -- small enough that the whole k-quartet structure is exercised
without the amplitude sizes hiding an index error in noise.

Run from THIS directory:
    PYTHONPATH=<workspace root> ../../../../.venv/bin/python -u gate_kccsd_ksymm.py
"""
import numpy as np
import pyscf
assert pyscf.__version__ == "2.12.1", pyscf.__version__
from pyscf import lib
from pyscf.pbc import gto, scf, cc

L = 2.
He = gto.Cell()
He.verbose = 0
He.a = np.eye(3) * L
He.atom = [['He', (L / 2., L / 2., L / 2.)]]
He.basis = {'He': [[0, (4.0, 1.0)], [0, (1.0, 1.0)]]}
He.space_group_symmetry = True
He.build()

nk = [2, 2, 2]
# `time_reversal_symmetry` is swept because this port's k-symmetric fixtures
# run with it OFF (D-17-07-01 in `17-07-SUMMARY.md`: `little_cogroup_ops`
# indexes `k2opk`'s doubled column space while `symm_adapted_basis` and
# `MORotationMatrix` index `ops`, so `rmat`'s op index goes out of range with
# time reversal on). Upstream's own test uses it ON, so both are recorded.
import sys
TR = (sys.argv[1] == "1") if len(sys.argv) > 1 else True
print(f"time_reversal_symmetry = {TR}")
kpts = He.make_kpts(nk, space_group_symmetry=True, time_reversal_symmetry=TR)
kmf = scf.KRHF(He, kpts, exxdiv=None).density_fit()
kmf.kernel()
print(f"nkpts = {kpts.nkpts}  nkpts_ibz = {kpts.nkpts_ibz}")
k4, w, _ = kpts.make_k4_ibz(sym='s1')
print(f"n_kqrts_ibz = {len(k4)}  of {kpts.nkpts**3} k-triples")
print(f"E_scf(ksymm) = {kmf.e_tot!r}")

kcc = cc.KsymAdaptedRCCSD(kmf)
kcc.kernel()
print(f"emp2(ksymm)   = {kcc.emp2!r}")
print(f"e_corr(ksymm) = {kcc.e_corr!r}")
t1 = kcc.t1.todense()
t2 = kcc.t2.todense()
print(f"lib.fp(t1).max() = {lib.fp(t1).max()!r}")
print(f"t1.shape = {t1.shape}  t2.shape = {t2.shape}")

# The full-BZ reference, from the SAME mean field (`to_khf`), which is what
# `test_vs_krccsd` compares against.
kmf0 = kmf.to_khf()
kccref = cc.krccsd.KRCCSD(kmf0)
kccref.kernel()
print(f"E_scf(full BZ, to_khf) = {kmf0.e_tot!r}")
print(f"emp2(full BZ)   = {kccref.emp2!r}")
print(f"e_corr(full BZ) = {kccref.e_corr!r}")
print(f"|d e_corr| = {abs(kcc.e_corr - kccref.e_corr):.6e}")
print(f"|d t1|max  = {abs(kccref.t1 - t1).max():.6e}")
print(f"|d t2|max  = {abs(kccref.t2 - t2).max():.6e}")

# Element dumps, so a Rust test can gate on more than two scalars.
print("t1_flat =", [complex(x) for x in np.asarray(t1).ravel()])
print("t2_fp =", repr(float(lib.fp(np.asarray(t2).ravel()).real)))
