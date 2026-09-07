#!/usr/bin/env python
"""17-09 Task 1 oracle -- `KPoints.make_k4_ibz(sym='s2')` (kpts.py:218-283,
:293-300), the k-quartet set `kmp2_ksymm.kernel` loops over.

`sym='s2'` is the ONE piece of `kpts.py` that 17-05 left refusing
(`PbcSymmError::UnsupportedK4Symmetry`), because at the time its only named
consumer was 17-09. It has no oracle-free invariant strong enough to stand
alone -- the weights summing to 1 is necessary but very far from sufficient,
and the *count* is what the phase's speed claim rests on -- so it is gated
against upstream directly.

Run from THIS directory:
    PYTHONPATH=<workspace root> ../../../../.venv/bin/python -u gate_k4_s2.py
"""
import numpy as np
import pyscf
assert pyscf.__version__ == "2.12.1", pyscf.__version__
from pyscf.pbc import gto


def dump(name, cell, nk, tr):
    kpts = cell.make_kpts(nk, space_group_symmetry=True, time_reversal_symmetry=tr)
    k4_s1, w1, _ = kpts.make_k4_ibz(sym='s1')
    k4, w, bz2ibz = kpts.make_k4_ibz(sym='s2')
    n3 = kpts.nkpts ** 3
    print(f"--- {name} {nk} time_reversal={tr}")
    print(f"nkpts={kpts.nkpts} nkpts_ibz={kpts.nkpts_ibz} "
          f"n_tuples={n3} n_s1={len(k4_s1)} n_s2={len(k4)}")
    print(f"weight_sum_s2 = {float(np.sum(w))!r}")

    # Is the s2 fold SOUND (every class member really is equivalent to the
    # class representative), and is it COMPLETE (every triple lands in the
    # same class as its dummy-index partner)?
    #
    # It is sound and it is NOT complete. The refine pass (`kpts.py:236-273`)
    # searches later representatives whose four k-indices form the right
    # multiset and then walks that representative's s1 star; when the partner
    # lives in a star whose representative has a DIFFERENT multiset, the
    # search misses it and the two classes stay separate. That costs
    # k-quartets, never accuracy -- so 17-09's Rust gate asserts SOUNDNESS as
    # an invariant and pins the measured incompleteness as a number, rather
    # than asserting a completeness upstream does not have.
    _, _, bz2ibz_s1, _, _, _ = kpts.make_ktuples_ibz(ntuple=3)
    kconserv = kpts.get_kconserv()
    n = kpts.nkpts
    unsound = 0
    partner_split = 0
    for ki in range(n):
        for kj in range(n):
            for ka in range(n):
                kb = kconserv[ki, ka, kj]
                t = ki * n * n + kj * n + ka
                tsw = kj * n * n + ki * n + kb
                rep = k4[bz2ibz[t]]
                trep = rep[0] * n * n + rep[1] * n + rep[2]
                if not (bz2ibz_s1[t] == bz2ibz_s1[trep]
                        or bz2ibz_s1[tsw] == bz2ibz_s1[trep]):
                    unsound += 1
                if bz2ibz[t] != bz2ibz[tsw]:
                    partner_split += 1
    print(f"unsound = {unsound}   partner_split = {partner_split}   of {n3}")
    print("k4_s2 =", [list(map(int, r)) for r in k4])
    print("weight_s2_x_nkpts3 =", [int(round(float(x) * n3)) for x in w])
    print("bz2ibz_s2 =", [int(x) for x in bz2ibz])
    print()


half = 5.4306 / 2
q = 5.4306 / 4
si = gto.Cell()
si.atom = f"Si 0. 0. 0.\nSi {q} {q} {q}"
si.a = [[0., half, half], [half, 0., half], [half, half, 0.]]
si.basis = 'gth-szv'
si.pseudo = 'gth-pade'
si.space_group_symmetry = True
si.verbose = 0
si.build()

a0 = 3.5668
q = a0 / 4
h = a0 / 2
dia = gto.Cell()
dia.atom = f"C 0. 0. 0.\nC {q} {q} {q}"
dia.a = [[0., h, h], [h, 0., h], [h, h, 0.]]
dia.basis = 'gth-szv'
dia.pseudo = 'gth-pade'
dia.space_group_symmetry = True
dia.verbose = 0
dia.build()

for tr in (False, True):
    dump("si", si, [1, 1, 2], tr)
    dump("si", si, [2, 2, 2], tr)
    dump("diamond", dia, [2, 2, 2], tr)
