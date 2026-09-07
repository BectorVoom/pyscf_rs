"""Does kccsd_rhf_ksymm.py:112's stale `kc` change the answer?

`update_amps`'s second T1 quartet loop unpacks `kk, kl, ki, kd = kq` and then
tests `if kk == ka and kl == kc` -- but `kc` is not bound in that loop; it is
left over from the PRECEDING loop's `ki, kk, kc, kd = kq`. Momentum
conservation on the quartet gives kd = kk + kl - ki, and ka = ki, so when
kk == ka the partner index is kl itself: the intended condition is `kk == ka`
alone, with `t1[kl]`.
"""
import numpy as np, pyscf
from pyscf import lib
from pyscf.pbc import gto, scf, cc
from pyscf.pbc.cc import kccsd_rhf_ksymm as m
from pyscf.pbc.cc import kintermediates_rhf_ksymm as imdk
from pyscf.pbc.mp.kmp2 import padding_k_idx
from pyscf.pbc.cc.kccsd_rhf import _get_epq
from pyscf.pbc.lib import ktensor
einsum = lib.einsum

orig = m.update_amps

def fixed_update_amps(cc, t1, t2, eris):
    """Byte-copy of upstream's update_amps with the ONE condition corrected."""
    import types, inspect, re
    src = inspect.getsource(orig)
    src = src.replace("if kk == ka and kl == kc:\n            tau_term_1 += einsum('ka,lc->klac', t1[ka], t1[kc])",
                      "if kk == ka:\n            tau_term_1 += einsum('ka,lc->klac', t1[ka], t1[kl])")
    src = src.replace("def update_amps(", "def _patched_update_amps(")
    g = dict(m.__dict__)
    exec(compile(src, "<patched>", "exec"), g)
    return g["_patched_update_amps"](cc, t1, t2, eris)

L = 2.
He = gto.Cell(); He.verbose=0; He.a=np.eye(3)*L
He.atom=[['He',(L/2.,L/2.,L/2.)]]
He.basis={'He':[[0,(4.0,1.0)],[0,(1.0,1.0)]]}
He.space_group_symmetry=True; He.build()
kpts = He.make_kpts([2,2,2], space_group_symmetry=True, time_reversal_symmetry=True)
kmf = scf.KRHF(He, kpts, exxdiv=None).density_fit(); kmf.kernel()

kcc = cc.KsymAdaptedRCCSD(kmf); kcc.kernel()
e_orig = kcc.e_corr

m.update_amps = fixed_update_amps
type(kcc).update_amps = staticmethod(fixed_update_amps) if False else fixed_update_amps
kcc2 = cc.KsymAdaptedRCCSD(kmf)
kcc2.update_amps = lambda t1,t2,eris: fixed_update_amps(kcc2,t1,t2,eris)
kcc2.kernel()
e_fixed = kcc2.e_corr

kmf0 = kmf.to_khf()
ref = cc.krccsd.KRCCSD(kmf0); ref.kernel()

print(f"e_corr upstream ksymm (stale kc) = {e_orig!r}")
print(f"e_corr ksymm with kk==ka fix     = {e_fixed!r}")
print(f"e_corr full BZ KRCCSD            = {ref.e_corr!r}")
print(f"|stale - fixed|   = {abs(e_orig-e_fixed):.6e}")
print(f"|stale - full BZ| = {abs(e_orig-ref.e_corr):.6e}")
print(f"|fixed - full BZ| = {abs(e_fixed-ref.e_corr):.6e}")
