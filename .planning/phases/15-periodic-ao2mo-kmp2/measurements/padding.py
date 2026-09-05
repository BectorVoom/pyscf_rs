import numpy as np
from pyscf import __version__
from pyscf.pbc.mp import kmp2

assert __version__ == "2.12.1", __version__


class FakeMP:
    frozen = None
    _nocc = None
    _nmo = None
    nkpts = 3
    mo_occ = [
        np.array([2, 2, 0, 0, 0, 0.0]),
        np.array([2, 2, 2, 0, 0, 0.0]),
        np.array([2, 2, 0, 0, 0.0]),
    ]

    def get_nocc(self, per_kpoint=False):
        return kmp2.get_nocc(self, per_kpoint=per_kpoint)

    def get_nmo(self, per_kpoint=False):
        return kmp2.get_nmo(self, per_kpoint=per_kpoint)

    def get_frozen_mask(self):
        return kmp2.get_frozen_mask(self)


mp = FakeMP()
for frozen in (None, 1, [0, 1], [[0], [0, 1], [1]]):
    mp.frozen = frozen
    print(f"frozen={frozen!r}")
    print("nocc_per_k=", kmp2.get_nocc(mp, per_kpoint=True))
    print("nmo_per_k=", kmp2.get_nmo(mp, per_kpoint=True))
    print("nocc_dense=", kmp2.get_nocc(mp))
    print("nmo_dense=", kmp2.get_nmo(mp))
    print("mask=", [x.tolist() for x in kmp2.get_frozen_mask(mp)])
    print("split=", tuple([x.tolist() for x in y] for y in kmp2.padding_k_idx(mp, "split")))
    print("joint=", [x.tolist() for x in kmp2.padding_k_idx(mp, "joint")])
