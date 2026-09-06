#!/usr/bin/env python
"""Phase 16 oracle emitter (vendored PySCF 2.12.1).

One section per invocation, so a Rust test pays only for what it diffs:

    PYTHONPATH=$PWD .venv/bin/python -u \
      .planning/phases/16-periodic-cc-ci/measurements/oracle_phase16.py <section>

Sections:

  eris      the seven `_ERIS` blocks + fock + mo_energy, diamond gth-szv [1,1,2]
  imds      the cc_* intermediates on a FIXED synthetic t1/t2, same cell
  krccsd    e_hf / emp2 / e_corr / t1 / t2 fingerprints, same cell
  krccsd222 the same at [2,2,2]
  triples   the (T) correction, fast and slow
  gamma     the [1,1,1] reduction

Numeric blocks are emitted as

    BEGIN <name> n=<count>
    <%.17g values, 8 per line; complex is interleaved re, im>
    END <name>

`%.17g` round-trips an f64 exactly, so the Rust side compares the same bits
upstream produced. The synthetic amplitudes in `imds` come from a fixed
SplitMix64 stream so both sides see the same numbers with no file exchange.
"""

import sys

import numpy as np

import pyscf

assert pyscf.__version__ == "2.12.1", pyscf.__version__

from pyscf.pbc import cc as pbcc
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbcscf
from pyscf.pbc.cc import kintermediates_rhf as imdk

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


def emit(name, arr):
    a = np.asarray(arr).ravel()
    if np.iscomplexobj(a):
        flat = np.empty(2 * a.size)
        flat[0::2] = a.real
        flat[1::2] = a.imag
    else:
        flat = a.astype(float)
    print(f"BEGIN {name} n={flat.size}")
    for i in range(0, flat.size, 8):
        print(" ".join("%.17g" % v for v in flat[i : i + 8]))
    print(f"END {name}")


def scalar(name, v):
    """`key=value`, the format `oracle_phase15.rs`'s parser already reads."""
    print(("%s=%.17g" % (name, float(v))))


class SplitMix64:
    """The PRNG the Rust side uses, so both see the same synthetic amplitudes."""

    def __init__(self, s):
        self.s = s & 0xFFFFFFFFFFFFFFFF

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    def unit(self):
        return (self.next() >> 11) / float(1 << 53) - 0.5


def synthetic_amps(shape1, shape2, seed=20260906):
    r = SplitMix64(seed)
    n1 = int(np.prod(shape1))
    n2 = int(np.prod(shape2))
    t1 = np.array([complex(0.05 * r.unit(), 0.05 * r.unit()) for _ in range(n1)])
    t2 = np.array([complex(0.05 * r.unit(), 0.05 * r.unit()) for _ in range(n2)])
    return t1.reshape(shape1), t2.reshape(shape2)


def build(nk):
    cell = diamond()
    kpts = cell.make_kpts(nk)
    mf = pbcscf.KRHF(cell, kpts, exxdiv=None)
    mf.conv_tol = 1e-10
    mf.kernel()
    cc = pbcc.KRCCSD(mf)
    cc.conv_tol = 1e-9
    cc.conv_tol_normt = 1e-7
    return cell, kpts, mf, cc


def section_eris(nk=(1, 1, 2)):
    cell, kpts, mf, cc = build(list(nk))
    eris = cc.ao2mo(cc.mo_coeff)
    scalar("e_hf", mf.e_tot)
    scalar("nkpts", len(kpts))
    scalar("nocc", cc.nocc)
    scalar("nmo", cc.nmo)
    emit("fock", eris.fock)
    emit("mo_energy", np.asarray(eris.mo_energy))
    # The PADDED MO coefficients the transform used, so the Rust side can build
    # its `_ERIS` from UPSTREAM's own mean field and the comparison stops being
    # a comparison of two SCFs.
    emit("mo_coeff", np.asarray(eris.mo_coeff))
    scalar("nao", eris.mo_coeff[0].shape[0])
    from pyscf.pbc.mp.kmp2 import get_nocc as _get_nocc
    nocc_per_kpt = _get_nocc(cc, per_kpoint=True)
    emit("nocc_per_kpt", np.asarray(nocc_per_kpt, dtype=float))
    for name in ("oooo", "ooov", "oovv", "ovov", "voov", "vovv", "vvvv"):
        emit(name, np.asarray(getattr(eris, name)))


def section_imds(nk=(1, 1, 2)):
    cell, kpts, mf, cc = build(list(nk))
    eris = cc.ao2mo(cc.mo_coeff)
    scalar("e_hf", mf.e_tot)
    emit("fock", eris.fock)
    emit("mo_energy", np.asarray(eris.mo_energy))
    emit("mo_coeff", np.asarray(eris.mo_coeff))
    scalar("nao", eris.mo_coeff[0].shape[0])
    from pyscf.pbc.mp.kmp2 import get_nocc as _get_nocc
    emit("nocc_per_kpt", np.asarray(_get_nocc(cc, per_kpoint=True), dtype=float))
    scalar("nkpts", len(kpts))
    scalar("nocc", cc.nocc)
    scalar("nmo", cc.nmo)
    nkpts, nocc = len(kpts), cc.nocc
    nvir = cc.nmo - nocc
    t1, t2 = synthetic_amps(
        (nkpts, nocc, nvir), (nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir)
    )
    kconserv = cc.khelper.kconserv
    emit("t1", t1)
    emit("t2", t2)
    emit("cc_Foo", imdk.cc_Foo(t1, t2, eris, kconserv))
    emit("cc_Fvv", imdk.cc_Fvv(t1, t2, eris, kconserv))
    emit("cc_Fov", imdk.cc_Fov(t1, t2, eris, kconserv))
    emit("Loo", imdk.Loo(t1, t2, eris, kconserv))
    emit("Lvv", imdk.Lvv(t1, t2, eris, kconserv))
    emit("cc_Woooo", imdk.cc_Woooo(t1, t2, eris, kconserv))
    emit("cc_Wvvvv", imdk.cc_Wvvvv(t1, t2, eris, kconserv))
    emit("cc_Wvoov", imdk.cc_Wvoov(t1, t2, eris, kconserv))
    emit("cc_Wvovo", imdk.cc_Wvovo(t1, t2, eris, kconserv))
    t1new, t2new = pbcc.kccsd_rhf.update_amps(cc, t1, t2, eris)
    emit("t1new", t1new)
    emit("t2new", t2new)
    scalar("energy_synth", pbcc.kccsd_rhf.energy(cc, t1, t2, eris))


def section_krccsd(nk=(1, 1, 2)):
    cell, kpts, mf, cc = build(list(nk))
    eris = cc.ao2mo(cc.mo_coeff)
    emit("fock", eris.fock)
    emit("mo_energy", np.asarray(eris.mo_energy))
    emit("mo_coeff", np.asarray(eris.mo_coeff))
    scalar("nao", eris.mo_coeff[0].shape[0])
    from pyscf.pbc.mp.kmp2 import get_nocc as _get_nocc
    emit("nocc_per_kpt", np.asarray(_get_nocc(cc, per_kpoint=True), dtype=float))
    scalar("nkpts", len(kpts))
    scalar("nocc", cc.nocc)
    scalar("nmo", cc.nmo)
    emp2, t1, t2 = cc.init_amps(eris)
    scalar("e_hf", mf.e_tot)
    scalar("emp2", emp2)
    e_corr, t1, t2 = cc.kernel(eris=eris)
    scalar("e_corr", e_corr)
    scalar("fp_t1_re", np.asarray(t1).real.sum())
    emit("t1", t1)
    scalar("e_tot", mf.e_tot + e_corr)


def section_triples(nk=(1, 1, 2)):
    from pyscf.pbc.cc import kccsd_t_rhf, kccsd_t_rhf_slow

    cell, kpts, mf, cc = build(list(nk))
    eris = cc.ao2mo(cc.mo_coeff)
    scalar("e_hf", mf.e_tot)
    emit("fock", eris.fock)
    emit("mo_energy", np.asarray(eris.mo_energy))
    emit("mo_coeff", np.asarray(eris.mo_coeff))
    scalar("nao", eris.mo_coeff[0].shape[0])
    from pyscf.pbc.mp.kmp2 import get_nocc as _get_nocc
    emit("nocc_per_kpt", np.asarray(_get_nocc(cc, per_kpoint=True), dtype=float))
    scalar("nkpts", len(kpts))
    scalar("nocc", cc.nocc)
    scalar("nmo", cc.nmo)
    e_corr, t1, t2 = cc.kernel(eris=eris)
    scalar("e_corr", e_corr)
    scalar("et_fast", kccsd_t_rhf.kernel(cc, eris, t1, t2))
    scalar("et_slow", kccsd_t_rhf_slow.kernel(cc, eris, t1, t2))


def section_kgccsd(nk=(1, 1, 2)):
    """KGCCSD on a KGHF mean field, emitting the spin-orbital `_PhysicistsERIs`.

    The Rust side rebuilds the seven blocks from these MO coefficients, so the
    comparison is of the CC code and not of two SCFs — see `README §10`.
    """
    from pyscf.pbc import scf as _pbcscf
    from pyscf.pbc.cc import kccsd as _kccsd

    cell = diamond()
    kpts = cell.make_kpts(list(nk))
    kmf = _pbcscf.KGHF(cell, kpts, exxdiv=None)
    kmf.conv_tol = 1e-10
    kmf.kernel()
    cc = _kccsd.KGCCSD(kmf)
    cc.conv_tol = 1e-9
    cc.conv_tol_normt = 1e-7
    eris = cc.ao2mo(cc.mo_coeff)

    scalar("e_hf", kmf.e_tot)
    scalar("nkpts", len(kpts))
    scalar("nocc", cc.nocc)
    scalar("nmo", cc.nmo)
    scalar("nao", np.asarray(eris.mo_coeff)[0].shape[0])
    emit("fock", np.asarray(eris.fock))
    emit("mo_energy", np.asarray(eris.mo_energy))
    emit("mo_coeff", np.asarray(eris.mo_coeff))
    emit("orbspin", np.asarray(eris.orbspin, dtype=float))
    from pyscf.pbc.mp.kmp2 import get_nocc as _get_nocc
    emit("nocc_per_kpt", np.asarray(_get_nocc(cc, per_kpoint=True), dtype=float))
    for name in ("oooo", "ooov", "ovoo", "oovv", "ovov", "ovvv", "vvvv"):
        emit(name, np.asarray(getattr(eris, name)))

    emp2, t1, t2 = cc.init_amps(eris)
    scalar("emp2", emp2)
    # A FIXED synthetic amplitude pair, so update_amps is isolated from the
    # iteration exactly as the RHF `imds` section does.
    nkpts, nocc = len(kpts), cc.nocc
    nvir = cc.nmo - nocc
    st1, st2 = synthetic_amps(
        (nkpts, nocc, nvir), (nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir)
    )
    emit("st1", st1)
    emit("st2", st2)
    scalar("energy_synth", _kccsd.energy(cc, st1, st2, eris))
    t1n, t2n = _kccsd.update_amps(cc, st1, st2, eris)
    emit("st1new", t1n)
    emit("st2new", t2n)

    e_corr, t1, t2 = cc.kernel(eris=eris)
    scalar("e_corr", e_corr)
    emit("t1", t1)
    emit("t2", t2)
    from pyscf.pbc.cc import kccsd_t as _kccsd_t
    scalar("et_spinorb", _kccsd_t.kernel(cc, eris, t1, t2))


def section_kcis(nk=(1, 1, 2)):
    """KCIS roots, plus the `_ERIS` inputs so the Rust side rebuilds them."""
    from pyscf.pbc.ci import kcis_rhf as _kcis

    cell, kpts, mf, cc = build(list(nk))
    cis = _kcis.KCIS(mf)
    cis.conv_tol = 1e-9
    eris = cis.ao2mo()

    scalar("e_hf", mf.e_tot)
    scalar("nkpts", len(kpts))
    scalar("nocc", cis.nocc)
    scalar("nmo", cis.nmo)
    scalar("nao", np.asarray(eris.mo_coeff)[0].shape[0])
    emit("fock", np.asarray(eris.fock))
    emit("mo_energy", np.asarray(eris.mo_energy))
    emit("mo_coeff", np.asarray(eris.mo_coeff))
    from pyscf.pbc.mp.kmp2 import get_nocc as _get_nocc
    emit("nocc_per_kpt", np.asarray(_get_nocc(cis, per_kpoint=True), dtype=float))

    emit("epsilons", np.asarray([eris.fock[k].diagonal().real for k in range(len(kpts))]))
    nroots = 3
    for kshift in range(len(kpts)):
        # `cis_diag` returns a COMPLEX array (dtype = eris.dtype), so this
        # block is INTERLEAVED; the Rust side must read it with `cblock`.
        emit("diag_%d" % kshift, _kcis.cis_diag(cis, kshift, eris))
        evals, _ = cis.kernel(nroots=nroots, eris=eris, kptlist=[kshift])
        emit("roots_%d" % kshift, np.asarray(evals[0]).real)
        # The DENSE fallback on the same fixture, so the Rust side can gate its
        # own Davidson-vs-dense agreement against upstream's.
        cis.davidson = False
        evals_d, _ = cis.kernel(nroots=nroots, eris=eris, kptlist=[kshift])
        emit("dense_%d" % kshift, np.asarray(evals_d[0]).real)
        cis.davidson = True


SECTIONS = {
    "kcis": lambda: section_kcis((1, 1, 2)),
    "kgccsd": lambda: section_kgccsd((1, 1, 2)),
    "eris": lambda: section_eris((1, 1, 2)),
    "eris222": lambda: section_eris((2, 2, 2)),
    "imds": lambda: section_imds((1, 1, 2)),
    "imds222": lambda: section_imds((2, 2, 2)),
    "krccsd": lambda: section_krccsd((1, 1, 2)),
    "krccsd222": lambda: section_krccsd((2, 2, 2)),
    "gamma": lambda: section_krccsd((1, 1, 1)),
    "triples": lambda: section_triples((1, 1, 2)),
}

if __name__ == "__main__":
    print(f"pyscf.__version__={pyscf.__version__}")
    for name in sys.argv[1:] or ["eris"]:
        SECTIONS[name]()
