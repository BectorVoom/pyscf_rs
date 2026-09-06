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

    # 16-09 Task 1 — the ten EOM intermediates `_IMDS` builds
    # (`eom_kccsd_ghf.py:1863-1966`), on the SAME synthetic amplitudes, so each
    # is gated on its own rather than only through an EOM root.
    from pyscf.pbc.cc import kintermediates as _gimd
    kcon = cc.khelper.kconserv
    emit("imds_Foo", _gimd.Foo(cc, st1, st2, eris, kcon))
    emit("imds_Fvv", _gimd.Fvv(cc, st1, st2, eris, kcon))
    emit("imds_Fov", _gimd.Fov(cc, st1, st2, eris, kcon))
    emit("imds_Woooo", _gimd.Woooo(cc, st1, st2, eris, kcon))
    emit("imds_Wovvo", _gimd.Wovvo(cc, st1, st2, eris, kcon))
    emit("imds_Wooov", _gimd.Wooov(cc, st1, st2, eris, kcon))
    emit("imds_Wvovv", _gimd.Wvovv(cc, st1, st2, eris, kcon))
    emit("imds_Wvvvv", _gimd.Wvvvv(cc, st1, st2, eris, kcon))
    emit("imds_Wovoo", _gimd.Wovoo(cc, st1, st2, eris, kcon))
    emit("imds_Wvvvo", _gimd.Wvvvo(cc, st1, st2, eris, kcon))

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


def section_eris_gdf(nk=(1, 1, 2)):
    """The seven `_ERIS` blocks on a **GDF** mean field — gate G2.

    `kccsd_rhf.py:37` imports `GDF, RSGDF` and branches the whole `_ERIS` build
    on the mean field's DF class, and 16-01 measured the plane-wave/Gaussian
    route split at `9.22e-4 Ha` on this cell. A gate that does not name its
    route is untestable, so the Gaussian route gets its own.
    """
    from pyscf.pbc import df as _pbcdf

    cell = diamond()
    kpts = cell.make_kpts(list(nk))
    mf = pbcscf.KRHF(cell, kpts, exxdiv=None)
    mf.with_df = _pbcdf.GDF(cell, kpts)
    mf.conv_tol = 1e-10
    mf.kernel()
    cc = pbcc.KRCCSD(mf)
    cc.conv_tol = 1e-9
    eris = cc.ao2mo(cc.mo_coeff)

    scalar("e_hf", mf.e_tot)
    scalar("nkpts", len(kpts))
    scalar("nocc", cc.nocc)
    scalar("nmo", cc.nmo)
    scalar("nao", np.asarray(eris.mo_coeff)[0].shape[0])
    emit("fock", eris.fock)
    emit("mo_energy", np.asarray(eris.mo_energy))
    emit("mo_coeff", np.asarray(eris.mo_coeff))
    from pyscf.pbc.mp.kmp2 import get_nocc as _get_nocc
    emit("nocc_per_kpt", np.asarray(_get_nocc(cc, per_kpoint=True), dtype=float))
    for name in ("oooo", "ooov", "oovv", "ovov", "voov", "vovv", "vvvv"):
        emit(name, np.asarray(getattr(eris, name)))
    emp2, t1, t2 = cc.init_amps(eris)
    scalar("emp2", emp2)
    e_corr, t1, t2 = cc.kernel(eris=eris)
    scalar("e_corr", e_corr)


def h3_openshell(nk, mesh=(13, 13, 13)):
    """`pbc/cc/test/test_kuccsd_openshell.py:10-24`, with the k-mesh's own spin.

    Three hydrogens in a large cubic-ish cell on a two-`s` basis: six AOs, and
    the ONLY fixture in this phase that is genuinely open shell. `cell.spin` is
    multiplied by `nkpts` exactly as upstream's test does at `:29`, so the
    supercell the k-mesh represents carries `nkpts` unpaired electrons.
    """
    cell = pbcgto.Cell()
    cell.unit = "B"
    cell.a = [
        [0.0, 6.74027466, 6.74027466],
        [6.74027466, 0.0, 6.74027466],
        [6.74027466, 6.74027466, 0.0],
    ]
    cell.mesh = list(mesh)
    cell.atom = """H 0 0 0
                   H 1.68506866 1.68506866 1.68506866
                   H 3.37013733 3.37013733 3.37013733"""
    cell.basis = [[0, (1.0, 1.0)], [0, (0.5, 1.0)]]
    cell.verbose = 0
    cell.charge = 0
    cell.spin = 1
    cell.build()
    cell.spin = cell.spin * int(np.prod(nk))
    return cell


UBLOCKS = (
    "oooo", "ooov", "oovv", "ovov", "voov", "vovv",
    "OOOO", "OOOV", "OOVV", "OVOV", "VOOV", "VOVV",
    "ooOO", "ooOV", "ooVV", "ovOV", "voOV", "voVV",
    "OOov", "OOvv", "OVov", "VOov", "VOvv",
    "vvvv", "VVVV", "vvVV",
)


def section_kuccsd(nk=(1, 1, 2), mesh=(31, 31, 31)):
    """KUCCSD on an OPEN-SHELL KUHF mean field.

    Everything the Rust side needs to rebuild `_ChemistsERIs` from upstream's
    own MO coefficients is emitted, so the gates measure the CC code and not
    two unrestricted SCFs — which, unlike the restricted ones, need not even
    find the same solution.

    The mesh DEFAULTS TO `[31,31,31]`, not upstream's pinned `[13,13,13]`, and
    that is a deliberate deviation recorded in `16-06-SUMMARY.md`. At `[13,13,13]`
    the port and upstream agree only to `1.2e-5` on `vvvv` while agreeing to
    `2.7e-9` on `oooo` — monotone in the count of VIRTUAL indices, because this
    fixture's virtuals are the antibonding combinations of an all-electron
    `0.5`-exponent `s` function in a 6.74-Bohr cell and a 13-point FFT does not
    resolve them. At `[31,31,31]` the whole table is flat at `~5e-10` and `vvvv`
    is TIGHTER than `oooo`. Gating on the coarse mesh would mean a `1e-4` gate
    hiding four orders of real agreement, so the primary fixture is the refined
    one and `kuccsd_coarse` keeps the coarse numbers for the measurement that
    proves the residual is the mesh.
    """
    from pyscf.pbc.cc import kccsd_uhf as _kuhf

    cell = h3_openshell(nk, mesh)
    kpts = cell.make_kpts(list(nk))
    kpts -= kpts[0]
    kmf = pbcscf.KUHF(cell, kpts, exxdiv=None)
    kmf.conv_tol = 1e-11
    kmf.kernel()
    cc = _kuhf.KUCCSD(kmf)
    cc.conv_tol = 1e-9
    cc.conv_tol_normt = 1e-7
    eris = cc.ao2mo(cc.mo_coeff)

    nkpts = len(kpts)
    nocca, noccb = cc.nocc
    nmoa, nmob = cc.nmo
    scalar("e_hf", kmf.e_tot)
    scalar("converged_hf", 1.0 if kmf.converged else 0.0)
    scalar("nkpts", nkpts)
    scalar("nocca", nocca)
    scalar("noccb", noccb)
    scalar("nmoa", nmoa)
    scalar("nmob", nmob)
    scalar("nao", np.asarray(eris.mo_coeff[0])[0].shape[0])
    from pyscf.pbc import tools as _pbctools
    scalar("madelung", _pbctools.madelung(cell, kpts))
    emit("kpts", np.asarray(kpts))
    emit("mesh", np.asarray(cell.mesh, dtype=float))
    emit("focka", np.asarray(eris.fock[0]))
    emit("fockb", np.asarray(eris.fock[1]))
    emit("mo_energy_a", np.asarray(eris.mo_energy[0]))
    emit("mo_energy_b", np.asarray(eris.mo_energy[1]))
    emit("mo_coeff_a", np.asarray(eris.mo_coeff[0]))
    emit("mo_coeff_b", np.asarray(eris.mo_coeff[1]))
    # `kump2.get_nocc(cc, per_kpoint=True)` is the upstream route, but it goes
    # through `kmp2.get_nocc`, which chokes on `KUCCSD.mo_occ` being a 2-list.
    # Counting the occupations directly is the same number and cannot break.
    emit(
        "nocc_per_kpt_a",
        np.asarray([int(np.count_nonzero(o > 0)) for o in kmf.mo_occ[0]], dtype=float),
    )
    emit(
        "nocc_per_kpt_b",
        np.asarray([int(np.count_nonzero(o > 0)) for o in kmf.mo_occ[1]], dtype=float),
    )
    for name in UBLOCKS:
        emit(name, np.asarray(getattr(eris, name)))

    emp2, t1, t2 = cc.init_amps(eris)
    scalar("emp2", emp2)

    # A FIXED synthetic amplitude quintuple, drawn from ONE SplitMix64 stream
    # in `amplitudes_to_vector`'s order, so `update_amps` is isolated from the
    # iteration exactly as the RHF `imds` section isolates the intermediates.
    nvira, nvirb = nmoa - nocca, nmob - noccb
    r = SplitMix64(20260906)

    def draw(shape):
        n = int(np.prod(shape))
        return np.array(
            [complex(0.05 * r.unit(), 0.05 * r.unit()) for _ in range(n)]
        ).reshape(shape)

    st1a = draw((nkpts, nocca, nvira))
    st1b = draw((nkpts, noccb, nvirb))
    st2aa = draw((nkpts, nkpts, nkpts, nocca, nocca, nvira, nvira))
    st2ab = draw((nkpts, nkpts, nkpts, nocca, noccb, nvira, nvirb))
    st2bb = draw((nkpts, nkpts, nkpts, noccb, noccb, nvirb, nvirb))
    st1, st2 = (st1a, st1b), (st2aa, st2ab, st2bb)
    z1 = (np.zeros_like(st1a), np.zeros_like(st1b))
    z2 = (np.zeros_like(st2aa), np.zeros_like(st2ab), np.zeros_like(st2bb))
    scalar("energy_synth", _kuhf.energy(cc, st1, st2, eris))
    # Three runs, so a `t2new` mismatch bisects into "a term linear in t1",
    # "a term in t2" or "a cross term" without any argument about which.
    for tag, a1, a2 in (("", st1, st2), ("_t1z", z1, st2), ("_t2z", st1, z2)):
        t1n, t2n = _kuhf.update_amps(cc, a1, a2, eris)
        for nm, arr in zip(
            ("st1anew", "st1bnew", "st2aanew", "st2abnew", "st2bbnew"),
            (t1n[0], t1n[1], t2n[0], t2n[1], t2n[2]),
        ):
            emit(nm + tag, arr)

    e_corr, t1, t2 = cc.kernel(eris=eris)
    scalar("e_corr", e_corr)
    scalar("converged_cc", 1.0 if cc.converged else 0.0)
    emit("t1a", t1[0])
    emit("t1b", t1[1])

    # 16-12 — the one-particle density matrix, on BOTH the converged amplitudes
    # and the synthetic ones. The synthetic pair is what actually exercises the
    # equations: at convergence `t1` is small, so a wrong `t1`-linear term
    # barely moves `dm1`.
    from pyscf.pbc.cc import kuccsd_rdm as _rdm
    dm1a, dm1b = _rdm.make_rdm1(cc, t1, t2)
    emit("rdm1a", dm1a)
    emit("rdm1b", dm1b)
    sdm1a, sdm1b = _rdm.make_rdm1(cc, st1, st2)
    emit("srdm1a", sdm1a)
    emit("srdm1b", sdm1b)


def section_kuccsd_imds(nk=(1, 1, 2), mesh=(31, 31, 31)):
    """Every `kintermediates_uhf` ground-state intermediate on the SAME fixed
    synthetic amplitudes `section_kuccsd` uses, so a `t2new` mismatch can be
    bisected to one intermediate instead of being argued about."""
    from pyscf.pbc.cc import kccsd_uhf as _kuhf
    from pyscf.pbc.cc import kintermediates_uhf as _imd

    cell = h3_openshell(nk, mesh)
    kpts = cell.make_kpts(list(nk))
    kpts -= kpts[0]
    kmf = pbcscf.KUHF(cell, kpts, exxdiv=None)
    kmf.conv_tol = 1e-11
    kmf.kernel()
    cc = _kuhf.KUCCSD(kmf)
    eris = cc.ao2mo(cc.mo_coeff)

    nkpts = len(kpts)
    nocca, noccb = cc.nocc
    nmoa, nmob = cc.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb
    scalar("nkpts", nkpts)
    scalar("nocca", nocca)
    scalar("noccb", noccb)
    scalar("nmoa", nmoa)
    scalar("nmob", nmob)
    scalar("nao", np.asarray(eris.mo_coeff[0])[0].shape[0])
    from pyscf.pbc import tools as _pbctools
    scalar("madelung", _pbctools.madelung(cell, kpts))
    emit("kpts", np.asarray(kpts))
    emit("mesh", np.asarray(cell.mesh, dtype=float))
    emit("focka", np.asarray(eris.fock[0]))
    emit("fockb", np.asarray(eris.fock[1]))
    emit("mo_energy_a", np.asarray(eris.mo_energy[0]))
    emit("mo_energy_b", np.asarray(eris.mo_energy[1]))
    emit("mo_coeff_a", np.asarray(eris.mo_coeff[0]))
    emit("mo_coeff_b", np.asarray(eris.mo_coeff[1]))
    emit(
        "nocc_per_kpt_a",
        np.asarray([int(np.count_nonzero(o > 0)) for o in kmf.mo_occ[0]], dtype=float),
    )
    emit(
        "nocc_per_kpt_b",
        np.asarray([int(np.count_nonzero(o > 0)) for o in kmf.mo_occ[1]], dtype=float),
    )

    r = SplitMix64(20260906)

    def draw(shape):
        n = int(np.prod(shape))
        return np.array(
            [complex(0.05 * r.unit(), 0.05 * r.unit()) for _ in range(n)]
        ).reshape(shape)

    t1 = (draw((nkpts, nocca, nvira)), draw((nkpts, noccb, nvirb)))
    t2 = (
        draw((nkpts, nkpts, nkpts, nocca, nocca, nvira, nvira)),
        draw((nkpts, nkpts, nkpts, nocca, noccb, nvira, nvirb)),
        draw((nkpts, nkpts, nkpts, noccb, noccb, nvirb, nvirb)),
    )

    for nm, arr in zip(("tau_aa", "tau_ab", "tau_bb"), _imd.make_tau(cc, t2, t1, t1)):
        emit(nm, arr)
    for nm, arr in zip(
        ("tauh_aa", "tauh_ab", "tauh_bb"), _imd.make_tau(cc, t2, t1, t1, fac=0.5)
    ):
        emit(nm, arr)
    for nm, arr in zip(
        ("tau2_aa", "tau2_ab", "tau2_bb"), _imd.make_tau2(cc, t2, t1, t1, fac=2.0)
    ):
        emit(nm, arr)
    for nm, arr in zip(("Fvv_a", "Fvv_b"), _imd.cc_Fvv(cc, t1, t2, eris)):
        emit(nm, arr)
    for nm, arr in zip(("Foo_a", "Foo_b"), _imd.cc_Foo(cc, t1, t2, eris)):
        emit(nm, arr)
    for nm, arr in zip(("Fov_a", "Fov_b"), _imd.cc_Fov(cc, t1, t2, eris)):
        emit(nm, arr)
    for nm, arr in zip(
        ("Woooo", "WooOO", "WOOOO"), _imd.cc_Woooo(cc, t1, t2, eris)
    ):
        emit(nm, arr)
    for nm, arr in zip(
        ("Wvvvv", "WvvVV", "WVVVV"), _imd.cc_Wvvvv_half(cc, t1, t2, eris)
    ):
        emit(nm, arr)
    for nm, arr in zip(
        ("Wovvo", "WovVO", "WOVvo", "WOVVO", "WoVVo", "WOvvO"),
        _imd.cc_Wovvo(cc, t1, t2, eris),
    ):
        emit(nm, arr)

    # `add_vvvv_` in isolation: it MUTATES its `Ht2` argument, so feed it zeros.
    z = (np.zeros_like(t2[0]), np.zeros_like(t2[1]), np.zeros_like(t2[2]))
    _kuhf.add_vvvv_(cc, z, t1, t2, eris)
    for nm, arr in zip(("vvvv_aa", "vvvv_ab", "vvvv_bb"), z):
        emit(nm, arr)


def section_kuccsd_wovvo(nk=(1, 1, 2), mesh=(31, 31, 31)):
    """`kccsd_uhf.py:230-386` run STANDALONE on zeroed `Ht2`.

    The body below is a VERBATIM copy of those lines — not a re-transcription —
    so that a mismatch against the Rust port's `wovvo_terms` is attributable to
    the port and not to a second reading of upstream. The only edits are the
    `Ht2*` initialisers (zeros instead of the partially-assembled doubles) and
    the emits at the end.
    """
    from pyscf.pbc.cc import kccsd_uhf as _kuhf
    from pyscf.pbc.cc import kintermediates_uhf
    from pyscf.pbc.lib import kpts_helper
    from pyscf import lib
    einsum = lib.einsum

    cell = h3_openshell(nk, mesh)
    kpts = cell.make_kpts(list(nk))
    kpts -= kpts[0]
    kmf = pbcscf.KUHF(cell, kpts, exxdiv=None)
    kmf.conv_tol = 1e-11
    kmf.kernel()
    cc = _kuhf.KUCCSD(kmf)
    eris = cc.ao2mo(cc.mo_coeff)

    nkpts = len(kpts)
    nocca, noccb = cc.nocc
    nmoa, nmob = cc.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb
    scalar("nkpts", nkpts)
    scalar("nocca", nocca)
    scalar("noccb", noccb)
    scalar("nmoa", nmoa)
    scalar("nmob", nmob)
    scalar("nao", np.asarray(eris.mo_coeff[0])[0].shape[0])
    from pyscf.pbc import tools as _pbctools
    scalar("madelung", _pbctools.madelung(cell, kpts))
    emit("kpts", np.asarray(kpts))
    emit("mesh", np.asarray(cell.mesh, dtype=float))
    emit("focka", np.asarray(eris.fock[0]))
    emit("fockb", np.asarray(eris.fock[1]))
    emit("mo_energy_a", np.asarray(eris.mo_energy[0]))
    emit("mo_energy_b", np.asarray(eris.mo_energy[1]))
    emit("mo_coeff_a", np.asarray(eris.mo_coeff[0]))
    emit("mo_coeff_b", np.asarray(eris.mo_coeff[1]))
    emit(
        "nocc_per_kpt_a",
        np.asarray([int(np.count_nonzero(o > 0)) for o in kmf.mo_occ[0]], dtype=float),
    )
    emit(
        "nocc_per_kpt_b",
        np.asarray([int(np.count_nonzero(o > 0)) for o in kmf.mo_occ[1]], dtype=float),
    )

    r = SplitMix64(20260906)

    def draw(shape):
        n = int(np.prod(shape))
        return np.array(
            [complex(0.05 * r.unit(), 0.05 * r.unit()) for _ in range(n)]
        ).reshape(shape)

    t1a = draw((nkpts, nocca, nvira))
    t1b = draw((nkpts, noccb, nvirb))
    t2aa = draw((nkpts, nkpts, nkpts, nocca, nocca, nvira, nvira))
    t2ab = draw((nkpts, nkpts, nkpts, nocca, noccb, nvira, nvirb))
    t2bb = draw((nkpts, nkpts, nkpts, noccb, noccb, nvirb, nvirb))
    t1 = (t1a, t1b)
    t2 = (t2aa, t2ab, t2bb)
    kconserv = cc.khelper.kconserv
    P = kintermediates_uhf.kconserv_mat(cc.nkpts, cc.khelper.kconserv)

    Ht2aa = np.zeros_like(t2aa)
    Ht2ab = np.zeros_like(t2ab)
    Ht2bb = np.zeros_like(t2bb)

    Wovvo, WovVO, WOVvo, WOVVO, WoVVo, WOvvO = \
            kintermediates_uhf.cc_Wovvo(cc, t1, t2, eris)

    #:Ht2ab += einsum('xwzimae,wvumeBJ,xwzv,wuvy->xyziJaB', t2aa, WovVO, P, P)
    #:Ht2ab += einsum('xwziMaE,wvuMEBJ,xwzv,wuvy->xyziJaB', t2ab, WOVVO, P, P)
    #:Ht2ab -= einsum('xie,zma,uwzBJme,zuwx,xyzu->xyziJaB', t1a, t1a, eris.VOov, P, P)
    for kx, kw, kz in kpts_helper.loop_kkk(nkpts):
        kv = kconserv[kx, kz, kw]
        for ku in range(nkpts):
            ky = kconserv[kw, kv, ku]
            Ht2ab[kx, ky, kz] += lib.einsum('imae,mebj->ijab', t2aa[kx,kw,kz], WovVO[kw,kv,ku])
            Ht2ab[kx, ky, kz] += lib.einsum('imae,mebj->ijab', t2ab[kx,kw,kz], WOVVO[kw,kv,ku])

    #for kz, ku, kw in kpts_helper.loop_kkk(nkpts):
    #    kx = kconserv[kz,kw,ku]
    #    ky = kconserv[kz,kx,ku]
    #    continue
    #    Ht2ab[kx, ky, kz] -= lib.einsum('ie, ma, emjb->ijab', t1a[kx], t1a[kz], eris.voOV[kx,kz,kw].conj())
    Ht2ab -= einsum('xie, yma, xyzemjb->xzyijab', t1a, t1a, eris.voOV[:].conj())
    #:Ht2ab += einsum('wxvmIeA,wvumebj,xwzv,wuvy->yxujIbA', t2ab, Wovvo, P, P)
    #:Ht2ab += einsum('wxvMIEA,wvuMEbj,xwzv,wuvy->yxujIbA', t2bb, WOVvo, P, P)
    #:Ht2ab -= einsum('xIE,zMA,uwzbjME,zuwx,xyzu->yxujIbA', t1b, t1b, eris.voOV, P, P)

    #for kx, kw, kz in kpts_helper.loop_kkk(nkpts):
    #    kv = kconserv[kx, kz, kw]
    #    for ku in range(nkpts):
    #        ky = kconserv[kw, kv, ku]
    #        #Ht2ab[ky,kx,ku] += lib.einsum('miea, mebj-> jiba', t2ab[kw,kx,kv], Wovvo[kw,kv,ku])
    #        #Ht2ab[ky,kx,ku] += lib.einsum('miea, mebj-> jiba', t2bb[kw,kx,kv], WOVvo[kw,kv,ku])

    for km, ke, kb in kpts_helper.loop_kkk(nkpts):
        kj = kconserv[km, ke, kb]
        Ht2ab[kj,:,kb] += einsum('xmiea, mebj->xjiba', t2ab[km,:,ke], Wovvo[km,ke,kb])
        Ht2ab[kj,:,kb] += einsum('xmiea, mebj->xjiba', t2bb[km,:,ke], WOVvo[km,ke,kb])


    for kz, ku, kw in kpts_helper.loop_kkk(nkpts):
        kx = kconserv[kz, kw, ku]
        ky = kconserv[kz, kx, ku]
        Ht2ab[ky,kx,ku] -= lib.einsum('ie, ma, bjme->jiba', t1b[kx], t1b[kz], eris.voOV[ku,kw,kz])


    #:Ht2ab += einsum('xwviMeA,wvuMebJ,xwzv,wuvy->xyuiJbA', t2ab, WOvvO, P, P)
    #:Ht2ab -= einsum('xie,zMA,zwuMJbe,zuwx,xyzu->xyuiJbA', t1a, t1b, eris.OOvv, P, P)
    #for kx, kw, kz in kpts_helper.loop_kkk(nkpts):
    #    kv = kconserv[kx, kz, kw]
    #    for ku in range(nkpts):
    #        ky = kconserv[kw, kv, ku]
    #        Ht2ab[kx,ky,ku] += lib.einsum('imea,mebj->ijba', t2ab[kx,kw,kv],WOvvO[kw,kv,ku])
    for km, ke, kb in kpts_helper.loop_kkk(nkpts):
        kj = kconserv[km, ke, kb]
        Ht2ab[:,kj,kb] += einsum('ximea, mebj->xijba', t2ab[:,km,ke], WOvvO[km,ke,kb])


    for kz,ku,kw in kpts_helper.loop_kkk(nkpts):
        kx = kconserv[kz, kw, ku]
        ky = kconserv[kz, kx, ku]
        Ht2ab[kx,ky,ku] -= lib.einsum('ie, ma, mjbe->ijba', t1a[kx], t1b[kz], eris.OOvv[kz, kw, ku])

    #:Ht2ab += einsum('wxzmIaE,wvumEBj,xwzv,wuvy->yxzjIaB', t2ab, WoVVo, P, P)
    #:Ht2ab -= einsum('xIE,zma,zwumjBE,zuwx,xyzu->yxzjIaB', t1b, t1a, eris.ooVV, P, P)
    for kx, kw, kz in kpts_helper.loop_kkk(nkpts):
        kv = kconserv[kx, kz, kw]
        for ku in range(nkpts):
            ky = kconserv[kw, kv, ku]
            Ht2ab[ky, kx, kz] += lib.einsum('miae,mebj->jiab', t2ab[kw,kx,kz], WoVVo[kw,kv,ku])

    for kz, ku, kw in kpts_helper.loop_kkk(nkpts):
        kx = kconserv[kz,kw,ku]
        ky = kconserv[kz,kx,ku]
        Ht2ab[ky,kx,kz] -= lib.einsum('ie, ma, mjbe->jiab', t1b[kx], t1a[kz], eris.ooVV[kz,kw,ku])

    #:u2aa  = einsum('xwzimae,wvumebj,xwzv,wuvy->xyzijab', t2aa, Wovvo, P, P)
    #:u2aa += einsum('xwziMaE,wvuMEbj,xwzv,wuvy->xyzijab', t2ab, WOVvo, P, P)
    #Left this in to keep proper shape, need to replace later
    u2aa  = np.zeros_like(t2aa)
    for kx, kw, kz in kpts_helper.loop_kkk(nkpts):
        kv = kconserv[kx, kz, kw]
        for ku in range(nkpts):
            ky = kconserv[kw, kv, ku]
            u2aa[kx,ky,kz] += lib.einsum('imae, mebj->ijab', t2aa[kx,kw,kz], Wovvo[kw,kv,ku])
            u2aa[kx,ky,kz] += lib.einsum('imae, mebj->ijab', t2ab[kx,kw,kz], WOVvo[kw,kv,ku])


    #:u2aa += einsum('xie,zma,zwumjbe,zuwx,xyzu->xyzijab', t1a, t1a, eris.oovv, P, P)
    #:u2aa -= einsum('xie,zma,uwzbjme,zuwx,xyzu->xyzijab', t1a, t1a, eris.voov, P, P)

    for kz, ku, kw in kpts_helper.loop_kkk(nkpts):
        kx = kconserv[kz,kw,ku]
        ky = kconserv[kz,kx,ku]
        u2aa[kx,ky,kz] += lib.einsum('ie,ma,mjbe->ijab',t1a[kx],t1a[kz],eris.oovv[kz,kw,ku])
        u2aa[kx,ky,kz] -= lib.einsum('ie,ma,bjme->ijab',t1a[kx],t1a[kz],eris.voov[ku,kw,kz])


    #:u2aa += np.einsum('xie,uyzbjae,uzyx->xyzijab', t1a, eris.vovv, P)
    #:u2aa -= np.einsum('zma,xzyimjb->xyzijab', t1a, eris.ooov.conj())

    for ky, kx, ku in kpts_helper.loop_kkk(nkpts):
        kz = kconserv[ky, ku, kx]
        u2aa[kx, ky, kz] += lib.einsum('ie, bjae->ijab', t1a[kx], eris.vovv[ku,ky,kz])
        u2aa[kx, ky, kz] -= lib.einsum('ma, imjb->ijab', t1a[kz], eris.ooov[kx,kz,ky].conj())

    u2aa = u2aa - u2aa.transpose(1,0,2,4,3,5,6)
    u2aa = u2aa - einsum('xyzijab,xyzu->xyuijba', u2aa, P)
    Ht2aa += u2aa

    #:u2bb  = einsum('xwzimae,wvumebj,xwzv,wuvy->xyzijab', t2bb, WOVVO, P, P)
    #:u2bb += einsum('wxvMiEa,wvuMEbj,xwzv,wuvy->xyzijab', t2ab, WovVO, P, P)
    #:u2bb += einsum('xie,zma,zwumjbe,zuwx,xyzu->xyzijab', t1b, t1b, eris.OOVV, P, P)
    #:u2bb -= einsum('xie,zma,uwzbjme,zuwx,xyzu->xyzijab', t1b, t1b, eris.VOOV, P, P)

    u2bb = np.zeros_like(t2bb)

    for kx, kw, kz in kpts_helper.loop_kkk(nkpts):
        kv = kconserv[kx, kz, kw]
        for ku in range(nkpts):
            ky = kconserv[kw,kv, ku]
            u2bb[kx, ky, kz] += lib.einsum('imae,mebj->ijab', t2bb[kx,kw,kz], WOVVO[kw,kv,ku])
            u2bb[kx, ky, kz] += lib.einsum('miea, mebj-> ijab', t2ab[kw,kx,kv],WovVO[kw,kv,ku])

    for kz, ku, kw in kpts_helper.loop_kkk(nkpts):
        kx = kconserv[kz, kw, ku]
        ky = kconserv[kz, kx, ku]
        u2bb[kx, ky, kz] += lib.einsum('ie, ma, mjbe->ijab',t1b[kx],t1b[kz],eris.OOVV[kz,kw,ku])
        u2bb[kx, ky, kz] -= lib.einsum('ie, ma, bjme->ijab', t1b[kx], t1b[kz],eris.VOOV[ku,kw,kz])

    #:u2bb += np.einsum('xie,uzybjae,uzyx->xyzijab', t1b, eris.VOVV, P)
    #:u2bb -= np.einsum('zma,xzyimjb->xyzijab', t1b, eris.OOOV.conj())

    for ky, kx, ku in kpts_helper.loop_kkk(nkpts):
        kz = kconserv[ky, ku, kx]
        u2bb[kx,ky,kz] += lib.einsum('ie,bjae->ijab', t1b[kx], eris.VOVV[ku,ky,kz])

    #for kx, kz, ky in kpts_helper.loop_kkk(nkpts):
    #    u2bb[kx,ky,kz] -= lib.einsum('ma, imjb-> ijab', t1b[kz], eris.OOOV[kx,kz,ky].conj())
    u2bb -= einsum('zma, xzyimjb->xyzijab', t1b, eris.OOOV[:].conj())

    u2bb = u2bb - u2bb.transpose(1,0,2,4,3,5,6)
    u2bb = u2bb - einsum('xyzijab,xyzu->xyuijba', u2bb, P)
    Ht2bb += u2bb

    #:Ht2ab += np.einsum('xie,uyzBJae,uzyx->xyziJaB', t1a, eris.VOvv, P)
    #:Ht2ab += np.einsum('yJE,zxuaiBE,zuxy->xyziJaB', t1b, eris.voVV, P)
    #:Ht2ab -= np.einsum('zma,xzyimjb->xyzijab', t1a, eris.ooOV.conj())
    #:Ht2ab -= np.einsum('umb,yuxjmia,xyuz->xyzijab', t1b, eris.OOov.conj(), P)
    for ky, kx, ku in kpts_helper.loop_kkk(nkpts):
        kz = kconserv[ky,ku,kx]
        Ht2ab[kx,ky,kz] += lib.einsum('ie, bjae-> ijab', t1a[kx], eris.VOvv[ku,ky,kz])
        Ht2ab[kx,ky,kz] += lib.einsum('je, aibe-> ijab', t1b[ky], eris.voVV[kz,kx,ku])

    #for kx, kz, ky in kpts_helper.loop_kkk(nkpts):
    #    Ht2ab[kx,ky,kz] -= lib.einsum('ma, imjb->ijab', t1a[kz], eris.ooOV[kx,kz,ky].conj())
    Ht2ab -= einsum('zma, xzyimjb->xyzijab', t1a, eris.ooOV[:].conj())

    for kx, ky, ku in kpts_helper.loop_kkk(nkpts):
        kz = kconserv[kx, ku, ky]
        Ht2ab[kx,ky,kz] -= lib.einsum('mb,jmia->ijab',t1b[ku],eris.OOov[ky,ku,kx].conj())
    emit("wovvo_aa", Ht2aa)
    emit("wovvo_ab", Ht2ab)
    emit("wovvo_bb", Ht2bb)


def section_kuccsd_woooo(nk=(1, 1, 2), mesh=(31, 31, 31)):
    """`kccsd_uhf.py:205-226` run STANDALONE on zeroed `Ht2` — the bare `ovov`
    driver plus the `Woooo` stage, verbatim, for the same reason
    `section_kuccsd_wovvo` exists."""
    from pyscf.pbc.cc import kccsd_uhf as _kuhf
    from pyscf.pbc.cc import kintermediates_uhf
    from pyscf.pbc.lib import kpts_helper
    from pyscf import lib
    einsum = lib.einsum

    cell = h3_openshell(nk, mesh)
    kpts = cell.make_kpts(list(nk))
    kpts -= kpts[0]
    kmf = pbcscf.KUHF(cell, kpts, exxdiv=None)
    kmf.conv_tol = 1e-11
    kmf.kernel()
    cc = _kuhf.KUCCSD(kmf)
    eris = cc.ao2mo(cc.mo_coeff)

    nkpts = len(kpts)
    nocca, noccb = cc.nocc
    nmoa, nmob = cc.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb
    scalar("nkpts", nkpts)
    scalar("nocca", nocca)
    scalar("noccb", noccb)
    scalar("nmoa", nmoa)
    scalar("nmob", nmob)
    scalar("nao", np.asarray(eris.mo_coeff[0])[0].shape[0])
    from pyscf.pbc import tools as _pbctools
    scalar("madelung", _pbctools.madelung(cell, kpts))
    emit("kpts", np.asarray(kpts))
    emit("mesh", np.asarray(cell.mesh, dtype=float))
    emit("focka", np.asarray(eris.fock[0]))
    emit("fockb", np.asarray(eris.fock[1]))
    emit("mo_energy_a", np.asarray(eris.mo_energy[0]))
    emit("mo_energy_b", np.asarray(eris.mo_energy[1]))
    emit("mo_coeff_a", np.asarray(eris.mo_coeff[0]))
    emit("mo_coeff_b", np.asarray(eris.mo_coeff[1]))
    emit(
        "nocc_per_kpt_a",
        np.asarray([int(np.count_nonzero(o > 0)) for o in kmf.mo_occ[0]], dtype=float),
    )
    emit(
        "nocc_per_kpt_b",
        np.asarray([int(np.count_nonzero(o > 0)) for o in kmf.mo_occ[1]], dtype=float),
    )

    r = SplitMix64(20260906)

    def draw(shape):
        n = int(np.prod(shape))
        return np.array(
            [complex(0.05 * r.unit(), 0.05 * r.unit()) for _ in range(n)]
        ).reshape(shape)

    t1a = draw((nkpts, nocca, nvira))
    t1b = draw((nkpts, noccb, nvirb))
    t2aa = draw((nkpts, nkpts, nkpts, nocca, nocca, nvira, nvira))
    t2ab = draw((nkpts, nkpts, nkpts, nocca, noccb, nvira, nvirb))
    t2bb = draw((nkpts, nkpts, nkpts, noccb, noccb, nvirb, nvirb))
    t1 = (t1a, t1b)
    t2 = (t2aa, t2ab, t2bb)
    kconserv = cc.khelper.kconserv

    Ht2aa = np.zeros_like(t2aa)
    Ht2ab = np.zeros_like(t2ab)
    Ht2bb = np.zeros_like(t2bb)

    eris_ovov = np.asarray(eris.ovov)
    eris_OVOV = np.asarray(eris.OVOV)
    eris_ovOV = np.asarray(eris.ovOV)
    Ht2aa += (eris_ovov.transpose(0,2,1,3,5,4,6) - eris_ovov.transpose(2,0,1,5,3,4,6)).conj()
    Ht2bb += (eris_OVOV.transpose(0,2,1,3,5,4,6) - eris_OVOV.transpose(2,0,1,5,3,4,6)).conj()
    Ht2ab += eris_ovOV.transpose(0,2,1,3,5,4,6).conj()

    tauaa, tauab, taubb = kintermediates_uhf.make_tau(cc, t2, t1, t1)
    Woooo, WooOO, WOOOO = kintermediates_uhf.cc_Woooo(cc, t1, t2, eris)

    # Add the contributions from Wvvvv
    for km, ki, kn in kpts_helper.loop_kkk(nkpts):
        kj = kconserv[km,ki,kn]
        Woooo[km,ki,kn] += .5 * einsum('xmenf, xijef->minj', eris_ovov[km,:,kn], tauaa[ki,kj])
        WOOOO[km,ki,kn] += .5 * einsum('xMENF, xIJEF->MINJ', eris_OVOV[km,:,kn], taubb[ki,kj])
        WooOO[km,ki,kn] += .5 * einsum('xmeNF, xiJeF->miNJ', eris_ovOV[km,:,kn], tauab[ki,kj])

    for km, ki, kn in kpts_helper.loop_kkk(nkpts):
        kj = kconserv[km,ki,kn]
        Ht2aa[ki,kj,:] += einsum('minj,wmnab->wijab', Woooo[km,ki,kn], tauaa[km,kn]) * .5
        Ht2bb[ki,kj,:] += einsum('MINJ,wMNAB->wIJAB', WOOOO[km,ki,kn], taubb[km,kn]) * .5
        Ht2ab[ki,kj,:] += einsum('miNJ,wmNaB->wiJaB', WooOO[km,ki,kn], tauab[km,kn])
    emit("woooo_aa", Ht2aa)
    emit("woooo_ab", Ht2ab)
    emit("woooo_bb", Ht2bb)


def section_kuccsd_fock(nk=(1, 1, 2), mesh=(31, 31, 31)):
    """`kccsd_uhf.py:65-202` run STANDALONE — the intermediates, the singles
    equation and the `Fvv`/`Foo` doubles driving loop, verbatim, on zeroed
    accumulators. The last of the five stages `update_amps` is made of."""
    from pyscf.pbc.cc import kccsd_uhf as _kuhf
    from pyscf.pbc.cc import kintermediates_uhf
    from pyscf.pbc.lib import kpts_helper
    from pyscf.pbc.mp.kump2 import padding_k_idx
    from pyscf import lib
    einsum = lib.einsum

    cell = h3_openshell(nk, mesh)
    kpts = cell.make_kpts(list(nk))
    kpts -= kpts[0]
    kmf = pbcscf.KUHF(cell, kpts, exxdiv=None)
    kmf.conv_tol = 1e-11
    kmf.kernel()
    cc = _kuhf.KUCCSD(kmf)
    eris = cc.ao2mo(cc.mo_coeff)

    nkpts = len(kpts)
    nocca, noccb = cc.nocc
    nmoa, nmob = cc.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb
    scalar("nkpts", nkpts)
    scalar("nocca", nocca)
    scalar("noccb", noccb)
    scalar("nmoa", nmoa)
    scalar("nmob", nmob)
    scalar("nao", np.asarray(eris.mo_coeff[0])[0].shape[0])
    from pyscf.pbc import tools as _pbctools
    scalar("madelung", _pbctools.madelung(cell, kpts))
    emit("kpts", np.asarray(kpts))
    emit("mesh", np.asarray(cell.mesh, dtype=float))
    emit("focka", np.asarray(eris.fock[0]))
    emit("fockb", np.asarray(eris.fock[1]))
    emit("mo_energy_a", np.asarray(eris.mo_energy[0]))
    emit("mo_energy_b", np.asarray(eris.mo_energy[1]))
    emit("mo_coeff_a", np.asarray(eris.mo_coeff[0]))
    emit("mo_coeff_b", np.asarray(eris.mo_coeff[1]))
    emit(
        "nocc_per_kpt_a",
        np.asarray([int(np.count_nonzero(o > 0)) for o in kmf.mo_occ[0]], dtype=float),
    )
    emit(
        "nocc_per_kpt_b",
        np.asarray([int(np.count_nonzero(o > 0)) for o in kmf.mo_occ[1]], dtype=float),
    )

    r = SplitMix64(20260906)

    def draw(shape):
        n = int(np.prod(shape))
        return np.array(
            [complex(0.05 * r.unit(), 0.05 * r.unit()) for _ in range(n)]
        ).reshape(shape)

    t1 = (draw((nkpts, nocca, nvira)), draw((nkpts, noccb, nvirb)))
    t2 = (
        draw((nkpts, nkpts, nkpts, nocca, nocca, nvira, nvira)),
        draw((nkpts, nkpts, nkpts, nocca, noccb, nvira, nvirb)),
        draw((nkpts, nkpts, nkpts, noccb, noccb, nvirb, nvirb)),
    )

    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    Ht1a = np.zeros_like(t1a)
    Ht1b = np.zeros_like(t1b)
    Ht2aa = np.zeros_like(t2aa)
    Ht2ab = np.zeros_like(t2ab)
    Ht2bb = np.zeros_like(t2bb)

    nkpts, nocca, nvira = t1a.shape
    noccb, nvirb = t1b.shape[1:]
    #fvv_ = eris.fock[0][:,nocca:,nocca:]
    #fVV_ = eris.fock[1][:,noccb:,noccb:]
    #foo_ = eris.fock[0][:,:nocca,:nocca]
    #fOO_ = eris.fock[1][:,:noccb,:noccb]
    fov_ = eris.fock[0][:,:nocca,nocca:]
    fOV_ = eris.fock[1][:,:noccb,noccb:]

    # Get location of padded elements in occupied and virtual space
    nonzero_padding_alpha, nonzero_padding_beta = padding_k_idx(cc, kind="split")
    nonzero_opadding_alpha, nonzero_vpadding_alpha = nonzero_padding_alpha
    nonzero_opadding_beta, nonzero_vpadding_beta = nonzero_padding_beta

    mo_ea_o = [e[:nocca] for e in eris.mo_energy[0]]
    mo_eb_o = [e[:noccb] for e in eris.mo_energy[1]]
    mo_ea_v = [e[nocca:] + cc.level_shift for e in eris.mo_energy[0]]
    mo_eb_v = [e[noccb:] + cc.level_shift for e in eris.mo_energy[1]]

    Fvv_, FVV_ = kintermediates_uhf.cc_Fvv(cc, t1, t2, eris)
    Foo_, FOO_ = kintermediates_uhf.cc_Foo(cc, t1, t2, eris)
    Fov_, FOV_ = kintermediates_uhf.cc_Fov(cc, t1, t2, eris)

    # Move energy terms to the other side
    for k in range(nkpts):
        Fvv_[k][np.diag_indices(nvira)] -= mo_ea_v[k]
        FVV_[k][np.diag_indices(nvirb)] -= mo_eb_v[k]
        Foo_[k][np.diag_indices(nocca)] -= mo_ea_o[k]
        FOO_[k][np.diag_indices(noccb)] -= mo_eb_o[k]

    # Get the momentum conservation array
    kconserv = cc.khelper.kconserv

    # T1 equation
    P = kintermediates_uhf.kconserv_mat(cc.nkpts, cc.khelper.kconserv)
    Ht1a += fov_.conj()
    Ht1b += fOV_.conj()
    Ht1a += einsum('xyximae,yme->xia', t2aa, Fov_)
    Ht1a += einsum('xyximae,yme->xia', t2ab, FOV_)
    Ht1b += einsum('xyximae,yme->xia', t2bb, FOV_)
    Ht1b += einsum('yxymiea,yme->xia', t2ab, Fov_)
    Ht1a -= einsum('xyzmnae, xzymine->zia', t2aa, eris.ooov)
    Ht1a -= einsum('xyzmNaE, xzymiNE->zia', t2ab, eris.ooOV)
    #Ht1a -= einsum('xyzmnae,xzymine,xyzw->zia', t2aa, eris.ooov, P)
    #Ht1a -= einsum('xyzmNaE,xzymiNE,xyzw->zia', t2ab, eris.ooOV, P)
    Ht1b -= einsum('xyzmnae, xzymine->zia', t2bb, eris.OOOV)
    #Ht1b -= einsum('xyzmnae,xzymine,xyzw->zia', t2bb, eris.OOOV, P)
    Ht1b -= einsum('yxwnmea,xzymine,xyzw->zia', t2ab, eris.OOov, P)

    for ka in range(nkpts):
        Ht1a[ka] += einsum('ie,ae->ia', t1a[ka], Fvv_[ka])
        Ht1b[ka] += einsum('ie,ae->ia', t1b[ka], FVV_[ka])
        Ht1a[ka] -= einsum('ma,mi->ia', t1a[ka], Foo_[ka])
        Ht1b[ka] -= einsum('ma,mi->ia', t1b[ka], FOO_[ka])

        for km in range(nkpts):
            # ka == ki; km == kf == km
            # <ma||if> = [mi|af] - [mf|ai]
            #         => [mi|af] - [fm|ia]
            Ht1a[ka] += einsum('mf,aimf->ia', t1a[km], eris.voov[ka, ka, km])
            Ht1a[ka] -= einsum('mf,miaf->ia', t1a[km], eris.oovv[km, ka, ka])
            Ht1a[ka] += einsum('MF,aiMF->ia', t1b[km], eris.voOV[ka, ka, km])

            # miaf - mfai => miaf - fmia
            Ht1b[ka] += einsum('MF,AIMF->IA', t1b[km], eris.VOOV[ka, ka, km])
            Ht1b[ka] -= einsum('MF,MIAF->IA', t1b[km], eris.OOVV[km, ka, ka])
            Ht1b[ka] += einsum('mf,fmIA->IA', t1a[km], eris.voOV[km, km, ka].conj())

            for kf in range(nkpts):
                ki = ka
                ke = kconserv[ki, kf, km]
                Ht1a[ka] += einsum('imef,fmea->ia', t2aa[ki,km,ke], eris.vovv[kf,km,ke].conj())
                Ht1a[ka] += einsum('iMeF,FMea->ia', t2ab[ki,km,ke], eris.VOvv[kf,km,ke].conj())
                Ht1b[ka] += einsum('IMEF,FMEA->IA', t2bb[ki,km,ke], eris.VOVV[kf,km,ke].conj())
                Ht1b[ka] += einsum('mIfE,fmEA->IA', t2ab[km,ki,kf], eris.voVV[kf,km,ke].conj())

    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        kb = kconserv[ki, ka, kj]

        # Fvv equation
        Ftmpa_kb = Fvv_[kb] - 0.5 * einsum('mb,me->be', t1a[kb], Fov_[kb])
        Ftmpb_kb = FVV_[kb] - 0.5 * einsum('MB,ME->BE', t1b[kb], FOV_[kb])

        Ftmpa_ka = Fvv_[ka] - 0.5 * einsum('mb,me->be', t1a[ka], Fov_[ka])
        Ftmpb_ka = FVV_[ka] - 0.5 * einsum('MB,ME->BE', t1b[ka], FOV_[ka])

        tmp = einsum('ijae,be->ijab', t2aa[ki, kj, ka], Ftmpa_kb)
        Ht2aa[ki, kj, ka] += tmp

        tmp = einsum('IJAE,BE->IJAB', t2bb[ki, kj, ka], Ftmpb_kb)
        Ht2bb[ki, kj, ka] += tmp

        tmp = einsum('iJaE,BE->iJaB', t2ab[ki, kj, ka], Ftmpb_kb)
        Ht2ab[ki, kj, ka] += tmp

        tmp = einsum('iJeB,ae->iJaB', t2ab[ki, kj, ka], Ftmpa_ka)
        Ht2ab[ki, kj, ka] += tmp

        #P(ab)
        tmp = einsum('ijbe,ae->ijab', t2aa[ki, kj, kb], Ftmpa_ka)
        Ht2aa[ki, kj, ka] -= tmp

        tmp = einsum('IJBE,AE->IJAB', t2bb[ki, kj, kb], Ftmpb_ka)
        Ht2bb[ki, kj, ka] -= tmp

        # Foo equation
        Ftmpa_kj = Foo_[kj] + 0.5 * einsum('je,me->mj', t1a[kj], Fov_[kj])
        Ftmpb_kj = FOO_[kj] + 0.5 * einsum('JE,ME->MJ', t1b[kj], FOV_[kj])

        Ftmpa_ki = Foo_[ki] + 0.5 * einsum('je,me->mj', t1a[ki], Fov_[ki])
        Ftmpb_ki = FOO_[ki] + 0.5 * einsum('JE,ME->MJ', t1b[ki], FOV_[ki])

        tmp = einsum('imab,mj->ijab', t2aa[ki, kj, ka], Ftmpa_kj)
        Ht2aa[ki, kj, ka] -= tmp

        tmp = einsum('IMAB,MJ->IJAB', t2bb[ki, kj, ka], Ftmpb_kj)
        Ht2bb[ki, kj, ka] -= tmp

        tmp = einsum('iMaB,MJ->iJaB', t2ab[ki, kj, ka], Ftmpb_kj)
        Ht2ab[ki, kj, ka] -= tmp

        tmp = einsum('mJaB,mi->iJaB', t2ab[ki, kj, ka], Ftmpa_ki)
        Ht2ab[ki, kj, ka] -= tmp

        #P(ij)
        tmp = einsum('jmab,mi->ijab', t2aa[kj, ki, ka], Ftmpa_ki)
        Ht2aa[ki, kj, ka] += tmp

        tmp = einsum('JMAB,MI->IJAB', t2bb[kj, ki, ka], Ftmpb_ki)
        Ht2bb[ki, kj, ka] += tmp
    emit("fock_aa", Ht2aa)
    emit("fock_ab", Ht2ab)
    emit("fock_bb", Ht2bb)
    emit("fock_t1a", Ht1a)
    emit("fock_t1b", Ht1b)


def section_kgccsd_eom_ip(nk=(1, 1, 2)):
    """EOM-IP-KCCSD's matvec, its left sibling and its diagonal, on a FIXED
    synthetic trial vector — one emit per `kshift`.

    The trial vector is drawn from its OWN SplitMix64 stream (seed 20260907) so
    it does not depend on how many amplitudes were drawn before it.
    """
    from pyscf.pbc import scf as _pbcscf
    from pyscf.pbc.cc import kccsd as _kccsd
    from pyscf.pbc.cc import eom_kccsd_ghf as _eom

    cell = diamond()
    kpts = cell.make_kpts(list(nk))
    kmf = _pbcscf.KGHF(cell, kpts, exxdiv=None)
    kmf.conv_tol = 1e-10
    kmf.kernel()
    cc = _kccsd.KGCCSD(kmf)
    cc.conv_tol = 1e-9
    cc.conv_tol_normt = 1e-7
    eris = cc.ao2mo(cc.mo_coeff)

    nkpts = len(kpts)
    nocc, nmo = cc.nocc, cc.nmo
    nvir = nmo - nocc
    scalar("e_hf", kmf.e_tot)
    scalar("nkpts", nkpts)
    scalar("nocc", nocc)
    scalar("nmo", nmo)
    scalar("nao", np.asarray(eris.mo_coeff)[0].shape[0])
    emit("fock", np.asarray(eris.fock))
    emit("mo_energy", np.asarray(eris.mo_energy))
    emit("mo_coeff", np.asarray(eris.mo_coeff))
    from pyscf.pbc.mp.kmp2 import get_nocc as _get_nocc
    emit("nocc_per_kpt", np.asarray(_get_nocc(cc, per_kpoint=True), dtype=float))

    # FIXED synthetic amplitudes, drawn exactly as `section_kgccsd` draws them,
    # so the EOM intermediates here are the ones that gate ran against.
    st1, st2 = synthetic_amps(
        (nkpts, nocc, nvir), (nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir)
    )
    cc.t1, cc.t2 = st1, st2

    eom = _eom.EOMIP(cc)
    imds = eom.make_imds(eris=eris)
    size = eom.vector_size()
    scalar("ip_vector_size", size)

    r = SplitMix64(20260907)
    vec = np.array([complex(r.unit(), r.unit()) for _ in range(int(size))])
    emit("ip_vec", vec)

    for kshift in range(nkpts):
        emit("ip_matvec_%d" % kshift, _eom.ipccsd_matvec(eom, vec, kshift, imds))
        emit("ip_lmatvec_%d" % kshift, _eom.lipccsd_matvec(eom, vec, kshift, imds))
        emit("ip_diag_%d" % kshift, _eom.ipccsd_diag(eom, kshift, imds))

    # --- EA. The vector length depends on `kshift` (the pair list does), so
    # the trial vector is drawn per shift from its own stream.
    eom_ea = _eom.EOMEA(cc)
    imds_ea = eom_ea.make_imds(eris=eris)
    size_ea = eom_ea.vector_size()
    scalar("ea_vector_size", size_ea)
    for kshift in range(nkpts):
        r = SplitMix64(20260908 + kshift)
        v = np.array([complex(r.unit(), r.unit()) for _ in range(int(size_ea))])
        emit("ea_vec_%d" % kshift, v)
        emit("ea_matvec_%d" % kshift, _eom.eaccsd_matvec(eom_ea, v, kshift, imds_ea))
        emit("ea_lmatvec_%d" % kshift, _eom.leaccsd_matvec(eom_ea, v, kshift, imds_ea))
        emit("ea_diag_%d" % kshift, _eom.eaccsd_diag(eom_ea, kshift, imds_ea))

    # --- The actual ROOTS, on the CONVERGED amplitudes.
    #
    # The matvec gates above run on synthetic amplitudes so they measure the
    # equations; the roots have to run on converged ones to be a root at all.
    # `t1`/`t2` are emitted alongside so the Rust side solves the SAME
    # eigenproblem rather than its own converged approximation to it.
    ecc, t1, t2 = cc.kernel(eris=eris)
    scalar("e_corr", ecc)
    emit("t1", t1)
    emit("t2", t2)
    nroots = 2
    scalar("nroots", nroots)
    for tag, cls, fn in (("ip", _eom.EOMIP, "ipccsd"), ("ea", _eom.EOMEA, "eaccsd")):
        e = cls(cc)
        e.conv_tol = 1e-8
        e.max_cycle = 100
        imd = e.make_imds(eris=eris)
        for kshift in range(nkpts):
            # `ipccsd`/`eaccsd` return `(e, v)` and stash convergence on the
            # object (`:620-623`), so the flags come from `e.converged`.
            evals, evecs = getattr(e, fn)(nroots=nroots, kptlist=[kshift], imds=imd)
            emit("%s_roots_%d" % (tag, kshift), np.asarray(evals).ravel())
            emit(
                "%s_conv_%d" % (tag, kshift),
                np.asarray(np.real(e.converged), dtype=float).ravel(),
            )


SECTIONS = {
    "eris_gdf": lambda: section_eris_gdf((1, 1, 2)),
    "kcis": lambda: section_kcis((1, 1, 2)),
    "kgccsd": lambda: section_kgccsd((1, 1, 2)),
    "kgccsd_eom_ip": lambda: section_kgccsd_eom_ip((1, 1, 2)),
    "kuccsd": lambda: section_kuccsd((1, 1, 2)),
    "kuccsd_coarse": lambda: section_kuccsd((1, 1, 2), (13, 13, 13)),
    "kuccsd_imds": lambda: section_kuccsd_imds((1, 1, 2)),
    "kuccsd_wovvo": lambda: section_kuccsd_wovvo((1, 1, 2)),
    "kuccsd_woooo": lambda: section_kuccsd_woooo((1, 1, 2)),
    "kuccsd_fock": lambda: section_kuccsd_fock((1, 1, 2)),
    "kuccsd311": lambda: section_kuccsd((3, 1, 1)),
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
