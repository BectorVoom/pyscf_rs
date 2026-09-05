#!/usr/bin/env python
"""Phase 15 rollup oracle emitter (vendored PySCF 2.12.1).

One section per invocation, so a Rust test pays only for what it diffs:

    PYTHONPATH=$PWD .venv/bin/python -u \
      .planning/phases/15-periodic-ao2mo-kmp2/measurements/oracle_rollup.py <section>

Sections (matching `15-07-PLAN.md` Task 1's nine parts):

  symm_map   1. KptsHelper.symm_map + _operation, diamond [1,1,2] and [2,2,2]
  padding    2. padding_k_idx / get_nocc / get_nmo / get_frozen_mask
  ao2mo7d    3. ao2mo and ao2mo_7d, one k-quadruple per DF implementor
  lov        4. _init_mp_df_eris, element-wise
  kmp2       5. e_corr / e_corr_ss / e_corr_os, both routes, both systems
  t2rdm      6. t2 blocks, make_rdm1 both kinds, gamma1_intermediates
  mofirst    9. the MO-first ao2mo block, one per plane-wave builder

Parts 7 (`kmp2_stagger`) and 8 (the KUMP2 refusal) live in `stagger.py` and in
`oracle_phase15.rs` itself.

Numeric blocks are emitted as

    BEGIN <name> n=<count>
    <%.17g values, 8 per line; complex is interleaved re, im>
    END <name>

`%.17g` round-trips an f64 exactly, so the Rust side compares the same bits
upstream produced.
"""

import sys

import numpy as np

import pyscf
from pyscf.pbc import df, gto, mp, scf
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.mp import kmp2

assert pyscf.__version__ == "2.12.1", pyscf.__version__
print("pyscf.__version__=%s" % pyscf.__version__)


def emit(name, values):
    """Emit a flat float64 block."""
    flat = np.asarray(values).ravel()
    if np.iscomplexobj(flat):
        out = np.empty(flat.size * 2, dtype=np.float64)
        out[0::2] = flat.real
        out[1::2] = flat.imag
        flat = out
    else:
        flat = flat.astype(np.float64)
    print("BEGIN %s n=%d" % (name, flat.size))
    for i in range(0, flat.size, 8):
        print(" ".join("%.17g" % v for v in flat[i : i + 8]))
    print("END %s" % name)


# ---------------------------------------------------------------- fixtures


def diamond():
    """The committed anchor cell, `kmp2.py:795-806` — Bohr, gth-szv/gth-pade."""
    cell = gto.Cell()
    h = 3.370137329
    q = 1.685068664391
    cell.atom = [["C", (0.0, 0.0, 0.0)], ["C", (q, q, q)]]
    cell.a = np.array([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]])
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pade"
    cell.unit = "B"
    cell.verbose = 0
    cell.build()
    return cell


def helium():
    """All-electron He/6-31g, mesh 9 — the second gate system (`routes.py`)."""
    cell = gto.Cell()
    cell.atom = [["He", (0.0, 0.0, 0.0)]]
    a = 2.834589
    cell.a = np.array([[0.0, a, a], [a, 0.0, a], [a, a, 0.0]])
    cell.basis = "6-31g"
    cell.mesh = [9, 9, 9]
    cell.unit = "B"
    cell.verbose = 0
    cell.build()
    return cell


def krhf(cell, kmesh, with_df=None, conv_tol=1e-11):
    kpts = cell.make_kpts(kmesh, with_gamma_point=True)
    mf = scf.KRHF(cell, kpts, exxdiv=None)
    if with_df is not None:
        mf.with_df = with_df(cell, kpts)
    mf.conv_tol = conv_tol
    mf.kernel()
    assert mf.converged
    return mf


# ---------------------------------------------------------------- 1. symm_map


def section_symm_map():
    cell = diamond()
    for tag, kmesh in (("112", [1, 1, 2]), ("222", [2, 2, 2])):
        kpts = cell.make_kpts(kmesh, with_gamma_point=True)
        kh = kpts_helper.KptsHelper(cell, kpts)
        nk = len(kpts)
        print("symm_map_%s_nkpts=%d" % (tag, nk))
        print("symm_map_%s_norbits=%d" % (tag, len(kh.symm_map)))
        # Insertion order matters: it is what a correlated method iterates.
        flat = []
        for key, members in kh.symm_map.items():
            flat.extend(key)
            flat.append(len(members))
            for m in members:
                flat.extend(m)
        emit("symm_map_%s" % tag, np.asarray(flat, dtype=np.float64))
        emit("operation_%s" % tag, np.asarray(kh._operation, dtype=np.float64))


# ---------------------------------------------------------------- 2. padding


class _RaggedMp:
    """Upstream's documented ragged example, `kmp2.py:229-249`."""

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


PADDING_FROZEN = (None, 1, [0, 1], [[0], [0, 1], [1]])


def section_padding():
    m = _RaggedMp()
    for i, frozen in enumerate(PADDING_FROZEN):
        m.frozen = frozen
        flat = list(kmp2.get_nocc(m, per_kpoint=True))
        flat += list(kmp2.get_nmo(m, per_kpoint=True))
        flat += [kmp2.get_nocc(m), kmp2.get_nmo(m)]
        for x in kmp2.get_frozen_mask(m):
            flat += [1.0 if v else 0.0 for v in x]
        occ, vir = kmp2.padding_k_idx(m, "split")
        for x in occ:
            flat += [len(x)] + list(x)
        for x in vir:
            flat += [len(x)] + list(x)
        for x in kmp2.padding_k_idx(m, "joint"):
            flat += [len(x)] + list(x)
        emit("padding_%d" % i, np.asarray(flat, dtype=np.float64))


# ---------------------------------------------------------------- 3. ao2mo_7d


def _mo_pair(nao, nmo, seed):
    """A deterministic complex MO block; the transform must not care."""
    rng = np.random.default_rng(seed)
    return np.asarray(
        rng.standard_normal((nao, nmo)) + 1j * rng.standard_normal((nao, nmo)),
        order="F",
    )


def section_ao2mo7d():
    cell = helium()
    kmesh = [1, 1, 2]
    kpts = cell.make_kpts(kmesh, with_gamma_point=True)
    nao = cell.nao_nr()
    nk = len(kpts)
    kconserv = kpts_helper.get_kconserv(cell, kpts)
    builders = [
        ("fftdf", df.FFTDF(cell, kpts)),
        ("aftdf", df.AFTDF(cell, kpts)),
        ("gdf", df.GDF(cell, kpts).build()),
        ("mdf", df.MDF(cell, kpts).build()),
    ]
    mos = [_mo_pair(nao, nao, 20 + k) for k in range(nk)]
    ki, kj, kk = 0, 1, 1
    kl = kconserv[ki, kj, kk]
    print("ao2mo7d_nao=%d nkpts=%d ki=%d kj=%d kk=%d kl=%d" % (nao, nk, ki, kj, kk, kl))
    # The MO blocks travel WITH the reference values: the Rust side must
    # transform the same coefficients, or the diff measures two SCFs instead of
    # one AO2MO.
    for k in range(nk):
        emit("ao2mo7d_mo_%d" % k, mos[k].ravel(order="C"))
    for name, builder in builders:
        eri = builder.ao2mo(
            (mos[ki], mos[kj], mos[kk], mos[kl]),
            (kpts[ki], kpts[kj], kpts[kk], kpts[kl]),
            compact=False,
        )
        emit("ao2mo_%s" % name, np.asarray(eri).ravel())
        # The AO-level block too, so a residual can be attributed to the
        # integral or to the transform without a second oracle run.
        ao = builder.get_eri(
            (kpts[ki], kpts[kj], kpts[kk], kpts[kl]), compact=False
        )
        emit("aoeri_%s" % name, np.asarray(ao).ravel())
        # `ao2mo_7d` builds the full (nk, nk, nk) tensor; diff one slot of it.
        seven = builder.ao2mo_7d(np.asarray(mos), kpts)
        emit("ao2mo7d_%s" % name, np.asarray(seven[ki, kj, kk]).ravel())


# ---------------------------------------------------------------- 4. Lov


def section_lov():
    cell = diamond()
    mf = krhf(cell, [1, 1, 2], with_df=lambda c, k: df.GDF(c, k).build())
    m = mp.KMP2(mf)
    lov = kmp2._init_mp_df_eris(m)
    nkpts = m.nkpts
    nocc, nmo = m.nocc, m.nmo
    print("lov_nkpts=%d nocc=%d nvir=%d" % (nkpts, nocc, nmo - nocc))
    mo = kmp2._add_padding(m, m.mo_coeff, m.mo_energy)[0]
    for k in range(nkpts):
        emit("lov_mo_%d" % k, np.asarray(mo[k]).ravel(order="C"))
    flat = []
    for ki in range(nkpts):
        for kj in range(nkpts):
            block = np.asarray(lov[ki, kj])
            flat.append(block.shape[0])
    emit("lov_naux", np.asarray(flat, dtype=np.float64))
    for ki in range(nkpts):
        for kj in range(nkpts):
            emit("lov_%d_%d" % (ki, kj), np.asarray(lov[ki, kj]).ravel())


# ---------------------------------------------------------------- 5. KMP2


def section_kmp2():
    for tag, cell, builders in (
        (
            "diamond",
            diamond(),
            (("fftdf", None), ("gdf", lambda c, k: df.GDF(c, k).build())),
        ),
        (
            "helium",
            helium(),
            (("fftdf", None), ("gdf", lambda c, k: df.GDF(c, k).build())),
        ),
    ):
        for name, builder in builders:
            mf = krhf(cell, [1, 1, 2], with_df=builder)
            m = mp.KMP2(mf)
            e_corr, _ = m.kernel()
            print(
                "kmp2_%s_%s e_corr=%.17g ss=%.17g os=%.17g e_hf=%.17g"
                % (tag, name, e_corr, m.e_corr_ss, m.e_corr_os, mf.e_tot)
            )
            if name == "gdf":
                # The same mean field forced through the four-index AO2MO path.
                m2 = mp.KMP2(mf)
                m2.with_df_ints = False
                e2, _ = m2.kernel()
                print("kmp2_%s_gdf_ao2mo e_corr=%.17g" % (tag, e2))


# ---------------------------------------------------------------- 6. t2/RDM


def section_t2rdm():
    cell = helium()
    mf = krhf(cell, [1, 1, 2])
    m = mp.KMP2(mf)
    e_corr, t2 = m.kernel()
    print("t2rdm_e_corr=%.17g nkpts=%d nocc=%d nmo=%d" % (e_corr, m.nkpts, m.nocc, m.nmo))
    t2 = np.asarray(t2)
    print("t2rdm_t2_shape=%s" % (list(t2.shape),))
    mo, moe = kmp2._add_padding(m, m.mo_coeff, m.mo_energy)
    print("t2rdm_mesh=%s" % (list(map(int, cell.mesh)),))
    for k in range(m.nkpts):
        emit("t2rdm_mo_%d" % k, np.asarray(mo[k]).ravel(order="C"))
        emit("t2rdm_moe_%d" % k, np.asarray(moe[k]).ravel())
    emit("t2", t2.ravel())
    dm1 = np.asarray(m.make_rdm1(t2))
    emit("rdm1_padded", dm1.ravel())
    dm1u = np.asarray(m.make_rdm1(t2, kind="compact"))
    emit("rdm1_compact", dm1u.ravel())
    doo, dvv = kmp2._gamma1_intermediates(m, t2)
    emit("gamma1_doo", np.asarray(doo).ravel())
    emit("gamma1_dvv", np.asarray(dvv).ravel())


# ---------------------------------------------------------------- 9. MO-first


def section_mofirst():
    """`fft_ao2mo.general` / `aft_ao2mo.general` on an (o,v,o,v) quadruple.

    This is the route the phase's own FFTDF anchor runs on: upstream transforms
    AO->MO on the real-space grid and never materialises the AO ERI.
    """
    cell = diamond()
    kmesh = [1, 1, 2]
    mf = krhf(cell, kmesh)
    m = mp.KMP2(mf)
    kpts = mf.kpts
    nk = len(kpts)
    nocc, nmo = m.nocc, m.nmo
    mo = kmp2._add_padding(m, m.mo_coeff, m.mo_energy)[0]
    kconserv = kpts_helper.get_kconserv(cell, kpts)
    print("mofirst_nkpts=%d nocc=%d nvir=%d" % (nk, nocc, nmo - nocc))
    print("mofirst_mesh=%s" % (list(map(int, cell.mesh)),))
    for k in range(nk):
        emit("mofirst_mo_%d" % k, np.asarray(mo[k]).ravel(order="C"))
    for name, builder in (("fftdf", df.FFTDF(cell, kpts)), ("aftdf", df.AFTDF(cell, kpts))):
        for ki in range(nk):
            for ka in range(nk):
                for kj in range(nk):
                    kb = kconserv[ki, ka, kj]
                    eri = builder.ao2mo(
                        (
                            mo[ki][:, :nocc],
                            mo[ka][:, nocc:],
                            mo[kj][:, :nocc],
                            mo[kb][:, nocc:],
                        ),
                        (kpts[ki], kpts[ka], kpts[kj], kpts[kb]),
                        compact=False,
                    )
                    emit("mofirst_%s_%d_%d_%d" % (name, ki, ka, kj), np.asarray(eri).ravel())


SECTIONS = {
    "symm_map": section_symm_map,
    "padding": section_padding,
    "ao2mo7d": section_ao2mo7d,
    "lov": section_lov,
    "kmp2": section_kmp2,
    "t2rdm": section_t2rdm,
    "mofirst": section_mofirst,
}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in SECTIONS:
        raise SystemExit("usage: oracle_rollup.py {%s}" % "|".join(SECTIONS))
    SECTIONS[sys.argv[1]]()
